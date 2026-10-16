# Exp #038 model-size follow-up: Qwen3-4B, LoRA r=16 + zero-shot, AG News, on a T4 (16 GB, no bf16).
# Same data / prompt / hyper-params / eval as train.py (0.6B). Differences: fp16 base weights, fp32 LoRA params,
# fp16 autocast + GradScaler, gradient checkpointing, batch 16 run as 4 micro-batches of 4 (grad accumulation).
# Eval size can be cut at runtime by writing an integer into /content/out4b/eval_n (read before each eval);
# every record also reports accuracy on the first 1000 test examples so shortened evals stay comparable.
import json, math, os, time
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

OUT = "/content/out4b"; os.makedirs(OUT, exist_ok=True)
LOG = open(f"{OUT}/progress_4b.log", "a", buffering=1)
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n")

MODEL = "Qwen/Qwen3-4B"; SEED = 1234
N_TRAIN, N_TEST, LR, BS, MAXLEN = 4000, 2000, 2e-4, 16, 192
MICRO = 4; ACC = BS // MICRO
RANK = 16
LABELS = [" World", " Sports", " Business", " Sci/Tech"]
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
dev = "cuda"

tok = AutoTokenizer.from_pretrained(MODEL)
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
HEAD = tok("Classify the topic of this news article as World, Sports, Business, or Sci/Tech.\nArticle: ", add_special_tokens=False).input_ids
TAIL = tok("\nTopic:", add_special_tokens=False).input_ids
LAB_IDS = [tok(l, add_special_tokens=False).input_ids for l in LABELS]
MAXLAB = max(len(l) for l in LAB_IDS)
log("label token ids", LAB_IDS)

def prompt_ids(text):
    budget = MAXLEN - len(HEAD) - len(TAIL) - MAXLAB
    body = tok(text, add_special_tokens=False).input_ids[:budget]
    return HEAD + body + TAIL

ds = load_dataset("fancyzhx/ag_news")
train = ds["train"].shuffle(seed=SEED).select(range(N_TRAIN))
test = ds["test"].shuffle(seed=SEED).select(range(N_TEST))
train_ex = [(prompt_ids(t), y) for t, y in zip(train["text"], train["label"])]
test_ex = [(prompt_ids(t), y) for t, y in zip(test["text"], test["label"])]

def collate(pairs):
    """pairs: list of (prompt_ids, answer_ids). Left-pad so answers sit at the end; labels -100 except answer tokens."""
    L = max(len(p) + len(a) for p, a in pairs)
    ids = torch.full((len(pairs), L), PAD); att = torch.zeros((len(pairs), L), dtype=torch.long)
    lab = torch.full((len(pairs), L), -100)
    for i, (p, a) in enumerate(pairs):
        s = p + a; o = L - len(s); ids[i, o:] = torch.tensor(s); att[i, o:] = 1
        lab[i, L - len(a):] = torch.tensor(a)
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    return ids.to(dev), att.to(dev), pos.to(dev), lab.to(dev)

K = MAXLAB + 1  # only compute logits for the last K positions (covers all answer tokens)
def answer_logits(model, ids, att, pos, lab):
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(input_ids=ids, attention_mask=att, position_ids=pos, logits_to_keep=K).logits[:, :-1]
    return logits.float(), lab[:, -(K - 1):]

def answer_logprobs(model, ids, att, pos, lab):
    """Return per-sequence summed log-prob of answer tokens and token count."""
    logits, tgt = answer_logits(model, ids, att, pos, lab); mask = tgt != -100
    lp = torch.log_softmax(logits, -1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (lp * mask).sum(1), mask.sum(1)

def eval_n():
    p = f"{OUT}/eval_n"
    return min(N_TEST, int(open(p).read().strip())) if os.path.exists(p) else N_TEST

@torch.no_grad()
def evaluate(model, ebs=8):
    model.eval(); n_eval = eval_n(); correct = []; losses = []
    for i in range(0, n_eval, ebs):
        chunk = test_ex[i:min(i + ebs, n_eval)]
        pairs = [(p, LAB_IDS[k]) for p, _ in chunk for k in range(4)]
        s, n = answer_logprobs(model, *collate(pairs))
        s = s.view(len(chunk), 4); n = n.view(len(chunk), 4)
        for j, (_, y) in enumerate(chunk):
            correct.append(int(s[j].argmax().item() == y))
            losses.append((-s[j, y] / n[j, y]).item())  # mean per-token CE on gold answer
        if (i // ebs) % 50 == 0: log(f"eval {i + len(chunk)}/{n_eval} acc_so_far {sum(correct)/len(correct):.4f}")
    return dict(n_eval=n_eval, eval_accuracy=sum(correct) / n_eval, eval_loss=sum(losses) / n_eval,
                eval_accuracy_first1000=sum(correct[:1000]) / min(1000, n_eval))

def load_base():
    torch.manual_seed(SEED)
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev)

res_path = f"{OUT}/runs_4b.json"
runs = json.load(open(res_path)) if os.path.exists(res_path) else []
def save():
    json.dump(runs, open(res_path, "w"), indent=2)

if not any(r["run"] == "zero_shot" for r in runs):
    m = load_base(); t0 = time.time()
    res = evaluate(m)
    runs.append(dict(run="zero_shot", model=MODEL, rank=0, trainable_params=0, train_seconds=0.0,
                     peak_gpu_mem_gb=None, final_train_loss=None, eval_seconds=time.time() - t0, **res))
    save(); log("zero_shot", runs[-1]); del m; torch.cuda.empty_cache()

if not any(x["run"] == f"lora_r{RANK}" for x in runs):
    r = RANK
    m = load_base()
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.enable_input_require_grads()
    m = get_peft_model(m, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, target_modules=TARGETS, task_type="CAUSAL_LM"))
    ntrain = sum(p.numel() for p in m.parameters() if p.requires_grad)
    log(f"rank {r}: trainable params {ntrain}, lora dtype {next(p for p in m.parameters() if p.requires_grad).dtype}")
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    g = torch.Generator().manual_seed(SEED); order = torch.randperm(N_TRAIN, generator=g).tolist()
    steps = math.ceil(N_TRAIN / BS)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / steps))
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    m.train(); recent = []
    for step in range(steps):
        batch = [train_ex[i] for i in order[step * BS:(step + 1) * BS]]
        opt.zero_grad(set_to_none=True); step_loss = 0.0
        for k in range(0, len(batch), MICRO):
            logits, tgt = answer_logits(m, *collate([(p, LAB_IDS[y]) for p, y in batch[k:k + MICRO]]))
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100)
            scaler.scale(loss / ACC).backward(); step_loss += loss.item() / ACC
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        recent.append(step_loss); recent = recent[-20:]
        if step % 10 == 0 or step == steps - 1:
            log(f"rank {r} step {step + 1}/{steps} loss {step_loss:.4f} avg20 {sum(recent)/len(recent):.4f} t={time.time()-t0:.0f}s peak {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    torch.cuda.synchronize(); train_s = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    t1 = time.time(); res = evaluate(m)
    runs.append(dict(run=f"lora_r{r}", model=MODEL, rank=r, lora_alpha=2 * r, trainable_params=ntrain, train_seconds=train_s,
                     peak_gpu_mem_gb=peak, final_train_loss=recent[-1], final_train_loss_avg20=sum(recent) / len(recent),
                     eval_seconds=time.time() - t1, **res))
    save(); log(f"lora_r{r}", runs[-1]); del m, opt; torch.cuda.empty_cache()
log("DONE")
