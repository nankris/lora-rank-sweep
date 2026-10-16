# Exp #038: LoRA rank sweep, Qwen3-0.6B on AG News. Writes /content/out/{runs.json,progress.log}
import json, math, os, sys, time
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

OUT = "/content/out038r"; os.makedirs(OUT, exist_ok=True)
LOG = open(f"{OUT}/progress.log", "a", buffering=1)
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n")

MODEL = "Qwen/Qwen3-0.6B"; SEED = 1234
N_TRAIN, N_TEST, LR, BS, MAXLEN = 4000, 2000, 2e-4, 16, 192
RANKS = [int(r) for r in os.environ.get("RANKS", "4,16,64").split(",")]
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

@torch.no_grad()
def evaluate(model, ebs=16):
    model.eval(); correct = 0; losses = []; predictions = []
    for i in range(0, len(test_ex), ebs):
        chunk = test_ex[i:i + ebs]
        pairs = [(p, LAB_IDS[k]) for p, _ in chunk for k in range(4)]
        s, n = answer_logprobs(model, *collate(pairs))
        s = s.view(len(chunk), 4); n = n.view(len(chunk), 4)
        for j, (_, y) in enumerate(chunk):
            predictions.append(int(s[j].argmax().item()))
            correct += int(s[j].argmax().item() == y)
            losses.append((-s[j, y] / n[j, y]).item())  # mean per-token CE on gold answer
    json.dump(predictions, open(f"{OUT}/pred_r{CURRENT_RANK}_s{RUN_SEED}.json", "w"))
    return correct / len(test_ex), sum(losses) / len(losses)

def load_base():
    torch.manual_seed(RUN_SEED)
    return AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev)

CURRENT_RANK = 0
RUN_SEED = 1234
res_path = f"{OUT}/runs.json"
runs = json.load(open(res_path)) if os.path.exists(res_path) else []
def save():
    json.dump(runs, open(res_path, "w"), indent=2)

if not any(r["run"] == "zero_shot" for r in runs):
    m = load_base(); t0 = time.time()
    acc, el = evaluate(m)
    runs.append(dict(run="zero_shot", rank=0, trainable_params=0, train_seconds=0.0,
                     peak_gpu_mem_gb=None, final_train_loss=None, eval_accuracy=acc, eval_loss=el,
                     eval_seconds=time.time() - t0))
    save(); log("zero_shot", runs[-1]); del m; torch.cuda.empty_cache()

for RUN_SEED in [int(s) for s in os.environ.get("SEEDS", "1234,1235,1236").split(",")]:
    for r in RANKS:
        CURRENT_RANK = r
        if any(x["run"] == f"lora_r{r}_s{RUN_SEED}" for x in runs): continue
        m = load_base()
        m = get_peft_model(m, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, target_modules=TARGETS, task_type="CAUSAL_LM"))
        ntrain = sum(p.numel() for p in m.parameters() if p.requires_grad)
        log(f"rank {r}: trainable params {ntrain}")
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
        scaler = torch.amp.GradScaler("cuda")
        g = torch.Generator().manual_seed(RUN_SEED); order = torch.randperm(N_TRAIN, generator=g).tolist()
        steps = math.ceil(N_TRAIN / BS)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / steps))
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        m.train(); recent = []
        for step in range(steps):
            batch = [train_ex[i] for i in order[step * BS:(step + 1) * BS]]
            logits, tgt = answer_logits(m, *collate([(p, LAB_IDS[y]) for p, y in batch]))
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            recent.append(loss.item()); recent = recent[-20:]
            if step % 25 == 0 or step == steps - 1:
                log(f"rank {r} step {step + 1}/{steps} loss {loss.item():.4f} avg20 {sum(recent)/len(recent):.4f} t={time.time()-t0:.0f}s")
        torch.cuda.synchronize(); train_s = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        t1 = time.time(); acc, el = evaluate(m)
        runs.append(dict(run=f"lora_r{r}_s{RUN_SEED}", rank=r, seed=RUN_SEED, lora_alpha=2 * r, trainable_params=ntrain, train_seconds=train_s,
                         peak_gpu_mem_gb=peak, final_train_loss=recent[-1], final_train_loss_avg20=sum(recent) / len(recent),
                         eval_accuracy=acc, eval_loss=el, eval_seconds=time.time() - t1))
        save(); log(f"lora_r{r}_s{RUN_SEED}", runs[-1]); del m, opt; torch.cuda.empty_cache()
log("DONE")
import importlib.metadata as md
json.dump(dict(gpu=torch.cuda.get_device_name(), train_fingerprint=train._fingerprint, test_fingerprint=test._fingerprint, versions={x:md.version(x) for x in ["torch","transformers","peft","datasets"]}),open(f"{OUT}/metadata.json","w"),indent=2)
