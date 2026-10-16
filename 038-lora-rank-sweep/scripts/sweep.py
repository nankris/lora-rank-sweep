"""#038 LoRA rank sweep: Qwen3-0.6B on AG News (4-way topic classification), ranks 4/16/64 x 3 seeds + zero-shot.
Run on a Colab T4. Writes /content/results038/{runs.jsonl,progress.log}.
Loss is computed only on the answer tokens. Accuracy = argmax over the 4 label strings' summed log-probs.
"""
import os, json, time, random, math, sys
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

OUT = "/content/results038"; os.makedirs(OUT, exist_ok=True)
MODEL = "Qwen/Qwen3-0.6B"
RANKS = [4, 16, 64]
SEEDS = [0, 1, 2]
N_TRAIN, N_EVAL = 4000, 2000
BATCH, LR, EPOCHS, MAXLEN = 16, 2e-4, 1, 192
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
T0 = time.time()
def plog(m):
    with open(f"{OUT}/progress.log", "a") as f: f.write(f"[{time.time()-T0:6.0f}s] {m}\n")

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "right"
ds = load_dataset("fancyzhx/ag_news")
train = ds["train"].shuffle(seed=1234).select(range(N_TRAIN))
evl = ds["test"].shuffle(seed=1234).select(range(N_EVAL))
def prompt(text):
    return ("Classify the news article into one of: World, Sports, Business, Sci/Tech.\n"
            f"Article: {text.strip()[:600]}\nAnswer:")
label_ids = [tok(" " + l, add_special_tokens=False)["input_ids"] + [tok.eos_token_id] for l in LABELS]
plog(f"label token ids {label_ids}")

def encode(text, lab):
    p = tok(prompt(text), add_special_tokens=False)["input_ids"][: MAXLEN - 8]
    a = label_ids[lab]
    return p + a, [-100]*len(p) + a

def collate(items):
    L = max(len(x[0]) for x in items)
    ids = torch.full((len(items), L), tok.pad_token_id); lab = torch.full((len(items), L), -100); att = torch.zeros((len(items), L), dtype=torch.long)
    for i, (x, y) in enumerate(items):
        ids[i, :len(x)] = torch.tensor(x); lab[i, :len(y)] = torch.tensor(y); att[i, :len(x)] = 1
    return ids.cuda(), att.cuda(), lab.cuda()

def seq_logprob(model, ids, att, lab):
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(input_ids=ids, attention_mask=att).logits.float()
    lp = torch.log_softmax(logits[:, :-1], -1); tgt = lab[:, 1:]
    m = tgt != -100
    g = lp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1) * m
    return g.sum(-1), m.sum(-1)

@torch.no_grad()
def evaluate(model):
    model.eval(); correct = 0; nll = 0.0; ntok = 0
    for s in range(0, N_EVAL, 32):
        rows = evl.select(range(s, min(s+32, N_EVAL)))
        scores = []
        for k in range(4):
            batch = [encode(r["text"], k) for r in rows]
            lp, n = seq_logprob(model, *collate(batch)); scores.append(lp)
            for j, r in enumerate(rows):
                if r["label"] == k: nll -= lp[j].item(); ntok += n[j].item()
        S = torch.stack(scores, 1)
        correct += (S.argmax(1).cpu() == torch.tensor(rows["label"])).sum().item()
    model.train()
    return correct / N_EVAL, nll / ntok

def fresh():
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).cuda()
    m.config.use_cache = False
    return m

runs = open(f"{OUT}/runs.jsonl", "a")
# zero-shot baseline
m = fresh(); t = time.time(); acc, loss = evaluate(m)
rec = {"rank": 0, "seed": None, "eval_acc": acc, "eval_loss": loss, "train_s": 0, "eval_s": time.time()-t, "trainable_params": 0}
runs.write(json.dumps(rec)+"\n"); runs.flush(); plog(json.dumps(rec)); del m; torch.cuda.empty_cache()

train_enc = [encode(r["text"], r["label"]) for r in train]
for r in RANKS:
    for seed in SEEDS:
        random.seed(seed); torch.manual_seed(seed)
        m = fresh()
        cfg = LoraConfig(r=r, lora_alpha=2*r, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], task_type="CAUSAL_LM")
        m = get_peft_model(m, cfg)
        ntrain = sum(p.numel() for p in m.parameters() if p.requires_grad)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
        scaler = torch.amp.GradScaler("cuda")
        order = list(range(N_TRAIN)); random.shuffle(order)
        steps = math.ceil(N_TRAIN / BATCH) * EPOCHS; warm = int(0.05*steps)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s+1)/max(1, warm)) * max(0.0, (steps - s) / steps))
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time(); losses = []
        for step in range(steps):
            idx = order[(step*BATCH) % N_TRAIN: (step*BATCH) % N_TRAIN + BATCH]
            ids, att, lab = collate([train_enc[i] for i in idx])
            with torch.autocast("cuda", dtype=torch.float16):
                out = m(input_ids=ids, attention_mask=att, labels=lab)
            scaler.scale(out.loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            losses.append(out.loss.item())
        torch.cuda.synchronize(); tt = time.time() - t
        t2 = time.time(); acc, loss = evaluate(m)
        rec = {"rank": r, "seed": seed, "eval_acc": acc, "eval_loss": loss, "train_s": tt, "eval_s": time.time()-t2,
               "trainable_params": ntrain, "peak_mem_gb": torch.cuda.max_memory_allocated()/1e9, "steps": steps,
               "train_loss_first20": sum(losses[:20])/20, "train_loss_last20": sum(losses[-20:])/20, "train_losses": losses}
        runs.write(json.dumps(rec)+"\n"); runs.flush()
        plog(json.dumps({k: v for k, v in rec.items() if k != "train_losses"}))
        del m, opt; torch.cuda.empty_cache()
plog("ALL DONE")
