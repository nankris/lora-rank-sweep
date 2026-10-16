import json, os, csv
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
a = json.load(open(f"{D}/results/colab/runs.json")); b = json.load(open(f"{D}/results/colab/runs_4b.json"))
rows = [dict(model="Qwen3-0.6B", **{k: r.get(k) for k in ("run","rank","trainable_params","train_seconds","peak_gpu_mem_gb","eval_accuracy","eval_loss")}) for r in a] + \
       [dict(model="Qwen3-4B", **{k: r.get(k) for k in ("run","rank","trainable_params","train_seconds","peak_gpu_mem_gb","eval_accuracy","eval_loss")}) for r in b]
w = csv.DictWriter(open(f"{D}/results/summary.csv","w"), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
lab = [f"{r['model']}\n{r['run']}" for r in rows]; col = ["#bbb" if r["rank"]==0 else ("#d17b4a" if r["model"]=="Qwen3-0.6B" else "#4a7bd1") for r in rows]
ax[0].bar(lab, [r["eval_accuracy"] for r in rows], color=col)
for i, r in enumerate(rows): ax[0].text(i, r["eval_accuracy"], f"{r['eval_accuracy']:.3f}", ha="center", va="bottom", fontsize=8)
ax[0].set_ylim(0, 1.05); ax[0].set_title("AG News accuracy (2,000 test ex, 1 seed)"); ax[0].tick_params(axis="x", labelsize=7)
tr = [r for r in rows if r["rank"]]
ax[1].bar([f"{r['model']}\n{r['run']}" for r in tr], [r["trainable_params"]/1e6 for r in tr], color=[c for c, r in zip(col, rows) if r["rank"]])
ax2 = ax[1].twinx(); ax2.plot(range(len(tr)), [r["train_seconds"] for r in tr], "ko-"); ax2.set_ylabel("train seconds on T4 (dots)")
ax[1].set_ylabel("trainable params (M, bars)"); ax[1].set_title("Cost side: params and T4 training time"); ax[1].tick_params(axis="x", labelsize=7)
plt.suptitle("#038c LoRA rank sweep, driven end-to-end by Claude Code on Colab T4", fontsize=11); plt.tight_layout(); plt.savefig(f"{D}/results/chart.png", dpi=110)
