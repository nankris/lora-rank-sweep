# #038c LoRA rank sweep on Colab, driven end-to-end by Claude Code

**Question.** (1) Does LoRA rank 4 / 16 / 64 matter for Qwen3-0.6B on a small classification fine-tune, and what does a bigger base model (Qwen3-4B, r16) add? (2) Can Claude Code, given only the `colab` CLI, write the training code, run it on a Colab GPU, babysit it and clean up — and what does that driving cost?

**Short answer.**
- **Rank barely matters.** Qwen3-0.6B on AG News (4,000 train, 1 epoch): r4 **89.6%**, r16 **90.9%**, r64 **90.65%** (zero-shot 31.9%). r16 ≈ r64 within noise (1 seed, ±0.65 pp SE); r4 ~1 pp lower. 16× more trainable params (2.5M → 40.4M) changed training time by 3% (170 → 175 s) and peak memory by 7% (11.0 → 11.7 GB).
- **A 6.7× bigger base adds little after fine-tuning.** Qwen3-4B r16: **91.55%** (+0.65 pp over 0.6B r16, within noise), eval loss 0.209 vs 0.243. Zero-shot it's far stronger (73.4% vs 31.9%). Training took 5.3× longer on the T4 (900 s vs 170 s).
- **Claude Code drove it all.** It wrote `train.py`/`train_4b.py`, launched training as a background process on the VM, polled, fixed one OOM (full-vocab logits → `logits_to_keep`) and the known peft/torchao import error (uninstalled torchao), downloaded results, stopped the sessions and verified. Driving cost (API-equivalent): **$0.66** for the 0.6B sweep (Opus 5.5, 27 turns, 24 tool calls, 26 min wall, 116 s of model time) and **~$0.04** for the 4B follow-ups (Haiku 5.5 resuming the same session, 31 turns, 38 tool calls). Session total **$0.70**.
- **GPU used: 59 min of T4 total** (22.6 min for the 0.6B sweep, 36.5 min for 4B incl. ~11 min of model load + zero-shot eval). Est. **≈1.1–1.8 CU** (at the 1.07 CU/hr per T4 seen in `colab usage` up to the 1.8 CU/hr quoted for T4); the account balance is shared with another agent's sessions, so CU can't be read off the balance exactly. Well under the 5 CU cap.

## Setup
- Claude Code 2.1.300, Pro. Driver run: `claude -p "$(cat scripts/prompt.txt)" --output-format stream-json --verbose --dangerously-skip-permissions --strict-mcp-config` with cwd `work/` (via `_harness/run_cc.py --persist`), default model Opus 5.5. Follow-ups resumed the same session (`--resume <session>`, `--model haiku`) with `scripts/prompt_4b.txt`, `scripts/prompt_4b_t4.txt`, `scripts/prompt_4b_resume.txt`.
- Colab: `colab new -s claude-038 --gpu T4` (0.6B), `colab new -s claude-038-4b --gpu T4` (4B). **GPU type: T4 for both.** The 4B run was asked to use an L4; L4 allocation was refused for ~15 min (the account's other sessions held the L4 slots), so it fell back to a T4.
- Data/model: `fancyzhx/ag_news`, 4,000 shuffled train (seed 1234), 2,000 shuffled test; `Qwen/Qwen3-0.6B` (0.6B params) and `Qwen/Qwen3-4B` (4B params). LoRA alpha = 2r on q/k/v/o/gate/up/down, lr 2e-4 (10-step warmup, linear decay), batch 16, max len 192, 1 epoch, loss on answer tokens only. Accuracy = argmax of summed log-probs of " World"/" Sports"/" Business"/" Sci/Tech". Same setup as the other agent's `../038-lora-rank-sweep` (Codex/Hark-driven) for comparability.
- 0.6B: fp32 weights + fp16 autocast. 4B (T4: 16 GB, no bf16): fp16 weights, fp32 LoRA params, fp16 autocast + GradScaler, gradient checkpointing, 4×4 micro-batches. No 4-bit.

## Results
| Model | Run | Trainable params | Train s (T4) | Peak GPU GB | Eval acc | Eval loss |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | zero-shot | 0 | – | – | 0.319 | 5.687 |
| Qwen3-0.6B | LoRA r4 | 2,523,136 | 170 | 10.99 | 0.896 | 0.256 |
| Qwen3-0.6B | LoRA r16 | 10,092,544 | 170 | 11.14 | **0.909** | 0.243 |
| Qwen3-0.6B | LoRA r64 | 40,370,176 | 175 | 11.75 | 0.9065 | 0.247 |
| Qwen3-4B | zero-shot | 0 | – | – | 0.734 | 2.652 |
| Qwen3-4B | LoRA r16 | 33,030,144 | 900 | 9.01 | **0.9155** | **0.209** |

Claude Code driving cost (from `results/runs.csv`; in a resumed session `total_cost_usd` is cumulative, so per-segment = difference):
| Segment | Model | Turns | Tool calls | Wall | Est. $ (segment) | Outcome |
|---|---|---|---|---|---|---|
| 0.6B sweep | Opus 5.5 | 27 | 24 | 26.4 min | $0.656 | done, session stopped |
| 4B on L4 | Haiku 5.5 | 7 | 6 | 15 min | $0.017 | L4 refused, no GPU used |
| 4B on T4 | Haiku 5.5 | — | 9 | — | ≈$0.001 | interrupted: this sandbox rebooted and killed the local `claude` process; the VM job kept running |
| 4B resume | Haiku 5.5 | 24 | 23 | 29 min | $0.026 | done, session stopped |

Files: `results/colab/` (runs.json, runs_4b.json, progress logs, stdout logs, summary.md written by Claude), `scripts/train.py`, `scripts/train_4b.py` (Claude-written), `results/raw/` (Claude stream-json), `results/summary.csv`, chart `results/chart.png`.

## Caveats
- 1 seed per run; differences under ~1.3 pp aren't significant.
- 0.6B and 4B differ in precision setup (fp32 vs fp16 weights + checkpointing), forced by the T4.
- One extra one-line turn ("say hi", ≈$0.02 Haiku-cheap... included in the session total) was sent to the session by me to check it could be resumed after the reboot.
- CU is estimated from GPU minutes × T4 rate, not measured per session.
- The driver was told to keep the GPU under 60 / 50 min and did (22.6 / 36.5 min); it never touched other sessions.
