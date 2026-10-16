# Exp #038 — LoRA rank sweep, Qwen3-0.6B on AG News (T4)

4000 train ex (shuffle seed 1234), 1 epoch, lr 2e-4 (10-step warmup, linear decay), batch 16, max len 192, fp16 autocast,
LoRA alpha=2r on q/k/v/o/gate/up/down, loss on answer tokens only. Eval: 2000 shuffled test ex; acc = argmax of summed
log-probs of " World"/" Sports"/" Business"/" Sci/Tech"; eval loss = mean per-token CE on the gold label tokens. 1 seed.

| Run | Trainable params | Train s | Peak GPU mem (GB) | Final train loss (last step / avg last 20) | Eval acc | Eval loss |
|---|---|---|---|---|---|---|
| zero-shot | 0 | – | – | – | 0.3190 | 5.687 |
| LoRA r=4  | 2,523,136  | 170.4 | 10.99 | 0.434 / 0.264 | 0.8960 | 0.256 |
| LoRA r=16 | 10,092,544 | 169.9 | 11.14 | 0.393 / 0.253 | **0.9090** | **0.243** |
| LoRA r=64 | 40,370,176 | 174.7 | 11.75 | 0.339 / 0.243 | 0.9065 | 0.247 |

Takeaway: all ranks reach ~90%; r=16 and r=64 are within noise (1 seed, ±~0.65pp binomial SE), r=4 is ~1pp lower.
Training time is flat across ranks; memory grows slightly with rank (peak is dominated by fp32 base weights + activations).

**GPU time: 1357 s = 22.6 min** (`colab new` succeeded at ~19:52:40 UTC to `colab stop` at ~20:15:17 UTC, 2026-10-14).
Includes one OOM crash (full-vocab eval logits; fixed with `logits_to_keep`) and one peft/torchao import error
(fixed by uninstalling torchao on the VM); the zero-shot result came from the second attempt and was reused when the run resumed.
Files: `runs.json`, `progress.log`, `stdout.log`, `train.py`.

## Qwen3-4B r16 (model-size follow-up)

Session `claude-038-4b`, **T4** (L4 was unavailable on the account). Same data, prompt, LR/schedule, batch 16, max len 192,
LoRA alpha=2r on all attention + MLP projections, answer-only loss, same eval (2000 test ex, 1 seed). Deviations forced by
the T4 (16 GB, no bf16): fp16 base weights (LoRA params fp32), fp16 autocast + GradScaler, gradient checkpointing, and
batch 16 run as 4 micro-batches of 4. No 4-bit. Full 4000 train and 2000 eval examples were used.

| Model | Run | Trainable params | Train s | Peak GPU mem (GB) | Final train loss (last step / avg last 20) | Eval acc | Eval loss |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | zero-shot | 0 | – | – | – | 0.3190 | 5.687 |
| Qwen3-0.6B | LoRA r=16 (0.6B sweep, T4) | 10,092,544 | 169.9 | 11.14 | 0.393 / 0.253 | **0.9090** | **0.243** |
| Qwen3-4B | zero-shot | 0 | – | – | – | 0.7340 | 2.652 |
| Qwen3-4B | LoRA r=16 | 33,030,144 | 899.998 | 9.01 | 0.579 / 0.232 | **0.9155** | **0.209** |

Qwen3-4B on the first 1000 test examples: zero-shot 0.736, LoRA r=16 0.913 (full 2000 was used, so no subsetting).

Takeaways: the 4B model gains about 0.7 points accuracy over 0.6B r=16 (0.9155 vs 0.9090, 1 seed, within noise) and lower
eval loss (0.209 vs 0.243). Zero-shot 4B is far stronger than zero-shot 0.6B (73.4% vs 31.9%). Its LoRA training was ~5.3x slower
on T4 (900 s vs 170 s), and it needed 9.0 GB peak in fp16 with checkpointing.

Caveat: the 0.6B and 4B runs differ in precision (fp32 weights + fp16 autocast vs fp16 weights + checkpointing), so the
comparison is not perfectly controlled.

**GPU time (4B session): 36 min (2190 s)** from `colab new` (T4, ~20:34 UTC) to `colab stop` (~21:10 UTC).
Includes ~11 min of model load and zero-shot eval before LoRA training started (training itself: 900 s). Files: `runs_4b.json`, `progress_4b.log`,
`stdout_4b.log`, `train_4b.py`.
