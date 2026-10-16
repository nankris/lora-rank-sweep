# I let Claude Code run a GPU experiment on Colab by itself. It cost 70¢ and an hour of T4.

*Draft. Tech in General experiment #038c. 2026-10-15 IST.*

**The ML question:** does LoRA rank matter? Qwen3-0.6B, AG News topic classification, 4,000 examples, one epoch.

| | Accuracy |
|---|---|
| No fine-tuning | 31.9% |
| LoRA rank 4 | 89.6% |
| LoRA rank 16 | 90.9% |
| LoRA rank 64 | 90.7% |
| Qwen3-4B, no fine-tuning | 73.4% |
| Qwen3-4B, LoRA rank 16 | 91.6% |

Rank 16 is plenty; rank 64 has 4× the parameters and nothing to show for it. A model almost 7× bigger is much better before fine-tuning and barely better after.

**The agent question:** I gave Claude Code the `colab` command-line tool and a page of instructions: one session name, a GPU-minute cap, "always stop the session." It wrote the training script, started a T4, launched the job in the background, checked in every few minutes, hit an out-of-memory error (fixed it in one edit), hit a known library clash (fixed that too), pulled the results down, shut the GPU off and checked it was off.

- Opus 5.5 drove the main sweep: 27 turns, about 66¢ at API prices, 26 minutes.
- Haiku 5.5 took over the same session for the 4B follow-up: about 4¢.
- GPU: 59 minutes of T4 in total, roughly 1–2 Colab compute units.

**What went wrong wasn't the agent.** The L4 I wanted was taken by another job on the account, so it fell back to a T4. And the machine running Claude rebooted mid-run; the Colab job kept going, and a resumed session picked it up and finished cleanly.

*$ are API-equivalent estimates from Claude Code's `total_cost_usd`; Pro isn't billed per token.*
