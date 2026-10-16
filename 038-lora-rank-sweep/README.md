# #038 LoRA rank sweep — DROPPED (2026-10-15, budget cut from 10 CU to 5 CU)
Not completed. A T4 session (`tig038`) ran 19:36–19:51 UTC (~15 min, ≈0.27 CU). It got a zero-shot baseline only:
Qwen3-0.6B on 2,000 AG News test items scored 28.2% (label+EOS log-prob scoring, near the 25% chance level).
The first LoRA run then crashed (peft needs torchao>0.16 and Colab had 0.10). Uninstalling torchao fixed it, the sweep was
relaunched, and then the session was stopped when the budget changed. `scripts/sweep.py` is ready to rerun (do `pip uninstall -y torchao` first).
