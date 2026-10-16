# 038r results

**Verdict: PROVEN.** Fixed Qwen3-0.6B / AG News protocol, 4,000 training examples and 2,000 test examples; paired training seeds are the replication units. All accuracies, SDs and differences below are in percentage points.

| Rank | Seeds | Accuracy mean ± sample SD | 95% t CI of mean | Individual scores |
|---|---:|---|---|---|
| 4 | 3 | 89.500 ± 0.346 | [88.639, 90.361] | 89.70, 89.70, 89.10 |
| 16 | 3 | 90.617 ± 0.144 | [90.258, 90.975] | 90.70, 90.70, 90.45 |
| 64 | 3 | 90.733 ± 0.257 | [90.096, 91.371] | 90.95, 90.80, 90.45 |

| Claim | Mean paired difference | 95% paired t CI | 95% seed bootstrap CI | Decision |
|---|---:|---|---|---|
| r16 > r4 | 1.117 | [0.615, 1.619] | [1.000, 1.350] | supported |
| r64 ≈ r16 | 0.117 | [-0.196, 0.429] | [0.000, 0.250] | supported |

Round 1 used seeds 1234–1236. r16 > r4: mean 1.117, t CI [0.615, 1.619], bootstrap CI [1.000, 1.350], supported; r64 ≈ r16: mean 0.117, t CI [-0.196, 0.429], bootstrap CI [0.000, 0.250], supported.

The equivalence margin is ±1 percentage point, chosen before execution. “Equivalent” is practical equivalence under that margin, not exact equality or proof that rank 16 is optimal. CIs characterize seed variability conditional on this fixed test subset; they do not include uncertainty over new test samples. Bootstrap intervals with few seeds can be narrow, so acceptance additionally requires the paired t interval. Two planned looks are descriptive and not adjusted for sequential familywise error.

The zero-shot scoring audit found that 038 scores label+EOS, whereas 038c scores label only; prompt wording and article truncation also differ. Both request the same seed-1234 2,000-item split, but the old dataset revision was unpinned. These are unmatched scoring protocols: the historical 28.2% versus 31.9% gap cannot be interpreted as a matched replication failure. The code diff explains the confound but cannot apportion the 3.7-point aggregate gap without additional evaluations. See [audit](results/scoring-audit.md) and [full diff](results/scoring.diff).

Zero-shot in this rerun: **31.9%**, matching 038c.

GPU: **NVIDIA L4**. Estimated session consumption: **0.756 CU (0.76 rounded)**; 29.46 minutes including setup and downloads at the measured incremental 1.54 CU/hour. This is a conservative elapsed-time estimate, not direct per-session metering; the allocation boundary has about 12 seconds uncertainty. Before/during/after account usage rates: 1.54/3.08/1.54 CU/hour. Session termination is recorded in results/stop_resume.txt. No second round was needed because both first-round criteria passed.
