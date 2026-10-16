# 038r: LoRA rank across seeds

## Preregistered hypothesis and decision rule (before execution)
Hypothesis: on Qwen3-0.6B AG News with the 038c data and scoring, r16 exceeds r4 and r64 is practically equivalent to r16.
Round 1: ranks 4,16,64 each on seeds 1234,1235,1236, paired by seed. Fixed 4,000 training and 2,000 test examples shuffled with data seed 1234. Exact 038c prompt, label-only summed-log-prob scoring, optimizer, precision, dropout=0 and training schedule. Model initialization and training-order seed vary together.
Compute paired seed differences in accuracy; use 95% Student-t confidence intervals (seed is the replication unit; sample SD, df=n-1). Also report paired nonparametric seed bootstrap 95% percentile CIs (100,000 draws), required as a robustness check: a claim is accepted only if both methods agree. r16>r4 is supported if both lower bounds exceed zero, contradicted if both upper bounds are <=0, otherwise inconclusive. r64≈r16 is supported if both intervals are wholly inside [-1,+1] percentage points, contradicted if both intervals lie wholly outside the band, otherwise inconclusive. Overall PROVEN if both supported; DISPROVEN if either contradicted; otherwise STILL INCONCLUSIVE. These are joint descriptive criteria, not a population-wide rank optimum claim.
If either claim is inconclusive, redesign exactly once by adding seeds 1237,1238,1239 (six total per arm), keeping all other settings fixed; apply the same rule to pooled six seeds and stop. No additional rounds. This increases independent training replicates rather than treating 2,000 correlated item predictions as independent training runs.
Budget <=5 CU. L4 only unless model cannot fit. Use own session 038r-rank-seeds; stop on completion/failure. Save predictions, data fingerprints, package versions and GPU metadata. Shared account balance changes cannot alone measure this session's CU; report session duration times measured incremental usage rate, alongside before/after snapshots.

## Execution and outcome
Resumed the existing partial folder as requested. Earlier allocation failures and the original local commit are retained as history; no training round had completed then. On 2026-10-15 the L4 allocation succeeded. Round 1 completed all nine runs on seeds 1234–1236. **PROVEN under the preregistered criteria**: both r16>r4 and r64≈r16 are supported, so no second round was needed.

| Rank | Mean accuracy ± sample SD | 95% t CI of mean |
|---|---|---|
| 4 | 89.500 ± 0.346% | [88.639, 90.361]% |
| 16 | 90.617 ± 0.144% | [90.258, 90.975]% |
| 64 | 90.733 ± 0.257% | [90.096, 91.371]% |

Paired r16−r4: +1.117 percentage points, 95% t CI [0.615, 1.619], seed bootstrap [1.000, 1.350]. Paired r64−r16: +0.117 points, t CI [−0.196, 0.429], bootstrap [0.000, 0.250]; both intervals lie inside the preregistered ±1 point equivalence margin. Conditional on this dataset, protocol and seeds; practical equivalence does not establish exact equality or a universal optimal rank.

Zero-shot baseline reproduced 038c at 31.9%. The scoring-code audit identifies label+EOS versus label-only scoring, different prompts and different truncation in 038 versus 038c. It explains why their historical baselines are unmatched, without numerically attributing the 3.7-point gap. See results/scoring-audit.md and results/scoring.diff.

GPU: **NVIDIA L4**. **Estimated compute used: 0.756 CU (rounded 0.76 CU)**, conservative upper estimate from 29.46 minutes × 1.54 CU/hour. Allocation happened between the before/allocated snapshots, a roughly 12-second uncertainty. Account usage rate was 1.54 before, 3.08 with our session, and 1.54 after stopping; the shared balance fell by more because another assignment remained active. See results/compute.json and usage_resume_*.txt. The session `038r-rank-seeds` was explicitly stopped; results/stop_resume.txt confirms termination. No other session was touched.

## Reproduce and artifacts
`source /workspace/tools/setup.sh`, create an L4 session, execute scripts/setup_remote.py, then scripts/train.py with `colab exec`. Round 1 defaults to seeds 1234,1235,1236. If required by the decision rule, rerun the same script in the same session with `SEEDS=1237,1238,1239`; it resumes completed runs. Download `/content/out038r` to results/, then run `python scripts/analyze.py` and `python scripts/report.py` locally. Always stop the session. Model and dataset were public; no private data is included.

results/runs.json contains all metrics; pred_*.json contains all 2,000 predictions for each model; metadata.json records GPU, package versions and data fingerprints; progress.log is the complete remote log (round1.log contains the console stream, which stalled locally). reference/ preserves original source snapshots. chart.png plots mean ± sample SD and paired seed trajectories. results/round1-analysis.json freezes the first-round decision. The source scoring protocol was preserved; predictions, data sizes, run completeness and analysis were checked locally. Originals were not modified.
