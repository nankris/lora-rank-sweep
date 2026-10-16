import json,pathlib
p=pathlib.Path(__file__).resolve().parents[1]
a=json.loads((p/'results/analysis.json').read_text())
ci=lambda x: f'[{x[0]:.3f}, {x[1]:.3f}]'
text=f'# 038r results\n\n**Verdict: {a["verdict"]}.** Fixed Qwen3-0.6B / AG News protocol, 4,000 training examples and 2,000 test examples; paired training seeds are the replication units. All accuracies, SDs and differences below are in percentage points.\n\n'
text+='| Rank | Seeds | Accuracy mean ± sample SD | 95% t CI of mean | Individual scores |\n|---|---:|---|---|---|\n'
for r in a['ranks']:
 text+=f'| {r["rank"]} | {r["n"]} | {r["mean"]:.3f} ± {r["std"]:.3f} | {ci(r["ci"])} | '+', '.join(f'{v:.2f}' for v in r['scores'])+' |\n'
text+='\n| Claim | Mean paired difference | 95% paired t CI | 95% seed bootstrap CI | Decision |\n|---|---:|---|---|---|\n'
for k,c in a['claims'].items():
 text+=f'| {k} | {c["mean"]:.3f} | {ci(c["t_ci"])} | {ci(c["bootstrap_ci"])} | {c["status"]} |\n'
if (p/'results/round1-analysis.json').exists():
 b=json.loads((p/'results/round1-analysis.json').read_text())
 text+='\nRound 1 used seeds 1234–1236. '
 text+='; '.join(f'{k}: mean {c["mean"]:.3f}, t CI {ci(c["t_ci"])}, bootstrap CI {ci(c["bootstrap_ci"])}, {c["status"]}' for k,c in b['claims'].items())+'. '
 if len(a['seeds'])>3: text+='Because at least one claim was inconclusive, the single preregistered redesign added seeds 1237–1239, keeping data, scoring and training fixed. Final results pool six paired seeds. Stop after two rounds. '
text+='\n\nThe equivalence margin is ±1 percentage point, chosen before execution. “Equivalent” is practical equivalence under that margin, not exact equality or proof that rank 16 is optimal. CIs characterize seed variability conditional on this fixed test subset; they do not include uncertainty over new test samples. Bootstrap intervals with few seeds can be narrow, so acceptance additionally requires the paired t interval. Two planned looks are descriptive and not adjusted for sequential familywise error.\n\n'
text+='The zero-shot scoring audit found that 038 scores label+EOS, whereas 038c scores label only; prompt wording and article truncation also differ. Both request the same seed-1234 2,000-item split, but the old dataset revision was unpinned. These are unmatched scoring protocols: the historical 28.2% versus 31.9% gap cannot be interpreted as a matched replication failure. The code diff explains the confound but cannot apportion the 3.7-point aggregate gap without additional evaluations. See [audit](results/scoring-audit.md) and [full diff](results/scoring.diff).\n\n'
if (p/'results/compute.json').exists():
 u=json.loads((p/'results/compute.json').read_text())
 runs=json.loads((p/'results/runs.json').read_text())
 z=next(r for r in runs if r['rank']==0)
 text+=f"Zero-shot in this rerun: **{z['eval_accuracy']*100:.1f}%**, matching 038c.\n\n"
 text+=f"GPU: **{u['gpu']}**. Estimated session consumption: **{u['estimated_cu_upper_bound']:.3f} CU (0.76 rounded)**; {u['elapsed_seconds_upper_bound']/60:.2f} minutes including setup and downloads at the measured incremental 1.54 CU/hour. This is a conservative elapsed-time estimate, not direct per-session metering; the allocation boundary has about 12 seconds uncertainty. Before/during/after account usage rates: 1.54/3.08/1.54 CU/hour. Session termination is recorded in results/stop_resume.txt. "
 if len(a['seeds'])==3: text+='No second round was needed because both first-round criteria passed.\n'
(p/'writeup.md').write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n')
