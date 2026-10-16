import json, pathlib
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=pathlib.Path(__file__).resolve().parents[1]
runs=json.loads((P/'results/runs.json').read_text())
seeds=sorted({r['seed'] for r in runs if r['rank']})
a={k:np.array([next(r['eval_accuracy']*100 for r in runs if r['rank']==k and r['seed']==s) for s in seeds]) for k in [4,16,64]}
rng=np.random.default_rng(380)
rows=[]
for k,v in a.items():
    half=t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))
    rows.append(dict(rank=k,n=len(v),mean=v.mean(),std=v.std(ddof=1),ci=[v.mean()-half,v.mean()+half],scores=v.tolist()))
claims={}
for name,d in [('r16 > r4',a[16]-a[4]),('r64 ≈ r16',a[64]-a[16])]:
    half=t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
    ci=[float(d.mean()-half),float(d.mean()+half)]
    boot=np.quantile(d[rng.integers(len(d),size=(100000,len(d)))].mean(1),[.025,.975]).tolist()
    if name=='r16 > r4':
        status='supported' if min(ci[0],boot[0])>0 else 'contradicted' if max(ci[1],boot[1])<=0 else 'inconclusive'
    else:
        status='supported' if min(ci[0],boot[0])>-1 and max(ci[1],boot[1])<1 else 'contradicted' if min(ci[0],boot[0])>1 or max(ci[1],boot[1])< -1 else 'inconclusive'
    claims[name]=dict(mean=float(d.mean()),t_ci=ci,bootstrap_ci=boot,status=status,differences=d.tolist())
verdict='DISPROVEN' if any(c['status']=='contradicted' for c in claims.values()) else 'PROVEN' if all(c['status']=='supported' for c in claims.values()) else 'STILL INCONCLUSIVE'
out=dict(seeds=seeds,ranks=rows,claims=claims,verdict=verdict)
(P/'results/analysis.json').write_text(json.dumps(out,indent=2))
fig,ax=plt.subplots(figsize=(7,4))
x=np.arange(3)
ax.errorbar(x,[r['mean'] for r in rows],yerr=[r['std'] for r in rows],fmt='o',capsize=7,label='Mean ± sample SD')
for j,s in enumerate(seeds): ax.plot(x,[a[k][j] for k in [4,16,64]],alpha=.3,lw=1)
ax.set(xticks=x,xticklabels=['r4','r16','r64'],ylabel='AG News accuracy (%)',title=f'038r: {len(seeds)} paired seeds; 2,000 fixed test items')
ax.legend(); fig.tight_layout(); fig.savefig(P/'chart.png',dpi=180)
print(json.dumps(out,indent=2))
