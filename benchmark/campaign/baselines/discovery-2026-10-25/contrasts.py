"""Campaign-specific contrasts; run under Slurm on the retained observations."""
import csv,json,math,statistics,sys
from pathlib import Path
root,out=map(Path,sys.argv[1:3]);out.mkdir(exist_ok=True)
with (out/'summary.csv').open() as stream:
    rows=list(csv.DictReader(stream))
measurements={}
for path in root.glob('*-metrics.tsv'):
    version=path.name.split('-rep')[0]
    with path.open() as stream:
        for r in csv.DictReader(stream,delimiter='\t'):
            measurements[(version,r['id'],int(r['rep']))]=float(r['mean_ns'])
index={(r['julia'],r['family'],r['name'],r['dtype'],r['shape'],r['layout'],r['phase']):r for r in rows}
contrasts=[]
for r in rows:
    other=None;kind=''
    if r['family']=='partial' and r['name'].endswith('/prox!') and '_full/' not in r['name']:
        other=(r['julia'],r['family'],r['name'].replace('/prox!','_full/prox!'),r['dtype'],r['shape'],r['layout'],r['phase'])
        kind='full / nearby-call partial'
    if r['family'] in ('least_squares','affine_constrained') and r['phase']=='warm' and r['name'].endswith('/prox!'):
        other=(r['julia'],r['family'],r['name'].removesuffix('/prox!'),r['dtype'],r['shape'],r['layout'],'construct_and_first_prox')
        kind='construction+first / warmed prox'
    if other is None or other not in index:continue
    before=index[other]
    ratios=[math.log(measurements[(r['julia'],before['id'],rep)]/measurements[(r['julia'],r['id'],rep)]) for rep in range(1,11)]
    mu=statistics.mean(ratios);half=2.262*statistics.stdev(ratios)/math.sqrt(10)
    contrasts.append(dict(kind=kind,julia=r['julia'],dtype=r['dtype'],shape=r['shape'],name=r['name'],
        before_id=before['id'],after_id=r['id'],before_mean_us=float(before['mean_ns'])/1000,
        after_mean_us=float(r['mean_ns'])/1000,paired_ratio=math.exp(mu),ci95_low=math.exp(mu-half),ci95_high=math.exp(mu+half)))
with (out/'contrasts.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(contrasts[0]));writer.writeheader();writer.writerows(contrasts)
versions=sorted({r['julia'] for r in rows})
stats={}
for version in versions:
    subset=[r for r in rows if r['julia']==version]
    cvs=sorted(float(r['cv']) for r in subset)
    stats[version]=dict(groups=len(subset),expected_zero_groups=sum(r['expected_zero']=='true' for r in subset),
        unexpected_allocation_groups=sum(r['expected_zero']=='true' and float(r['max_bytes'])>0 for r in subset),
        cv_median=statistics.median(cvs),cv_p90=cvs[math.ceil(.9*len(cvs))-1],cv_max=max(cvs),
        cv_above_15pct=sum(v>.15 for v in cvs),cv_above_50pct=sum(v>.5 for v in cvs),
        fewer_than_40_samples_groups=sum(int(r['min_samples'])<40 for r in subset),
        min_samples=min(int(r['min_samples']) for r in subset),
        gc_affected_groups=sum(float(r['max_gc_share'])>0 for r in subset))
(out/'insights.json').write_text(json.dumps(stats,indent=2)+'\n')
lines=['# Baseline contrasts','',
    'Ten paired repetitions within each Julia version. Times are arithmetic means of run means. Ratios use geometric within-repetition ratios and a t interval on log ratios. Rows compare different APIs/cache phases of the same package, not a before/after implementation change.','',
    '| Julia | Phase contrast | Type | Shape | Operation | Before / after (µs) | Paired ratio [95% interval] |',
    '|---|---|---|---|---|---:|---:|']
for r in contrasts:
    lines.append(f"| {r['julia']} | {r['kind']} | {r['dtype']} | {r['shape']} | {r['name']} | {r['before_mean_us']:.2f} / {r['after_mean_us']:.2f} | {r['paired_ratio']:.2f} [{r['ci95_low']:.2f}, {r['ci95_high']:.2f}] |")
lines+=['','## Representative expensive warmed operators','',
    '| Julia | Case | Mean ± SD (µs) | 95% interval (µs) | Maximum bytes |','|---|---|---:|---:|---:|']
selected=[r for r in rows if r['dtype']=='Float64' and r['phase']=='warm' and r['family'] in ('separable','ordered','constrained','smooth','solver_traces','logdet') and (r['shape']=='100000' or r['shape']=='256' or r['shape']=='1024x256' or r['shape']=='256x256')]
for r in sorted(selected,key=lambda r:float(r['mean_ns']),reverse=True)[:45]:
    lines.append(f"| {r['julia']} | `{r['id']}` | {float(r['mean_ns'])/1000:.2f} ± {float(r['sd_between_runs_ns'])/1000:.2f} | [{float(r['ci95_low_ns'])/1000:.2f}, {float(r['ci95_high_ns'])/1000:.2f}] | {float(r['max_bytes']):.0f} |")
(out/'CONTRASTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(stats,indent=2))
