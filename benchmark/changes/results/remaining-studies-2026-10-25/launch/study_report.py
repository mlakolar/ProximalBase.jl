"""Within-process, within-seed comparisons of benchmark-only alternatives."""
import csv,json,math,statistics,sys
from pathlib import Path
root,summary=map(Path,sys.argv[1:3])
def read(path,sep=','):
    with path.open() as stream:return list(csv.DictReader(stream,delimiter=sep))
rows=read(summary/'summary.csv')
key=lambda r:(r['julia'],r['family'],r['dtype'],r['shape'],r['layout'],r['phase'],r['name'])
index={key(r):r for r in rows}
measurements={}
for path in root.glob('*-metrics.tsv'):
    version=path.name.split('-rep')[0]
    for r in read(path,'\t'):measurements[(version,r['id'],int(r['rep']))]=float(r['mean_ns'])
contrasts=[]
for r in rows:
    name=r['name']
    if r['family']=='ls_backend' and name.startswith('Cholesky/'):
        before_name=name.replace('Cholesky/','ThinSVD/',1)
    elif r['family']=='slope_sort' and not name.startswith('BaseDefault/'):
        before_name='BaseDefault/'+name.split('/',1)[1]
    else:continue
    before=index[key(r)[:-1]+(before_name,)]
    logs=[math.log(measurements[(r['julia'],before['id'],rep)]/measurements[(r['julia'],r['id'],rep)]) for rep in range(1,11)]
    mu=statistics.mean(logs);half=2.262*statistics.stdev(logs)/math.sqrt(10)
    contrasts.append(dict(julia=r['julia'],family=r['family'],name=name,dtype=r['dtype'],shape=r['shape'],phase=r['phase'],before_mean_ns=before['mean_ns'],after_mean_ns=r['mean_ns'],paired_speedup=math.exp(mu),ci95_low=math.exp(mu-half),ci95_high=math.exp(mu+half),after_max_bytes=r['max_bytes']))
with (summary/'contrasts.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(contrasts[0]),lineterminator='\n');writer.writeheader();writer.writerows(contrasts)
lines=['# Sorting and least-squares prototype study','',
       'Ten fresh processes per Julia version, with common seeded inputs and shuffled case order within each process. These are kernel/backend prototypes; package dispatch and defaults are unchanged. Least-squares changing-step traces contain four proximal calls. Constructor costs include setup; fixed-step calls reuse factors. Sorting results exclude pooling and output restoration.','',
       '| Julia | Alternative | Type | Shape | Before / after (µs) | Paired speedup [95% CI] | After Julia bytes |',
       '|---|---|---|---|---:|---:|---:|']
for r in contrasts:
    lines.append(f"| {r['julia']} | {r['name']} | {r['dtype']} | {r['shape']} | {float(r['before_mean_ns'])/1000:.3f} / {float(r['after_mean_ns'])/1000:.3f} | {r['paired_speedup']:.3f} [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}] | {float(r['after_max_bytes']):.0f} |")
    if r['dtype']=='Float64' and r['shape'] in ('100000','1024x256'):print(json.dumps(r))
(summary/'STUDY.md').write_text('\n'.join(lines)+'\n')
