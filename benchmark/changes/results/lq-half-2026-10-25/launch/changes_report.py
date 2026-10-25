"""Summarize all interleaved changes, retaining unfavorable and inconclusive cases."""
import csv,json,sys
from pathlib import Path
before,after,out=map(Path,sys.argv[1:4])
def read(path):
    with path.open() as stream:return list(csv.DictReader(stream))
old={(r['julia'],r['id']):r for r in read(before/'summary.csv')}
new={(r['julia'],r['id']):r for r in read(after/'summary.csv')}
rows=read(out/'comparison.csv')
lines=['# Ten-run interleaved comparison','',
       'Before/after use the same keyed input, Julia version, CPU and node within a pair. Order alternates across repetitions. Times are arithmetic means of ten run means. Ratios are geometric paired speedups with nominal t intervals on log ratios. Every valid run and unfavorable result is retained.','',
       '| Julia | Case | Before / after (µs) | Paired speedup [95% CI] | After maximum Julia bytes |',
       '|---|---|---:|---:|---:|']
for r in rows:
    key=(r['julia'],r['id']);a=new[key]
    lines.append(f"| {r['julia']} | `{r['id']}` | {float(r['before_mean_ns'])/1000:.3f} / {float(r['after_mean_ns'])/1000:.3f} | {float(r['paired_geometric_speedup']):.3f} [{float(r['ci95_low']):.3f}, {float(r['ci95_high']):.3f}] | {float(a['max_bytes']):.0f} |")
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
stats={v:dict(groups=sum(r['julia']==v for r in rows),
    faster=sum(r['julia']==v and float(r['ci95_low'])>1 for r in rows),
    slower=sum(r['julia']==v and float(r['ci95_high'])<1 for r in rows),
    slower_more_than_10pct=sum(r['julia']==v and float(r['paired_geometric_speedup'])<1/1.1 and float(r['ci95_high'])<1 for r in rows)) for v in sorted({r['julia'] for r in rows})}
(out/'status.json').write_text(json.dumps(stats,indent=2)+'\n')
print(json.dumps(stats,indent=2))
print('LARGEST_SLOWDOWNS')
for r in sorted(rows,key=lambda r:float(r['paired_geometric_speedup']))[:12]:print(r)
print('TARGET_GAINS')
for r in rows:
    if '/nuclear/rank4/gapped/nearby/' in r['id']:print(r)
