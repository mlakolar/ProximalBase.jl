import csv, json, statistics as st, random, sys, pathlib
root=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else '/tmp/pb849-performance/results')
rows=list(csv.DictReader(open(root/'raw.csv')))
by={}
for r in rows: by.setdefault(r['case'],{}).setdefault(int(r['round']),{})[r['side']]=r
rng=random.Random(849)
output=[]
for case,rounds in by.items():
    ratios=[float(r['B']['median_ns'])/float(r['A']['median_ns']) for r in rounds.values()]
    controls=[float(r['Acontrol']['median_ns'])/float(r['A']['median_ns']) for r in rounds.values()]
    boot=sorted(st.median(rng.choices(ratios,k=len(ratios))) for _ in range(10000))
    aa=st.median(controls);ratio=st.median(ratios)
    d=dict(case=case,baseline_ns=st.median(float(r['A']['median_ns']) for r in rounds.values()),candidate_ns=st.median(float(r['B']['median_ns']) for r in rounds.values()),ratio=ratio,ci_low=boot[250],ci_high=boot[9749],aa_ratio=aa,aa_min=min(controls),aa_max=max(controls),rounds=len(ratios),slower5=sum(v>1.05 for v in ratios),baseline_bytes=st.median(int(r['A']['memory']) for r in rounds.values()),candidate_bytes=st.median(int(r['B']['memory']) for r in rounds.values()),baseline_allocs=st.median(int(r['A']['allocs']) for r in rounds.values()),candidate_allocs=st.median(int(r['B']['allocs']) for r in rounds.values()))
    d['flag']=ratio>1.05 and d['ci_low']>1.05
    output.append(d)
with open(root/'summary.csv','w') as f:
    w=csv.DictWriter(f,fieldnames=output[0].keys());w.writeheader();w.writerows(sorted(output,key=lambda r:-r['ratio']))
json.dump(output,open(root/'summary.json','w'),indent=2)
print(f'{len(output)} cases; {len(rows)} raw rows')
for r in sorted(output,key=lambda r:-r['ratio']):
    if r['ratio']>1.04 or r['candidate_bytes']>r['baseline_bytes']:
        print(f"{r['case']}: B/A={r['ratio']:.3f} CI=[{r['ci_low']:.3f},{r['ci_high']:.3f}] A/A={r['aa_ratio']:.3f} bytes={r['baseline_bytes']}->{r['candidate_bytes']} {'FLAG' if r['flag'] else ''}")
