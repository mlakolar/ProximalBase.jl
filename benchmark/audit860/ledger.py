#!/usr/bin/env python3
"""Account for every >5% flag without pooling discovery and confirmation."""
import argparse,csv
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('stats',type=Path);p.add_argument('output',type=Path)
a=p.parse_args()
def read(name):
    return {r['case']:r for r in csv.DictReader((a.stats/(name+'.csv')).open())}
rows=[]
for strategy,broadname,aaname in (
    ('bulk','discovery-bulk','discovery-AA'),
    ('outlined','placement-outlined','placement-AA'),
    ('singleton','placement-singleton','placement-AA')):
    broad=read(broadname);confirm=read('confirmation-'+strategy)
    aa_broad=read(aaname);aa_confirm=read('confirmation-AA')
    assert broad.keys()==confirm.keys()==aa_broad.keys()==aa_confirm.keys()
    for key in sorted(broad):
        b,c=broad[key],confirm[key]
        br,cr=float(b['ratio']),float(c['ratio'])
        if max(br,cr)<=1.05:continue
        op=key.rsplit('/',1)[-1]
        kind='bcd' if op.startswith('residual_') else 'copy' if op.startswith('copy_') else 'diagnostic'
        if cr<=1.05:
            outcome='initial flag not repeated in fresh confirmation'
        elif float(c['low'])>1.05:
            outcome='confirmation interval entirely above 1.05; strategy rejected'
        else:
            outcome='confirmation median flag; interval includes <=1.05; strategy rejected for other conclusive regressions'
        mechanism=('unchanged timed API body; compiler/placement context, not a bulk-copy claim' if kind=='diagnostic' or op.startswith('residual_bcd') else
                   'extra singleton guards on Base fallback; both callers retain one copy call' if strategy=='singleton' else
                   'outlined recognition/reduction/call boundary; zero writes have little activation work to remove' if strategy=='outlined' else
                   'inlined bulk code shape; per-layout loop evidence retained; residual arithmetic remains vectorized')
        rows.append(dict(strategy=strategy,case=key,kind=kind,broad_ratio=br,
            confirm_ratio=cr,confirm_low=float(c['low']),confirm_high=float(c['high']),
            repeated_median_above_5pct=br>1.05 and cr>1.05,
            base_ns=c['base_ns'],candidate_ns=c['candidate_ns'],
            added_us=(float(c['candidate_ns'])-float(c['base_ns']))/1000,
            timer_limited=min(float(c['base_ns']),float(c['candidate_ns']))<100,
            broad_AA=aa_broad[key]['ratio'],confirm_AA=aa_confirm[key]['ratio'],
            allocation_delta=int(c['candidate_bytes'])-int(c['base_bytes']),
            outcome=outcome,investigation=mechanism))
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(f'{len(rows)} flagged contrasts accounted for; no candidate accepted')
