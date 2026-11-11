"""Small negative checks for audit integrity and paired statistics; no timing."""
import csv,tempfile
from pathlib import Path
from analyze import interval,summarize,verify_samples

def row(case,role,r,t):
    return dict(case=case,role=role,round=str(r),seed='42',status='pass',median_ns=t,min_ns=t,p90_ns=t,
                bytes=0,allocs=0,samples=3,work=10,memory_bytes=1024,warm_seconds=.01)
rows=[]
for r in range(1,8):
    rows += [row('known','original',r,100*r),row('known','post850',r,200*r),
             row('known','originalAA',r,100*r),row('known','post850AA',r,200*r)]
s=summarize(rows,1000)
p=next(x for x in s if x['before']=='original' and x['after']=='post850')
assert p['ratio']==p['ci95_low']==p['ci95_high']==2
assert p['aa_ratio']==1
assert interval([1]*7,draws=1000)==(1,1)
try:summarize(rows[:-2],100)
except AssertionError:pass
else:raise AssertionError('missing pair accepted')
with tempfile.TemporaryDirectory() as d:
    root=Path(d);(root/'original').mkdir();f=root/'original'/'fixture-000.csv'
    Path(str(f)+'.fixture.toml').write_text('seed = 42\n')
    sample=Path(str(f)+'.known.samples.csv');sample.write_text('time_ns,gc_ns\n10,0\n20,0\n30,0\n')
    with f.open('w') as io:
        w=csv.DictWriter(io,fieldnames=['case','status','samples','median_ns','min_ns','bytes','allocs'])
        w.writeheader();w.writerow(dict(case='x/known',status='pass',samples=3,median_ns=20,min_ns=10,bytes=0,allocs=0))
    verify_samples(root,[row('x/known','original',1,20)])
    sample.write_text('time_ns,gc_ns\n10,0\n200,0\n300,0\n')
    try:verify_samples(root,[])
    except AssertionError:pass
    else:raise AssertionError('tampered samples accepted')
print('known ratios, exact A/A, missing pairs and tampered samples: pass')
