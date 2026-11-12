#!/usr/bin/env python3
"""Paired process medians, raw-sample verification and round bootstrap."""
import argparse,csv,hashlib,json,random,statistics,tomllib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('campaign',type=Path)
p.add_argument('--numerator',default='candidate');p.add_argument('--denominator',default='base')
p.add_argument('--output',required=True,type=Path);a=p.parse_args()
rows={}
campaign=json.loads((a.campaign/'campaign.json').read_text())
expected_rounds=set(range(campaign['argv']['start'],campaign['argv']['start']+campaign['argv']['rounds']))
expected_roles=set(campaign['roles'])
assert a.numerator in expected_roles and a.denominator in expected_roles
assert {p.stem for p in a.campaign.glob('*.toml')}=={f'{r:03}-{role}' for r in expected_rounds for role in expected_roles},'incomplete campaign'
expected_keys=None
binary_hash=None
for file in sorted(a.campaign.glob('*.toml')):
    roundstr,role=file.stem.split('-',1);r=int(roundstr)
    data=tomllib.loads(file.read_text())
    meta=data['meta']
    assert meta['seed']==860000+101*r
    assert Path(meta['source']).resolve()==Path(campaign['roles'][role]).resolve()
    assert meta['threads']==meta['blas_threads']==1
    assert meta['julia']=='1.13.3' and meta['opt']==2
    if binary_hash is None:binary_hash=meta['binary_sha256']
    assert meta['binary_sha256']==binary_hash,'Julia binary changed'
    assert meta['bounds']==(1 if campaign['argv']['check'] else 2)
    for path,digest in meta['source_hashes'].items():
        assert digest==campaign['source_hashes'][role][path]
    role_keys=set(data['results'])
    if expected_keys is None:expected_keys=role_keys
    # The historical source lacks the activity-check-only diagnostic.
    assert {k for k in role_keys if not k.endswith('/sweep_once')}=={k for k in expected_keys if not k.endswith('/sweep_once')}
    for key,value in data['results'].items():
        assert value['median_ns']==statistics.median(value['times_ns'])
        assert len(value['times_ns'])==len(value['gc_ns'])
        assert all(t>0 for t in value['times_ns'])
        rows[r,role,key]=value
keys=sorted({k for r,role,k in rows if role==a.denominator})
rng=random.Random(860)
summary=[]
for key in keys:
    rounds=sorted(r for r,role,k in rows if role==a.denominator and k==key and (r,a.numerator,k) in rows)
    if key.endswith('/sweep_once') and not any(role==a.numerator and k==key for r,role,k in rows):continue
    assert set(rounds)==expected_rounds,(key,'missing pair')
    den=[rows[r,a.denominator,key] for r in rounds];num=[rows[r,a.numerator,key] for r in rounds]
    ratios=[n['median_ns']/d['median_ns'] for n,d in zip(num,den)]
    boots=sorted(statistics.median(rng.choices(ratios,k=len(ratios))) for _ in range(10000))
    summary.append(dict(case=key,rounds=len(rounds),base_ns=statistics.median(d['median_ns'] for d in den),
                        candidate_ns=statistics.median(n['median_ns'] for n in num),ratio=statistics.median(ratios),
                        low=boots[249],high=boots[9749],minimum=min(ratios),maximum=max(ratios),
                        base_bytes=max(d['bytes'] for d in den),candidate_bytes=max(n['bytes'] for n in num),
                        base_allocs=max(d['allocs'] for d in den),candidate_allocs=max(n['allocs'] for n in num)))
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('w') as f:
    w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
print(json.dumps({'cases':len(summary),'slowdowns_over_5pct':[x for x in summary if x['ratio']>1.05],
                  'improvements_over_5pct':sum(x['ratio']<.95 for x in summary)},indent=2))
