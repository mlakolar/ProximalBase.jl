#!/usr/bin/env python3
"""Verify #872 campaign samples and summarize paired per-round ratios of every role
against the baseline (standard library only; run with python3 -I)."""
import argparse,csv,json,random,statistics,tomllib
from collections import defaultdict
from pathlib import Path

def interval(values):
    rng=random.Random(872)
    draws=sorted(statistics.median(rng.choices(values,k=len(values))) for _ in range(10000))
    return draws[249],draws[9749]

def main():
    p=argparse.ArgumentParser();p.add_argument('campaigns',type=Path,nargs='+')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    by=defaultdict(dict);digests={};binaries=set();rawcount=0;roles_all=None
    for campaign in a.campaigns:
        assert json.loads((campaign/'failures.json').read_text())==[]
        cfg=json.loads((campaign/'campaign.json').read_text())
        roles=cfg['roles'];roles_all=roles_all or roles;assert roles==roles_all
        first=int(cfg['argv']['start_round']);rounds=range(first,first+int(cfg['argv']['rounds']))
        expected={(r,role) for r in rounds for role in roles};seen=set()
        for path in campaign.glob('round-*/*/provenance.toml'):
            meta=tomllib.loads(path.read_text());role=path.parent.name
            proc=(int(path.parent.parent.name.split('-')[-1]),role)
            assert proc not in seen;seen.add(proc)
            assert meta['label']==role and meta['julia']=='1.13.3'
            assert meta['threads']==meta['blas_threads']==1 and meta['bounds']==2 and meta['opt_level']==2
            digests.setdefault(role,meta['source_digest']);assert digests[role]==meta['source_digest']
            binaries.add(meta['binary_sha256'])
        assert seen==expected,(len(seen),len(expected))
        verified={}
        for path in campaign.glob('round-*/*/fixture-*.csv'):
            if path.name.endswith('.samples.csv'):continue
            r=int(path.parent.parent.name.split('-')[-1]);role=path.parent.name
            for row in csv.DictReader(path.open()):
                assert row['status']=='pass',(path,row)
                sp=Path(str(path)+'.'+row['case'].split('/')[-1]+'.samples.csv')
                samples=[float(x['time_ns']) for x in csv.DictReader(sp.open())]
                assert len(samples)==int(row['samples'])
                assert statistics.median(samples)==float(row['median_ns']) and min(samples)==float(row['min_ns'])
                key=(r,role,row['case']);assert key not in verified;verified[key]=row;rawcount+=len(samples)
        used=set()
        for row in csv.DictReader((campaign/'observations.csv').open()):
            r=int(row['round']);role=row['role'];key=(r,role,row['case'])
            assert key in verified and key not in used
            assert all(row[k]==v for k,v in verified[key].items())
            rot=roles[r%len(roles):]+roles[:r%len(roles)]
            assert int(row['order'])==rot.index(role)+1
            used.add(key);assert (r,role) not in by[row['case']];by[row['case']][r,role]=row
        assert used==set(verified)
    assert len(binaries)==1
    if 'baselineAA' in digests:assert digests['baseline']==digests['baselineAA']
    results=[]
    for case,data in sorted(by.items()):
        rounds=sorted({r for r,_ in data});roles=sorted({role for _,role in data})
        assert all((r,role) in data for r in rounds for role in roles)
        for r in rounds:assert len({data[r,role]['seed'] for role in roles})==1
        row={'case':case,'rounds':len(rounds)}
        for role in roles:
            rows=[data[r,role] for r in rounds]
            row[role+'_ns']=statistics.median(float(x['median_ns']) for x in rows)
            row[role+'_bytes']=max(int(x['bytes']) for x in rows)
            row[role+'_allocs']=max(int(x['allocs']) for x in rows)
        for role in roles:
            if role=='baseline':continue
            ratios=[float(data[r,role]['median_ns'])/float(data[r,'baseline']['median_ns']) for r in rounds]
            row[role+'_ratio']=statistics.median(ratios)
            row[role+'_low'],row[role+'_high']=interval(ratios)
            row[role+'_maxratio']=max(ratios)
        results.append(row)
    with (a.output/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    report={'cases':len(results),'samples':rawcount,'source_digests':digests,
            'julia_binary_sha256':next(iter(binaries))}
    for role in roles_all:
        if role=='baseline':continue
        rs=[x[role+'_ratio'] for x in results]
        report[role]={'median_of_case_ratios':statistics.median(rs),'min':min(rs),'max':max(rs),
                      'cases_above_1.05':sum(x>1.05 for x in rs),'cases_below_0.95':sum(x<0.95 for x in rs),
                      'alloc_increase_cases':sum(x[role+'_bytes']>x['baseline_bytes'] or x[role+'_allocs']>x['baseline_allocs'] for x in results)}
    (a.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
