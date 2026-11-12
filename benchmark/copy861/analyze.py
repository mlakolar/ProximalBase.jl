#!/usr/bin/env python3
"""Verify individual #852 samples, then summarize paired round ratios."""
import argparse,csv,json,random,statistics,tomllib
from collections import defaultdict
from pathlib import Path

def interval(values):
    rng=random.Random(861)
    draws=sorted(statistics.median(rng.choices(values,k=len(values))) for _ in range(10000))
    return draws[249],draws[9749]

def main():
    p=argparse.ArgumentParser();p.add_argument('campaigns',type=Path,nargs='+')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    by=defaultdict(dict);rawcount=0;source_digests={};harness_digests={};binary_hashes=set()
    for campaign in a.campaigns:
        assert json.loads((campaign/'failures.json').read_text())==[]
        config=json.loads((campaign/'campaign.json').read_text())
        first=int(config['argv']['start_round'])
        expected_rounds=range(first,first+int(config['argv']['rounds']))
        expected_roles=config['argv']['roles'].split(',')
        expected_processes={(r,role) for r in expected_rounds for role in expected_roles}
        seen_processes=set()
        verified={};verified_seeds={}
        for path in campaign.glob('round-*/*/provenance.toml'):
            meta=tomllib.loads(path.read_text());role=path.parent.name
            process=(int(path.parent.parent.name.split('-')[-1]),role)
            assert process not in seen_processes
            seen_processes.add(process)
            assert meta['label']==role
            assert meta['julia']=='1.13.3' and meta['threads']==meta['blas_threads']==1
            assert meta['bounds']==2 and meta['opt_level']==2
            assert meta['kernel']=='Linux'
            source_digests.setdefault(role,meta['source_digest'])
            assert source_digests[role]==meta['source_digest']
            harness_digests.setdefault(str(campaign),meta['harness_digest'])
            assert harness_digests[str(campaign)]==meta['harness_digest']
            binary_hashes.add(meta['binary_sha256'])
        assert seen_processes==expected_processes
        for path in campaign.glob('round-*/*/fixture-*.csv'):
            if path.name.endswith('.samples.csv'):continue
            r=int(path.parent.parent.name.split('-')[-1]);role=path.parent.name
            seed=tomllib.loads(Path(str(path)+'.fixture.toml').read_text())['seed']
            for row in csv.DictReader(path.open()):
                assert row['status']=='pass',(path,row)
                samplepath=Path(str(path)+'.'+row['case'].split('/')[-1]+'.samples.csv')
                samples=[float(r['time_ns']) for r in csv.DictReader(samplepath.open())]
                assert len(samples)==int(row['samples'])
                assert statistics.median(samples)==float(row['median_ns'])
                assert min(samples)==float(row['min_ns'])
                key=(r,role,row['case'])
                assert key not in verified
                verified[key]=row;verified_seeds[key]=seed
                rawcount+=len(samples)
        used=set()
        for row in csv.DictReader((campaign/'observations.csv').open()):
            key=row['case'];r=int(row['round']);role=row['role']
            rawkey=(r,role,key)
            assert rawkey in verified and rawkey not in used
            assert all(row[k]==v for k,v in verified[rawkey].items())
            assert int(row['seed'])==verified_seeds[rawkey]
            rotation=expected_roles[r%len(expected_roles):]+expected_roles[:r%len(expected_roles)]
            assert int(row['order'])==rotation.index(role)+1
            used.add(rawkey)
            assert (r,role) not in by[key]
            by[key][r,role]=row
        assert used==set(verified)
        cases={case for r,role,case in used}
        assert used=={(r,role,case) for r,role in expected_processes for case in cases}
    assert len(binary_hashes)==1
    if 'baselineAA' in source_digests:
        assert source_digests['baseline']==source_digests['baselineAA']
    if 'candidateAA' in source_digests:
        assert source_digests['candidate']==source_digests['candidateAA']
    results=[]
    for case,data in sorted(by.items()):
        rounds=sorted({r for r,role in data})
        roles=sorted({role for r,role in data})
        assert all((r,role) in data for r in rounds for role in roles)
        for r in rounds:
            assert len({data[r,role]['seed'] for role in roles})==1
        row={'case':case,'rounds':len(rounds)}
        for role in roles:
            rows=[data[r,role] for r in rounds]
            row[role+'_ns']=statistics.median(float(x['median_ns']) for x in rows)
            row[role+'_bytes']=statistics.median(int(x['bytes']) for x in rows)
            row[role+'_allocs']=statistics.median(int(x['allocs']) for x in rows)
        for numer,denom,label in [('candidate','baseline','ratio'),('baselineAA','baseline','AA'),
                                 ('candidate','original','vs_original'),
                                 ('candidateAA','candidate','candidateAA')]:
            if numer not in roles or denom not in roles:continue
            ratios=[float(data[r,numer]['median_ns'])/float(data[r,denom]['median_ns']) for r in rounds]
            row[label]=statistics.median(ratios)
            row[label+'_low'],row[label+'_high']=interval(ratios)
        results.append(row)
    with (a.output/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    flagged=[x['case'] for x in results if x.get('ratio',1)>1.05]
    (a.output/'regressions.txt').write_text('\n'.join(flagged)+'\n')
    (a.output/'verification.json').write_text(json.dumps({'cases':len(results),'samples':rawcount,
        'rows':sum(map(len,by.values())),'regressions_above_5_percent':len(flagged),
        'source_digests':source_digests,'harness_digests':harness_digests,
        'julia_binary_sha256':next(iter(binary_hashes))},indent=2)+'\n')
    print((a.output/'verification.json').read_text())
if __name__=='__main__':main()
