#!/usr/bin/env python3
"""Verify individual #852 samples, then summarize paired round ratios."""
import argparse,csv,json,random,statistics
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
    by=defaultdict(dict);rawcount=0
    for campaign in a.campaigns:
        assert json.loads((campaign/'failures.json').read_text())==[]
        for path in campaign.glob('round-*/*/fixture-*.csv'):
            if path.name.endswith('.samples.csv'):continue
            for row in csv.DictReader(path.open()):
                assert row['status']=='pass',(path,row)
                samplepath=Path(str(path)+'.'+row['case'].split('/')[-1]+'.samples.csv')
                samples=[float(r['time_ns']) for r in csv.DictReader(samplepath.open())]
                assert len(samples)==int(row['samples'])
                assert statistics.median(samples)==float(row['median_ns'])
                assert min(samples)==float(row['min_ns'])
                rawcount+=len(samples)
        for row in csv.DictReader((campaign/'observations.csv').open()):
            key=row['case'];r=int(row['round']);role=row['role']
            assert (r,role) not in by[key]
            by[key][r,role]=row
    results=[]
    for case,data in sorted(by.items()):
        rounds=sorted({r for r,role in data})
        roles=sorted({role for r,role in data})
        assert all((r,role) in data for r in rounds for role in roles)
        row={'case':case,'rounds':len(rounds)}
        for role in roles:
            rows=[data[r,role] for r in rounds]
            row[role+'_ns']=statistics.median(float(x['median_ns']) for x in rows)
            row[role+'_bytes']=statistics.median(int(x['bytes']) for x in rows)
            row[role+'_allocs']=statistics.median(int(x['allocs']) for x in rows)
        for numer,denom,label in [('candidate','baseline','ratio'),('baselineAA','baseline','AA'),
                                 ('candidate','original','vs_original')]:
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
        'rows':sum(map(len,by.values())),'regressions_above_5_percent':len(flagged)},indent=2)+'\n')
    print((a.output/'verification.json').read_text())
if __name__=='__main__':main()
