#!/usr/bin/env python3
"""Additional independent sample, source, runtime and calibration checks.

analyze.py --verify checks the expected case/role/round Cartesian product and
exact fixture seeds. This pass separately checks p90, numeric validity and
common runtime/source provenance, and exports sample/GC/calibration diagnostics.
Run only after all timing, to avoid extra I/O during measurement.
"""
import argparse,csv,hashlib,json,math,statistics as st,tomllib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
p.add_argument('--raw-parent',type=Path,default=Path('/tmp'));p.add_argument('--sources',type=Path,required=True)
p.add_argument('--julia',type=Path,required=True);a=p.parse_args()
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
def tree(root):
    files=sorted(f for f in root.rglob('*') if f.is_file() and f.suffix in ('.jl','.py','.toml'))
    return hashlib.sha256('\n'.join(str(f.relative_to(root))+':'+sha(f) for f in files).encode()).hexdigest()
def p90(values):
    x=sorted(values);t=.9*(len(x)-1);j=math.floor(t);k=math.ceil(t)
    return x[j]+(t-j)*(x[k]-x[j])
versions={k:tree(a.sources/k/'src') for k in ('original','historical05','post848','post850')}
repo=Path(__file__).resolve().parent.parent.parent
runtime=json.loads((repo/'docs/audit/iterate-performance-852/research/runtime-hashes.json').read_text())
assert all(sha(repo/name)==expected for name,expected in runtime.items()),'frozen primary timing file changed'
binary=sha(a.julia);sample_diagnostics=[];calibrations=[];processes=0
for campaign in csv.DictReader((a.results/'campaigns.csv').open()):
    raw=a.raw_parent/campaign['raw_relative_directory']
    assert sha(raw/'observations.csv')==campaign['observations_sha256']
    config=json.loads((raw/'campaign.json').read_text());assert not config['argv']['persistent']
    args=config['argv'];start=args['start_round']
    expected={(f'round-{r:02}',role) for r in range(start,start+args['rounds']) for role in args['roles'].split(',')}
    provenance=list(raw.glob('round-*/*/provenance.toml'))
    actual={(file.parent.parent.name,file.parent.name) for file in provenance}
    assert actual==expected,('missing/extra process provenance',campaign['campaign'],expected-actual,actual-expected)
    for prov in provenance:
        d=tomllib.loads(prov.read_text());source='original' if d['label']=='originalAA' else 'post850' if d['label']=='post850AA' else d['label']
        assert d['source_digest']==versions[source],prov
        assert d['binary_sha256']==binary and d['julia']=='1.13.3',prov
        assert d['kernel']=='Linux' and d['threads']==d['blas_threads']==1,prov
        assert d['bounds']==2 and d['opt_level']==2,prov
        for endpoint in ('calibration_start','calibration_end'):
            cal=d[endpoint];assert st.median(cal['times_ns'])==cal['median_ns']
            assert min(cal['times_ns'])==cal['min_ns']
            calibrations.append(dict(campaign=campaign['campaign'],round=prov.parent.parent.name,role=d['label'],
              endpoint=endpoint,median_ns=cal['median_ns'],min_ns=cal['min_ns'],source_load_seconds=d['source_load_seconds'],harness_digest=d['harness_digest']))
        processes+=1
    counters={};gc_samples=0;total_samples=0;zero_samples=0
    for file in raw.rglob('fixture-*.csv'):
        if file.name.endswith('.samples.csv'):continue
        for row in csv.DictReader(file.open()):
            if row['status']!='pass':continue
            values=list(csv.DictReader(Path(str(file)+'.'+row['case'].rsplit('/',1)[1]+'.samples.csv').open()))
            times=[float(v['time_ns']) for v in values];gcs=[float(v['gc_ns']) for v in values]
            assert len(times)==int(row['samples'])>0
            assert all(math.isfinite(t) and t>=0 for t in times) and st.median(times)>0
            assert all(math.isfinite(g) and g>=0 for g in gcs)
            assert st.median(times)==float(row['median_ns']) and min(times)==float(row['min_ns'])
            assert math.isclose(p90(times),float(row['p90_ns']),rel_tol=1e-12,abs_tol=1e-7),(file,row['case'],'p90')
            assert float(row['bytes'])>=0 and float(row['allocs'])>=0
            counters[len(times)]=counters.get(len(times),0)+1;gc_samples+=sum(g>0 for g in gcs)
            total_samples+=len(times);zero_samples+=sum(t==0 for t in times)
    sample_diagnostics.append(dict(campaign=campaign['campaign'],trials=sum(counters.values()),
       sample_counts=dict(sorted(counters.items())),total_samples=total_samples,samples_with_gc=gc_samples,
       zero_time_samples=zero_samples))
with (a.results/'calibration.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(calibrations[0]));writer.writeheader();writer.writerows(calibrations)
(a.results/'sample-diagnostics.json').write_text(json.dumps(sample_diagnostics,indent=2)+'\n')
(a.results/'integrity.json').write_text(json.dumps({'status':'pass','processes':processes,
 'julia_binary_sha256':binary,'source_digests':versions,'primary_runtime_files_sha256':runtime,
 'checks':['frozen primary runtime files','sample numeric validity','median','minimum','p90',
 'GC nonnegative','source tree identity','Julia/binary identity','compiler/bounds flags','thread counts',
 'Linux','fresh-process mode','start/end calibration retained']},indent=2)+'\n')
print('Independent sample/provenance checks pass;',processes,'fresh source processes')
