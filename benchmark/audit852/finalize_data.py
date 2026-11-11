#!/usr/bin/env python3
"""Post-timing validation and export; never overlaps the measurement campaigns.

Run after confirm_controls.py, or --wait for its verified outputs. Packaging,
report conclusions, independent final review and GitHub publication remain
separate reviewable steps. Use a fresh prefix/output set for a fresh replay.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parent.parent
p=argparse.ArgumentParser();p.add_argument('--prefix',type=Path,required=True)
p.add_argument('--broad',type=Path,required=True);p.add_argument('--julia',type=Path,required=True)
p.add_argument('--sources',type=Path,required=True);p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)))
p.add_argument('--wait',action='store_true');p.add_argument('--skip-plots',action='store_true')
a=p.parse_args();prefix=str(a.prefix.resolve());sources=a.sources.resolve();julia=a.julia.resolve();steps=[]
research=REPO/'docs/audit/iterate-performance-852/research';results=research.parent/'results'
def ready():
    raw=Path(prefix+'-raw-block-complete.json')
    if not raw.exists() or json.loads(raw.read_text())['status']!='pass':return False
    adapter=Path(prefix+'-adapter-control-complete.json')
    if not adapter.exists() or json.loads(adapter.read_text())['status']!='pass':return False
    for name in ['supplement-restore','active-restore']:
        f=Path(prefix+'-'+name+'-summary/analysis.json')
        if not f.exists() or not json.loads(f.read_text())['raw_samples_verified']:return False
    for name in ['restore','active-restore']:
        flags=Path(prefix+'-'+name+'-summary/suspicious.txt')
        if flags.read_text().strip():
            f=Path(prefix+'-'+name+'-confirmation-summary/analysis.json')
            if not f.exists() or not json.loads(f.read_text())['raw_samples_verified']:return False
    return True
if a.wait:
    while not ready():time.sleep(5)
assert ready(),'all timing and confirmation analyses must finish first'

def run(name,cmd):
    log=Path(prefix+'-post-'+name+'.log');start=time.time()
    print(name,flush=True)
    with log.open('w') as f:result=subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,cwd=REPO)
    steps.append(dict(stage=name,command=list(map(str,cmd)),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(start)),
      seconds=time.time()-start,status='pass' if result.returncode==0 else 'failure',log=log.name))
    (research/'post-timing-validation.json').write_text(json.dumps(steps,indent=2)+'\n')
    if result.returncode:raise SystemExit('failed '+name+'; inspect '+str(log))

python=sys.executable;native=Path(prefix+'-supplement-restore-harness')
run('analysis-invariants',[python,HERE/'check_analysis.py'])
for name,harness in [('primary',HERE),('supplement',native)]:
    run('strict-'+name,[python,HERE/'strict_correctness.py','--julia',julia,'--sources',sources,
       '--harness-source',harness,'--output',prefix+'-strict-'+name])
run('strict-raw-block',[python,HERE/'strict_correctness.py','--julia',julia,'--sources',sources,
    '--harness-source',prefix+'-raw-block-harness','--output',prefix+'-strict-raw-block'])
for source in ['original','historical05','post848','post850']:
    flags=['taskset','-c',a.cpu,julia,'--startup-file=no','--threads=1','-O2','--project='+str(HERE)]
    run('layout-'+source,flags+['--check-bounds=yes',HERE/'layout_correctness.jl',sources/source,native])
    run('generated-code-'+source,flags+['--check-bounds=no',HERE/'diagnose.jl',sources/source,
        prefix+'-generated-code/'+source,native])
    run('compilation-'+source,flags+['--check-bounds=no',HERE/'compile_diagnostic.jl',sources/source,
        prefix+'-pinned-compilation-'+source+'.toml'])
run('partition-supplement',[python,HERE/'split_confirmation.py',prefix+'-supplement-restore',native,prefix+'-supplement-partitions'])
run('export-results',[python,HERE/'export_results.py','--prefix',prefix,'--broad',a.broad,'--output',results])
run('independent-integrity',[python,HERE/'verify_integrity.py','--results',results,'--raw-parent',Path(prefix).parent,
       '--sources',sources,'--julia',julia])
run('regression-ledger',[python,HERE/'regression_ledger.py','--results',results])
if not a.skip_plots:run('plots',[python,HERE/'plot_report.py',results,results.parent/'figures'])
print('Post-timing validation/export complete; final report/review/packaging remain.',flush=True)
