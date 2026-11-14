#!/usr/bin/env python3
"""Sequentially finish an existing broad campaign, follow-ups and controls.

No stages overlap timing. Existing output directories are refused by the runner;
use a fresh prefix to reproduce. Leave an existing broad runner running, or run
it first, then invoke this script with its output path.
"""
import argparse,csv,json,os,shutil,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--broad',type=Path,required=True);p.add_argument('--prefix',type=Path,required=True)
p.add_argument('--julia',type=Path,required=True);p.add_argument('--sources',type=Path,required=True)
p.add_argument('--follow-rounds',type=int,default=21);p.add_argument('--control-rounds',type=int,default=7)
p.add_argument('--supplement-rounds',type=int,default=7);p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)))
a=p.parse_args();prefix=str(a.prefix.resolve());julia=str(a.julia.resolve());sources=str(a.sources.resolve())
status=Path(prefix+'-stages.json');stages=[]
def stage(name,cmd,log):
    print(name,flush=True);started=time.time();stages.append(dict(stage=name,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),status='running'))
    status.write_text(json.dumps(stages,indent=2)+'\n')
    with Path(log).open('w') as f:result=subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT)
    stages[-1]|={'status':'pass' if result.returncode==0 else 'failure','seconds':time.time()-started,'log':str(log)}
    status.write_text(json.dumps(stages,indent=2)+'\n')
    if result.returncode:raise SystemExit(f'{name} failed; see {log}')
while not (a.broad/'failures.json').exists():time.sleep(5)
assert json.loads((a.broad/'failures.json').read_text())==[]
broad_summary=Path(prefix+'-broad-summary')
stage('broad analysis',[sys.executable,HERE/'analyze.py',a.broad,broad_summary,'--verify','--plots'],prefix+'-broad-analysis.log')
for label in ('original','historical05','post848','post850'):
    stage('extra correctness '+label,[julia,'--startup-file=no','--threads=1','--check-bounds=yes','--project='+str(HERE),HERE/'extra_correctness.jl',Path(sources)/label],prefix+'-extra-'+label+'.log')
for label in ('original','historical05','post848','post850'):
    stage('separate compilation diagnostic '+label,[julia,'--startup-file=no','--threads=1','--check-bounds=no','-O2','--project='+str(HERE),HERE/'compile_diagnostic.jl',Path(sources)/label,prefix+'-compilation-'+label+'.toml'],prefix+'-compilation-'+label+'.log')
flags=broad_summary/'suspicious.txt';follow=Path(prefix+'-followup');fs=Path(prefix+'-followup-summary')
if flags.read_text().strip():
    stage('fresh-process regression follow-up',[sys.executable,HERE/'run.py','--julia',julia,'--sources',sources,'--output',follow,'--rounds',a.follow_rounds,'--start-round',8,'--cases-file',flags,'--cpu',a.cpu],prefix+'-followup.log')
    shutil.copyfile(flags,follow/'selection.txt')
    stage('follow-up analysis',[sys.executable,HERE/'analyze.py',follow,fs,'--verify'],prefix+'-followup-analysis.log')
stage('isolated allocation-only fresh-process controls',[sys.executable,HERE/'isolate_allocations.py','--julia',julia,'--sources',sources,'--output',prefix+'-allocation-isolation','--rounds',21,'--cpu',a.cpu],prefix+'-allocation-isolation.log')
sh=Path(prefix+'-supplement-harness');sc=Path(prefix+'-supplement-check');sr=Path(prefix+'-supplement');ss=Path(prefix+'-supplement-summary')
stage('generate supplemental harness',[sys.executable,HERE/'supplement.py','--harness',sh,'--sources',sources],prefix+'-supplement-generate.log')
common=['--julia',julia,'--sources',sources,'--cpu',a.cpu]
stage('supplemental correctness',[sys.executable,sh/'run.py',*common,'--output',sc,'--check','--rounds',1,'--roles','original,historical05,post848,post850'],prefix+'-supplement-check.log')
stage('supplemental seven fresh-process rounds',[sys.executable,sh/'run.py',*common,'--output',sr,'--rounds',a.supplement_rounds],prefix+'-supplement.log')
stage('supplemental analysis',[sys.executable,sh/'analyze.py',sr,ss,'--verify'],prefix+'-supplement-analysis.log')
# Investigate common setup for all primary warmed scalar paths, including
# apparent improvements. Allocation-dominated construction/copy/import cases
# have no scalar x restoration in their timed context and are excluded here.
rows=list(csv.DictReader((broad_summary/'summary.csv').open()))
selected=sorted({r['case'] for r in rows if r['before']=='original' and r['after']=='post850' and not r['case'].startswith('block/')
                 and r['case'].rsplit('/',1)[1] not in ('construct_empty','construct_populated','copy','copyto','dense_import')})
selection=Path(prefix+'-restore-selection.txt');selection.write_text('\n'.join(selected)+'\n')
rr=Path(prefix+'-restore');rs=Path(prefix+'-restore-summary')
stage('common full-field restoration control',[sys.executable,HERE/'restore_control.py','--julia',julia,'--sources',sources,'--cases-file',selection,'--output',rr,'--rounds',a.control_rounds,'--cpu',a.cpu],prefix+'-restore.log')
stage('restoration control analysis',[sys.executable,HERE/'analyze.py',rr,rs,'--verify'],prefix+'-restore-analysis.log')
print('All planned campaigns and raw-to-summary checks complete.',flush=True)
