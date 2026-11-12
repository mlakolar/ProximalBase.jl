#!/usr/bin/env python3
"""Sequential fresh-process, rotated source roles; one immutable tree per role."""
import argparse, hashlib, json, os, platform, shutil, subprocess, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--julia',required=True,type=Path)
p.add_argument('--output',required=True,type=Path)
p.add_argument('--role',action='append',required=True,help='label=/immutable/source/tree')
p.add_argument('--rounds',type=int,default=7)
p.add_argument('--start',type=int,default=1)
p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)))
p.add_argument('--check',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
roles=dict(x.split('=',1) for x in a.role)
assert len(roles)==len(a.role),'duplicate role label'
# Retain exactly the runtime harness even if the working copy is later edited.
snapshot=a.output.resolve()/'harness'
for name in ('audit860','audit852'):
    shutil.copytree(HERE.parent/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__'))
def hashes(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ('.jl','.py','.toml')}
provenance={'argv':vars(a),'python':platform.python_version(),'roles':roles,
            'source_hashes':{k:hashes(Path(v)/'src') for k,v in roles.items()},
            'harness_hashes':hashes(snapshot/'audit860'),'audit852_hashes':hashes(snapshot/'audit852'),
            'machine':subprocess.getoutput('uname -a'),'cpu':subprocess.getoutput('lscpu'),
            'governor':subprocess.getoutput(f'cat /sys/devices/system/cpu/cpu{a.cpu}/cpufreq/scaling_governor')}
(a.output/'campaign.json').write_text(json.dumps(provenance,indent=2,default=str)+'\n')
for r in range(a.start,a.start+a.rounds):
    labels=list(roles);shift=(r-1)%len(labels);labels=labels[shift:]+labels[:shift]
    if r%2==0:labels.reverse()
    for label in labels:
        seed=860000+101*r
        command=['taskset','-c',str(a.cpu),str(a.julia),'--startup-file=no','--threads=1',
                 '--check-bounds='+('yes' if a.check else 'no'),'-O2',
                 '--project='+str(snapshot/'audit852'),str(snapshot/'audit860/worker.jl'),
                 roles[label],str((a.output/f'{r:03}-{label}.toml').resolve()),str(seed)]
        start=time.monotonic()
        with (a.output/f'{r:03}-{label}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,
                           env=os.environ|{'JULIA_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        print(f'round {r} {label}: {time.monotonic()-start:.1f}s',flush=True)
for label,path in roles.items():
    assert hashes(Path(path)/'src')==provenance['source_hashes'][label],f'source changed: {label}'
(a.output/'complete.json').write_text(json.dumps({'rounds':a.rounds,'roles':list(roles)})+'\n')
