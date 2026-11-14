#!/usr/bin/env python3
"""Sequential paired process controller; source archives are never edited."""
import argparse,csv,hashlib,io,json,os,platform,re,subprocess,tarfile,time
from pathlib import Path
from spec import scenarios,VERSIONS
HERE=Path(__file__).resolve().parent; REPO=HERE.parent.parent

def archive_sources(parent):
    parent.mkdir(parents=True,exist_ok=True)
    for label,sha in VERSIONS.items():
        root=parent/label
        if root.exists():
            assert (root/'.audit852-sha').read_text().strip()==sha,root
            continue
        root.mkdir()
        data=subprocess.check_output(['git','-C',str(REPO),'archive',sha])
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:tf.extractall(root,filter='data')
        (root/'.audit852-sha').write_text(sha+'\n')

class Worker:
    def __init__(self,label,a,out,round):
        source='original' if label=='originalAA' else 'post850' if label=='post850AA' else label
        self.path=out/f'round-{round:02}'/label;self.path.mkdir(parents=True)
        self.err=(self.path/'stderr.log').open('w')
        env=os.environ|{'JULIA_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
        cmd=['taskset','-c',str(a.cpu),str(a.julia),'--startup-file=no','--threads=1',
             '--check-bounds='+('yes' if a.check else 'no'),'-O2','--project='+str(HERE),
             str(HERE/'worker.jl'),str(a.sources/source),label,str(self.path)]
        self.cmd=cmd
        self.p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,text=True,env=env)
        if self.p.stdout.readline().strip()!='READY':raise RuntimeError(f'worker failed startup: {self.path}')
    def run(self,x,ops,seed,check,index):
        target=self.path/f'fixture-{index:03}-seed-{seed}.csv'
        fields=[x['key'],x['family'],x['dtype'],x['n'],x['s'],x['b'],x['layout'],','.join(ops),seed,'check' if check else 'time',target]
        self.p.stdin.write('\t'.join(map(str,fields))+'\n');self.p.stdin.flush()
        response=self.p.stdout.readline().strip()
        if response!='DONE':raise RuntimeError(f'worker {response}: {target}')
        return list(csv.DictReader(target.open()))
    def close(self):
        self.p.stdin.write('QUIT\n');self.p.stdin.flush();self.p.wait();self.err.close()
        if self.p.returncode:raise RuntimeError(f'worker exit: {self.path}')

def main():
    p=argparse.ArgumentParser();p.add_argument('--julia',type=Path,required=True)
    p.add_argument('--sources',type=Path,default=Path('/tmp/pb852-frozen'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--rounds',type=int,default=7)
    p.add_argument('--start-round',type=int,default=1);p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)))
    p.add_argument('--check',action='store_true');p.add_argument('--filter',default='')
    p.add_argument('--cases-file',type=Path);p.add_argument('--roles',default='original,post850,originalAA,historical05,post848,post850AA')
    p.add_argument('--persistent',action='store_true',help='follow-up only; keep worker processes across rounds')
    a=p.parse_args();a.output=a.output.resolve();a.sources=a.sources.resolve()
    a.output.mkdir(parents=True,exist_ok=False);archive_sources(a.sources)
    selected=set(a.cases_file.read_text().splitlines()) if a.cases_file else None
    chosen=[]
    for index,x in enumerate(scenarios()):
        ops=[op for op in x['ops'] if re.search(a.filter,x['key']+'/'+op) and (selected is None or x['key']+'/'+op in selected)]
        if ops:chosen.append((index,x,ops))
    assert chosen,'no cases selected'
    machine={k:subprocess.getoutput(cmd) for k,cmd in {'uname':'uname -a','lscpu':'lscpu','hostname':'hostname','os':'cat /etc/os-release','governor':f'cat /sys/devices/system/cpu/cpu{a.cpu}/cpufreq/scaling_governor','topology':f'cat /sys/devices/system/cpu/cpu{a.cpu}/topology/thread_siblings_list'}.items()}
    machine|={'python':platform.python_version(),'versions':VERSIONS,'argv':vars(a)|{'chosen_case_count':sum(len(ops) for _,_,ops in chosen)},'cpu_affinity':sorted(os.sched_getaffinity(0)),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    (a.output/'campaign.json').write_text(json.dumps(machine,indent=2,default=str)+'\n')
    roles=a.roles.split(',');workers={};failures=[]
    with (a.output/'observations.csv').open('w') as f:
        fields=['round','order','role','seed','case','status','median_ns','min_ns','p90_ns','bytes','allocs','samples','work','memory_bytes','warm_seconds','error']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in range(a.start_round,a.start_round+a.rounds):
            order=roles if r%2 else roles[::-1]
            for role in order:
                if role not in workers:workers[role]=Worker(role,a,a.output,r)
            fixtures=chosen if r%2 else chosen[::-1]
            for index,x,ops in fixtures:
                seed=852000+r*101+index
                for ordinal,role in enumerate(order,1):
                    rows=workers[role].run(x,ops,seed,a.check,index)
                    for row in rows:
                        w.writerow(dict(round=r,order=ordinal,role=role,seed=seed)|row)
                        if row['status']=='failure':failures.append((r,role,row['case'],row['error']))
                    f.flush()
                if index%10==0:print(f'round {r}: fixture {index}, failures {len(failures)}',flush=True)
            if not a.persistent:
                for worker in workers.values():worker.close()
                workers={}
            print(f'Completed round {r}; failures={len(failures)}',flush=True)
        for worker in workers.values():worker.close()
    (a.output/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    if failures:raise SystemExit(f'{len(failures)} correctness failures retained')
if __name__=='__main__':main()
