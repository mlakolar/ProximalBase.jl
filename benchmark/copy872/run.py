#!/usr/bin/env python3
"""#872 separate-process paired protocol: #861's driver with any number of source roles."""
import argparse,csv,hashlib,json,os,platform,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
FRACTIONS=(1,10,50,100)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def fixtures(selected,grid):
    out=[]
    for dt in ('Float32','Float64'):
        for n in (8192,65536):
            for b in (1,8,64):
                for layout in ('contiguous','strided','arbitrary'):
                    K=n//b
                    for frac in FRACTIONS:
                        s=max(1,round(K*frac/100))
                        key=f'block/{dt}/n{n}/b{b}/{layout}/f{frac}/s{s}'
                        names=['shared_empty','shared_active','shared_destonly','equal_empty','equal_active','cached_active',
                               'mixed_shared','restart_shared','restart_equal']
                        if s==K:names.remove('shared_destonly')   # no group outside src
                        if grid=='small' and n==65536:
                            names=[x for x in names if x in ('shared_active','equal_active','restart_shared')]
                        ops=[op for op in names if selected is None or key+'/'+op in selected]
                        if ops:out.append((key,dt,n,s,b,layout,ops))
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--julia',type=Path,required=True)
    p.add_argument('--source',action='append',required=True,help='role=path; baseline and baselineAA must name equal trees')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=7)
    p.add_argument('--start-round',type=int,default=1)
    p.add_argument('--check',action='store_true')
    p.add_argument('--grid',default='full',choices=('full','small'))
    p.add_argument('--cases-file',type=Path)
    p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sources=dict(x.split('=',1) for x in a.source)
    roles=list(sources)
    selected=set(a.cases_file.read_text().split()) if a.cases_file else None
    fx=fixtures(selected,a.grid)
    assert fx
    paths=sorted(HERE.glob('*.jl'))+sorted(HERE.glob('*.py'))+sorted(HERE.glob('candidates/*.jl'))
    paths+=[ROOT/'benchmark'/'copy861'/'cases.jl']
    paths+=list((ROOT/'benchmark'/'audit852').glob('*.toml'))
    paths+=[ROOT/'benchmark'/'audit852'/f for f in ('worker.jl','cases.jl','adapters.jl')]
    provenance={'argv':{k:str(v) for k,v in vars(a).items()},'roles':roles,'sources':sources,
                'platform':platform.platform(),'started':time.time(),
                'harness_sha256':{str(f.relative_to(ROOT)):digest(f) for f in paths},
                'lscpu':subprocess.check_output(['lscpu'],text=True),'rounds':a.rounds,
                'fixtures':len(fx),'cases':sum(len(f[-1]) for f in fx),
                'case_keys':[f[0]+'/'+op for f in fx for op in f[-1]]}
    (a.output/'campaign.json').write_text(json.dumps(provenance,indent=2)+'\n')
    failures=[]
    with (a.output/'observations.csv').open('w') as output:
        fields=['round','order','role','seed','case','status','median_ns','min_ns','p90_ns',
                'bytes','allocs','samples','work','memory_bytes','warm_seconds','error']
        writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader()
        for r in range(a.start_round,a.start_round+a.rounds):
            order=roles[r%len(roles):]+roles[:r%len(roles)]
            for ordinal,role in enumerate(order,1):
                source=Path(sources[role])
                procdir=a.output/f'round-{r:02}'/role;procdir.mkdir(parents=True)
                cmd=['taskset','-c',str(a.cpu),str(a.julia),'--startup-file=no','--threads=1','-O2',
                     '--check-bounds='+('yes' if a.check else 'no'),'--project='+str(ROOT/'benchmark'/'audit852'),
                     str(HERE/'worker.jl'),str(source.resolve()),role,str(procdir.resolve())]
                env=os.environ|{'JULIA_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
                t0=time.time()
                with (procdir/'stderr.log').open('w') as err:
                    worker=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,env=env)
                    assert worker.stdout.readline().strip()=='READY',procdir
                    fs=fx if r%2 else fx[::-1]
                    for index,(key,dt,n,s,b,layout,ops) in enumerate(fs):
                        seed=872000+r*101+sum(map(ord,key))
                        target=procdir/f'fixture-{index:03}.csv'
                        worker.stdin.write('\t'.join(map(str,[key,'block',dt,n,s,b,layout,','.join(ops),seed,
                                                             'check' if a.check else 'time',target.resolve()]))+'\n')
                        worker.stdin.flush()
                        assert worker.stdout.readline().strip()=='DONE',target
                        for row in csv.DictReader(target.open()):
                            writer.writerow(dict(round=r,order=ordinal,role=role,seed=seed)|row)
                            if row['status']!='pass':failures.append((r,role,row))
                        output.flush()
                    worker.stdin.write('QUIT\n');worker.stdin.flush();assert worker.wait()==0
                print(f'round {r} {role}: {len(fx)} fixtures; {time.time()-t0:.0f}s; failures={len(failures)}',flush=True)
    (a.output/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    assert not failures,failures[:3]
if __name__=='__main__':main()
