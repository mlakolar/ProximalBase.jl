#!/usr/bin/env python3
"""#861 separate-process paired protocol, reusing the #852 worker."""
import argparse,csv,hashlib,json,os,platform,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--julia',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--candidate',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=7)
    p.add_argument('--start-round',type=int,default=1)
    p.add_argument('--check',action='store_true')
    p.add_argument('--exact852',action='store_true',help='unchanged #852 focal copyto/copy fixtures')
    p.add_argument('--roles',default='original,baseline,candidate,baselineAA')
    p.add_argument('--cases-file',type=Path)
    p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    roles=a.roles.split(',')
    selected=set(a.cases_file.read_text().splitlines()) if a.cases_file else None
    fixtures=[]
    for dt in ('Float32','Float64'):
        for n,b in ((8192,1),(8192,8),(8192,64),(65536,8),(65536,64)):
            for layout in ('contiguous','strided','arbitrary'):
                s=(n//b)//10
                key=f'block/{dt}/n{n}/s{s}/b{b}/{layout}'
                names=['shared_empty','shared_active','shared_zeros','equal_active','equal_empty',
                       'cached_active','different','copy','self','copybang_shared',
                       'mixed_shared','mixed_equal','restart_shared','restart_equal']
                if n==65536:
                    names=['shared_active','equal_active','copy','restart_shared']
                ops=[op for op in names if selected is None or key+'/'+op in selected]
                if ops:fixtures.append((key,dt,n,s,b,layout,ops))
    if a.exact852:
        fixtures=[]
        for dt,layout in (('Float32','strided'),('Float64','contiguous')):
            for s in (81,819):
                key=f'block/{dt}/n8192/s{s}/b1/{layout}'
                ops=[op for op in ('copyto','copy') if selected is None or key+'/'+op in selected]
                if ops:fixtures.append((key,dt,8192,s,1,layout,ops))
    assert fixtures
    paths=list(HERE.glob('*.jl'))+list(HERE.glob('*.py'))
    paths+=list((HERE.parent/'audit852').glob('*.toml'))
    paths+=[HERE.parent/'audit852'/f for f in ('worker.jl','cases.jl','adapters.jl')]
    provenance={'argv':{k:str(v) for k,v in vars(a).items()},'platform':platform.platform(),
                'started':time.time(),'harness_sha256':{str(f.relative_to(ROOT)):digest(f) for f in paths},
                'lscpu':subprocess.check_output(['lscpu'],text=True),'rounds':a.rounds}
    (a.output/'campaign.json').write_text(json.dumps(provenance,indent=2)+'\n')
    failures=[]
    with (a.output/'observations.csv').open('w') as output:
        fields=['round','order','role','seed','case','status','median_ns','min_ns','p90_ns',
                'bytes','allocs','samples','work','memory_bytes','warm_seconds','error']
        writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader()
        for r in range(a.start_round,a.start_round+a.rounds):
            order=roles[r%len(roles):]+roles[:r%len(roles)]
            for ordinal,role in enumerate(order,1):
                source=a.baseline if role in ('baseline','baselineAA') else a.original if role=='original' else a.candidate
                procdir=a.output/f'round-{r:02}'/role;procdir.mkdir(parents=True)
                cmd=['taskset','-c',str(a.cpu),str(a.julia),'--startup-file=no','--threads=1','-O2',
                     '--check-bounds='+('yes' if a.check else 'no'),'--project='+str(HERE.parent/'audit852'),
                     str(HERE/'worker.jl'),str(source.resolve()),role,str(procdir.resolve())]
                env=os.environ|{'JULIA_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1',
                                'PB_EXACT852':'1' if a.exact852 else '0'}
                with (procdir/'stderr.log').open('w') as err:
                    worker=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,env=env)
                    assert worker.stdout.readline().strip()=='READY',procdir
                    fs=fixtures if r%2 else fixtures[::-1]
                    for index,(key,dt,n,s,b,layout,ops) in enumerate(fs):
                        seed=861000+r*101+sum(map(ord,key))
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
                print(f'round {r} {role}: {len(fixtures)} fixtures; failures={len(failures)}',flush=True)
    (a.output/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    assert not failures,failures[:3]
if __name__=='__main__':main()
