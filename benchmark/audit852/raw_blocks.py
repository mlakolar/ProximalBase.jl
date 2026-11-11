#!/usr/bin/env python3
"""Fill original Float32/arbitrary block coverage using existing raw native views.

All view/owner construction is outside timing. No historical kernel or package
method is supplied. Full construction is intentionally absent from this variant.
Run after scalar confirmations; --wait enforces that serialization.
"""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
from spec import scenarios,VERSIONS
from run import archive_sources
HERE=Path(__file__).resolve().parent;REPO=HERE.parent.parent
p=argparse.ArgumentParser();p.add_argument('--prefix',type=Path,required=True)
p.add_argument('--julia',type=Path,required=True);p.add_argument('--sources',type=Path,required=True)
p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)));p.add_argument('--wait',action='store_true')
p.add_argument('--generate-only',action='store_true',help='create the raw-native harness for CI/reference checks, without measurements')
p.add_argument('--harness-source',type=Path,help='existing generated common-full/native-index supplemental harness')
a=p.parse_args();prefix=str(a.prefix.resolve())
def ready():
    if not Path(prefix+'-active-restore-summary/analysis.json').exists():return False
    for name in ('restore','active-restore'):
        flags=Path(prefix+'-'+name+'-summary/suspicious.txt')
        if not flags.exists():return False
        if flags.read_text().strip():
            f=Path(prefix+'-'+name+'-confirmation-summary/analysis.json')
            if not f.exists() or not json.loads(f.read_text())['raw_samples_verified']:return False
    return True
if a.wait and not a.generate_only:
    while not ready():time.sleep(5)
if not a.generate_only:assert ready(),'finish prior timing/confirmations first'
archive_sources(a.sources)
harness=Path(prefix+'-raw-block-harness');assert not harness.exists()
shutil.copytree(a.harness_source or Path(prefix+'-supplement-restore-harness'),harness,ignore=shutil.ignore_patterns('__pycache__'))
adapters=harness/'adapters.jl';s=adapters.read_text();line=next(l for l in s.splitlines() if l.startswith('newblock(T,n,b,groups)='))
s=s.replace(line,'''function newblock(::Type{T},n,b,groups) where T
    if ORIGINAL
        storage=zeros(T,n)
        atoms=[view(storage,G) for G in groups]
        PB.AtomIterate(storage,atoms)
    else
        BT(T,n,groups)
    end
end''');adapters.write_text(s)
worker=harness/'worker.jl';s=worker.read_text();line=next(l for l in s.splitlines() if l.strip().startswith('incompatible=ORIGINAL'))
s=s.replace(line,'    incompatible=false # existing raw original constructor represents these layouts/precisions')
worker.write_text(s)
file=harness/'cases.jl';s=file.read_text();needle='if ORIGINAL && layout=="strided";@assert size(block(src,1))==(1,b);end'
assert needle in s;s=s.replace(needle,'if ORIGINAL;@assert size(block(src,1))==(b,);end');file.write_text(s)
chosen=[]
for x in scenarios():
    if x['family']!='block':continue
    x=dict(x);x['ops']=[op for op in x['ops'] if not op.startswith('construct')]+['copyto_shared','residual_bcd']
    assert len(set(x['ops']))==len(x['ops']);chosen.append(x)
(harness/'spec.py').write_text('VERSIONS='+repr(VERSIONS)+'\nSCENARIOS='+repr(chosen)+'\ndef scenarios():return SCENARIOS\n')
(harness/'raw-block-control.json').write_text(json.dumps({'construction':'excluded; raw original storage/views/owner prepared outside timing',
 'original_api':'existing AtomIterate(storage, atoms) with native vector SubArray views',
 'source_patches':False,'restore':'common full fields outside timing','trial':'80 samples / 1 second',
 'roles':'original,post850,originalAA,post850AA','fixtures':len(chosen),'case_ids':sum(len(x['ops']) for x in chosen)},indent=2)+'\n')
if a.generate_only:
    print('Generated raw-native harness:',harness,flush=True);raise SystemExit(0)
base=[sys.executable,str(harness/'run.py'),'--julia',str(a.julia),'--sources',str(a.sources),'--cpu',str(a.cpu)]
steps=[]
def run(name,cmd):
    print(name,flush=True);log=Path(prefix+'-raw-block-'+name+'.log');start=time.time()
    with log.open('w') as f:subprocess.run(list(map(str,cmd)),cwd=REPO,stdout=f,stderr=subprocess.STDOUT,check=True)
    steps.append({'stage':name,'seconds':time.time()-start,'log':log.name})
run('check',base+['--output',prefix+'-raw-block-check','--check','--rounds','1','--start-round','301','--roles','original,post850'])
for source in ('original','post850'):
    run('layout-'+source,['taskset','-c',a.cpu,a.julia,'--startup-file=no','--threads=1','--check-bounds=yes','--project='+str(HERE),HERE/'layout_correctness.jl',a.sources/source,harness])
roles=['--roles','original,post850,originalAA,post850AA']
run('broad7',base+['--output',prefix+'-raw-block','--rounds','7']+roles)
run('analysis7',[sys.executable,harness/'analyze.py',prefix+'-raw-block',prefix+'-raw-block-summary','--verify'])
flags=Path(prefix+'-raw-block-summary/suspicious.txt')
if flags.read_text().strip():
    output=Path(prefix+'-raw-block-confirmation')
    run('confirmation21',base+['--output',output,'--rounds','21','--start-round','8','--cases-file',flags]+roles)
    shutil.copyfile(flags,output/'selection.txt')
    run('analysis21',[sys.executable,harness/'analyze.py',output,str(output)+'-summary','--verify'])
Path(prefix+'-raw-block-complete.json').write_text(json.dumps({'status':'pass','steps':steps},indent=2)+'\n')
print('Raw-constructor block coverage and confirmation complete.',flush=True)
