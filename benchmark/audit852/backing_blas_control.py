#!/usr/bin/env python3
"""Compare original backing-array BLAS with current sparse-aware owner products.

Both strategies already exist; no kernel shim or timed shape conversion is
added. Current storage can also be passed to BLAS. Finite inputs only, exactly
as in the shared fixtures. Keep this strategy comparison separate from default
owner-API timings. Serialize after the static-adapter campaign.
"""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
from spec import scenarios,VERSIONS
from control_manifest import record_control
HERE=Path(__file__).resolve().parent;REPO=HERE.parent.parent
p=argparse.ArgumentParser();p.add_argument('--prefix',type=Path,required=True)
p.add_argument('--julia',type=Path,required=True);p.add_argument('--sources',type=Path,required=True)
p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)));p.add_argument('--wait',action='store_true')
a=p.parse_args();prefix=str(a.prefix.resolve());marker=Path(prefix+'-adapter-control-complete.json')
if a.wait:
    while not marker.exists():time.sleep(5)
assert json.loads(marker.read_text())['status']=='pass'
harness=Path(prefix+'-backing-blas-harness');assert not harness.exists()
shutil.copytree(Path(prefix+'-raw-block-harness'),harness,ignore=shutil.ignore_patterns('__pycache__'))
file=harness/'cases.jl';text=file.read_text();head,tail=text.split('function block_cases',1)
needle='        elseif name=="dot"';assert tail.count(needle)==1
extra='''        elseif name=="dot_backing_native"
            v=rand(rng,T,n);expected=dot(v,base)
            f=let v=v
                ORIGINAL ? (()->dot(v,x.storage)) : (()->dot(v,x))
            end
            check=r->(@assert isapprox(r,expected;rtol=8e-4,atol=1e-5);verify_block(x,base,active,groups))
        elseif name=="matvec_backing_native"
            A=rand(rng,T,16,n);out=zeros(T,16);expected=A*base;work=16s*b
            f=let A=A,out=out
                ORIGINAL ? (()->mul!(out,A,x.storage)) : (()->mul!(out,A,x))
            end
            check=r->(reference_equal(out,expected);verify_block(x,base,active,groups))
'''
tail=tail.replace(needle,extra+needle);file.write_text(head+'function block_cases'+tail)
chosen=[]
for x in scenarios():
    if x['family']=='block':
        x=dict(x);x['ops']=['dot_backing_native','matvec_backing_native'];chosen.append(x)
(harness/'spec.py').write_text('VERSIONS='+repr(VERSIONS)+'\nSCENARIOS='+repr(chosen)+'\ndef scenarios():return SCENARIOS\n')
(harness/'backing-blas-control.json').write_text(json.dumps({'before':'original dot/mul! on its existing dense backing Vector',
 'after':'post850 dot/mul! on BlockIterate owner with active groups','timed_conversion':False,
 'domain':'finite deterministic operands','current_alternative':'current storage Vector also supports BLAS',
 'interpretation':'available strategy comparison, separate from default owner API'},indent=2)+'\n')
record_control(harness,'original backing-array BLAS versus current sparse-aware owner products')
base=[sys.executable,harness/'run.py','--julia',a.julia,'--sources',a.sources,'--cpu',a.cpu,
      '--roles','original,post850,originalAA,post850AA']
steps=[]
def run(name,cmd):
    print(name,flush=True);log=Path(prefix+'-backing-blas-'+name+'.log');start=time.time()
    with log.open('w') as f:subprocess.run(list(map(str,cmd)),cwd=REPO,stdout=f,stderr=subprocess.STDOUT,check=True)
    steps.append(dict(stage=name,seconds=time.time()-start,log=log.name))
run('check',base+['--output',prefix+'-backing-blas-check','--check','--rounds','1','--start-round','701'])
run('broad7',base+['--output',prefix+'-backing-blas','--rounds','7','--start-round','601'])
run('analysis7',[sys.executable,harness/'analyze.py',prefix+'-backing-blas',prefix+'-backing-blas-summary','--verify'])
flags=Path(prefix+'-backing-blas-summary/suspicious.txt')
if flags.read_text().strip():
    output=Path(prefix+'-backing-blas-confirmation')
    run('confirmation21',base+['--output',output,'--rounds','21','--start-round','608','--cases-file',flags])
    shutil.copyfile(flags,output/'selection.txt')
    run('analysis21',[sys.executable,harness/'analyze.py',output,str(output)+'-summary','--verify'])
Path(prefix+'-backing-blas-complete.json').write_text(json.dumps({'status':'pass','steps':steps},indent=2)+'\n')
print('Backing-array BLAS strategy control complete.',flush=True)
