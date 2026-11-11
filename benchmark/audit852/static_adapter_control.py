#!/usr/bin/env python3
"""Resolve historical API-selection inference before source attribution.

Waits for all primary/raw block campaigns, then diagnoses allocation/inference
for all sources. If runtime API selection changes historical05 allocations,
measure a separately preserved static-adapter block campaign. No package method
or production file is changed, and original measurements are never overwritten.
"""
import argparse,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
from spec import scenarios,VERSIONS
from control_manifest import record_control
HERE=Path(__file__).resolve().parent;REPO=HERE.parent.parent
p=argparse.ArgumentParser();p.add_argument('--prefix',type=Path,required=True)
p.add_argument('--julia',type=Path,required=True);p.add_argument('--sources',type=Path,required=True)
p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)));p.add_argument('--wait',action='store_true')
a=p.parse_args();prefix=str(a.prefix.resolve());marker=Path(prefix+'-raw-block-complete.json')
if a.wait:
    while not marker.exists():time.sleep(5)
assert json.loads(marker.read_text())['status']=='pass'
records={};steps=[]
def run(name,cmd):
    print(name,flush=True);log=Path(prefix+'-adapter-'+name+'.log');start=time.time()
    with log.open('w') as f:subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,cwd=REPO,check=True)
    steps.append(dict(stage=name,seconds=time.time()-start,log=log.name));return log
for source in ('original','historical05','post848','post850'):
    log=run('diagnostic-'+source,['taskset','-c',a.cpu,a.julia,'--startup-file=no','--threads=1','-O2','--check-bounds=no',
        '--project='+str(HERE),HERE/'adapter_diagnostic.jl',a.sources/source])
    match=re.search(r'ADAPTER_ALLOCATION_BYTES runtime=(\d+) static=(\d+)',log.read_text());assert match,log
    records[source]=dict(runtime_bytes=int(match[1]),static_bytes=int(match[2]))
artifact=records['historical05']['runtime_bytes']>records['historical05']['static_bytes']
assert all(records[source]['runtime_bytes']==records[source]['static_bytes'] for source in ('original','post848','post850')), 'API adapter affected a primary/extraction source; investigate before export'
if artifact:
    harness=Path(prefix+'-static-adapter-harness');assert not harness.exists()
    shutil.copytree(Path(prefix+'-supplement-restore-harness'),harness,ignore=shutil.ignore_patterns('__pycache__'))
    file=harness/'adapters.jl';s=file.read_text();needle='    block(x,k)=isdefined(PB,:blockview) ? PB.blockview(x,k) : PB.atom(x,k)'
    assert needle in s;s=s.replace(needle,'''    @static if isdefined(PB,:blockview)
        block(x,k)=PB.blockview(x,k)
    else
        block(x,k)=PB.atom(x,k)
    end''');file.write_text(s)
    # Historical05 native fill removes activity, unlike current retained-zero
    # fill. Preserve the destination working set before both copy operations.
    file=harness/'cases.jl';s=file.read_text()
    assert 'prep=()->fill!(dest,zero(T));work=n' in s
    assert 'prep=()->fill!(shared,zero(T));work=n' in s
    s=s.replace('prep=()->fill!(dest,zero(T));work=n','prep=()->retained_zero_blocks!(dest);work=n')
    s=s.replace('prep=()->fill!(shared,zero(T));work=n','prep=()->retained_zero_blocks!(shared);work=n')
    s='''# Benchmark-only state preparation; no package methods added.
function retained_zero_blocks!(x)
    before=PB.nactive(x)
    for k in PB.activecoordinates(x)
        fill!(block(x,k),zero(eltype(x)))
    end
    @assert PB.nactive(x)==before
    x
end
'''+s
    file.write_text(s)
    chosen=[]
    for x in scenarios():
        if x['family']=='block':
            x=dict(x);x['ops']=list(x['ops'])+['copyto_shared','residual_bcd'];chosen.append(x)
    (harness/'spec.py').write_text('VERSIONS='+repr(VERSIONS)+'\nSCENARIOS='+repr(chosen)+'\ndef scenarios():return SCENARIOS\n')
    (harness/'static-adapter-control.json').write_text(json.dumps({'change':'API choice at adapter definition time; no inline hint or package method change',
      'original_observations':'retained; legacy historical05 block contrasts superseded for source attribution',
      'copy_preparation':'explicit zero writes retain the destination active groups in every source; native historical05 fill removes them',
      'restore':'common full state, 80 samples/1 second'},indent=2)+'\n')
    record_control(harness,'static API selection and equal retained-zero block-copy preparation')
    base=[sys.executable,harness/'run.py','--julia',a.julia,'--sources',a.sources,'--cpu',a.cpu,
          '--roles','historical05,post848,post850,post850AA']
    run('check',base+['--output',prefix+'-static-adapter-check','--check','--rounds','1','--start-round','501'])
    run('broad7',base+['--output',prefix+'-static-adapter','--rounds','7','--start-round','401'])
    run('analysis7',[sys.executable,harness/'analyze.py',prefix+'-static-adapter',prefix+'-static-adapter-summary','--verify'])
    flags=Path(prefix+'-static-adapter-summary/suspicious.txt')
    if flags.read_text().strip():
        output=Path(prefix+'-static-adapter-confirmation')
        run('confirmation21',base+['--output',output,'--rounds','21','--start-round','408','--cases-file',flags])
        shutil.copyfile(flags,output/'selection.txt')
        run('analysis21',[sys.executable,harness/'analyze.py',output,str(output)+'-summary','--verify'])
Path(prefix+'-adapter-control-complete.json').write_text(json.dumps({'status':'pass','historical05_runtime_adapter_artifact':artifact,
 'allocations':records,'steps':steps},indent=2)+'\n')
print('Adapter attribution resolved.',flush=True)
