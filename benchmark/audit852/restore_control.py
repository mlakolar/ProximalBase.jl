#!/usr/bin/env python3
"""Generate diagnostic common-state restorations outside timing.

State restoration changes outside timing. The optional native-block-indexing
control also changes residual block access to the representation's native
indices, without a timed conversion. Package source is always unchanged;
generated harness files/digests are retained with the campaign.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
from run import archive_sources
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--julia',type=Path,required=True);p.add_argument('--sources',type=Path,required=True)
p.add_argument('--cases-file',type=Path,required=True);p.add_argument('--rounds',type=int,default=21)
p.add_argument('--start-round',type=int,default=101)
p.add_argument('--check',action='store_true',help='bounds-checked references only, without measurements')
p.add_argument('--harness-source',type=Path,default=HERE,help='use a generated supplemental harness when controlling its workloads')
p.add_argument('--restore-mode',choices=('full','active'),default='full')
p.add_argument('--native-block-indexing',action='store_true',help='use native 2-index access for original row-block matrix views')
p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)))
a=p.parse_args();a.output=a.output.resolve();harness=a.output.with_name(a.output.name+'-harness')
archive_sources(a.sources)
assert not harness.exists();shutil.copytree(a.harness_source.resolve(),harness,ignore=shutil.ignore_patterns('__pycache__'))
file=harness/'cases.jl';s=file.read_text();old='setup=()->copyto!(x,src)'
assert s.count(old)==2
# Replace scalar and block native state restoration; timed callbacks stay unchanged.
s=s.replace(old,'setup=()->restore_full_fields!(x,src)')
s='''# Generated cache-conditioning control; no timed callback or package source changed.
function restore_full_fields!(x,y)
    if hasproperty(x,:nzval)
        if AUDIT_RESTORE_MODE=="active"
            for j in 1:x.nnz;x.ind2nzval[x.nzval2ind[j]]=0;end
            for j in 1:y.nnz
                k=y.nzval2ind[j];x.nzval[j]=y.nzval[j];x.nzval2ind[j]=k;x.ind2nzval[k]=j
            end
        else
            copyto!(x.nzval,y.nzval)
            copyto!(x.nzval2ind,y.nzval2ind)
            copyto!(x.ind2nzval,y.ind2nzval)
        end
        x.nnz=y.nnz
    else
        copyto!(x.storage,y.storage)
        if hasproperty(x,:active)
            copyto!(x.active,y.active)
            copyto!(x.position,y.position)
            x.nactive=y.nactive
        end
    end
    x
end
'''+s
s='const AUDIT_RESTORE_MODE = '+repr(a.restore_mode).replace("'",'"')+'\n'+s
file.write_text(s)
if a.native_block_indexing:
    helper=harness/'residual_workloads.jl';helper_text=helper.read_text()
    assert 'old=v[j];newvalues[j]' in helper_text and 'v[j]=newvalues[j]' in helper_text
    helper_text=helper_text.replace('old=v[j];newvalues[j]','old=native_block_value(v,j);newvalues[j]').replace('v[j]=newvalues[j]','native_block_write!(v,newvalues[j],j)')
    helper.write_text('''# Native representation access, with no timed shape conversion.
@inline native_block_value(v::AbstractVector,j)=v[j]
@inline native_block_value(v::AbstractMatrix,j)=v[1,j]
@inline native_block_write!(v::AbstractVector,z,j)=(v[j]=z)
@inline native_block_write!(v::AbstractMatrix,z,j)=(v[1,j]=z)
'''+helper_text)
    cases_text=file.read_text();needle_shape='    src=make()\n    for i in findall(!iszero,base);src[i]=base[i];end'
    assert needle_shape in cases_text
    cases_text=cases_text.replace(needle_shape,'''    src=make()
    if ORIGINAL && layout=="strided";@assert size(block(src,1))==(1,b);end
    for i in findall(!iszero,base);src[i]=base[i];end''')
    file.write_text(cases_text)
worker=harness/'worker.jl';worker_text=worker.read_text();assert 'samples=80 seconds=.008' in worker_text
worker.write_text(worker_text.replace('samples=80 seconds=.008','samples=80 seconds=1.0'))
(harness/'control.json').write_text(json.dumps({'original_cases_sha256':hashlib.sha256((a.harness_source/'cases.jl').read_bytes()).hexdigest(),
 'control_cases_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'restore_mode':a.restore_mode,'native_block_indexing':a.native_block_indexing,'change':'common untimed state restoration and 1-second sampling budget; optional native matrix indexing in residual block access; no timed shape conversion' },indent=2)+'\n')
cmd=[sys.executable,str(harness/'run.py'),'--julia',str(a.julia),'--sources',str(a.sources),'--output',str(a.output),
 '--cases-file',str(a.cases_file.resolve()),'--rounds',str(a.rounds),'--start-round',str(a.start_round),'--cpu',str(a.cpu),
 '--roles','original,post850,originalAA,post848,post850AA']
if a.check:cmd.append('--check')
result=subprocess.run(cmd)
if a.output.exists():shutil.copyfile(a.cases_file,a.output/'selection.txt')
raise SystemExit(result.returncode)
