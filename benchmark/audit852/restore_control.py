#!/usr/bin/env python3
"""Generate a diagnostic harness with identical full-field restoration outside timing.

Only scalar fixture setup changes. Package source and timed callbacks are copied
unchanged; generated harness files/digests are retained with the campaign.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
from run import archive_sources
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--julia',type=Path,required=True);p.add_argument('--sources',type=Path,required=True)
p.add_argument('--cases-file',type=Path,required=True);p.add_argument('--rounds',type=int,default=21)
p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)))
a=p.parse_args();a.output=a.output.resolve();harness=a.output.with_name(a.output.name+'-harness')
archive_sources(a.sources)
assert not harness.exists();shutil.copytree(HERE,harness,ignore=shutil.ignore_patterns('__pycache__'))
file=harness/'cases.jl';s=file.read_text();old='setup=()->copyto!(x,src)'
assert s.count(old)==2
# Change only scalar setup; the second occurrence belongs to blocks.
s=s.replace(old,'setup=()->restore_full_fields!(x,src)',1)
s='''# Generated cache-conditioning control; no timed callback or package source changed.
function restore_full_fields!(x,y)
    copyto!(x.nzval,y.nzval)
    copyto!(x.nzval2ind,y.nzval2ind)
    copyto!(x.ind2nzval,y.ind2nzval)
    x.nnz=y.nnz
    x
end
'''+s
file.write_text(s)
worker=harness/'worker.jl';worker_text=worker.read_text();assert 'samples=80 seconds=.008' in worker_text
worker.write_text(worker_text.replace('samples=80 seconds=.008','samples=80 seconds=1.0'))
(harness/'control.json').write_text(json.dumps({'original_cases_sha256':hashlib.sha256((HERE/'cases.jl').read_bytes()).hexdigest(),
 'control_cases_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'change':'scalar x restore copies all three full-capacity arrays in the same order for every source; timed callbacks unchanged; 1 second budget permits equal 80-sample counts despite O(q) restore' },indent=2)+'\n')
cmd=[sys.executable,str(harness/'run.py'),'--julia',str(a.julia),'--sources',str(a.sources),'--output',str(a.output),
 '--cases-file',str(a.cases_file.resolve()),'--rounds',str(a.rounds),'--start-round','101','--cpu',str(a.cpu),
 '--roles','original,post850,originalAA,post848,post850AA']
result=subprocess.run(cmd)
if a.output.exists():shutil.copyfile(a.cases_file,a.output/'selection.txt')
raise SystemExit(result.returncode)
