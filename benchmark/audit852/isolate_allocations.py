#!/usr/bin/env python3
"""One selected allocation operation per fresh process, with A/A controls.

This removes preceding fixture/case history from order-sensitive constructor
observations. Every role uses the same standard BenchmarkTools pretrial GC,
80-sample cap and 1-second budget. No timed callback or package source changes.
"""
import argparse,json,os,re,shutil,subprocess,sys
from pathlib import Path
from run import archive_sources
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--julia',type=Path,required=True);p.add_argument('--sources',type=Path,required=True)
p.add_argument('--rounds',type=int,default=21);p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)))
a=p.parse_args();a.output=a.output.resolve();assert not a.output.exists();a.output.mkdir()
archive_sources(a.sources.resolve());h=a.output/'harness';shutil.copytree(HERE,h,ignore=shutil.ignore_patterns('__pycache__'))
p=h/'worker.jl';s=p.read_text();assert 'samples=80 seconds=.008' in s;p.write_text(s.replace('samples=80 seconds=.008','samples=80 seconds=1.0'))
cases=['sparse/Float32/n10000/s100/b1/scalar/construct_empty',
       'sparse/Float64/n1000000/s1000/b1/scalar/construct_empty',
       'sparse/Float64/n1000000/s1000/b1/scalar/dense_import']
(a.output/'isolation.json').write_text(json.dumps({'cases':cases,'rounds':a.rounds,'protocol':'one operation per fresh source process; source/timed callbacks unchanged; matched 80 samples and 1 second budget; standard BenchmarkTools gctrial=true','roles':['original','post850','originalAA','post850AA']},indent=2)+'\n')
for i,case in enumerate(cases):
    out=a.output/f'case-{i:02}';summary=a.output/f'summary-{i:02}'
    cmd=[sys.executable,str(h/'run.py'),'--julia',str(a.julia.resolve()),'--sources',str(a.sources.resolve()),'--output',str(out),
         '--rounds',str(a.rounds),'--start-round','201','--filter','^'+re.escape(case)+'$','--roles','original,post850,originalAA,post850AA','--cpu',str(a.cpu)]
    with (a.output/f'case-{i:02}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,str(h/'analyze.py'),str(out),str(summary),'--verify'],check=True)
print('Isolated allocation controls complete.')
