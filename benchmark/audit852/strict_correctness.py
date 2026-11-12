#!/usr/bin/env python3
"""Repeat all references with tighter Float64 scalar-product tolerances.

Timed code is frozen. Only untimed validation is changed in a generated copy;
Float32 tolerances and the already-strict full-state references stay unchanged.
"""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
from run import archive_sources
from control_manifest import record_control
p=argparse.ArgumentParser();p.add_argument('--julia',type=Path,required=True)
p.add_argument('--sources',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--harness-source',type=Path,default=Path(__file__).resolve().parent)
a=p.parse_args();archive_sources(a.sources);harness=a.output.with_name(a.output.name+'-harness')
assert not harness.exists();shutil.copytree(a.harness_source,harness,ignore=shutil.ignore_patterns('__pycache__'))
file=harness/'cases.jl';original=file.read_text();text=original
for tolerance in ('2e-4','8e-4'):
    text=text.replace('rtol='+tolerance+',atol=1e-5',
      'rtol=T==Float32 ? '+tolerance+' : 2e-10,atol=T==Float32 ? 1e-5 : 1e-10')
assert text!=original;file.write_text(text)
(harness/'correctness-control.json').write_text(json.dumps({
 'change':'Float64 scalar result reference checks use rtol=2e-10, atol=1e-10; untimed validation only',
 'input_cases_sha256':hashlib.sha256(original.encode()).hexdigest(),
 'output_cases_sha256':hashlib.sha256(text.encode()).hexdigest()},indent=2)+'\n')
record_control(harness,'untimed strict Float64 references; no timing campaign')
subprocess.run([sys.executable,str(harness/'run.py'),'--julia',str(a.julia),'--sources',str(a.sources),
 '--output',str(a.output),'--check','--rounds','1','--start-round','301',
 '--roles','original,historical05,post848,post850'],check=True)
