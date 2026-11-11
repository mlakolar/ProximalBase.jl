#!/usr/bin/env python3
"""Recompute every campaign from raw samples and compare exported metric cells.

Run after all timing. MAIN denotes this committed harness; generated harnesses
are resolved relative to --raw-parent, alongside extracted raw archives.
Source-attribution annotations are reproduced by export_results.py, separately
from the numeric analysis. All numeric/boolean/string metric cells must match.
"""
import argparse,csv,hashlib,json,subprocess,sys,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
p.add_argument('--raw-parent',type=Path,required=True);a=p.parse_args();records=[]
def canonical(rows):
    return [{k:v for k,v in r.items() if k!='source_attribution'} for r in rows]
for entry in csv.DictReader((a.results/'campaigns.csv').open()):
    raw=a.raw_parent/entry['raw_relative_directory']
    harness=Path(__file__).resolve().parent if entry['analysis_harness']=='MAIN' else a.raw_parent/entry['analysis_harness']
    with tempfile.TemporaryDirectory(prefix='pb852-replay-') as temp:
        output=Path(temp)/'analysis'
        result=subprocess.run([sys.executable,str(harness/'analyze.py'),str(raw),str(output),'--verify'],capture_output=True,text=True)
        assert result.returncode==0,(entry['campaign'],result.stdout,result.stderr)
        expected=list(csv.DictReader((a.results/entry['campaign']/'summary.csv').open()))
        reproduced=list(csv.DictReader((output/'summary.csv').open()))
        assert canonical(expected)==reproduced,(entry['campaign'],'summary metric cell mismatch')
        original_compat=list(csv.DictReader((output/'incompatibilities.csv').open()))
        exported_compat=list(csv.DictReader((a.results/entry['campaign']/'incompatibilities.csv').open()))
        assert [{k:v for k,v in r.items() if k!='audit_scope'} for r in exported_compat]==original_compat
        digest=hashlib.sha256(json.dumps(reproduced,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        records.append(dict(campaign=entry['campaign'],status='pass',comparison_rows=len(reproduced),metric_sha256=digest))
        print(entry['campaign'],'raw samples → all exported metric cells match',flush=True)
(a.results/'replay.json').write_text(json.dumps({'status':'pass','python':sys.version,'campaigns':records},indent=2)+'\n')
