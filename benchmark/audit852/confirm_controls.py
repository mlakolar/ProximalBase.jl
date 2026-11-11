#!/usr/bin/env python3
"""Confirm newly exposed scalar control flags in fresh, nonoverlapping rounds.

Run after supplement_controls.py, or use --wait to wait for its verified hot
control. Each context keeps its own selection and independent confirmation
summary; exploratory and confirmation rounds are never silently pooled.
"""
import argparse,csv,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',type=Path,required=True)
    p.add_argument('--julia',type=Path,required=True);p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)));p.add_argument('--wait',action='store_true')
    a=p.parse_args();prefix=str(a.prefix.resolve())
    ready=Path(prefix+'-active-restore-summary/analysis.json')
    if a.wait:
        while not ready.exists():time.sleep(5)
    assert json.loads(ready.read_text())['raw_samples_verified']
    for context,mode in [('restore','full'),('active-restore','active')]:
        source=Path(prefix+'-'+context+'-summary')
        assert json.loads((source/'analysis.json').read_text())['raw_samples_verified']
        selected=sorted(set((source/'suspicious.txt').read_text().splitlines())-{''})
        selection=Path(prefix+'-'+context+'-confirmation-selection.txt')
        selection.write_text('\n'.join(selected)+'\n')
        if not selected:continue
        out=Path(prefix+'-'+context+'-confirmation')
        cmd=[sys.executable,str(HERE/'restore_control.py'),'--julia',str(a.julia),'--sources',str(a.sources),
             '--cases-file',str(selection),'--output',str(out),'--rounds','21','--start-round','201',
             '--cpu',str(a.cpu),'--restore-mode',mode]
        print(context,len(selected),'cases; 21 independent confirmation rounds',flush=True)
        with Path(str(out)+'.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        with Path(str(out)+'-analysis.log').open('w') as log:
            subprocess.run([sys.executable,str(out.with_name(out.name+'-harness')/'analyze.py'),str(out),str(out)+'-summary','--verify'],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Scalar control confirmation complete.',flush=True)
if __name__=='__main__':main()
