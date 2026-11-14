#!/usr/bin/env python3
"""Native-index supplemental controls: seven baseline plus twenty-one confirmation rounds."""
import argparse,csv,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent

def run_controls(prefix,julia,sources,cpu):
    prefix=str(Path(prefix).resolve())
    summary=Path(prefix+'-supplement-summary')
    rows=list(csv.DictReader((summary/'summary.csv').open()))
    selected=set((summary/'suspicious.txt').read_text().splitlines())
    selected|={r['case'] for r in rows if r['before']=='original' and r['after']=='post850'
               and r['case'].rsplit('/',1)[1] in ('residual_cd_fixed','residual_cd_expand','residual_bcd','bilinear_native')}
    selected.discard('')
    selection=Path(prefix+'-supplement-restore-selection.txt');selection.write_text('\n'.join(sorted(selected))+'\n')
    out=Path(prefix+'-supplement-restore');harness=Path(prefix+'-supplement-harness')
    cmd=[sys.executable,str(HERE/'restore_control.py'),'--julia',str(julia),'--sources',str(sources),
         '--harness-source',str(harness),'--cases-file',str(selection),'--output',str(out),'--rounds','28','--cpu',str(cpu),'--native-block-indexing']
    with Path(prefix+'-supplement-restore.log').open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    with Path(prefix+'-supplement-restore-analysis.log').open('w') as log:
        subprocess.run([sys.executable,str(out.with_name(out.name+'-harness')/'analyze.py'),str(out),prefix+'-supplement-restore-summary','--verify'],stdout=log,stderr=subprocess.STDOUT,check=True)
    # Complement the full-field/cold-cache control with identical active-prefix
    # restoration for both scalar sources. This isolates hot working-set kernels.
    hot=Path(prefix+'-active-restore')
    hotselection=Path(prefix+'-restore-selection.txt')
    hotcmd=[sys.executable,str(HERE/'restore_control.py'),'--julia',str(julia),'--sources',str(sources),
            '--cases-file',str(hotselection),'--output',str(hot),'--rounds','7','--cpu',str(cpu),'--restore-mode','active']
    with Path(prefix+'-active-restore.log').open('w') as log:
        subprocess.run(hotcmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    with Path(prefix+'-active-restore-analysis.log').open('w') as log:
        subprocess.run([sys.executable,str(hot.with_name(hot.name+'-harness')/'analyze.py'),str(hot),prefix+'-active-restore-summary','--verify'],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Supplemental flags, full solver controls and hot scalar controls complete.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix',type=Path,required=True);p.add_argument('--julia',type=Path,required=True)
    p.add_argument('--sources',type=Path,required=True);p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)))
    p.add_argument('--wait-for-main',action='store_true');a=p.parse_args()
    if a.wait_for_main:
        while True:
            f=Path(str(a.prefix)+'-stages.json')
            if f.exists():
                stages=json.loads(f.read_text())
                if stages and stages[-1]['status']=='failure':raise SystemExit('main campaign failed; inspect its stage log')
                if any(s['stage']=='restoration control analysis' and s['status']=='pass' for s in stages):break
            time.sleep(5)
    run_controls(a.prefix,a.julia,a.sources,a.cpu)
