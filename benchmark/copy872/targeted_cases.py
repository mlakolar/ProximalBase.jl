#!/usr/bin/env python3
"""Print the targeted case list: every 1% case, the 100% cases with group size 1 or 64,
and the 10% restart_shared cases. usage: python3 targeted_cases.py > cases-targeted.txt"""
import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from run import fixtures
for key,dt,n,s,b,layout,ops in fixtures(None,'full'):
    frac=int(key.split('/f')[1].split('/')[0])
    for op in ops:
        if frac==1 or (frac==100 and b in (1,64)) or (frac==10 and op=='restart_shared'):
            print(key+'/'+op)
