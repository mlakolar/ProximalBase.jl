#!/usr/bin/env python3
"""Print the targeted case list: every 1% and every 100% case, and the 10% restart_shared
cases. usage: python3 targeted_cases.py > cases-targeted.txt"""
import sys,pathlib
sys.dont_write_bytecode=True;sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from run import fixtures
for key,dt,n,s,b,layout,ops in fixtures(None,'full'):
    frac=int(key.split('/f')[1].split('/')[0])
    for op in ops:
        if frac in (1,100) or (frac==10 and op=='restart_shared'):
            print(key+'/'+op)
