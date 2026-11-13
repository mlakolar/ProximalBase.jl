import sys,collections
rows=[l.rstrip('\n').split(',',8) for l in open(sys.argv[1]) if l.startswith('solve,')]
d=collections.defaultdict(dict)
for r in rows: d[tuple(r[2:6])].setdefault(r[6],{})[r[7]]=r[8]
eps={'Float64':2.220446049250313e-16,'Float32':float.fromhex('0x1p-23')}
fam=collections.defaultdict(lambda: collections.defaultdict(float))
for k,v in d.items():
    for nm,m in v.items():
        if 'scaled_reported_objective_error' in m:
            e=float(m['scaled_reported_objective_error'])/eps[k[0]]
            key=(k[0],k[1],k[3]); fam[key][nm]=max(fam[key][nm],e)
V=['K','B1','B2','B3','H','H1','C']
print('max over λ of |reported - F(x)| / S, in units of eps(T); S = α(‖|X||x|‖+‖y‖)²/2 + λ‖x‖₁')
print('%-8s %-11s %-4s '%('T','family','sol')+' '.join('%9s'%v for v in V))
for key in sorted(fam):
    print('%-8s %-11s %-4s '%key+' '.join('%9.2f'%fam[key][v] for v in V))
