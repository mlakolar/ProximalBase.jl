import sys,collections
rows=[l.rstrip('\n').split(',',8) for l in open(sys.argv[1]) if l.startswith('solve,')]
d=collections.defaultdict(dict)
for r in rows:
    d[tuple(r[2:6])].setdefault(r[6],{})[r[7]]=r[8]
diff=0
for k,v in sorted(d.items()):
    K=v.get('K',{})
    for nm,m in v.items():
        if nm=='K': continue
        if m.get('meets')!=K.get('meets') or ('error' in m)!=('error' in K):
            diff+=1; print('DIFF',k,nm,'meets',m.get('meets'),'K',K.get('meets'),'kkt %s vs K %s'%(m.get('kkt'),K.get('kkt')),m.get('error','')[:70],'|',K.get('error','')[:70])
print('configs',len(d),'status differences vs K',diff)
for k,v in sorted(d.items()):
    K=v.get('K',{})
    if K.get('meets')=='false': print('K-miss',k,{nm:(m.get('kkt','err')[:9]) for nm,m in v.items()})
    if 'error' in K: print('K-error',k,K['error'][:100])
