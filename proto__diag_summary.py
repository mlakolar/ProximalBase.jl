import sys,collections
for path in sys.argv[1:]:
    rows=[l.rstrip('\n').split(',',8) for l in open(path) if l.startswith(('diag,','alloc,','count,'))]
    D=collections.defaultdict(dict); A=collections.defaultdict(dict); C=collections.defaultdict(dict)
    for r in rows:
        k=(r[2],r[3],r[4]); tgt={'diag':D,'alloc':A,'count':C}[r[0]]
        tgt[k].setdefault(r[5],{})[r[6]]=r[8]
    print('=====',path)
    variants=sorted({v for d in D.values() for v in d})
    stat=collections.Counter(); bit=collections.Counter(); itdiff=collections.Counter(); better=collections.Counter(); worse=collections.Counter()
    for k,d in sorted(D.items()):
        K=d['K']
        for v,m in d.items():
            if v=='K': continue
            if m['meets_target']!=K['meets_target']:
                stat[v]+=1; print('  status differs',k,v,'meets',m['meets_target'],'K',K['meets_target'],'cert %s vs K %s'%(m['cert_value'],K['cert_value']),'iters',m['iterations'],K['iterations'])
            if m['bitwise_x_equal_K']=='true': bit[v]+=1
            if m['iterations']!=K['iterations']: itdiff[v]+=1
            cv,ck=float(m['cert_value']),float(K['cert_value'])
            if cv<ck: better[v]+=1
            if cv>ck: worse[v]+=1
    n=len(D)
    print('  configs',n)
    for v in variants:
        if v=='K': continue
        print('  %-4s status_diff=%d bitwise_x=%d/%d iter_diff=%d cert_better=%d cert_worse=%d'%(v,stat[v],bit[v],n,itdiff[v],better[v],worse[v]))
    nz=[(k,v,m) for k,d in A.items() for v,m in d.items() if any(m.get(x,'0')!='0' for x in ('iteration_bytes','pass_bytes','loss_value_bytes'))]
    print('  nonzero warm alloc entries:',nz[:5], len(nz))
    sb=collections.defaultdict(set)
    for k,d in A.items():
        for v,m in d.items(): sb[k].add(m['solve_bytes'])
    print('  configs with differing solve_bytes across variants:',sum(1 for s in sb.values() if len(s)>1))
    fo=[(k,v) for k,d in C.items() for v,m in d.items() if m.get('formula_ok','true')!='true']
    print('  count formula failures:',fo)
    miss=collections.Counter((k[1],k[2]) for k,d in D.items() if d['K']['meets_target']=='false')
    print('  K misses target by (solver,T):',dict(miss))
