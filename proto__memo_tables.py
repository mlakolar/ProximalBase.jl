# Markdown tables for memo §5 from the exclusive-node outputs (no retyping).
import collections, statistics as st
def med(xs):
    s=sorted(xs); n=len(s); return (s[(n-1)//2]+s[n//2])/2
out=[]
# (d) end-to-end
out.append('| Julia | Solver | K2/K (A/A) | K3/K (A/A) | B1 | B2 | B3 | **H** | C |')
out.append('|---|---|---|---|---|---|---|---|---|')
for v in ['1.13.3','1.10.12']:
    rows=[l.rstrip('\n').split(',',8) for l in open(f'out/excl/xeon-out-{v}.txt') if l.startswith(('time,','diag,'))]
    T=collections.defaultdict(dict); D=collections.defaultdict(dict)
    for r in rows:
        k=(r[2],r[3],r[4])
        if r[0]=='time' and r[6] in ('ratio_median_paired','ratio_ci_lo','ratio_ci_hi'): T[k].setdefault(r[5],{})[r[6]]=float(r[8])
        elif r[0]=='diag': D[k].setdefault(r[5],{})[r[6]]=r[8]
    for s in ['cd','cds','pg','apg']:
        keys=[k for k in T if k[1]==s]
        cells=[]
        for nm in ['K2','K3','B1','B2','B3','H','C']:
            same=[k for k in keys if D[k][nm]['bitwise_x_equal_K']=='true' and D[k][nm]['iterations']==D[k]['K']['iterations']]
            r=[T[k][nm]['ratio_median_paired'] for k in same]
            c='%.3f / %.3f / %.3f [%.3f, %.3f]'%(min(r),med(r),max(r),min(T[k][nm]['ratio_ci_lo'] for k in same),max(T[k][nm]['ratio_ci_hi'] for k in same)) if r else '—'
            if len(same)<len(keys):
                ch=[k for k in keys if k not in same]
                q=[T[k][nm]['ratio_median_paired']*float(D[k]['K']['iterations'])/float(D[k][nm]['iterations']) for k in ch]
                c+=' (n=%d); changed trajectory n=%d, per-iteration %.3f–%.3f'%(len(same),len(ch),min(q),max(q))
            cells.append(c)
        out.append('| %s | %s (%d) | '%(v,s,len(keys))+' | '.join(cells)+' |')
out.append('')
# (c) kernels: per op, range over sizes and scales, ordinary inputs; and recovery inputs
out.append('| Julia | Call | inputs | K (median ns, range) | K2/K3 | B1 | B2 | B3 | **H** | C |')
out.append('|---|---|---|---|---|---|---|---|---|---|')
for v in ['1.13.3','1.10.12']:
    rows=[l.rstrip('\n').split(',',8) for l in open(f'out/excl/xeon-out-{v}.txt') if l.startswith('kern,')]
    K=collections.defaultdict(dict)
    for r in rows:
        if r[6] in ('ratio_median_paired','median_ns'): K[(r[2],r[3],r[4])].setdefault(r[5],{})[r[6]]=float(r[8])
    for op in ['value','value_and_gradient','residual_value']:
        for kind,lab in (('ordinary','ordinary, n = 2000 and 240'),('underflow','squares underflow (recovery), n = 240')):
            ks=[k for k in K if k[1]==op and kind in k[0]]
            kn=[K[k]['K']['median_ns'] for k in ks]
            def rng(nm): r=[K[k][nm]['ratio_median_paired'] for k in ks]; return '%.2f–%.2f'%(min(r),max(r))
            aa='%.2f–%.2f'%(min(min(K[k]['K2']['ratio_median_paired'],K[k]['K3']['ratio_median_paired']) for k in ks),max(max(K[k]['K2']['ratio_median_paired'],K[k]['K3']['ratio_median_paired']) for k in ks))
            out.append('| %s | `%s` | %s | %.0f–%.0f | %s | %s | %s | %s | **%s** | %s |'%(v,op,lab,min(kn),max(kn),aa,rng('B1'),rng('B2'),rng('B3'),rng('H'),rng('C')))
out.append('')
# (e) compile
out.append('| Julia | Variant | precompile PB (s, median of 3) | precompile PS (s) | cache PB / PS (MB) | load (s, median of 5) | first 6 solves, scaled LS (s) | first 6 solves, raw LS (s) | added in the oracle file: functions / methods / specializations after load |')
out.append('|---|---|---|---|---|---|---|---|---|')
for v in ['1.13.3','1.10.12']:
    L=open(f'out/excl/xeon-out-compile-{v}.txt').read().splitlines()
    pre=collections.defaultdict(list); tt=collections.defaultdict(list); cnt={}
    for l in L:
        if l.startswith('CSV,precompile'):
            _,_,_,var,rep,tpb,tps,bpb,bps=l.split(','); pre[var].append((float(tpb),float(tps),int(bpb),int(bps)))
        elif l.startswith('CSV,ttfx'):
            parts=l.split(','); d={}
            for p in parts[5:]:
                if p.startswith('load='): d['load']=float(p[5:])
                elif '=' in p:
                    k,val=p.split('='); d[k]=tuple(map(float,val.split('/')))
            tt[parts[3]].append(d)
        elif l.startswith('CSV,count'):
            f=dict(x.split('=',1) for x in l.split(',')[5:8]); cnt[l.split(',')[3]]='%s / %s / %s'%(f['functions'],f['methods'],f['specializations_after_load'])
    for var in ['K','B1','B2','B3','H','C']:
        p=pre[var]; t=tt[var]; keys=[k for k in t[0] if k!='load']
        out.append('| %s | %s | %.2f (%s) | %.2f (%s) | %.2f / %.2f | %.3f | %.3f | %.3f | %s |'%(v,'**H**' if var=='H' else var,med([x[0] for x in p]),', '.join('%.2f'%x[0] for x in p),med([x[1] for x in p]),', '.join('%.2f'%x[1] for x in p),p[0][2]/1e6,p[0][3]/1e6,med([d['load'] for d in t]),med([sum(d[k][0] for k in keys) for d in t]),med([sum(d[k][2] for k in keys) for d in t]),cnt.get(var,'—')))
print('\n'.join(out))
