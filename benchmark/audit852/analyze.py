#!/usr/bin/env python3
"""Recompute paired summaries from retained observations and individual samples."""
import argparse,csv,hashlib,json,math,random,statistics as st,tomllib,re
from spec import scenarios
from collections import defaultdict
from pathlib import Path

def interval(values,seed=852,draws=10000):
    # Percentile bootstrap of the paired process-round median. Each round is
    # resampled as a unit; never treat within-trial timing samples as independent.
    rng=random.Random(seed);n=len(values)
    bs=sorted(st.median(rng.choices(values,k=n)) for _ in range(draws))
    return bs[int(.025*draws)],bs[int(.975*draws)]

def load(path,verify=True):
    rows=list(csv.DictReader((path/'observations.csv').open()));seen=set()
    for row in rows:
        key=(row['round'],row['role'],row['case']);assert key not in seen,key;seen.add(key)
        assert row['status'] in ('pass','incompatible'),row
        if row['status']=='incompatible':continue
        for col in ('median_ns','min_ns','p90_ns','bytes','allocs','samples','work','memory_bytes','warm_seconds'):
            row[col]=float(row[col])
    return rows

def verify_completeness(path,rows):
    campaign=json.loads((path/'campaign.json').read_text());args=campaign['argv']
    selected=None
    if args.get('cases_file'):
        selection=path/'selection.txt'
        if not selection.exists():selection=Path(args['cases_file'])
        assert selection.exists(), 'selected case manifest missing; retain selection.txt in campaign'
        selected=set(selection.read_text().splitlines())
    cases={x['key']+'/'+op for x in scenarios() for op in x['ops']
           if re.search(args.get('filter',''),x['key']+'/'+op)
           and (selected is None or x['key']+'/'+op in selected)}
    assert len(cases)==args['chosen_case_count'], 'specification/selection count mismatch'
    start=args['start_round'];rounds=range(start,start+args['rounds'])
    expected={(str(r),role,case) for r in rounds for role in args['roles'].split(',') for case in cases}
    observed={(r['round'],r['role'],r['case']) for r in rows}
    assert expected==observed, ('missing/extra campaign records',len(expected-observed),len(observed-expected))
    return len(expected)

def verify_samples(path,rows):
    metrics={}
    for file in path.rglob('fixture-*.csv'):
        if file.name.endswith('.samples.csv'):continue
        records=list(csv.DictReader(file.open()))
        for r in records:
            if r['status']!='pass':continue
            samplepath=Path(str(file)+'.'+r['case'].rsplit('/',1)[1]+'.samples.csv')
            assert samplepath.exists(),samplepath
            sample=list(csv.DictReader(samplepath.open()));times=[float(x['time_ns']) for x in sample]
            assert len(times)==int(r['samples'])
            assert st.median(times)==float(r['median_ns']),(file,r['case'],'median')
            assert min(times)==float(r['min_ns']),(file,r['case'],'minimum')
            assert all(float(x['gc_ns'])>=0 for x in sample)
            fixture=tomllib.loads(Path(str(file)+'.fixture.toml').read_text())
            key=(str(fixture['seed']),file.parent.name,r['case'])
            assert key not in metrics,key
            metrics[key]=(float(r['median_ns']),float(r['bytes']),float(r['allocs']))
    for r in rows:
        if r['status']=='pass':
            key=(str(r['seed']),r['role'],r['case'])
            assert metrics.get(key)==(r['median_ns'],r['bytes'],r['allocs']),(key,'observation/trial mismatch')
    assert len(metrics)==sum(r['status']=='pass' for r in rows), 'orphan trial records'

def summarize(rows,draws):
    grouped=defaultdict(dict)
    for r in rows:grouped[(r['case'],r['role'])][int(r['round'])]=r
    pairs=[('original','post850'),('historical05','post850'),('post848','post850'),
           ('original','historical05'),('historical05','post848'),('original','originalAA'),('post850','post850AA')]
    out=[]
    for case in sorted({r['case'] for r in rows}):
        for before,after in pairs:
            A=grouped.get((case,before),{});B=grouped.get((case,after),{})
            if not A or not B:continue
            assert set(A)==set(B),(case,before,after,'missing pair')
            rs=sorted(A);statuses={(A[r]['status'],B[r]['status']) for r in rs}
            assert len(statuses)==1,(case,'changing compatibility')
            if statuses!={('pass','pass')}:continue
            ratios=[B[r]['median_ns']/A[r]['median_ns'] for r in rs]
            lo,hi=interval(ratios,draws=draws)
            aa_role='originalAA' if before=='original' else 'post850AA'
            ref_role='original' if before=='original' else 'post850'
            AA=grouped.get((case,aa_role),{});R=grouped.get((case,ref_role),{})
            aa=[AA[r]['median_ns']/R[r]['median_ns'] for r in rs if r in AA and r in R and AA[r]['status']==R[r]['status']=='pass']
            at=st.median(A[r]['median_ns'] for r in rs);bt=st.median(B[r]['median_ns'] for r in rs)
            out.append(dict(case=case,before=before,after=after,rounds=len(rs),before_ns=at,after_ns=bt,
                ratio=st.median(ratios),ci95_low=lo,ci95_high=hi,change_percent=(st.median(ratios)-1)*100,
                ratio_min=min(ratios),ratio_max=max(ratios),ratio_pstdev=st.pstdev(ratios),
                before_bytes=st.median(A[r]['bytes'] for r in rs),after_bytes=st.median(B[r]['bytes'] for r in rs),
                before_allocs=st.median(A[r]['allocs'] for r in rs),after_allocs=st.median(B[r]['allocs'] for r in rs),
                before_memory=st.median(A[r]['memory_bytes'] for r in rs),after_memory=st.median(B[r]['memory_bytes'] for r in rs),
                aa_ratio=st.median(aa) if aa else '',aa_min=min(aa) if aa else '',aa_max=max(aa) if aa else '',
                before_p90=st.median(A[r]['p90_ns'] for r in rs),after_p90=st.median(B[r]['p90_ns'] for r in rs),
                pairs_above_5=sum(v>1.05 for v in ratios),work=B[rs[0]]['work'],
                before_ops_per_second=A[rs[0]]['work']*1e9/at,after_ops_per_second=B[rs[0]]['work']*1e9/bt,
                timer_limited=min(at,bt)<100,
                warm_before_median_seconds=st.median(A[r]['warm_seconds'] for r in rs),
                warm_after_median_seconds=st.median(B[r]['warm_seconds'] for r in rs)))
    return out

def writecsv(path,rows):
    if not rows:path.write_text('');return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def tables(out,summary):
    primary=[x for x in summary if x['before']=='original' and x['after']=='post850']
    lines=['# Measured primary comparisons','', 'R = post850/original; intervals resample paired process rounds. Values are ns per complete case.','',
           '| Case | Original ns | Post850 ns | R [95% CI] | Bytes old→new | Allocs old→new | A/A |','|---|---:|---:|---|---|---|---:|']
    for x in primary:
        lines.append(f"| {x['case']} | {x['before_ns']:.1f} | {x['after_ns']:.1f} | {x['ratio']:.3f} [{x['ci95_low']:.3f}, {x['ci95_high']:.3f}] | {x['before_bytes']:g}→{x['after_bytes']:g} | {x['before_allocs']:g}→{x['after_allocs']:g} | {x['aa_ratio']:.3f} |")
    (out/'tables.md').write_text('\n'.join(lines)+'\n')

def plots(out,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'svg.hashsalt':'audit852'})
    primary={r['case']:r for r in summary if r['before']=='original' and r['after']=='post850'}
    for family,ops in [('sparse',['copyto','clear','dot','cd_fixed']),('symmetric',['copyto','clear','matmat','symmetric_updates'])]:
        fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
        for ax,op in zip(axes.flat,ops):
            for n in ([10000,1000000] if family=='sparse' else [144,1414]):
                for label,col,sty in [('original','#8c510a','--'),('post850','#01665e','-')]:
                    values=[]
                    for s in [100,1000,10000]:
                        case=f'{family}/Float64/n{n}/s{s}/b1/'+('scalar' if family=='sparse' else 'triangle')+'/'+op
                        r=primary.get(case)
                        if r:values.append((s,r['before_ns' if label=='original' else 'after_ns']/1000))
                    if values:ax.plot(*zip(*values),marker='o',ls=sty,color=col,alpha=1 if n in [1000000,1414] else .5,label=f'{label}, n={n}')
            ax.set(xscale='log',yscale='log',xlabel='Active coordinates',ylabel='Microseconds / operation',title=op);ax.grid(alpha=.25)
        axes.flat[0].legend(fontsize=7);fig.suptitle(f'{family}: Float64 warmed execution (same machine)')
        fig.savefig(out/f'{family}-scaling.svg',metadata={'Date':None});fig.savefig(out/f'{family}-scaling.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,op in zip(axes,['block_cd','matvec','construct_empty']):
        for pct in [1,10]:
            for label,col,sty in [('original','#8c510a','--'),('post850','#01665e','-')]:
                vals=[]
                for b in [1,8,64]:
                    s=max(1,(8192//b)*pct//100);r=primary.get(f'block/Float64/n8192/s{s}/b{b}/contiguous/{op}')
                    if r:vals.append((b,r['before_ns' if label=='original' else 'after_ns']/1000))
                if vals:ax.plot(*zip(*vals),marker='o',ls=sty,color=col,alpha=1 if pct==10 else .5,label=f'{label}, {pct}%')
        ax.set(xscale='log',yscale='log',xlabel='Block size',ylabel='Microseconds / operation',title=op);ax.grid(alpha=.25)
    axes[0].legend(fontsize=7);fig.suptitle('Contiguous blocks: value-equivalent operations, Float64')
    fig.savefig(out/'block-scaling.svg',metadata={'Date':None});fig.savefig(out/'block-scaling.png',dpi=160);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--draws',type=int,default=10000);p.add_argument('--plots',action='store_true');p.add_argument('--verify',action='store_true')
    a=p.parse_args();rows=load(a.campaign,verify=False)
    verify_completeness(a.campaign,rows)
    if a.verify:verify_samples(a.campaign,rows)
    summary=summarize(rows,a.draws);a.output.mkdir(parents=True,exist_ok=True)
    writecsv(a.output/'summary.csv',summary);tables(a.output,summary)
    compat=[{k:r[k] for k in ('case','role','status','error')} for r in rows if r['status']!='pass']
    compat=list({tuple(r.values()):r for r in compat}.values());writecsv(a.output/'incompatibilities.csv',compat)
    flags=sorted({r['case'] for r in summary if r['after']=='post850' and r['before'] in ('original','post848') and r['ratio']>1.05})
    (a.output/'suspicious.txt').write_text('\n'.join(flags)+'\n')
    (a.output/'analysis.json').write_text(json.dumps({'observations_sha256':hashlib.sha256((a.campaign/'observations.csv').read_bytes()).hexdigest(),'rows':len(rows),'comparisons':len(summary),'flagged_cases':len(flags),'bootstrap_draws':a.draws,'bootstrap_seed':852,'unit':'paired process round','raw_samples_verified':a.verify},indent=2)+'\n')
    if a.plots:plots(a.output,summary)
    print(json.loads((a.output/'analysis.json').read_text()))
if __name__=='__main__':main()
