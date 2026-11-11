#!/usr/bin/env python3
"""Build an explicitly labeled shape-adapted/reconstruction supplemental harness.

The generated copy preserves the headline harness. Original strided row-block
matrices are compared to current vector groups without conversion in timing.
New reconstruction cases explicitly retain zeros, compact, then rebuild four times.
"""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
from spec import scenarios,VERSIONS
from run import archive_sources
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--harness',type=Path,required=True);p.add_argument('--sources',type=Path,default=Path('/tmp/pb852-frozen'));a=p.parse_args()
archive_sources(a.sources.resolve())
h=a.harness.resolve();assert not h.exists()
shutil.copytree(HERE,h,ignore=shutil.ignore_patterns('__pycache__'))
file=h/'adapters.jl';s=file.read_text()
s=s.replace('newblock(T,n,b,groups)=ORIGINAL ? BT(n,n÷b) : BT(T,n,groups)',
 '''newblock(T,n,b,groups)=ORIGINAL ? (eltype(groups)<:StepRange ? BT((n÷b,b),n÷b,true) : BT(n,n÷b)) : BT(T,n,groups)''')
file.write_text(s)
file=h/'worker.jl';s=file.read_text().replace('(T==Float32 || layout!="contiguous")','(T==Float32 || layout=="arbitrary")')
s=s.replace('p.id in ("activate_batch","dropzeros")','p.id in ("activate_batch","dropzeros")')
file.write_text(s)
file=h/'cases.jl';s='include("residual_workloads.jl")\n'+file.read_text();needle='        elseif name=="prune_rebuild"'
assert needle in s
s=s.replace(needle,'''        elseif name=="rebuild_cycles"
            f=()->begin
                for _ in 1:4
                    populate_scalar!(x,idx,zerovals,rows,cols)
                    compact!(x)
                    populate_scalar!(x,idx,vals,rows,cols)
                end
                x
            end
            add(name,f,setup,base,active;work=12s)
'''+needle)
# Separate state-copy use contexts: make() owns another equal layout, whereas
# similar(src) shares the validated layout and avoids O(number of groups) equality.
needle2='        elseif name=="clear"\n            f=()->clear!(x);ref=zeros(T,n)'
assert needle2 in s
s=s.replace(needle2,'''        elseif name=="copyto_shared"
            shared=similar(src)
            f=()->copyto!(shared,src);prep=()->fill!(shared,zero(T));work=n
            check=r->verify_block(shared,base,active,groups)
'''+needle2)
# Original has a dedicated active-set bilinear helper even though generic
# dot(a,x,b) traverses its AbstractMatrix interface. Compare that native helper
# against the current dot API, with symmetric operand columns prepared outside
# timing so the mathematical inputs and dense accesses agree.
needle3='        elseif name in ("bilinear","matmat")'
assert needle3 in s
s=s.replace(needle3,'''        elseif name=="bilinear_native"
            av=rand(rng,T,n);bv=rand(rng,T,n)
            AP=zeros(T,n,n);BP=zeros(T,n,n);AP[:,1]=av;AP[1,:]=av;BP[:,1]=bv;BP[1,:]=bv
            AS=Symmetric(AP,:L);BS=Symmetric(BP,:L)
            va=view(AP,:,1);vb=view(BP,:,1)
            Aref=zeros(T,n,n)
            for k in idx;r=rows[k];c=cols[k];Aref[r,c]=Aref[c,r]=base[k];end
            expected=dot(av,Aref*bv)
            f=ORIGINAL ? (()->PB.A_mul_X_mul_B_rc(AS,x,BS,1,1)) : (()->dot(va,x,vb))
            add(name,f,setup,base,active;work=s,check=r->(@assert isapprox(r,expected;rtol=8e-4,atol=1e-5);verify_scalar(x,base,active,rows,cols)))
'''+needle3)
# Independent dense/BLAS references for complete residual-maintaining CD/BCD.
needle4='        elseif name=="rebuild_cycles"'
assert needle4 in s
s=s.replace(needle4,'''        elseif name in ("residual_cd_fixed","residual_cd_expand")
            design=randn(rng,T,16,q);target=randn(rng,T,16)
            lips=vec(sum(abs2,design;dims=1));lambda=T(8)
            initial_residual=target-design[:,idx]*vals;residual=copy(initial_residual)
            schedule=name=="residual_cd_fixed" ? idx : vcat(idx,absent)
            refbeta=copy(base);refresidual=copy(initial_residual);refactive=copy(active)
            for _ in 1:4,k in schedule
                column=view(design,:,k)
                u=refbeta[k]+dot(column,refresidual)/lips[k]
                next=sign(u)*max(abs(u)-lambda/lips[k],zero(T));change=next-refbeta[k]
                refresidual .-= change .* column;refbeta[k]=next
                iszero(next) || push!(refactive,k)
            end
            reference_equal(refresidual,target-design*refbeta)
            f=let design=design,residual=residual,lips=lips,lambda=lambda,schedule=schedule
                ()->residual_cd!(x,schedule,rows,cols,design,residual,lips,lambda,4)
            end
            prep=let residual=residual,initial_residual=initial_residual
                ()->begin setup();copyto!(residual,initial_residual);end
            end
            check=let refbeta=refbeta,refactive=refactive,residual=residual,refresidual=refresidual
                r->begin verify_scalar(x,refbeta,refactive,rows,cols);reference_equal(residual,refresidual);end
            end
            add(name,f,prep,refbeta,refactive;work=4length(schedule),check=check)
'''+needle4)
needle5='        elseif name=="copyto_shared"'
assert needle5 in s
s=s.replace(needle5,'''        elseif name=="residual_bcd"
            design=randn(rng,T,16,n);target=randn(rng,T,16)
            lips=vec(sum(abs2,design;dims=1));steps=T[inv(sum(lips[G])) for G in groups];lambda=T(8)
            initial_residual=target-design*base;residual=copy(initial_residual)
            refbeta=copy(base);refresidual=copy(initial_residual)
            for _ in 1:4,k in ks
                columns=design[:,groups[k]];old=refbeta[groups[k]]
                u=old+steps[k]*(transpose(columns)*refresidual)
                next=sign.(u).*max.(abs.(u).-steps[k]*lambda,zero(T))
                refresidual .-= columns*(next-old);refbeta[groups[k]]=next
            end
            reference_equal(refresidual,target-design*refbeta)
            newvalues=zeros(T,b);deltas=zeros(T,b)
            f=let design=design,residual=residual,steps=steps,lambda=lambda,newvalues=newvalues,deltas=deltas
                ()->residual_bcd!(x,ks,groups,design,residual,steps,lambda,newvalues,deltas,4)
            end
            prep=let residual=residual,initial_residual=initial_residual
                ()->begin setup();copyto!(residual,initial_residual);end
            end
            check=let refbeta=refbeta,residual=residual,refresidual=refresidual
                r->begin verify_block(x,refbeta,active,groups);reference_equal(residual,refresidual);end
            end
            work=4s*b
'''+needle5)
# Original row-block views are native 1×b matrices. Flatten only the
# collected validation result, never the timed view or update operation.
s=s.replace('reference_equal(collect(v),ref[groups[k]])','reference_equal(vec(collect(v)),ref[groups[k]])')
s=s.replace('@assert isapprox(a,b;rtol=eltype(a)==Float32 ? 8e-4 : 2e-10,atol=eltype(a)==Float32 ? 2e-5 : 1e-10)',
'''@assert isapprox(a,b;rtol=eltype(a)==Float32 ? 8e-4 : 2e-10,atol=eltype(a)==Float32 ? 2e-5 : 1e-10) (type=eltype(a), count=length(a), difference_norm=norm(a-b), reference_norm=norm(b), actual_norm=norm(a))''')
file.write_text(s)
chosen=[]
for x in scenarios():
    x=dict(x)
    if x['family']=='block' and x['dtype']=='Float64' and x['layout']=='strided' and x['n']==8192:
        x['ops']=['construct_empty','construct_populated','write_active','block_read','block_sweep','copy','copyto','copyto_shared','clear','block_cd','residual_bcd'];chosen.append(x)
    elif x['family']=='block':
        x['ops']=['copyto_shared','residual_bcd'];chosen.append(x)
    elif x['family'] in ('sparse','symmetric'):
        x['ops']=['rebuild_cycles']+(['bilinear_native'] if x['family']=='symmetric' else ['residual_cd_fixed','residual_cd_expand']);chosen.append(x)
(h/'spec.py').write_text('VERSIONS='+repr(VERSIONS)+'\nSCENARIOS='+repr(chosen)+'\ndef scenarios():return SCENARIOS\n')
(h/'supplement.json').write_text(json.dumps({'cases':sum(len(x['ops']) for x in chosen),'scope':'original strided matrix blocks vs new vector groups; four explicit zero/compact/reconstruct cycles','changes':['original strided public native matrix constructor','permit supported original strided value cases','add rebuild_cycles workload','add copyto_shared layout context','add bilinear_native historical optimized helper; modern prebuilt strided dense-column views','add residual-maintaining lasso CD and BCD workloads with independent dense references']},indent=2)+'\n')
print(h)
