using Random, LinearAlgebra, SparseArrays
struct AuditCase{F,S,V}
    id::String
    op::F
    setup::S
    validate::V
    work::Int
end
noop()=nothing
function reference_equal(a,b)
    @assert size(a)==size(b)
    @assert isapprox(a,b;rtol=eltype(a)==Float32 ? 8e-4 : 2e-10,atol=eltype(a)==Float32 ? 2e-5 : 1e-10)
end
# Explicit schedules are shared. References never call package index mappings.
function triangle(p)
    rows=Int[];cols=Int[]
    for c in 1:p,r in c:p;push!(rows,r);push!(cols,c);end
    rows,cols
end
function group_layout(n,b,layout,rng)
    K=n÷b
    layout=="contiguous" && return [(k-1)*b+1:k*b for k in 1:K]
    layout=="strided" && return [k:K:n for k in 1:K]
    perm=randperm(rng,n)
    [perm[(k-1)*b+1:k*b] for k in 1:K]
end
function verify_scalar(x,ref,active,rows,cols)
    # Check all independent values, exact activity set/count, map consistency,
    # and symmetry at every stored coordinate.
    got=[coord(x,k,rows,cols) for k in eachindex(ref)]
    reference_equal(got,ref)
    a=collect(ids(x)); @assert length(a)==length(active)==x.nnz
    @assert Set(a)==Set(active)
    for (pos,k) in enumerate(a)
        @assert x.ind2nzval[k]==pos
        if x isa IS_SYM
            @assert x[rows[k],cols[k]]==x[cols[k],rows[k]]==ref[k]
        end
    end
    @assert count(!iszero,x.ind2nzval)==length(active)
end
function verify_block(x,ref,active,groups)
    reference_equal([x[i] for i in eachindex(ref)],ref)
    if !ORIGINAL
        @assert Set(ids(x))==Set(active)
        @assert PB.nactive(x)==length(active)
        for k in eachindex(groups); @assert PB.isactive(x,k)==(k in active);end
    end
    for k in (isempty(active) ? Int[] : [minimum(active),maximum(active)])
        v=block(x,k);reference_equal(collect(v),ref[groups[k]])
        old=v[1];v[1]=old+one(old);@assert x[groups[k][1]]==old+one(old);v[1]=old
    end
end
function populate_scalar!(x,idx,vals,rows,cols)
    for (j,k) in enumerate(idx);coordset!(x,vals[j],k,rows,cols);end
    x
end
function read_sum(x,idx,rows,cols)
    value=zero(eltype(x))
    for k in idx;value+=coord(x,k,rows,cols);end
    value
end
function update!(x,idx,rows,cols,repeats)
    T=eltype(x)
    for _ in 1:repeats,k in idx
        coordset!(x,coord(x,k,rows,cols)+T(1/8),k,rows,cols)
    end
    x
end
function cd!(x,idx,rows,cols,repeats)
    T=eltype(x)
    for _ in 1:repeats,k in idx
        u=T(7/8)*coord(x,k,rows,cols)+T(1/4)
        coordset!(x,max(u-T(1/16),zero(T)),k,rows,cols)
    end
    x
end
function scalar_cases(fam,T,n,s,ops,seed)
    rng=Xoshiro(seed); rows,cols=fam=="sparse" ? (Int[],Int[]) : triangle(n)
    q=fam=="sparse" ? n : length(rows)
    idx=randperm(rng,q)[1:s]; sequential=sort(idx)
    absent=setdiff(1:q,idx)[1:min(256,q-s)];batchvals=fill(T(1/2),length(absent));zerovals=zeros(T,s)
    vals=T.((mod.(1:s,31).+1)./32); base=zeros(T,q);base[idx]=vals
    make=()->newscalar(fam,T,n)
    src=populate_scalar!(make(),idx,vals,rows,cols);x=make();dest=make()
    setup=()->copyto!(x,src)
    dense=copy(base); densev=rand(rng,T,q)
    active=Set(idx); cases=AuditCase[]
    function add(op,f,prep,ref,act;work=s,check=nothing)
        valid=check===nothing ? ((result)->verify_scalar(result isa typeof(x) ? result : x,ref,act,rows,cols)) : check
        push!(cases,AuditCase(op,f,prep,valid,work))
    end
    # Copy independence, self-copy and zero semantics are checked independently
    # of case timing; original generic copy failures are reported per case below.
    z=make(); k=idx[1];coordset!(z,one(T),k,rows,cols);coordset!(z,zero(T),k,rows,cols)
    verify_scalar(z,zeros(T,q),Set([k]),rows,cols)
    if !isempty(absent);coordset!(z,zero(T),absent[1],rows,cols);@assert z.nnz==1;end
    for name in ops
        if name=="construct_empty"
            add(name,make,noop,zeros(T,q),Set{Int}();work=q)
        elseif name=="construct_populated"
            add(name,()->populate_scalar!(make(),idx,vals,rows,cols),noop,base,active;work=q+s)
        elseif name=="activate_batch"
            ref=copy(base);ref[absent].=T(1/2)
            add(name,()->populate_scalar!(x,absent,batchvals,rows,cols),setup,ref,union(active,absent);work=length(absent))
        elseif name=="write_active"
            add(name,()->populate_scalar!(x,idx,vals,rows,cols),setup,base,active)
        elseif name=="repeat_update"
            ref=copy(base);ref[idx].+=T(1/8)*8
            add(name,()->update!(x,idx,rows,cols,8),setup,ref,active;work=8s)
        elseif name=="active_sweep"
            ref=copy(base);ref[idx].+=T(1/8)
            add(name,()->update!(x,ids(x),rows,cols,1),setup,ref,active)
        elseif name in ("read_random","read_sequential","read_scalar","read_diagonal","read_offdiagonal")
            inds=name=="read_sequential" ? sequential : name=="read_scalar" ? [idx[1]] : name=="read_diagonal" ? findall(rows.==cols) : name=="read_offdiagonal" ? idx[rows[idx].!=cols[idx]] : idx
            expected=sum(base[inds])
            f=let inds=inds
                name=="read_scalar" ? (()->coord(x,inds[1],rows,cols)) : (()->read_sum(x,inds,rows,cols))
            end
            add(name,f,setup,base,active;work=length(inds),check=r->(@assert isapprox(r,expected;rtol=2e-4,atol=1e-5);verify_scalar(x,base,active,rows,cols)))
        elseif name=="write_scalar"
            add(name,()->coordset!(x,vals[1],idx[1],rows,cols),setup,base,active;work=1,check=r->verify_scalar(x,base,active,rows,cols))
        elseif name=="dense_import"
            add(name,()->PB.SparseIterate(dense),noop,base,active;work=q)
        elseif name=="copy"
            add(name,()->copy(src),noop,base,active;work=q,check=r->begin
                verify_scalar(r,base,active,rows,cols)
                coordset!(r,T(9),idx[1],rows,cols);@assert coord(src,idx[1],rows,cols)==vals[1]
            end)
        elseif name=="copyto"
            prep=()->begin copyto!(dest,src);coordset!(dest,T(3),isempty(absent) ? idx[1] : absent[1],rows,cols);end
            add(name,()->copyto!(dest,src),prep,base,active;work=q,check=r->verify_scalar(dest,base,active,rows,cols))
        elseif name=="clear"
            add(name,()->clear!(x),setup,zeros(T,q),active;work=s)
        elseif name=="zero_active"
            add(name,()->populate_scalar!(x,idx,zerovals,rows,cols),setup,zeros(T,q),active)
        elseif name=="dropzeros"
            half=idx[1:2:end];ref=copy(base);ref[half].=0
            prep=()->begin setup();populate_scalar!(x,half,zeros(T,length(half)),rows,cols);end
            add(name,()->compact!(x),prep,ref,setdiff(active,half))
        elseif name=="dot"
            expected=sum(densev.*base)
            add(name,()->dot(densev,x),setup,base,active;check=r->(@assert isapprox(r,expected;rtol=2e-4,atol=1e-5);verify_scalar(x,base,active,rows,cols)))
        elseif name=="matvec"
            # 16 dense rows at both ambient sizes; matrix generation outside timing.
            A=rand(rng,T,16,q);out=zeros(T,16);expected=A[:,idx]*vals
            f=let A=A,out=out;()->mul!(out,A,x);end
            add(name,f,setup,base,active;work=16s,check=r->(reference_equal(out,expected);verify_scalar(x,base,active,rows,cols)))
        elseif name in ("cd_fixed","cd_expand")
            inds=name=="cd_fixed" ? idx : vcat(idx,absent)
            ref=copy(base)
            for _ in 1:4,k in inds;ref[k]=max(T(7/8)*ref[k]+T(1/4)-T(1/16),zero(T));end
            f=let inds=inds;()->cd!(x,inds,rows,cols,4);end
            add(name,f,setup,ref,union(active,inds);work=4length(inds))
        elseif name=="prune_rebuild"
            half=idx[1:2:end];rebuild=isempty(absent) ? half[1:min(256,length(half))] : absent;ref=copy(base);ref[half].=0;ref[rebuild].=T(1/2)
            f=let half=half,rebuild=rebuild
              ()->begin
                for k in half;coordset!(x,zero(T),k,rows,cols);end
                compact!(x)
                for k in rebuild;coordset!(x,T(1/2),k,rows,cols);end
                x
            end
            end
            add(name,f,setup,ref,union(setdiff(active,half),rebuild);work=s+length(rebuild))
        elseif name=="symmetric_updates"
            # Identical public matrix indexing, independent of coordinate adapters.
            ref=copy(base)
            for _ in 1:8,k in idx;ref[k]+=T(1/8);end
            f=()->begin
                for _ in 1:8,k in idx;r=rows[k];c=cols[k];x[r,c]=x[r,c]+T(1/8);end
                x
            end
            add(name,f,setup,ref,active;work=8s)
        elseif name in ("bilinear","matmat")
            Aref=zeros(T,n,n)
            for k in idx;r=rows[k];c=cols[k];Aref[r,c]=Aref[c,r]=base[k];end
            if name=="bilinear"
                a=rand(rng,T,n);b=rand(rng,T,n);expected=dot(a,Aref*b)
                f=let a=a,b=b;()->dot(a,x,b);end
                add(name,f,setup,base,active;check=r->(@assert isapprox(r,expected;rtol=8e-4,atol=1e-5);verify_scalar(x,base,active,rows,cols)))
            else
                B=rand(rng,T,n,8);out=zeros(T,n,8);expected=Aref*B
                f=let out=out,B=B;()->mul!(out,x,B);end
                add(name,f,setup,base,active;work=8s,check=r->(reference_equal(out,expected);verify_scalar(x,base,active,rows,cols)))
            end
        else
            error("unknown scalar case: $name")
        end
    end
    cases,src
end
function block_update!(x,ks,repeats,cd)
    T=eltype(x)
    for _ in 1:repeats,k in ks
        v=block(x,k)
        for j in eachindex(v)
            v[j]=cd ? max(T(7/8)*v[j]+T(1/4)-T(1/16),zero(T)) : v[j]+T(1/8)
        end
    end
    x
end
function block_cases(T,n,s,b,layout,ops,seed)
    rng=Xoshiro(seed);groups=group_layout(n,b,layout,rng);K=length(groups)
    ks=randperm(rng,K)[1:s];active=Set(ks);absent=setdiff(1:K,ks)[1:min(256,K-s)]
    base=zeros(T,n)
    for k in ks;base[groups[k]].=T(1/2);end
    make=()->newblock(T,n,b,groups)
    src=make()
    for i in findall(!iszero,base);src[i]=base[i];end
    x=make();dest=make();setup=()->copyto!(x,src);cases=AuditCase[]
    for name in ops
        f=noop;prep=setup;ref=base;act=active;work=s*b;check=nothing
        if name=="construct_empty"
            f=make;prep=noop;ref=zeros(T,n);act=Set{Int}();work=n
        elseif name=="construct_populated"
            f=()->begin y=make();for i in eachindex(base);iszero(base[i]) || (y[i]=base[i]);end;y;end
            prep=noop;work=n
        elseif name=="activate_batch"
            # Original has no active metadata: driver reports incompatibility.
            f=()->begin for k in absent;PB.activate!(x,k);end;x;end
            act=union(active,absent);work=length(absent)
        elseif name=="write_active"
            f=()->begin for k in ks;fill!(block(x,k),T(1/2));end;x;end
        elseif name=="block_read"
            f=()->sum(sum(block(x,k)) for k in ks);expected=sum(base)
            check=r->(@assert isapprox(r,expected;rtol=2e-4,atol=1e-5);verify_block(x,base,active,groups))
        elseif name=="block_sweep"
            f=()->block_update!(x,ks,1,false);ref=copy(base)
            for k in ks;ref[groups[k]].+=T(1/8);end
        elseif name=="copy"
            f=()->copy(src);prep=noop;work=n
            check=r->begin verify_block(r,base,active,groups);r[groups[ks[1]][1]]=T(9);@assert src[groups[ks[1]][1]]==T(1/2);end
        elseif name=="copyto"
            f=()->copyto!(dest,src);prep=()->fill!(dest,zero(T));work=n
            check=r->verify_block(dest,base,active,groups)
        elseif name=="clear"
            f=()->clear!(x);ref=zeros(T,n)
        elseif name=="dropzeros"
            half=ks[1:2:end];ref=copy(base);for k in half;ref[groups[k]].=0;end
            prep=()->begin setup();for k in half;fill!(block(x,k),zero(T));end;end
            f=()->compact!(x);act=setdiff(active,half)
        elseif name=="dot"
            v=rand(rng,T,n);expected=dot(v,base);f=()->dot(v,x)
            check=r->(@assert isapprox(r,expected;rtol=8e-4,atol=1e-5);verify_block(x,base,active,groups))
        elseif name=="matvec"
            A=rand(rng,T,16,n);out=zeros(T,16);expected=A*base;f=()->mul!(out,A,x);work=16s*b
            check=r->(reference_equal(out,expected);verify_block(x,base,active,groups))
        elseif name=="block_cd"
            f=()->block_update!(x,ks,4,true);ref=copy(base);work=4s*b
            for _ in 1:4,k in ks,j in groups[k];ref[j]=max(T(7/8)*ref[j]+T(1/4)-T(1/16),zero(T));end
        else
            error("unknown block case: $name")
        end
        valid=check===nothing ? (r->verify_block(r isa typeof(x) ? r : x,ref,act,groups)) : check
        push!(cases,AuditCase(name,f,prep,valid,work))
    end
    # Prove a zero write to an absent block does not activate it; existing
    # active blocks retain explicit zeros until dropzeros!.
    if !ORIGINAL
        copyto!(x,src);fill!(block(x,ks[1]),zero(T));@assert PB.isactive(x,ks[1])
        if !isempty(absent);fill!(block(x,absent[1]),zero(T));@assert !PB.isactive(x,absent[1]);end
    end
    cases,src
end
