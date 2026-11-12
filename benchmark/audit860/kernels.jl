# Reuse #852's mathematical kernels and independent layout/reference helpers.
include(joinpath(@__DIR__, "../audit852/adapters.jl"))
include(joinpath(@__DIR__, "../audit852/cases.jl"))
include(joinpath(@__DIR__, "../audit852/residual_workloads.jl"))

# Diagnostics only: the raw route deliberately does not maintain activity.
@inline diagnostic_view(x, k, ::Val{:owner}) = block(x, k)
@inline diagnostic_view(x, k, ::Val{:raw}) = ORIGINAL ? block(x,k) : view(x.storage, x.groups[k])
@inline function diagnostic_view(x, k, ::Val{:once})
    PB.isactive(x, k) || error("once diagnostic requires active groups")
    view(x.storage, x.groups[k])
end

function diagnostic_sweep!(x, ks, route)
    T = eltype(x)
    for _ in 1:4, k in ks
        v = diagnostic_view(x, k, route)
        for j in eachindex(v)
            v[j] = max(T(7/8)*v[j]+T(1/4)-T(1/16), zero(T))
        end
    end
    x
end

# Same residual BCD, using the documented buffer + copyto!(blockview, buffer)
# idiom. The coefficient block is committed before its residual update; the
# gradient and residual arithmetic and the coordinate schedule are unchanged.
function residual_bulk!(x,ks,groups,A,r,steps,lambda,nv,d,repeats)
    T=eltype(x)
    for _ in 1:repeats,k in ks
        v=block(x,k); gamma=steps[k]
        for (j,i) in enumerate(groups[k])
            g=zero(T)
            for row in axes(A,1); g+=A[row,i]*r[row]; end
            old=v[j]; nv[j]=residual_soft(old+gamma*g,gamma*lambda)
            d[j]=nv[j]-old
        end
        copyto!(v,nv)
        for (j,i) in enumerate(groups[k]),row in axes(A,1)
            r[row]-=d[j]*A[row,i]
        end
    end
    x
end

function copy_blocks!(x,ks,buffer)
    for k in ks
        copyto!(block(x,k),buffer)
    end
    x
end

function copy_fixture(T,n,b,groups,ks,state)
    initial=zeros(T,n)
    if state in (:active,:zero_active)
        for k in ks,i in groups[k];initial[i]=T(1/2);end
    end
    x=if ORIGINAL
        storage=copy(initial)
        PB.AtomIterate(storage,[view(storage,G) for G in groups])
    else
        PB.BlockIterate(initial,groups)
    end
    if !ORIGINAL && state==:active_zero
        for k in ks;PB.activate!(x,k);end
    end
    buffer=fill(state in (:zero_inactive,:zero_active) ? zero(T) : one(T),b)
    active0=ORIGINAL ? Int[] : copy(x.active)
    position0=ORIGINAL ? Int[] : copy(x.position)
    count0=ORIGINAL ? 0 : x.nactive
    prep=()->begin
        copyto!(x.storage,initial)
        if !ORIGINAL
            copyto!(x.active,active0);copyto!(x.position,position0);x.nactive=count0
        end
        nothing
    end
    op=()->copy_blocks!(x,ks,buffer)
    expected=copy(initial)
    for k in ks;expected[groups[k]]=buffer;end
    prep();op();reference_equal(x.storage,expected)
    if !ORIGINAL
        expected_active=state==:zero_inactive ? Set{Int}() : Set(ks)
        @assert Set(PB.activecoordinates(x))==expected_active
        @assert x.nactive==length(expected_active)
        for (pos,k) in enumerate(PB.activecoordinates(x));@assert x.position[k]==pos;end
    end
    (;op,prep)
end

function make_fixture(T, n, b, layout, seed; lambda=T(8))
    rng = Xoshiro(seed)
    groups = layout == "reversed" ? [k*b:-1:(k-1)*b+1 for k in 1:n÷b] : group_layout(n,b,layout,rng)
    ks = randperm(rng,n÷b)[1:max(1,n÷b÷10)]
    base = zeros(T,n)
    for k in ks, i in groups[k]; base[i] = T(1/2); end
    x = if ORIGINAL
        storage = copy(base)
        PB.AtomIterate(storage, [view(storage,G) for G in groups])
    else
        PB.BlockIterate(base, groups)
    end
    A = randn(rng,T,16,n); target = randn(rng,T,16)
    steps = T[inv(sum(abs2,A[:,G])) for G in groups]
    r0 = target-A*base; r = copy(r0)
    nv = zeros(T,b); d = zeros(T,b)
    ref = copy(base); rr = copy(r0)
    # Independent dense block algebra, not either implementation's entry loop.
    for _ in 1:4, k in ks
        cols = A[:,groups[k]]; old = ref[groups[k]]
        u = old+steps[k]*(transpose(cols)*rr)
        new = sign.(u).*max.(abs.(u).-steps[k]*lambda,zero(T))
        rr -= cols*(new-old); ref[groups[k]] = new
    end
    reference_equal(rr,target-A*ref)
    active0 = ORIGINAL ? Int[] : copy(x.active)
    position0 = ORIGINAL ? Int[] : copy(x.position)
    count0 = ORIGINAL ? 0 : x.nactive
    prep = () -> begin
        copyto!(x.storage,base)
        if !ORIGINAL
            copyto!(x.active,active0); copyto!(x.position,position0); x.nactive=count0
        end
        copyto!(r,r0)
        nothing
    end
    bcd = () -> residual_bcd!(x,ks,groups,A,r,steps,lambda,nv,d,4)
    bulk = () -> residual_bulk!(x,ks,groups,A,r,steps,lambda,nv,d,4)
    check = () -> begin
        reference_equal(Vector(x.storage),ref); reference_equal(r,rr)
        if !ORIGINAL
            @assert x.active == active0 && x.position == position0 && x.nactive == count0
        end
    end
    (;x,ks,groups,A,r,steps,nv,d,base,prep,bcd,bulk,check)
end
