# Independent nonuniform fingerprints for every ordered block, outside timing.
using LinearAlgebra,SparseArrays,Random
include(joinpath(ARGS[1],"src","ProximalBase.jl"));const PB=Main.ProximalBase
harness=length(ARGS)>=2 ? ARGS[2] : @__DIR__
include(joinpath(harness,"adapters.jl"));include(joinpath(harness,"cases.jl"))
raw_variant=isfile(joinpath(harness,"raw-block-control.json"))
for T in (Float32,Float64),b in (1,8,64),layout in ("contiguous","strided","arbitrary")
    ORIGINAL && !raw_variant && (T==Float32 || layout=="arbitrary") && continue
    n=8192;groups=group_layout(n,b,layout,Xoshiro(852));x=newblock(T,n,b,groups)
    # Every coordinate has a distinct exactly representable binary fraction.
    ref=T.(1:n)./T(n)
    for i in eachindex(ref);x[i]=ref[i];end
    for (k,G) in enumerate(groups)
        v=block(x,k);@assert vec(collect(v))==ref[G]
        for (j,i) in enumerate(G)
            value=-T(j)/T(b)
            if v isa AbstractMatrix;v[1,j]=value;else;v[j]=value;end
            ref[i]=value;@assert x[i]==value
        end
    end
    @assert [x[i] for i in eachindex(ref)]==ref
    if !ORIGINAL;@assert PB.nactive(x)==length(groups);end
    y=copy(x);@assert [y[i] for i in eachindex(ref)]==ref
    y[1]=T(9);@assert x[1]==ref[1]
    println(T,'/',b,'/',layout," all group orders and write aliases pass")
end
