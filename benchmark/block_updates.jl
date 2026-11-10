# PB-only diagnostics for #787/#844. The same driver runs in immutable baseline
# and candidate environments; no ProximalSolvers dependency or unchecked public API.
# --server warms once, then accepts one output filename per line on stdin.
# Every sample has one evaluation: setup is outside timing, including restoration
# of in-place inputs and initially inactive blocks. Warm-up never enters results.
using ProximalBase, BenchmarkTools, LinearAlgebra, Statistics, SHA, Random
const PB = ProximalBase
Base.JLOptions().check_bounds == 2 || error("run with --check-bounds=no")
BLAS.set_num_threads(1)

struct BlockProbe{F,S,R}
    id::String
    operation::F
    prepare::S
    result::R
end
const probes = Any[]
noop() = nothing
addprobe(id, op, setup, result) = push!(probes, BlockProbe(id, op, setup, result))

function math_probes(::Type{T}, n, f, wrapper, buffer, family) where T
    fixture = family in ("GroupMCP","GroupSCAD") ? T[(.4+.2sin(i))/sqrt(n) for i in 1:n] : T[sin(i)+2 for i in 1:n]
    if buffer === :vector || buffer === :inplace
        u = copy(fixture)
        out = buffer === :inplace ? u : similar(u)
    else
        u = view(zeros(T,n+2),2:n+1)
        out = buffer === :view_inplace ? u : view(zeros(T,n+2),2:n+1)
    end
    math_buffer_probes(fixture,f,u,out)
end

# A function barrier keeps vector/view union choices out of the timed closures.
function math_buffer_probes(fixture::Vector{T},f,u::U,out::O) where {T,U,O}
    setup = () -> (copyto!(u,fixture); nothing)
    setup()
    gamma = T(.7)
    public = () -> (prox_block!(out,f,u,1,gamma); nothing)
    kernel = () -> (PB._prox_block!(out,f,u,1,gamma); nothing)
    check = () -> begin
        out ≈ prox_block(f,fixture,1,gamma) || error("wrong mathematical result")
        out
    end
    public, kernel, setup, check
end

function make_math_probes(::Type{T}, n, leaf, family) where T
    center = T[cos(i) for i in 1:2n]
    wrappers = (("plain",leaf), ("scaled",Scaled(leaf,T(1.3))),
        ("regularize",Regularize(leaf,T(.2),center)),
        ("nested_scaled",Scaled(Scaled(leaf,T(1.2)),T(.8))),
        ("nested_mixed",Regularize(Scaled(Regularize(leaf,T(.1),center),T(1.2)),T(.2),center)))
    for (wrapper,f) in wrappers, buffer in (:vector,:inplace,:view,:view_inplace)
        public,kernel,setup,check = math_probes(T,n,f,wrapper,buffer,family)
        id = "$T/$family/n$n/$wrapper/$buffer"
        addprobe("math/public/$id",public,setup,check)
        addprobe("math/kernel/$id",kernel,setup,check)
        # Exact in-place input is restored outside timing even for check-only probes.
        u = buffer in (:vector,:inplace) ? ones(T,n) : view(ones(T,n+2),2:n+1)
        y = buffer in (:inplace,:view_inplace) ? u : buffer === :view ? view(zeros(T,n+2),2:n+1) : similar(u)
        addprobe("math/preflight/$id",()->PB._check_block_buffers(y,f,u,1),noop,()->T[])
    end
end

function writer_probes(::Type{T}, n, layout, state, source) where T
    groups = layout === :range ? [1:n,n+1:2n] :
        layout === :strided ? [collect(1:2:2n),collect(2:2:2n)] :
        [collect(n:-1:1),collect(2n:-1:n+1)]
    x = BlockIterate(T,2n,groups)
    vals = state in (:active_zero,:inactive_zero) ? zeros(T,n) : T[sin(i)+2 for i in 1:n]
    src = source === :vector ? vals : source === :view ? view(vcat(vals,vals),2:2:2n) :
        source === :conversion ? (T === Float32 ? Float64.(vals) : Float32.(vals)) : Any[vals...]
    inactive = state in (:inactive_nonzero,:inactive_zero)
    function setup()
        fill!(x,zero(T))
        activate!(x,2)
        inactive || activate!(x,1)
        nothing
    end
    setup()
    expected = T.(src)
    op = () -> (copyto!(blockview(x,1),src); nothing)
    check = () -> begin
        collect(blockview(x,1)) == expected || error("wrong stored values")
        ids = collect(activecoordinates(x))
        ids == (inactive && all(iszero,expected) ? [2] : [2,1]) || error("wrong active order")
        vcat(Vector(x),T.(ids))
    end
    id = "$T/n$n/$layout/$state/$source"
    addprobe("writer/$id",op,setup,check)
    if state === :active_nonzero && source === :vector
        # A separately constructed equal index vector deliberately misses identity
        # recognition; arbitrary/prefix views must retain generic value-write rules.
        v = view(x,collect(groups[1]))
        addprobe("arbitrary_view/$T/n$n/$layout",()->(copyto!(v,src);nothing),setup,check)
    end
end

function combined_probe(::Type{T},n,strided) where T
    gs = strided ? [collect(1:2:2n),collect(2:2:2n)] : [collect(1:n),collect(n+1:2n)]
    u = T[sin(i)+2 for i in 1:n]
    for f in (GroupNormL2(T(.3),gs,ones(T,2)),GroupNormLinf(T(.3),gs,ones(T,2)),
              SparseGroupNormL2(T(.1),T(.3),gs,ones(T,2)))
        x,out = BlockIterate(T,2n,gs),similar(u)
        prox_block!(out,f,u,1,T(.7));copyto!(blockview(x,1),out)
        tag = "$T/$(nameof(typeof(f)))/n$n/strided$strided"
        addprobe("pure_group/$tag",()->(prox_block!(out,f,u,1,T(.7));nothing),noop,()->out)
        addprobe("public_write/$tag",()->(copyto!(blockview(x,1),out);nothing),noop,()->Vector(x))
        addprobe("group_write/$tag",()->(prox_block!(out,f,u,1,T(.7));copyto!(blockview(x,1),out);nothing),noop,()->Vector(x))
    end
end

function sweep!(x,f,scratch,gradient,kernel::F) where F
    for k in 1:24
        G = blockindices(x,k)
        @inbounds for j in eachindex(scratch)
            scratch[j] = x[G[j]] - gradient[G[j]]
        end
        kernel(scratch,f,scratch,k,eltype(x)(.7))
        @inbounds for j in eachindex(scratch)
            x[G[j]] = scratch[j]
        end
    end
    nothing
end

# Self-contained fixed-pass quadratic CD diagnostic. Only PB kernels/public
# scalar writes are used; this is not a PS implementation or a convergence test.
function quadratic_sweeps!(x,A,g,curvature,scratch,gradient,kernel::F) where F
    for _ in 1:10, k in 1:24
        G=blockindices(x,k)
        L=curvature[k]
        @inbounds for (i,j) in enumerate(G)
            scratch[i]=x[j]-gradient[j]/L
        end
        kernel(scratch,g,scratch,k,inv(L))
        @inbounds for (i,j) in enumerate(G)
            step=scratch[i]-x[j]
            iszero(step) && continue
            x[j]=scratch[i]
            @simd for row in eachindex(gradient)
                gradient[row] += step*A[row,j]
            end
        end
    end
    nothing
end
function quadratic_probes(::Type{T}) where T
    rng=Xoshiro(844);n,p=160,120
    X=randn(rng,T,n,p);beta=zeros(T,p);beta[1:10].=T(2)
    response=X*beta+T(.1)*randn(rng,T,n)
    A=X'X/T(n);b=-X'response/T(n)
    groups=[i:i+4 for i in 1:5:p]
    curvature=T[opnorm(A[G,G]) for G in groups]
    for (label,kernel) in (("public",prox_block!),("kernel",PB._prox_block!))
        x=BlockIterate(T,p,groups);g=GroupNormL2(T(.1),groups)
        scratch=view(zeros(T,p),1:5);gradient=copy(b)
        setup=()->(fill!(x,zero(T));copyto!(gradient,b);nothing)
        op=()->quadratic_sweeps!(x,A,g,curvature,scratch,gradient,kernel)
        check=()->begin
            values=Vector(x)
            isapprox(gradient,A*values+b;atol=100eps(T),rtol=sqrt(eps(T))) || error("quadratic gradient cache differs")
            vcat(values,gradient,T[dot(values,A*values)/2+dot(b,values)+g(values)])
        end
        addprobe("quadratic_fixed10/$label/$T/n5/groups24",op,setup,check)
    end
end
function control_probes(::Type{T}) where T
    for n in (1,4,5,8,32,256), strided in (false,true)
        gs = strided ? [collect(1:2:2n),collect(2:2:2n)] : [1:n,n+1:2n]
        x = BlockIterate(T[sin(i)+2 for i in 1:2n],gs)
        v = blockview(x,1); raw = PB._block_storage(x,1); z = ones(T,n)
        addprobe("norm/public/$T/n$n/strided$strided",()->norm(v),noop,()->[norm(v)])
        addprobe("norm/storage/$T/n$n/strided$strided",()->norm(raw),noop,()->[norm(raw)])
        addprobe("dot/public/$T/n$n/strided$strided",()->dot(v,z),noop,()->[dot(v,z)])
    end
    s = SparseIterate(ones(T,32))
    b = BlockIterate(ones(T,32),[1:32])
    addprobe("scalar/SparseIterate/$T",()->(s[3]=T(2);nothing),noop,()->Vector(s))
    addprobe("scalar/BlockIterate/$T",()->(b[3]=T(2);nothing),noop,()->Vector(b))
    for (label,kernel) in (("public",prox_block!),("kernel",PB._prox_block!))
        x = BlockIterate(ones(T,120),[i:i+4 for i in 1:5:120])
        f = GroupNormL2(T(.3),[i:i+4 for i in 1:5:120],ones(T,24))
        scratch = view(zeros(T,120),1:5)
        gradient = T[sin(i)/10 for i in 1:120]
        setup = () -> (fill!(x,one(T));nothing)
        addprobe("quadratic_like/$label/$T/n5/groups24",()->sweep!(x,f,scratch,gradient,kernel),setup,()->Vector(x))
    end
end

for T in (Float32,Float64), n in (1,5,8,32,256)
    gs = [1:n,n+1:2n]
    for f in (GroupNormL2(T(.3),gs,ones(T,2)),GroupNormLinf(T(.3),gs,ones(T,2)),
              SparseGroupNormL2(T(.1),T(.3),gs,ones(T,2)),GroupMCP(T(.3),gs,ones(T,2)),
              GroupSCAD(T(.3),gs,ones(T,2)))
        make_math_probes(T,n,f,string(nameof(typeof(f))))
    end
    for layout in (:range,:strided,:reversed), state in (:active_nonzero,:active_zero,:inactive_nonzero,:inactive_zero),
        source in (:vector,:view,:conversion,:checked)
        writer_probes(T,n,layout,state,source)
    end
    for strided in (false,true)
        combined_probe(T,n,strided)
    end
end
for T in (Float32,Float64)
    control_probes(T)
    quadratic_probes(T)
end
case_filter = get(ENV,"PB_BLOCK_FILTER","")
isempty(case_filter) || filter!(p->occursin(Regex(case_filter),p.id),probes)
isempty(probes) && error("empty probe selection")
length(unique(p.id for p in probes)) == length(probes) || error("duplicate case ids")

function warm!(p::BlockProbe)
    for _ in 1:3
        p.prepare();p.operation()
    end
    result = p.result()
    string(p.id,'\t',bytes2hex(sha256(reinterpret(UInt8,collect(result)))))
end
function measure(io,p::BlockProbe; samples,seconds)
    op,setup = p.operation,p.prepare
    bench = @benchmarkable $op() setup=($setup())
    trial = run(bench;evals=1,samples,seconds,gctrial=false,gcsample=false)
    t = trial.times
    println(io,p.id,'\t',median(t),'\t',minimum(t),'\t',quantile(t,.25),'\t',quantile(t,.75),
        '\t',trial.memory,'\t',trial.allocs,'\t',length(t))
end
function measure_all(path)
    samples = parse(Int,get(ENV,"PB_BLOCK_SAMPLES","500"))
    seconds = parse(Float64,get(ENV,"PB_BLOCK_SECONDS",".02"))
    GC.gc()
    open(path,"w") do io
        println(io,"case\tmedian_ns\tminimum_ns\tp25_ns\tp75_ns\tbytes\tallocations\tsamples")
        for p in probes
            measure(io,p;samples,seconds)
        end
    end
    map(warm!,probes) == metadata || error("benchmark results changed after measurement")
end
println(stderr,"Julia=",VERSION," PB=",pathof(PB)," threads=",Threads.nthreads()," BLAS=",BLAS.get_num_threads()," cases=",length(probes))
if get(ENV,"PB_BLOCK_INSPECT","")=="true"
    for p in probes
        println(stderr,p.id," operation=",typeof(p.operation)," fields=",fieldtypes(typeof(p.operation)))
    end
end
metadata = map(warm!,probes)
metadata_path = get(ENV,"PB_BLOCK_METADATA","")
isempty(metadata_path) || write(metadata_path,join(metadata,'\n')*"\n")
values_path = get(ENV,"PB_BLOCK_VALUES","")
if !isempty(values_path)
    open(values_path,"w") do io
        for p in probes
            p.prepare();p.operation()
            values = collect(p.result())
            println(io,p.id,'\t',eltype(values),'\t',bytes2hex(reinterpret(UInt8,values)))
        end
    end
end
if "--server" in ARGS
    println("READY ",length(probes));flush(stdout)
    for path in eachline(stdin)
        path == "STOP" && break
        measure_all(path)
        println("DONE ",path);flush(stdout)
    end
else
    length(ARGS)==1 || error("supply an output path or --server")
    measure_all(only(ARGS))
end
