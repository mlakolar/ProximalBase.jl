using BenchmarkTools, LinearAlgebra, Statistics, TOML, SHA, InteractiveUtils
VERSION==v"1.13.3" || error("audit requires Julia 1.13.3")
Sys.islinux() || error("Linux required")
Threads.nthreads()==1 || error("one Julia thread required")
BLAS.set_num_threads(1)
source, output, seedstr = ARGS[1:3]
include(joinpath(source,"src/ProximalBase.jl")); const PB = Main.ProximalBase
include("kernels.jl")
seed = parse(Int,seedstr)
function measure(op,prep)
    for _ in 1:3; prep(); op(); end
    run(@benchmarkable $op() setup=($prep()) evals=1 samples=3 seconds=.001)
    t = run(@benchmarkable $op() setup=($prep()) evals=1 samples=80 seconds=.1)
    Dict("median_ns"=>median(t).time,"bytes"=>t.memory,"allocs"=>t.allocs,
         "times_ns"=>t.times,"gc_ns"=>t.gctimes)
end
function main()
    results = Dict{String,Any}()
    fixtures = Dict{String,Any}()
    specs = [(T,n,b,l) for T in (Float32,Float64), n in (8192,65536),
             b in (1,8,64), l in ("contiguous","strided","arbitrary","reversed")]
    isodd(seed) || reverse!(specs)
    for (index,(T,n,b,l)) in enumerate(specs)
        # Same padding schedule across sources, varied across rounds/fixtures.
        # Hold it throughout timing, as in the #852 placement controls.
        padseed=seed+31index
        padding=[zeros(UInt8,mod(padseed*131+j*4093,65536)+1) for j in 1:mod(padseed,13)]
        GC.@preserve padding begin
            f = make_fixture(T,n,b,l,seed)
            # Build both designs before numerical-kernel warmup. Allocations
            # from warming f's kernels therefore occur after z is allocated.
            z=make_fixture(T,n,b,l,seed;lambda=T(1e6))
            key = "$T/n$n/b$b/$l"
            fixtures[key]=Dict("storage_pointer"=>UInt(pointer(f.x.storage)),
                "design_pointer"=>UInt(pointer(f.A)),"residual_pointer"=>UInt(pointer(f.r)),
                "buffer_pointer"=>UInt(pointer(f.nv)),"zero_design_pointer"=>UInt(pointer(z.A)),
                "zero_storage_pointer"=>UInt(pointer(z.x.storage)),
                "zero_buffer_pointer"=>UInt(pointer(z.nv)),"held_padding_bytes"=>sum(length,padding;init=0))
            f.prep(); f.bcd(); f.check()
            results[key*"/residual_bcd"] = measure(f.bcd,f.prep)
            f.prep(); f.bulk(); f.check()
            results[key*"/residual_bulk"] = measure(f.bulk,f.prep)
            z.prep();z.bcd();z.check()
            @assert all(iszero,z.x.storage)
            results[key*"/residual_bcd_zero"] = measure(z.bcd,z.prep)
            z.prep();z.bulk();z.check()
            @assert all(iszero,z.x.storage)
            results[key*"/residual_bulk_zero"] = measure(z.bulk,z.prep)
            for state in (:active,:activate,:zero_inactive,:zero_active,:active_zero)
                c=copy_fixture(T,n,b,f.groups,f.ks,state)
                results[key*"/copy_"*string(state)]=measure(c.op,c.prep)
            end
            for route in (Val(:owner),Val(:raw),Val(:once))
                ORIGINAL && route == Val(:once) && continue
                op = () -> diagnostic_sweep!(f.x,f.ks,route)
                f.prep();op()
                expected=copy(f.base)
                for _ in 1:4,k in f.ks,i in f.groups[k]
                    expected[i]=max(T(7/8)*expected[i]+T(1/4)-T(1/16),zero(T))
                end
                reference_equal(f.x.storage,expected)
                results[key*"/sweep_"*string(typeof(route).parameters[1])] = measure(op,f.prep)
            end
        end
    end
    meta = Dict("julia"=>string(VERSION),"cpu"=>Sys.CPU_NAME,"threads"=>Threads.nthreads(),
        "blas"=>string(BLAS.get_config()),"blas_threads"=>BLAS.get_num_threads(),
        "bounds"=>Base.JLOptions().check_bounds,"opt"=>Base.JLOptions().opt_level,
        "seed"=>seed,"source"=>source,"binary_sha256"=>bytes2hex(sha256(read(joinpath(Sys.BINDIR,"julia")))),
        "source_hashes"=>Dict(p=>bytes2hex(sha256(read(joinpath(source,"src",p)))) for p in readdir(joinpath(source,"src")) if endswith(p,".jl")))
    open(output,"w") do io; TOML.print(io,Dict("meta"=>meta,"fixtures"=>fixtures,"results"=>results)); end
end
main()
