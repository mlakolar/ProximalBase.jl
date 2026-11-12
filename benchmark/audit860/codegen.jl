# Run after timing, under exactly the same compiler flags as the campaign.
using InteractiveUtils, LinearAlgebra
BLAS.set_num_threads(1)
include(joinpath(ARGS[1],"src/ProximalBase.jl"));const PB=Main.ProximalBase
include("kernels.jl")
mkpath(ARGS[2])
function dump(name,f,types)
    for (ext,printer) in (("llvm",code_llvm),("native",code_native))
        open(joinpath(ARGS[2],name*"."*ext),"w") do io
            printer(io,f,types;debuginfo=:none)
        end
    end
    open(joinpath(ARGS[2],name*".typed"),"w") do io
        show(io,MIME("text/plain"),code_typed(f,types))
    end
end
for T in (Float32,Float64),layout in ("contiguous","arbitrary","reversed","strided")
    f=make_fixture(T,8192,64,layout,860)
    for route in (Val(:owner),Val(:raw),Val(:once))
        ORIGINAL && route==Val(:once) && continue
        dump("$T-$layout-$(typeof(route).parameters[1])",diagnostic_sweep!,Tuple{typeof(f.x),Vector{Int},typeof(route)})
    end
    args=Tuple{typeof(f.x),typeof(f.ks),typeof(f.groups),typeof(f.A),typeof(f.r),typeof(f.steps),T,typeof(f.nv),typeof(f.d),Int}
    dump("$T-$layout-bcd",residual_bcd!,args)
    dump("$T-$layout-bulk",residual_bulk!,args)
    dump("$T-$layout-copy",copyto!,Tuple{typeof(block(f.x,1)),Vector{T}})
    dump("$T-$layout-copysweep",copy_blocks!,Tuple{typeof(f.x),typeof(f.ks),Vector{T}})
end
