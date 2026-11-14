# Compare with identical flags/runtime, each source in its own fresh process:
# julia -O2 --check-bounds=no --project=benchmark/audit852 codegen.jl SOURCE OUT
using InteractiveUtils
include(joinpath(ARGS[1], "src", "ProximalBase.jl"))
mkpath(ARGS[2])
for T in (Float32, Float64), S in (Float32, Float64), (label, gs) in (
    ("contiguous", [1:4, 5:8, 9:12]),
    ("strided", [1:3:12, 2:3:12, 3:3:12]),
    ("arbitrary", [[10,4,7,1], [11,5,8,2], [12,6,9,3]]))
    d=ProximalBase.BlockIterate(T,12,gs)
    s=ProximalBase.BlockIterate(S,12,gs)
    for (name,f) in (("public",copyto!), ("active",ProximalBase._copy_active!))
        stem=joinpath(ARGS[2],"$(T)-$(S)-$(label)-$(name)")
        open(stem*".ll","w") do io
            code_llvm(io,f,Tuple{typeof(d),typeof(s)};debuginfo=:none)
        end
        open(stem*".s","w") do io
            code_native(io,f,Tuple{typeof(d),typeof(s)};debuginfo=:none,syntax=:intel)
        end
    end
end
