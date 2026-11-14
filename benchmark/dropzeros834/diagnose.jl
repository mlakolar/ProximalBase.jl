# Optimized LLVM IR of dropzeros! for each iterate type and element type in one source tree, with
# type names and debug information removed, so that two trees (or the two types) can be diffed.
# usage: julia --project=ENV diagnose.jl SOURCE OUTPUT_DIR
using InteractiveUtils
module Src end
Base.include(Src, joinpath(ARGS[1], "src", "ProximalBase.jl"))
const PB = Src.ProximalBase
using SparseArrays: dropzeros!
mkpath(ARGS[2])
for (name, X) in (("sparse", PB.SparseIterate), ("symmetric", PB.SymmetricSparseIterate)), T in (Float32, Float64)
    S = X === PB.SparseIterate ? X{T, 1} : X{T}
    ir = sprint(io -> code_llvm(io, dropzeros!, (S,); debuginfo=:none, raw=false))
    ir = replace(ir, r"SparseIterate|SymmetricSparseIterate" => "ITERATE", r"julia_dropzeros!_\d+" => "julia_dropzeros!",
                 r";.*" => "")
    write(joinpath(ARGS[2], "$name-$T.ll"), ir)
end
