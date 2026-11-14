# Every example must be self-contained: use a fresh process, not shared definitions.
using LinearAlgebra
BLAS.set_num_threads(1)
for file in sort(filter(endswith(".jl"), readdir(joinpath(@__DIR__, "..", "examples"); join=true)))
    println("Isolated example: ", basename(file))
    flush(stdout)
    run(`$(Base.julia_cmd()) --startup-file=no --check-bounds=yes --project=$(dirname(Base.active_project())) $file`)
end
