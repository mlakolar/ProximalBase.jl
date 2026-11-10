# PB-only block math and storage measurements. Compare immutable source checkouts
# with the same numerical fixtures, one thread, --check-bounds=no and fresh source.
# The earlier PB/PS whole-solve program is preserved under docs/audit/block-prox-844/measurements.
using ProximalBase, BenchmarkTools, LinearAlgebra, Statistics
const PB = ProximalBase
Base.JLOptions().check_bounds == 2 || error("run with --check-bounds=no")
BLAS.set_num_threads(1)

function measure(label, op)
    op(); op()
    trial = run(@benchmarkable($op()); evals=200, samples=400, seconds=.15)
    println(label, '\t', median(trial).time, '\t', trial.memory, '\t', trial.allocs)
    flush(stdout)
end
function block_cases(::Type{T}, n, strided) where T
    groups = strided ? [collect(1:2:2n), collect(2:2:2n)] : [collect(1:n), collect(n+1:2n)]
    u = T[sin(i)+2 for i in 1:n]
    for f in (GroupNormL2(T(.3),groups,ones(T,2)), GroupNormLinf(T(.3),groups,ones(T,2)),
              SparseGroupNormL2(T(.1),T(.3),groups,ones(T,2)))
        x, out = BlockIterate(T,2n,groups), similar(u)
        dense = zeros(T,2n); dense[groups[1]] .= u
        expected = prox(f,dense,T(.7))[groups[1]]
        prox_block!(out,f,u,1,T(.7))
        @assert out ≈ expected
        copyto!(blockview(x,1),out)
        @assert collect(blockview(x,1)) == out && nactive(x) == 1
        label = "$T/$(nameof(typeof(f)))/n$n/strided$strided"
        measure("pure_group/$label", () -> (prox_block!(out,f,u,1,T(.7)); nothing))
        measure("public_write/$label", () -> (copyto!(blockview(x,1),out); nothing))
        measure("group_write/$label", () -> (prox_block!(out,f,u,1,T(.7)); copyto!(blockview(x,1),out); nothing))
        # Logical values and IDs make the before/after numerical fixture explicit.
        println(stderr,label,'\t',repr(out),'\t',collect(activecoordinates(x)))
    end
end
function benchmark_round()
    println("case\tmedian_ns\tbytes\tallocations")
    for T in (Float32,Float64), n in (8,32,256), strided in (false,true)
        block_cases(T,n,strided)
    end
end
repeats=parse(Int,get(ENV,"PB_BLOCK_BENCH_REPEATS","1"))
repeats>0 || error("PB_BLOCK_BENCH_REPEATS must be positive")
prefix=get(ENV,"PB_BLOCK_BENCH_OUTPUT","")
for rep in 1:repeats
    if isempty(prefix)
        benchmark_round()
    else
        open("$prefix-$rep.tsv","w") do output
            open("$prefix-$rep.stderr","w") do metadata
                redirect_stdio(;stdout=output,stderr=metadata) do
                    benchmark_round()
                end
            end
        end
        println("Completed PB-only repetition ",rep)
    end
end
