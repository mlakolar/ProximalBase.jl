# Fixed-seed probes for issue #129. Run this same file against every revision.
# The ablations below are diagnostics for ordinary finite inputs, not alternative
# implementations: in particular the optimistic store is unsafe in-place when
# an exceptional coefficient occurs.
using ProximalBase, BenchmarkTools, Random, Statistics, InteractiveUtils, TOML, SHA

function plain_prox!(y, f, x, γ)
    a = γ * f.λ0
    @inbounds for i in eachindex(y, x)
        y[i] = ProximalBase.shrink(x[i], a * f.w[i])
    end
    y
end

function checked_prox!(y, f, x, γ)
    a = γ * f.λ0
    good = true
    @inbounds @simd for i in eachindex(y, x)
        c = a * f.w[i]
        good &= (floatmin(eltype(x)) <= a <= floatmax(eltype(x))) &
                (floatmin(eltype(x)) <= c <= floatmax(eltype(x)))
        y[i] = ProximalBase.shrink(x[i], c)
    end
    good
end

function prepass_prox!(y, f, x, γ)
    a = γ * f.λ0
    good = true
    @inbounds @simd for i in eachindex(x)
        c = a * f.w[i]
        good &= (floatmin(eltype(x)) <= a <= floatmax(eltype(x))) &
                (floatmin(eltype(x)) <= c <= floatmax(eltype(x)))
    end
    good || error("ordinary-input diagnostic")
    plain_prox!(y, f, x, γ)
end

# Same arithmetic as the old dense loop, even on the repaired revision.
function entry_prox!(y, f, x, γ)
    @inbounds for i in eachindex(y, x)
        y[i] = prox_entry(f, x[i], i, γ)
    end
    y
end

function entry_value(f, x)
    s = zero(eltype(x))
    @inbounds @simd for i in eachindex(x)
        s += value_entry(f, x[i], i)
    end
    s
end

function checked_value(f, x)
    s = zero(eltype(x))
    good = true
    @inbounds @simd for i in eachindex(x)
        a = f.λ0 * f.w[i]
        s += a * abs(x[i])
        good &= floatmin(eltype(x)) <= a <= floatmax(eltype(x))
    end
    good ? s : entry_value(f, x)
end

function counted_value(f, x)
    s = zero(eltype(x))
    bad = 0
    @inbounds @simd for i in eachindex(x)
        a = f.λ0 * f.w[i]
        s += a * abs(x[i])
        bad += !(floatmin(eltype(x)) <= a <= floatmax(eltype(x)))
    end
    bad == 0 ? s : entry_value(f, x)
end

function coordinate_sweep(f, x)
    s = zero(eltype(x))
    @inbounds for i in eachindex(x)
        s += prox_entry(f, x[i], i, 1.0)
    end
    s
end

function measure(io, raw, name, op::F) where {F}
    first = @timed op()
    trial = @benchmark $op() samples=1000 seconds=0.2 evals=1
    println(io, join((name, minimum(trial).time, mean(trial).time, trial.memory,
                      first.time, first.compile_time), '\t'))
    for (i, t) in enumerate(trial.times)
        println(raw, name, '\t', i, '\t', t)
    end
end

function main()
    output = ARGS[1]
    mkpath(output)
    rng = Xoshiro(1)
    x = randn(rng, 100_000)
    w = rand(rng, length(x)) .+ 0.5
    y = similar(x)
    positive = abs.(x)
    f, g = NormL1(0.5, w), NormL1Nonneg(0.5, w)
    ops = (("weighted_prox", () -> prox!(y, f, x, 1.0)),
           ("weighted_nonneg_prox", () -> prox!(y, g, x, 1.0)),
           ("weighted_value", () -> f(x)),
           ("weighted_nonneg_value", () -> g(positive)),
           ("unweighted_prox", () -> prox!(y, NormL1(0.5), x, 1.0)),
           ("coordinate_sweep", () -> coordinate_sweep(f, x)),
           ("entry_prox", () -> entry_prox!(y, f, x, 1.0)),
           ("entry_value", () -> entry_value(f, x)),
           ("plain_prox", () -> plain_prox!(y, f, x, 1.0)),
           ("checked_prox", () -> checked_prox!(y, f, x, 1.0)),
           ("prepass_prox", () -> prepass_prox!(y, f, x, 1.0)),
           ("checked_value", () -> checked_value(f, x)),
           ("counted_value", () -> counted_value(f, x)))
    open(joinpath(output, "metrics.tsv"), "w") do io
        open(joinpath(output, "samples.tsv"), "w") do raw
            println(io, "operation\tmin_ns\tmean_ns\tbytes\tfirst_s\tcompile_s")
            println(raw, "operation\tsample\ttime_ns")
            for (name, op) in ops
                measure(io, raw, name, op)
            end
        end
    end
    root = pkgdir(ProximalBase)
    open(joinpath(output, "metadata.toml"), "w") do io
        TOML.print(io, Dict("julia"=>string(VERSION), "cpu"=>Sys.CPU_NAME,
            "cpu_model"=>Sys.cpu_info()[1].model, "threads"=>Threads.nthreads(),
            "root"=>root, "sha"=>readchomp(`git -C $root rev-parse HEAD`),
            "probe_sha256"=>bytes2hex(sha256(read(@__FILE__))),
            "seed"=>1, "n"=>length(x), "benchmarktools"=>string(pkgversion(BenchmarkTools))))
    end
    if get(ENV, "PB_COMPILER", "false") == "true"
        for (name, fn, args) in (("package_prox", ProximalBase._prox_sep_dense!, (y,f,x,1.0)),
                                ("package_value", ProximalBase._value_dense, (f,x)),
                                ("entry_prox", entry_prox!, (y,f,x,1.0)),
                                ("entry_value", entry_value, (f,x)),
                                ("checked_prox", checked_prox!, (y,f,x,1.0)),
                                ("prepass_prox", prepass_prox!, (y,f,x,1.0)),
                                ("checked_value", checked_value, (f,x)),
                                ("counted_value", counted_value, (f,x)))
            open(joinpath(output, name * ".ll"), "w") do io
                code_llvm(io, fn, typeof.(args); debuginfo=:none)
            end
            open(joinpath(output, name * ".native"), "w") do io
                code_native(io, fn, typeof.(args); debuginfo=:none)
            end
            open(joinpath(output, name * ".types"), "w") do io
                code_warntype(io, fn, typeof.(args); debuginfo=:none)
            end
        end
        if isdefined(ProximalBase, :_positive_product)
            open(joinpath(output, "product.ll"), "w") do io
                code_llvm(io, ProximalBase._positive_product, (Float64,Float64,Float64); debuginfo=:none)
            end
        end
    end
end
main()
