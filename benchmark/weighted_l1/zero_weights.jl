# Focused follow-up to PR #131: use the same script/environments on both sides.
using ProximalBase, BenchmarkTools, Random, Statistics, InteractiveUtils, TOML

function measure(io, raw, name, f::F, x::Vector{T}) where {F,T}
    expected = sum(value_entry(f, x[i], i) for i in eachindex(x))
    @assert isapprox(f(x), expected; rtol=T(8)*sqrt(eps(T)), atol=zero(T))
    trial = @benchmark $f($x) samples=1000 seconds=0.1 evals=1
    @assert trial.memory == 0
    println(io, join((name, T, minimum(trial).time, mean(trial).time, trial.memory), '\t'))
    for (i,t) in enumerate(trial.times)
        println(raw, name, '\t', T, '\t', i, '\t', t)
    end
end

function main()
    output = ARGS[1]
    mkpath(output)
    open(joinpath(output,"metrics.tsv"),"w") do io
        open(joinpath(output,"samples.tsv"),"w") do raw
            println(io,"operation\ttype\tmin_ns\tmean_ns\tbytes")
            println(raw,"operation\ttype\tsample\ttime_ns")
            for T in (Float32, Float64)
                rng = Xoshiro(1)
                x = randn(rng,T,100_000)
                w = rand(rng,T,length(x)) .+ T(0.5)
                wz = copy(w); wz[1:10:end] .= zero(T)
                for F in (NormL1, NormL1Nonneg)
                    z = F === NormL1Nonneg ? abs.(x) : x
                    for (name, level, weights) in (("ordinary",T(0.5),w),
                        ("zero_weights",T(0.5),wz), ("zero_level",zero(T),w))
                        f = F(level,weights)
                        measure(io,raw,string(nameof(F),"/",name),f,z)
                    end
                    if get(ENV,"PB_COMPILER","false") == "true"
                        f = F(T(0.5),wz)
                        open(joinpath(output,string(nameof(F),"-",T,".ll")),"w") do ll
                            code_llvm(ll,ProximalBase._value_dense,typeof.((f,z));debuginfo=:none)
                        end
                    end
                end
            end
        end
    end
    open(joinpath(output,"metadata.toml"),"w") do io
        TOML.print(io,Dict("julia"=>string(VERSION),"cpu"=>Sys.CPU_NAME,
            "cpu_model"=>Sys.cpu_info()[1].model,"threads"=>Threads.nthreads(),
            "source"=>pkgdir(ProximalBase),"seed"=>1,"n"=>100_000))
    end
end
main()
