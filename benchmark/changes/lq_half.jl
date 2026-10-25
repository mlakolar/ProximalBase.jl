include(joinpath(@__DIR__, "..", "campaign", "run.jl"))

function lq_half_cases(rep)
    cases = PerfCase[]
    for T in (Float32,Float64), n in (256,100_000)
        rng = scenario_rng(rep, "lq_half/$T/$n")
        x = randn(rng,T,n)
        w = rand(rng,T,n) .+ T(0.2)
        steps = rand(rng,T,n) .+ T(0.2)
        for q in (T(0.5),T(2)/3,T(0.7))
            f = NormLq(T(0.3),q)
            addprox!(cases,"lq_half","q$q",f,x;alias=true,
                     profile=T==Float64 && n==100_000 && q==T(0.5))
            addprox!(cases,"lq_half","q$q",NormLq(T(0.3),q,w),x;layout="weighted")
            addprox!(cases,"lq_half","q$q",f,x;γ=steps,layout="diagonal_step")
            sparsex=SparseIterate(sprandn(rng,T,n,0.01))
            addprox!(cases,"lq_half","q$q",f,sparsex;layout="sparse_1pct")
        end
    end
    cases
end

main(lq_half_cases)
