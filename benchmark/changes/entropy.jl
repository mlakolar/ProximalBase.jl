include(joinpath(@__DIR__, "..", "campaign", "run.jl"))

function entropy_cases(rep)
    cases = PerfCase[]
    for T in (Float32,Float64), n in (256,100_000)
        rng = scenario_rng(rep, "entropy_start/$T/$n")
        step, λ = T(0.7), T(0.2)
        μ = step * λ
        q = rand(rng,T,n) .+ T(0.2)
        us = (("normal",nothing), ("middle",4rand(rng,T,n).-2),
              ("outside",vcat(-3 .- 20rand(rng,T,n÷2),3 .+ 30rand(rng,T,n-n÷2))))
        for (name,u) in us
            x = u === nothing ? randn(rng,T,n) : μ .* (u .+ 1 .+ log(μ))
            z = u === nothing ? copy(x) : μ .* (u .- log.(q ./ μ))
            addprox!(cases,"entropy_start","NegEntropy/$name",NegEntropy(λ),x;
                     γ=step,alias=true,profile=T==Float64 && n==100_000)
            addprox!(cases,"entropy_start","KL/$name",KLDivergence(q;λ),z;
                     γ=step,alias=true,profile=T==Float64 && n==100_000)
        end
    end
    cases
end

main(entropy_cases)
