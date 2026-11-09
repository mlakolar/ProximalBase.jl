module MetricGridsTest

# The GroupNormLinf and GroupSCAD cases of the main loop of "per-group steps for regularization". metric.jl
# runs GroupNormL2, SparseGroupNormL2 and GroupMCP in that loop, and all five penalties in the loop over
# reordered groups after it. The fixtures and check_metric_optimality are copied from metric.jl.

using Test
using ProximalBase
using LinearAlgebra
using Random

Random.seed!(1)

# the subgradient inequality in the metric Diagonal(1 ./ γ): p = prox_{γf}(x) when
# f(z) ≥ f(p) + ⟨(x - p) ./ γ, z - p⟩ for every z
function check_metric_optimality(f, x, γ, p; points = 5)
  fp = f(p)
  for _ in 1:points, δ in (1e-3, 1e-1, 1.)
    z = p .+ δ .* randn(size(p)...)
    fz = f(z)
    isfinite(fz) && @test fz >= fp + dot((x .- p) ./ γ, z .- p) - 1e-9 * max(1, abs(fp))
  end
end

@testset "per-group steps for regularization" begin
  groups = [1:2, [3, 5], [4, 6]]
  γ = [0.2, 0.9, 1.4]
  weights = [1.0, 0.0, 0.7]
  penalties = ((G, w) -> GroupNormLinf(0.5, G, w),
               (G, w) -> GroupSCAD(0.5, G, w; a = 3.5))
  for make in penalties, dims in ((6,), (2, 3)), centered in (false, true)
    x = reshape([3.0, -4.0, 2.0, -1.0, 0.5, 2.5], dims)
    c = centered ? reshape([0.5, 1.0, -2.0, 3.0, 1.5, -0.5], dims) : nothing
    r = Regularize(make(groups, weights), 0.7, c)
    reference = similar(x)
    γe = similar(x)
    for k in eachindex(groups)
      G = groups[k]
      shrunk = (x[G] .+ γ[k] * r.ρ .* (c === nothing ? zeros(length(G)) : c[G])) ./ (1 + γ[k] * r.ρ)
      reference[G] = prox(make([1:length(G)], [weights[k]]), shrunk, γ[k] / (1 + γ[k] * r.ρ))
      γe[G] .= γ[k]
    end
    p = prox(r, x, γ)
    @test p ≈ reference
    @test prox(r, x, fill(0.6, 3)) ≈ prox(r, x, 0.6)
    @test prox(r, x, view([0.2, 0.0, 0.9, 0.0, 1.4, 0.0], 1:2:6)) ≈ p
    for input in (x, SparseIterate(x))
      y = similar(input)
      @test prox!(y, r, input, γ) === y
      @test Array(y) ≈ p
      @test value_and_prox!(y, r, input, γ) ≈ r(p)
      @test Array(y) ≈ p
      alias = copy(input)
      @test Array(prox!(alias, r, alias, γ)) ≈ p
      alias = copy(input)
      @test value_and_prox!(alias, r, alias, γ) ≈ r(p)
      @test Array(alias) ≈ p
    end
    y32 = similar(x, Float32)
    @test prox!(y32, r, x, γ) === y32
    @test y32 ≈ p rtol = 1e-5
    @test value_and_prox!(y32, r, x, γ) isa Float32
    @test eltype(prox(r, Float32.(x), γ)) == Float32
    @test prox(r, round.(Int, x), γ) ≈ prox(r, Float64.(round.(Int, x)), γ)
    isconvex(r) && check_metric_optimality(r, x, γe, p)
    if centered
      for op in (prox!, value_and_prox!)
        centre = copy(c)
        rc = Regularize(r.f, r.ρ, centre)
        result = op(centre, rc, x, γ)
        @test centre ≈ p
        op === value_and_prox! && @test result ≈ r(p)
      end
    end
  end
end

end
