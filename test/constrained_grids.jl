module ConstrainedGridsTest

# The cells of "a separable function on a box" with the box IndBox(l, u) for the six functions after
# NormL1(0.3, w) and ElasticNet(0.2, 0.5), and the Float32 pass of "variable fixing with extreme normals".
# constrained.jl runs the other cells and the Float64 pass. The initial fixtures (x, the weights, the bounds
# and the LinearFunction) are copied from constrained.jl, with the same seed. The random perturbations of the
# optimality checks differ from those of constrained.jl, which skips the cells that run here.

using Test
using ProximalBase
using LinearAlgebra
using Random

using ..TestUtils

Random.seed!(1)

@testset "a separable function on a box" begin
  n = 8
  x = 2 .* randn(n)
  w = rand(n)
  l, u = -rand(n), rand(n)
  fs = (NormL1(0.3, w), ElasticNet(0.2, 0.5), NormL1Nonneg(0.2), SqrNormL2(2.),
        ZeroFunction(), LinearFunction(randn(n)), Scaled(NormL1(1.), 0.4),
        IndBox(-0.3, 0.6))
  # the first two functions are drawn and run in constrained.jl; the six after them run here
  for f in fs[3:end], C in (IndBox(l, u),)
    @testset "$(nameof(typeof(f))) on $(nameof(typeof(C)))" begin
      c = Constrained(f, C)
      check_traits(c; proximable = true, separable = true, convex = true)
      for γ in (0.4, 1.7)
        p = check_prox(c, x, γ)
        @test p ≈ clamp.(prox(f, x, γ), c.box.l, c.box.u)
        @test all(i -> prox_entry(c, x[i], i, γ) ≈ p[i], 1:n)
      end
    end
  end
end

# and one that counts the passes of the search over x, and the entries whose
# slopes it computes (copied with the loop of constrained.jl; the Float32 loop below does not reach it)
const CALLS = Ref(0)
const SLOPES = Ref(0)
struct Counted{F} <: SeparableFunction
  f::F
end
ProximalBase.entry_parameters(f::Counted) = ProximalBase.entry_parameters(f.f)
ProximalBase.prox_entry(g::Counted, v, i, γ) = (CALLS[] += 1; prox_entry(g.f, v, i, γ))
ProximalBase.value_entry(g::Counted, v, i) = value_entry(g.f, v, i)
ProximalBase.isconvex(::Type{<:Counted}) = true
ProximalBase._has_prox_slope(::Type{<:Counted{F}}) where {F} = ProximalBase._has_prox_slope(F)
ProximalBase._prox_entry_slope(g::Counted, v, i, γ) =
  (CALLS[] += 1; SLOPES[] += 1; ProximalBase._prox_entry_slope(g.f, v, i, γ))

# and one that also keeps the ranges of its scalar proximal operators, so that
# the search fixes entries; Counted hides them, and searches as before
struct Ranged{F} <: SeparableFunction
  f::F
end
ProximalBase.entry_parameters(f::Ranged) = ProximalBase.entry_parameters(f.f)
ProximalBase.prox_entry(g::Ranged, v, i, γ) = (CALLS[] += 1; prox_entry(g.f, v, i, γ))
ProximalBase.value_entry(g::Ranged, v, i) = value_entry(g.f, v, i)
ProximalBase.isconvex(::Type{<:Ranged}) = true
ProximalBase._has_prox_slope(::Type{<:Ranged{F}}) where {F} = ProximalBase._has_prox_slope(F)
ProximalBase._prox_entry_slope(g::Ranged, v, i, γ) =
  (CALLS[] += 1; SLOPES[] += 1; ProximalBase._prox_entry_slope(g.f, v, i, γ))
ProximalBase._has_prox_range(::Type{<:Ranged{F}}) where {F} = ProximalBase._has_prox_range(F)
ProximalBase._prox_range(g::Ranged, i::Integer, ::Type{S}) where {S} = ProximalBase._prox_range(g.f, i, S)
ProximalBase._has_uniform_range(::Type{<:Ranged{F}}) where {F} = ProximalBase._has_uniform_range(F)
# the recovery of a Float32 call whose multiplier the Float32 check does not accept is refused for a function without the
# hook of its entry maps (a call of the fixing tests below needs it)
ProximalBase._has_wide(::Type{<:Ranged{F}}) where {F} = ProximalBase._has_wide(F)
ProximalBase._prox_entry_wide(g::Ranged, v::Float64, i::Integer, γ::Float32) =
  (CALLS[] += 1; ProximalBase._prox_entry_wide(g.f, v, i, γ))

@testset "variable fixing with extreme normals" begin
  rng = Xoshiro(140)
  n = 20_000
  # constrained.jl loops over (Float32, Float64); this is the Float32 pass
  for T in (Float32,)
    e = round(Int, 0.6 * exponent(floatmax(T)))
    x = randn(rng, T, n)
    a = randn(rng, T, n)
    a[1:7:end] .= zero(T)
    positive = abs.(a) .+ T(0.5)
    γv = T(0.2) .+ rand(rng, T, n)
    cases = ((IndNonnegative(), one(T), one(T)),
             (IndBox(zero(T), one(T)), one(T), T(10)),
             (Constrained(NormL1(T(0.3)), IndBox(-one(T), one(T))), -one(T), -T(n - 5)),
             (NormL1Nonneg(T(0.1)), positive, T(0.5)),
             (IndBox(zero(T), one(T)), a, sum(min.(a, zero(T))) + one(T)))
    rtol, atol = T === Float64 ? (1e-10, 1e-12) : (1e-4, 1e-6)
    tol = T === Float64 ? 1e-10 : 16Float64(eps(T))
    for (f, normal, b) in cases, F in (IndHyperplane, IndHalfspace), γ in (T(0.7), γv)
      H = F(normal, b)
      r = Constrained(Ranged(f), H)
      CALLS[] = SLOPES[] = 0
      ref = prox(r, x, γ)
      unfixed = T === Float64 ? prox(Constrained(Counted(f), H), x, γ) : nothing
      for c in (ldexp(one(T), -e), ldexp(one(T), e))
        ac, bc = c .* normal, c * b
        @test ac ./ c == normal
        @test bc / c == b
        Hc = F(ac, bc)
        g = Constrained(Ranged(f), Hc)
        CALLS[] = SLOPES[] = 0
        y = prox(g, x, γ)                            # from a fresh multiplier
        @test SLOPES[] == 0
        @test g.live[6].buf[].compact
        @test g.live[6].buf[].n < n
        @test isapprox(y, ref; rtol, atol = atol * max(1, norm(x)))
        T === Float64 && @test isapprox(y, unfixed; rtol = 1e-10)
        m = normal isa Real ? abs(normal) : maximum(abs, normal)
        s = ldexp(1.0, -exponent(m))
        an, bn = Float64.(normal) .* s, Float64(b) * s
        products = an .* Float64.(y)
        gap = sum(products) - bn
        scale = sum(abs, products) + abs(bn)
        @test (F === IndHalfspace ? max(gap, zero(gap)) : abs(gap)) <= tol * scale
      end
    end
  end
end

end
