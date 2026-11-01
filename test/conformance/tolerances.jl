module ConformanceTolerancesTest

using Test
using ProximalBase
using LinearAlgebra
using Random
using ..Conformance

# A4: direct entries have s=0; computed spectra use their own scale.
for T in (Float32, Float64), bound in (T(0), T(8)), side in (:inside, :outside)
  τ = sqrt(eps(T)) * max(one(T), abs(bound))
  δ = side == :inside ? τ / T(4) : T(4) * τ
  @row begin
    id = "tolerances.membership.$T.$(Int(bound)).$side"; case = ""; family = :tolerances; oracle = :value
    types = "argument=$T bound=$T"; backend = :cpu
    rules = "A4"; problem = "upper-bound residual $(side == :inside ? "1/4" : "4") times sqrt(eps(T))*max(1,abs(bound)); s=0"
    expected = side == :inside ? zero(T) : T(Inf)
    observed = IndBox(T(-1), bound)(T[bound + δ])
  end
end
for T in (Float32, Float64), k in (0, 20), side in (:inside, :outside)
  scale = pow2(T, k)
  δ = (side == :inside ? sqrt(eps(T))/T(4) : T(4)*sqrt(eps(T))) * scale
  @row begin
    id = "tolerances.spectrum.$T.$k.$side"; case = ""; family = :tolerances; oracle = :value
    types = "spectrum=$T"; backend = :cpu
    rules = "A4 A5"; problem = "#307: spectrum=(2^$k,-delta); negativity residual is 1/4 or 4 times sqrt(eps(T))*max(1,s)"
    expected = side == :inside ? zero(T) : T(Inf)
    observed = Spectral(IndNonnegative())(Matrix(Diagonal(T[scale, -δ])))
  end
  @row begin
    id = "tolerances.rank.$T.$k.$side"; case = ""; family = :tolerances; oracle = :value
    types = "matrix=$T"; backend = :cpu
    rules = "A4 A5"; problem = "rank-one ball: sigma2/sigma1 is sqrt(eps(T))/4 or 4sqrt(eps(T)); scale invariant"
    expected = side == :inside ? zero(T) : T(Inf)
    observed = IndBallRank(1)(Matrix(Diagonal(T[scale, δ])))
  end
end
# Decision 16 identifies these fixed defaults. sqrt(eps(T)) is the A4
# precision-aware comparison used here; the exact replacement is a Claims item.
@row begin
  id = "tolerances.largest-eigenvalue-default"; case = ""; family = :tolerances; oracle = :decision
  types = "Float32 operator and Ritz residual"; backend = :cpu
  rules = "A4"; problem = "diag(1,1+2^-15): first Ritz residual below sqrt(eps(Float32)); stop at precision-aware boundary"
  expected = 1
  observed = let calls = Ref(0)
    op! = (y, x) -> (calls[] += 1; y .= Float32[1, 1 + pow2(Float32, -15)] .* x)
    ProximalBase._largest_eigenvalue(op!, 2, Float32)
    calls[]
  end
  broken = (group = :G10, baseline = Returned(v -> v === 2))
end
@row begin
  id = "tolerances.intersection-default"; case = ""; family = :tolerances; oracle = :decision
  types = "Float32 iterate and step-length residual"; backend = :cpu
  rules = "A4"; problem = "projection of 2 onto [0,2-2^-15]: first displacement below sqrt(eps(Float32))*norm(y)"
  expected = 1
  observed = let f = IndIntersection(IndBox(0f0, 2f0 - pow2(Float32, -15)))
    prox(f, Float32[2], 1f0)
    f.iterations
  end
  broken = (group = :G10, baseline = Returned(v -> v === 2))
end
@row begin
  id = "tolerances.kkt-default"; case = ""; family = :tolerances; oracle = :decision
  types = "Float32 level/argument; Float64 generated gradient and violation"; backend = :cpu
  rules = "A4"; problem = "seed 20 gives |violation|=1.2596413523358052e-8, between 1e-8 and sqrt(eps(Float64)); precision-aware band skips check"
  expected = 0
  observed = let
    ts = @testset "KKT boundary probe" begin
      ProximalBase.Testing.test_kkt(NormL1(0.35540783f0), Float32[0], 1f0;
                                  trials = 1, rng = MersenneTwister(20))
    end
    Test.get_test_counts(ts).passes
  end
  broken = (group = :G10, baseline = Returned(v -> v === 1))
end

end
