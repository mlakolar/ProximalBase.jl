module ConformanceTolerancesTest

using Test
using ProximalBase
using LinearAlgebra
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
# A4 does not select an algorithmic default. Both eps(T) and sqrt(eps(T))
# are precision-aware but give different counts on the former fixtures.
# Finite observations cannot distinguish every such rule from a fixed default.
@notcovered :tolerances :decision "Float32/Float64 Ritz residual" :cpu "tolerances.largest-eigenvalue-default: A4 algorithmic tolerance; no prescribed default or stopping count, and no finite observation separates a fixed default from every precision-aware rule."
@notcovered :tolerances :decision "Float32/Float64 step-length residual" :cpu "tolerances.intersection-default: A4 algorithmic tolerance; no prescribed default or round count, and no finite observation separates a fixed default from every precision-aware rule."
@notcovered :tolerances :decision "Float32 argument; Float64 generated violation" :cpu "tolerances.kkt-default: A4 algorithmic tolerance; no prescribed default or check count, and no finite observation separates a fixed default from every precision-aware rule."

end
