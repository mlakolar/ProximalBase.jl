module HingeArrayExceptionsTest
using Test, ProximalBase, Random, SparseArrays
using ..TestUtils

# HingeLoss on arrays with one exceptional entry among ordinary ones (issue #519, PR A): the entry whose
# coefficient γwᵢ is not a normal number (zero, infinite, subnormal weight; γwᵢ overflowing or
# underflowing while γwᵢyᵢ² is in range) is planted at the first, middle and last index of otherwise
# ordinary data. The expected values of the planted entry are exact by construction (powers of two),
# independent of the implementation; the ordinary entries are compared to a BigFloat oracle on a
# sample and, all of them, to the scalar prox_entry. The same is required of prox!, in place prox!
# (y === x: no original value may be read after it is overwritten), value_and_prox! and
# prox_coordinate!.

const PB = ProximalBase

# the oracle: s = γwy² exactly, u = y v rounded to S as the model forms the margin, the three branches
# decided with a band of 8 ulp around u = 1 - s (inside it the model's 2-ulp step decides; nothing
# is claimed), the value of the branch rounded from the exact quantities
function oracle(::Type{S}, v, y, w, γ) where {S}
  setprecision(BigFloat, 512) do
    s = big(γ) * big(w) * big(y)^2
    u = S(big(y) * big(v))
    bu = big(u)
    band = 8 * big(eps(S)) * max(abs(bu), min(s, big(1)), big(1))
    isinf(s) || s > big(floatmax(S)) * 4 ? (u >= 1 ? S(v) : S(big(1) / big(y))) :
      abs(bu - (1 - s)) <= band && bu < 1 ? nothing :
      bu >= 1 ? S(bu / big(y)) :
      bu < 1 - s ? (S((bu + s) / big(y)), 8 * big(eps(S)) * max(abs(bu), s, big(1)) / abs(big(y))) :
      S(big(1) / big(y))
  end
end
function oracle_ok(::Type{S}, got, want) where {S}
  want === nothing && return true
  want isa Tuple ? abs(big(got) - big(want[1])) <= want[2] : got == want
end

# the ordinary entries: weights near wscale, labels ±(1 or a nonunit finite value)
function ordinary(rng, ::Type{S}, n, γ, wscale) where {S}
  x = randn(rng, S, n)
  y = S[(rand(rng, Bool) ? 1 : -1) * (rand(rng) < 0.5 ? S(1) : S(0.5) + S(1.5) * rand(rng, S)) for _ in 1:n]
  w = S[wscale * (S(0.5) + rand(rng, S)) for _ in 1:n]
  x, y, w
end

# name => (γ, ordinary weight scale, planted (x, y, w), expected result or :entry for the oracle)
function scenarios(::Type{S}) where {S}
  P = S === Float64 ? 600 : 70
  q(k) = S(2)^k
  tiny = nextfloat(zero(S))
  [("zero weight", S(7) / 10, S(1), (S(3), S(-2), S(0)), S(3)),
   ("zero weight, margin ≥ 1", S(7) / 10, S(1), (S(-3), S(-2), S(0)), S(-3)),
   ("infinite weight, u < 1", S(7) / 10, S(1), (S(3), S(-2), S(Inf)), S(-1) / 2),
   ("infinite weight, u ≥ 1", S(7) / 10, S(1), (S(-3), S(-2), S(Inf)), S(-3)),
   ("subnormal weight", S(7) / 10, S(1), (S(3), S(-2), 3 * tiny), S(3)),
   ("overflowing γw, step 1: u+s = 0", q(P), q(-P), (-q(P), q(-P), q(P)), S(0)),
   ("overflowing γw, step 1: tie u = 1-s", q(P), q(-P), (S(0), q(-P), q(P)), q(P)),
   ("overflowing γw, step 1: u ≥ 1", q(P), q(-P), (4q(P), q(-P), q(P)), 4q(P)),
   ("overflowing γw, negative label", q(P), q(-P), (q(P), -q(-P), q(P)), S(0)),
   ("underflowing γw, step 1: u+s = 0", q(-P), q(P), (-q(-P), q(P), q(-P)), S(0)),
   ("underflowing γw, step 1: tie u = 1-s", q(-P), q(P), (S(0), q(P), q(-P)), q(-P)),
   ("underflowing γw, negative label", q(-P), q(P), (q(-P), -q(P), q(-P)), S(0))]
end

positions(n) = n == 1 ? (1,) : (1, (n + 1) ÷ 2, n)

@testset "one exceptional entry among ordinary entries: $S" for S in (Float32, Float64)
  rng = MersenneTwister(5551)
  for (name, γ, wscale, (xe, ye, we), expected) in scenarios(S), n in (1, 256, 100_000), k in positions(n)
    x0, y0, w0 = ordinary(rng, S, n, γ, wscale)
    x0[k], y0[k], w0[k] = xe, ye, we
    f = HingeLoss(y0, w0)
    want = S[prox_entry(f, x0[i], i, γ) for i in 1:n]
    # the planted entry is exactly the expected value, whatever the implementation of the loop
    bad = Any[]
    want[k] == expected || push!(bad, "scalar prox_entry of the planted entry: $(want[k]) ≠ $expected")
    # the scalar path against the oracle on the planted entry and a sample of the ordinary entries
    for i in unique(vcat(k, rand(rng, 1:n, 40)))
      oracle_ok(S, want[i], oracle(S, x0[i], y0[i], w0[i], γ)) || push!(bad, "oracle at $i: $(want[i])")
    end
    # prox!, out of place and in place, and value_and_prox!
    out = fill!(similar(x0), S(NaN))
    PB.prox!(out, f, x0, γ)
    z = copy(x0)
    PB.prox!(z, f, z, γ)
    vout = fill!(similar(x0), S(NaN))
    val = PB.value_and_prox!(vout, f, x0, γ)
    vz = copy(x0)
    PB.value_and_prox!(vz, f, vz, γ)
    for (nm, got) in (("prox!", out), ("prox! in place", z), ("value_and_prox!", vout), ("value_and_prox! in place", vz))
      isequal(got, want) || push!(bad, "$nm differs from prox_entry at $(findfirst(.!isequal.(got, want)))")
    end
    isempty(bad) || @info "hinge exception" name n k S bad
    @test isempty(bad)
    x0 == x0 && @test x0[k] == xe                       # the argument is not modified
    # a coordinate update
    if n <= 256
      xi = SparseIterate(sparsevec(copy(x0)))
      @test PB.prox_coordinate!(xi, f, k, γ) == expected
    end
  end
end

@testset "several exceptional entries: $S" for S in (Float32, Float64)
  rng = MersenneTwister(5552)
  for (name, γ, wscale, (xe, ye, we), expected) in scenarios(S)
    n = 1000
    x0, y0, w0 = ordinary(rng, S, n, γ, wscale)
    ks = sort(unique(rand(rng, 1:n, 120)))
    x0[ks] .= xe; y0[ks] .= ye; w0[ks] .= we
    f = HingeLoss(y0, w0)
    want = S[prox_entry(f, x0[i], i, γ) for i in 1:n]
    @test all(want[ks] .== expected)
    out = PB.prox!(similar(x0), f, x0, γ)
    z = copy(x0); PB.prox!(z, f, z, γ)
    @test isequal(out, want) && isequal(z, want)
  end
end

@testset "the review example, from arrays" begin
  γ = 2.0^600
  f = HingeLoss([2.0^-600], [2.0^600])
  @test PB.prox_entry(f, -2.0^600, 1, γ) === 0.0      # γ w label² = 1; the margin is -1: u + s = 0
  y = [0.0]; PB.prox!(y, f, [-2.0^600], γ); @test y == [0.0]
  z = [-2.0^600]; PB.prox!(z, f, z, γ); @test z == [0.0]
  @test PB.kkt_violation(f, [2.0^-1], 1) isa Float64
end

end
