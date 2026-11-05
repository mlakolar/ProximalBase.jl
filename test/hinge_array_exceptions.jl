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

@testset "the destination is a parameter array" begin
  # a destination that is the weight or label array: every parameter is read before it is written
  S = Float64
  γ = 2.0^600
  w = [2.0^-600, 2.0^600, 2.0^-600]; y = [1.0, 2.0^-600, -1.0]; x = [0.5, -2.0^600, 2.0]
  want = [prox_entry(HingeLoss(copy(y), copy(w)), x[i], i, γ) for i in 1:3]
  fw = HingeLoss(copy(y), w)
  @test PB.prox!(w, fw, x, γ) === w && w == want
  w2 = [2.0^-600, 2.0^600, 2.0^-600]; y2 = [1.0, 2.0^-600, -1.0]
  fy = HingeLoss(y2, copy(w2))
  @test PB.prox!(y2, fy, x, γ) === y2 && y2 == want
  # LogPenalty lla_weights! into its weight array
  lw = [1e300, 1.0]
  f = LogPenalty(1e-200, 1e200, lw)
  @test PB.lla_weights!(lw, f, [0.5, 0.5]) == [1e-100, 0.0]
end

@testset "the destination is the array of steps" begin
  # y === γ for an array step: every step is read before it is written, as the entry loop did
  n = 5
  for S in (Float64, Float32), exc in (1, 3, 5)
    big = S === Float64 ? S(2)^1020 : S(2)^120
    w = ones(S, n); w[exc] = S === Float64 ? S(2)^20 : S(2)^20     # γ w overflows at the exceptional entry
    γ = S[0.5, 1, 2, 3, 0.25]; γ[exc] = big
    f = HingeLoss(ones(S, n), w); x = fill(S(-3), n)
    want = [PB.prox_entry(f, x[i], i, γ[i]) for i in 1:n]
    g = copy(γ); @test PB.prox!(g, f, x, g) === g && g == want
    g = copy(γ); o = similar(g); PB.prox!(o, f, x, γ); @test o == want
    g = copy(γ); v = PB.value_and_prox!(g, f, x, g); @test g == want
    z = copy(x); g = copy(γ); PB.prox!(z, f, z, g); @test z == want        # y === x with an array step
  end
end

@testset "blocks of the in-place and repair loops (exceptional entries at block edges)" begin
  # the loops work on blocks of 128 entries: planted entries at the last and first index of a block, in
  # a short last block, and many of them; in place, out of place and entry by entry agree bit for bit
  rng = MersenneTwister(519)
  for S in (Float32, Float64), n in (127, 128, 129, 300, 1000), labelled in (true, false)
    x = randn(rng, S, n); w = rand(rng, S, n) .+ S(0.2)
    tiny = 3 * nextfloat(zero(S))
    for pos in (Int[], [1], [128], [129], [n], [1, 128, 129, 256, n], collect(1:7:n))
      ww = copy(w); foreach(i -> i <= n && (ww[i] = tiny), pos)
      f = labelled ? HingeLoss(S.(rand(rng, (-1, 1), n)), ww) : HingeLoss(nothing, ww)
      γ = S(0.7)
      want = [PB.prox_entry(f, x[i], i, γ) for i in 1:n]
      y = similar(x); PB.prox!(y, f, x, γ); @test y == want
      z = copy(x); @test PB.prox!(z, f, z, γ) === z; @test z == want
    end
  end
end

@testset "destinations that share memory with a parameter, integer outputs, abstract element types" begin
  # independent review of the block loops: every entry reads its parameters before its slot is written
  n = 6
  f = HingeLoss(fill(3.0, n), fill(1e-300, n)); sc = PB.Scaled(f, 2.0); x = zeros(n); γ = fill(1.2345e-20, n)
  want = PB.prox!(zeros(n), sc, x, copy(γ))
  g = copy(γ); PB.prox!(g, sc, x, g); @test g == want                    # y === γ through a wrapper's step array
  w = [0.0, 1, 1, 1, 1, 1]; fw = HingeLoss(ones(n), w)
  @test PB.prox!(view(w, :), fw, fill(0.5, n), Inf) == [0.5, 1, 1, 1, 1, 1]   # y is a view of the weights
  lab = fill(3.0, n); fl = HingeLoss(lab, fill(1e-300, n))
  @test PB.prox!(view(lab, :), fl, zeros(n), 1e-15) == fill(3.0e-315, n)
  fi = HingeLoss([2, 2, 2], [0.0, 1, 1])                                 # zero weight, infinite step, integer output
  @test PB.prox!(zeros(Int, 3), fi, [0, 5, 7], Inf) == [0, 5, 7]
  ff = HingeLoss(Float16[-6.758], Float16[0.00681]); γ16 = Float16(2.253e-5)   # abstract element type, narrow entries
  @test PB.prox!(Real[0.0], ff, Real[Float16(0)], γ16)[1] == PB.prox_entry(ff, Float16(0), 1, γ16)
  z = Real[Float16(0)]; @test PB.prox!(z, ff, z, γ16)[1] == PB.prox_entry(ff, Float16(0), 1, γ16)
end

@testset "the range test as one unsigned comparison equals floatmin <= ab <= floatmax" begin
  ref(ab::R) where {R} = (floatmin(R) <= ab) & (ab <= floatmax(R))
  @test all(u -> PB._normal_range(reinterpret(Float16, u)) == ref(reinterpret(Float16, u)), typemin(UInt16):typemax(UInt16))
  rng = MersenneTwister(5)
  @test all(_ -> (u = rand(rng, UInt32); PB._normal_range(reinterpret(Float32, u)) == ref(reinterpret(Float32, u))), 1:200_000)
  @test all(_ -> (u = rand(rng, UInt64); PB._normal_range(reinterpret(Float64, u)) == ref(reinterpret(Float64, u))), 1:200_000)
  for T in (Float16, Float32, Float64), v in (zero(T), -zero(T), floatmin(T), prevfloat(floatmin(T)), nextfloat(floatmin(T)),
                                               floatmax(T), prevfloat(floatmax(T)), T(Inf), -T(Inf), T(NaN), -floatmin(T), -floatmax(T))
    @test PB._normal_range(v) == ref(v)
  end
  @test PB._normal_range(big"1.0") && !PB._normal_range(big"0.0")
end

end
