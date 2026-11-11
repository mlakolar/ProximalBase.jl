module CheckArrayExceptionsTest
using Test, ProximalBase, Random
using ..TestUtils

# The array loop of CheckLoss (issue #519, PR A): the ordinary arithmetic of every entry is a branch-free
# expression over the thresholds γτwᵢ and γ(1-τ)wᵢ, formed in the promotion of the model precision and
# the factors and rounded once, so that the loop vectorizes; blocks of 128 entries whose step-and-level
# parts γτ and γ(1-τ) are not normal numbers (zero, infinite, NaN, overflowing or underflowing steps)
# are recovered entry by entry through prox_block, and every other block stays on the ordinary loop.
# The array result must equal the scalar prox_block, bit for bit and NaN for NaN, whatever the shape of
# the data: ordinary, with exceptional weights or steps at the edges of a block, NaN entries, in place
# (y === x), into an integer destination, with a step array, and in the three float types.

const PB = ProximalBase

entrywise(f, x, γ) = [PB.prox_block(f, x[i], i, γ isa AbstractArray ? γ[i] : γ) for i in eachindex(x)]
function same(a, b)
  length(a) == length(b) && all(i -> isequal(a[i], b[i]), eachindex(a))
end

# the indices around the edges of the blocks of 128 and the ends of the array
edges(n) = unique(filter(k -> 1 <= k <= n, (1, 2, 127, 128, 129, 130, 255, 256, 257, n - 1, n)))

@testset "ordinary data is the scalar prox_block and the thresholds are rounded once: $S" for S in (Float16, Float32, Float64)
  rng = MersenneTwister(5191)
  for n in (1, 7, 128, 129, 1000), τ in (S(0.3), S(0.5), 0.3), γ in (S(0.7), S(2), 0.7)
    x = randn(rng, S, n)
    w = rand(rng, S, n) .+ S(0.2)
    for f in (CheckLoss(τ, w), CheckLoss(τ))
      want = entrywise(f, x, γ)
      @test same(PB.prox!(similar(x), f, x, γ), want)
      z = copy(x); @test PB.prox!(z, f, z, γ) === z; @test same(z, want)
      @test same(PB.prox(f, x, γ), want)
    end
  end
  # against an oracle formed from the exact factors, for a Float64 step and level with data of type S
  if S !== Float64
    n = 2000
    x = randn(rng, S, n) .* 3
    w = rand(rng, n) .+ 0.2
    τ, γ = 0.3, 0.7
    f = CheckLoss(τ, w)
    got = PB.prox!(similar(x), f, x, γ)
    bad = 0
    setprecision(BigFloat, 256) do
      for i in 1:n
        hi, lo = S(big(γ) * big(τ) * big(w[i])), S(big(γ) * (1 - big(τ)) * big(w[i]))
        u = x[i]
        want = u > hi ? u - hi : (u < -lo ? u + lo : zero(S))
        abs(got[i] - want) <= 2 * eps(S) * (abs(u) + hi + lo) || (bad += 1)
      end
    end
    @test bad == 0
  end
end

# name => (step, weight) of the planted entry, and the value that the entry must have for the datum 3
function planted(::Type{S}) where {S}
  [("zero weight (free)", S(0.7), S(0), :x),
   ("infinite weight (fixed at 0)", S(0.7), S(Inf), S(0)),
   ("subnormal weight", S(0.7), nextfloat(zero(S)) * 3, :entry),
   ("huge weight", S(0.7), floatmax(S), :entry)]
end

@testset "exceptional weights at the edges of the blocks: $S" for S in (Float16, Float32, Float64)
  rng = MersenneTwister(5192)
  γ = S(0.7)
  for n in (129, 300, 1000), (name, _, we, expected) in planted(S), k in edges(n), τ in (S(0.3), 0.3)
    x = randn(rng, S, n); w = rand(rng, S, n) .+ S(0.2)
    x[k] = S(3); w[k] = we
    f = CheckLoss(τ, w)
    want = entrywise(f, x, γ)
    expected === :x && @test want[k] == S(3)
    expected isa Number && @test want[k] == expected
    out = fill!(similar(x), S(NaN))
    @test same(PB.prox!(out, f, x, γ), want)
    z = copy(x); PB.prox!(z, f, z, γ)
    @test same(z, want)
    @test x[k] == S(3)                                   # the argument is not modified
  end
end

@testset "NaN entries: a NaN stays NaN unless the map is constant: $S" for S in (Float16, Float32, Float64)
  rng = MersenneTwister(5193)
  for n in (129, 300), k in edges(n)
    x = randn(rng, S, n); w = rand(rng, S, n) .+ S(0.2)
    x[k] = S(NaN)
    f = CheckLoss(S(0.3), w)
    out = PB.prox!(similar(x), f, x, S(0.7))
    @test isnan(out[k]) && all(isfinite, out[setdiff(1:n, k)])
    z = copy(x); PB.prox!(z, f, z, S(0.7)); @test same(z, out)
    # a zero weight leaves the entry free (NaN stays), a hard weight fixes it at 0
    for (we, ok) in ((S(0), isnan), (S(Inf), iszero))
      w2 = copy(w); w2[k] = we
      f2 = CheckLoss(S(0.3), w2)
      o = PB.prox!(similar(x), f2, x, S(0.7))
      @test ok(o[k]) && same(o, entrywise(f2, x, S(0.7)))
    end
    # an infinite step with a positive weight is a constant map: 0, also at the NaN entry
    o = PB.prox!(similar(x), f, x, S(Inf))
    @test all(iszero, o) && same(o, entrywise(f, x, S(Inf)))
  end
end

@testset "steps that are not ordinary for the whole array: $S" for S in (Float16, Float32, Float64)
  rng = MersenneTwister(5194)
  n = 400
  x = randn(rng, S, n); w = rand(rng, S, n) .+ S(0.2)
  x[[1, 128, 129, n]] .= S[NaN, S(2), -S(2), S(Inf)]
  w[[2, 127]] .= S[0, Inf]
  for f in (CheckLoss(S(0.3), w), CheckLoss(S(0.3)), CheckLoss(0.3, w), CheckLoss(nextfloat(zero(S)), w), CheckLoss(prevfloat(one(S)), w))
    for γ in (S(0), nextfloat(zero(S)), floatmin(S), floatmax(S), S(Inf), S(0.7))
      want = entrywise(f, x, γ)
      @test same(PB.prox!(similar(x), f, x, γ), want)
      z = copy(x); PB.prox!(z, f, z, γ); @test same(z, want)
    end
  end
end

@testset "a step array with exceptional entries, a quantile of another precision, an integer destination" begin
  rng = MersenneTwister(5195)
  for S in (Float32, Float64), n in (129, 700)
    x = randn(rng, S, n); w = rand(rng, S, n) .+ S(0.2)
    γ = rand(rng, S, n) .+ S(0.1)
    γ[[1, 128, 129, 500 % n + 1]] .= S[0, S(Inf), nextfloat(zero(S)), floatmax(S)]
    for f in (CheckLoss(S(0.3), w), CheckLoss(0.3, w), CheckLoss(BigFloat("0.3"), w))
      want = entrywise(f, x, γ)
      @test same(PB.prox!(similar(x), f, x, γ), want)
      z = copy(x); PB.prox!(z, f, z, γ); @test same(z, want)
    end
  end
  # float quantile with data of another type
  x32 = randn(rng, Float32, 300); w64 = rand(rng, 300) .+ 0.2
  for τ in (prevfloat(1.0), nextfloat(0.0), 0.3), γ in (1.0, 1e-30, 1e30, Inf)
    f = CheckLoss(τ, w64)
    @test same(PB.prox!(similar(x32), f, x32, γ), entrywise(f, x32, γ))
    f = CheckLoss(τ, Float32.(w64))
    @test same(PB.prox!(similar(x32), f, x32, γ), entrywise(f, x32, γ))
  end
  # an integer destination: the results are the free entries and the entries fixed at 0
  x = Float64.(rand(rng, -5:5, 300)); wi = ifelse.(rand(rng, 300) .< 0.5, 0.0, Inf)
  f = CheckLoss(0.3, wi)
  out = zeros(Int, 300)
  PB.prox!(out, f, x, 0.7)
  @test out == Int.(entrywise(f, x, 0.7))
  # the destination is a parameter array: every parameter is read before it is written
  w = rand(rng, 300) .+ 0.2; w0 = copy(w); x = randn(rng, 300)
  f = CheckLoss(0.3, w)
  want = entrywise(CheckLoss(0.3, copy(w0)), x, 0.7)
  @test PB.prox!(w, f, x, 0.7) === w && same(w, want)
  γarr = rand(rng, 300) .+ 0.1; γ0 = copy(γarr)
  want = entrywise(CheckLoss(0.3, copy(w0)), x, γ0)
  @test PB.prox!(γarr, CheckLoss(0.3, copy(w0)), x, γarr) === γarr && same(γarr, want)
end

# a function barrier: the measurement must not include the boxing of arguments built from a type variable
allocs_out(y, f, x, γ) = (PB.prox!(y, f, x, γ); PB.prox!(y, f, x, γ); @allocated PB.prox!(y, f, x, γ))
allocs_in(z, f, γ) = (PB.prox!(z, f, z, γ); PB.prox!(z, f, z, γ); @allocated PB.prox!(z, f, z, γ))

@testset "no allocation in the array loop" begin
  rng = MersenneTwister(5196)
  for S in (Float32, Float64)
    x = randn(rng, S, 1000); w = rand(rng, S, 1000) .+ S(0.2); y = similar(x)
    f = CheckLoss(S(0.3), w)
    for γ in (S(0.7), S(Inf), fill(S(0.7), 1000))
      @test allocs_out(y, f, x, γ) == 0
      @test allocs_in(copy(x), f, γ) == 0
    end
    # entries that need recovery (the repair is entrywise, with no allocation either)
    w[500] = S(0)
    @test allocs_out(y, f, x, S(0.7)) == 0
    @test allocs_out(y, f, x, nextfloat(zero(S))) == 0
    @test allocs_in(copy(x), f, nextfloat(zero(S))) == 0
  end
end

end
