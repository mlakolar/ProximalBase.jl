module FibersMidpointOverflowTest

# #667 (O2, O7, E2, E4): FiberSum of NormL1Fused on two slices, a separate output of value_and_prox!,
# in an identity regime (a zero level, or a zero step) without an infinite level. The loop takes the
# unchecked pair midpoint (u + w) / 2 there, which overflows for a pair near floatmax and wrote Inf
# into the output; and a NaN in the fibers of the identity function stopped the recovery of an
# unrelated fiber, so the value was NaN or Inf, while prox! and the aliased call stored finite entries
# and a finite value. Now the output is that of prox! (O7), the value is that of the stored output
# (O2): a zero level adds no value whatever its entries (E4), a NaN in a coupled fused pair gives NaN
# (E2), and an overflow of the stored result stays Inf.
# The value of the aliased call is compared with that of the stored output (aliasvalue) except where NaN
# or infinite entries stand in the free fibers and the sum overflows in the loop: there the recovery of
# the aliased call (unchanged here, as at base) returns the overflowed sum, so those rows compare the
# outputs of the aliased call and the values of the separate calls only.
# Rows marked "regression guard" pass at base and keep their results: the K = 3 view route and the
# ordinary route. Outputs are compared with isequal.

using Test, ProximalBase, Random

const PB = ProximalBase

# the value of the stored output: equal to fb(y) where the call decided it, within the reduction order
# (A1) where the ordinary loop did
samevalue(a, b) = isequal(a, b) || (isfinite(a) && isfinite(b) && isapprox(a, b; rtol = 64 * eps(typeof(a))))

# A destination of another type (dest, a mixed-precision call) leaves the aliased call out.
# prox!, and value_and_prox! into a fresh output, an output holding a sentinel, a NaN-filled output and
# the argument itself (a copy): the outputs are identical (O7), the values are those of the stored
# output (O2), checked exactly (exact = true) or within the reduction order. The result is the tuple of
# the separate value, the output and the aliased value.
function check(fb, x, γ; exact = true, expected = nothing, value = nothing, aliasvalue = true, dest = float(eltype(x)))
  T = dest
  mixed = T !== float(eltype(x))
  p = fill(T(-7), size(x))
  prox!(p, fb, x, γ)
  outs = Any[]
  vals = Any[]
  for fill_ in (:fresh, :sentinel, :nan)
    y = fill_ === :fresh ? similar(x, T) : fill(fill_ === :nan ? T(NaN) : T(-7), size(x))
    push!(vals, value_and_prox!(y, fb, x, γ))
    push!(outs, y)
  end
  # the aliased call stores into the argument: its type is the destination's only when they agree
  a = mixed ? copy(p) : copy(x)
  va = mixed ? T(fb(p)) : value_and_prox!(a, fb, a, γ)
  fy = fb(p)
  bad = Symbol[]
  isequal(a, p) || push!(bad, :aliased_output)
  all(y -> isequal(y, p), outs) || push!(bad, :separate_output)
  (all(v -> v isa T, vals) && va isa T) || push!(bad, :value_type)
  (exact ? all(v -> isequal(v, fy), vals) : all(v -> samevalue(v, fy), vals)) || push!(bad, :separate_value)
  aliasvalue && !samevalue(va, fy) && push!(bad, :aliased_value)
  expected === nothing || isequal(p, expected) || push!(bad, :expected_output)
  value === nothing || all(v -> isequal(v, value), vals) || push!(bad, :expected_value)
  @test isempty(bad)
  isempty(bad) || @info "check failed" bad vals va fy
  (vals[1], outs[1], va)
end

# an n × n × K array whose diagonal fibers hold d and whose other entries hold off; sym makes it symmetric
function build(T, n, K; d = floatmax(T), off = T(NaN))
  x = fill(T(off), n, n, K)
  for i in 1:n
    x[i, i, :] .= d
  end
  x
end

zero_level() = NormL1Fused(0.0, 0.0)

####################################
#  the table of the issue
####################################

@testset "separate output at floatmax pairs beside NaN in the zero-level fibers" begin
  for T in (Float32, Float64), sym in (false, true), n in (2, 3)
    fm = floatmax(T)
    x = build(T, n, 2)
    # off-diagonal function a zero level, diagonal function fused, value overflows (1 * 2 floatmax)
    fb = FiberSum(zero_level(); diagonal = NormL1Fused(1.0, 1.0), symmetric = sym)
    v, y, va = check(fb, x, 1.0; value = T(Inf))
    @test v === T(Inf) && va === T(Inf)
    for i in 1:n, j in 1:n
      i == j ? (@test y[i, i, 1] === fm && y[i, i, 2] === fm) : (@test isnan(y[i, j, 1]) && isnan(y[i, j, 2]))
    end
    # a level that keeps the value finite: n diagonal fibers of 0.125 * 2 floatmax
    fb = FiberSum(zero_level(); diagonal = NormL1Fused(0.125, 0.125), symmetric = sym)
    v, y, va = check(fb, x, 1.0; aliasvalue = false)
    @test isfinite(v) && isapprox(v, T(0.25) * n * fm; rtol = 1e-6) && isequal(v, fb(y))
    # the same at the fusion level only, equal entries: the value is 0 and the output floatmax
    fb = FiberSum(zero_level(); diagonal = NormL1Fused(0.0, 1.0), symmetric = sym)
    v, y, va = check(fb, x, 1.0; value = zero(T))
    @test all(i -> y[i, i, 1] === fm && y[i, i, 2] === fm, 1:n)
    # unequal entries beside the zero level: a fusion and a shrink, no midpoint overflow
    xu = build(T, n, 2)
    xu[1, 1, 1], xu[1, 1, 2] = fm, -fm
    fb = FiberSum(zero_level(); diagonal = NormL1Fused(0.5, 0.5), symmetric = sym)
    check(fb, xu, 1.0)
  end
end

@testset "zero step: the identity at every level, floatmax pairs beside NaN" begin
  for T in (Float32, Float64), sym in (false, true)
    n = 3
    x = build(T, n, 2)
    # a zero level for the off-diagonal fibers, finite levels on the diagonal, at a zero step
    fb = FiberSum(zero_level(); diagonal = NormL1Fused(0.125, 0.125), symmetric = sym)
    v, y, va = check(fb, x, 0.0; aliasvalue = false)
    @test isfinite(v) && isapprox(v, T(0.25) * n * floatmax(T); rtol = 1e-6)
    fb = FiberSum(zero_level(); diagonal = NormL1Fused(1.0, 1.0), symmetric = sym)
    v, y, va = check(fb, x, 0.0; value = T(Inf))
    @test isequal(y, x)
    fb = FiberSum(zero_level(); diagonal = NormL1Fused(0.125, 0.125), symmetric = sym)
    v, y, va = check(fb, x, 0; aliasvalue = false)
    @test isfinite(v) && isapprox(v, T(0.25) * n * floatmax(T); rtol = 1e-6)
  end
end

@testset "roles swapped: the zero-level function on the diagonal" begin
  for T in (Float32, Float64), sym in (false, true)
    n = 3
    fm = floatmax(T)
    # NaN in the zero-level fibers (the diagonal), floatmax pairs in the fused fibers
    x = build(T, n, 2; d = T(NaN), off = fm)
    fb = FiberSum(NormL1Fused(0.0625, 0.0625); diagonal = zero_level(), symmetric = sym)
    # the aliased value is not compared where the zero-level fibers on the diagonal hold NaN: its recovery
    # of a sum that overflows in the loop returns Inf there, as at base, and is not part of this change
    v, y, va = check(fb, x, 1.0; aliasvalue = false)
    @test isfinite(v) && isapprox(v, T(0.125) * n * (n - 1) * fm; rtol = 1e-6)
    @test all(i -> isnan(y[i, i, 1]) && isnan(y[i, i, 2]), 1:n)
    fb = FiberSum(NormL1Fused(1.0, 1.0); diagonal = zero_level(), symmetric = sym)
    v, y, va = check(fb, x, 1.0; value = T(Inf))
    @test v === T(Inf)
    # a NaN in a fused pair (the data of the first table with the roles swapped): E2, the value is NaN
    x = build(T, n, 2)
    fb = FiberSum(NormL1Fused(1.0, 1.0); diagonal = zero_level(), symmetric = sym)
    v, y, va = check(fb, x, 1.0)
    @test isnan(v) && isnan(va)
    @test all(i -> y[i, i, 1] === fm && y[i, i, 2] === fm, 1:n)
  end
end

@testset "a free diagonal beside NaN (diagonal = nothing)" begin
  for T in (Float32, Float64), sym in (false, true)
    n = 3
    fm = floatmax(T)
    # NaN on the free diagonal, floatmax pairs elsewhere
    x = build(T, n, 2; d = T(NaN), off = fm)
    fb = FiberSum(NormL1Fused(0.0625, 0.0625); diagonal = nothing, symmetric = sym)
    v, y, va = check(fb, x, 1.0; aliasvalue = false)
    @test isfinite(v) && isapprox(v, T(0.125) * n * (n - 1) * fm; rtol = 1e-6)
    @test all(i -> isnan(y[i, i, 1]) && isnan(y[i, i, 2]), 1:n)
    @test all(I -> I[1] == I[2] || (y[I, 1] === fm && y[I, 2] === fm), CartesianIndices((n, n)))
  end
end

####################################
#  the same checks on the K = 3 view route and the rows of the ordinary route (regression guards)
####################################

@testset "regression guard: K = 3 view route" begin
  for T in (Float32, Float64), sym in (false, true)
    n = 3
    x = build(T, n, 3)
    for fb in (FiberSum(zero_level(); diagonal = NormL1Fused(0.0, 1.0), symmetric = sym),
               FiberSum(zero_level(); diagonal = NormL1Fused(1.0, 1.0), symmetric = sym),
               FiberSum(NormL1Fused(0.25, 1.0); diagonal = zero_level(), symmetric = sym),
               FiberSum(NormL1(0.0); diagonal = NormL1(0.25), symmetric = sym))
      check(fb, x, 1.0; exact = false)
      check(fb, x, 0.0; exact = false)
    end
    fb = FiberSum(zero_level(); diagonal = NormL1Fused(0.0, 1.0), symmetric = sym)
    v, y, va = check(fb, x, 1.0; value = zero(T))
    @test all(i -> all(k -> y[i, i, k] === floatmax(T), 1:3), 1:n)
  end
end

@testset "NaN in a coupled pair stays NaN (E2), an overflowing result stays infinite" begin
  for T in (Float32, Float64), sym in (false, true)
    n = 3
    fm = floatmax(T)
    x = build(T, n, 2)
    # a NaN in one entry of a fused diagonal pair beside a floatmax pair: NaN (E2)
    x[1, 1, 1] = NaN
    fb = FiberSum(zero_level(); diagonal = NormL1Fused(0.125, 0.125), symmetric = sym)
    v, y, va = check(fb, x, 1.0)
    @test isnan(v) && isnan(va) && isnan(y[1, 1, 1]) && isnan(y[1, 1, 2])
    @test y[2, 2, 1] === fm && y[2, 2, 2] === fm
    # a coupled NaN and a zero step
    fb = FiberSum(zero_level(); diagonal = NormL1Fused(0.125, 0.125), symmetric = sym)
    v, y, va = check(fb, x, 0.0)
    @test isnan(v) && isnan(va)
    # the stored result overflows: the value stays infinite
    x = build(T, n, 2)
    fb = FiberSum(zero_level(); diagonal = NormL1Fused(1.0, 0.0), symmetric = sym)
    v, y, va = check(fb, x, 1.0; value = T(Inf))
    @test v === T(Inf) && va === T(Inf)
  end
end

@testset "mixed entries: floatmax, Inf, NaN, signed zeros, in free and fused fibers" begin
  rng = Random.MersenneTwister(667)
  for T in (Float32, Float64), sym in (false, true)
    fm = floatmax(T)
    pool = T[fm, -fm, fm / 2, T(Inf), T(-Inf), T(0), -T(0), T(1), T(-2), T(NaN), floatmin(T)]
    levels = ((0.0, 0.0), (0.125, 0.125), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (0.5, 0.25))
    for trial in 1:60
      n = 3
      x = fill(T(NaN), n, n, 2)
      for j in 1:n, i in j:n, k in 1:2
        x[i, j, k] = rand(rng, pool)
        x[j, i, k] = x[i, j, k]
      end
      # the free fibers (those of the zero-level function) hold NaN and infinities
      (fl, dl) = (rand(rng, levels), rand(rng, levels))
      swap = rand(rng, Bool)
      fb = swap ? FiberSum(NormL1Fused(fl...); diagonal = NormL1Fused(0.0, 0.0), symmetric = sym) :
        FiberSum(NormL1Fused(0.0, 0.0); diagonal = NormL1Fused(dl...), symmetric = sym)
      γ = rand(rng, (1.0, 0.5, 0.0, 2.0))
      # the aliased value is left out here: its recovery of an overflowed sum is not that of the value of the
      # stored output where free fibers hold NaN or an infinity; the aliased output is compared
      check(fb, x, γ; exact = false, aliasvalue = false)
      fb = FiberSum(NormL1Fused(fl...); diagonal = nothing, symmetric = sym)
      check(fb, x, γ; exact = false, aliasvalue = false)
    end
  end
end

@testset "regression guard: the ordinary route keeps its results and allocates nothing" begin
  rng = Random.MersenneTwister(668)
  for T in (Float32, Float64), sym in (false, true), n in (2, 5)
    x = randn(rng, T, n, n, 2)
    sym && (x .= (x .+ permutedims(x, (2, 1, 3))) ./ 2)
    for fb in (FiberSum(NormL1Fused(0.3, 0.2); diagonal = NormL1Fused(0.1, 0.1), symmetric = sym),
               FiberSum(NormL1Fused(0.3, 0.2); symmetric = sym),
               FiberSum(NormL1Fused(0.3, 0.2); diagonal = nothing, symmetric = sym))
      check(fb, x, 0.7; exact = false)
    end
    fb = FiberSum(NormL1Fused(0.3, 0.2); diagonal = NormL1Fused(0.1, 0.1), symmetric = sym)
    y = similar(x)
    value_and_prox!(y, fb, x, 0.7)
    a = copy(x)
    value_and_prox!(a, fb, a, 0.7)
    # nothing that grows with the array: Julia 1.10 reports a fixed 128 bytes for the call, at base too
    @test (@allocated value_and_prox!(y, fb, x, 0.7)) <= 128
    a .= x
    @test (@allocated value_and_prox!(a, fb, a, 0.7)) <= 128
  end
end

####################################
#  a destination narrower than the argument: the stored output decides (finite x, infinite y)
####################################

diagonal_idx(I) = I[1] == I[2]
offdiag_idx(I) = I[1] != I[2]

# Float64 data into a Float32 output: floatmax(Float64) is Inf32 in the output, and the free fibers
# hold it, so x is finite and the stored output is not. The value is that of the stored output: the
# infinities of the free fibers add no value (E4) and do not scale the sum of the other fibers.
@testset "Float64 input into a Float32 output" begin
  fm32, fm64 = Float64(floatmax(Float32)), floatmax(Float64)
  # the reviewer's example
  x = fill(fm32, 2, 2, 2)
  x[1, 2, :] .= fm64
  x[2, 1, :] .= fm64
  fb = FiberSum(NormL1Fused(0.0, 0.0); diagonal = NormL1Fused(0.125, 0.125))
  y = fill(-7.0f0, size(x))
  v = value_and_prox!(y, fb, x, 1.0)
  @test (all(isfinite, x), v, fb(y)) === (true, 1.7014117f38, 1.7014117f38)
  for sym in (false, true), γ in (1.0, 0.5, 0.0, 0), n in (2, 3)
    # (a) the zero-level function on the off-diagonal fibers, fused pairs on the diagonal
    # (b) roles swapped, (c) a free diagonal
    for (kind, fbm) in ((:a, FiberSum(NormL1Fused(0.0, 0.0); diagonal = NormL1Fused(0.125, 0.125), symmetric = sym)),
                        (:b, FiberSum(NormL1Fused(0.0625, 0.0625); diagonal = NormL1Fused(0.0, 0.0), symmetric = sym)),
                        (:c, FiberSum(NormL1Fused(0.0625, 0.0625); diagonal = nothing, symmetric = sym)))
      free = kind === :a ? offdiag_idx : diagonal_idx
      x = fill(fm32, n, n, 2)
      for I in CartesianIndices((n, n))
        free(I) && (x[I, :] .= fm64)
      end
      v, y, _ = check(fbm, x, γ; dest = Float32)
      @test isfinite(v) && v === fbm(y)
      @test any(isinf, y)
    end
    # a coupled overflow: a fused pair at floatmax(Float64) stores Inf32 in both entries, the value is
    # that of the stored output, whatever it is
    x = fill(fm32, n, n, 2)
    x[1, 1, :] .= fm64
    check(FiberSum(NormL1Fused(0.0, 0.0); diagonal = NormL1Fused(0.125, 0.125), symmetric = sym), x, γ; dest = Float32)
    check(FiberSum(NormL1Fused(0.0625, 0.0625); diagonal = nothing, symmetric = sym), x, γ; dest = Float32)
  end
  # Float32 data into a Float64 output, and equal types: the same checks
  for sym in (false, true)
    x = fill(floatmax(Float32), 3, 3, 2)
    x[1, 2, :] .= NaN
    x[2, 1, :] .= NaN
    check(FiberSum(NormL1Fused(0.0, 0.0); diagonal = NormL1Fused(0.125, 0.125), symmetric = sym), x, 1.0; dest = Float64)
  end
end

@testset "regression guard: K = 3 view route, Float64 input into a Float32 output" begin
  fm32, fm64 = Float64(floatmax(Float32)), floatmax(Float64)
  for sym in (false, true), γ in (1.0, 0.0)
    x = fill(fm32, 3, 3, 3)
    for I in CartesianIndices((3, 3))
      I[1] != I[2] && (x[I, :] .= fm64)
    end
    check(FiberSum(NormL1Fused(0.0, 0.0); diagonal = NormL1Fused(0.125, 0.125), symmetric = sym), x, γ; dest = Float32, exact = false)
    check(FiberSum(NormL1(0.0); diagonal = NormL1(0.125), symmetric = sym), x, γ; dest = Float32, exact = false)
  end
end

####################################
#  a slice axis that does not start at 1 (the pair kernel accepts it; the fiber views of fb(y) do not)
####################################

struct Shifted{T, N} <: AbstractArray{T, N}
  data::Array{T, N}
  starts::NTuple{N, Int}
end
Base.size(a::Shifted) = size(a.data)
Base.axes(a::Shifted{T, N}) where {T, N} =
  ntuple(d -> Base.IdentityUnitRange(a.starts[d]:a.starts[d] + size(a, d) - 1), N)
Base.IndexStyle(::Type{<:Shifted}) = IndexCartesian()
Base.getindex(a::Shifted{T, N}, i::Vararg{Int, N}) where {T, N} = a.data[ntuple(d -> i[d] - a.starts[d] + 1, N)...]
Base.setindex!(a::Shifted{T, N}, v, i::Vararg{Int, N}) where {T, N} =
  setindex!(a.data, v, ntuple(d -> i[d] - a.starts[d] + 1, N)...)

@testset "an output with a slice axis that does not start at 1 keeps its results" begin
  fb = FiberSum(NormL1Fused(0.0, 0.0); diagonal = NormL1Fused(0.125, 0.125))
  for starts in ((1, 1, -2), (0, 0, 1), (1, 1, 5))
    # a NaN in a fused diagonal pair beside free NaN-free fibers: the value is NaN (E2), no exception
    x = Shifted(ones(2, 2, 2), starts)
    x.data[1, 1, 1] = NaN
    y = Shifted(fill(-7.0, 2, 2, 2), starts)
    v = value_and_prox!(y, fb, x, 1.0)
    @test isnan(v)
    @test isequal(y.data[:, :, 1], [NaN 1.0; 1.0 0.875]) && isequal(y.data[:, :, 2], [NaN 1.0; 1.0 0.875])
    z = Shifted(fill(-7.0, 2, 2, 2), starts)
    prox!(z, fb, x, 1.0)
    @test isequal(z.data, y.data)
    a = Shifted(copy(x.data), starts)
    @test isnan(value_and_prox!(a, fb, a, 1.0)) && isequal(a.data, y.data)
    # a zero step and the roles swapped
    y = Shifted(fill(-7.0, 2, 2, 2), starts)
    @test isnan(value_and_prox!(y, fb, x, 0.0))
    fs = FiberSum(NormL1Fused(0.125, 0.125); diagonal = NormL1Fused(0.0, 0.0))
    y = Shifted(fill(-7.0, 2, 2, 2), starts)
    # the NaN sits in a fiber of the zero-level function here: free, no value (E4)
    @test value_and_prox!(y, fs, x, 1.0) === 0.4375
    # entries that are finite and large: the call returns, as at base (no exception)
    xb = Shifted(fill(floatmax(Float64), 2, 2, 2), starts)
    y = Shifted(fill(-7.0, 2, 2, 2), starts)
    @test value_and_prox!(y, fb, xb, 1.0) isa Float64
  end
end

end
