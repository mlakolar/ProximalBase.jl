module NormL1NonnegTest
# NormL1Nonneg (#947): the ℓ1 penalty λ0 Σᵢ wᵢ xᵢ on the nonnegative orthant, Inf outside it. Constructor and traits,
# analytic cases, proximal points against independent one-dimensional minima, the rejection of negative and NaN scalar
# steps by the public array oracles, the representations, the value at the stored output of value_and_prox!, restriction
# and inferred types, agreement with the archived implementation (legacy/full eff21b6d) and warmed allocations. The
# endpoint truth table, the values against BigFloat, the point-aware KKT scores and Scaled are in
# norml1nonneg_extremes.jl.
#
# References are independent of the kernels: a golden-section search of the objective c u + (u - v)²/(2γ) on u ≥ 0 in
# 4096-bit BigFloat, and the closed form max(v - t̂, 0) in BigFloat with t̂ the threshold that the NormL1 kernel forms
# (a proximal point is within 4 units in the last place of it, and within the rounding of the threshold of the exact
# minimum), and the value λ0 Σ wᵢ xᵢ in BigFloat, within (n + 2) eps relative for n terms. Equalities are of two kinds:
# with an exact answer (zeros, identities, infinities, representable values), and between routes that compute the same
# numbers (prox! and prox_block, in place and out of place, value_and_prox! and the value at its stored output), whose
# common result need not be exact.
using Test, ProximalBase, LinearAlgebra, SparseArrays, Random
const PB = ProximalBase

big4096(f) = setprecision(f, BigFloat, 4096)
weight(f, i) = f.w === nothing ? 1 : f.w[i]
# the threshold γλ0wᵢ as the NormL1 kernel rounds it (it does not depend on the trial)
threshold(::Type{T}, f, i, γ) where {T} = PB._l1_prox_threshold(zero(T), γ, f.λ0, PB._entry(f.w, i), T)[2]
# max(v - t̂, 0) in BigFloat, with the threshold of the kernel
function proxref(::Type{T}, f, v, i, γ) where {T}
  t = threshold(T, f, i, γ)
  big4096(() -> max(big(T(v)) - big(t), big(0)))
end
# `within` measures the error in BigFloat against k units in the last place of the reference rounded to the type of the
# result (an infinite rounded reference must be reproduced)
function within(got::T, ref::BigFloat, k = 4) where {T}
  r = T(ref)
  isfinite(r) || return isequal(got, r)
  isequal(got, r) || big4096(() -> abs(big(got) - ref) <= k * big(eps(abs(r))))
end
# the minimiser of u -> c u + (u - v)²/(2γ) over [0, max(v, 0) + 1] by golden-section search (convex objective), in
# BigFloat: 500 steps shrink the interval to 2^-347 of its length
function golden(obj, a, b, steps)
  φ = (sqrt(big(5)) - 1) / 2
  c = b - φ * (b - a)
  d = a + φ * (b - a)
  gc, gd = obj(c), obj(d)
  for _ in 1:steps
    if gc < gd
      b, d, gd = d, c, gc
      c = b - φ * (b - a)
      gc = obj(c)
    else
      a, c, gc = c, d, gd
      d = a + φ * (b - a)
      gd = obj(d)
    end
  end
  (a + b) / 2
end
function minimiser(::Type{T}, f, v, i, γ) where {T}
  big4096() do
    c = (iszero(f.λ0) || iszero(weight(f, i))) ? big(0) : big(f.λ0) * big(weight(f, i))
    vb = big(T(v))
    golden(u -> c * u + (u - vb)^2 / (2 * big(γ)), big(0), max(vb, big(0)) + 1, 500)
  end
end
# λ0 Σ wᵢ xᵢ over the entries of a dense array in BigFloat: NaN when an entry is NaN, otherwise Inf when one is
# negative; a zero coefficient λ0wᵢ and a zero entry give a zero term (0⋅Inf = 0, as the value of NormL1)
function valref(f, x)
  any(isnan, x) && return big4096(() -> big(NaN))
  any(<(0), x) && return big4096(() -> big(Inf))
  big4096() do
    s = big(0)
    for i in eachindex(x)
      c = (iszero(f.λ0) || iszero(weight(f, i))) ? big(0) : big(f.λ0) * big(weight(f, i))
      s += (iszero(c) || iszero(x[i])) ? big(0) : c * big(x[i])
    end
    s
  end
end
function valok(got::T, ref, n) where {T}
  isfinite(ref) || return isequal(got, T(ref))
  isfinite(got) && big4096(() -> abs(big(got) - ref) <= (n + 2) * eps(T) * abs(ref))
end
# a vector with linear indices lo:lo+n-1
struct ShiftedVector{T} <: AbstractVector{T}
  data::Vector{T}
  lo::Int
end
Base.size(a::ShiftedVector) = size(a.data)
Base.axes(a::ShiftedVector) = (a.lo:(a.lo + length(a.data) - 1),)
Base.IndexStyle(::Type{<:ShiftedVector}) = IndexLinear()
Base.getindex(a::ShiftedVector, i::Int) = a.data[i - a.lo + 1]
Base.setindex!(a::ShiftedVector, v, i::Int) = (a.data[i - a.lo + 1] = v)

# a weight array that counts its reads, to show that values and proximal points on sparse and symmetric iterates
# read the weights of the stored entries only
mutable struct CountingArray{T, N} <: AbstractArray{T, N}
  data::Array{T, N}
  reads::Int
end
CountingArray(a) = CountingArray(a, 0)
Base.size(a::CountingArray) = size(a.data)
Base.IndexStyle(::Type{<:CountingArray}) = IndexLinear()
Base.getindex(a::CountingArray, i::Int) = (a.reads += 1; a.data[i])

# The archived NormL1Nonneg (legacy/full eff21b6d, src/functions/separable.jl, lines 400-475), reproduced from that source
# tree at one entry: prox_block is max(v - γλ0w, 0), and value_entry accepts v ≥ -sqrt(eps) and clamps it at 0.
legacy_prox(λ, w, v::T, γ) where {T} = max(v - PB._weight_product(float(T), γ, λ, w), zero(float(T)))
function legacy_value_entry(λ, w, v::T) where {T}
  S = float(T)
  wi = w === nothing ? true : w
  if v >= -sqrt(eps(S))
    a = max(S(v), zero(S))
    p = PB._positive_product(λ, wi, a)
    ifelse(isinf(wi) & !iszero(a), oftype(p, Inf) * a, p)
  else
    oftype(PB._positive_product(λ, wi, zero(S)), isnan(v) ? NaN : Inf)
  end
end

# warmed allocations of the dense oracles
alloc_prox(y, f, x, γ) = @allocated prox!(y, f, x, γ)
alloc_vprox(y, f, x, γ) = @allocated value_and_prox!(y, f, x, γ)
alloc_value(f, x) = @allocated f(x)
alloc_screen(f, g, i) = @allocated kkt_violation(f, g, i)
alloc_local(f, g, x, i) = @allocated kkt_violation(f, g, x, i)
alloc_point(f, g, x) = @allocated kkt_violation(f, g, x)
function loop_prox_block(f, x, γ)
  s = zero(eltype(x))
  @inbounds for i in eachindex(x)
    s += prox_block(f, x[i], i, γ)
  end
  s
end
alloc_loop(f, x, γ) = @allocated loop_prox_block(f, x, γ)

@testset "NormL1Nonneg" begin
  @testset "constructor, traits and parameters" begin
    f = NormL1Nonneg(1)
    @test f isa NormL1Nonneg{Int, Nothing} && f.λ0 === 1 && f.w === nothing && f.l1 isa NormL1{Int, Nothing}
    @test NormL1Nonneg().λ0 === 1 && NormL1Nonneg(0.5).λ0 === 0.5 && NormL1Nonneg(0.5f0) isa NormL1Nonneg{Float32, Nothing}
    w = [1.0, 0.0, Inf]
    g = NormL1Nonneg(0.25f0, w)
    @test g isa NormL1Nonneg{Float32, Vector{Float64}} && g.λ0 === 0.25f0 && g.w === w && g.l1.w === w && g.l1.λ0 === 0.25f0
    @test isconvex(g) && isseparable(g) && isproximable(g) && !issmooth(g) && iszeropreserving(g) && !PB._isindicator(g)
    @test iszeropreserving(NormL1Nonneg(Inf, w)) && iszeropreserving(NormL1Nonneg(0.0))
    @test PB.entry_parameters(g) === (:w => w,)
    for λ in (-1.0, NaN, -Inf)
      @test_throws ArgumentError NormL1Nonneg(λ)
      @test_throws ArgumentError NormL1Nonneg(λ, [1.0])
    end
    @test_throws ArgumentError NormL1Nonneg(1.0, [1.0, -1.0])
    @test_throws ArgumentError NormL1Nonneg(1.0, [1.0, NaN])
    @test_throws ArgumentError NormL1Nonneg(1.0, [1.0 + 0im])
    @test_throws ArgumentError NormL1Nonneg(1.0, ShiftedVector([1.0, 2.0], 0))
    @test_throws ArgumentError NormL1Nonneg(1.0)(ComplexF64[1, 2])
    @test_throws DimensionMismatch NormL1Nonneg(1.0, [1.0, 2.0])([1.0, 2.0, 3.0])
    @test_throws DimensionMismatch prox(NormL1Nonneg(1.0, [1.0, 2.0]), [1.0, 2.0, 3.0])
    @test_throws ArgumentError lipschitz(g)
    # the vector form of prox_block has no NormL1Nonneg hook
    @test_throws MethodError prox_block(NormL1Nonneg(1.0), [1.0, 2.0], 1, 1.0)
    # weights are kept by reference and read at the call, not revalidated
    wm = [1.0, 2.0]
    fm = NormL1Nonneg(1.0, wm)
    wm[2] = 5.0
    @test fm([0.0, 1.0]) === 5.0 && fm.l1.w === wm
  end

  for T in (Float32, Float64)
    @testset "$T analytic cases" begin
      λ0, γ = T(0.5), T(0.25)
      x = T[3, -2, 0.1, -0.05, 0, 1.25, -0.0, 7]
      w = T[1, 2, 0, 0.5, 1, 3, 1, 0.25]
      n = length(x)
      for f in (NormL1Nonneg(λ0), NormL1Nonneg(λ0, w))
        y = prox(f, x, γ)
        @test eltype(y) === T
        @test all(within(y[i], proxref(T, f, x[i], i, γ)) for i in 1:n)
        @test all(y[i] === prox_block(f, x[i], i, γ) for i in 1:n)
        @test all(≥(0), y) && all(!signbit, y)                           # a zero result is +0, also from -0
        # optimality of min over u ≥ 0 of c u + (u - xᵢ)²/(2γ): the derivative c + (yᵢ - xᵢ)/γ is zero where yᵢ > 0
        # and not negative where yᵢ = 0
        for i in 1:n
          g = big4096() do
            c = big(λ0) * big(weight(f, i))
            c + (big(y[i]) - big(x[i])) / big(γ)
          end
          tol = 64 * eps(T) * (1 + abs(x[i])) / γ
          @test iszero(y[i]) ? g >= -tol : abs(g) <= tol
        end
        # the value: Inf at a point with a negative entry, and the weighted sum at a feasible one
        @test f(x) === T(Inf)
        xp = abs.(x)
        @test valok(f(xp), valref(f, xp), n)
        @test valok(T(sum(value_entry(f, xp[i], i) for i in 1:n)), valref(f, xp), n)
        @test value_entry(f, x[2], 2) === T(Inf) && value_entry(f, -T(0.05), 4) === T(Inf)
        @test value_entry(f, x[7], 7) === zero(T) && value_entry(f, x[5], 5) === zero(T)
        # in place, out of place over a nonzero output, the allocating form and the value at the stored point
        z = fill(T(99), n)
        @test prox!(z, f, x, γ) === z && z == y
        a = copy(x)
        @test prox!(a, f, a, γ) === a && a == y
        z .= T(99)
        v = value_and_prox!(z, f, x, γ)
        @test z == y && v === f(z)
        a .= x
        @test value_and_prox!(a, f, a, γ) === f(y) && a == y
        @test prox!(z, f, x) == prox(f, x, 1)                              # the default step is one
      end
      # a singleton, a matrix with matrix weights, and views
      @test prox(NormL1Nonneg(λ0), T[3], γ) == [prox_block(NormL1Nonneg(λ0), T(3), 1, γ)]
      X, W = reshape(x, 2, 4), reshape(w, 2, 4)
      @test prox(NormL1Nonneg(λ0, W), X, γ) == reshape(prox(NormL1Nonneg(λ0, w), x, γ), 2, 4)
      xp = abs.(x)
      @test valok(NormL1Nonneg(λ0, W)(reshape(xp, 2, 4)), valref(NormL1Nonneg(λ0, w), xp), n)
      u = vcat(T[5], x, T[5])
      @test prox(NormL1Nonneg(λ0, w), view(u, 2:n+1), γ) == prox(NormL1Nonneg(λ0, w), x, γ)
      up = vcat(T[5], xp, T[5])
      @test valok(NormL1Nonneg(λ0, w)(view(up, 2:n+1)), valref(NormL1Nonneg(λ0, w), xp), n)
      @test NormL1Nonneg(λ0, w)(view(u, 2:n+1)) === T(Inf)
      # exact threshold ties give +0: γλ0 = 1 at v = 1, and -0 gives +0
      tie = NormL1Nonneg(T(0.5))
      @test prox_block(tie, one(T), 1, T(2)) === zero(T) && prox_block(tie, -zero(T), 1, T(2)) === zero(T)
      @test prox_block(NormL1Nonneg(zero(T)), -zero(T), 1, γ) === zero(T)
      # a function of integers
      @test prox(NormL1Nonneg(1), [3, -1, 0], 1) == [2.0, 0.0, 0.0]
      @test NormL1Nonneg(1)([3, 1, 0]) === 4.0 && NormL1Nonneg(1)([3, -1, 0]) === Inf
      # a level that is an integer with data of a type: the result takes the type of the data
      @test NormL1Nonneg(2)(T[1, 0.5]) === T(3) && NormL1Nonneg(2, T[1, 1])(T[1, 0.5]) === T(3)
    end
  end

  # a golden-section search of the objective, and the closed form with the exact threshold, at ordinary inputs
  for T in (Float32, Float64)
    @testset "$T proximal points against constrained minima" begin
      trials = T[-2, -0.0, 0.001, 0.3, 1, 2.5, 10]
      n = 0
      for λ0 in T[0, 0.5, 2.5, 1e-3], w in (nothing, T[0], T[0.5], T[3]), γ in T[0.25, 1, 7], v in trials
        f = NormL1Nonneg(λ0, w)
        y = prox_block(f, v, 1, γ)
        t = threshold(T, f, 1, γ)
        ref = proxref(T, f, v, 1, γ)
        @test y isa T && within(y, ref)
        # the exact threshold γ λ0 w, and the minimiser of the objective
        exact = big4096() do
          c = (iszero(λ0) || iszero(weight(f, 1))) ? big(0) : big(λ0) * big(weight(f, 1))
          max(big(v) - big(γ) * c, big(0))
        end
        u = minimiser(T, f, v, 1, γ)
        tol = big4096(() -> 4 * big(eps(T(ref))) + 2 * big(eps(t)))
        slack = big4096(() -> ldexp(big(1.0), -300))
        @test big4096(() -> abs(u - exact) <= slack)
        @test big4096(() -> abs(big(y) - exact) <= tol + slack)
        @test big4096(() -> abs(big(y) - u) <= tol + slack)
        n += 1
      end
      @test n == 4 * 4 * 3 * 7
    end
  end

  for T in (Float32, Float64)
    @testset "$T negative and NaN scalar steps are rejected before writes" begin
      f = NormL1Nonneg(T(1), T[1, 2])
      x = T[3, -2]
      for γ in (-one(T), T(NaN), -T(Inf), -T(0.5))
        y = T[7, 8]
        @test_throws ArgumentError prox!(y, f, x, γ)
        @test y == T[7, 8]
        @test_throws ArgumentError value_and_prox!(y, f, x, γ)
        @test y == T[7, 8]
        a = copy(x)
        @test_throws ArgumentError prox!(a, f, a, γ)
        @test a == x
        @test_throws ArgumentError value_and_prox!(a, f, a, γ)
        @test a == x
        @test_throws ArgumentError prox(f, x, γ)
        s = SparseIterate(x); s0 = copy(s)
        @test_throws ArgumentError prox!(s, f, s, γ)
        @test Vector(s) == Vector(s0) && collect(activecoordinates(s)) == collect(activecoordinates(s0))
        b = BlockIterate(copy(x), [[1], [2]])
        @test_throws ArgumentError prox!(b, f, b, γ)
        @test Vector(b) == x
        S = SymmetricSparseIterate(T[1 2; 2 0]); S0 = copy(S)
        fs = NormL1Nonneg(T(1), ones(T, 2, 2))
        @test_throws ArgumentError prox!(S, fs, S, γ)
        @test_throws ArgumentError value_and_prox!(S, fs, S, γ)
        @test_throws ArgumentError prox(fs, S, γ)
        @test Matrix(S) == Matrix(S0) && collect(activecoordinates(S)) == collect(activecoordinates(S0))
      end
      msg = try prox(f, x, -1) catch e; sprint(showerror, e) end
      @test occursin("NormL1Nonneg", msg) && occursin("nonnegative step", msg)
      # a zero step is accepted, -0 included: the projection onto the orthant, and -0 gives +0
      for γ in (zero(T), -zero(T))
        y0 = prox(f, T[3, -0.0], γ)
        @test isequal(y0, T[3, 0]) && isequal(prox(f, x, γ), T[3, 0])
      end
      @test isequal(prox(f, x, T(Inf)), T[0, 0])
      # the entries of an array of steps are not read for this check (a documented precondition)
      @test prox(f, x, T[-1, 1]) isa Vector{T}
      # prox_block does not check, and Scaled keeps its own step rule: Inf * f at γ = 0 forms Inf⋅0 = NaN (#311)
      @test prox_block(f, T(3), 1, -one(T)) isa T
      @test all(isnan, prox(T(Inf) * NormL1Nonneg(T(1)), x, zero(T)))
    end
  end

  for T in (Float32, Float64)
    @testset "$T representations" begin
      λ0, γ = T(0.4), T(0.5)
      n = 10
      w = T[1, 2, 0, 0.5, Inf, 1, 3, 1, 0.25, 2]
      f = NormL1Nonneg(λ0, w)
      # SparseIterate: stored entries only; in place, a stored ±0 stays stored as +0; absent entries are not activated
      x = SparseIterate(T, n)
      x[2] = 3; x[4] = -1; x[9] = T(0.1)
      activate!(x, 5); activate!(x, 7); x[7] = -zero(T)
      dense = Vector(x)
      y = prox(f, x, γ)                                                      # a new output stores the nonzero results
      @test y isa SparseIterate{T} && Vector(y) == prox(f, dense, γ)
      @test sort(collect(activecoordinates(y))) == [2, 9]
      @test f(x) === T(Inf) && f(dense) === T(Inf)                           # a stored negative entry
      xf = SparseIterate(T, n)
      xf[2] = 3; xf[4] = T(0.5); xf[9] = T(0.1)
      activate!(xf, 5)
      @test valok(f(xf), valref(f, Vector(xf)), n)
      out = SparseIterate(T, n); out[1] = 5                                  # a stored entry of the output stays stored
      @test value_and_prox!(out, f, x, γ) === f(out) && Vector(out) == Vector(y) && isactive(out, 1) && out[1] === zero(T)
      @test prox!(x, f, x, γ) === x && Vector(x) == Vector(y)                # in place, every stored entry stays stored
      @test sort(collect(activecoordinates(x))) == [2, 4, 5, 7, 9] && x[7] === zero(T) && x[5] === zero(T) && x[4] === zero(T)
      # array steps on dense and sparse arguments
      steps = T(0.25) .+ T.(1:n) ./ 8
      dd = T[1, -2, 3, 0.5, 4, 0, 2, -0.0, 0.1, 6]
      ys = prox(f, dd, steps)
      @test all(within(ys[i], proxref(T, f, dd[i], i, steps[i])) for i in 1:n)
      @test Vector(prox(f, SparseIterate(dd), steps)) == ys
      # BlockIterate: inactive groups stay inactive and zero
      groups = [[1, 3], [2, 4], [5, 6, 7], [8, 9, 10]]
      b = BlockIterate(T[1, -2, 3, 0.5, 0, 0, 0, 2, 0, -1], groups)
      @test sort(collect(activecoordinates(b))) == [1, 2, 4]
      yb = prox(f, b, γ)
      @test Vector(yb) == prox(f, Vector(b), γ) && sort(collect(activecoordinates(yb))) == [1, 2, 4]
      @test f(b) === T(Inf)
      bf = BlockIterate(T[1, 2, 3, 0.5, 0, 0, 0, 2, 0, 1], groups)
      @test valok(f(bf), valref(f, Vector(bf)), n)
      vb = copy(b)
      @test value_and_prox!(vb, f, b, γ) === f(vb) && Vector(vb) == Vector(yb)
      @test Vector(prox(f, b, steps)) == prox(f, Vector(b), steps)
      # SymmetricSparseIterate: weights read at both positions, an off-diagonal coordinate counted twice in the value
      p = 4
      Wm = T[1 2 0 1; 2 1 3 0.5; 0 3 2 1; 1 0.5 1 4]
      S = SymmetricSparseIterate(T, p)
      S[1, 1] = 2; S[3, 1] = -1.5; S[4, 2] = T(0.2); S[4, 4] = -3
      fs = NormL1Nonneg(λ0, Wm)
      @test PB.check_symmetric_parameters(fs, S)
      M = Matrix(S)
      YS = prox(fs, S, γ)
      @test YS isa SymmetricSparseIterate{T} && Matrix(YS) == prox(fs, M, γ)
      @test sort(collect(activecoordinates(YS))) == sort([tril_index(p, 1, 1), tril_index(p, 4, 2)])
      @test prox!(zeros(T, p, p), fs, S, γ) == prox(fs, M, γ)              # a dense output from a symmetric argument
      @test fs(S) === T(Inf)
      SF = SymmetricSparseIterate(T, p)
      SF[1, 1] = 2; SF[3, 1] = T(1.5); SF[4, 2] = T(0.2); SF[4, 4] = 3
      MF = Matrix(SF)
      @test valok(fs(SF), valref(fs, MF), p^2)
      @test fs(SF) ≈ λ0 * sum(Wm .* MF)
      ZS = copy(S)
      @test value_and_prox!(ZS, fs, S, γ) === fs(ZS) && Matrix(ZS) == Matrix(YS)
      # the explicit check rejects asymmetric weights; the oracles do not compare them
      bad = copy(Wm); bad[2, 1] = 7
      @test_throws ArgumentError PB.check_symmetric_parameters(NormL1Nonneg(λ0, bad), S)
      @test PB.check_symmetric_parameters(NormL1Nonneg(λ0, Symmetric(bad, :L)), S)
      # a weight array of the wrong length is rejected before anything is written
      S0 = copy(S); short = NormL1Nonneg(λ0, ones(T, 3, 3))
      @test_throws DimensionMismatch prox!(S, short, S, γ)
      @test_throws DimensionMismatch value_and_prox!(S, short, S, γ)
      @test_throws DimensionMismatch short(S)
      @test Matrix(S) == Matrix(S0) && collect(activecoordinates(S)) == collect(activecoordinates(S0))
      yd = fill(T(9), n)
      @test_throws DimensionMismatch prox!(yd, NormL1Nonneg(λ0, ones(T, 3)), dd, γ)
      @test yd == fill(T(9), n)
      # unsupported: array steps with a symmetric output, or with a dense output and a symmetric argument
      @test_throws ArgumentError prox!(S, fs, S, ones(T, p, p))
      @test_throws ArgumentError value_and_prox!(S, fs, S, ones(T, p, p))
      @test Matrix(S) == Matrix(S0)
      @test_throws MethodError prox!(zeros(T, p, p), fs, S, ones(T, p, p))
      # an immutable argument is read and left as it was
      r = T(1):T(4)
      @test prox(NormL1Nonneg(λ0), r, γ) == prox(NormL1Nonneg(λ0), collect(r), γ) && r == T(1):T(4)
      @test NormL1Nonneg(λ0)(r) ≈ λ0 * 10

      # a sparse iterate and a symmetric iterate read the weights of the stored entries only
      m = 20_000
      cw = CountingArray(T(0.5) .+ rand(Xoshiro(1), T, m))
      for λ in (T(0.7), zero(T), T(Inf))
        g = NormL1Nonneg(λ, cw)
        z = SparseIterate(T, m)
        for k in (3, 4000, 17_777)
          z[k] = T(k) / 1000
        end
        cw.reads = 0
        val = g(z)
        @test cw.reads <= 2 * nactive(z)
        @test isequal(val, g(Vector(z))) || isapprox(val, g(Vector(z)); rtol = 8 * eps(T))
        z2 = copy(z)
        cw.reads = 0
        prox!(z2, g, z2, T(0.5))
        @test cw.reads <= nactive(z) && nactive(z2) == nactive(z)
        # the recovery of terms below floatmin reads the stored entries too
        h = NormL1Nonneg(floatmin(T), cw)
        cw.reads = 0
        u = h(z)
        @test cw.reads <= 3 * nactive(z)
        @test u ≈ h(Vector(z))
      end
      pp = 150
      CW = CountingArray(ones(T, pp, pp))
      SS = SymmetricSparseIterate(T, pp)
      SS[7, 3] = 2; SS[100, 100] = 1; SS[150, 1] = T(0.5)
      fc = NormL1Nonneg(T(0.3), CW)
      CW.reads = 0
      @test fc(SS) ≈ T(0.3) * (2 * 2 + 1 + 2 * T(0.5))
      @test CW.reads <= 4 * nactive(SS)
    end
  end

  # value_and_prox! returns the value at the stored output, bit for bit, at lengths on both sides of the loop switches
  for T in (Float32, Float64)
    @testset "$T value_and_prox! is the value at the stored output" begin
      rng = Xoshiro(947)
      for n in (1, 7, 8, 15, 16, 17, 100, 1023, 1024, 1025, 2048, 5000)
        x = T(2) .* randn(rng, T, n)
        x[rand(rng, n) .< 0.1] .= zero(T)
        w = T(0.5) .+ rand(rng, T, n)
        for f in (NormL1Nonneg(T(0.4)), NormL1Nonneg(T(0.4), w), NormL1Nonneg(zero(T)), NormL1Nonneg(zero(T), w),
                  NormL1Nonneg(T(Inf)), NormL1Nonneg(T(0.4), [Inf; w[2:end]]))
          for γ in (T(0.3), T(2), zero(T))
            y = similar(x)
            @test isequal(value_and_prox!(y, f, x, γ), f(y))
            @test isequal(y, prox(f, x, γ)) && all(!signbit, y)
            a = copy(x)
            @test isequal(value_and_prox!(a, f, a, γ), f(y)) && isequal(a, y)
            sx = SparseIterate(x .* (rand(rng, T, n) .< T(0.3))); sy = copy(sx)
            @test isequal(value_and_prox!(sy, f, sx, γ), f(sy))
          end
        end
      end
      # the value is NaN at a NaN output, and a +Inf entry stays +Inf at a zero level, where its value is 0
      f0 = NormL1Nonneg(zero(T))
      y = zeros(T, 3)
      @test isnan(value_and_prox!(y, f0, T[NaN, 1, 2], T(0.5))) && isnan(y[1]) && y[2:3] == T[1, 2]
      @test value_and_prox!(y, f0, T[Inf, 1, -2], T(0.5)) === zero(T) && y == T[Inf, 1, 0]
    end
  end

  for T in (Float32, Float64)
    @testset "$T restriction and inferred types" begin
      w = T[1, 0, 2, Inf]
      f = NormL1Nonneg(T(0.5), w)
      r = restrict(f, [3, 1])
      @test r isa NormL1Nonneg && r.λ0 === f.λ0 && r.w == T[2, 1] && r.l1.w == T[2, 1] && r(T[1, 2]) == f(T[2, 0, 1, 0])
      @test r(T[1, -2]) === T(Inf)
      @test restrict(NormL1Nonneg(T(1)), [2]).w === nothing && restrict(NormL1Nonneg(T(1)), [2]) isa NormL1Nonneg
      # the entry value is one type: an unweighted Float32 function is Float32 (#684)
      @test value_entry(NormL1Nonneg(one(T)), one(T), 1) isa T
      @test @inferred(value_entry(NormL1Nonneg(one(T)), one(T), 1)) === one(T)
      @test @inferred(value_entry(NormL1Nonneg(one(T)), -one(T), 1)) === T(Inf)
      @test @inferred(value_entry(NormL1Nonneg(one(T)), T(NaN), 1)) |> isnan
      @test @inferred(value_entry(NormL1Nonneg(one(T), T[2]), one(T), 1)) === T(2)
      @test @inferred(prox_block(NormL1Nonneg(one(T)), one(T), 1, one(T))) === zero(T)
      @test @inferred(prox_block(NormL1Nonneg(one(T), T[2]), T(5), 1, one(T))) === T(3)
      x = T[1, 2, 0.5, 3]
      for f in (NormL1Nonneg(T(0.5)), NormL1Nonneg(T(0.5), T[1, 2, 3, 4]), NormL1Nonneg(zero(T)), NormL1Nonneg(T(Inf), T[1, 2, 3, 4]))
        @test @inferred(f(x)) isa T
        @test @inferred(f(view(x, :))) isa T
        @test @inferred(f(SparseIterate(x))) isa T
        @test @inferred(prox(f, x, T(0.5))) isa Vector{T}
        @test value_and_prox!(similar(x), f, x, T(0.5)) isa T
      end
      @test @inferred(kkt_violation(NormL1Nonneg(one(T)), x, 1)) isa T
      @test @inferred(kkt_violation(NormL1Nonneg(one(T)), x, x, 1)) isa T
      @test @inferred(kkt_violation(NormL1Nonneg(one(T)), x, x)) isa T
    end
  end

  @testset "agreement with the archived implementation" begin
    # the proximal points are those of the archive wherever the current policy did not change them: bit for bit,
    # signed zeros included, over levels, weights, steps and trials with the extremes of each type
    for T in (Float32, Float64)
      λs = T[0, 1, 2.5, floatmin(T), floatmax(T)]
      ws = T[0, 0.5, 1, Inf]
      γs = T[0, 0.5, Inf, nextfloat(zero(T)), floatmax(T)]
      vs = T[-Inf, -3, -nextfloat(zero(T)), -0.0, 0.0, nextfloat(zero(T)), 3, floatmax(T), Inf, NaN]
      rows = 0
      for λ in λs, wv in ws, γ in γs, v in vs
        rows += 1
        @test isequal(prox_block(NormL1Nonneg(λ, T[wv]), v, 1, γ), legacy_prox(λ, wv, v, γ))
      end
      @test rows == 5 * 4 * 5 * 10
    end
    # an infinite level with a positive weight fixes the entry at +0 for every step, and the archive did not: a positive
    # trial at the zero step stays itself (an infinite one too), +Inf at a positive step gives NaN, and NaN gives NaN
    for T in (Float32, Float64), wv in T[1, 0.5, Inf]
      f = NormL1Nonneg(T(Inf), T[wv])
      for v in T[3, floatmax(T), nextfloat(zero(T))]
        wv == Inf || @test legacy_prox(T(Inf), wv, v, zero(T)) === v
        @test prox_block(f, v, 1, zero(T)) === zero(T)
      end
      wv == Inf || @test legacy_prox(T(Inf), wv, T(Inf), zero(T)) === T(Inf)
      @test prox_block(f, T(Inf), 1, zero(T)) === zero(T)
      for γ in T[0.5, 1, floatmax(T), nextfloat(zero(T)), Inf]
        @test isnan(legacy_prox(T(Inf), wv, T(Inf), γ)) && prox_block(f, T(Inf), 1, γ) === zero(T)
        @test isnan(legacy_prox(T(Inf), wv, T(NaN), γ)) && prox_block(f, T(NaN), 1, γ) === zero(T)
      end
      @test isnan(legacy_prox(T(Inf), wv, T(NaN), zero(T))) && prox_block(f, T(NaN), 1, zero(T)) === zero(T)
    end
    # the archive accepted entries down to -sqrt(eps) and clamped them at zero; the domain is exact now
    for T in (Float32, Float64)
      tol = sqrt(eps(T))
      for v in (-tol / 2, -tol, -tol / 1000, -nextfloat(zero(T)))
        @test iszero(legacy_value_entry(T(1), nothing, v)) && value_entry(NormL1Nonneg(T(1)), v, 1) === T(Inf)
      end
      @test isinf(legacy_value_entry(T(1), nothing, -2 * tol))
      @test NormL1Nonneg(T(1))(T[1, -tol / 2]) === T(Inf)
      # the archived unweighted entry of Float32 data is a Float64 (#684); it is a Float32 now
      T === Float32 && @test typeof(legacy_value_entry(T(1), nothing, T(1))) === Float64
      @test typeof(value_entry(NormL1Nonneg(T(1)), T(1), 1)) === T
    end
    # at feasible points the values agree within the sum of their error bounds, 2(n + 2) eps
    rng = Xoshiro(947)
    for T in (Float32, Float64), n in (1, 8, 100, 1024, 2048)
      x = abs.(T(3) .* randn(rng, T, n))
      x[1:2:end] .= zero(T)
      w = T(0.5) .+ rand(rng, T, n)
      for (λ, ww) in ((T(0.3), nothing), (T(0.3), w), (zero(T), w), (T(2.5), nothing))
        f = NormL1Nonneg(λ, ww)
        old = sum(legacy_value_entry(λ, ww === nothing ? nothing : ww[i], x[i]) for i in 1:n; init = zero(T))
        got = f(x)
        @test abs(got - old) <= 2 * (n + 2) * eps(T) * abs(old)
      end
    end
  end

  # warmed allocations of the dense oracles
  for T in (Float32, Float64)
    @testset "$T allocations" begin
      for n in (8, 100, 1024, 2048)
        x = T(2) .* randn(Xoshiro(947), T, n)
        xp = abs.(x)
        w = T(0.5) .+ rand(Xoshiro(948), T, n)
        y = similar(x)
        g = randn(Xoshiro(949), T, n)
        for f in (NormL1Nonneg(T(0.4)), NormL1Nonneg(T(0.4), w), NormL1Nonneg(zero(T)), NormL1Nonneg(T(Inf), w))
          for γ in (T(0.3), zero(T))
            prox!(y, f, x, γ); alloc_prox(y, f, x, γ)
            @test alloc_prox(y, f, x, γ) == 0
            value_and_prox!(y, f, x, γ); alloc_vprox(y, f, x, γ)
            @test alloc_vprox(y, f, x, γ) == 0
          end
          f(xp); alloc_value(f, xp)
          @test alloc_value(f, xp) == 0
          f(x); @test alloc_value(f, x) == 0                                # a negative entry
          kkt_violation(f, g, 3); kkt_violation(f, g, xp, 3); kkt_violation(f, g, xp)
          alloc_screen(f, g, 3); alloc_local(f, g, xp, 3); alloc_point(f, g, xp)
          @test alloc_screen(f, g, 3) == 0 && alloc_local(f, g, xp, 3) == 0 && alloc_point(f, g, xp) == 0
        end
        f = NormL1Nonneg(T(0.4), w)
        loop_prox_block(f, x, T(0.3)); alloc_loop(f, x, T(0.3))
        @test alloc_loop(f, x, T(0.3)) == 0
      end
    end
  end
end
end
