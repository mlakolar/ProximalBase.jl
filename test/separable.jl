module SeparableTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

# separable functions of a user: one that preserves zeros, and one that does not
struct WeightedSquare <: SeparableFunction       # Σᵢ wᵢ xᵢ² / 2
  w::Vector{Float64}
end
ProximalBase.value_entry(f::WeightedSquare, v, i) = f.w[i] * v^2 / 2
ProximalBase.prox_entry(f::WeightedSquare, v, i, γ) = v / (1 + γ * f.w[i])
ProximalBase.iszeropreserving(::WeightedSquare) = true
ProximalBase.isconvex(::Type{WeightedSquare}) = true
function ProximalBase.check_parameters(f::WeightedSquare, x)
  length(f.w) == length(x) || throw(DimensionMismatch("a parameter array has length $(length(f.w)) and the argument has length $(length(x))"))
  nothing
end

struct ShiftedAbs <: SeparableFunction           # Σᵢ |xᵢ - 1|, which does not preserve zeros
end
ProximalBase.value_entry(::ShiftedAbs, v, i) = abs(v - 1)
ProximalBase.prox_entry(::ShiftedAbs, v, i, γ) = 1 + sign(v - 1) * max(abs(v - 1) - γ, 0)

@testset "NormL1" begin
  @testset "shrink" begin
    x = randn(10)
    λ = maximum(abs, x) + 0.1
    @test prox(NormL1(λ), x) == zeros(10)
    @test NormL1(λ)(x) ≈ λ * sum(abs, x)
    x = [1.0, 1.4, -3.2]
    @test prox(NormL1(0.1), x) ≈ [0.9, 1.3, -3.1]
    w = [0.1, 1.5, 1]
    f = NormL1(1., w)
    @test prox(f, x) ≈ [0.9, 0., -2.2]
    @test f(x) ≈ dot(w, abs.(x))
    @test NormL1(2., nothing) == NormL1(2.)
    @test typeof(NormL1(2.)) == NormL1{Float64, Nothing}
    @test_throws ArgumentError NormL1(-1.)
    @test_throws ArgumentError NormL1(1., [-1., 1.])
  end
  for f in (NormL1(0.4), NormL1(0.4, rand(8)))
    x = randn(8)
    check_prox(f, x, 0.7)
    @test value_and_prox!(similar(x), f, x, 0.7) ≈ f(prox(f, x, 0.7))
  end
  @testset "Float32" begin
    x = randn(Float32, 10)
    @test NormL1(0.5)(x) isa Float32
    @test prox(NormL1(0.5), x, 0.3) isa Vector{Float32}
    @test NormL1(0.5, rand(10))(x) isa Float32
    @test prox(NormL1(0.5f0, rand(Float32, 10)), x, 0.3f0) isa Vector{Float32}
  end
end

@testset "weighted L1 dense loops" begin
  for T in (Float32, Float64), F in (NormL1, NormL1Nonneg)
    x = randn(T, 513)
    w = rand(T, length(x)) .+ T(0.2)
    f = F(T(0.3), w)
    step = T(0.7)
    expected = [prox_entry(f, x[i], i, step) for i in eachindex(x)]
    @test prox(f, x, step) == expected
    y = copy(x)
    @test prox!(y, f, y, step) === y
    @test y == expected
    @test value_and_prox!(similar(x), f, x, step) ≈ f(expected)
    for z in (x, abs.(x), expected)
      @test f(z) ≈ sum(value_entry(f, z[i], i) for i in eachindex(z))
    end
    # Matrices, views and mixed precision retain the entrywise interface.
    for z in (reshape(copy(x), 27, 19), view(x, :), Float64.(x))
      @test vec(prox(f, z, step)) ≈ expected
    end
    y = copy(w)
    @test prox!(y, F(f.λ0, y), x, step) == expected  # output may be the weights
    @test F(T(0.3), T[])(T[]) == zero(T)
    @test isempty(prox(F(T(0.3), T[]), T[], step))

    # True zero factors suppress NaN/Inf products, but the nonnegative
    # penalty still rejects infeasible entries, even at zero level/weight.
    # Finite sums may reassociate: n*eps(T) covers two orders of summation
    # of n nonnegative terms. Nonfinite values must still agree exactly.
    for level in (T(0), T(0.3), T(Inf)), v in (T(Inf), T(-Inf), T(NaN), T(-1), -zero(T))
      z = ones(T, 513)
      z[1] = v
      weights = ones(T, length(z))
      isinf(level) && fill!(weights, zero(T))
      weights[1] = zero(T)
      f = F(level, weights)
      reference = sum(value_entry(f, z[i], i) for i in eachindex(z))
      @test isfinite(reference) ? isapprox(f(z), reference; rtol=length(z)*eps(T)) : isequal(f(z), reference)
      if iszero(level)
        weights[1] = T(Inf)  # an infinite weight fixes the entry at zero, even at a zero level
        f = F(level, weights)
        reference = sum(value_entry(f, z[i], i) for i in eachindex(z))
        @test isfinite(reference) ? isapprox(f(z), reference; rtol=length(z)*eps(T)) : isequal(f(z), reference)
      end
    end

    # The total is normal even though one coefficient underflows: testing
    # only the aggregate would silently lose half the value.
    small = T === Float32 ? T(1e-30) : T(1e-200)
    f = F(small, T[small, inv(small)])
    z = T[inv(small), small]
    reference = T(sum(BigFloat(f.λ0) .* BigFloat.(f.w) .* BigFloat.(z)))
    @test f(z) ≈ reference rtol=8eps(T) atol=zero(T)

    tiny = nextfloat(zero(T))
    @test F(T(1.5), T[2])(T[tiny]) == 3tiny
    @test prox(F(T(2), T[tiny]), T[16tiny], T(1.5)) == T[13tiny]
    large = floatmax(T) / 2
    for (level, weights, z, γ) in (
        (large, T[inv(large), 0], T[7, 3], T(4)),
        (T(0), T[1, Inf], T[7, Inf], step),
        (T(Inf), T[0, 1], T[7, 3], T(0)))
      f = F(level, weights)
      # an infinite threshold at an infinite entry gives NaN, as soft thresholding does
      reference = [prox_entry(f, z[i], i, γ) for i in eachindex(z)]
      @test isequal(prox(f, z, γ), reference)
      out = copy(z)
      @test isequal(prox!(out, f, out, γ), reference)
      reference = sum(value_entry(f, z[i], i) for i in eachindex(z))
      @test isfinite(reference) ? isapprox(f(z), reference; rtol=length(z)*eps(T)) : isequal(f(z), reference)
    end
  end
end

@testset "sparse iterates" begin
  # the value and the proximal operator touch the stored entries of a SparseIterate,
  # and give the results of the dense array
  for f in (NormL1(0.3), NormL1(0.3, rand(50)), NormL1Nonneg(0.2), ElasticNet(0.3, 0.4), NormL0(0.05),
            MCP(0.3), SCAD(0.3), IndNonnegative(), IndBox(-0.5, 0.5))
    xs = SparseIterate(sprandn(50, 0.2))
    x = Vector(xs)
    @test f(xs) ≈ f(x)
    ys = prox(f, xs, 0.6)
    @test Vector(ys) ≈ prox(f, x, 0.6)
    # in place, the stored entries stay stored
    zs = copy(xs)
    prox!(zs, f, zs, 0.6)
    @test Vector(zs) ≈ prox(f, x, 0.6)
    @test nnz(zs) == nnz(xs)
    @test value_and_prox!(similar(xs), f, xs, 0.6) ≈ f(prox(f, x, 0.6))
    # into another iterate, its stored entries are kept, as for a dense array
    out = SparseIterate(50)
    out[1] = 3.
    prox!(out, f, xs, 0.6)
    @test Vector(out) ≈ prox(f, x, 0.6)
    @test isactive(out, 1)
  end
  # a box that excludes zero visits every entry
  f = IndBox(0.5, 1.)
  xs = SparseIterate(sprandn(20, 0.2))
  @test Vector(prox(f, xs, 1.)) == clamp.(Vector(xs), 0.5, 1.)
  @test f(xs) == Inf
  # on a SymmetricSparseIterate: the dense symmetric matrix, each coordinate once
  p = 6
  D = Matrix(Symmetric(sprandn(p, p, 0.4))) + Diagonal(randn(p))
  W = rand(p, p); W = W + W'
  for f in (NormL1(0.2), NormL1(0.2, W), MCP(0.2), IndBox(-0.3, 0.3), SqrNormL2(2.), IndBox(0.1, 1.))
    s = SymmetricSparseIterate(D)
    @test f(s) ≈ f(D)
    @test Matrix(prox(f, s, 0.5)) ≈ prox(f, D, 0.5)
    prox!(s, f, s, 0.5)                                    # in place
    @test Matrix(s) ≈ prox(f, D, 0.5)
  end
  # on an AtomIterate: the active groups for a function that preserves zeros
  xa = AtomIterate([0., 0., 1., -2., 0., 0.], [1:2, 3:4, 5:6])
  @test NormL1(0.5)(xa) ≈ 1.5
  @test MCP(1.)(xa) ≈ MCP(1.)(Vector(xa))
end

@testset "parameter arrays must fit" begin
  # weights and bounds of the wrong length raise an error instead of reading out of bounds
  for f in (NormL1(1., [1., 2.]), NormL1Nonneg(1., [1., 2.]), ElasticNet(1., 1., [1., 2.]), IndBox([-1., -1.], 1.),
            IndBox(-1., [1., 1.]), LinearFunction([1., 2.]), SqrNormL2(1., [1., 2.]), Tilt(NormL1(1.), [1., 2.]),
            Translate(NormL1(1.), [1., 2.]), 2 * NormL1(1., [1., 2.]))
    x = ones(6)
    @test_throws DimensionMismatch prox(f, x, 0.1)
    @test_throws DimensionMismatch prox_coordinate!(SparseIterate(x), f, 3, 0.1)
    @test_throws DimensionMismatch prox_coordinate!(AtomIterate(x, 3), f, 2, 0.1)
    f isa Union{LinearFunction, SqrNormL2, Tilt, Translate} || @test_throws DimensionMismatch f(x)
  end
  # a direct call of the scalar operator checks the index
  @test_throws BoundsError prox_entry(NormL1(1., [1., 2.]), 1.0, 5, 0.1)
  # on a SymmetricSparseIterate the weights follow the p × p matrix: a vector over the
  # lower triangle, the convention of 0.4, is rejected
  p = 4
  S = SymmetricSparseIterate(Matrix(1.0I, p, p))
  wtri = rand(p * (p + 1) ÷ 2)
  @test_throws DimensionMismatch prox_coordinate!(S, NormL1(0.1, wtri), ncoordinates(S), 0.5)
  @test_throws DimensionMismatch prox(NormL1(0.1, wtri), S, 0.5)
  @test_throws DimensionMismatch NormL1(0.1, wtri)(S)
  @test prox_coordinate!(S, NormL1(0.1, ones(p, p)), ncoordinates(S), 0.5) ≈ 1 - 0.05
end

@testset "public parameter checks before writes" begin
  f = WeightedSquare([1., 2., 3.])
  @test ProximalBase.check_parameters(f, ones(3)) === nothing
  @test ProximalBase.check_parameters(ShiftedAbs(), ones(4)) === nothing
  @test prox(ShiftedAbs(), zeros(4), 0.5) == fill(0.5, 4)
  points = (ones(4), SparseIterate([1., 0., -2., 0.]),
            SymmetricSparseIterate([1. 0.; 0. -2.]),
            AtomIterate([1., 0., -2., 0.], [1:2, 3:4]))
  for x in points
    fs = (f, Scaled(f, 2.), Tilt(f, ones(size(x))), Translate(f, ones(size(x))),
          Regularize(f, 1.), Regularize(f, 1., ones(size(x))),
          Constrained(f, IndBox(-1., 1.)), Constrained(Scaled(f, 2.), IndBox(-1., 1.)))
    for h in fs
      @test_throws DimensionMismatch h(x)
      for step in (0.5, fill(0.5, size(x))), op in (prox!, value_and_prox!), alias in (false, true)
        step isa AbstractArray && h isa Regularize && continue
        input = copy(x)
        output = alias ? input : copy(x)
        before = Array(output)
        stored = output isa Union{SparseIterate, SymmetricSparseIterate, AtomIterate} ? copy(activecoordinates(output)) : nothing
        @test_throws DimensionMismatch op(output, h, input, step)
        @test Array(output) == before
        @test Array(input) == Array(x)
        stored === nothing || @test activecoordinates(output) == stored
      end
      if x isa Union{SparseIterate, SymmetricSparseIterate, AtomIterate}
        for k in (1, ncoordinates(x))
          input = copy(x)
          before = Array(input)
          stored = copy(activecoordinates(input))
          @test_throws DimensionMismatch prox_coordinate!(input, h, k, 0.5)
          @test Array(input) == before
          @test activecoordinates(input) == stored
        end
      end
    end
  end
  # Parameters are checked even when the argument is outside the box.
  @test_throws DimensionMismatch Constrained(f, IndBox(-1., 1.))(fill(2., 4))
  # An empty stored set must not skip the hook.
  for x in (SparseIterate(4), SymmetricSparseIterate(2), AtomIterate(4, [1:2, 3:4]))
    @test_throws DimensionMismatch f(x)
    @test_throws DimensionMismatch prox!(x, f, x, 0.5)
    @test all(iszero, x)
  end
  for T in (Float32, Float64)
    A = Matrix{T}(I, 3, 3)
    h = Precompose(WeightedSquare(ones(2)), A, one(T))
    for f in (Translate(h, ones(T, 3)), Regularize(h, one(T))),
        op in (prox!, value_and_prox!), alias in (false, true)
      input = fill(T(3), 3)
      output = alias ? input : copy(input)
      @test_throws DimensionMismatch op(output, f, input, T(0.5))
      @test output == fill(T(3), 3)
      @test input == fill(T(3), 3)
    end
    @test h.image.buf === nothing
    h = Precompose(WeightedSquare(ones(2)), Matrix{T}(I, 2, 3), one(T))
    @test prox(Scaled(h, T(2)), ones(T, 3), T(0.5)) ≈ T[0.5, 0.5, 1]
    @test h.image.buf isa Vector{T} && length(h.image.buf) == 2
    h = Precompose(WeightedSquare(ones(2)), A, one(T))
    for f in (Translate(h, ones(T, 4)), Regularize(h, one(T)))
      input = fill(T(3), 4)
      @test_throws DimensionMismatch prox!(input, f, input, T(0.5))
      @test input == fill(T(3), 4)
    end
  end
end

@testset "SymmetricSparseIterate into another iterate" begin
  p = 5
  D = Matrix(Symmetric(sprandn(p, p, 0.4))) + I
  s = SymmetricSparseIterate(D)
  y = SymmetricSparseIterate(p)
  y[p, 1] = 3.                     # stored in y only: becomes a stored zero
  prox!(y, NormL1(0.2), s, 1.)
  @test Matrix(y) ≈ prox(NormL1(0.2), D, 1.)
  @test isactive(y, tril_index(p, p, 1)) && y[p, 1] == 0
end

@testset "NormL1Nonneg" begin
  x = [1.0, -0.4, 2.5, 0.1]
  f = NormL1Nonneg(0.5)
  @test prox(f, x) ≈ [0.5, 0., 2., 0.]
  @test f([1., 2.]) == 1.5 && f([1., -1.]) == Inf
  w = rand(4)
  check_prox(NormL1Nonneg(0.3, w), randn(4), 0.9)
end

@testset "ElasticNet" begin
  x = randn(8)
  f = ElasticNet(0.3, 0.5)
  @test f(x) ≈ 0.3 * sum(abs, x) + 0.25 * sum(abs2, x)
  @test prox(f, x, 0.7) ≈ shrink.(x, 0.21) ./ 1.35
  check_prox(f, x, 0.7)
  check_prox(ElasticNet(0.3, 0.5, rand(8)), x, 0.7)
  # the limits: the lasso and ridge
  @test prox(ElasticNet(0.3, 0.), x, 0.7) ≈ prox(NormL1(0.3), x, 0.7)
  @test prox(ElasticNet(0., 0.5), x, 0.7) ≈ prox(SqrNormL2(0.5), x, 0.7)
end

@testset "boxes" begin
  x = 3 .* randn(10)
  l, u = -rand(10), rand(10)
  for f in (IndBox(-1., 2.), IndBox(l, u), IndBallLinf(0.8), IndNonnegative(), IndBox(-Inf, 0.5))
    z = check_prox(f, x, 1.)
    @test f(z) == 0
    @test value_and_prox!(similar(x), f, x, 1.) == 0
  end
  @test prox(IndBox(l, u), x) == clamp.(x, l, u)
  @test prox(IndNonnegative(), x) == max.(x, 0)
  @test IndBallLinf(0.8) isa IndBox
  @test IndBox(0, 1)([0.5, 2.]) == Inf
  @test IndBox(0, 1)([1 + 1e-12, 0.]) == 0                # the tolerance of rounding
  @test IndNonnegative()([-1e-12, 1.]) == 0 && IndNonnegative()([-1e-3, 1.]) == Inf
  @test_throws ArgumentError IndBox(1, 0)
  @test_throws ArgumentError IndBallLinf(-1.)

  @test_throws ArgumentError("the lower bounds must be below Inf and the upper bounds above -Inf") IndBox(Inf, Inf)
  @test_throws ArgumentError("the lower bounds must be below Inf and the upper bounds above -Inf") IndBox([0.0, Inf], [1.0, Inf])
  @test_throws ArgumentError IndBox(-Inf, -Inf)
  @test_throws ArgumentError IndBox(Inf32, Inf32)
  @test_throws ArgumentError IndBox(Inf, Inf32)
  @test_throws ArgumentError IndBox(Float16(Inf), Float16(Inf))
  @test_throws ArgumentError IndBox(big(Inf), big(Inf))
  @test_throws ArgumentError IndBox(1//0, 1//0)
  @test_throws ArgumentError IndBox([-Inf, 0.0], [-Inf, 1.0])
  @test_throws ArgumentError IndBox(-Inf, [Inf, -Inf])
  @test_throws ArgumentError IndBox([0.0, Inf], Inf)
  @test_throws ArgumentError IndBox([0.0 Inf; 0.0 0.0], [1.0 Inf; 1.0 1.0])
  @test_throws ArgumentError IndBox(view([0.0, Inf, 2.0], 1:2), [1.0, Inf])
  @test_throws ArgumentError("the lower bounds must not exceed the upper bounds") IndBox(NaN, 1.0)
  @test_throws ArgumentError("the lower bounds must not exceed the upper bounds") IndBox([0.0, NaN], 1.0)
  @test_throws ArgumentError("the lower bounds must not exceed the upper bounds") IndBox(1, 0)
  @test_throws ArgumentError("the lower bounds must not exceed the upper bounds") IndBox([Inf, 2.0], [Inf, 1.0])

  for (l, u) in ((-Inf, Inf), (0.0, Inf), (-Inf, 0.0), ([-Inf, 0.0], [Inf, Inf]),
                 (0, 0), (-2.0, -2.0), (floatmax(Float64), floatmax(Float64)),
                 (false, true), (-1//2, 1//2), (big(-1.0), big(1.0)))
    f = IndBox(l, u)
    x = [-3.0, 2.0]
    @test prox(f, x) == clamp.(x, l, u)
    @test f(clamp.(x, l, u)) == 0
  end
  @test IndBox(Inf, Float64[])(Float64[]) == 0           # zero coordinates
  @test prox(IndBox(Inf, Float64[]), Float64[]) == Float64[]
  @test IndBallLinf(Inf)([-3.0, 2.0]) == 0
  @test prox(IndBallLinf(Inf), [-3.0, 2.0]) == [-3.0, 2.0]
  @test IndBox(0.0, Inf)([Inf]) == 0                    # infinite entries remain members
  @test IndBox(-Inf, Inf)([Inf]) == 0
end

@testset "restrict of a box drops terms that must be finite at zero" begin
  padded(z, S, n) = (x = zeros(eltype(z), n); x[S] = z; x)
  for T in (Float32, Float64)
    f = IndBox(T[1, 0], T[2, 2])
    @test f(T[0, 1]) == Inf
    @test_throws ArgumentError restrict(f, [2])
    @test_throws ArgumentError restrict(f, 2:2)
    @test_throws ArgumentError restrict(f, BitVector([false, true]))
    @test_throws ArgumentError restrict(f, Int[])
    r = restrict(f, [1, 2])                            # nothing is dropped
    @test r.l == f.l && r.u == f.u
    @test restrict(f, [1])(T[1.5]) == f(T[1.5, 0]) == 0
    @test_throws ArgumentError restrict(IndBox(T[-2, 0], T[-1, 2]), [2])
    @test_throws ArgumentError restrict(IndBox(T(1), T(2)), [1, 2])
    @test restrict(IndBox(T(-1), T(2)), [1]) == IndBox(T(-1), T(2))
  end
  @test_throws ArgumentError restrict(IndBox(1, [2, 2]), [2])
  @test restrict(IndBox(1, [2, 2]), [1, 2])([1.5, 1.5]) == 0
  # every excluded coordinate contains zero, with a bound at zero or infinite bounds
  for f in (IndBox([0., -1., -Inf], [1., 0., Inf]), IndBox(-1., 2.), IndBox(-Inf, 0.)),
      S in ([1], [2], [3], [1, 3])
    for v in (-0.5, 0.5, 3.)
      z = fill(v, length(S))
      @test restrict(f, S)(z) == f(padded(z, S, 3))
    end
  end
  # the excluded coordinates contain zero, however S selects the rest
  f = IndBox([0., 1.], [2., 2.])
  for S in ([2], 2:2, [2, 2], BitVector([false, true]))
    r = restrict(f, S)
    @test r.l == f.l[S] && r.u == f.u[S]
  end
  M = IndBox([1. 0.; 2. 0.], [2. 1.; 3. 1.])        # zero lies outside the first column only
  @test restrict(M, [1, 2])([1.5, 2.5]) == 0
  @test restrict(M, [1, 2, 3])([1.5, 2.5, 2.]) == Inf
  @test_throws ArgumentError restrict(M, [2, 3])
  # wrappers pass the check on
  box = IndBox([1., 0.], [2., 2.])
  for g in (Scaled(box, 2.), Tilt(box, [1., 1.]), Regularize(box, 1.), Precompose(box, [1., 2.]),
            Precompose(box, 2.), Constrained(NormL1(1.), box))
    @test_throws ArgumentError restrict(g, [2])
    @test all(v -> restrict(g, [1])([v]) ≈ g([v, 0.]), (0.75, 1.5, 3.))
  end
end

@testset "NormL0" begin
  x = [0.5, -2., 0.05, 1.2]
  f = NormL0(0.5)
  @test f(x) == 2.
  @test prox(f, x, 1.) == [0., -2., 0., 1.2]       # kept when x² > 2γλ = 1
  for v in (-3., -1.1, -0.9, 0.3, 1.0, 2.5), γ in (0.3, 1., 4.)
    t, _ = brute_force_prox(u -> 0.5 * (u != 0), v, γ)
    @test prox_entry(f, v, 1, γ) ≈ t atol = 1e-6
  end
  check_prox(f, randn(10), 0.8)
end

@testset "MCP" begin
  λ, a = 0.6, 2.5
  f = MCP(λ, a)
  pen(t) = abs(t) <= a * λ ? λ * abs(t) - t^2 / (2a) : a * λ^2 / 2
  x = 3 .* randn(20)
  @test f(x) ≈ sum(pen, x)
  # the proximal operator is the global minimizer, also for steps γ ≥ a that make the
  # problem nonconvex
  for v in (-3., -1.5, -0.7, -0.2, 0., 0.35, 0.65, 1.0, 1.5, 2.2, 4.), γ in (0.3, 1., 2.4, 2.5, 3., 10.)
    t, ht = brute_force_prox(pen, v, γ)
    z = prox_entry(f, v, 1, γ)
    @test pen(z) + (z - v)^2 / (2γ) <= ht + 1e-9
  end
  # firm thresholding for γ < a
  @test prox_entry(f, 0.5, 1, 1.) == 0.
  @test prox_entry(f, 1.0, 1, 1.) ≈ (1.0 - 0.6) / (1 - 1 / 2.5)
  @test prox_entry(f, 2.0, 1, 1.) == 2.0
  check_prox(f, x, 0.8)
  @test_throws ArgumentError MCP(0.5, 0.)
end

@testset "SCAD" begin
  λ, a = 0.6, 3.7
  f = SCAD(λ, a)
  function pen(t)
    s = abs(t)
    s <= λ ? λ * s : s <= a * λ ? (2a * λ * s - s^2 - λ^2) / (2 * (a - 1)) : λ^2 * (a + 1) / 2
  end
  x = 3 .* randn(20)
  @test f(x) ≈ sum(pen, x)
  for v in (-4., -2., -1.1, -0.7, 0., 0.3, 0.9, 1.3, 2.1, 3., 5.), γ in (0.3, 1., 2.6, 2.7, 3., 10.)
    t, ht = brute_force_prox(pen, v, γ)
    z = prox_entry(f, v, 1, γ)
    @test pen(z) + (z - v)^2 / (2γ) <= ht + 1e-9
  end
  # the SCAD thresholding rule for γ = 1: soft thresholding up to 2λ
  @test prox_entry(f, 1.0, 1, 1.) ≈ 0.4
  @test prox_entry(f, 1.5, 1, 1.) ≈ ((a - 1) * 1.5 - a * λ) / (a - 2)
  @test prox_entry(f, 3.0, 1, 1.) == 3.0
  check_prox(f, x, 0.8)
  @test_throws ArgumentError SCAD(0.5, 2.)
end

@testset "a separable function of your own" begin
  p = 4
  W = rand(p, p) .+ 0.5
  w = vec(W + W')                         # symmetric, as weights on a SymmetricSparseIterate must be
  f = WeightedSquare(w)
  check_traits(f; proximable = true, separable = true, convex = true)
  @test iszeropreserving(f) && !iszeropreserving(ShiftedAbs())
  @test iszeropreserving(IndBox(-1., 1.)) && !iszeropreserving(IndBox(1., 2.))   # a property of the instance
  # the value and the proximal operator on a vector, from value_entry and prox_entry
  x = randn(p * p)
  @test f(x) ≈ sum(w .* x .^ 2) / 2
  check_prox(f, x, 0.7)
  @test prox(f, x, 0.7) ≈ x ./ (1 .+ 0.7 .* w)
  # on the three iterates, as on the dense arrays they hold
  for g in (f, ShiftedAbs())
    xs = SparseIterate(sprandn(p * p, 0.3))
    @test g(xs) ≈ g(Vector(xs))
    @test Vector(prox!(SparseIterate(p * p), g, xs, 0.7)) ≈ prox(g, Vector(xs), 0.7)
    D = randn(p, p); D = D + D'
    S = SymmetricSparseIterate(D)
    @test g(S) ≈ g(Matrix(S))
    @test Matrix(prox!(SymmetricSparseIterate(p), g, S, 0.7)) ≈ prox(g, Matrix(S), 0.7)
    xa = AtomIterate(randn(p * p), p)
    @test g(xa) ≈ g(Vector(xa))
  end
  # coordinate steps
  xs = SparseIterate(p * p)
  xs[3] = 2.
  @test prox_coordinate!(xs, f, 3, 0.5) ≈ 2 / (1 + 0.5 * w[3])
  prox_coordinate!(xs, f, 5, 0.5)
  @test nnz(xs) == 1                      # a zero stays zero, and unstored
  S = SymmetricSparseIterate(p)
  S[3, 2] = 1.
  k = tril_index(p, 3, 2)
  @test prox_coordinate!(S, f, k, 0.5) ≈ 1 / (1 + 0.5 * w[(2 - 1) * p + 3])
  # zero-preserving functions touch only the stored entries of a sparse iterate
  xs = SparseIterate(sprandn(p * p, 0.3))
  y = SparseIterate(p * p)
  prox!(y, f, xs, 0.7)
  @test nnz(y) == nnz(xs)
  prox!(y, ShiftedAbs(), xs, 0.7)
  @test nnz(y) > nnz(xs)                  # the prox of |x - 1| moves zeros
end

end
