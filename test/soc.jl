module SOCTest

using Test
using ProximalBase
using LinearAlgebra
using Random

using ..TestUtils

const PB = ProximalBase
const PT = ProximalBase.Testing
const rng = MersenneTwister(169)

# For a fixed head a, the nearest tail lies in the ball of radius a. Bisect
# the derivative of (a-t)^2 + max(s-a, 0)^2, to relative width 2^-200.
function reference(v)
  setprecision(BigFloat, 256) do
    x = BigFloat.(v)
    t, s = x[1], norm(x[2:end])
    lo, hi = zero(t), max(abs(t), s)
    if -t - s >= 0
      a = zero(t)
    else
      tol = ldexp(hi, -200)
      for _ in 1:210
        hi - lo <= tol && break
        a = (lo + hi) / 2
        if a - t - max(s - a, zero(t)) > 0
          hi = a
        else
          lo = a
        end
      end
      a = (lo + hi) / 2
    end
    p = copy(x)
    p[1] = a
    s > a && (p[2:end] .*= a / s)
    p
  end
end

function check_reference(v; rtol = eltype(v) === Float32 ? 1e-5 : 1e-12)
  p, ref = prox(IndSOC(), v), reference(v)
  scale = maximum(abs, BigFloat.(v))
  allowance = BigFloat(rtol) * scale + 4BigFloat(nextfloat(zero(eltype(v))))
  @test all(isfinite, p)
  @test maximum(abs.(BigFloat.(p) .- ref)) <= allowance
  p
end

# Normalize each cone before forming products; retain an absolute allowance
# of four subnormal units for rounded outputs.
function check_moreau(v::AbstractVector{T}, p, q) where {T}
  scale = maximum(abs, BigFloat.(v))
  iszero(scale) && (scale = one(scale))
  u, z, w = BigFloat.(v) ./ scale, BigFloat.(p) ./ scale, BigFloat.(q) ./ scale
  tol = BigFloat(T === Float32 ? 1e-5 : 1e-12)
  δ = 4BigFloat(nextfloat(zero(T))) / scale
  @test all(isfinite, p) && all(isfinite, q)
  @test max(norm(z[2:end]) - z[1], 0) <= tol + length(v) * δ
  @test max(norm(w[2:end]) + w[1], 0) <= tol + length(v) * δ
  @test maximum(abs.(u .- z .- w)) <= tol + δ
  @test abs(dot(z, w)) <= tol + 2length(v) * δ
end

@testset "IndSOC branches and Moreau characterization" begin
  f = IndSOC()
  for T in (Float64, Float32)
    cases = [T[5, 3, 4], T[6, 3, 4], T[-5, 3, 4], T[-6, 3, 4],
             T[0, 3, 4], T[2, 3, 4], T[0, 0, 0], T[-2, 0, 0],
             T[2, 0, 0], T[0], T[-3], T[3]]
    append!(cases, [randn(rng, T, n) for n in 1:7 for _ in 1:5])
    for v in cases
      p = check_reference(v)
      q = prox(Conjugate(f), v)
      check_moreau(v, p, q)
      @test f(p) === zero(T)
      @test Conjugate(f)(q) === zero(T)
      @test q ≈ -prox(f, -v) rtol = T === Float32 ? 1e-5 : 1e-12
      y = copy(v)
      @test prox!(y, f, y, 0.3) === y
      @test y == p
      @test value_and_prox!(similar(v), f, v, 2.0) === zero(T)
    end
    @test prox(f, T[0, 3, 4]) ≈ T[2.5, 1.5, 2]
    @test prox(f, T[-2]) == T[0]
    tol = PB._tol(T)
    @test f(T[-tol / 2, 0]) == 0
    @test isinf(f(T[-2tol, 0]))
    @test f(T[1, 1 + tol / 2]) == 0
    @test isinf(f(T[1, 1 + 2tol]))
  end
end

@testset "IndSOC one-dimensional reference" begin
  for n in 1:5, _ in 1:8
    v = randn(rng, n)
    p, ref = prox(IndSOC(), v), reference(v)
    @test maximum(abs.(BigFloat.(p) .- ref)) <= 1e-8
  end
end

@testset "IndSOC products and linear indices" begin
  groups = [[3, 1, 2], [5, 4], [7]]
  f = IndSOC(groups)
  v = [3.0, 4, 0, -2, 0, 13, -1, 17]
  p = prox(f, v)
  @test p ≈ [1.5, 2, 2.5, -1, 1, 13, 0, 17]
  @test f(p) == 0
  q = prox(Conjugate(f), v)
  @test q[[6, 8]] == zeros(2)
  @test Conjugate(f)(q) == 0
  @test v ≈ p + q rtol = 1e-12
  for G in groups
    check_moreau(v[G], p[G], q[G])
    @test p[G] ≈ prox(IndSOC(), v[G]) rtol = 1e-12
  end
  @test prox(f, reshape(v, 2, 4)) == reshape(p, 2, 4)
  @test f(reshape(p, 2, 4)) == 0
  @test prox(IndSOC(), reshape(v, 2, 4)) == reshape(prox(IndSOC(), v), 2, 4)
  @test prox(IndSOC([3:-1:1]), [3.0, 4, 0]) ≈ [1.5, 2, 2.5]

  full = IndSOC([1:3, 4:6])
  w = randn(rng, 6)
  @test w ≈ prox(full, w) - prox(full, -w) rtol = 1e-12
  @test prox(Conjugate(full), w) ≈ -prox(full, -w) rtol = 1e-12

  empty = IndSOC(UnitRange{Int}[])
  @test empty(v) == 0
  @test prox(empty, v) == v
  @test empty(Float64[]) == 0
  @test prox(empty, Float64[]) == Float64[]
  @test Conjugate(empty)(zeros(8)) == 0
  @test isinf(Conjugate(empty)(v))
  @test prox(Conjugate(empty), v) == zeros(8)

  polar = Conjugate(IndSOC([[3, 1, 2]]))
  y = [3.0, 4, -5, 0, 0]
  @test polar(y) == 0
  @test isinf(polar(-y))
  for i in (4, 5)
    z = copy(y)
    z[i] = 1
    @test isinf(polar(z))
    z[i] = PB._tol(Float64) / 2
    @test polar(z) == 0
  end
  gap = Conjugate(IndSOC([2:3]))
  @test gap([0.0, -2, 1, 0]) == 0
  @test isinf(gap([1.0, -2, 1, 0]))
  @test isinf(gap([0.0, -2, 1, 1]))

  groups[1][1] = 6
  push!(groups, [8])
  @test f.groups == [[3, 1, 2], [5, 4], [7]]
  ranges = [1:2, 4:5]
  owned = IndSOC(ranges)
  ranges[1] = 2:3
  @test owned.groups == [1:2, 4:5]
end

@testset "IndSOC extreme magnitudes" begin
  for T in (Float64, Float32)
    M, q = floatmax(T), nextfloat(zero(T))
    large, small = T === Float64 ? (T(1e200), T(1e-200)) : (T(1e30), T(1e-30))
    cases = [T[0, M, M], T[0, M, M, small], T[q, 5q],
             T[0, q, q], T[-q, 5q], T[0, large, small],
             T[small, 3small, 4small], T[-large / 2, large],
             T[M, M], T[-M, M], T[M / 2, M], T[-M / 2, M],
             T[-prevfloat(2floatmin(T)), 2floatmin(T)],
             T[-prevfloat(large), large], T[-nextfloat(large), large],
             T[-prevfloat(small), small], T[-nextfloat(small), small]]
    for v in cases
      p = check_reference(v)
      c = Conjugate(IndSOC())
      w = prox(c, v)
      check_moreau(v, p, w)
      @test IndSOC()(p) == 0
      @test c(w) == 0
      @test value_and_prox!(similar(v), c, v, one(T)) === zero(T)
    end
    p = prox(IndSOC(), T[0, M, M])
    @test p[1] / M ≈ inv(sqrt(T(2))) rtol = 4eps(T)
    @test p[2:3] == T[M / 2, M / 2]
    p = prox(IndSOC(), T[0, M, M, small])
    @test p[4] == small / 2
    @test prox(IndSOC(), T[q, 5q]) == T[3q, 3q]
    check_reference(T[-M, M, M, M, M])
    @test IndSOC()(T[M, M]) == 0
    @test isinf(IndSOC()(T[M, M, M]))
    @test isinf(IndSOC()(T[-M, M]))
    @test Conjugate(IndSOC())(T[-M, M]) == 0
    @test isinf(Conjugate(IndSOC())(T[-M, M, M]))
  end
end

struct ZeroBased{T} <: AbstractVector{T}
  data::Vector{T}
end
Base.size(x::ZeroBased) = size(x.data)
Base.axes(x::ZeroBased) = (0:length(x.data)-1,)
Base.getindex(x::ZeroBased, i::Int) = x.data[i + 1]

@testset "IndSOC contracts and validation before writes" begin
  f = IndSOC()
  @test_throws ArgumentError f(Float64[])
  @test_throws ArgumentError prox(f, Float64[])
  @test_throws ArgumentError Conjugate(f)(Float64[])
  for groups in ([Int[]], [1:0], [[1, 1]], [[0, 1]], [[-1, 2]], [1:3, 3:5])
    @test_throws ArgumentError IndSOC(groups)
  end
  bad = IndSOC([1:2, 5:7])
  @test_throws BoundsError bad(ones(6))
  @test_throws BoundsError Conjugate(bad)(ones(6))
  for g in (bad, Scaled(bad, 2.0), Translate(bad, ones(6)),
            Tilt(bad, ones(6)), Conjugate(bad)), op in (prox!, value_and_prox!)
    for alias in (false, true)
      x = [-1.0, 5, 7, 8, 0, 9]
      y = alias ? x : fill(13.0, 6)
      input, before = copy(x), copy(y)
      @test_throws BoundsError op(y, g, x, 1.0)
      @test x == input
      @test y == before
    end
  end
  for op in (prox!, value_and_prox!)
    x, y = [0.0, 3, 4], fill(17.0, 1, 3)
    @test_throws DimensionMismatch op(y, f, x, 1.0)
    @test y == fill(17.0, 1, 3)
    @test_throws MethodError op(similar(x), f, x, ones(3))
    y = fill(17.0, 3)
    @test_throws ArgumentError("IndSOC requires real entries") op(y, f, complex.(x), 1.0)
    @test y == fill(17.0, 3)
    @test_throws ArgumentError op(y, f, ZeroBased(x), 1.0)
    @test y == fill(17.0, 3)
    @test_throws ArgumentError("IndSOC requires real entries") op(complex.(y), f, x, 1.0)
    @test_throws ArgumentError op(ZeroBased(y), f, x, 1.0)
    x = ([3.0, 4], ones(6))
    y = deepcopy(x)
    @test_throws BoundsError op(y, SeparableSum(NormL1(), bad), x, 1.0)
    @test y == x
  end
  @test_throws ArgumentError("IndSOC requires real entries") f(ComplexF64[0, 1])
  @test_throws ArgumentError f(ZeroBased([0.0, 1]))

  for T in (Float64, Float32), nonfinite in (T(NaN), T(Inf), T(-Inf)), i in 1:3
    x = T[0, 3, 4]
    x[i] = nonfinite
    @test isinf(f(x))
    @test isinf(Conjugate(f)(x))
    @test all(isnan, prox(f, x))
    y = copy(x)
    prox!(y, f, y, one(T))
    @test all(isnan, y)
  end
  free = IndSOC([1:2])
  @test isinf(free([1.0, 0, Inf]))
  @test isinf(Conjugate(free)([-1.0, 0, NaN]))
  @test prox(free, [1.0, 0, Inf]) == [1.0, 0, Inf]
  @test isinf(IndSOC(UnitRange{Int}[])([Inf]))

  m = typemin(Int)
  @test isinf(f([m, 0]))
  @test prox(f, [m, 0]) == zeros(2)
  @test prox(f, [0, m]) == [abs(float(m)) / 2, float(m) / 2]
  @test Conjugate(f)([m, 0]) == 0
  @test isinf(Conjugate(f)([typemax(Int), 0]))
end

@testset "IndSOC precision and generic helpers" begin
  for f in (IndSOC(), IndSOC([1:3]), IndSOC([[3, 1, 2]]))
    @test f isa AbstractFunction
    @test !(f isa PB.AbstractGroupPenalty)
    check_traits(f; proximable = true, convex = true)
    @test PB._isindicator(f)
    x = [0.0, 3, 4]
    check_prox(f, x, 0.7; rng)
    PT.test_conjugate(f, x, 0.7; conjugate = Conjugate(f))
    for T in (Float32, Float64)
      v, y = T.(x), similar(x, T)
      @test f(v) isa T
      @test prox(f, v) isa Vector{T}
      @test value_and_prox!(y, f, v, 0.7) isa T
      @test Conjugate(f)(v) isa T
      @test value_and_prox!(y, Conjugate(f), v, 0.7) isa T
      for S in (Float32, Float64)
        z = similar(x, S)
        @test prox!(z, f, v, 0.7) === z
        @test z ≈ S.(prox(f, Float64.(v))) rtol = 1e-5
        @test value_and_prox!(z, f, v, 0.7) isa S
        @test value_and_prox!(z, Conjugate(f), v, 0.7) isa S
      end
    end
  end
  grouped = IndSOC([1:3, 5:6])
  x = randn(rng, 7)
  check_prox(grouped, x, 0.8; rng)
  PT.test_conjugate(grouped, x, 0.8; conjugate = Conjugate(grouped))
  # A narrower output must not turn a finite input into infinities before
  # deciding that the projection is zero.
  y = zeros(Float32, 2)
  @test prox!(y, IndSOC(), [-1e100, 1e100], 1.0) == zeros(Float32, 2)
end

function fresh_allocations(op::O, f::F, x::X) where {O, F, X}
  y = copy(x)
  op(y, f, y, 1)
  copyto!(y, x)
  @allocated op(y, f, y, 1)
  copyto!(y, x)
  @allocated op(y, f, y, 1)
end

function check_allocations(f::F, x::X) where {F, X}
  y = similar(x)
  for g in (f, Conjugate(f))
    @test allocs(g, x) == 0
    for op in (prox!, value_and_prox!)
      @test allocs(op, y, g, x, 1) == 0
      @test fresh_allocations(op, g, x) == 0
    end
  end
end

@testset "IndSOC allocations on fresh branches" begin
  for T in (Float64, Float32)
    M, q = floatmax(T), nextfloat(zero(T))
    cases = [T[5, 3, 4], T[-5, 3, 4], T[0, 3, 4], T[0, 0, 0],
             T[-1, 0, 0], T[0, M, M], T[q, 5q, 0]]
    for v in cases
      for f in (IndSOC(), IndSOC([1:3, 5:7]), IndSOC([[3, 1, 2], [7, 5, 6]]))
        if f.groups === nothing
          x = v
        else
          x = fill(T(7), 8)
          for G in f.groups
            x[G] = v
          end
        end
        check_allocations(f, x)
        check_allocations(f, reshape(copy(x), 1, length(x)))
      end
    end
  end
end

end
