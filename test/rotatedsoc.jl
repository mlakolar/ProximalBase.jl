module RotatedSOCTest

using Test
using ProximalBase
using LinearAlgebra
using Random

using ..TestUtils

const PB = ProximalBase
const PT = ProximalBase.Testing
const rng = MersenneTwister(170)

function rotation(::Type{T}, n) where {T}
  Q = Matrix{T}(I, n, n)
  Q[1:2, 1:2] = T[1 1; 1 -1] / sqrt(T(2))
  Q
end

# In rotated coordinates, fix the SOC head h and minimize over the tail ball.
# Bisect the derivative to relative width 2^-(bits-64). At extreme scales,
# 4096 bits retain the small head through the rotation and its inverse.
function reference(v; bits = 256)
  setprecision(BigFloat, bits) do
    x = BigFloat.(v)
    root2 = sqrt(BigFloat(2))
    t = (x[1] + x[2]) / root2
    w = vcat((x[1] - x[2]) / root2, x[3:end])
    s = norm(w)
    lo, hi = zero(t), max(abs(t), s)
    if s <= t
      return x
    elseif s <= -t
      return zero.(x)
    end
    tol = ldexp(hi, -(bits - 64))
    for _ in 1:bits+8
      hi - lo <= tol && break
      h = (lo + hi) / 2
      if h - t - max(s - h, zero(t)) > 0
        hi = h
      else
        lo = h
      end
    end
    h = (lo + hi) / 2
    w .*= h / s
    vcat((h + w[1]) / root2, (h - w[1]) / root2, w[2:end])
  end
end

function check_reference(v; bits = 256)
  T = eltype(v)
  p, ref = prox(IndRotatedSOC(), v), reference(v; bits)
  scale = maximum(abs, BigFloat.(v))
  rtol = BigFloat(T === Float32 ? 1e-5 : 1e-12)
  atol = 4BigFloat(nextfloat(zero(T)))
  @test all(isfinite, p)
  @test maximum(abs.(BigFloat.(p) .- ref)) <= rtol * scale + atol
  p
end

# Check primal and polar feasibility independently of the value oracles.
# Normalization avoids overflowing products; four subnormal units account
# for the absolute rounding error of representable outputs.
function check_moreau(v::AbstractVector{T}, p, q) where {T}
  scale = maximum(abs, BigFloat.(v))
  iszero(scale) && (scale = one(scale))
  x, a, b = BigFloat.(v) ./ scale, BigFloat.(p) ./ scale, BigFloat.(q) ./ scale
  tol = BigFloat(T === Float32 ? 1e-5 : 1e-12)
  δ = 4BigFloat(nextfloat(zero(T))) / scale
  @test all(isfinite, p) && all(isfinite, q)
  @test min(a[1], a[2]) >= -tol - δ
  @test max(b[1], b[2]) <= tol + δ
  @test sum(abs2, a[3:end]) - 2a[1] * a[2] <= tol + 4length(v) * δ
  @test sum(abs2, b[3:end]) - 2b[1] * b[2] <= tol + 4length(v) * δ
  @test maximum(abs.(x .- a .- b)) <= tol + δ
  @test abs(dot(a, b)) <= tol + 2length(v) * δ
end

@testset "IndRotatedSOC branches and orthogonal composition" begin
  f = IndRotatedSOC()
  for T in (Float64, Float32)
    cases = [T[2, 4, 4], T[3, 4, 4], T[-2, -4, 4], T[-3, -4, 4],
             T[0, 0, 4], T[0, 2, 4], T[2, 0, 4], T[-2, 3, 4],
             T[0, 0, 0], T[-2, 3, 0], T[2, -3, 0], T[-2, -3, 0],
             T[0, 0], T[-2, 3], T[2, -3], T[-2, -3], T[2, 3]]
    append!(cases, [randn(rng, T, n) for n in 2:8 for _ in 1:8])
    rtol = T === Float32 ? 1e-5 : 1e-12
    for v in cases
      p = check_reference(v)
      q = prox(Conjugate(f), v)
      h = Precompose(IndSOC(), rotation(T, length(v)), one(T))
      @test p ≈ prox(h, v) rtol = rtol atol = rtol
      check_moreau(v, p, q)
      @test v ≈ p - prox(f, -v) rtol = rtol atol = rtol
      @test q ≈ -prox(f, -v) rtol = rtol atol = rtol
      @test f(p) === zero(T)
      @test Conjugate(f)(q) === zero(T)
      y = copy(v)
      @test prox!(y, f, y, 0.3) === y
      @test y == p
      @test value_and_prox!(similar(v), f, v, 2.0) === zero(T)
    end
    @test prox(f, T[0, 0, 4]) ≈ T[sqrt(T(2)), sqrt(T(2)), 2]
    tol = PB._tol(T)
    @test f(T[-tol / 4, 1, 0]) == 0
    @test isinf(f(T[-2tol, 1, 0]))
    @test f(T[1, 1, sqrt(T(2))]) == 0
    @test f(T[1, 1, sqrt(T(2)) * (1 + tol / 4)]) == 0
    @test isinf(f(T[1, 1, sqrt(T(2)) * (1 + 2tol)]))
    @test f(T[-tol / 4, 0, 0]) == 0
    @test isinf(f(T[-2tol, 0, 0]))
  end
end

@testset "IndRotatedSOC products and linear indices" begin
  groups = [[3, 1, 2], [5, 4], [7, 8]]
  f = IndRotatedSOC(groups)
  v = [0.0, 4, 0, -2, 3, 13, -1, 2, 17]
  p = prox(f, v)
  @test p ≈ [sqrt(2), 2, sqrt(2), 0, 3, 13, 0, 2, 17]
  @test f(p) == 0
  q = prox(Conjugate(f), v)
  @test q[[6, 9]] == zeros(2)
  @test Conjugate(f)(q) == 0
  @test v ≈ p + q rtol = 1e-12
  @test abs(dot(p, q)) <= 1e-12
  for G in groups
    check_moreau(v[G], p[G], q[G])
    @test p[G] ≈ prox(IndRotatedSOC(), v[G]) rtol = 1e-12
  end
  @test prox(f, reshape(v, 3, 3)) == reshape(p, 3, 3)
  @test f(reshape(p, 3, 3)) == 0
  @test prox(IndRotatedSOC(), reshape(v, 3, 3)) == reshape(prox(IndRotatedSOC(), v), 3, 3)
  @test prox(IndRotatedSOC([3:-1:1]), [4.0, 0, 0]) ≈ [2, sqrt(2), sqrt(2)]
  y = copy(v)
  @test prox!(y, f, y, 1) == p

  full = IndRotatedSOC([1:3, 4:6])
  w = randn(rng, 6)
  @test w ≈ prox(full, w) - prox(full, -w) rtol = 1e-12
  @test prox(Conjugate(full), w) ≈ -prox(full, -w) rtol = 1e-12

  empty = IndRotatedSOC(UnitRange{Int}[])
  @test empty(v) == 0
  @test prox(empty, v) == v
  @test empty(Float64[]) == 0
  @test prox(empty, Float64[]) == Float64[]
  @test Conjugate(empty)(zeros(9)) == 0
  @test isinf(Conjugate(empty)(v))
  @test prox(Conjugate(empty), v) == zeros(9)

  polar = Conjugate(IndRotatedSOC([[3, 1, 2]]))
  y = [-2.0, 4, -4, 0, 0]
  @test polar(y) == 0
  @test isinf(polar(-y))
  for i in (4, 5)
    z = copy(y)
    z[i] = 1
    @test isinf(polar(z))
    z[i] = PB._tol(Float64) / 2
    @test polar(z) == 0
  end
  gap = Conjugate(IndRotatedSOC([2:3]))
  @test gap([0.0, -2, -1, 0]) == 0
  @test isinf(gap([1.0, -2, -1, 0]))
  @test isinf(gap([0.0, -2, -1, 1]))

  groups[1][1] = 6
  push!(groups, [9, 10])
  @test f.groups == [[3, 1, 2], [5, 4], [7, 8]]
  ranges = [1:2, 4:5]
  owned = IndRotatedSOC(ranges)
  ranges[1] = 2:3
  @test owned.groups == [1:2, 4:5]
end

@testset "IndRotatedSOC extreme magnitudes" begin
  f = IndRotatedSOC()
  for T in (Float64, Float32)
    M, q = floatmax(T), nextfloat(zero(T))
    large, small = T === Float64 ? (T(1e200), T(1e-200)) : (T(1e30), T(1e-30))
    cases = [T[M, M, 0], T[M, M, M, M], T[0, 0, M, M],
             T[0, 0, M, M, small], T[-M / 2, M / 2, M], T[M / 2, -M / 2, M],
             T[q, -q, 0], T[q, q, q], T[-q, q, q], T[0, 0, q, q],
             T[small, small, 3small], T[large, -large / 2, large],
             T[-large, small, small], T[small, -large, small],
             vcat(T[-M / 4, -M / 4], fill(M, 4))]
    @testset "$T case $j" for (j, v) in enumerate(cases)
      p = check_reference(v; bits = 4096)
      check_moreau(v, p, -prox(f, -v))
      @test f(p) == 0
      # The generic conjugate subtracts projections; allow its usual loss
      # of very small components, and check its defining characterization.
      c = Conjugate(f)
      w = prox(c, v)
      check_moreau(v, p, w)
      @test c(w) == 0
      @test value_and_prox!(similar(v), c, v, one(T)) === zero(T)
    end
    @test prox(f, T[M, M, 0]) == T[M, M, 0]
    @test prox(f, T[q, -q, 0]) == T[q, 0, 0]
    p = prox(f, T[0, 0, M, M, small])
    @test p[end] == small / 2
    @test f(T[M, M, M, M]) == 0
    @test isinf(f(T[M, M, M, M, M]))

    a = ldexp(one(T), T === Float64 ? 60 : 30)
    for v in (T[-a, 1, 0], T[1, -a, 0])
      @test prox(f, v) == max.(v, zero(T))
    end
    p = check_reference(T[-a, 1, 1]; bits = 4096)
    @test p == T[inv(2a^2), 1, inv(a)]
    @test prox(f, T[1, -a, 1]) == p[[2, 1, 3]]
    @test prox(f, T[-a, 0, 1])[2] ≈ inv(2a) rtol = 4eps(T)
  end

  a = ldexp(1.0, 1000)
  p = check_reference([-a, ldexp(1.0, -1000), ldexp(1.0, 400)]; bits = 4096)
  @test p == [0, ldexp(1.0, -201), ldexp(1.0, -801)]
  p = check_reference([-a, ldexp(1.0, -1000), ldexp(1.0, -500)]; bits = 4096)
  @test p == [0, ldexp(1.0, -1000), 0]
end

struct ZeroBased{T} <: AbstractVector{T}
  data::Vector{T}
end
Base.size(x::ZeroBased) = size(x.data)
Base.axes(x::ZeroBased) = (0:length(x.data)-1,)
Base.getindex(x::ZeroBased, i::Int) = x.data[i + 1]

@testset "IndRotatedSOC contracts and validation before writes" begin
  f = IndRotatedSOC()
  for v in (Float64[], [1.0])
    @test_throws ArgumentError f(v)
    @test_throws ArgumentError prox(f, v)
    @test_throws ArgumentError Conjugate(f)(v)
    for op in (prox!, value_and_prox!)
      y = fill(17.0, length(v))
      @test_throws ArgumentError op(y, f, v, 1.0)
      @test y == fill(17.0, length(v))
    end
  end
  for groups in ([Int[]], [1:0], [[1]], [1:1], [[1, 1]], [[0, 1]], [[-1, 2]], [1:3, 3:5])
    @test_throws ArgumentError IndRotatedSOC(groups)
  end
  bad = IndRotatedSOC([1:2, 5:7])
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
    x, y = [0.0, 0, 4], fill(17.0, 1, 3)
    @test_throws DimensionMismatch op(y, f, x, 1.0)
    @test y == fill(17.0, 1, 3)
    @test_throws MethodError op(similar(x), f, x, ones(3))
    y = fill(17.0, 3)
    @test_throws ArgumentError op(y, f, complex.(x), 1.0)
    @test y == fill(17.0, 3)
    @test_throws ArgumentError op(y, f, ZeroBased(x), 1.0)
    @test y == fill(17.0, 3)
    @test_throws ArgumentError op(complex.(y), f, x, 1.0)
    @test_throws ArgumentError op(ZeroBased(y), f, x, 1.0)
    x = ([3.0, 4], ones(6))
    y = deepcopy(x)
    @test_throws BoundsError op(y, SeparableSum(NormL1(), bad), x, 1.0)
    @test y == x
  end
  @test_throws ArgumentError f(ComplexF64[0, 1])
  @test_throws ArgumentError f(ZeroBased([0.0, 1]))

  for T in (Float64, Float32), nonfinite in (T(NaN), T(Inf), T(-Inf)), i in 1:3
    x = T[0, 0, 4]
    x[i] = nonfinite
    @test isinf(f(x))
    @test isinf(Conjugate(f)(x))
    @test all(isnan, prox(f, x))
    y = copy(x)
    prox!(y, f, y, one(T))
    @test all(isnan, y)
  end
  free = IndRotatedSOC([1:2])
  @test isinf(free([1.0, 0, Inf]))
  @test isinf(Conjugate(free)([-1.0, 0, NaN]))
  @test prox(free, [1.0, 0, Inf]) == [1.0, 0, Inf]
  @test isinf(IndRotatedSOC(UnitRange{Int}[])([Inf]))
  @test prox(IndRotatedSOC([1:3]), [NaN, 0.0, 1, 7])[4] == 7

  m = typemin(Int)
  @test isinf(f([m, 0]))
  @test prox(f, [m, 0]) == zeros(2)
  @test f([typemax(Int), typemax(Int), m]) == 0
  @test Conjugate(f)([m, m, m]) == 0
  @test isinf(Conjugate(f)([typemax(Int), 0]))
  @test prox(f, [0, 0, m]) ≈ [abs(float(m)) / (2sqrt(2)), abs(float(m)) / (2sqrt(2)), float(m) / 2]
end

@testset "IndRotatedSOC precision and generic helpers" begin
  for f in (IndRotatedSOC(), IndRotatedSOC([1:3]), IndRotatedSOC([[3, 1, 2]]))
    @test f isa AbstractFunction
    @test !(f isa PB.AbstractGroupPenalty)
    check_traits(f; proximable = true, convex = true)
    @test PB._isindicator(f)
    @test Scaled(f, 0) === f
    x = [0.0, 2, 4]
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
  grouped = IndRotatedSOC([1:3, 5:6])
  x = randn(rng, 7)
  check_prox(grouped, x, 0.8; rng)
  PT.test_conjugate(grouped, x, 0.8; conjugate = Conjugate(grouped))
  y = zeros(Float32, 3)
  @test prox!(y, IndRotatedSOC(), [-1e100, -1e100, 1e100], 1.0) == zeros(Float32, 3)
end

@testset "IndRotatedSOC perspective and Berhu" begin
  λ0, λ2 = 3.0, 2.0
  M = sqrt(λ0 / λ2)
  f = Berhu(2sqrt(λ0 * λ2), M)
  cone = IndRotatedSOC()
  # The derivative λ0 - λ2 β²/a² changes sign at |β|/M.
  for β in (0.0, -M / 2, M / 2, -M, M, -2M, 2M)
    a = min(abs(β) / M, 1.0)
    s = iszero(a) ? 0.0 : β^2 / a
    minimum = λ0 * a + λ2 * s
    @test f([β]) ≈ minimum rtol = 1e-12 atol = 1e-14
    @test cone([a, s / 2, β]) == 0
    if !iszero(β)
      @test isinf(cone([a, s / 4, β]))
    end
    for candidate in range(0.01, 1.0; length = 101)
      @test λ0 * candidate + λ2 * β^2 / candidate >= minimum - 1e-12
    end
  end
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

@testset "IndRotatedSOC allocations on fresh branches" begin
  for T in (Float64, Float32)
    M, q = floatmax(T), nextfloat(zero(T))
    a = ldexp(one(T), T === Float64 ? 60 : 30)
    cases = [T[3, 4, 4], T[-3, -4, 4], T[0, 0, 4], T[0, 0, 0],
             T[-1, 2, 0], T[0, 0, M], T[q, -q, 0], T[-a, 1, 1]]
    for v in cases
      for f in (IndRotatedSOC(), IndRotatedSOC([1:3, 5:7]),
                IndRotatedSOC([[3, 1, 2], [7, 5, 6]]))
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
