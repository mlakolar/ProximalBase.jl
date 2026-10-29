module BinaryTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

const PT = ProximalBase.Testing
const PB = ProximalBase

function exact_endpoint(l, u, v)
  v, l, u = promote(float(v), float(l), float(u))
  q = Rational{BigInt}(v)
  abs(q - Rational{BigInt}(l)) <= abs(q - Rational{BigInt}(u)) ? l : u
end

# Record bound reads to check that implicit zeros are skipped.
struct ReadBounds{T} <: AbstractVector{T}
  data::Vector{T}
  reads::Vector{Int}
end
Base.size(b::ReadBounds) = size(b.data)
Base.IndexStyle(::Type{<:ReadBounds}) = IndexLinear()
Base.getindex(b::ReadBounds, i::Int) = (push!(b.reads, i); b.data[i])

@testset "IndBinary" begin
  for T in (Float32, Float64)
    signs = IndBinary(-1, 1)
    bits = IndBinary(0, 1)
    @test prox(signs, T[-2, -1, 0, 1, 2]) == T[-1, -1, -1, 1, 1]
    @test prox(bits, T[-1, 0, 0.5, 0.6, 2]) == T[0, 0, 0, 1, 1]
    @test signs(T[-1, 1]) === zero(T)
    @test bits(T[0, 1]) === zero(T)
    @test signs(T[0, 1]) === T(Inf)
    @test bits(T[0.5, 1]) === T(Inf)
    t = PB._tol(T)
    for (l, u) in ((T(-1), T(1)), (T(0), T(1)), (T(4), T(8)))
      f = IndBinary(l, u)
      @test f(T[l + t * max(one(T), abs(l)) / 2, u - t * max(one(T), abs(u)) / 2]) == 0
      @test f(T[l - 2t * max(one(T), abs(l))]) == Inf
      @test f(T[u + 2t * max(one(T), abs(u))]) == Inf
      @test prox(f, T[(l + u) / 2]) == T[l]
    end
    @test !iszeropreserving(signs)
    @test iszeropreserving(bits)
    @test !iszeropreserving(IndBinary(nextfloat(zero(T)), one(T)))
    @test !iszeropreserving(IndBinary(-one(T), -nextfloat(zero(T))))
    for f in (signs, bits, IndBinary(T[-2, 0, -1, 2], T[1, 1, 0, 3]))
      x = T[-1.3, 0.2, 1.6, 0]
      PT.test_traits(f; proximable = true, separable = true)
      PT.test_separable(f, x, T(0.7); groups = [1:2, 3:4], rng = MersenneTwister(174))
      PT.test_prox(f, x, T(0.7); local_points = false, rng = MersenneTwister(174))
      @test f(x) isa T
      @test prox(f, x) isa Vector{T}
      @test PB._isindicator(f) && PB._isindicator(typeof(f))
      @test value_and_prox!(similar(x), f, x, 0.7) === zero(T)
    end
  end
end

@testset "exact nearest endpoint" begin
  rng = MersenneTwister(174)
  for (T, U) in ((Float32, UInt32), (Float64, UInt64))
    e, η, large = eps(T), nextfloat(zero(T)), floatmax(T)
    cases = [(one(T) + e, one(T) + 6e, one(T) + 4e),
             (zero(T), 3η, 2η), (-3η, zero(T), -2η)]
    for (l, u) in ((-large, large), (large / 2, large), (-large, -large / 2),
                   (-one(T), one(T)), (zero(T), large), (-large, zero(T)))
      for v in (l, nextfloat(l), u, prevfloat(u), zero(T), η, -η, l / 2 + u / 2)
        push!(cases, (l, u, v))
      end
    end
    for l in -8:7, u in (l + 1):8, v in -9:9
      push!(cases, (l * η, u * η, v * η))
    end
    for _ in 1:128
      a, b, v = reinterpret(T, rand(rng, U, 3))
      if all(isfinite, (a, b, v)) && a != b
        l, u = minmax(a, b)
        push!(cases, (l, u, v))
      end
      if isfinite(a)
        l, u = a, nextfloat(a, 7)
        if isfinite(u)
          for k in 0:7
            push!(cases, (l, u, nextfloat(l, k)))
          end
        end
      end
    end
    for (l, u, v) in cases
      f = IndBinary(l, u)
      expected = exact_endpoint(l, u, v)
      @test prox_entry(f, v, 1, one(T)) == expected
    end
    l, u, x = first.(cases), getindex.(cases, 2), last.(cases)
    f = IndBinary(l, u)
    expected = [exact_endpoint(l[i], u[i], x[i]) for i in eachindex(x)]
    @test prox(f, x) == expected
    @test prox!(x, f, x, one(T)) == expected
    @test f(expected) == 0
    @test prox_entry(IndBinary(one(T) + e, one(T) + 6e), one(T) + 4e, 1, one(T)) == one(T) + 6e
    @test prox_entry(IndBinary(zero(T), 3η), 2η, 1, one(T)) == 3η
  end
end

@testset "nonfinite entries and mixed precision" begin
  for T in (Float32, Float64), f in (IndBinary(-1, 1), IndBinary(-floatmax(T), floatmax(T)),
                                    IndBinary(floatmax(T) / 2, floatmax(T)))
    x = T[-Inf, Inf, NaN]
    y = prox(f, x)
    @test y[1] == T(f.l) && y[2] == T(f.u) && isnan(y[3])
    @test all(v -> value_entry(f, v, 1) === T(Inf), x)
    @test f(x) === T(Inf)
    @test value_and_prox!(y, f, x, 0.7) === zero(T)
    @test isnan(y[3])
  end
  e = eps()
  f = IndBinary(1 + e, 1 + 6e)
  @test prox!(zeros(1), f, Float32[1], 1) == [1 + e]
  @test prox(f, Float32[1]) == Float32[1]
  large = 2 * Float64(floatmax(Float32))
  f = IndBinary(-large, large)
  @test prox!(zeros(2), f, Float32[-1, 1], 1) == [-large, large]
  @test prox(f, Float32[-1, 1]) == Float32[-Inf, Inf]
  @test IndBinary(1 + e, 1 + 6e)(Float32[1]) === 0f0
  f = IndBinary(typemin(Int), typemax(Int))
  @test prox(f, [0.0]) == [float(typemin(Int))]
  @test prox(IndBinary(-2, 3), [-4, 1, 4]) == [-2.0, 3.0, 3.0]
end

@testset "array bounds, restriction and diagonal steps" begin
  l, u = [0., -1., 0., -2.], [1., 0., 2., 0.]
  f = IndBinary(l, u)
  @test f.l === l && f.u === u
  @test iszeropreserving(f)
  x = [0.5, -0.5, 1.1, -1.1]
  expected = [0., -1., 2., -2.]
  steps = view([0.1, 9., 0.2, 9., 0.3, 9., 0.4, 9.], 1:2:8)
  for op in (prox!, value_and_prox!), alias in (false, true)
    input = copy(x)
    output = alias ? input : similar(input)
    result = op(output, f, input, steps)
    @test output == expected
    @test op === prox! ? result === output : result === 0.0
  end
  @test prox(IndBinary(reshape(l, 2, 2), u), reshape(x, 2, 2), reshape(collect(steps), 2, 2)) == reshape(expected, 2, 2)
  @test prox(IndBinary(0, u .+ 3), x) == zeros(4)
  @test prox(IndBinary(l .- 3, 0), x) == zeros(4)
  S = [4, 1, 2]
  r = restrict(f, S)
  @test r.l == l[S] && r.u == u[S]
  @test r(expected[S]) == f(expected)
  @test prox(r, x[S]) == prox(f, x)[S]
  @test restrict(IndBinary(0, 1), S)([0., 1., 0.]) == 0
  @test_throws ArgumentError restrict(IndBinary(-1, 1), S)
  @test_throws ArgumentError restrict(IndBinary([0., 1.], [1., 2.]), [1])
  l[1] = -1
  @test !iszeropreserving(IndBinary(l, u))
end

@testset "sparse and atom iterates" begin
  l = ReadBounds(zeros(6), Int[])
  u = ReadBounds(ones(6), Int[])
  f = IndBinary(l, u)
  x = SparseIterate(6)
  x[2] = 1.
  activate!(x, 4)                         # a stored zero must remain stored
  stored = Set(activecoordinates(x))
  empty!(l.reads); empty!(u.reads)
  @test f(x) == 0
  @test Set(l.reads) == Set(u.reads) == stored
  y = SparseIterate(6)
  y[6] = 1.
  empty!(l.reads); empty!(u.reads)
  @test prox!(y, f, x, 0.7) === y
  @test Set(l.reads) == Set(u.reads) == stored
  @test Vector(y) == Vector(x)
  @test Set(activecoordinates(y)) == Set([2, 6])
  @test nnz(x) == 2 && x[4] == 0
  z = copy(x)
  prox!(z, f, z, 0.7)
  @test Set(activecoordinates(z)) == stored && Vector(z) == Vector(x)
  for f in (IndBinary(0, 1), IndBinary([0., -1., 0., -1., 0., -1.], [1., 0., 1., 0., 1., 0.]),
            IndBinary(-1, 1), IndBinary(nextfloat(0.), 1.))
    for input in (SparseIterate(6), x, SparseIterate([0., -0.6, 0., 0.8, 0., 0.]),
                  AtomIterate([0., -0.6, 0., 0.8, 0., 0.], [1:2, 3:4, 5:6]))
      dense = Vector(input)
      expected = prox(f, dense)
      @test f(input) == f(dense)
      for op in (prox!, value_and_prox!), alias in (false, true)
        src = copy(input)
        out = alias ? src : copy(input)
        op(out, f, src, 0.7)
        @test Vector(out) == expected
      end
    end
  end
  @test nnz(prox(IndBinary(-1, 1), SparseIterate(6))) == 6
  a = AtomIterate(6, 3)
  prox!(a, IndBinary(-1, 1), a, 1.)
  @test Vector(a) == fill(-1., 6) && nactive(a) == 3
  for f in (IndBinary(0, 1), IndBinary(Float64[], Float64[]))
    @test f(Float64[]) === 0.0
    @test isempty(prox(f, Float64[]))
    @test f(SparseIterate(0)) === 0.0
    @test isempty(prox(f, SparseIterate(0)))
  end
end

@testset "symmetric bounds and spectral projection" begin
  X = [-0.2 -0.8; -0.8 1.3]
  for f in (IndBinary(0, 1), IndBinary(-1, 1), IndBinary([0. -2.; -2. 0.], [1. 0.; 0. 2.]))
    PT.test_separable(f, X, 0.7; symmetric = true, rng = MersenneTwister(174))
    s = SymmetricSparseIterate(X)
    @test f(s) == f(X)
    for op in (prox!, value_and_prox!), alias in (false, true)
      src = copy(s)
      out = alias ? src : SymmetricSparseIterate(ones(2, 2))
      op(out, f, src, [0.2 0.3; 0.3 0.7])
      @test Matrix(out) == prox(f, X)
    end
  end
  @test Matrix(prox(IndBinary(-1, 1), SymmetricSparseIterate(2))) == -ones(2, 2)
  for T in (Float32, Float64)
    X = T[0.2 0.4; 0.4 0.8]
    F = Spectral(IndBinary(0, 1))
    P = PT.test_prox(F, X, T(0.7); symmetric = true, local_points = false, rng = MersenneTwister(174))
    @test P isa Matrix{T}
    @test P ≈ X
    @test P * P ≈ P
    @test F(P) === zero(T)
    PT.test_traits(F; proximable = true)
    @test prox(F, Matrix(Diagonal(T[0.5, 0.6]))) == Matrix(Diagonal(T[0, 1]))
  end
  @test_throws ArgumentError Spectral(IndBinary(zeros(2), ones(2)))
  @test_throws ArgumentError Conjugate(IndBinary(0, 1))
  @test_throws ArgumentError Constrained(IndBinary(0, 1), IndBox(0, 1))
  @test_throws ArgumentError Constrained(IndBinary(0, 1), IndHyperplane(1, 1))
end

@testset "parameter checks before writes" begin
  @test_throws MethodError IndBinary()
  for (l, u) in ((1, 1), (2, 1), (-Inf, 1), (0, Inf), (NaN, 1), (0, NaN),
                 ([0., 1.], [1., 1.]), ([0., NaN], 1.), (0., [1., Inf]))
    @test_throws ArgumentError IndBinary(l, u)
  end
  @test_throws DimensionMismatch IndBinary(zeros(2), ones(3))
  @test_throws BoundsError prox_entry(IndBinary(zeros(2), ones(2)), 0., 3, 1.)
  points = (ones(4), SparseIterate([1., 0., -2., 0.]),
            SymmetricSparseIterate([1. 0.; 0. -2.]), AtomIterate([1., 0., -2., 0.], 2))
  for x in points
    for f in (IndBinary(zeros(2), 1), IndBinary(0, ones(2))),
        step in (0.7, fill(0.7, size(x))), op in (prox!, value_and_prox!), alias in (false, true)
      src = copy(x)
      out = alias ? src : copy(x)
      before = Array(out)
      stored = out isa Array ? nothing : copy(activecoordinates(out))
      @test_throws DimensionMismatch f(src)
      @test_throws DimensionMismatch op(out, f, src, step)
      @test Array(out) == before && Array(src) == Array(x)
      stored === nothing || @test activecoordinates(out) == stored
    end
    for op in (prox!, value_and_prox!), alias in (false, true)
      src = copy(x)
      out = alias ? src : copy(x)
      stored = out isa Array ? nothing : copy(activecoordinates(out))
      @test_throws DimensionMismatch op(out, IndBinary(0, 1), src, ones(3))
      @test Array(out) == Array(x) && Array(src) == Array(x)
      stored === nothing || @test activecoordinates(out) == stored
    end
    if !(x isa Array)
      for f in (IndBinary(zeros(2), 1), IndBinary(0, ones(2))), k in (1, ncoordinates(x))
        out = copy(x)
        stored = copy(activecoordinates(out))
        @test_throws DimensionMismatch prox_coordinate!(out, f, k, 0.7)
        @test Array(out) == Array(x) && activecoordinates(out) == stored
      end
      out = copy(x)
      @test_throws MethodError prox_coordinate!(out, IndBinary(0, 1), 1, ones(3))
      @test Array(out) == Array(x)
    end
  end
  y = ones(3)
  @test_throws DimensionMismatch prox!(y, IndBinary(0, 1), zeros(4), 1.)
  @test y == ones(3)
end

@testset "allocations" begin
  for T in (Float32, Float64), f in (IndBinary(0, 1), IndBinary(zero(T), one(T)),
                                    IndBinary(zeros(T, 8), ones(T, 8)))
    x, y, γ = ones(T, 8), zeros(T, 8), T(0.7)
    s = SparseIterate(x)
    @test allocs(f, x) == 0
    @test allocs(prox!, y, f, x, γ) == 0
    @test allocs(value_and_prox!, y, f, x, γ) == 0
    @test allocs(prox_entry, f, T(0.5), 1, γ) == 0
    @test allocs(prox_coordinate!, s, f, 1, γ) == 0
  end
  for T in (Float32, Float64)
    f = IndBinary(-floatmax(T), floatmax(T))
    @test allocs(prox_entry, f, nextfloat(zero(T)), 1, one(T)) == 0
  end
end

end
