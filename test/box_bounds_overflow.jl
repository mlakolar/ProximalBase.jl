module BoxBoundsOverflowTest

using Test
using Test: Serialization
using ProximalBase
using Adapt
using JLArrays
using GPUArraysCore: allowscalar

using ..TestUtils

const PB = ProximalBase

entry(b::Real, i) = b
entry(b::AbstractArray, i) = b[i]

function oracle(l, u, x)
  T = float(eltype(x))
  ok = true
  for i in eachindex(IndexLinear(), x)
    ok &= PB._inbox(x[i], entry(l, i), entry(u, i), PB._tol(T))
  end
  ok ? zero(T) : T(Inf)
end

function check_box(l, u, x)
  f = IndBox(l, u)
  @test f(x) == oracle(l, u, x)
  f
end

inflates_finitely(::Type{T}, m) where {T} = isfinite(T(m) + PB._tol(T) * max(one(T), T(m)))

@testset "mixed arithmetic counterexamples" begin
  # Vary the lower and upper types independently, including scalar/array pairs.
  for L in (Float32, Float64), U in (Float32, Float64)
    l, u = -L(floatmax(Float32)), U(floatmax(Float32))
    for v in Float32[-Inf, -floatmax(Float32), 0, floatmax(Float32), Inf, NaN]
      for (lo, hi) in (([l], [u]), (l, [u]), ([l], u), (l, u))
        f = check_box(lo, hi, [v])
        @test L === U ? f.bmax == Float64(floatmax(Float32)) : isnan(f.bmax)
      end
    end
  end
  for (l, u, x) in (([-floatmax(Float32)], [1.0], Float32[-Inf]),
                    ([-1.0], [floatmax(Float32)], Float32[Inf]),
                    ([-1], [65504], Float16[Inf]),
                    ([-65504], [1], Float16[-Inf]),
                    ([-1f0], [typemax(UInt128)], Float32[Inf]),
                    ([zero(UInt128)], [floatmax(Float32)], Float32[Inf]),
                    ([zero(UInt128)], [typemax(UInt128)], Float32[-Inf]))
    @test isnan(check_box(l, u, x).bmax)
  end
end

@testset "evaluated overflow transition" begin
  for T in (Float32, Float64)
    m = floatmax(T) / (one(T) + PB._tol(T))
    while !inflates_finitely(T, m)
      m = prevfloat(m)
    end
    while inflates_finitely(T, nextfloat(m))
      m = nextfloat(m)
    end
    first_overflow = nextfloat(m)
    for b in (prevfloat(m, 2), prevfloat(m), m, first_overflow, nextfloat(first_overflow))
      f = IndBox([-b], [b])
      @test f.bmax === Float64(b)
      @test inflates_finitely(T, f.bmax) == (b <= m)
      for v in T[-Inf, -floatmax(T), -b, 0, b, floatmax(T), Inf, NaN]
        check_box(f.l, f.u, [v])
        check_box(-b, [b], [v])
        check_box([-b], b, [v])
      end
    end
    # Infinite bounds are excluded from the cached finite maximum.
    f = check_box(T[-Inf, -2, -Inf], T[Inf, Inf, 3], T[-Inf, 0, Inf])
    @test f.bmax === 3.0
    @test_throws ArgumentError IndBox(T[NaN], T[1])
    @test_throws ArgumentError IndBox(T[-1], T[NaN])
  end
end

@testset "fallback types and array layouts" begin
  for T in (Float16, BigFloat)
    for v in T[-Inf, -floatmax(T), 0, floatmax(T), Inf, NaN]
      @test isnan(check_box(T[-floatmax(T)], T[floatmax(T)], [v]).bmax)
    end
  end
  setprecision(BigFloat, 64) do
    setrounding(BigFloat, RoundDown) do
      M = floatmax(BigFloat)
      for v in BigFloat[-Inf, -M, 0, M, Inf, NaN]
        @test isnan(check_box([-M], [M], [v]).bmax)
      end
    end
  end
  for (l, u, x) in (([typemin(Int)], [typemax(Int)], [typemin(Int)]),
                    ([typemin(Int)], [typemax(Int)], [typemax(Int)]),
                    ([zero(UInt128)], [typemax(UInt128)], [typemax(UInt128)]),
                    ([-1//2], [1//2], [1//3]),
                    (Real[-1f0, -2.0], Real[1.0, 2f0], Float32[0, Inf]),
                    (Any[-1, -2.0], Any[1f0, 2], [NaN, 0.0]))
    @test isnan(check_box(l, u, x).bmax)
  end
  # Integer inputs and mixed bounds must fall back even if the cache is finite.
  check_box([-1.0], [1.0], [1])
  check_box([-1.0], [floatmax(Float64)], Float32[Inf])
  check_box([-floatmax(Float64)], [1.0], Float32[-Inf])
  for T in (Float32, Float64)
    l, u, x = fill(-one(T), 2, 2), fill(one(T), 2, 2), T[0 Inf; -1 NaN]
    check_box(l, u, x)
    check_box(view(l, :, 1), view(u, :, 1), view(x, :, 1))
    check_box(-one(T), u, x)
    check_box(l, one(T), x)
    M = floatmax(T)
    check_box(fill(-M, 2, 2), fill(M, 2, 2), x)
    check_box(view(fill(-M, 4), 1:2:4), view(fill(M, 4), 1:2:4), T[-Inf, Inf])
  end
end

@testset "empty bounds and reconstructed metadata" begin
  for T in (Float32, Float64)
    for (l, u) in ((T[], T[]), (T(Inf), T[]), (T[], T(-Inf)),
                   (T[-Inf, -Inf], T[Inf, Inf]))
      x = zeros(T, l isa AbstractArray ? length(l) : length(u))
      @test check_box(l, u, x).bmax === 0.0
    end
    f = IndBox(T[-1, -floatmax(T)], T[2, floatmax(T)])
    g = restrict(f, [1])
    @test g.bmax === 2.0
    @test g(T[0]) == oracle(g.l, g.u, T[0])
    @test IndBallLinf(T(3)).bmax === 3.0
    @test IndBallLinf(T(Inf)).bmax === 0.0
    @test IndBox{Vector{T}, Vector{T}}(T[-1], T[2]).bmax === 2.0
  end
  @test isnan(IndBox(Int[], Int[]).bmax)
  @test IndBox{Real, Real}(-1.0, 2.0).bmax === 2.0
  @test IndBox{AbstractArray, AbstractArray}([-1f0], [2f0]).bmax === 2.0
end

@testset "device construction and adaptation" begin
  allowscalar(false)
  l32, u32 = JLArray(Float32[-Inf, -1]), JLArray(Float32[Inf, 1])
  @test PB._box_bound_max(l32) === 1f0
  @test PB._box_bound_max(u32) === 1f0
  @test PB._box_bound_max(JLArray(Float32[])) === 0f0
  @test IndBox(l32, u32).bmax === 1.0
  @test Adapt.adapt(JLArray, IndBox(Float32[-1], Float32[1])).bmax === 1.0
  for T in (Float32, Float64), b in (T(2), floatmax(T))
    l, u = T[-Inf, -b], T[Inf, b]
    f = IndBox(l, u)
    for g in (IndBox(JLArray(l), JLArray(u)), Adapt.adapt(JLArray, f))
      @test g.bmax === Float64(b)
      for x in (T[0, 1], T[-Inf, Inf], T[0, NaN])
        @test g(JLArray(x)) == oracle(l, u, x)
      end
      h = Adapt.adapt(Array, g)
      @test h.bmax === f.bmax
      @test h.l == l && h.u == u
    end
  end
  @test IndBox(JLArray(Float64[]), JLArray(Float64[])).bmax === 0.0
end

@testset "display, equality and serialization" begin
  # Fix the display context so the runner's imports do not change qualification.
  @test repr(IndBox(-1.0, 1.0); context = :module => @__MODULE__) == "IndBox{Float64, Float64}(-1.0, 1.0, true, 1.0)"
  for (l, u) in (([-1.0], [1.0]), ([-1], [1]))
    f, g = IndBox(l, u), IndBox(l, u)
    @test f == g
    @test hash(f) == hash(g)
    io = IOBuffer()
    Serialization.serialize(io, f)
    seekstart(io)
    h = Serialization.deserialize(io)
    @test h.l == l && h.u == u
    @test h.zp == f.zp && isequal(h.bmax, f.bmax)
    @test h([0.0]) === f([0.0])
  end
end

@testset "both membership paths allocate nothing" begin
  for T in (Float32, Float64), b in (one(T), floatmax(T))
    f = IndBox(fill(-b, 4), fill(b, 4))
    for x in (zeros(T, 4), T[Inf, -Inf, 0, NaN])
      f(x)
      @test allocs(f, x) == 0
    end
  end
end

end
