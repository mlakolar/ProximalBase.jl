module SparseValueCostTest

using Test, ProximalBase, LinearAlgebra, SparseArrays

const PB = ProximalBase

# zero(C) is counted: a pass over unstored entries creates one for each entry.
const ZEROS = Ref(0)
struct C <: AbstractFloat
  v::Float64
end
Base.zero(::Type{C}) = (ZEROS[] += 1; C(0.0))
Base.zero(::C) = zero(C)
Base.float(::Type{C}) = Float64
Base.float(x::C) = x.v
Base.convert(::Type{C}, x::Real) = C(Float64(x))
Base.convert(::Type{Float64}, x::C) = x.v
Base.Float64(x::C) = x.v
Base.promote_rule(::Type{C}, ::Type{T}) where {T<:Real} = promote_type(Float64, T)
for op in (:abs, :abs2, :-)
  @eval Base.$op(x::C) = C($op(x.v))
end
Base.isnan(x::C) = isnan(x.v)
Base.isfinite(x::C) = isfinite(x.v)
Base.iszero(x::C) = iszero(x.v)
Base.:(==)(x::C, y::C) = x.v == y.v
Base.:<(x::C, y::C) = x.v < y.v
Base.:<=(x::C, y::C) = x.v <= y.v
Base.:+(x::C, y::C) = C(x.v + y.v)
Base.:*(x::C, y::C) = C(x.v * y.v)

zeros_made(f) = (ZEROS[] = 0; v = f(); (ZEROS[], v))

# Writing a nonzero first keeps the coordinate stored when its value is zero.
function make_iterate(T, kind, data; p = 100)
  x = kind == :sym ? SymmetricSparseIterate(T, p) :
      kind == :mat ? SparseIterate(T, p, p) : SparseIterate(T, p)
  for (r, c, v) in data
    if kind == :vec
      x[r] = T(1)
      x[r] = T(v)
    else
      x[r, c] = T(1)
      x[r, c] = T(v)
    end
  end
  x
end

function helpers(x)
  a = ones(size(x))
  (PB._sumabs2, PB._norm2, PB._norm2_scaled,
   y -> PB._scaled_sum(y, (1.0, 1.0), abs),
   y -> PB._scaled_sumabs2(y, (1.0, 1.0)),
   y -> PB._scaled_dot(a, y, (1.0, 1.0)))
end

# Conversion and the independent dense reference stay outside the measured call.
function check_oracle(f, kind, data; expected = nothing)
  x = make_iterate(C, kind, data)
  y = make_iterate(Float64, kind, data)
  dense = Array(y)
  n, value = zeros_made(() -> f(x))
  @test n == 0
  @test isequal(value, f(y))
  reference = f(dense)
  @test isequal(value, reference) || isapprox(value, reference)
  expected === nothing || (@test isequal(value, expected))
end

@testset "Sparse value cost" begin
  @testset "Detector control" begin
    for kind in (:sym, :mat)
      x = make_iterate(C, kind, [(1, 1, 1.0), (2, 1, 0.5)]; p = 200)
      controls = (y -> invoke(PB._sumabs2, Tuple{AbstractArray}, y),
                  y -> invoke(norm, Tuple{Any, Real}, y, 2),
                  y -> sum(abs2, y))
      for (name, f) in zip(("generic sumabs2", "generic norm", "sum abs2"), controls)
        @testset "$kind $name" begin
          n, _ = zeros_made(() -> f(x))
          @test n >= length(x) - 2nnz(x)
        end
      end
    end
  end

  @testset "Helper guard" begin
    for kind in (:vec, :mat, :sym), data in
        ([(1, 1, 1.0), (2, 1, 0.5), (3, 3, 0.0)],
         [(1, 1, 1e-300), (2, 1, -1e-300), (3, 3, 0.0)],
         [(1, 1, 1e300), (2, 1, -1e300), (3, 3, 0.0)],
         Tuple{Int, Int, Float64}[])
      x = make_iterate(C, kind, data)
      y = make_iterate(Float64, kind, data)
      names = ("sumabs2", "norm2", "norm2_scaled", "scaled_sum", "scaled_sumabs2", "scaled_dot")
      for (name, f, g) in zip(names, helpers(x), helpers(y))
        @testset "$kind $name stored=$(nnz(x))" begin
          n, value = zeros_made(() -> f(x))
          if nnz(x) == 0 && name in ("norm2", "norm2_scaled")
            # An empty-view norm may create one zero; a dense pass would create length(x) zeros.
            @test n <= 1
          else
            @test n == 0
          end
          @test isequal(value, g(y))
        end
      end
    end
  end

  @testset "Oracle guard" begin
    p = 100
    a, biga = ones(p, p), fill(1e200, p, p)
    oracles = (IndHyperplane(a, 0.0), IndHyperplane(biga, 0.0),
      IndHalfspace(a, 0.0), IndHalfspace(biga, 0.0),
      IndHyperslab(a, -1.0, 1.0), SqrNormL2(), NormL2(), NormL1(),
      NormL1Nonneg(), NormL0(), MCP(1.0, 3.0), SCAD(1.0, 3.7),
      ElasticNet(1.0, 1.0), CubeNormL2(), IndBallL2(1.0),
      IndBallL2(1e308), IndBallL2(floatmax(Float64)), IndSphereL2(1.0),
      IndBallLinf(1.0), IndBox(-1.0, 1.0), IndNonnegative(), NormLinf(),
      IndBallL1(1.0), IndBallL1(1e308), IndBallL1(floatmax(Float64)))
    for kind in (:sym, :mat), magnitude in (1.0, 1e-300, 1e300),
        sign in (-1, 1), f in oracles
      data = [(1, 1, sign * magnitude), (2, 1, magnitude), (3, 3, 0.0)]
      check_oracle(f, kind, data)
    end

    # The radius, rather than the norm, forces the scaled membership branch.
    for r in (1e308, floatmax(Float64))
      slack = r + PB._tol(Float64) * max(1.0, r)
      @test isfinite(slack) == (r == 1e308)
      for f in (IndBallL1(r), IndBallL2(r)), kind in (:sym, :mat)
        check_oracle(f, kind, [(1, 1, 1e300), (2, 1, 1e300)]; expected = 0.0)
        data = [(1, 1, floatmax(Float64)), (2, 2, floatmax(Float64))]
        check_oracle(f, kind, data; expected = Inf)
        if !isfinite(slack)
          x = make_iterate(Float64, kind, data)
          @test isequal(f(x), PB._ball_scaled_value(f, x, Float64))
        end
      end
    end

    # Each indicator has an explicit interior/boundary member and a nonmember.
    indicators = ((IndHyperplane(a, 0.0), 0.0, 2.0),
      (IndHyperplane(biga, 0.0), 0.0, 2.0),
      (IndHalfspace(a, 0.0), -1.0, 2.0),
      (IndHalfspace(biga, 0.0), -1.0, 2.0),
      (IndHyperslab(a, -1.0, 1.0), 0.0, 2.0),
      (IndBallL1(1.0), 0.0, 2.0), (IndBallL2(1.0), 0.0, 2.0),
      (IndSphereL2(1.0), 1.0, 2.0), (IndBallLinf(1.0), 0.0, 2.0),
      (IndBox(-1.0, 1.0), 0.0, 2.0), (IndNonnegative(), 0.0, -1.0))
    for kind in (:sym, :mat), (f, inside, outside) in indicators
      check_oracle(f, kind, [(1, 1, inside)]; expected = 0.0)
      check_oracle(f, kind, [(1, 1, outside)]; expected = Inf)
    end
  end

  @testset "Known dense paths" begin
    for kind in (:sym, :mat)
      x = make_iterate(C, kind, [(1, 1, 1e300), (2, 1, 1e300)])
      n, _ = zeros_made(() -> NormL2(1e100)(x))
      @test_broken n == 0
      x = make_iterate(C, kind, [(1, 1, 1.0), (2, 1, 0.5)])
      n, _ = zeros_made(() -> IndSimplex(2.0)(x))
      @test_broken n == 0
    end
    x = make_iterate(C, :sym, [(1, 1, 1.0), (2, 1, 0.5)])
    n, _ = zeros_made(() -> IndBallL0(3)(x))
    @test_broken n == 0
    x = make_iterate(C, :mat, [(1, 1, 1.0), (2, 1, 0.5)])
    f = LinearFunction(ones(size(x)))
    n, _ = zeros_made(() -> f(x))
    @test_broken n == 0
  end

  @testset "norm of a SymmetricSparseIterate" begin
    for T in (Float64, Float32, BigFloat, Int)
      cases = Any[(3, [(1, 1, 1), (2, 1, -2), (3, 3, 0)]),
                  (3, Tuple{Int, Int, Int}[]), (0, Tuple{Int, Int, Int}[])]
      if T <: Integer
        push!(cases, (3, [(1, 1, typemax(Int)), (2, 1, typemax(Int))]))
        push!(cases, (3, [(1, 1, typemin(Int)), (2, 1, typemin(Int))]))
      else
        for v in (NaN, Inf, -Inf, 0.0, -0.0, 0.1)
          push!(cases, (3, [(1, 1, v), (2, 1, v), (3, 3, 0.0)]))
        end
        push!(cases, (3, [(1, 1, NaN), (2, 1, Inf)]))
      end
      for (d, data) in cases
        x = make_iterate(T, :sym, data; p = d)
        for p in (1, Inf, 0, 2, 3, -Inf)
          value = norm(x, p)
          generic = invoke(norm, Tuple{Any, Real}, x, p)
          reference = norm(Matrix(x), p)
          @test typeof(value) == typeof(generic) == typeof(reference)
          if p == Inf || p in (0, -Inf) || all(t -> isinteger(abs(t[3])), data)
            @test isequal(value, reference)
            @test isequal(value, generic)
          else
            @test isequal(value, reference) || isapprox(value, reference)
            @test isequal(value, generic) || isapprox(value, generic)
          end
        end
        @test isequal(norm(x), invoke(norm, Tuple{Any, Real}, x, 2))
      end
    end
  end
end

end
