module SparseValueCostTest
# #808: the NormL2 value visits only the stored entries of a matrix SparseIterate and of a
# SymmetricSparseIterate, also on its recovery path. Ported from the historical
# test/sparse_value_cost.jl (legacy/full at eff21b6d7d33c622313122f46c6ee71693c101ee): the element type
# C counts its zero(C) calls, and a pass over unstored entries creates one for each, so a count of
# zero rules out the O(p²) traversal deterministically, without timing. The two "Known dense paths"
# checks of NormL2 were @test_broken there and are assertions here.
using Test, ProximalBase, LinearAlgebra

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
Base.isinf(x::C) = isinf(x.v)
Base.iszero(x::C) = iszero(x.v)
Base.:(==)(x::C, y::C) = x.v == y.v
Base.:<(x::C, y::C) = x.v < y.v
Base.:<=(x::C, y::C) = x.v <= y.v
Base.:+(x::C, y::C) = C(x.v + y.v)
Base.:*(x::C, y::C) = C(x.v * y.v)

zeros_made(f) = (ZEROS[] = 0; v = f(); (ZEROS[], v))

# Writing a nonzero first keeps the coordinate stored when its value is zero.
function make_iterate(T, kind, data; p = 100)
  x = kind == :sym ? SymmetricSparseIterate(T, p) : SparseIterate(T, p, p)
  for (r, c, v) in data
    x[r, c] = T(1)
    x[r, c] = T(v)
  end
  x
end
as_matrix(x) = x isa SymmetricSparseIterate ? Matrix(x) : Array(x)
bigfro(A) = setprecision(() -> sqrt(sum(big.(A) .^ 2; init = big(0))), BigFloat, 512)

@testset "Sparse value cost (#808)" begin
  @testset "Detector control" begin
    # a generic pass over the logical matrix is counted: the detector sees an O(p²) traversal
    for kind in (:sym, :mat)
      x = make_iterate(C, kind, [(1, 1, 1.0), (2, 1, 0.5)]; p = 200)
      M = kind == :sym ? matrixview(x) : x
      n, _ = zeros_made(() -> sum(abs2, M))
      @test n >= 200^2 - 4
    end
  end

  @testset "Known dense paths (formerly broken)" begin
    # NormL2(1e100) at two stored entries of magnitude 1e300 in a 100 × 100 logical matrix: the sum of
    # squares overflows and the value is recomputed; the value itself overflows to Inf
    for kind in (:sym, :mat)
      x = make_iterate(C, kind, [(1, 1, 1e300), (2, 1, 1e300)])
      n, v = zeros_made(() -> NormL2(1e100)(x))
      @test n == 0
      @test v == Inf == NormL2(1e100)(make_iterate(Float64, kind, [(1, 1, 1e300), (2, 1, 1e300)]))
    end
  end

  @testset "Recovered values visit stored entries only ($kind)" for kind in (:sym, :mat)
    # finite recovered values, against BigFloat; the symmetric off-diagonal entry counts twice
    for (λ, data) in ((1e-100, [(1, 1, 1e300), (2, 1, 1e300)]),
                      (1e-100, [(1, 1, 1e300), (2, 1, -1e300), (3, 3, 0.0), (5, 4, -0.0)]),
                      (1e200, [(1, 1, 1e-310), (2, 1, 3e-320)]),        # subnormal entries
                      (1.0, [(1, 1, 3.0), (2, 1, -4.0), (3, 3, 0.0)]),  # ordinary path
                      (1.0, Tuple{Int, Int, Float64}[]),
                      (Inf, Tuple{Int, Int, Float64}[]), (Inf, [(4, 2, -0.0)]))
      x = make_iterate(C, kind, data)
      y = make_iterate(Float64, kind, data)
      n, v = zeros_made(() -> NormL2(λ)(x))
      @test n == 0
      @test isequal(v, NormL2(λ)(y))
      ref = setprecision(() -> big(λ) * bigfro(as_matrix(y)), BigFloat, 512)
      if isnan(ref) || iszero(ref)
        @test v === 0.0
      else
        @test abs(big(v) - ref) <= 4 * big(eps(Float64)) * ref
      end
    end
  end
end

end
