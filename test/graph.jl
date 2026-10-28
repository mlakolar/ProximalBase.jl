module GraphTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

# No indexing or AbstractMatrix interface: only the operator contract is available.
mutable struct GraphOperator{T}
  A::Matrix{T}
  products::Int
  nonfinite::Bool
  failat::Int
end
GraphOperator(A::Matrix) = GraphOperator(A, 0, false, typemax(Int))
struct GraphAdjoint{T}
  parent::GraphOperator{T}
end
Base.size(A::GraphOperator) = size(A.A)
Base.eltype(::GraphOperator{T}) where {T} = T
Base.adjoint(A::GraphOperator) = GraphAdjoint(A)
function LinearAlgebra.mul!(y, A::GraphOperator, x)
  A.products += 1
  mul!(y, A.A, x)
  (A.nonfinite || A.products == A.failat) && !isempty(y) && (y[1] = Inf)
  y
end
function LinearAlgebra.mul!(y, A::GraphAdjoint, x)
  A.parent.products += 1
  mul!(y, A.parent.A', x)
  (A.parent.nonfinite || A.parent.products == A.parent.failat) && !isempty(y) && (y[1] = Inf)
  y
end

stack(z) = [z[1]; z[2]]
buffers(z::Tuple) = map(buffers, z)
buffers(z::AbstractArray) = fill(7.0, size(z))
buffers(z::Number) = 7.0

# A QR least-squares solve on a basis of the graph avoids forming either Gram matrix.
function qr_reference(A, z)
  n = size(A, 2)
  B = [Matrix{Float64}(I, n, n); Matrix{Float64}(A)]
  x = qr(B) \ Float64.(stack(z))
  [x; A * x]
end

function kkt_reference(A, z)
  m, n = size(A)
  B = [Matrix{Float64}(A) -Matrix{Float64}(I, m, m)]
  K = [Matrix{Float64}(I, n + m, n + m) B'; B zeros(m, m)]
  (K \ [stack(z); zeros(m)])[1:(n + m)]
end

function unchanged(op, f, point)
  for alias in (false, true)
    x = deepcopy(point)
    out = alias ? x : buffers(point)
    before = deepcopy(out)
    @test_throws DimensionMismatch op(out, f, x, 0.7)
    @test out == before
  end
end

@testset "IndGraph projections" begin
  rng = Xoshiro(175)
  for T in (Float32, Float64), (m, n) in ((5, 3), (3, 5), (3, 3))
    A = randn(rng, T, m, n)
    z = (randn(rng, T, n), randn(rng, T, m))
    reference = qr_reference(A, z)
    tol = T === Float32 ? 1e-4 : 1e-10
    scale = norm(stack(z))
    @test reference ≈ kkt_reference(A, z) rtol = tol
    affine = IndAffine([A -Matrix{T}(I, m, m)], zeros(T, m))
    @test prox(affine, stack(z)) ≈ reference rtol = tol
    for B in (A, sparse(A), GraphOperator(A))
      f = IndGraph(B)
      @test f.A === B
      check_traits(f; proximable = true, convex = true)
      B isa AbstractMatrix && @test size(f.fact) == (min(m, n), min(m, n))
      p = prox(f, z, T(0.3))
      @test eltype(p[1]) === T && eltype(p[2]) === T
      @test norm(stack(p) - reference) <= tol * scale
      @test norm(A * p[1] - p[2]) <= tol * scale
      @test norm(A' * (z[2] - p[2]) + z[1] - p[1]) <= tol * scale
      @test f(p) === zero(T)
      @test f.converged
      @test norm(stack(prox(f, p)) - stack(p)) <= tol * scale
      @test norm(stack(prox(f, z, T(7))) - stack(p)) <= tol * scale
      out = map(similar, z)
      @test prox!(out, f, z, T(0.3)) === out
      @test norm(stack(out) - stack(p)) <= tol * scale
      @test value_and_prox!(out, f, z, T(0.3)) === zero(T)
      for op in (prox!, value_and_prox!)
        inplace = deepcopy(z)
        op(inplace, f, inplace, T(0.3))
        @test norm(stack(inplace) - reference) <= tol * scale
      end
      # The polar graph is the null space of [I Aᵀ].
      polar = IndAffine([Matrix{Float64}(I, n, n) Float64.(A')], zeros(n))
      conjugate = Conjugate(f)
      q = prox(conjugate, z, T(0.7))
      @test norm(stack(q) - prox(polar, Float64.(stack(z)))) <= tol * scale
      @test norm(q[1] + A' * q[2]) <= tol * scale
      @test abs(value_and_prox!(out, conjugate, z, T(0.7))) <= tol * scale^2
      @test_throws ArgumentError reset_cache!(f)
      @test_throws ArgumentError reset_cache!(Scaled(f, 2))
    end
    # The explicit operator keyword is also supported for matrices.
    @test stack(prox(IndGraph(A; operator = true), z)) ≈ reference rtol = tol
  end

  @testset "scalar and diagonal formulas" begin
    for T in (Float32, Float64), a in (zero(T), T(2), T(-3))
      f = IndGraph(reshape([a], 1, 1))
      c, d = T(3), T(-2)
      x = (c + a * d) / (1 + a^2)
      p = prox(f, ([c], [d]))
      @test p[1] ≈ [x]
      @test p[2] ≈ [a * x]
    end
    A = Diagonal([0.0, 2.0, -3.0])
    z = ([1.0, -2.0, 3.0], [4.0, 5.0, -6.0])
    x = (z[1] + diag(A) .* z[2]) ./ (1 .+ diag(A).^2)
    @test stack(prox(IndGraph(A), z)) ≈ [x; A * x]
    # All input reads precede writes even when the output swaps the input blocks.
    f = IndGraph(Matrix(A))
    expected = prox(f, z)
    swapped = (z[2], z[1])
    prox!(swapped, f, z)
    @test stack(swapped) ≈ stack(expected)
  end
end

@testset "IndGraph domains and numerical range" begin
  for T in (Float32, Float64), (m, n) in ((0, 0), (0, 3), (3, 0), (2, 3), (3, 2))
    A = zeros(T, m, n)
    z = (ones(T, n), ones(T, m))
    for B in (A, sparse(A), GraphOperator(A))
      f = IndGraph(B)
      expected = (copy(z[1]), zeros(T, m))
      @test prox(f, z) == expected
      @test f(expected) === zero(T)
      @test value_and_prox!(z, f, z) === zero(T)
      @test z == expected
    end
  end
  f = IndGraph([1 2; 0 -1; 3 0])
  z = ([2, -1], [1, 3, 2])
  @test stack(prox(f, z)) ≈ qr_reference(f.A, z)
  @test f((z[1], f.A * z[1])) === 0.0
  for A in (reshape([3], 1, 1), [3 0])
    f = IndGraph(A)
    z = (fill(typemax(Int), size(A, 2)), [typemax(Int)])
    @test stack(prox(f, z)) ≈ qr_reference(A, z) rtol = 1e-14
    @test f((z[1], [0])) == Inf
  end
  @test_throws ArgumentError IndGraph(ones(ComplexF64, 2, 2))
  for bad in (NaN, Inf, -Inf)
    @test_throws ArgumentError IndGraph(fill(bad, 2, 3))
    @test_throws ArgumentError IndGraph(sparse(fill(bad, 2, 3)))
  end
  @test_throws ArgumentError IndGraph(ones(2, 2); rtol = 0)
  @test_throws ArgumentError IndGraph(ones(2, 2); rtol = Inf)
  @test_throws ArgumentError IndGraph(ones(2, 2); maxiter = 0)
  @test_throws ArgumentError IndGraph(GraphOperator(ones(2, 2)); operator = false)
  for A in ([2.0^27 2.0^27; 0 0], fill(floatmax(Float64), 2, 3))
    @test_throws ArgumentError IndGraph(A)
    @test_throws ArgumentError IndGraph(sparse(A))
  end
  # The wide formula retains the small second block that A*x would cancel away.
  A = reshape([2.0^27, 0.0], 1, 2)
  for B in (A, sparse(A))
    p = prox(IndGraph(B), ([1.0, 0.0], [0.0]))
    @test p[1][2] == 0
    @test p[2][1] ≈ 2.0^-27 rtol = 1e-15
  end
end

@testset "IndGraph values" begin
  for T in (Float32, Float64), sparse_matrix in (false, true), wide in (false, true)
    A = wide ? T[1 0] : reshape(T[1, 0], 2, 1)
    f = IndGraph(sparse_matrix ? sparse(A) : A)
    x = ones(T, size(A, 2))
    y = A * x
    slack = sqrt(eps(T)) * max(one(T), norm(y), norm(A) * norm(x))
    near, far = copy(y), copy(y)
    near[1] += slack / 4
    far[1] += 8slack
    @test f((x, y)) === zero(T)
    @test f((x, near)) === zero(T)
    @test f((x, far)) === T(Inf)
    @test f((fill(T(Inf), length(x)), y)) === T(Inf)
    @test f((x, fill(T(NaN), length(y)))) === T(Inf)
    op = IndGraph(GraphOperator(Matrix(A)))
    @test op((x, y)) === zero(T)
    @test op((x, near)) === zero(T)
    @test op((x, far)) === T(Inf)
  end
  for T in (Float32, Float64), B in (Matrix{T}(I, 2, 2), sparse(Matrix{T}(I, 2, 2)))
    f = IndGraph(B)
    huge = T[floatmax(T) / 2, floatmax(T) / 2]
    @test f((huge, huge)) === zero(T)
    @test f((huge, -huge)) === T(Inf)
    op = IndGraph(GraphOperator(Matrix(B)))
    @test op((huge, huge)) === zero(T)
    @test op((huge, -huge)) === T(Inf)
  end
  # The product overflows although the entries and the Gram matrix are representable.
  for B in (reshape([2.0], 1, 1), sparse(reshape([2.0], 1, 1)))
    f = IndGraph(B)
    x = [floatmax(Float64)]
    @test f((x, [0.0])) == Inf
    @test f((x, [floatmax(Float64)])) == Inf
  end
  for T in (Float32, Float64), sparse_matrix in (false, true)
    A = reshape(T[2, -2, 1], 1, 3)
    f = IndGraph(sparse_matrix ? sparse(A) : A)
    x = T[floatmax(T), floatmax(T), 0]
    @test f((x, T[0])) === zero(T)
    @test f((x, T[floatmax(T)])) === T(Inf)
  end
  # Operator membership uses the image, rather than a possibly large matrix norm.
  A = [1.0 0.0; 0.0 1e6]
  point = ([1.0, 0.0], [1.001, 0.0])
  @test IndGraph(A)(point) == 0
  @test IndGraph(GraphOperator(A))(point) == Inf
end

@testset "IndGraph preflight" begin
  f = IndGraph(ones(3, 2))
  bad = ((ones(2),), (ones(2), ones(3), ones(1)),
    (ones(1, 2), ones(3)), (ones(2), ones(3, 1)),
    (ones(1), ones(3)), (ones(2), ones(4)), ones(5))
  for z in bad
    @test_throws DimensionMismatch f(z)
    for op in (prox!, value_and_prox!)
      unchanged(op, f, z)
      point = ([3.0], z)
      s = SeparableSum(SqrNormL2(2), f)
      unchanged(op, s, point)
      unchanged(op, Scaled(s, 2), point)
      unchanged(op, SeparableSum(SqrNormL2(), Scaled(f, 2)), point)
    end
  end
  z = (ones(2), ones(3))
  for out in ((fill(7.0, 2),), (fill(7.0, 2), fill(7.0, 4)),
              (fill(7.0, 1, 2), fill(7.0, 3)))
    for op in (prox!, value_and_prox!)
      before = deepcopy(out)
      @test_throws DimensionMismatch op(out, f, z, 1)
      @test out == before
      nested_out = ([7.0], out)
      nested_before = deepcopy(nested_out)
      @test_throws DimensionMismatch op(nested_out,
        Scaled(SeparableSum(SqrNormL2(), f), 2), ([1.0], z), 1)
      @test nested_out == nested_before
    end
  end
  @test_throws ArgumentError f((ones(ComplexF64, 2), ones(3)))
  out = deepcopy(z)
  @test_throws MethodError prox!(out, f, z, (1.0, 2.0))
  @test out == z
end

@testset "IndGraph conjugate gradients" begin
  for wide in (false, true)
    A = wide ? [1.0 0 0 0; 0 2 0 0; 0 0 4 0] : [1.0 0 0; 0 2 0; 0 0 4; 0 0 0]
    m, n = size(A)
    op = GraphOperator(A)
    f = IndGraph(op; rtol = 1e-12)
    z = (ones(n), -ones(m))
    p = prox(f, z)
    @test f.converged && f.iterations > 0
    rhs = wide ? z[2] - A * z[1] : z[1] + A' * z[2]
    G = wide ? I + A * A' : I + A' * A
    q = wide ? z[2] - p[2] : p[1]
    @test norm(G * q - rhs) <= 1e-12 * norm(rhs)
    @test stack(p) ≈ qr_reference(A, z) rtol = 1e-10
    prox(f, z)
    @test f.converged && f.iterations == 0
    # A nonzero warm start must not survive a zero right-hand side.
    zero_rhs = (zeros(n), zeros(m))
    @test prox(f, zero_rhs) == zero_rhs
    @test f.converged && f.iterations == 0
    limited = IndGraph(GraphOperator(A); rtol = 1e-12, maxiter = 1)
    out = map(similar, z)
    @test value_and_prox!(out, limited, z) == 0
    @test !limited.converged && limited.iterations == 1
    @test norm(stack(out) - qr_reference(A, z)) > 1e-6
    op.nonfinite = true
    for oracle in (prox!, value_and_prox!), alias in (false, true)
      input = deepcopy(z)
      out = alias ? input : buffers(z)
      before = deepcopy(out)
      @test_throws ArgumentError oracle(out, f, input, 1)
      @test out == before
      @test !f.converged
    end
    @test f(z) == Inf
    op.nonfinite = false
    @test stack(prox(f, z)) ≈ qr_reference(A, z) rtol = 1e-10
    # Failure during CG, after the rhs and initial residual have been computed.
    for oracle in (prox!, value_and_prox!), alias in (false, true)
      input = (-copy(z[1]), -copy(z[2]))
      out = alias ? input : buffers(z)
      before = deepcopy(out)
      op.failat = op.products + 4
      @test_throws ArgumentError oracle(out, f, input, 1)
      @test out == before
      @test !f.converged
    end
    op.failat = typemax(Int)
    @test stack(prox(f, z)) ≈ qr_reference(A, z) rtol = 1e-10
  end
  for T in (Float32, Float64), wide in (false, true)
    A = wide ? T[1 0] : reshape(T[1, 0], 2, 1)
    f = IndGraph(GraphOperator(A))
    magnitudes = T === Float32 ? T[1e-30, 1e30] : T[1e-200, 1e200]
    for magnitude in magnitudes
      z = (fill(magnitude, size(A, 2)), zeros(T, size(A, 1)))
      p = prox(f, z)
      @test f.converged && f.iterations > 0
      @test p[1][1] ≈ magnitude / 2
      @test p[2][1] ≈ magnitude / 2
    end
  end
end

@testset "IndGraph precision" begin
  for T in (Float32, Float64), (m, n) in ((3, 2), (2, 3))
    S = T === Float32 ? Float64 : Float32
    A = reshape(T.(1:(m * n)), m, n) / T(4)
    f = IndGraph(A)
    for (C, D, X, Y) in ((T, T, T, T), (T, S, S, T), (S, T, T, S),
                         (T, T, S, S), (S, S, T, T))
      z = (C.(1:n), -D.(1:m))
      out = (zeros(X, n), zeros(Y, m))
      @test typeof(f(z)) === float(promote_type(T, C, D))
      @test typeof(value_and_prox!(out, f, z)) === float(promote_type(T, C, D, X, Y))
      @test eltype(out[1]) === X && eltype(out[2]) === Y
      p = prox(f, z)
      @test eltype(p[1]) === C && eltype(p[2]) === D
      @test stack(out) ≈ qr_reference(A, z) rtol = 1e-4
    end
  end
end

function alternating!(op::O, out, f, z, w, γ) where {O}
  op(out, f, z, γ)
  @assert f.iterations > 0 && f.converged
  v = op(out, f, w, γ)
  @assert f.iterations > 0 && f.converged
  v
end

@testset "IndGraph zero allocations" begin
  rng = Xoshiro(176)
  for T in (Float32, Float64), (m, n) in ((7, 3), (3, 7))
    A = randn(rng, T, m, n)
    z = (randn(rng, T, n), randn(rng, T, m))
    w = (randn(rng, T, n), randn(rng, T, m))
    out = map(similar, z)
    f = IndGraph(A)
    @test allocs(f, z) == 0
    @test allocs(prox!, out, f, z, one(T)) == 0
    @test allocs(value_and_prox!, out, f, z, one(T)) == 0
    operator = GraphOperator(A)
    iterative = IndGraph(operator)
    @test allocs(iterative, z) == 0
    for op in (prox!, value_and_prox!)
      before = operator.products
      @test allocs(alternating!, op, out, iterative, z, w, one(T)) == 0
      @test operator.products > before
    end
  end
end

@testset "IndGraph review regressions" begin
  @testset "promoted membership products" begin
    a = Float32(1 + 2.0^-12)
    for A in (reshape([a], 1, 1), sparse(reshape([a], 1, 1))), operator in (false, true)
      f = IndGraph(A; operator)
      point = ([a], [Float64(a)^2])
      @test f(point) === 0.0
      @test f((Float64.(point[1]), point[2])) === 0.0
      @test allocs(f, point) == 0
    end
    f = IndGraph(GraphOperator(reshape([a], 1, 1)))
    @test f(([a], [Float64(a)^2])) === 0.0
  end

  @testset "integer operator membership" begin
    for A in (reshape([2], 1, 1), sparse(reshape([2], 1, 1)),
              GraphOperator(reshape([2], 1, 1)))
      f = IndGraph(A; operator = true)
      point = ([typemax(Int)], [-2])
      @test f(point) === Inf
      @test allocs(f, point) == 0
    end
  end

  @testset "warm starts across magnitudes" begin
    for T in (Float32, Float64), wide in (false, true)
      A = wide ? T[1 0] : ones(T, 1, 1)
      tiny = T === Float32 ? T(1e-30) : T(1e-200)
      for B in (A, GraphOperator(A))
        f = IndGraph(B; operator = true)
        prox(f, (fill(T(2), size(A, 2)), T[0]))
        point = (fill(tiny, size(A, 2)), T[0])
        p = prox(f, point)
        @test f.converged
        @test p[1][1] ≈ tiny / 2 rtol = 8eps(T) atol = 0
        @test p[2][1] ≈ tiny / 2 rtol = 8eps(T) atol = 0
      end
      # Keep the pair at unit scale while the system's rhs becomes tiny.
      small_A = wide ? T[tiny 0] : reshape([tiny], 1, 1)
      f = IndGraph(small_A; operator = true)
      warm = wide ? (zeros(T, 2), T[2]) : (T[2], T[0])
      prox(f, warm)
      point = wide ? (T[1, 0], T[0]) : (T[0], T[1])
      p = prox(f, point)
      @test f.converged
      @test p[1][1] ≈ (wide ? one(T) : tiny) rtol = 8eps(T) atol = 0
      @test p[2][1] ≈ (wide ? tiny : zero(T)) rtol = 8eps(T) atol = 0
    end
  end

  @testset "scaled projection right-hand sides" begin
    for T in (Float32, Float64), wide in (false, true)
      A = wide ? T[1 0] : ones(T, 1, 1)
      largest = floatmax(T)
      point = wide ? (T[-largest, 0], T[largest]) : (T[largest], T[largest])
      for B in (A, sparse(A), GraphOperator(A)), operator in (false, true)
        !(B isa AbstractMatrix) && !operator && continue
        f = IndGraph(B; operator)
        for oracle in (prox!, value_and_prox!), alias in (false, true)
          z = deepcopy(point)
          out = alias ? z : map(similar, z)
          oracle(out, f, z, one(T))
          @test f.converged
          @test all(isfinite, stack(out))
          expected = wide ? zeros(T, 3) : ones(T, 2)
          @test stack(out) / largest ≈ expected rtol = 8eps(T) atol = 8eps(T)
        end
        if !(B isa SparseMatrixCSC)
          out = map(similar, point)
          @test allocs(prox!, out, f, point, one(T)) == 0
          @test allocs(value_and_prox!, out, f, point, one(T)) == 0
        end
      end
    end
  end
end

end
