module LinearShapesTest
using Test, ProximalBase, LinearAlgebra, SparseArrays, Random
const PB = ProximalBase

# The shape contract of LinearFunction(c, d): c and the argument x have the same
# dimensions, size(c) == size(x), where a SymmetricSparseIterate has the size of the
# p × p matrix it represents. Every oracle applies it, and refuses any other pair with a
# DimensionMismatch before it writes, whatever dot would accept for the array types.

# the observable state of an output: values, and the working set of an iterate
state(x::AbstractArray) = copy(x)
state(x::SparseIterate) = (Array(x), collect(activecoordinates(x)), nactive(x))
state(x::BlockIterate) = (Vector(x), collect(activecoordinates(x)), nactive(x))
state(x::SymmetricSparseIterate) = (Matrix(x), collect(activecoordinates(x)), nactive(x))

dense(x::SymmetricSparseIterate) = Matrix(x)
dense(x::AbstractArray) = Array(x)
# ⟨c, x⟩ + d in BigFloat, entry by entry over the matrix or array that x represents
reference(c, d, x) = sum(big.(Array(c)) .* big.(dense(x))) + big(d)
close_to(v, r, T, n) = abs(big(v) - r) <= 4 * n * eps(T) * (sum(abs, big.(r)) + 1)

function sparse_iterate(T, dims...; rng)
  x = SparseIterate(T, dims...)
  for i in 1:2:length(x)
    x[i] = randn(rng, T)
  end
  x
end
function symmetric_iterate(T, p; rng)
  x = SymmetricSparseIterate(T, p)
  for c in 1:p, r in c:p
    isodd(r + c) || (x[r, c] = randn(rng, T))
  end
  x
end
symmetric(T, p; rng) = (a = randn(rng, T, p, p); a + a')

# every oracle on a valid pair: values against the BigFloat reference, the gradient c, the prox x - γc
function check_valid(f, x, out, T)
  c, d = f.c, f.d
  n = length(c)
  r = reference(c, d, x)
  γ = T(0.375)
  @test close_to(f(x), r, T, n)
  g = copy(out)
  @test gradient!(g, f, x) === g && dense(g) == Array(c)
  g = copy(out)
  @test close_to(value_and_gradient!(g, f, x), r, T, n) && dense(g) == Array(c)
  # the allocating oracles write into similar(x), which a Diagonal argument cannot hold
  allocating = !(x isa Diagonal)
  allocating && @test dense(gradient(f, x)) == Array(c)
  if allocating
    v, g2 = value_and_gradient(f, x)
    @test close_to(v, r, T, n) && dense(g2) == Array(c)
  end
  y = copy(out)
  expected = dense(x) .- γ .* Array(c)
  @test prox!(y, f, x, γ) === y && dense(y) ≈ expected
  y = copy(out)
  @test close_to(value_and_prox!(y, f, x, γ), reference(c, d, y), T, n) && dense(y) ≈ expected
  allocating && @test dense(prox(f, x, γ)) ≈ expected
  # a scaled linear function forwards the same check and the same kernels
  s = T(2) * f
  @test close_to(s(x), 2r, T, n)
  y = copy(out)
  @test dense(prox!(y, s, x, γ)) ≈ dense(x) .- 2γ .* Array(c)
  g = copy(out)
  @test dense(gradient!(g, s, x)) == 2 .* Array(c)
end

# every oracle on an invalid pair: a DimensionMismatch, and the output, the argument and c are unchanged
function check_rejected(f, x, out, T)
  γ = T(0.375)
  xs, cs = state(x), copy(f.c)
  for h in (f, T(2) * f)
    @test_throws DimensionMismatch h(x)
    @test_throws DimensionMismatch prox(h, x, γ)
    @test_throws DimensionMismatch gradient(h, x)
    @test_throws DimensionMismatch value_and_gradient(h, x)
    for op! in (y -> gradient!(y, h, x), y -> value_and_gradient!(y, h, x),
                y -> prox!(y, h, x, γ), y -> value_and_prox!(y, h, x, γ))
      y = copy(out); before = state(y)
      @test_throws DimensionMismatch op!(y)
      @test state(y) == before
    end
  end
  @test state(x) == xs && f.c == cs
end

@testset "LinearFunction shape contract ($T)" for T in (Float32, Float64)
  rng = Xoshiro(894)
  d = T(0.625)
  A, B = randn(rng, T, 2, 3), randn(rng, T, 3, 2)          # the same length, different shapes
  v6 = randn(rng, T, 6)

  @testset "dense arrays of the same shape" begin
    check_valid(LinearFunction(v6, d), randn(rng, T, 6), zeros(T, 6), T)
    check_valid(LinearFunction(A, d), randn(rng, T, 2, 3), zeros(T, 2, 3), T)
    check_valid(LinearFunction(randn(rng, T, 2, 2, 3), d), randn(rng, T, 2, 2, 3), zeros(T, 2, 2, 3), T)
    # views and wrappers of the same size take their own dot and the same checks
    check_valid(LinearFunction(view(randn(rng, T, 4, 5), 1:2, 2:4), d), randn(rng, T, 2, 3), zeros(T, 2, 3), T)
    check_valid(LinearFunction(copy(B'), d), view(randn(rng, T, 3, 2)', :, :), zeros(T, 2, 3), T)
    check_valid(LinearFunction(Diagonal(randn(rng, T, 3)), d), randn(rng, T, 3, 3), zeros(T, 3, 3), T)
    check_valid(LinearFunction(randn(rng, T, 3, 3), d), Diagonal(randn(rng, T, 3)), zeros(T, 3, 3), T)
    check_valid(LinearFunction(Symmetric(randn(rng, T, 3, 3)), d), randn(rng, T, 3, 3), zeros(T, 3, 3), T)
  end

  @testset "sparse arrays and sparse wrappers of the same shape" begin
    S = sprandn(rng, T, 5, 4, 0.4); X = sprandn(rng, T, 5, 4, 0.5); M = randn(rng, T, 5, 4)
    check_valid(LinearFunction(S, d), X, zeros(T, 5, 4), T)                          # CSC, CSC
    check_valid(LinearFunction(S, d), M, zeros(T, 5, 4), T)                          # CSC, dense
    check_valid(LinearFunction(M, d), X, zeros(T, 5, 4), T)                          # dense, CSC
    check_valid(LinearFunction(sparse(S'), d), copy(X'), zeros(T, 4, 5), T)
    check_valid(LinearFunction(S', d), sparse(M'), zeros(T, 4, 5), T)                # adjoint, CSC
    check_valid(LinearFunction(sparse(M'), d), transpose(X), zeros(T, 4, 5), T)      # CSC, transpose
    check_valid(LinearFunction(S', d), transpose(X), zeros(T, 4, 5), T)              # adjoint, transpose
    check_valid(LinearFunction(view(S, :, 2:4), d), view(X, :, 1:3), zeros(T, 5, 3), T)  # column subsets
    check_valid(LinearFunction(view(S, :, 1:3)', d), sparse(M[:, 1:3]'), zeros(T, 3, 5), T)
    v = sprandn(rng, T, 6, 0.5)
    check_valid(LinearFunction(v, d), randn(rng, T, 6), zeros(T, 6), T)
    check_valid(LinearFunction(v6, d), v, zeros(T, 6), T)
  end

  @testset "the same length with another shape is refused" begin
    S = sparse(A); SB = sparse(B)
    for (c, x) in ((A, B), (B, A), (S, SB), (S, B), (A, SB), (S', SB'), (transpose(S), SB'),
                   (S', transpose(SB)), (view(SB, :, 1:2), S), (A', A), (view(B, :, :), A),
                   (Diagonal(randn(rng, T, 3)), randn(rng, T, 1, 9)), (randn(rng, T, 1, 2, 3), A))
      check_rejected(LinearFunction(c, d), x, zeros(T, size(x)), T)
    end
  end

  @testset "a vector and a matrix of the same length are refused" begin
    for (c, x) in ((v6, A), (A, v6), (sparse(A), v6), (v6, sparse(A)), (sparsevec(v6), A),
                   (Diagonal(randn(rng, T, 3)), randn(rng, T, 9)), (randn(rng, T, 9), Diagonal(randn(rng, T, 3))))
      check_rejected(LinearFunction(c, d), x, zeros(T, size(x)), T)
    end
    # the literal case of the former linear-indexed prox: a coefficient vector and a 2 × 2 argument
    y = zeros(T, 2, 2)
    @test_throws DimensionMismatch prox!(y, LinearFunction(T[1, 2, 3, 4]), T[1 2; 3 4], T(2))
    @test y == zeros(T, 2, 2)
    # the exact alias of the argument is not changed either
    x = T[1 2; 3 4]
    @test_throws DimensionMismatch prox!(x, LinearFunction(T[1, 2, 3, 4]), x, T(2))
    @test x == T[1 2; 3 4]
  end

  @testset "SparseIterate and BlockIterate" begin
    xv = sparse_iterate(T, 6; rng); xm = sparse_iterate(T, 2, 3; rng)
    check_valid(LinearFunction(v6, d), xv, zeros(T, 6), T)
    check_valid(LinearFunction(v6, d), xv, SparseIterate(T, 6), T)
    check_valid(LinearFunction(sparse(A), d), xm, zeros(T, 2, 3), T)
    check_valid(LinearFunction(A, d), xm, SparseIterate(T, 2, 3), T)
    check_rejected(LinearFunction(A, d), xv, zeros(T, 6), T)
    check_rejected(LinearFunction(A, d), xv, sparse_iterate(T, 6; rng), T)
    check_rejected(LinearFunction(v6, d), xm, zeros(T, 2, 3), T)
    check_rejected(LinearFunction(B, d), xm, sparse_iterate(T, 2, 3; rng), T)
    groups = [[1, 4], [2, 3, 6], [5]]
    xb = BlockIterate(randn(rng, T, 6), groups)
    check_valid(LinearFunction(v6, d), xb, zeros(T, 6), T)
    check_valid(LinearFunction(v6, d), xb, BlockIterate(zeros(T, 6), groups), T)
    check_rejected(LinearFunction(A, d), xb, zeros(T, 6), T)
    check_rejected(LinearFunction(reshape(v6, 6, 1), d), xb, BlockIterate(randn(rng, T, 6), groups), T)
  end

  @testset "SymmetricSparseIterate: the p × p matrix it represents" begin
    p = 4
    x = symmetric_iterate(T, p; rng)
    C = symmetric(T, p; rng)
    for c in (C, sparse(C), Symmetric(C, :L), Symmetric(C, :U))
      check_valid(LinearFunction(c, d), x, SymmetricSparseIterate(T, p), T)      # owner output
      check_valid(LinearFunction(c, d), x, zeros(T, p, p), T)                     # dense output
    end
    # a dense symmetric argument written into a symmetric output
    xm = Matrix(x)
    f = LinearFunction(C, d)
    y = SymmetricSparseIterate(T, p)
    @test Matrix(prox!(y, f, xm, T(0.5))) ≈ xm .- T(0.5) .* C
    # coefficients with p² entries but not the p × p shape are refused, whatever the output
    for c in (vec(C), reshape(C, 1, p * p), reshape(C, 2, 8), randn(rng, T, p + 1, p), sparse(vec(C)))
      for out in (SymmetricSparseIterate(T, p), symmetric_iterate(T, p; rng), zeros(T, p, p))
        check_rejected(LinearFunction(c, d), x, out, T)
      end
      # the zero-coordinate screen reads the same extent
      @test_throws DimensionMismatch kkt_violation(LinearFunction(c, d), zeros(T, p, p), x, 1)
    end
    @test kkt_violation(f, zeros(T, p, p), x, 1) == abs(C[1, 1])
  end

  @testset "numerical edge cases on valid shapes" begin
    E = exponent(floatmax(T)); a = ldexp(one(T), E ÷ 2 + 4)
    # a dot that overflows in a partial sum, at a representable value, in matrix and sparse forms
    c = T[a a; 1 -a]; x = T[a 1; -a a]
    r = reference(c, T(0.5), x)
    for (cc, xx) in ((c, x), (sparse(c), x), (c, sparse(x)), (c', copy(x')))
      @test close_to(LinearFunction(cc, T(0.5))(xx), r, T, 4)
    end
    xs = SparseIterate(T, 2, 2); copyto!(xs, x)
    @test close_to(LinearFunction(c, T(0.5))(xs), r, T, 4)
    # a symmetric iterate counts an off-diagonal coordinate at both of its positions
    p = 3
    xo = SymmetricSparseIterate(T, p); xo[2, 1] = a; xo[3, 1] = -a; xo[3, 3] = T(2)
    cm = fill(a, p, p); cm[3, 3] = T(5)
    @test close_to(LinearFunction(cm, T(1))(xo), reference(cm, T(1), xo), T, p * p)
    # genuine infinities and NaNs follow arithmetic
    @test LinearFunction(T[1 1; 1 1])(T[Inf 1; 1 1]) === T(Inf)
    @test isnan(LinearFunction(T[1 1; 1 1])(T[Inf 1; -Inf 1]))
    @test isnan(LinearFunction(T[NaN 1; 1 1])(T[1 1; 1 1]))
    @test LinearFunction(T[1 2; 3 4], T(Inf))(ones(T, 2, 2)) === T(Inf)
    @test LinearFunction(sparse(T[1 0; 0 1]))(T[0 1; 1 -0.0]) === zero(T)
    # a refused shape is refused before any of these values is computed
    @test_throws DimensionMismatch LinearFunction(T[NaN, 1, 1, 1])(T[1 1; 1 1])
    @test_throws DimensionMismatch LinearFunction(T[a, a, 1, -a])(x)
  end
end

end
