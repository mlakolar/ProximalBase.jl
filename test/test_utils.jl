facts("shrink") do

  @fact shrink(3.0, 1.0) --> 2.0
  @fact shrink(-2.0, 1.0) --> -1.0
  @fact shrink(0.5, 1.0) --> 0.

  srand(1)
  p = 10
  x = randn(p)
  c = vecnorm(x) + 0.1
  @fact shrinkL2!(similar(x), x, c) --> zeros(p)
  c = c - 0.15
  @fact shrinkL2!(similar(x), x, c) --> max(1. - c / vecnorm(x), 0,) * x

  x = randn(p, 2*p)
  c = vecnorm(x) + 0.1
  @fact shrinkL2!(similar(x), x, c) --> zeros(p, 2*p)
  c = c - 0.15
  @fact shrinkL2!(similar(x), x, c) --> max(1. - c / vecnorm(x), 0,) * x
end



facts("multiply") do

  srand(1)
  n, m = 100, 100
  A = randn(n, m)
  b = randn(m)
  x = A * b
  xt = A' * b
  for i=1:n
    @fact A_mul_B_row(A, b, i) --> x[i]
    @fact At_mul_B_row(A, b, i) --> roughly(xt[i])
  end

  bs = sprand(m, 0.3)
  x = A * bs
  xt = A' * bs
  bi = convert(SparseIterate, bs)
  xi = A * bi
  xti = A' * bi
  for i=1:n
    @fact A_mul_B_row(A, bs, i) --> x[i]
    @fact At_mul_B_row(A, bs, i) --> roughly(xt[i])

    @fact A_mul_B_row(A, bi, i) --> x[i]
    @fact At_mul_B_row(A, bi, i) --> roughly(xt[i])

    @fact xi[i] --> x[i]
    @fact xti[i] --> xt[i]
  end

  # symmetric A * X * B
  X = randn(100, 50)
  Σx = Symmetric(cov(X))
  Y = randn(100, 50)
  Σy = Symmetric(cov(Y))
  Δ = sprand(50, 50, 0.1)
  Δ = (Δ + Δ') / 2.

  A = Σx * Δ * Σy
  Δs = convert(SymmetricSparseIterate, tril(Δ))
  @fact A_mul_X_mul_B(Σx, Δs, Σy) --> roughly(A)

  # symmetric A * U * U' * B
  X = randn(100, 50)
  Σx = Symmetric(cov(X))
  Y = randn(100, 50)
  Σy = Symmetric(cov(Y))
  U = randn(50, 5)

  A = (Σx * U)*(U' * Σy)
  @fact A_mul_UUt_mul_B(Σx, U, Σy) --> roughly(A)
end
