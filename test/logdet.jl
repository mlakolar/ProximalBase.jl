using Test, ProximalBase, LinearAlgebra, OffsetArrays

@testset "LogDetLoss" begin
  for T in (Float32, Float64)
    @testset "$T" begin
      S = T[1 0.5; 0.5 -1]
      X = T[2 1; 1 3]
      w, γ = T(0.75), T(0.5)
      f = LogDetLoss(S, w)
      @test issmooth(f) && isconvex(f) && isproximable(f) && !isseparable(f)
      @test lipschitz(f) === T(Inf)

      v = w * (T(0) - log(T(5)))
      G = w * (S - T[3 -1; -1 2] / 5)
      @test f(X) isa T
      @test f(X) ≈ v
      @test f(X) ≈ w * (tr(S * X) - logdet(cholesky(Symmetric(X))))
      @test gradient(f, X) ≈ G
      value, grad = value_and_gradient(f, X)
      @test value === f(X)
      @test grad ≈ G
      @test issymmetric(grad)
      D = T[0.5 1; 1 -0.75]
      h = cbrt(eps(T))
      @test (f(X + h * D) - f(X - h * D)) / (2h) ≈ dot(G, D) rtol=20h^2

      # The quadratic proximal term makes this well posed even for indefinite S
      # and when the supplied input has a skew part and a negative eigenvalue.
      V = T[-1 2; 0 1]
      Z = prox(f, V, γ)
      @test Z isa Matrix{T}
      @test issymmetric(Z) && isposdef(Z)
      @test norm(w * (S - inv(Z)) + (Z - (V + V') / 2) / γ) ≤ 80eps(T)
      @test prox(f, V, γ) ≈ prox(f, (V + V') / 2, γ)
      @test prox(f, X) ≈ prox(f, X, one(T))

      out = copy(V)
      @test prox!(out, f, out, γ) === out
      @test out ≈ Z
      copyto!(out, V)
      @test value_and_prox!(out, f, out, γ) === f(out)
      @test out ≈ Z
      backing = fill(T(-7), 4, 5)
      viewout = @view backing[1:2:3, 2:2:4]
      @test prox!(viewout, f, V, γ) === viewout
      @test value_and_prox!(viewout, f, V, γ) === f(viewout)
      @test viewout ≈ Z
      @test gradient!(viewout, f, X) === viewout
      @test viewout ≈ G
      @test all(==(-7), backing[[2, 4], :])
      @test all(==(-7), backing[:, [1, 3, 5]])

      owner = SymmetricSparseIterate(X)
      @test f(owner) === f(X)
      @test gradient(f, owner) ≈ G
      @test value_and_gradient(f, owner) == value_and_gradient(f, X)
      @test gradient!(out, f, owner) ≈ G
      @test value_and_gradient!(out, f, owner) === f(X)
      expected = prox(f, X, γ)
      @test prox(f, owner, γ) ≈ expected
      @test prox!(out, f, owner, γ) ≈ expected
      @test value_and_prox!(out, f, owner, γ) === f(out)
      @test prox(f, matrixview(owner), γ) ≈ expected
      @test prox(f, Symmetric(X), γ) ≈ expected
      @test prox(f, Hermitian(X), γ) ≈ expected
      @test_throws MethodError gradient!(owner, f, X)
      @test_throws MethodError value_and_gradient!(owner, f, X)
      @test_throws MethodError prox!(owner, f, X, γ)
      @test_throws MethodError value_and_prox!(owner, f, X, γ)
      @test Matrix(owner) == X

      for (bad, value) in ((T[1 1; 1 1], T(Inf)), (-X, T(Inf)), (V, T(Inf)),
                           (T[1 1; 0 1], T(Inf)), (T[Inf 0; 0 1], T(Inf)),
                           (T[Inf 1; 0 NaN], T(NaN)))
        @test isequal(f(bad), value)
        @test isequal(value_and_gradient!(out, f, bad), value)
        @test all(isnan, out)
        @test isequal(value_and_prox!(out, f, bad, -zero(T)), value)
        @test isequal(out, bad)
        if any(!isfinite, bad)
          @test_throws ArgumentError prox!(out, f, bad, γ)
          @test isequal(out, bad)
        end
      end

      @test_throws DimensionMismatch LogDetLoss(zeros(T, 2, 3))
      @test_throws ArgumentError LogDetLoss(zeros(T, 0, 0))
      @test_throws ArgumentError LogDetLoss(complex.(S))
      @test_throws ArgumentError LogDetLoss(big.(S))
      @test_throws ArgumentError LogDetLoss(OffsetArray(S, 0:1, 1:2))
      for weight in (zero(T), -one(T), T(NaN), T(Inf))
        @test_throws ArgumentError LogDetLoss(S, weight)
      end
      for x in (T(NaN), T(Inf))
        @test_throws ArgumentError LogDetLoss(T[x 0; 0 1])
      end
      asymmetric = copy(S)
      asymmetric[1, 2] = nextfloat(asymmetric[2, 1])
      @test_throws ArgumentError LogDetLoss(asymmetric)
      @test LogDetLoss(Symmetric(asymmetric, :L))(X) == LogDetLoss(S)(X)
      @test LogDetLoss(Symmetric(asymmetric, :U))(X) == LogDetLoss(Matrix(Symmetric(asymmetric, :U)))(X)
      @test LogDetLoss(Symmetric(T[1 NaN; 0 -1], :L))(X) == LogDetLoss(T[1 0; 0 -1])(X)
      owned = copy(S)
      private = LogDetLoss(owned, w)
      fill!(owned, T(NaN))
      @test private(X) === f(X)

      fill!(out, T(-7))
      XT = T === Float32 ? Float64 : Float32
      mixed = fill(XT(-7), 2, 2)
      offsetX = OffsetArray(X, 0:1, 1:2)
      offsetY = OffsetArray(out, 1:2, 0:1)
      @test_throws DimensionMismatch f(zeros(T, 2, 3))
      @test_throws ArgumentError f(offsetX)
      for op in (prox!, value_and_prox!)
        for invalid in (-γ, T(NaN), T(Inf), -T(Inf), XT(γ))
          @test_throws ArgumentError op(out, f, X, invalid)
        end
        @test_throws DimensionMismatch op(out, f, zeros(T, 1, 1), γ)
        @test_throws DimensionMismatch op(view(out, :, 1:1), f, X, γ)
        @test_throws ArgumentError op(out, f, offsetX, γ)
        @test_throws ArgumentError op(offsetY, f, X, γ)
        @test_throws ArgumentError op(mixed, f, X, γ)
        @test_throws ArgumentError op(out, f, XT.(X), γ)
        @test_throws ArgumentError op(Symmetric(out), f, X, γ)
        @test_throws ArgumentError op(f.S, f, X, γ)
      end
      for op in (gradient!, value_and_gradient!)
        @test_throws ArgumentError op(X, f, X)
        @test_throws ArgumentError op(f.S, f, X)
        @test_throws ArgumentError op(mixed, f, X)
        @test_throws ArgumentError op(out, f, offsetX)
        @test_throws ArgumentError op(offsetY, f, X)
        @test_throws DimensionMismatch op(out, f, zeros(T, 2, 3))
        @test_throws DimensionMismatch op(view(out, :, 1:1), f, X)
      end
      @test all(==(-7), out) && all(==(-7), mixed)
      @test X == T[2 1; 1 3] && f.S == S

      scalar = LogDetLoss(fill(T(2), 1, 1), w)
      @test scalar(fill(T(3), 1, 1)) ≈ w * (T(6) - log(T(3)))
      @test only(gradient(scalar, fill(T(3), 1, 1))) ≈ w * (T(2) - inv(T(3)))
      diagonal = LogDetLoss(Diagonal(T[1, 2]))
      @test diagonal(Diagonal(T[2, 4])) ≈ T(10) - log(T(8))
      # The shifted eigenvalues are 0 and 2 at a unit step.
      @test prox(diagonal, Diagonal(T[1, 4])) ≈ Diagonal(T[1, 1 + sqrt(T(2))])
      # In the (1,1), (1,-1) directions the shifted eigenvalues are 2 and 0.
      rotated = T[1 1; 1 1]
      @test prox(LogDetLoss(zeros(T, 2, 2)), rotated) ≈
        T[2 + sqrt(T(2)) sqrt(T(2)); sqrt(T(2)) 2 + sqrt(T(2))] / 2
      tiny = floatmin(T)
      near_boundary = Diagonal(T[tiny, 1])
      @test diagonal(near_boundary) ≈ tiny + T(2) - log(tiny)
      @test gradient(diagonal, near_boundary)[1, 1] ≈ one(T) - inv(tiny)

      # #280: squaring d would overflow, or the positive root would be
      # erased by cancellation. The oracle computes from the quadratic
      # equation in independent precision, with enough digits for both signs.
      far = T === Float32 ? T(2)^100 : T(2)^800
      barrier = LogDetLoss(zeros(T, 1, 1))
      for d in (-far, far)
        ref = setprecision(BigFloat, 2048) do
          D = BigFloat(d)
          T((D + sqrt(D^2 + 4)) / 2)
        end
        @test only(prox(barrier, fill(d, 1, 1))) ≈ ref rtol=8eps(T)
      end
      rounded = fill(far, 2, 2)
      matrix_barrier = LogDetLoss(zeros(T, 2, 2))
      @test value_and_prox!(out, matrix_barrier, rounded) === matrix_barrier(out)
      @test isfinite(out[1, 1]) && out[1, 1] ≈ far
      @test value_and_prox!(out, matrix_barrier, Diagonal(T[-far, 1]), nextfloat(zero(T))) === T(Inf)
      @test iszero(out[1, 1]) && iszero(out[1, 2]) && iszero(out[2, 1])
      @test out[2, 2] ≈ one(T) rtol=4eps(T)

      # A representable scalar root even when γw rounds to zero or infinity.
      for scale in (far, inv(far))
        @test only(prox(LogDetLoss(zeros(T, 1, 1), scale), zeros(T, 1, 1), scale)) ≈ scale rtol=8eps(T)
      end
      # The trace overflows in T before a finite scaling is applied.
      m = floatmax(T)
      scaled = LogDetLoss(fill(m, 1, 1), T(0.125))
      ref = T((BigFloat(m) * 2 - log(big"2.0")) / 8)
      @test scaled(fill(T(2), 1, 1)) ≈ ref rtol=4eps(T)
      fill!(out, T(-7))
      @test_throws ArgumentError prox!(out, LogDetLoss(zeros(T, 2, 2)), fill(m, 2, 2), one(T))
      @test all(==(-7), out)
    end
  end
end
