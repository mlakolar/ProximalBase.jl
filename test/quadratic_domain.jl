module QuadraticDomainTest

# QuadraticFunction for a symmetric A that need not be positive semidefinite: the oracles, the step
# domain of the proximal operator, and the trait. References are independent of the package's spectral
# formula: analytic diagonal solutions, and BigFloat solves of (I + γA) y = x - γb with the step
# converted to the function's type first, as the kernel does.
#
# Tolerance used for the proximal point: relative error (2-norm) at most 100 eps(T) cond(I + γA), where
# the condition number is that of the BigFloat matrix, computed in Float64.

using Test, ProximalBase, LinearAlgebra
const PB = ProximalBase

# the exact solution of (I + γA) y = x - γb for the stored A, b and the converted step γ
function refprox(A, b, x, γ::T) where {T}
  g = big(γ)
  Ab = big.(Matrix(A))
  rhs = big.(x) .- (b === nothing ? zero(Ab[:, 1]) : g .* big.(b))
  (I + g * Ab) \ rhs
end
condmat(A, γ) = cond(Matrix{Float64}(I + big(γ) * big.(Matrix(A))))
relerr(y, r) = Float64(norm(big.(y) .- r) / norm(r))
# the proximal point is within 100 eps(T) cond(I + γA) of the reference
function proxok(f, A, b, x, γ::T) where {T}
  y = prox(f, x, γ)
  all(isfinite, y) && relerr(y, refprox(A, b, x, γ)) <= 100 * Float64(eps(T)) * condmat(A, γ)
end

# a rotation of the plane (i, j) by the angle θ, built in BigFloat
function givens(n, i, j, θ)
  R = Matrix{BigFloat}(I, n, n)
  c, s = cos(θ), sin(θ)
  R[i, i] = c; R[j, j] = c; R[i, j] = -s; R[j, i] = s
  R
end
# R D R' with R a product of rotations kept in BigFloat: only the final product is rounded to T and
# symmetrized, and the reference is built from the stored A and not from D
function rotated(::Type{T}, d) where {T}
  n = length(d)
  R = givens(n, 1, 2, big"0.7") * givens(n, 3, 4, big"-1.1") * givens(n, 2, 3, big"0.4") * givens(n, 1, 4, big"0.9")
  M = R * Diagonal(big.(d)) * R'
  Matrix{T}((M + M') / 2)
end

# X X' for an integer X: exact data of a singular positive semidefinite matrix when X has fewer columns than rows
gram(::Type{T}, X) where {T} = Matrix{T}(X * X')

# the allocations of a second call
allocs(y, f, v, γ) = (prox!(y, f, v, γ); @allocated prox!(y, f, v, γ))

for T in (Float32, Float64)
  v = T[1, 2]
  Adiag = T[-2 0; 0 1]
  # the proximal point of the diagonal example, for a step that the guard accepts
  analytic(γ, b = T[0, 0]) = (v .- γ .* b) ./ (1 .+ γ .* diag(Adiag))

  @testset "$T traits" begin
    f = QuadraticFunction(Adiag)
    @test isconvex(QuadraticFunction) == false
    @test isconvex(typeof(f)) == false
    @test isconvex(f) == false
    @test issmooth(f) && isproximable(f) && !isseparable(f)
    s = Scaled(f, T(2))
    @test isconvex(s) == false
    @test issmooth(s) && isproximable(s)
  end

  @testset "$T positive semidefinite matrices keep their behavior" begin
    x = T[1, -2, 3, 0.5, -1, 2]
    b = T[0.5, 1, -1, 2, 0, 3]
    X = [1 0 2; 2 1 -1; 0 3 1; 1 1 1; -2 0 1; 1 -1 0]
    cases = (
      Diagonal(T[1, 2, 3, 4, 5, 6]),
      gram(T, X) + T(0.5) * I,
      rotated(T, T[0.5, 1, 2, 3]),
      gram(T, X),                       # rank 3, n = 6, exact integer data
    )
    for A in cases
      n = size(A, 1)
      xx = n == 4 ? x[1:4] : x
      bb = n == 4 ? b[1:4] : b
      for bvec in (nothing, bb), γ in (T(1e-3), T(0.5), T(1), T(1e3))
        f = QuadraticFunction(A, bvec)
        λ = eigvals(Symmetric(Matrix(A)))[1]
        # the singular matrix has a computed λmin of either sign; at these steps 1 + γλmin > 0
        λ < 0 && @test 1 + γ * λ > 0
        @test proxok(f, A, bvec, xx, γ)
        @test proxok(f, A, bvec, xx, γ)      # warm
      end
    end
    for bvec in (nothing, T[1, -2, 0.5])
      f = QuadraticFunction(zeros(T, 3, 3), bvec)
      xx = T[3, -1, 2]
      for γ in (T(1e6), floatmax(T) / 4)
        y = prox(f, xx, γ)
        @test f.Λ == zeros(T, 3)
        @test y == (bvec === nothing ? xx : xx .- γ .* bvec)
      end
    end
  end

  @testset "$T indefinite matrices, accepted steps" begin
    f = QuadraticFunction(Adiag)
    @test prox(f, v, T(0.25)) ≈ T[2, 1.6]
    @test prox(f, v, T(0.25)) ≈ analytic(T(0.25))
    b = T[0.5, -1]
    fb = QuadraticFunction(Adiag, b)
    for γ in (T(0.25), T(0.1), T(1e-3), prevfloat(T(0.5)))
      @test prox(fb, v, γ) ≈ analytic(γ, b)
      @test proxok(fb, Adiag, b, v, γ)
    end
    @test all(isfinite, prox(f, v, prevfloat(T(0.5))))
    # a subnormal step: the product 2 nextfloat(zero(T)) with λmin = -2 is exact and below 1
    @test 2 * nextfloat(zero(T)) < 1
    @test prox(f, v, nextfloat(zero(T))) ≈ v
    # a product that rounds to zero: 0.5 nextfloat(zero(T)) is a tie that rounds to even, that is to 0
    @test T(0.5) * nextfloat(zero(T)) == 0
    @test prox(QuadraticFunction(Diagonal(T[-0.5, 1])), v, nextfloat(zero(T))) ≈ v
    # a rotated dense matrix with eigenvalues -3, -0.5, 1, 4 (up to the rounding of the entries)
    A = rotated(T, T[-3, -0.5, 1, 4])
    vv = T[1, -2, 0.5, 3]
    bb = T[0.25, 1, -1, 2]
    g = QuadraticFunction(A, bb)
    γ = T(0.9) / 3
    @test proxok(g, A, bb, vv, γ)
    y = prox(g, vv, γ)
    resid = norm(big.(Matrix(I + big(γ) * big.(A)) * big.(y) .- (big.(vv) .- big(γ) .* big.(bb))))
    @test Float64(resid) <= 100 * Float64(eps(T)) * condmat(A, γ) * Float64(norm(big.(vv)))
  end

  @testset "$T indefinite matrices, rejected steps" begin
    f = QuadraticFunction(Adiag)
    for γ in (T(0.5), T(1), nextfloat(T(0.5)), floatmax(T), T(Inf), T(NaN))
      @test_throws DomainError prox(f, v, γ)
    end
    @test f.Λ[1] == -2
    A = rotated(T, T[-3, -0.5, 1, 4])
    g = QuadraticFunction(A)
    λ1 = eigvals(Symmetric(Matrix(A)))[1]
    @test_throws DomainError prox(g, T[1, -2, 0.5, 3], T(1.1) / abs(λ1))
    # a Float64 or Int step is converted to T first; 0.5 and 1 are the same number in either type
    @test_throws DomainError prox(f, v, 0.5)
    @test_throws DomainError prox(f, v, 1)
    # the message names the step and λmin
    e = try prox(f, v, T(1)); nothing catch err; err end
    @test e isa DomainError && occursin("λmin", e.msg)
  end

  @testset "$T the output is preserved on DomainError" begin
    γ = T(1)
    for warm in (false, true)
      mk() = (f = QuadraticFunction(Adiag); warm && prox(f, v, T(0.25)); f)
      f = mk()
      @test isempty(f.Λ) == !warm
      y = fill(T(7), 2)
      @test_throws DomainError prox!(y, f, v, γ)
      @test y == fill(T(7), 2)
      z = copy(v)
      @test_throws DomainError prox!(z, mk(), z, γ)
      @test z == v
      y = fill(T(7), 2)
      @test_throws DomainError value_and_prox!(y, mk(), v, γ)
      @test y == fill(T(7), 2)
      z = copy(v)
      @test_throws DomainError value_and_prox!(z, mk(), z, γ)
      @test z == v
      @test_throws DomainError prox(mk(), v, γ)
    end
  end

  @testset "$T oracles on an indefinite matrix" begin
    A = rotated(T, T[-3, -0.5, 1, 4])
    b = T[0.25, 1, -1, 2]
    c = T(0.75)
    f = QuadraticFunction(A, b, c)
    for x in (T[1, -2, 0.5, 3], T[0, 0, 0, 0], T[100, -30, 7, 1])
      Ab = big.(A); xb = big.(x); bb = big.(b)
      ref = xb' * Ab * xb / 2 + bb' * xb + big(c)
      mag = abs.(xb)' * abs.(Ab) * abs.(xb) / 2 + abs.(bb)' * abs.(xb) + abs(big(c))
      @test abs(big(f(x)) - ref) <= 8 * eps(T) * mag
      g = similar(x)
      gradient!(g, f, x)
      @test all(abs.(big.(g) .- (Ab * xb .+ bb)) .<= 8 * eps(T) * (abs.(Ab) * abs.(xb) .+ abs.(bb)))
      g2 = similar(x)
      val = value_and_gradient!(g2, f, x)
      @test val == f(x) && g2 == g && g == gradient(f, x)
    end
    x = T[1, -2, 0.5, 3]
    γ = T(0.9) / 3
    y = similar(x)
    val = value_and_prox!(y, f, x, γ)
    @test y == prox(f, x, γ)
    @test val == f(y)
  end

  @testset "$T Scaled" begin
    f = QuadraticFunction(Adiag)
    s = Scaled(f, T(2))
    @test prox(s, v, T(0.2)) == prox(f, v, T(0.4))
    y = similar(v)
    @test value_and_prox!(y, s, v, T(0.2)) ≈ 2 * f(y)
    @test y == prox(f, v, T(0.4))
    for γ in (T(0.25), T(0.3))
      y = fill(T(7), 2)
      @test_throws DomainError prox!(y, s, v, γ)
      @test y == fill(T(7), 2)
      @test_throws DomainError value_and_prox!(y, s, v, γ)
      @test y == fill(T(7), 2)
      @test_throws DomainError prox(s, v, γ)
    end
    # a nested wrapper whose factors are not folded: the product 8 is the effective factor, and the
    # leaf rejects the effective step 1/2 (the step 1/16 times the factor 8) and accepts 1/4
    inner = Scaled(f, T(2))
    nested = Scaled{typeof(inner), T}(inner, T(4))
    @test nested isa Scaled{<:Scaled} && !PB._chained(nested)
    @test prox(nested, v, T(1) / 32) == prox(f, v, T(1) / 4)
    for γ in (T(1) / 16, T(1) / 8)
      y = fill(T(7), 2)
      @test_throws DomainError prox!(y, nested, v, γ)
      @test_throws DomainError value_and_prox!(y, nested, v, γ)
      @test y == fill(T(7), 2)
    end
    # a nested wrapper whose factors 2^a1 and 2^a2 have a product below floatmin: it takes the chain route,
    # and the effective step 2^(a1 + a2 + k) of the step 2^k is 1/4 (accepted) or 1/2 (rejected)
    e0 = exponent(floatmin(T))
    a1 = e0 ÷ 2
    a2 = e0 - 1 - a1
    chain = Scaled(Scaled(f, ldexp(one(T), a1)), ldexp(one(T), a2))
    @test chain isa Scaled{<:Scaled} && PB._chained(chain)
    γa = ldexp(one(T), -e0 - 1)
    γr = ldexp(one(T), -e0)
    @test prox(chain, v, γa) == prox(f, v, T(0.25))
    y = similar(v)
    value_and_prox!(y, chain, v, γa)
    @test y == prox(f, v, T(0.25))
    y = fill(T(7), 2)
    @test_throws DomainError prox!(y, chain, v, γr)
    @test_throws DomainError value_and_prox!(y, chain, v, γr)
    @test y == fill(T(7), 2)
  end

  @testset "$T mutation and reset_cache!" begin
    for wrapped in (false, true)
      A = T[2 0; 0 1]
      f = QuadraticFunction(A)
      h = wrapped ? Scaled(f, T(2)) : f
      m = wrapped ? 2 : 1                    # the effective step is m γ
      big_step = T(10)
      @test prox(h, v, big_step) ≈ prox(f, v, m * big_step)
      A[1, 1] = -2
      # the documented stale-cache behavior (reset_cache! is required after a change), not a guarantee
      @test prox(h, v, big_step) ≈ prox(QuadraticFunction(T[2 0; 0 1]), v, m * big_step)
      reset_cache!(h)
      @test_throws DomainError prox(h, v, big_step)
      @test prox(h, v, T(0.25) / m) ≈ T[2, 1.6]
      A[1, 1] = 3
      reset_cache!(h)
      @test prox(h, v, big_step) ≈ T[v[1] / (1 + 3 * m * big_step), v[2] / (1 + m * big_step)]
      @test proxok(f, A, nothing, v, m * big_step)
    end
  end

  @testset "$T edge cases" begin
    f = QuadraticFunction(zeros(T, 0, 0))
    @test prox(f, T[], T(1)) == T[]
    y = T[]
    @test value_and_prox!(y, f, T[], T(1)) == 0
  end

  # unsupported input: A is not symmetric. The constructor accepts it and the prox reads the upper
  # triangle of Symmetric(Matrix(A)); this records the current behavior and is not part of the contract
  @testset "$T nonsymmetric A (unsupported precondition, current behavior)" begin
    A = T[1 2; 0 1]
    f = QuadraticFunction(A)
    g = QuadraticFunction(Matrix(Symmetric(A)))
    @test prox(f, v, T(0.5)) == prox(g, v, T(0.5))
    @test_throws DomainError prox(f, v, T(2))
  end

  @testset "$T allocations of a warm prox" begin
    y = zeros(T, 4)
    w = T[1, -2, 0.5, 3]
    psd = QuadraticFunction(rotated(T, T[0.5, 1, 2, 3]), T[1, 2, 3, 4])
    dia = QuadraticFunction(Diagonal(T[1, 2, 3, 4]))
    ind = QuadraticFunction(rotated(T, T[-3, -0.5, 1, 4]), T[1, 2, 3, 4])
    @test allocs(y, psd, w, T(1)) == 0
    @test allocs(y, dia, w, T(1)) == 0
    @test allocs(y, ind, w, T(0.1)) == 0
  end
end

# The step is converted to the element type before the check. For Float32 data and the Float64 step 1/41,
# the Float64 product with 41 is not below 1 but the converted step times 41 is (accepted), and
# prevfloat(0.5) converts to the boundary step 0.5f0 (rejected). Both are verified in the test.
@testset "the step is checked after conversion to Float32" begin
  γ = 1 / 41
  @test Float32(γ) * 41f0 < 1
  @test γ * 41 >= 1
  f = QuadraticFunction(Diagonal(Float32[-41, 1]))
  v = Float32[1, 2]
  y = prox(f, v, γ)
  @test f.Λ[1] == -41f0
  @test all(isfinite, y)
  @test relerr(y, refprox(Diagonal(Float32[-41, 1]), nothing, v, Float32(γ))) <= 100 * Float64(eps(Float32)) * condmat(Diagonal(Float32[-41, 1]), Float32(γ))
  γ = prevfloat(0.5)
  @test Float32(γ) == 0.5f0
  @test_throws DomainError prox(QuadraticFunction(Diagonal(Float32[-2, 1])), v, γ)
end

end
