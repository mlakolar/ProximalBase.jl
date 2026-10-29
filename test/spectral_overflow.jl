module SpectralOverflowTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

const PB = ProximalBase

threshold_matrix(c::T, d::T) where {T} = T[c + d c - d; c - d c + d]
samebits(a::T, b::T) where {T} =
  reinterpret(T === Float32 ? UInt32 : UInt64, a) == reinterpret(T === Float32 ? UInt32 : UInt64, b)
samebits(a::AbstractArray, b::AbstractArray) = size(a) == size(b) && all(samebits.(a, b))

function check_rounded(v::T, reference) where {T}
  r = big(reference)
  k = iszero(r) ? exponent(floatmin(T)) : max(exponent(abs(r)), exponent(floatmin(T)))
  δ = 2ldexp(one(r), k - precision(T) + 1)
  lo, hi = r - δ, r + δ
  if isfinite(T(hi))
    @test isfinite(v) && lo <= big(v) <= hi
  elseif T(lo) === T(Inf)
    @test v === T(Inf)
  else
    @test v === T(Inf) || (isfinite(v) && lo <= big(v) <= hi)
  end
end

logdet_reference(f, X) = big(f.w) * (dot(big.(f.S), big.(Matrix(X))) - logdet(Symmetric(big.(Matrix(X)))))

# The proximal outputs before value recovery was added.
function legacy_prox!(y, f::NormL2, x, γ)
  T = eltype(y)
  iszero(f.λ) && return copyto!(y, x)
  c, nx = T(γ) * T(f.λ), T(PB._norm2(x))
  if nx > c
    if isinf(nx) && isfinite(c)
      G = eachindex(IndexLinear(), x)
      PB._shrink_group!(GroupNormL2(one(T), [G]), y, x, G, c)
    else
      y .= (1 - c / nx) .* x
    end
  else
    fill!(y, zero(T))
  end
  y
end

function legacy_prox!(Y, f::NuclearNorm, X, γ)
  T = eltype(Y)
  iszero(f.λ) && return copyto!(Y, X)
  c = T(γ) * T(f.λ)
  ws = PB._svd_workspace(f.ws, T, size(X)...)
  copyto!(ws.A, X)
  s, U, Vt = PB._svd!(ws)
  k = 0
  @inbounds for j in eachindex(s)
    t = s[j] - c
    t > 0 || break
    k = j
    @simd for i in axes(U, 1)
      U[i, j] *= t
    end
  end
  PB._mul_lowrank!(Y, U, Vt, k, ws)
end

function legacy_prox!(Y, f::TracePSD, X, γ)
  T = eltype(Y)
  c = T(γ) * T(f.λ)
  ws = PB._eig_workspace(f.eig, T, size(X, 1))
  copyto!(ws.A, X)
  λ, V = PB._eigen!(ws)
  PB._rebuild_sym_out!(Y, V, λ, t -> max(t - c, zero(T)), Y)
  Y
end

function legacy_prox!(Y, f::LogDetLoss{T}, X, γ) where {T}
  root_step = sqrt(T(γ)) * sqrt(f.w)
  A = f.eig.A
  if root_step > one(T)
    A .= (((X ./ 2 .+ transpose(X) ./ 2) ./ root_step) ./ root_step) ./ 2 .- f.S ./ 2
    t, U = PB._eigen!(f.eig)
    PB._rebuild_sym_out!(Y, U, t, s -> PB._logdet_scaled_root_half(s, root_step), f.C; logs = true)
  else
    A .= X ./ 4 .+ transpose(X) ./ 4 .- ((f.S ./ 2 .* root_step) .* root_step)
    t, U = PB._eigen!(f.eig)
    PB._rebuild_sym_out!(Y, U, t, s -> PB._logbarrier_root_half(s, root_step), f.C; logs = true)
  end
  Y
end

function legacy_gradient!(G, f::LogDetLoss, X)
  PB._logdet_chol!(f, X)
  C, S, w = f.C, f.S, f.w
  PB._potri!('U', C)
  @inbounds for j in axes(C, 2), i in 1:j
    g = w * (S[i, j] - C[i, j])
    G[i, j] = G[j, i] = g
  end
  G
end

function value_functions(::Type{T}) where {T}
  (IndBallRank(1), IndBallRank(1; partial = true), IndBallRank(2),
   IndPSD(), IndPSD(rank = 1), IndPSD(partial = true), IndPSD(rank = 1, partial = true),
   TracePSD(one(T)), TracePSD(one(T); partial = true),
   SingularValue(IndBallL0(1)), SingularValue(IndBallLinf(one(T))),
   SingularValue(NormL1(one(T))), SingularValue(NormL0(one(T))),
   Spectral(IndNonnegative()), Spectral(IndBox(-one(T), one(T))))
end

function logical_inputs(X::Matrix{T}) where {T}
  padded = fill(T(NaN), size(X, 1) + 2, size(X, 2) + 2)
  V = view(padded, 2:size(X, 1)+1, 2:size(X, 2)+1)
  copyto!(V, X)
  # an iterate stores the lower triangle of X explicitly, since X may not be symmetric
  (X, sparse(X), SparseIterate(X), SymmetricSparseIterate(Symmetric(X, :L)), V)
end

# The value formulas at the base, through the same LAPACK workspaces.
function base_value(f::IndBallRank, X)
  T = float(eltype(X))
  minimum(size(X)) <= f.r && return zero(T)
  ws = PB._svd_workspace(f.ws, T, size(X)...)
  copyto!(ws.A, X)
  s, = PB._svd!(ws; vectors = false)
  s[f.r + 1] <= sqrt(eps(T)) * max(one(T), s[1]) ? zero(T) : T(Inf)
end

function base_ispsd(f, X, r)
  T = float(eltype(X))
  n = LinearAlgebra.checksquare(X)
  PB._issymmetric_tol(X) || return false
  ws = PB._eig_workspace(f.eig, T, n)
  copyto!(ws.A, X)
  λ = PB._eigvals!(ws)
  n == 0 && return true
  tol = sqrt(eps(T)) * max(one(T), abs(λ[1]), abs(λ[n]))
  λ[1] >= -tol || return false
  r === nothing || count(>(tol), λ) <= r
end

base_value(f::IndPSD, X) = base_ispsd(f, X, f.rank) ? zero(eltype(X)) : eltype(X)(Inf)
base_value(f::TracePSD, X) =
  base_ispsd(f, X, nothing) ? convert(eltype(X), f.λ) * tr(X) : eltype(X)(Inf)

function base_value(F::SingularValue, X)
  T = float(eltype(X))
  ws = PB._svd_workspace(F.ws, T, size(X)...)
  copyto!(ws.A, X)
  s, = PB._svd!(ws; vectors = false)
  c = isempty(s) ? zero(T) : sqrt(eps(T)) * s[1]
  convert(T, F.f([v <= c ? zero(T) : v for v in s]))
end

function base_value(f::Spectral, X)
  T = float(eltype(X))
  PB._issymmetric_tol(X) || return T(Inf)
  ws = PB._eig_workspace(f.eig, T, LinearAlgebra.checksquare(X))
  copyto!(ws.A, X)
  convert(T, f.f(PB._eigvals!(ws)))
end

@testset "spectral overflow" begin
  for T in (Float32, Float64)
    @testset "$T" begin
      M = floatmax(T)
      τ = sqrt(eps(T))
      c = (T(3) / T(4)) * M
      indefinite = T[M M; M -M]
      psd = threshold_matrix(c, M / 8)

      @testset "audit" begin
        for partial in (false, true)
          for X in (indefinite, T[M M M; M -M M])
            @test IndBallRank(1; partial)(X) === T(Inf)
          end
          for X in (indefinite, T[M M; M M / 2])
            @test IndPSD(; partial)(X) === T(Inf)
            @test IndPSD(; rank = 1, partial)(X) === T(Inf)
            @test TracePSD(one(T); partial)(X) === T(Inf)
          end
          for X in (fill(M, 2, 2), fill(M, 2, 3))
            @test IndBallRank(1; partial)(X) === zero(T)
          end
          @test IndPSD(; partial)(fill(M, 2, 2)) === zero(T)
          @test IndPSD(; rank = 1, partial)(fill(M, 2, 2)) === zero(T)
          @test IndPSD(; partial)(psd) === zero(T)
          @test IndPSD(; rank = 1, partial)(psd) === T(Inf)
          @test IndPSD(; rank = 2, partial)(psd) === zero(T)
        end
        @test SingularValue(IndBallL0(1))(indefinite) === T(Inf)
        @test SingularValue(NormL1(one(T)))(indefinite) === T(Inf)
        @test SingularValue(NormL0(one(T)))(indefinite) === T(2)
        for X in (fill(M, 2, 2), fill(M, 2, 3))
          @test SingularValue(IndBallL0(1))(X) === zero(T)
          @test SingularValue(NormL0(one(T)))(X) === one(T)
        end
        @test NuclearNorm(one(T))(indefinite) === T(Inf)
        @test Spectral(IndNonnegative())(indefinite) === T(Inf)
        @test TracePSD(one(T))(fill(M, 2, 2)) === T(Inf)
      end

      @testset "thresholds" begin
        @test isinf(2c)
        for factor in (T(7) / T(8), T(9) / T(8)), sign in (-one(T), one(T))
          d = sign * factor * τ * c
          X = threshold_matrix(c, d)
          @test all(isfinite, X)
          inside = factor < one(T)
          expected_rank = inside ? zero(T) : T(Inf)
          expected_psd = sign < 0 && !inside ? T(Inf) : zero(T)
          for Z in logical_inputs(X)
            before = Matrix(Z)
            for partial in (false, true)
              @test IndBallRank(1; partial)(Z) === expected_rank
              @test IndPSD(; partial)(Z) === expected_psd
              @test IndPSD(; rank = 1, partial)(Z) === expected_rank
            end
            @test SingularValue(IndBallL0(1))(Z) === expected_rank
            @test SingularValue(NormL0(one(T)))(Z) === T(inside ? 1 : 2)
            @test isequal(Matrix(Z), before)
          end
        end
        for X in (psd, indefinite)
          X3 = zeros(T, 3, 3)
          X3[1:2, 1:2] .= X
          Z = SymmetricSparseIterate(X3)
          @test IndBallRank(1)(Z) === T(Inf)
          @test IndPSD()(Z) === (X === psd ? zero(T) : T(Inf))
          @test IndPSD(rank = 1)(Z) === T(Inf)
          @test IndPSD(rank = 2)(Z) === (X === psd ? zero(T) : T(Inf))
          @test TracePSD(one(T))(Z) === T(Inf)
          @test SingularValue(IndBallL0(1))(Z) === T(Inf)
          @test SingularValue(NormL0(one(T)))(Z) === T(2)
          @test Matrix(Z) == X3
        end
      end

      @testset "nonfinite entries" begin
        functions = filter(f -> PB._isindicator(f) || f isa TracePSD, value_functions(T))
        for v in (T(NaN), T(Inf), T(-Inf))
          for X in (T[v 0; 0 1], T[1 v; v 1]), Z in logical_inputs(X)
            before = Matrix(Z)
            for f in functions
              @test f(Z) === T(Inf)
            end
            @test isequal(Matrix(Z), before)
          end
          for Z in logical_inputs(fill(v, 1, 1))
            @test IndBallRank(1)(Z) === T(Inf)
          end
        end
      end

      @testset "logical storage" begin
        X = Matrix{T}(I, 2, 2)
        padded = fill(T(NaN), 4, 4)
        padded[2:3, 2:3] .= X
        V = view(padded, 2:3, 2:3)
        S = view(sparse(padded), 2:3, 2:3)
        U = Symmetric(T[1 0; NaN 1], :U)
        for Z in (V, S, U), f in value_functions(T)
          @test samebits(f(Z), f(X))
        end
        for Z in (SparseIterate(X), SymmetricSparseIterate(X))
          @test nnz(Z) < length(Z.nzval)
          Z.nzval[nnz(Z)+1:end] .= T(NaN)
          before = copy(Z.nzval)
          for f in value_functions(T)
            @test samebits(f(Z), f(X))
          end
          @test isequal(Z.nzval, before)
        end
      end

      @testset "empty inputs" begin
        for X in (zeros(T, 0, 0), zeros(T, 0, 2), zeros(T, 2, 0))
          @test IndBallRank(1)(X) === zero(T)
          @test SingularValue(IndBallL0(1))(X) === zero(T)
          @test SingularValue(NormL1(one(T)))(X) === zero(T)
          @test SingularValue(NormL0(one(T)))(X) === zero(T)
          for f in (IndPSD(), IndPSD(rank = 1), TracePSD(one(T)), Spectral(IndNonnegative()))
            if size(X, 1) == size(X, 2)
              @test f(X) === zero(T)
            else
              @test_throws DimensionMismatch f(X)
            end
          end
        end
      end

      @testset "ordinary values match the base bitwise" begin
        rng = Xoshiro(165)
        for (m, n) in ((3, 3), (4, 2), (2, 5))
          X = randn(rng, T, m, n)
          L = randn(rng, T, m, 1) * randn(rng, T, 1, n)
          for f in (IndBallRank(1), IndBallRank(2), SingularValue(IndBallL0(1)),
                    SingularValue(NormL1(one(T))), SingularValue(NormL0(one(T)))), Z in (X, L)
            before = copy(Z)
            @test samebits(f(Z), base_value(f, Z))
            @test Z == before
          end
        end
        for n in (2, 4)
          B = randn(rng, T, n, n)
          for X in (B + B', B * B', B[:, 1:1] * B[:, 1:1]', B)
            for Z in logical_inputs(X), f in value_functions(T)
              before = Matrix(Z)
              @test samebits(f(Z), base_value(f, Z))
              @test Matrix(Z) == before
            end
          end
        end
      end

      @testset "workspace reuse and zero allocations" begin
        ordinary = T[2 1; 1 2]
        for wrap in (identity, SymmetricSparseIterate), f in value_functions(T)
          A, B, C = wrap(ordinary), wrap(psd), wrap(indefinite)
          @testset "$(typeof(f)), $(typeof(A))" begin
            before = f(A)
            f(B)
            @test samebits(f(A), before)
            f(C)
            @test samebits(f(A), before)
            for (label, X) in zip(("ordinary", "overflow PSD", "overflow indefinite"), (A, B, C))
              @testset "$label" begin
                input = Matrix(X)
                @test allocs(f, X) == 0
                @test Matrix(X) == input
              end
            end
          end
        end
      end
    end
  end
end

@testset "weighted spectral values at overflow" begin
  for T in (Float32, Float64)
    @testset "$T" begin
      M = floatmax(T)
      D = Matrix{T}(M * I, 2, 2)
      P = fill(M, 2, 2)
      E = Matrix{T}(I, 2, 2)
      γ = T(1) / 8
      for λ in (zero(T), T(1) / 4, T(1) / 2, one(T))
        @testset "weight $λ" begin
          l1, l2 = big(λ) * 2big(M), big(λ) * sqrt(big(2)) * big(M)
          for X in (D, P), f in (TracePSD(λ), NuclearNorm(λ))
            @testset "$(nameof(typeof(f))), $(X === D ? "diagonal" : "constant")" begin
              check_rounded(f(X), l1)
            end
          end
          for F in (SingularValue, Spectral)
            check_rounded(F(NormL1(λ))(D), l1)
            check_rounded(F(NormL2(λ))(D), l2)
          end
          check_rounded(NormL1(λ)(T[M, M]), l1)
          check_rounded(NormL2(λ)(T[M, M]), l2)
          for f in (NormL2(λ), NuclearNorm(λ), TracePSD(λ))
            X = f isa NormL2 ? T[0 M; M 0] : D
            Y, expected = similar(X), similar(X)
            v = value_and_prox!(Y, f, X, γ)
            legacy_prox!(expected, f, X, γ)
            @test samebits(Y, expected)
            if f isa NormL2
              @test prox!(Y, f, X, γ) === Y
              @test samebits(Y, expected)
            end
            reference = f isa NormL2 ? big(λ) * sqrt(sum(abs2, big.(Y))) : big(λ) * tr(big.(Y))
            check_rounded(v, reference)
          end
          if iszero(λ)
            @test_throws ArgumentError LogDetLoss(E, λ)
          else
            f = LogDetLoss(E, λ)
            reference = big(λ) * (2big(M) - 2log(big(M)))
            check_rounded(f(D), reference)
            G, expected = similar(D), similar(D)
            check_rounded(value_and_gradient!(G, f, D), reference)
            legacy_gradient!(expected, f, D)
            @test samebits(G, expected)
            Y = similar(D)
            v = value_and_prox!(Y, f, D, γ)
            legacy_prox!(expected, f, D, γ)
            @test samebits(Y, expected)
            check_rounded(v, logdet_reference(f, Y))
          end
        end
      end

      @testset "scaling arithmetic" begin
        u = nextfloat(zero(T))
        check_rounded(NormL2(u)(T[M, M]), big(u) * sqrt(big(2)) * big(M))
        small = ldexp(M, -(precision(T) ÷ 2 + 1))
        x = vcat(T[M, M], fill(small, 2046))
        check_rounded(NormL2(T(1) / 4)(x), sqrt(sum(abs2, big.(x))) / 4)
        x = T[M, M]
        y, expected = similar(x), similar(x)
        f = NormL2(T(1) / 2)
        G = eachindex(x)
        n = PB._shrink_group!(GroupNormL2(one(T), [G]), expected, x, G, M / 2)
        @test isfinite(n)
        @test samebits(value_and_prox!(y, f, x, M), f.λ * n)
        @test samebits(y, expected)
        for S in (D, fill(M, 2, 2)), X in (E, T[1 1/4; 1/4 1] / 2)
          f = LogDetLoss(S, T(1) / 4)
          for wrap in (identity, sparse, SparseIterate, SymmetricSparseIterate)
            check_rounded(f(wrap(X)), logdet_reference(f, X))
          end
        end
        f = LogDetLoss(D, u)
        check_rounded(f(D), logdet_reference(f, D))
        for X in (E, SymmetricSparseIterate(E))
          f = LogDetLoss(-D, T(1) / 4)
          @test f(X) < 0
          check_rounded(f(X), -big(M) / 2)
          G, expected = similar(E), similar(E)
          check_rounded(value_and_gradient!(G, f, X), -big(M) / 2)
          legacy_gradient!(expected, f, X)
          @test samebits(G, expected)
        end
        f = LogDetLoss(zeros(T, 2, 2), T(1) / 4)
        @test f(2E) < 0
        check_rounded(f(2E), -log(big(2)) / 2)
      end

      @testset "nonfinite entries and zero weights" begin
        for v in (T(Inf), T(-Inf), T(NaN)), λ in (zero(T), -zero(T))
          X = T[v 0; 0 1]
          for f in (NormL1(λ), NormL2(λ), NuclearNorm(λ))
            # the domain of the spectral functions is the finite matrices
            @test f(X) === (f isa NuclearNorm ? T(Inf) : zero(T))
            Y = similar(X)
            @test value_and_prox!(Y, f, X, γ) === zero(T)
            @test samebits(Y, X)
          end
          @test TracePSD(λ)(X) === T(Inf)
          @test LogDetLoss(E, one(T))(X) === T(Inf)
        end
        for f in (NormL1(one(T)), NormL2(one(T)))
          @test f(T[Inf, 1]) === T(Inf)
          @test isnan(f(T[NaN, 1]))
        end
        @test samebits(TracePSD(one(T))(fill(-eps(T), 1, 1)), -eps(T))
        @test samebits(TracePSD(-zero(T))(E), -zero(T))
        @test samebits(TracePSD(zero(T))(fill(-eps(T), 1, 1)), -zero(T))
      end

      @testset "accepted partial values keep the partial path" begin
        X = zeros(T, 64, 64)
        X[1, 1] = T(4)
        for F in (NuclearNorm, TracePSD), λ in (M, T(1) / 4)
          f = F(λ; partial = true)
          Y = fill(-one(T), size(X))
          if f isa NuclearNorm
            @test PB._leading_svd!(f, X, T, one(T), 0) == 1
            v = PB._partial_nuclear!(Y, f, X, one(T))
          else
            @test PB._leading_eigen!(f, X, T, one(T), 0) == 1
            v = PB._partial_psd!(Y, f, X, one(T), 0, λ)
          end
          # a partial decomposition is approximate: the value overflows for real
          # at λ = M, and is close to 3λ otherwise
          check(v) = λ == M ? (@test v === T(Inf)) : (@test isapprox(v, 3λ; rtol = 100eps(T)))
          check(v)
          @test (f isa NuclearNorm ? f.ws : f.eig) === nothing
          Z = copy(X)
          check(value_and_prox!(Z, f, Z, inv(λ)))
          @test (f isa NuclearNorm ? f.ws : f.eig) === nothing
          @test isapprox(Z, Y; rtol = 100eps(T))
        end
      end

      @testset "legacy reductions and workspace reuse" begin
        rng = Xoshiro(195)
        for n in (0, 1, 17, 2000), λ in (zero(T), T(1) / 4, one(T), T(3))
          x = randn(rng, T, n)
          @test samebits(NormL1(λ)(x), λ * sum(abs, x))
          @test samebits(NormL2(λ)(x), λ * PB._norm2(x))
          y, expected = similar(x), similar(x)
          legacy_prox!(expected, NormL2(λ), x, T(1) / 3)
          @test prox!(y, NormL2(λ), x, T(1) / 3) === y
          @test samebits(y, expected)
        end
        δ = eps(T) / 4
        X = sparse(Diagonal(vcat(one(T), fill(δ, 2047))))
        @test samebits(TracePSD(one(T))(X), tr(X))
        for wrap in (identity, sparse, SparseIterate, SymmetricSparseIterate, Symmetric)
          f = TracePSD(T(1) / 4)
          for n in (33, 2, 17), λ in (one(T), -zero(T))
            X = wrap(Matrix(Diagonal(vcat(one(T), fill(δ, n - 1)))))
            f.λ = λ
            @test samebits(f(X), λ * tr(X))
          end
        end
        for wrap in (identity, SymmetricSparseIterate, SparseIterate)
          A, B = wrap(T[2 1; 1 2]), wrap(D)
          for f in (NormL1(T(1) / 4), NormL2(T(1) / 4), NuclearNorm(T(1) / 4), TracePSD(T(1) / 4), LogDetLoss(E, T(1) / 4))
            before = f(A)
            f(B)
            @test samebits(f(A), before)
            for X in (A, B)
              @test allocs(f, X) == 0
              Y = zeros(T, 2, 2)
              @test allocs(value_and_prox!, Y, f, X, γ) == 0
              if f isa NormL2
                @test allocs(prox!, Y, f, X, γ) == 0
              end
              if f isa LogDetLoss
                @test allocs(value_and_gradient!, Y, f, X) == 0
              end
            end
          end
        end
        f = NuclearNorm(T(1) / 4)
        @test allocs(f, P) == 0
      end
    end
  end
end


@testset "scaled values: cancellation, mixed precision, non-finite weights" begin
  M = floatmax(Float64)
  D = Matrix(Diagonal([1.0, -1.0]))
  # Huge products that cancel keep the moderate products and the log-determinant.
  for (S, X) in ((M * D, Matrix(M * I(2))), (2.0^100 * D, Matrix(2.0^1000 * I(2))),
                 ([2.0^100 1; 1 -2.0^100], [2.0^1000 500; 500 2.0^1000]),
                 (2.0^40 * D, Matrix(2.0^1000 * I(2))), (2.0^60 * D, Matrix(2.0^1000 * I(2))))
    f = LogDetLoss(S, 0.25)
    reference = setprecision(4096) do
      big(0.25) * (dot(big.(S), big.(X)) - logdet(Symmetric(big.(X))))
    end
    check_rounded(f(X), reference)
    check_rounded(f(SymmetricSparseIterate(X)), reference)
    G = similar(X)
    check_rounded(value_and_gradient!(G, f, X), reference)
  end
  S, X = 2f0^40 * Matrix{Float32}(Diagonal([1, -1])), 2f0^120 * Matrix{Float32}(I, 2, 2)
  check_rounded(LogDetLoss(S, 0.25f0)(X), big(0.25f0) * -logdet(big.(X)))
  f = LogDetLoss(M * D, 1e-3)
  Y = zeros(2, 2)
  v = value_and_prox!(Y, f, Matrix(2.0^60 * I(2)), 1e-307)
  check_rounded(v, setprecision(() -> big(1e-3) * (dot(big.(f.S), big.(Y)) - logdet(Symmetric(big.(Y)))), 4096))
  # Inputs of another type than the data keep a finite value, or the base value.
  M32 = floatmax(Float32)
  f32 = LogDetLoss(M32 * Matrix{Float32}(I, 2, 2), 0.25f0)
  for X in (Matrix(I(2)), Matrix(I(2)) .* 1, Matrix(1.0I, 2, 2))
    v = f32(X)
    @test v isa typeof(LogDetLoss(Matrix{Float32}(I, 2, 2), 0.25f0)(X))   # the type of the ordinary value
    check_rounded(v, big(M32) / 2)
  end
  @test LogDetLoss(Matrix{Float32}(I, 2, 2), 0.25f0)(1e300 * Matrix(1.0I, 2, 2)) === -Inf
  # Non-finite weights keep their base values; complex entries give a real value.
  @test isnan(NormL1(Inf)(zeros(2)))
  @test isnan(NormL2(Inf)(zeros(2)))
  @test isnan(NuclearNorm(Inf)(zeros(2, 2)))
  z = ComplexF64[M, M]
  @test NormL2(0.25)(z) === Inf
end

end
