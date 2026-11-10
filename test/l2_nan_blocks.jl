module L2NaNBlocksTest

using ..TestUtils: take_block_step!

# E2 for the Euclidean block norms (#559, LIM-047): a NaN entry makes every output of its block
# NaN, since each output depends on every entry of the block, and the value at the stored output
# NaN. E4 comes first: a zero level or weight copies, and an infinite level, weight or step
# fixes the block at zero. Independent groups and indices in no group are not poisoned.
# A comparison with NaN selects no branch, so these tests never compare NaN with ==.

using Test, ProximalBase, LinearAlgebra, SparseArrays, JLArrays

# the same NaN pattern, and the finite entries equal to rounding
function agree(a, b; rtol = 64 * eps(float(real(eltype(a)))))
  size(a) == size(b) || return false
  all(eachindex(a)) do i
    isnan(a[i]) || isnan(b[i]) ? isnan(a[i]) && isnan(b[i]) : isapprox(a[i], b[i]; rtol, atol = rtol)
  end
end
agree(a::Real, b::Real; kwargs...) = isnan(a) || isnan(b) ? isnan(a) && isnan(b) : isapprox(a, b; kwargs...)

# run both oracles, with a caller-owned output filled with a sentinel, and in place
function check(f, x, γ, want, wantvalue; sentinel = -7)
  T = float(eltype(x))
  y = fill(T(sentinel), size(x))
  @test prox!(y, f, copy(x), γ) === y
  @test agree(y, want)
  y = fill(T(sentinel), size(x))
  v = value_and_prox!(y, f, copy(x), γ)
  @test agree(y, want)
  @test agree(v, wantvalue)
  z = copy(x)
  prox!(z, f, z, γ)
  @test agree(z, want)
  z = copy(x)
  @test agree(value_and_prox!(z, f, z, γ), wantvalue)
  @test agree(z, want)
  @test agree(prox(f, x, γ), want)
end

@testset "NormL2" begin
  for T in (Float32, Float64)
    f = NormL2(T(1))
    nan = T(NaN)
    @testset "NaN gives a NaN block, $T" begin
      check(f, T[NaN, 4], T(1), T[NaN, NaN], nan)
      check(f, T[3, NaN, 4], T(1), T[NaN, NaN, NaN], nan)
      check(f, T[NaN Inf; 1 2], T(1), fill(nan, 2, 2), nan)
      check(f, T[NaN, Inf], T(1), T[NaN, NaN], nan)    # NaN takes precedence over infinity
      check(f, T[Inf, NaN], T(1), T[NaN, NaN], nan)
      check(f, T[-Inf, NaN, Inf], T(1), T[NaN, NaN, NaN], nan)
      @test isnan(f(T[NaN, 4]))
      @test isnan(f(T[Inf, NaN]))
    end
    @testset "endpoint steps and E4, $T" begin     # the steps are accepted on master (#311)
      check(f, T[NaN, 4], T(0), T[NaN, 4], nan)                       # the identity: NaN only in its entry
      check(f, T[3, 4], T(0), T[3, 4], T(5))
      check(f, T[NaN, 4], T(Inf), T[0, 0], zero(T))                   # a constant map keeps zeros
      check(f, T[3, 4], T(Inf), T[0, 0], zero(T))
      check(NormL2(T(0)), T[NaN, 4], T(1), T[NaN, 4], zero(T))        # zero level: the identity
      check(NormL2(T(0)), T[NaN, 4], T(Inf), T[NaN, 4], zero(T))
      check(NormL2(T(Inf)), T[NaN, 4], T(1), T[0, 0], zero(T))        # infinite level: fixed at zero
      # strict E2: a positive step whose level product underflows still gives the NaN block
      tiny = sqrt(floatmin(T)) * eps(T)                                # tiny^2 underflows to 0
      @test iszero(tiny * tiny)
      check(NormL2(tiny), T[NaN, 4], tiny, T[NaN, NaN], nan)
    end
    @testset "finite inputs are unchanged, $T" begin
      check(f, T[3, 4], T(1), T[2.4, 3.2], T(4))
      check(f, T[0.3, 0.4], T(1), T[0, 0], zero(T))
      check(NormL2(T(2)), T[3, 4], T(1), T[1.8, 2.4], T(6))
      check(f, T[Inf, 1], T(1), T[Inf, 1], T(Inf))
    end
  end
end

@testset "GroupNormL2" begin
  for T in (Float32, Float64)
    g = GroupNormL2(T(1), [1:2, 3:4], T[1, 1])
    nan = T(NaN)
    x = T[NaN, 4, 3, 4]
    shrunk = T[3, 4] .* (1 - 1 / T(5))
    @testset "a NaN group is NaN and the other group is not, $T" begin
      check(g, x, T(1), [nan, nan, shrunk...], nan)
      check(g, T[3, 4, 3, NaN], T(1), [shrunk..., nan, nan], nan)
      check(g, T[NaN, 4, 3, NaN], T(1), fill(nan, 4), nan)
      check(g, T[NaN, Inf, 3, 4], T(1), [nan, nan, shrunk...], nan)
      check(g, T[3, 4, 3, 4], T(1), [shrunk..., shrunk...], 2 * T(4))
      @test isnan(g(x))
      @test isnan(g(T[NaN, Inf, 3, 4]))
    end
    @testset "indices in no group, singleton groups, $T" begin
      check(GroupNormL2(T(1), [1:2], T[1]), T[NaN, 4, 5], T(1), [nan, nan, T(5)], nan)
      check(GroupNormL2(T(1), [1:2], T[1]), T[3, 4, NaN], T(1), [shrunk..., nan], T(4))     # the free NaN is no part of the value
      check(GroupNormL2(T(1), [3:4], T[1]), T[NaN, 4, 3, 4], T(1), [nan, T(4), shrunk...], T(4))
      check(GroupNormL2(T(1), [1:1, 2:2], T[1, 1]), T[NaN, 4], T(1), T[NaN, 3], nan)
    end
    @testset "steps per group, endpoints, $T" begin
      check(g, x, T[1, 1], [nan, nan, shrunk...], nan)
      # a zero step in a vector is the identity for its group, NaN only in its own entry (#559, validation 3.4)
      check(g, T[3, 4, 3, 4], T[0, 1], [T(3), T(4), shrunk...], T(5) + T(4))
      check(g, x, T[0, 1], [nan, T(4), shrunk...], nan)
      check(g, x, T[1, 0], [nan, nan, T(3), T(4)], nan)
      check(g, T[3, 4, NaN, 4], T[1, 0], [shrunk..., nan, T(4)], nan)
      check(g, x, T[0, 0], x, nan)
      # a positive step whose γλw only underflowed is not the identity: the NaN block (strict E2)
      tiny = sqrt(floatmin(T)) * eps(T)
      gt = GroupNormL2(tiny, [1:2, 3:4], T[1, 1])
      @test iszero(tiny * tiny)
      check(gt, x, T[tiny, tiny], [nan, nan, T(3), T(4)], nan)
      check(gt, x, T[tiny, 0], [nan, nan, T(3), T(4)], nan)
      check(gt, x, T[0, tiny], [nan, T(4), T(3), T(4)], nan)
      check(gt, x, tiny, [nan, nan, T(3), T(4)], nan)
      check(g, x, T(0), x, nan)
      check(g, x, T(Inf), zeros(T, 4), zero(T))
      check(g, T[3, 4, 3, 4], T(Inf), zeros(T, 4), zero(T))
    end
    @testset "E4 comes before NaN, $T" begin
      w0 = GroupNormL2(T(1), [1:2, 3:4], T[0, 1])               # a zero weight copies, free of the value
      check(w0, x, T(1), [nan, T(4), shrunk...], T(4))
      @test agree(w0(T[NaN, 4, 3, 4]), T(5))                    # the value does not depend on the free group
      winf = GroupNormL2(T(1), [1:2, 3:4], T[Inf, 1])           # an infinite weight fixes the group at zero
      check(winf, x, T(1), [zero(T), zero(T), shrunk...], T(4))
      check(winf, T[3, 4, 3, 4], T(1), [zero(T), zero(T), shrunk...], T(4))
      check(GroupNormL2(T(0), [1:2, 3:4], T[1, 1]), x, T(1), x, zero(T))
      check(GroupNormL2(T(0), [1:2, 3:4], T[1, 1]), x, T(Inf), x, zero(T))
      check(GroupNormL2(T(0), [1:2, 3:4], T[Inf, 1]), x, T(1), [zero(T), zero(T), T(3), T(4)], zero(T))
      check(GroupNormL2(T(1), [1:2, 3:4], T[0, 1]), T[3, 4, 3, 4], T(Inf), T[3, 4, 0, 0], zero(T))
    end
  end
end

@testset "SparseGroupNormL2" begin
  for T in (Float32, Float64)
    nan = T(NaN)
    s = SparseGroupNormL2(T(0.1), T(1), [1:2, 3:4], T[1, 1])
    x = T[NaN, 4, 3, 4]
    soft = T[2.9, 3.9]
    other = soft .* (1 - 1 / sqrt(sum(abs2, soft)))
    @testset "NaN couples its group, $T" begin
      check(s, x, T(1), [nan, nan, other...], nan)
      check(s, T[3, 4, NaN, 4], T(1), [other..., nan, nan], nan)
      check(s, T[3, 4, 3, 4], T(1), [other..., other...], 2 * (norm(other) + T(0.1) * sum(other)))
      check(SparseGroupNormL2(T(0), T(1), [1:2, 3:4], T[1, 1]), T[NaN, 4, 3, 4], T(1),
            [nan, nan, T[3, 4] .* (1 - 1 / T(5))...], nan)
      @test isnan(s(x))
    end
    @testset "E4, steps and decoupled regimes, $T" begin
      # no group term: soft thresholding of each entry on its own
      check(SparseGroupNormL2(T(0.1), T(0), [1:2, 3:4], T[1, 1]), x, T(1), T[NaN, 3.9, 2.9, 3.9], nan)
      check(SparseGroupNormL2(T(0.1), T(1), [1:2, 3:4], T[0, 1]), x, T(1), [nan, T(3.9), other...], nan)
      check(SparseGroupNormL2(T(0.1), T(1), [1:2, 3:4], T[Inf, 1]), x, T(1), [zero(T), zero(T), other...],
            norm(other) + T(0.1) * sum(other))     # the fixed group is no part of the NaN
      check(s, x, T(0), x, nan)
      check(s, x, T(Inf), zeros(T, 4), zero(T))
      # a zero step in a vector is the identity for its group (each entry as it is), NaN only in its own
      check(s, x, T[0, 1], [nan, T(4), other...], nan)
      check(s, x, T[1, 0], [nan, nan, T(3), T(4)], nan)
      check(s, T[3, 4, NaN, 4], T[1, 0], [other..., nan, T(4)], nan)
      check(s, x, T[0, 0], x, nan)
      # a positive step whose group threshold only underflowed is the NaN block
      tiny = sqrt(floatmin(T)) * eps(T)
      st = SparseGroupNormL2(T(0.1), tiny, [1:2, 3:4], T[1, 1])
      check(st, x, tiny, [nan, nan, T(3), T(4)], nan)
      check(st, x, T[tiny, 0], [nan, nan, T(3), T(4)], nan)
      check(st, x, T[0, tiny], [nan, T(4), T(3), T(4)], nan)
      # a zero level is the identity at every step, also at γ = Inf (E4 before the step)
      z = SparseGroupNormL2(T(0), T(0), [1:2, 3:4], T[1, 1])
      for γ in (T(0), T(1), T(Inf))
        check(z, x, γ, x, zero(T))               # the zero function: its value is free of the NaN
        check(z, T[3, 4, 3, 4], γ, T[3, 4, 3, 4], zero(T))
      end
      zinf = SparseGroupNormL2(T(0), T(0), [1:2, 3:4], T[Inf, 1])
      check(zinf, x, T(Inf), T[0, 0, 3, 4], zero(T))
    end
  end
end

@testset "GroupNormLinf shares the drivers and is unchanged" begin
  for T in (Float32, Float64)
    f = GroupNormLinf(T(1), [1:2, 3:4], T[1, 1])
    check(f, T[NaN, 4, 3, 4], T(1), T[NaN, NaN, 3, 3], T(NaN))
    check(f, T[3, 4, 3, 4], T(1), T[3, 3, 3, 3], T(6))
    check(f, T[3, 4, 3, 4], T(0), T[3, 4, 3, 4], T(8))
  end
end

@testset "a finite parameter whose product overflowed or underflowed is no endpoint" begin
  for T in (Float32, Float64)
    nan = T(NaN)
    x = T[NaN, 4, 0, 0, 0, 0, 0, 0, 0]
    big = floatmax(T)
    # γλ and γλw overflow to Inf, but the step, level and weights are finite: the NaN block, not zeros
    for f in (NormL2(big), GroupNormL2(big, [1:9], T[1]), SparseGroupNormL2(T(0.1), big, [1:9], T[1]))
      check(f, x, T(2), fill(nan, 9), nan)
      f isa NormL2 || check(f, x, T[2], fill(nan, 9), nan)           # a step per group
      # the actual endpoints and fixes stay: γ = Inf is the constant zero map, γ = 0 the identity
      check(f, x, T(Inf), zeros(T, 9), zero(T))
      check(f, x, T(0), x, nan)
    end
    # an infinite level or weight is the hard fix
    check(NormL2(T(Inf)), x, T(2), zeros(T, 9), zero(T))
    check(GroupNormL2(big, [1:9], T[Inf]), x, T(2), zeros(T, 9), zero(T))
    check(SparseGroupNormL2(T(0.1), big, [1:9], T[Inf]), x, T(2), zeros(T, 9), zero(T))
    # AtomIterate
    for f in (GroupNormL2(big, [1:9], T[1]), SparseGroupNormL2(T(0.1), big, [1:9], T[1]))
      a = AtomIterate(x, f)
      y = copy(a)
      @test isnan(value_and_prox!(y, f, a, T(2)))
      @test all(isnan, y.storage)
      a = AtomIterate(x, f)
      take_block_step!(a, f, 1, T(2))
      @test all(isnan, a.storage)
      a = AtomIterate(x, f)
      take_block_step!(a, f, 1, T(Inf))                               # the constant zero map
      @test all(iszero, a.storage)
    end
    # device arrays
    for f in (NormL2(big), GroupNormL2(big, [1:9], T[1]), SparseGroupNormL2(T(0.1), big, [1:9], T[1]))
      for op in (prox!, value_and_prox!)
        y = JLArray(fill(T(-7), 9))
        r = op(y, f, JLArray(x), T(2))
        @test all(isnan, Array(y))
        op === value_and_prox! && @test isnan(r)
        z = JLArray(copy(x))
        op(z, f, z, T(2))
        @test all(isnan, Array(z))
      end
    end
  end
  # the product underflows on the mixed-precision path: the level and the step are positive, so the NaN block
  x = Float32[NaN, 4]
  for f in (GroupNormL2(1e-100, [1:2], [1.0]), SparseGroupNormL2(0.0, 1e-100, [1:2], [1.0]))
    check(f, x, 1e-100, Float32[NaN, NaN], NaN32)
    for op in (prox!, value_and_prox!)
      y = JLArray(fill(-7f0, 2))
      r = op(y, f, JLArray(x), 1e-100)
      @test all(isnan, Array(y))
      op === value_and_prox! && @test isnan(r)
      z = JLArray(copy(x))
      op(z, f, z, 1e-100)
      @test all(isnan, Array(z))
    end
  end
end

@testset "FiberSum(NormL2)" begin
  for T in (Float32, Float64)
    f = FiberSum(NormL2(T(1)))
    X = T[NaN 1; 2 3]
    Y = prox(f, X, T(1))
    @test all(isnan, Y[1, :])
    @test agree(Y[2, :], T[2, 3] .* (1 - 1 / sqrt(T(13))))
    z = copy(X)
    prox!(z, f, z, T(1))
    @test agree(z, Y)
    @test isnan(value_and_prox!(similar(X), f, X, T(1)))
  end
end

@testset "iterates" begin
  for T in (Float32, Float64)
    nan = T(NaN)
    @testset "SparseIterate, $T" begin
      y = SparseIterate(zeros(T, 3))
      @test isnan(value_and_prox!(y, NormL2(T(1)), T[NaN, 1, 2], T(1)))
      @test all(isnan, Vector(y))
      y = SparseIterate(zeros(T, 3))
      prox!(y, NormL2(T(1)), SparseIterate(T[NaN, 1, 2]), T(1))
      @test all(isnan, Vector(y))
      y = SparseIterate(T[1, 2, 3])
      prox!(y, NormL2(T(1)), SparseIterate(T[1, 2, NaN]), T(1))
      @test all(isnan, Vector(y))
      @test isnan(NormL2(T(1))(SparseIterate(T[NaN, 1, 2])))
    end
    @testset "SymmetricSparseIterate, $T" begin
      A = T[1 NaN 0.5; NaN 2 1; 0.5 1 3]
      # a group of linear indices that is symmetric: the 2 × 2 leading block
      groups = [[1], [2, 4], [3, 7], [5], [6, 8], [9]]      # singletons and mirrored pairs
      block = GroupNormL2(T(1), [[1, 2, 4, 5]], T[1])
      pairs = GroupNormL2(T(1), groups)
      sparse_pairs = SparseGroupNormL2(T(0.1), T(1), groups)
      for (f, want) in ((NormL2(T(1)), fill(nan, 3, 3)),
                        (block, T[nan nan 0.5; nan nan 1; 0.5 1 3]),
                        (pairs, T[0 nan 0; nan 1 0; 0 0 2]),
                        (sparse_pairs, T[0 nan 0; nan 0.9 0; 0 0 1.9]))
        for op in (prox!, value_and_prox!)
          y = SymmetricSparseIterate(T, 3)
          fill!(coordinates(y), -1)
          v = op(y, f, A, T(1))
          @test agree(Matrix(y), want)
          op === value_and_prox! && @test isnan(v)
          x = SymmetricSparseIterate(A)
          op(x, f, x, T(1))
          @test agree(Matrix(x), want)
          x = SymmetricSparseIterate(A)
          y = SymmetricSparseIterate(T, 3)
          op(y, f, x, T(1))
          @test agree(Matrix(y), want)
          @test isequal(Matrix(x), A)
        end
      end
      # a group step per group, on a dense output: the identity at a zero step (NaN in its own entry), the NaN
      # block when the product only underflowed
      A2 = T[NaN 1 0.5; 1 2 1; 0.5 1 3]
      tiny = sqrt(floatmin(T)) * eps(T)
      for (f, γ, want) in ((GroupNormL2(T(1), [[1, 2, 4, 5]], T[1]), T[0], T[nan 1 0.5; 1 2 1; 0.5 1 3]),
                           (GroupNormL2(T(1), [[1, 2, 4, 5]], T[1]), T[1], T[nan nan 0.5; nan nan 1; 0.5 1 3]),
                           (GroupNormL2(tiny, [[1, 2, 4, 5]], T[1]), T[tiny], T[nan nan 0.5; nan nan 1; 0.5 1 3]),
                           (SparseGroupNormL2(T(0.1), T(1), [[1, 2, 4, 5]], T[1]), T[0], T[nan 1 0.5; 1 2 1; 0.5 1 3]),
                           (SparseGroupNormL2(T(0.1), tiny, [[1, 2, 4, 5]], T[1]), T[tiny], T[nan nan 0.5; nan nan 1; 0.5 1 3]))
        for op in (prox!, value_and_prox!)
          # an owner takes no step per group: refused before anything is written
          y = SymmetricSparseIterate(T, 3)
          fill!(coordinates(y), -1)
          before = (Matrix(y), nactive(y))
          @test_throws ArgumentError op(y, f, A2, γ)
          @test isequal((Matrix(y), nactive(y)), before)
          x = SymmetricSparseIterate(A2)
          @test_throws ArgumentError op(x, f, x, γ)
          @test isequal(Matrix(x), A2)
          yd = fill(T(-7), 3, 3)
          op(yd, f, copy(A2), γ)
          @test agree(yd, want)
        end
      end
    end
    @testset "AtomIterate, $T" begin
      g = GroupNormL2(T(0.5), [1:2, 3:4])
      x = AtomIterate(T[NaN, 1, 2, 3], g)
      y = copy(x)
      @test isnan(value_and_prox!(y, g, x, T(1)))
      @test isnan(y.storage[1]) && isnan(y.storage[2])
      @test all(isfinite, y.storage[3:4])
      @test y.nactive == 2
      x = AtomIterate(T[NaN, 1, 2, 3], g)
      take_block_step!(x, g, 1, T(1))
      @test isnan(x.storage[1]) && isnan(x.storage[2])
      @test x.storage[3:4] == T[2, 3]
      @test x.nactive == 2                                # the group is nonzero, so it is active
      x = AtomIterate(T[NaN, 1, 2, 3], g)
      take_block_step!(x, g, 1, T(0))
      @test isnan(x.storage[1]) && x.storage[2] == 1
      for γ in (T[0, 1], T[1, 0])                         # a zero step in a vector: the identity for its group
        x = AtomIterate(T[NaN, 1, 2, 3], g)
        y = copy(x)
        v = value_and_prox!(y, g, x, γ)
        @test isnan(v)
        if γ[1] == 0
          @test isnan(y.storage[1]) && y.storage[2] == 1 && all(isfinite, y.storage[3:4])
        else
          @test isnan(y.storage[1]) && isnan(y.storage[2]) && y.storage[3:4] == T[2, 3]
        end
        z = AtomIterate(T[NaN, 1, 2, 3], g)           # in place
        value_and_prox!(z, g, z, γ)
        @test isequal(z.storage, y.storage)
      end
      tiny = sqrt(floatmin(T)) * eps(T)
      gt = GroupNormL2(tiny, [1:2, 3:4])
      x = AtomIterate(T[NaN, 1, 2, 3], gt)
      y = copy(x)
      @test isnan(value_and_prox!(y, gt, x, tiny))        # underflowed, not zero: the NaN block
      @test isnan(y.storage[1]) && isnan(y.storage[2])
      x = AtomIterate(T[NaN, 1, 2, 3], gt)
      take_block_step!(x, gt, 1, tiny)
      @test isnan(x.storage[1]) && isnan(x.storage[2])
      s = SparseGroupNormL2(T(0.1), T(1), [1:2, 3:4])
      for γ in (T[0, 1], T[1, 0])
        x = AtomIterate(T[NaN, 1, 2, 3], s)
        y = copy(x)
        @test isnan(value_and_prox!(y, s, x, γ))
        if γ[1] == 0
          @test isnan(y.storage[1]) && y.storage[2] == 1 && all(isfinite, y.storage[3:4])
        else
          @test isnan(y.storage[1]) && isnan(y.storage[2]) && y.storage[3:4] == T[2, 3]
        end
      end
      x = AtomIterate(T[NaN, 1, 2, 3], s)
      y = copy(x)
      @test isnan(value_and_prox!(y, s, x, T(1)))
      @test isnan(y.storage[1]) && isnan(y.storage[2]) && all(isfinite, y.storage[3:4])
      x = AtomIterate(T[NaN, 1, 2, 3], s)
      take_block_step!(x, s, 1, T(1))
      @test isnan(x.storage[1]) && isnan(x.storage[2])
      @test x.nactive == 2
    end
  end
end

@testset "device arrays (JLArrays)" begin
  host(a) = Array(a)
  for T in (Float32, Float64)
    nan = T(NaN)
    dev(a) = JLArray(a)
    @testset "NormL2, $T" begin
      f = NormL2(T(1))
      for op in (prox!, value_and_prox!)
        y = dev(fill(T(-7), 2))
        r = op(y, f, dev(T[NaN, 4]), T(1))
        @test all(isnan, host(y))
        op === value_and_prox! && @test isnan(r)
      end
      @test isnan(f(dev(T[NaN, 4])))
      y = dev(fill(T(-7), 2))
      prox!(y, f, dev(T[NaN, Inf]), T(1))
      @test all(isnan, host(y))
      y = dev(fill(T(-7), 2))
      prox!(y, f, dev(T[NaN, 4]), T(Inf))
      @test host(y) == zeros(T, 2)
      y = dev(fill(T(-7), 2))
      prox!(y, f, dev(T[NaN, 4]), T(0))
      @test agree(host(y), T[NaN, 4])
    end
    @testset "groups, $T" begin
      g = GroupNormL2(T(1), [1:2, 3:4], T[1, 1])
      s = SparseGroupNormL2(T(0.1), T(1), [1:2, 3:4], T[1, 1])
      for f in (g, s), op in (prox!, value_and_prox!)
        y = dev(fill(T(-7), 4))
        r = op(y, f, dev(T[NaN, 4, 3, 4]), T(1))
        h = host(y)
        ref = prox(f, T[NaN, 4, 3, 4], T(1))
        @test agree(h, ref)
        @test isnan(h[1]) && isnan(h[2]) && isfinite(h[3]) && isfinite(h[4])
        op === value_and_prox! && @test isnan(r)
        y = dev(fill(T(-7), 4))
        op(y, f, dev(T[NaN, 4, 3, 4]), T(0))
        @test agree(host(y), T[NaN, 4, 3, 4])
        y = dev(fill(T(-7), 4))
        op(y, f, dev(T[NaN, 4, 3, 4]), T(Inf))
        @test host(y) == zeros(T, 4)
      end
      @test isnan(g(dev(T[NaN, 4, 3, 4])))
      # E4 first: a zero weight copies, an infinite weight fixes the group at zero
      w0 = GroupNormL2(T(1), [1:2, 3:4], T[0, 1])
      y = dev(fill(T(-7), 4))
      value_and_prox!(y, w0, dev(T[NaN, 4, 3, 4]), T(1))
      @test agree(host(y)[1:2], T[NaN, 4]) && all(isfinite, host(y)[3:4])
      winf = GroupNormL2(T(1), [1:2, 3:4], T[Inf, 1])
      y = dev(fill(T(-7), 4))
      value_and_prox!(y, winf, dev(T[NaN, 4, 3, 4]), T(1))
      @test host(y)[1:2] == zeros(T, 2) && all(isfinite, host(y)[3:4])
    end
    @testset "zero levels with mixed finite and infinite weights, device and host agree, $T" begin
      G = [1:2, 3:4]
      # a zero group level with a finite weight is the identity (NaN in its own entry), while the
      # infinite weight keeps its hard fix; the value of the zero level is 0
      for (f, x) in ((GroupNormL2(T(0), G, T[1, Inf]), T[NaN, 4, 0, 0]),
                     (GroupNormL2(T(0), G, T[Inf, 1]), T[0, 0, NaN, 4]),
                     (SparseGroupNormL2(T(0.1), T(0), G, T[1, Inf]), T[NaN, 4, 0, 0]),
                     (SparseGroupNormL2(T(0.1), T(0), G, T[Inf, 1]), T[0, 0, NaN, 4]),
                     (GroupNormL2(T(0), G, T[1, Inf]), T[3, 4, 0, 0]),
                     (SparseGroupNormL2(T(0.1), T(0), G, T[1, Inf]), T[3, 4, 0, 0]))
        for γ in (any(isnan, x) ? (T(1), T(0)) : (T(1), T(0), T(Inf)))
          ref = zeros(T, 4)
          refv = value_and_prox!(ref, f, copy(x), γ)
          for op in (prox!, value_and_prox!)
            y = dev(fill(T(-7), 4))
            r = op(y, f, dev(x), γ)
            @test agree(host(y), ref)
            op === value_and_prox! && @test agree(r, refv)
            z = dev(copy(x))
            op(z, f, z, γ)
            @test agree(host(z), ref)
          end
        end
      end
      @test agree(prox(GroupNormL2(T(0), G, T[1, Inf]), T[NaN, 4, 0, 0], T(1)), T[NaN, 4, 0, 0])
      @test agree(prox(SparseGroupNormL2(T(0.1), T(0), G, T[1, Inf]), T[NaN, 4, 0, 0], T(1)), T[NaN, 3.9, 0, 0])
      # a zero ℓ1 level gives a zero threshold at every step: γ·0 is never formed (D1)
      f = SparseGroupNormL2(T(0), T(1), G, T[0, 1])
      x = T[2, 4, 3, 4]
      for γ in (T(1), T(Inf))
        ref = prox(f, x, γ)
        γ == T(Inf) && @test agree(ref, T[2, 4, 0, 0])
        for op in (prox!, value_and_prox!)
          y = dev(fill(T(-7), 4))
          op(y, f, dev(x), γ)
          @test agree(host(y), ref)
          z = dev(copy(x))
          op(z, f, z, γ)
          @test agree(host(z), ref)
        end
      end
      @test agree(host(prox(SparseGroupNormL2(T(0), T(0), G, T[1, 1]), dev(x), T(Inf))), x)
    end
    @testset "infinite group and ℓ1 levels, device and host agree, $T" begin
      G = [1:2, 3:4]
      # an infinite weight fixes its group at every step; an infinite group level fixes a group at a positive
      # step only: at the step 0 a finite weight keeps the identity (NaN in its own entry)
      for f in (GroupNormL2(T(Inf), G, T[1, Inf]), SparseGroupNormL2(T(0.1), T(Inf), G, T[1, Inf]))
        x = T[NaN, 4, NaN, 2]
        for γ in (T(0), T(1))
          ref = zeros(T, 4)
          refv = value_and_prox!(ref, f, copy(x), γ)
          if γ == 0
            @test agree(ref, T[NaN, 4, 0, 0]) && isnan(refv)
          else
            @test agree(ref, zeros(T, 4)) && iszero(refv)
          end
          for op in (prox!, value_and_prox!)
            y = dev(fill(T(-7), 4))
            r = op(y, f, dev(x), γ)
            @test agree(host(y), ref)
            op === value_and_prox! && @test agree(r, refv)
            z = dev(copy(x))
            op(z, f, z, γ)
            @test agree(host(z), ref)
          end
        end
      end
      # an infinite ℓ1 level maps the penalized group to zero and copies the ungrouped entry
      f = SparseGroupNormL2(T(Inf), T(1), [1:2], T[1])
      for x in (T[NaN, 4, 9], T[3, 4, 9])
        ref = prox(f, x, T(1))
        @test agree(ref, T[0, 0, 9])
        for op in (prox!, value_and_prox!)
          y = dev(fill(T(-7), 3))
          op(y, f, dev(x), T(1))
          @test agree(host(y), ref)
          z = dev(copy(x))
          op(z, f, z, T(1))
          @test agree(host(z), ref)
        end
      end
    end
  end
end

@testset "constrained group norms at NaN reach the ball projection" begin
  for T in (Float32, Float64)
    for c in (Constrained(GroupNormL2(T(1), [1:2]), IndBallL2(T(1))),
              Constrained(NormL2(T(1)), IndBallL2(T(1))),
              Constrained(GroupNormL2(T(1), [1:2]), IndSphereL2(T(1))))
      y = prox(c, T[NaN, 0], T(1))
      @test all(isnan, y)
      @test all(isnan, prox(c, T[0, NaN], T(1)))
    end
  end
end

end
