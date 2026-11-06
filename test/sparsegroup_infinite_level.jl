module SparseGroupInfiniteLevelTest

# SparseGroupNormL2 with an infinite ℓ1 level (#585). The level fixes every grouped entry at zero, as
# the proximal operator does, so the ℓ1 term is the indicator of that set: the value is 0 where the
# grouped entries are zero (whatever the group term's level and weights), Inf where one is nonzero
# and NaN where one is NaN (E2); indices in no group stay free. value_and_prox! returns the value at
# the stored output (O2), on the CPU and on device arrays alike. A comparison with NaN selects no
# branch, so these tests never compare NaN with ==.

using Test, ProximalBase, SparseArrays, JLArrays, Adapt

# The grid of the proximal map over group levels, weights, steps, iterates, device routes and
# wrappers is in sparsegroup_infinite_prox.jl (#620).

# the same type, the same class (NaN, ±Inf, finite) and the same value
agree(a::Real, b::Real) = typeof(a) == typeof(b) && (isnan(a) || isnan(b) ? isnan(a) && isnan(b) : a == b)
agree(a::AbstractArray, b::AbstractArray) =
  size(a) == size(b) && all(i -> isnan(a[i]) || isnan(b[i]) ? isnan(a[i]) && isnan(b[i]) : a[i] == b[i], eachindex(a))

# The value of an infinite ℓ1 level from its definition, independent of the package kernels: the
# grouped entries decide; the group term is 0 at a zero group, also with an infinite weight (E4).
function ref(f::SparseGroupNormL2, x)
  T = float(eltype(x))
  P = reduce(vcat, collect.(f.groups); init = Int[])
  a = [x[i] for i in P]
  any(isnan, a) ? T(NaN) : any(!iszero, a) ? T(Inf) : zero(T)
end

# value_and_prox! into a separate output (filled with a sentinel) and in place: the value is the one
# at the stored output (O2), against both f and the reference; returns the stored outputs
function fused(f, x, γ)
  T = float(eltype(x))
  y = fill(T(-7), size(x))
  v = value_and_prox!(y, f, copy(x), γ)
  @test agree(v, f(y))
  @test agree(v, ref(f, y))
  z = copy(x)
  u = value_and_prox!(z, f, z, γ)
  @test agree(u, f(z))
  @test agree(u, ref(f, z))
  @test agree(z, y)
  @test agree(u, v)
  v, y
end

@testset "standalone value, $T" for T in (Float32, Float64)
  nan = T(NaN)
  f = SparseGroupNormL2(T(Inf), T(1), [1:2], T[1])
  # the reported case: the penalized entries are zero, coordinate 3 is in no group
  @test f(T[0, 0, 9]) === zero(T)
  @test f(T[-0.0, -0.0, 9]) === zero(T)
  # the value does not depend on an index in no group (E2)
  for a in (nan, T(Inf), -T(Inf), T(-3))
    @test f(T[0, 0, a]) === zero(T)
  end
  # a nonzero grouped entry is outside the domain
  for x in (T[1, 0, 9], T[0, -1, 9], T[Inf, 0, 9], T[0, -Inf, nan], T[nextfloat(zero(T)), 0, 9])
    @test f(x) === T(Inf)
  end
  # a NaN grouped entry gives NaN, also beside an infinite one (NaN over Inf)
  for x in (T[nan, 0, 9], T[0, nan, 9], T[nan, nan, 9], T[nan, Inf, 9], T[1, nan, 9], T[nan, 0, nan])
    @test isnan(f(x))
  end
  # NaN in the first, middle and last entry of a group
  g = SparseGroupNormL2(T(Inf), T(1), [1:3], T[1])
  for i in 1:3
    x = T[0, 0, 0, 9]
    x[i] = nan
    @test isnan(g(x))
    x[i] = 2
    @test g(x) === T(Inf)
  end
  @test g(T[0, 0, 0, nan]) === zero(T)
  # the group level and weights do not change the indicator: zero, infinite, default
  for h in (SparseGroupNormL2(T(Inf), T(0), [1:2], T[1]), SparseGroupNormL2(T(Inf), T(1), [1:2], T[0]),
            SparseGroupNormL2(T(Inf), T(1), [1:2], T[Inf]), SparseGroupNormL2(T(Inf), T(0), [1:2], T[Inf]),
            SparseGroupNormL2(T(Inf), T(Inf), [1:2], T[1]), SparseGroupNormL2(T(Inf), T(1), [1:2]),
            SparseGroupNormL2(T(Inf), T(1), [1:2, 4:5], T[Inf, 1]))
    for x in (T[0, 0, 9, 0, 0], T[0, 0, nan, 0, 0], T[-0.0, 0, Inf, 0, 0], T[1, 0, 9, 0, 0], T[0, 0, 9, 0, 3],
              T[nan, 0, 9, 0, 0], T[0, 0, 9, nan, 0], T[nan, 0, 9, Inf, 0])
      @test agree(h(x), ref(h, x))
    end
  end
  # groups that cover every index
  f2 = SparseGroupNormL2(T(Inf), T(1), [1:2, 3:3], T[1, 1])
  @test f2(T[0, 0, 0]) === zero(T)
  @test f2(T[0, 0, 1]) === T(Inf)
  @test isnan(f2(T[0, nan, 0]))
end

@testset "value_and_prox! at the stored output, $T" for T in (Float32, Float64)
  nan = T(NaN)
  # the configurations whose proximal output is the hard fix: grouped entries zero, the rest copied
  for f in (SparseGroupNormL2(T(Inf), T(1), [1:2], T[1]), SparseGroupNormL2(T(Inf), T(1), [1:2], T[Inf]),
            SparseGroupNormL2(T(Inf), T(0), [1:2], T[Inf]), SparseGroupNormL2(T(Inf), T(Inf), [1:2], T[1]),
            SparseGroupNormL2(T(Inf), T(1), [1:2]))
    for (x, want) in ((T[3, 4, 9], T[0, 0, 9]), (T[nan, 4, 9], T[0, 0, 9]), (T[3, nan, 9], T[0, 0, 9]),
                      (T[3, 4, nan], T[0, 0, nan]), (T[Inf, 4, 9], T[0, 0, 9]), (T[3, 4, -Inf], T[0, 0, -Inf]))
      for γ in (T(1), T(0.5), T(Inf), T[1])
        v, y = fused(f, x, γ)
        @test agree(y, want)
        @test v === zero(T)
      end
    end
  end
  # a zero group level or weight: the infinite ℓ1 level still fixes the grouped entries at zero,
  # whatever they hold (#620)
  for f in (SparseGroupNormL2(T(Inf), T(0), [1:2], T[1]), SparseGroupNormL2(T(Inf), T(1), [1:2], T[0]))
    for γ in (T(1), T(0.5), T(Inf), T[1])
      for (x, want) in ((T[3, 4, 9], T[0, 0, 9]), (T[3, 4, nan], T[0, 0, nan]), (T[nan, 4, 9], T[0, 0, 9]),
                        (T[Inf, 4, 9], T[0, 0, 9]), (T[nan, nan, nan], T[0, 0, nan]))
        v, y = fused(f, x, γ)
        @test agree(y, want)
        @test v === zero(T)
      end
    end
  end
  # the step 0 also fixes the grouped entries at zero, for every group level and weight (#620)
  for f in (SparseGroupNormL2(T(Inf), T(1), [1:2], T[1]), SparseGroupNormL2(T(Inf), T(0), [1:2], T[1]),
            SparseGroupNormL2(T(Inf), T(1), [1:2], T[0]))
    for (x, want) in ((T[3, 4, 9], T[0, 0, 9]), (T[nan, 4, nan], T[0, 0, nan]))
      v, y = fused(f, x, zero(T))
      @test agree(y, want)
      @test v === zero(T)
    end
  end
  # a step per group, two groups
  f = SparseGroupNormL2(T(Inf), T(1), [1:2, 4:5], T[1, 1])
  v, y = fused(f, T[3, 4, 9, 1, 2], T[0.5, 2])
  @test agree(y, T[0, 0, 9, 0, 0])
  @test v === zero(T)
  f2 = SparseGroupNormL2(T(Inf), T(1), [1:2, 3:3], T[1, 1])
  v, y = fused(f2, T[3, 4, 9], T(1))
  @test agree(y, zeros(T, 3))
  @test v === zero(T)
end

@testset "AtomIterate, $T" for T in (Float32, Float64)
  nan = T(NaN)
  for f in (SparseGroupNormL2(T(Inf), T(1), [1:2, 3:3], T[1, 1]), SparseGroupNormL2(T(Inf), T(0), [1:2, 3:3], T[1, Inf]))
    for x in (T[0, 0, 0], T[0, -0.0, -0.0], T[1, 0, 0], T[0, 0, -2], T[nan, 0, 0], T[0, 0, nan], T[nan, Inf, 0])
      @test agree(f(AtomIterate(copy(x), f)), ref(f, x))
    end
    for x in (T[3, 4, 9], T[nan, 4, 9], T[3, 4, Inf])
      # every group is fixed at zero, also the first group of the second function (a zero group
      # level and a finite weight) at a NaN entry
      xa = AtomIterate(copy(x), f)
      ya = AtomIterate(fill(T(-7), 3), f)
      v = value_and_prox!(ya, f, xa, T(1))
      @test agree(v, f(ya))
      @test agree(v, ref(f, collect(ya)))
      @test agree(collect(ya), zeros(T, 3))
      @test v === zero(T)
      za = AtomIterate(copy(x), f)
      u = value_and_prox!(za, f, za, T(1))
      @test agree(u, f(za))
      @test agree(collect(za), collect(ya))
      @test u === zero(T)
    end
  end
end

@testset "wrappers and storage, $T" for T in (Float32, Float64)
  nan = T(NaN)
  f = SparseGroupNormL2(T(Inf), T(1), [1:2], T[1])
  # a restriction with no group left penalizes nothing
  @test restrict(f, [3])(T[9]) === zero(T)
  @test restrict(f, [3])(T[nan]) === zero(T)
  @test restrict(f, [1, 3])(T[0, 9]) === zero(T)
  @test restrict(f, [1, 3])(T[1, 9]) === T(Inf)
  @test isnan(restrict(f, [1, 3])(T[nan, 9]))
  # sparse storage
  for x in (T[0, 0, 9], T[2, 0, 9], T[nan, 0, 9])
    @test agree(f(SparseIterate(sparsevec(x))), ref(f, x))
    @test agree(f(sparsevec(x)), ref(f, x))
    @test agree(f(sparse(reshape([x; 0], 2, 2))), ref(f, x))
  end
  # an infinite level is not folded into a scaling
  s = Scaled(f, T(2))
  @test s(T[0, 0, 9]) === zero(T)
  @test s(T[1, 0, 9]) === T(Inf)
  @test isnan(s(T[nan, 0, 9]))
  y = similar(T[3, 4, 9])
  @test value_and_prox!(y, s, T[3, 4, 9], T(1)) === zero(T)
  @test agree(y, T[0, 0, 9])
  # with a ball constraint
  c = Constrained(f, IndBallL2(T(1)))
  @test c(T[0, 0, 1]) === zero(T)
  @test c(T[0, 0, 2]) === T(Inf)
  @test c(T[1, 0, 0]) === T(Inf)
  y = similar(T[3, 4, 9])
  v = value_and_prox!(y, c, T[3, 4, 9], T(1))
  @test agree(y, T[0, 0, 1])
  @test v === zero(T)
  @test agree(v, c(y))
  # as the fiber function of a FiberSum
  fb = FiberSum(SparseGroupNormL2(T(Inf), T(1), [1:2], T[1]))
  X = zeros(T, 2, 2, 2)
  @test fb(X) === zero(T)
  X[1, 2, 2] = 1
  @test fb(X) === T(Inf)
  X[2, 1, 1] = nan
  @test isnan(fb(X))
  Y = similar(X)
  v = value_and_prox!(Y, fb, randn(T, 2, 2, 2), T(1))
  @test agree(Y, zeros(T, 2, 2, 2))
  @test v === zero(T)
end

@testset "mixed precision and converted levels, $T" for T in (Float32, Float64)
  # Float64 parameters on data of type T: the stored level is infinite
  fm = SparseGroupNormL2(Inf, 1.0, [1:2], [1.0])
  @test fm(T[0, 0, 9]) === zero(T)
  @test fm(T[1, 0, 9]) === T(Inf)
  @test isnan(fm(T[NaN, 0, 9]))
  v, y = fused(fm, T[3, 4, 9], 1.0)
  @test agree(y, T[0, 0, 9])
  @test v === zero(T)
  # a finite level is never treated as infinite, even where it does not fit the data type (R3): the
  # stored level 2^128 with a zero group level; the step 0 is the identity, with the value 2^28 there
  fr = SparseGroupNormL2(ldexp(1.0, 128), 0.0, [1:2], [1.0])
  x = T[ldexp(one(T), -100), 0, 9]
  @test fr(x) === T(ldexp(1.0, 28))
  for g in (fr, adapt(JLArray, fr)), dev in (identity, JLArray)
    g === fr || dev === JLArray || continue
    y = dev(fill(T(-7), 3))
    @test value_and_prox!(y, g, dev(copy(x)), 0.0) === T(ldexp(1.0, 28))
    @test Array(y) == x
    y = dev(fill(T(-7), 3))
    @test value_and_prox!(y, g, dev(copy(x)), 1.0) === zero(T)
    @test Array(y) == T[0, 0, 9]
  end
end

@testset "device arrays (JLArrays), $T" for T in (Float32, Float64)
  nan = T(NaN)
  host(a) = Array(a)
  dev(a) = JLArray(a)
  fs = (SparseGroupNormL2(T(Inf), T(1), [1:2], T[1]), SparseGroupNormL2(T(Inf), T(0), [1:2], T[1]),
        SparseGroupNormL2(T(Inf), T(1), [1:2], T[0]), SparseGroupNormL2(T(Inf), T(1), [1:2], T[Inf]),
        SparseGroupNormL2(T(Inf), T(0), [1:2], T[Inf]), SparseGroupNormL2(Inf, 1.0, [1:2], [1.0]))
  for f in fs, g in (f, adapt(JLArray, f))
    # the value agrees with the CPU in class and value
    for x in (T[0, 0, 9], T[-0.0, 0, 9], T[0, 0, nan], T[0, 0, Inf], T[1, 0, 9], T[Inf, 0, 9], T[nan, 0, 9],
              T[0, nan, nan], T[nan, Inf, 9])
      @test agree(g(dev(x)), f(x))
      @test agree(g(dev(x)), ref(f, x))
    end
    # value_and_prox!, separate output and in place: the stored output of the CPU, and the value
    # there, also with a zero group level or weight at a NaN grouped entry
    for x in (T[3, 4, 9], T[3, 4, nan], T[nan, 4, 9])
      ref_y = fill(T(-7), 3)
      ref_v = value_and_prox!(ref_y, f, copy(x), T(1))
      @test agree(ref_y, T[0, 0, x[3]])
      @test ref_v === zero(T)
      y = dev(fill(T(-7), 3))
      v = value_and_prox!(y, g, dev(x), T(1))
      @test agree(v, f(host(y)))
      @test agree(host(y), ref_y)
      @test agree(v, ref_v)
      z = dev(copy(x))
      u = value_and_prox!(z, g, z, T(1))
      @test agree(u, f(host(z)))
      @test agree(host(z), ref_y)
      @test agree(u, ref_v)
    end
  end
end

@testset "finite levels unchanged, $T" for T in (Float32, Float64)
  # exactly representable controls: 0.25⋅7 + 0.5⋅5, and soft thresholding at 0.25 followed by the
  # shrinkage of the norm 5 at 2.5 (the factor 1/2), with the value 0.25⋅3.5 + 2.5⋅2.5
  f = SparseGroupNormL2(T(0.25), T(0.5), [1:2], T[1])
  @test f(T[3, 4, 9]) === T(4.25)
  g = SparseGroupNormL2(T(0.25), T(2.5), [1:2], T[1])
  y = fill(T(-7), 3)
  @test value_and_prox!(y, g, T[3.25, 4.25, 9], T(1)) === T(7.125)
  @test y == T[1.5, 2, 9]
  @test g(JLArray(T[3, 4, 9])) === g(T[3, 4, 9])
  @test isnan(f(T[NaN, 0, 9]))
  @test f(T[Inf, 0, 9]) === T(Inf)
  @test f(T[0, 0, NaN]) === zero(T)
end

end
