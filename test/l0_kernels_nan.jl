module L0KernelsNaNTest

using Test, ProximalBase, LinearAlgebra, SparseArrays, Random
using JLArrays: JLArray
using GPUArraysCore: allowscalar
using ..TestUtils

# NaN in the ℓ0 kernels with their own selection (E2, with E4 first): MCP, SCAD and NormL0 on a
# box, the counting functions of the singular values and IndBallRank, and IndBallGroupL0. A NaN
# entry gives NaN wherever the result depends on it; a map that the parameters make constant keeps
# its value. The fused value is the value at the stored output (O2). NaN is compared with isequal.

same(a, b) = isequal(a, b)

####################################
#  MCP, SCAD and NormL0 on a box
####################################

# The proximal entry at NaN: the clamped zero where the parameters fix the coordinate (a box of one
# point, an infinite level), NaN otherwise, also at γ = Inf, whose constant map a tilt would move.
# MCP and SCAD with an infinite level at γ = 0 are NaN at every entry, finite ones included (the
# unconstrained proximal map there, a separate gap).
ref_box_nan(name, λ, l, u, γ, ::Type{T}) where {T} =
  name != "NormL0" && isinf(λ) && iszero(γ) ? T(NaN) :
  (l == u || isinf(λ)) ? clamp(zero(T), T(l), T(u)) : T(NaN)

# The value at an argument with a NaN entry: NaN at a NaN entry that the value depends on, zero at
# one of an unbounded coordinate of the zero level, and the membership of the box elsewhere.
function ref_box_value(λ, l, u, x::AbstractVector{T}) where {T}
  s = zero(T)
  for i in eachindex(x)
    li, ui = l isa AbstractArray ? l[i] : l, u isa AbstractArray ? u[i] : u
    v = x[i]
    s += isnan(v) ? (iszero(λ) && li == -Inf && ui == Inf ? zero(T) : T(NaN)) :
         li <= v <= ui ? zero(T) : T(Inf)
  end
  s     # zero only when λ = 0, where the penalty is zero
end

penalties(T) = (("NormL0", λ -> NormL0(T(λ))), ("MCP", λ -> MCP(T(λ), T(3))), ("SCAD", λ -> SCAD(T(λ), T(37) / 10)))
boxes(T) = (IndBox(T(-2), T(2)), IndBox(T(1), T(2)), IndBox(T(1), T(1)), IndBox(T(-Inf), T(Inf)),
            IndBox(zero(T), T(Inf)), IndBox(T(-Inf), zero(T)))

@testset "proximal entries on a box at NaN, $T" for T in (Float32, Float64)
  for (name, mk) in penalties(T), λ in (0, 1, Inf), B in boxes(T), γ in T.((0, 0.5, 1, 5, Inf))
    c = Constrained(mk(λ), B)
    expected = ref_box_nan(name, λ, B.l, B.u, γ, T)
    @test same(prox_entry(c, T(NaN), 1, γ), expected)
    # NaN first, in the middle and last, and two NaNs; the finite entries are those of the
    # proximal map at the same vector with the NaN replaced (the map is separable)
    for x in (T[NaN, 1.5, -0.5], T[1.5, NaN, -0.5], T[1.5, -0.5, NaN], T[NaN, 1.5, NaN])
      filled = map(v -> isnan(v) ? T(0.7) : v, x)
      reference = prox(c, filled, γ)
      want = [isnan(x[i]) ? expected : reference[i] for i in eachindex(x)]
      @test same(prox(c, x, γ), want)
      z = copy(x); @test same(prox!(z, c, z, γ), want)                       # aliased
      @test same(prox(c, x, fill(γ, 3)), want)                               # a step per entry
      y = similar(x); v = value_and_prox!(y, c, x, γ)
      @test same(y, want)
      @test same(v, c(y))                                                    # O2
      z = copy(x); @test same(value_and_prox!(z, c, z, γ), v) && same(z, want)
    end
  end
  # the issue's case and IndNonnegative
  c = Constrained(NormL0(T(1)), IndBox(T(-2), T(2)))
  @test same(prox(c, T[NaN, 1.5], T(1)), T[NaN, 1.5])
  y = zeros(T, 2); @test same(value_and_prox!(y, c, T[NaN, 1.5], T(1)), T(NaN)) && same(y, T[NaN, 1.5])
  @test same(prox(Constrained(NormL0(T(1)), IndBox(T(1), T(2))), T[NaN, 1.5], T(1)), T[NaN, 1.5])
  @test same(prox(Constrained(NormL0(T(1)), IndNonnegative()), T[NaN, 1.5], T(1)), T[NaN, 1.5])
  # at γ = Inf the finite entries take the constant map and a NaN stays NaN
  @test same(prox(c, T[NaN, 1.5], T(Inf)), T[NaN, 0])
  @test same(value_and_prox!(zeros(T, 2), c, T[NaN, 1.5], T(Inf)), T(NaN))
  # the maps that the parameters fix keep their values
  @test same(prox(Constrained(NormL0(T(Inf)), IndBox(T(-2), T(2))), T[NaN, 1.5], T(Inf)), T[0, 0])
  @test same(prox(Constrained(SCAD(T(1), T(37) / 10), IndBox(T(1), T(1))), T[NaN, 1.5], T(Inf)), T[1, 1])
  @test same(prox(Constrained(MCP(T(1), T(3)), IndBox(T(1), T(1))), T[NaN, 1.5], T(1 // 2)), T[1, 1])
end

@testset "values on a box at NaN, $T" for T in (Float32, Float64)
  for (name, mk) in penalties(T), λ in (0, 1, Inf), B in boxes(T)
    c = Constrained(mk(λ), B)
    for x in (T[NaN, 1.5], T[NaN, 3], T[NaN], T[3, NaN], T[NaN, NaN], T[NaN, Inf], T[-Inf, NaN])
      @test same(c(x), ref_box_value(λ, B.l, B.u, x))
    end
  end
  @test same(Constrained(NormL0(T(1)), IndBox(T(-2), T(2)))(T[NaN, 1.5]), T(NaN))       # the issue: Inf before
  @test Constrained(NormL0(T(0)), IndBox(T(-Inf), T(Inf)))(T[NaN]) === zero(T)          # the zero function
  # a vector box with an unbounded and a bounded coordinate, at the zero level
  c0 = Constrained(NormL0(T(0)), IndBox(T[-Inf, 0], T[Inf, 1]))
  @test c0(T[NaN, 0.5]) === zero(T)
  @test c0(T[NaN, 2]) === T(Inf)
  @test same(c0(T[0.5, NaN]), T(NaN))
  y = zeros(T, 2); @test value_and_prox!(y, c0, T[NaN, 0.5], T(1)) === zero(T) && same(y, T[NaN, 0.5])
  y = zeros(T, 2); @test same(value_and_prox!(y, c0, T[0.5, NaN], T(1)), T(NaN)) && same(y, T[0.5, NaN])
  # the zero level ignores a NaN that the box still depends on: the fused value is that of the box
  c1 = Constrained(NormL0(T(0)), IndBox(T(-2), T(2)))
  y = zeros(T, 1); @test same(value_and_prox!(y, c1, T[NaN], T(1)), T(NaN)) && same(y, T[NaN])
  # ordinary and infinite controls keep their values
  c = Constrained(NormL0(T(1)), IndBox(T(-Inf), T(Inf)))
  @test c(T[Inf, 1.5]) === T(2) && c(T[0, 1.5]) === T(1)
  @test Constrained(NormL0(T(1)), IndBox(T(-2), T(2)))(T[3, 1.5]) === T(Inf)
  # wrappers and a symmetric iterate
  c = Constrained(NormL0(T(1)), IndBox(T(-2), T(2)))
  @test same(Scaled(c, T(2))(T[NaN, 1.5]), T(NaN))
  @test same(Translate(c, zeros(T, 2))(T[NaN, 1.5]), T(NaN))
  S = SymmetricSparseIterate(T[NaN 0.5; 0.5 1])
  @test same(c(S), T(NaN))
  @test Constrained(NormL0(T(0)), IndBox(T(-Inf), T(Inf)))(S) === zero(T)
  @test c(SymmetricSparseIterate(T[1 0.5; 0.5 3])) === T(Inf)
  x = SparseIterate(3); x[2] = T(NaN)
  @test same(Constrained(NormL0(1.0), IndBox(-2.0, 2.0))(x), NaN)
end

@testset "a box on a device array, $T" for T in (Float32, Float64)
  allowscalar(false)
  for f in (NormL0(T(1)), MCP(T(1), T(3)), SCAD(T(1), T(37) / 10)), γ in (T(1), T(5))
    c = Constrained(f, IndBox(T(-2), T(2)))
    x = T[NaN, 1.5, -0.5]
    d = JLArray(x)
    @test same(Array(prox(c, d, γ)), prox(c, x, γ))
    @test same(c(d), c(x))
    yd = similar(d); y = similar(x)
    @test same(value_and_prox!(yd, c, d, γ), value_and_prox!(y, c, x, γ)) && same(Array(yd), y)
  end
  c0 = Constrained(NormL0(T(0)), IndBox(JLArray(T[-Inf, 0]), JLArray(T[Inf, 1])))
  @test c0(JLArray(T[NaN, 0.5])) === zero(T)
  @test same(c0(JLArray(T[0.5, NaN])), T(NaN))
  @test c0(JLArray(T[NaN, 2])) === T(Inf)
end

####################################
#  Counting functions of the singular values, and IndBallRank
####################################

# the spectral functions, with whether they are constant on 2 × 2 and 2 × 3 matrices
spectral_cases(T) = (
  (SingularValue(NormL0(T(1))), false), (SingularValue(NormL0(T(Inf))), false),
  (SingularValue(IndBallL0(0)), false), (SingularValue(IndBallL0(1)), false),
  (IndBallRank(1), false), (IndBallRank(1; partial = true), false),
  (Scaled(SingularValue(NormL0(T(1))), T(2)), false), (SingularValue(Scaled(NormL0(T(1)), T(2))), false),
  (SingularValue(Constrained(NormL0(T(1)), IndBox(T(-5), T(5)))), false),
  (SingularValue(Constrained(NormL0(T(0)), IndBox(T(-5), T(5)))), false),
  (SingularValue(NormL0(T(0))), true), (SingularValue(IndBallL0(2)), true), (IndBallRank(2), true),
  (SingularValue(Constrained(NormL0(T(0)), IndBox(T(-Inf), T(Inf)))), true),
  (SingularValue(Constrained(NormL0(T(0)), IndNonnegative())), true))

copies(f) = f isa IndBallRank && f.r >= 2

@testset "counting functions of the singular values at NaN, $T" for T in (Float32, Float64)
  for (f, constant) in spectral_cases(T), dims in ((2, 2), (2, 3)),
      I in (CartesianIndex(1, 1), CartesianIndex(1, 2), CartesianIndex(2, 1)), w in (:nan, :inf, :both)
    input = reshape(T.(1:prod(dims)), dims)
    input[I] = w === :inf ? T(Inf) : T(NaN)
    w === :both && (input[2, 2] = -T(Inf))
    hasinf = w !== :nan
    # the value: NaN where it depends on the NaN, zero for a constant function at NaN alone, and
    # Inf at an infinite entry otherwise
    expected = constant ? (hasinf ? T(Inf) : zero(T)) : (w === :inf ? T(Inf) : T(NaN))
    @test same(f(input), expected)
    # the fused value at the output: a copy (IndBallRank(2)) or all NaN
    for alias in (false, true)
      x = copy(input)
      y = alias ? x : zeros(T, dims)
      v = value_and_prox!(y, f, x, one(T))
      if copies(f)
        @test same(y, input) && same(v, expected)
      else
        @test all(isnan, y) && same(v, constant ? zero(T) : T(NaN))
      end
    end
    @test same(f(JLArray(input)), expected)
  end
  # a NaN entry, its fused value on the device, and no allocation on the NaN path
  X = T[NaN 1; 2 3]
  f = SingularValue(NormL0(T(1)))
  @test same(value_and_prox!(similar(JLArray(X)), f, JLArray(X), one(T)), T(NaN))
  for g in (f, SingularValue(IndBallL0(1)), IndBallRank(1), IndBallRank(2))
    y = similar(X)
    # warmed by a finite call, which sets up the workspace that the checks read
    value_and_prox!(y, g, T[3 0; 0 1], one(T)); g(X); value_and_prox!(y, g, X, one(T))
    @test allocs(g, X) == 0
    @test allocs(value_and_prox!, y, g, X, one(T)) == 0
  end
  # the functions that do not count nonzeros follow E2 at a NaN entry too (PR #384)
  @test same(SingularValue(NormL1(T(1)))(X), T(NaN))
  @test same(value_and_prox!(similar(X), SingularValue(NormL1(T(1))), X, one(T)), T(NaN))
  # finite matrices keep their values
  F = T[3 0; 0 1]
  @test SingularValue(NormL0(T(1)))(F) === T(2) && IndBallRank(1)(F) === T(Inf) && IndBallRank(2)(F) === zero(T)
end

####################################
#  IndBallGroupL0
####################################

# The projection, written independently of the package's selection: s = 0 and k = 0 zero every
# group; with fewer places than groups a NaN in a group makes every grouped entry NaN; the groups
# of largest norm of their k largest entries are kept, the earlier first among ties, and within
# each the k largest entries, the earlier first, a NaN making the group NaN.
function ref_group_prox(x::AbstractVector{T}, groups, s, k) where {T}
  y = float.(copy(x))
  K = length(groups)
  if s == 0 || k == 0
    for G in groups, i in G
      y[i] = 0
    end
    return y
  end
  if s < K && any(G -> any(isnan, x[G]), groups)
    for G in groups, i in G
      y[i] = NaN
    end
    return y
  end
  topnorm(G) = norm(sort(abs.(x[G]); rev = true)[1:min(k, length(G))])
  kept = s < K ? sortperm([topnorm(G) for G in groups]; rev = true, alg = MergeSort)[1:s] : collect(1:K)
  for (j, G) in enumerate(groups)
    if !(j in kept)
      y[G] .= 0
    elseif length(G) > k
      if any(isnan, x[G])
        y[G] .= NaN
      else
        order = sortperm(abs.(x[G]); rev = true, alg = MergeSort)
        for t in order[k+1:end]
          y[G[t]] = 0
        end
      end
    end
  end
  y
end

# The indicator: NaN at a NaN in a group that the value depends on (every group when s < K, and
# a group with more than k entries), and membership otherwise.
function ref_group_value(x::AbstractVector{T}, groups, s, k) where {T}
  K = length(groups)
  any(G -> (s < K || length(G) > k) && any(isnan, x[G]), groups) && return float(T)(NaN)
  nz = count(G -> any(!iszero, x[G]), groups)
  fits = all(G -> length(G) <= k || count(!iszero, x[G]) <= k, groups)
  nz <= s && fits ? zero(float(T)) : float(T)(Inf)
end

group_rows(T) = (T[NaN, 1, 2, 3], T[NaN, 1, 0, 0], T[1, NaN, 2, 3], T[1, 2, 3, NaN], T[NaN, NaN, 2, 3],
                 T[Inf, NaN, 2, 3], T[NaN, Inf, 2, 3], T[Inf, 1, 2, 3], T[NaN, 0, 0, 0], T[1, 0, 2, 3, NaN],
                 T[0, 0, 2, 3, NaN], T[NaN, 0, 2, 3, 1], T[3, 1, 2, 2], T[0, 0, 0, 0])

@testset "IndBallGroupL0 at NaN, $T" for T in (Float32, Float64)
  for groups in ([1:2, 3:4], [[1, 3], [2, 4]]), s in (0, 1, 2, 3), k in (nothing, 0, 1, 2)
    f = IndBallGroupL0(s, groups; k)
    kk = k === nothing ? typemax(Int) : k
    for x in group_rows(T)
      expected = ref_group_prox(x, groups, s, kk)
      @test same(prox(f, x), expected)
      z = copy(x); @test same(prox!(z, f, z, one(T)), expected)
      @test same(f(x), ref_group_value(x, groups, s, kk))
      y = similar(x)
      v = value_and_prox!(y, f, x, one(T))
      @test same(y, expected)
      @test same(v, ref_group_value(expected, groups, s, kk))               # O2
      @test same(v, f(y))
      z = copy(x); @test same(value_and_prox!(z, f, z, one(T)), v) && same(z, expected)
    end
  end
  # the issue's cases
  f = IndBallGroupL0(1, [1:2, 3:4])
  @test all(isnan, prox(f, T[NaN, 1, 2, 3])) && all(isnan, prox(f, T[NaN, 1, 0, 0]))
  @test same(f(T[NaN, 1, 0, 0]), T(NaN)) && same(IndBallGroupL0(0, [1:2, 3:4])(T[NaN, 1, 0, 0]), T(NaN))
  @test same(value_and_prox!(zeros(T, 4), f, T[NaN, 1, 2, 3], one(T)), T(NaN))
  @test same(prox(IndBallGroupL0(2, [1:2, 3:4]; k = 1), T[NaN, 1, 2, 3]), T[NaN, NaN, 0, 3])
  # wrappers of the projection
  @test same(SqrDistL2(f)(T[NaN, 1, 2, 3]), T(NaN)) && all(isnan, prox(SqrDistL2(f), T[NaN, 1, 2, 3], one(T)))
  @test same(DistL2(f)(T[NaN, 1, 2, 3]), T(NaN))
end

@testset "IndBallGroupL0 on an AtomIterate at NaN, $T" for T in (Float32, Float64)
  groups = [1:2, 3:4, 5:6]
  for s in (0, 1, 2, 3), k in (nothing, 1),
      x0 in (T[NaN, 1, 2, 3, 0, 0], T[NaN, 1, 0, 0, 0, 0], T[0, 0, 2, NaN, 0, 0], T[1, 0, 2, 3, 0, 0])
    f = IndBallGroupL0(s, groups; k)
    dense = prox(f, x0)
    a = AtomIterate(x0, groups)
    b = AtomIterate(zeros(T, 6), groups)
    @test same(Vector(prox!(b, f, a, one(T))), dense)
    @test nactive(b) == count(G -> any(!iszero, dense[G]), groups)
    @test same(f(a), f(x0))
    b = AtomIterate(zeros(T, 6), groups)
    @test same(value_and_prox!(b, f, a, one(T)), value_and_prox!(similar(x0), f, x0, one(T)))
    @test same(Vector(b), dense)
  end
  # iterates whose groups differ from those of f take the dense route
  f = IndBallGroupL0(1, groups)
  x0 = T[NaN, 1, 2, 3, 0, 0]
  a = AtomIterate(x0, [1:3, 4:6]); b = AtomIterate(zeros(T, 6), [1:3, 4:6])
  @test same(Vector(prox!(b, f, a, one(T))), prox(f, x0))
  b = AtomIterate(zeros(T, 6), [1:3, 4:6])
  @test same(value_and_prox!(b, f, a, one(T)), T(NaN)) && same(Vector(b), prox(f, x0))
end

@testset "IndBallGroupL0 on a device array, $T" for T in (Float32, Float64)
  allowscalar(false)
  for groups in ([1:2, 3:4], [[1, 3], [2, 4]]), s in (0, 1, 2), x in group_rows(T)
    f = IndBallGroupL0(s, groups)
    d = JLArray(x)
    @test same(Array(prox(f, d)), prox(f, x))
    yd = similar(d)
    @test same(value_and_prox!(yd, f, d, one(T)), value_and_prox!(similar(x), f, x, one(T)))
    @test same(Array(yd), prox(f, x))
    @test same(f(d), f(x))
    @test same(IndBallGroupL0(s, groups; k = 1)(d), IndBallGroupL0(s, groups; k = 1)(x))
  end
end

@testset "IndBallGroupL0 on finite inputs, $T" for T in (Float32, Float64)
  rng = MersenneTwister(592)
  groups = [1:3, 4:6, 7:9, 10:12]
  for trial in 1:50, s in (1, 2, 4), k in (nothing, 2)
    # small integers: the sums of squares are exact, so ties between groups are exact too
    x = T.(rand(rng, -3:3, 12))
    f = IndBallGroupL0(s, groups; k)
    kk = k === nothing ? typemax(Int) : k
    @test prox(f, x) == ref_group_prox(x, groups, s, kk)
    @test f(x) === ref_group_value(x, groups, s, kk)
    @test value_and_prox!(similar(x), f, x, one(T)) === zero(T)
  end
end

end
