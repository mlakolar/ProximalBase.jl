module IndNonnegativeTest
# IndNonnegative (#948): the indicator of {x : x ≥ 0}. The value rule (NaN at a NaN entry, Inf at a negative entry, 0
# otherwise), the projection max(v, 0) (-0.0 and -Inf give +0.0, NaN stays NaN), the rejection of negative and NaN scalar
# steps by prox, prox! and value_and_prox!, the routes that do not check the step, the representations (dense arrays,
# SparseIterate, BlockIterate, SymmetricSparseIterate), validation before writes, the fused value at the stored output,
# Scaled, the screen of kkt_violation and the allocations.
#
# The references are written here with comparisons and ifelse, not with max or the kernels of the package, and are
# compared with isequal and the type of the result, so that signed zeros and NaN count.
using Test, ProximalBase, LinearAlgebra, SparseArrays, OffsetArrays, Random
using ..TestUtils
const PB = ProximalBase

const TYPES = (Float32, Float64)
const IND = IndNonnegative()

# ---- references ----------------------------------------------------------------------------------------------
refentry(v::T) where {T<:AbstractFloat} = isnan(v) ? T(NaN) : (v >= 0 ? zero(T) : T(Inf))
refprox(v::T) where {T<:AbstractFloat} = isnan(v) ? v : (v > 0 ? v : zero(T))
function refvalue(x)
  T = float(eltype(x))
  nan = neg = false
  for v in x
    isnan(v) && (nan = true)
    v < 0 && (neg = true)
  end
  nan ? T(NaN) : (neg ? T(Inf) : zero(T))
end
refproject(x) = map(refprox, float.(x))
same(a, b) = typeof(a) === typeof(b) && isequal(a, b)
sameall(a, b) = size(a) == size(b) && eltype(a) === eltype(b) && all(isequal(u, v) for (u, v) in zip(a, b))

# the kinds of entry: the infinities, the extremes, the subnormals, the signed zeros, ordinary numbers, NaN
specials(::Type{T}) where {T} = T[-Inf, -floatmax(T), -one(T), -floatmin(T), -nextfloat(zero(T)), -zero(T), zero(T),
                                  nextfloat(zero(T)), floatmin(T), one(T), floatmax(T), Inf, NaN]
nonan(::Type{T}) where {T} = filter(!isnan, specials(T))
# the steps that the public oracles admit, and those that they refuse
admitted(::Type{T}) where {T} = Any[T(0), -zero(T), floatmin(T), one(T), floatmax(T), T(Inf), 0, 1]
refused(::Type{T}) where {T} = Any[-one(T), T(-Inf), T(NaN), -1, -floatmin(T)]

# a counting element type: zero(Counted) is counted, so that reading an unstored entry of an iterate through getindex
# (which forms a zero) is seen. The same device as in sparse_value_cost.jl, kept to what these oracles use.
const ZEROS = Ref(0)
struct Counted <: AbstractFloat
  v::Float64
end
Base.zero(::Type{Counted}) = (ZEROS[] += 1; Counted(0.0))
Base.zero(::Counted) = zero(Counted)
Base.float(::Type{Counted}) = Float64
Base.float(x::Counted) = x.v
Base.convert(::Type{Counted}, x::Real) = Counted(Float64(x))
Base.convert(::Type{Float64}, x::Counted) = x.v
Base.Float64(x::Counted) = x.v
Base.promote_rule(::Type{Counted}, ::Type{T}) where {T<:Real} = promote_type(Float64, T)
Base.isnan(x::Counted) = isnan(x.v)
Base.iszero(x::Counted) = iszero(x.v)
Base.:>=(x::Counted, y::Real) = x.v >= y
Base.:<(x::Counted, y::Real) = x.v < y
zeros_made(g) = (ZEROS[] = 0; r = g(); (ZEROS[], r))

# Writing a nonzero first keeps a coordinate stored when its value is zero.
function fill_entries!(x, data)
  T = eltype(x)
  for (i, v) in data
    x[i] = T(1)
    x[i] = T(v)
  end
  x
end

# the same for a matrix iterate, with the entries (row, column, value)
function fill_matrix!(x, data)
  T = eltype(x)
  for (r, c, v) in data
    x[r, c] = T(1)
    x[r, c] = T(v)
  end
  x
end

@testset "truth table, entries" begin
  for T in TYPES
    vals = specials(T)
    for v in vals
      @test same(value_entry(IND, v, 1), refentry(v))
      for γ in (T(0), one(T), T(Inf), -one(T), T(NaN))         # prox_block ignores the step
        @test same(prox_block(IND, v, 1, γ), refprox(v))
      end
      @test same(IND([v]), refvalue([v]))
      @test sameall(prox(IND, [v], one(T)), [refprox(v)])
    end
    # the explicit cases of the contract
    @test same(prox_block(IND, T(-2), 1, one(T)), zero(T))
    @test same(prox_block(IND, -zero(T), 1, one(T)), zero(T))
    @test same(prox_block(IND, T(-Inf), 1, one(T)), zero(T))
    @test same(prox_block(IND, T(3), 1, one(T)), T(3))
    @test same(prox_block(IND, T(Inf), 1, one(T)), T(Inf))
    @test same(prox_block(IND, T(NaN), 1, one(T)), T(NaN))
    @test same(IND(T[NaN, -1]), T(NaN))
    @test same(IND(T[1, -1e-30]), T(Inf))
    @test same(IND(T[-Inf]), T(Inf))
    @test same(IND(T[-0.0, 0.0, Inf]), zero(T))
    @test same(IND(T[]), zero(T))
    @test same(IND(zeros(T, 0, 3)), zero(T))
    # pairs of entries: the sum of the entry values is NaN, then Inf, then 0
    for a in vals, b in vals
      x = [a, b]
      @test same(IND(x), refvalue(x))
      @test sameall(prox(IND, x, one(T)), refproject(x))
    end
    # the whole table in one array
    @test same(IND(vals), T(NaN))
    @test same(IND(nonan(T)), T(Inf))
    @test same(IND(filter(v -> v >= 0, nonan(T))), zero(T))
    @test sameall(prox(IND, vals, one(T)), refproject(vals))
    @test sameall(prox(IND, vals, T(Inf)), refproject(vals))
  end
end

@testset "truth table, integer entries" begin
  @test same(IND([1, -2]), Inf)
  @test same(IND([1, 2]), 0.0)
  @test same(IND(Int32[1, 2]), 0.0)
  @test same(IND(Int[]), 0.0)
  @test same(value_entry(IND, -2, 1), Inf)
  @test same(value_entry(IND, Int8(-1), 1), Inf) && same(IND(Int32[-1]), Inf) && same(IND(Int8[0, 3]), 0.0)
  @test same(value_entry(IND, 0, 1), 0.0)
  @test same(prox_block(IND, -2, 1, 1.0), 0.0)
  @test same(prox_block(IND, 3, 1, 1.0), 3.0)
  @test sameall(prox(IND, [1, -2, 0], 1.0), [1.0, 0.0, 0.0])
  @test sameall(prox(IND, Int32[1, -2, 0], 1.0), Float64[1, 0, 0])
  y = zeros(3)
  @test same(value_and_prox!(y, IND, [1, -2, 0], 1.0), 0.0) && sameall(y, [1.0, 0.0, 0.0])
end

@testset "lengths and the position of the exceptional entry" begin
  for T in TYPES, n in (1, 2, 7, 8, 15, 16, 17, 100, 1000), v in specials(T)
    for pos in unique((1, n))
      x = ones(T, n); x[pos] = v
      @test same(IND(x), refvalue(x))
      y = similar(x)
      @test prox!(y, IND, x, one(T)) === y && sameall(y, refproject(x))
    end
    x = [specials(T)[mod1(3i, 13)] for i in 1:n]
    @test same(IND(x), refvalue(x))
    @test sameall(prox(IND, x, T(2)), refproject(x))
    xn = filter(!isnan, x)
    @test same(IND(xn), refvalue(xn))
  end
end

@testset "steps" begin
  for T in TYPES
    x = specials(T)
    ref = refproject(x)
    for γ in admitted(T)
      @test sameall(prox(IND, x, γ), ref)
      y = fill(T(7), length(x))
      @test prox!(y, IND, x, γ) === y && sameall(y, ref)
      y = fill(T(7), length(x))
      v = value_and_prox!(y, IND, x, γ)
      @test same(v, T(NaN)) && sameall(y, ref)
    end
    xn = nonan(T)
    for γ in admitted(T)
      y = similar(xn)
      @test same(value_and_prox!(y, IND, xn, γ), zero(T)) && sameall(y, refproject(xn))
    end
    for γ in refused(T)
      @test_throws "IndNonnegative" prox(IND, x, γ)
      y = fill(T(7), length(x))
      @test_throws ArgumentError prox!(y, IND, x, γ)
      @test y == fill(T(7), length(x))
      @test_throws ArgumentError value_and_prox!(y, IND, x, γ)
      @test y == fill(T(7), length(x))
      # an aliased argument is left as it was
      a = copy(x)
      @test_throws ArgumentError prox!(a, IND, a, γ)
      @test sameall(a, x)
      @test_throws ArgumentError value_and_prox!(a, IND, a, γ)
      @test sameall(a, x)
      # the same for iterates
      s = SparseIterate(T, 4); fill_entries!(s, [(2, -1.0), (3, 2.0)]); s0 = copy(s)
      @test_throws ArgumentError prox!(s, IND, s, γ)
      @test_throws ArgumentError value_and_prox!(s, IND, s, γ)
      @test sameall(Vector(s), Vector(s0)) && collect(activecoordinates(s)) == collect(activecoordinates(s0))
      b = BlockIterate(T[1, -2], [[1], [2]]); b0 = copy(b)
      @test_throws ArgumentError prox!(b, IND, b, γ)
      @test sameall(Vector(b), Vector(b0))
      S = SymmetricSparseIterate(T[1 -2; -2 0]); S0 = copy(S)
      @test_throws ArgumentError prox!(S, IND, S, γ)
      @test_throws ArgumentError value_and_prox!(S, IND, S, γ)
      @test_throws ArgumentError prox(IND, S, γ)
      @test sameall(Matrix(S), Matrix(S0)) && collect(activecoordinates(S)) == collect(activecoordinates(S0))
    end
    # an array of steps, any entries, projects: the entries are read by the loops, not checked, and do not change the projection
    steps = T[mod(i, 3) == 0 ? -one(T) : (mod(i, 3) == 1 ? T(NaN) : T(Inf)) for i in eachindex(x)]
    @test sameall(prox(IND, x, steps), ref)
    y = similar(x)
    @test prox!(y, IND, x, steps) === y && sameall(y, ref)
    @test same(value_and_prox!(y, IND, x, steps), T(NaN)) && sameall(y, ref)
    @test sameall(prox(IND, x, fill(-one(T), length(x))), ref)
    # prox_block, and the routes of Scaled, do not check the step, and project
    @test same(prox_block(IND, T(-3), 1, -one(T)), zero(T))
    @test same(prox_block(IND, T(3), 1, T(NaN)), T(3))
    for γ in (-one(T), T(NaN))
      @test sameall(prox(Scaled(IND, 2), x, γ), ref)
      y = similar(x)
      @test sameall(prox!(y, Scaled(IND, 2), x, γ), ref)
    end
  end
end

@testset "representations of dense arguments" begin
  for T in TYPES
    vals = nonan(T)                                         # 12 entries
    x = copy(vals)
    ref = refproject(x)
    # Matrix and reshape
    M = reshape(copy(vals), 3, 4)
    @test same(IND(M), T(Inf)) && sameall(prox(IND, M, one(T)), reshape(ref, 3, 4))
    R = reshape(copy(vals), 4, 3)
    ym = similar(R)
    @test prox!(ym, IND, R, one(T)) === ym && sameall(ym, reshape(ref, 4, 3))
    @test same(value_and_prox!(ym, IND, R, one(T)), zero(T))
    # a strided view, and a view of a matrix
    bigx = vcat(vals, vals)
    V = view(bigx, 1:2:24)
    @test same(IND(V), refvalue(V)) && sameall(prox(IND, V, one(T)), refproject(V))
    yv = fill(T(9), 24); out = view(yv, 2:2:24)
    @test prox!(out, IND, V, one(T)) === out && sameall(out, refproject(V))
    @test all(yv[1:2:24] .== T(9))                          # the other entries of the buffer are not written
    @test same(value_and_prox!(out, IND, V, one(T)), zero(T))
    Mv = view(reshape(copy(bigx), 4, 6), 2:3, 2:5)
    @test same(IND(Mv), refvalue(Mv)) && sameall(prox(IND, Mv, one(T)), refproject(Mv))
    # in place
    z = copy(x)
    @test prox!(z, IND, z, one(T)) === z && sameall(z, ref)
    z = copy(x)
    @test same(value_and_prox!(z, IND, z, one(T)), zero(T)) && sameall(z, ref)
    z = copy(vals); z[3] = T(NaN)
    @test same(value_and_prox!(z, IND, z, one(T)), T(NaN)) && isnan(z[3]) && z[1] === zero(T)
    # a one-based array of another type
    @test same(IND(1:3), 0.0) && sameall(prox(IND, 1:3, 1.0), [1.0, 2.0, 3.0])
    @test same(IND(-1:1), Inf)
  end
end

@testset "SparseIterate" begin
  for T in TYPES
    n = 8
    x = SparseIterate(T, n)
    x[2] = 3; x[4] = -1; x[6] = T(0.5)
    activate!(x, 5); activate!(x, 7); x[7] = -zero(T)         # stored zeros, one of them -0.0
    dense = Vector(x)
    @test same(IND(x), refvalue(dense)) && same(IND(x), T(Inf))
    γ = T(0.5)
    # a fresh output: the zero results are not stored
    y = prox(IND, x, γ)
    @test y isa SparseIterate{T} && sameall(Vector(y), refproject(dense))
    @test sort(collect(activecoordinates(y))) == [2, 6]
    # in place: every stored entry stays stored, a stored negative becomes a stored +0
    z = copy(x)
    @test prox!(z, IND, z, γ) === z && sameall(Vector(z), refproject(dense))
    @test nactive(z) == nactive(x) && sort(collect(activecoordinates(z))) == sort(collect(activecoordinates(x)))
    @test z[4] === zero(T) && z[7] === zero(T) && z[5] === zero(T) && isactive(z, 4) && isactive(z, 7)
    @test same(IND(z), zero(T))
    # an output with a stored entry of its own: it is overwritten, and stays stored
    out = SparseIterate(T, n); out[1] = 5
    v = value_and_prox!(out, IND, x, γ)
    @test same(v, zero(T)) && same(v, IND(out)) && isactive(out, 1) && out[1] === zero(T)
    @test sameall(Vector(out), refproject(dense))
    # a stored NaN: the value is NaN and the entry stays NaN
    xn = copy(x); xn[3] = T(NaN)
    @test same(IND(xn), T(NaN))
    out = SparseIterate(T, n)
    v = value_and_prox!(out, IND, xn, γ)
    @test same(v, T(NaN)) && same(v, IND(out)) && isnan(out[3]) && sameall(Vector(out), refproject(Vector(xn)))
    zn = copy(xn)
    @test same(value_and_prox!(zn, IND, zn, γ), T(NaN)) && isnan(zn[3])
    # array steps, any entries
    steps = T[-1, NaN, Inf, 0, -2, 1, 3, 4]
    @test sameall(Vector(prox(IND, x, steps)), refproject(dense))
    zs = copy(x)
    @test same(value_and_prox!(zs, IND, zs, steps), zero(T)) && sameall(Vector(zs), refproject(dense))
    # nothing stored
    e = SparseIterate(T, n)
    @test same(IND(e), zero(T)) && nactive(prox(IND, e, γ)) == 0
    # an infeasible stored value only
    f1 = fill_entries!(SparseIterate(T, n), [(3, -2.0)])
    @test same(IND(f1), T(Inf)) && same(IND(fill_entries!(SparseIterate(T, n), [(3, 2.0)])), zero(T))
    # a matrix iterate
    A = SparseIterate(T, 3, 3); fill_entries!(A, [(2, -1.0), (5, 2.0), (9, -3.0)])
    @test same(IND(A), T(Inf)) && sameall(Array(prox(IND, A, γ)), refproject(Array(A)))
  end
end

@testset "SparseIterate reads stored entries only" begin
  p = 10_000
  data = [(1, 1.0), (7, -4.0), (9, 0.0), (500, 2.5)]
  vec_iterate(T) = fill_entries!(SparseIterate(T, p), data)
  # control: a generic pass over the logical vector is counted
  n, _ = zeros_made(() -> count(isnan, fill_entries!(SparseIterate(Counted, 300), [(1, 1.0), (2, 0.5)])))
  @test n >= 300 - 4
  x = vec_iterate(Counted); xr = vec_iterate(Float64)
  n, v = zeros_made(() -> IND(x))
  @test n == 0 && same(v, IND(xr)) && same(v, Inf)
  y = vec_iterate(Counted)
  n, r = zeros_made(() -> prox!(y, IND, y, 1.0))
  @test n == 0 && r === y && sameall(Vector{Float64}(Vector(y)), refproject(Vector(xr)))
  y = vec_iterate(Counted)
  n, v = zeros_made(() -> value_and_prox!(y, IND, y, 1.0))
  @test n == 0 && same(v, 0.0)
  out = SparseIterate(Counted, p)
  n, r = zeros_made(() -> prox!(out, IND, x, 1.0))
  @test n == 0 && sort(collect(activecoordinates(out))) == [1, 500]
  # a stored NaN
  xn = vec_iterate(Counted); xn[3] = Counted(NaN)
  n, v = zeros_made(() -> IND(xn))
  @test n == 0 && isnan(v)
  # a symmetric iterate
  q = 100
  sdata = [(1, 1, 1.0), (2, 1, -4.0), (3, 3, 0.0), (50, 10, 2.5)]
  mk(T) = fill_matrix!(SymmetricSparseIterate(T, q), sdata)
  S = mk(Counted)
  n, v = zeros_made(() -> IND(S))
  @test n == 0 && same(v, Inf)
  n, r = zeros_made(() -> prox!(S, IND, S, 1.0))
  @test n == 0 && r === S
  n, v = zeros_made(() -> IND(S))
  @test n == 0 && same(v, 0.0)
  S = mk(Counted)
  n, v = zeros_made(() -> value_and_prox!(S, IND, S, 1.0))
  @test n == 0 && same(v, 0.0)
end

@testset "BlockIterate" begin
  for T in TYPES
    groups = [[1, 3], [2, 4], [5, 6, 7], [8, 9, 10]]
    b = BlockIterate(T[1, -2, 3, 0.5, 0, 0, 0, 2, 0, -1], groups)
    γ = T(0.5)
    @test sort(collect(activecoordinates(b))) == [1, 2, 4]
    @test same(IND(b), T(Inf))
    yb = prox(IND, b, γ)
    @test sameall(Vector(yb), refproject(Vector(b)))
    vb = copy(b)
    @test same(value_and_prox!(vb, IND, b, γ), zero(T)) && same(IND(vb), zero(T)) && sameall(Vector(vb), Vector(yb))
    ib = copy(b)
    @test prox!(ib, IND, ib, γ) === ib && sameall(Vector(ib), Vector(yb))
    @test same(IND(BlockIterate(T[1, 2, 3, 0.5, 0, 0, 0, 2, 0, 1], groups)), zero(T))
    bn = BlockIterate(T[1, NaN, 3, 0.5, 0, 0, 0, 2, 0, 1], groups)
    @test same(IND(bn), T(NaN))
    vn = copy(bn)
    @test same(value_and_prox!(vn, IND, bn, γ), T(NaN)) && same(IND(vn), T(NaN))
    @test sameall(Vector(prox(IND, b, fill(-one(T), 10))), refproject(Vector(b)))
    # an iterate with no active group is feasible
    @test same(IND(BlockIterate(T, 10, groups)), zero(T))
  end
end

@testset "SymmetricSparseIterate" begin
  for T in TYPES
    p = 4
    S = SymmetricSparseIterate(T, p)
    S[1, 1] = 2; S[3, 1] = -1.5; S[4, 2] = T(0.2); S[4, 4] = -3
    M = Matrix(S)
    γ = T(0.5)
    @test PB.check_symmetric_parameters(IND, S) === true
    @test PB.check_symmetric_parameters(Scaled(IND, 2), S) === true
    @test same(IND(S), refvalue(M)) && same(IND(S), T(Inf))
    YS = prox(IND, S, γ)
    @test YS isa SymmetricSparseIterate{T} && sameall(Matrix(YS), refproject(M))
    @test nactive(YS) == 2                                                       # the two positive entries
    @test sameall(prox!(zeros(T, p, p), IND, S, γ), refproject(M))             # a dense output from a symmetric input
    ZS = copy(S)
    @test same(value_and_prox!(ZS, IND, S, γ), zero(T)) && same(IND(ZS), zero(T)) && sameall(Matrix(ZS), Matrix(YS))
    IS = copy(S)
    @test prox!(IS, IND, IS, γ) === IS && sameall(Matrix(IS), refproject(M)) && nactive(IS) == nactive(S)
    # a stored NaN
    SN = copy(S); SN[2, 2] = T(NaN)
    @test same(IND(SN), T(NaN))
    ZN = copy(SN)
    @test same(value_and_prox!(ZN, IND, SN, γ), T(NaN)) && same(IND(ZN), T(NaN)) && isnan(ZN[2, 2])
    # a matrix that is not stored symmetrically: a dense symmetric matrix
    D = SymmetricSparseIterate(T[1 2; 2 3])
    @test same(IND(D), zero(T))
    # array steps with a symmetric output are unsupported, as for the other separable functions of the package
    @test_throws ArgumentError prox!(S, IND, S, ones(T, p, p))
  end
end

@testset "validation before writes" begin
  for T in TYPES
    x = T[1, -2, 3]
    # a size mismatch
    y = fill(T(9), 2)
    @test_throws DimensionMismatch prox!(y, IND, x, one(T))
    @test_throws DimensionMismatch value_and_prox!(y, IND, x, one(T))
    @test y == fill(T(9), 2)
    y3 = fill(T(9), 3)
    @test_throws DimensionMismatch prox!(y3, IND, x, ones(T, 2))
    @test y3 == fill(T(9), 3)
    # complex entries
    yc = fill(complex(T(9)), 3)
    @test_throws ArgumentError prox!(yc, IND, x, one(T))
    @test_throws ArgumentError value_and_prox!(yc, IND, x, one(T))
    @test yc == fill(complex(T(9)), 3)
    xc = complex.(x)
    @test_throws ArgumentError IND(xc)
    @test_throws ArgumentError prox!(y3, IND, xc, one(T))
    @test_throws ArgumentError prox(IND, xc, one(T))
    @test y3 == fill(T(9), 3)
    # axes that do not start at one: the input, the output, the array of steps and the screen
    xo = OffsetVector(copy(x), 0:2)
    yo = OffsetVector(fill(T(9), 3), 0:2)
    @test_throws ArgumentError IND(xo)
    @test_throws ArgumentError prox(IND, xo, one(T))
    @test_throws ArgumentError prox!(y3, IND, xo, one(T))
    @test_throws ArgumentError value_and_prox!(y3, IND, xo, one(T))
    @test_throws ArgumentError prox!(yo, IND, x, one(T))
    @test_throws ArgumentError value_and_prox!(yo, IND, x, one(T))
    @test_throws ArgumentError prox!(y3, IND, x, OffsetVector(ones(T, 3), 0:2))
    @test_throws ArgumentError kkt_violation(IND, xo, 1)
    @test y3 == fill(T(9), 3) && parent(yo) == fill(T(9), 3)
  end
end

@testset "fused value at the stored output" begin
  for T in TYPES
    # the kinds of entry of specials, alone and with the others, with scalar steps and with an array of steps
    arrays = Any[specials(T), nonan(T), T[NaN, -1], T[-1, NaN], T[NaN], T[-1], T[1], T[0, -0.0], T[Inf, NaN, -Inf], T[]]
    for v in specials(T)
      push!(arrays, T[1, v]); push!(arrays, T[v, 1, -3])
    end
    for x in arrays
      for γ in admitted(T)
        y = similar(x)
        v = value_and_prox!(y, IND, x, γ)
        @test same(v, IND(y)) && same(v, any(isnan, x) ? T(NaN) : zero(T)) && sameall(y, refproject(x))
      end
      steps = T[mod(i, 2) == 0 ? -one(T) : T(Inf) for i in eachindex(x)]
      y = similar(x)
      v = value_and_prox!(y, IND, x, steps)
      @test same(v, IND(y)) && same(v, any(isnan, x) ? T(NaN) : zero(T)) && sameall(y, refproject(x))
      z = copy(x)
      v = value_and_prox!(z, IND, z, one(T))
      @test same(v, IND(z)) && same(v, any(isnan, x) ? T(NaN) : zero(T))
    end
    # the case of the contract
    y = similar(T[NaN, -1])
    @test same(value_and_prox!(y, IND, T[NaN, -1], one(T)), T(NaN)) && sameall(y, T[NaN, 0])
    @test same(value_and_prox!(y, IND, T[1, -1], one(T)), zero(T)) && sameall(y, T[1, 0])
    # a matrix and a long array
    M = reshape(specials(T)[1:12], 3, 4)
    ym = similar(M)
    @test same(value_and_prox!(ym, IND, M, one(T)), zero(T)) && same(IND(ym), zero(T))
    long = ones(T, 1000); long[end] = T(NaN); long[1] = -one(T)
    yl = similar(long)
    @test same(value_and_prox!(yl, IND, long, one(T)), T(NaN)) && same(IND(yl), T(NaN))
  end
end

@testset "Scaled" begin
  @test Scaled(IND, 0) === IND
  @test Scaled(IND, 0.0) === IND
  @test 0 * IND === IND
  @test 0.0 * IND === IND
  @test PB._isindicator(Scaled(IND, 2))
  for T in TYPES
    x = specials(T)
    xn = nonan(T)
    xneg = T[1, -1e-30]
    xfeas = filter(v -> v >= 0, xn)
    chains = (("2", Scaled(IND, 2)), ("1e300", Scaled(IND, 1e300)), ("Inf", Scaled(IND, Inf)),
              ("2f0", Scaled(IND, T(2))),
              ("1e-200 * 1e-200", Scaled(Scaled(IND, 1e-200), 1e-200)),
              ("1e200 * 1e200", Scaled(Scaled(IND, 1e200), 1e200)))
    for (name, s) in chains
      @testset "$name, $T" begin
        for z in (x, xn, xneg, xfeas, T[NaN, -1], T[-1, NaN], T[])
          ref = refproject(z)
          @test isequal(s(z), refvalue(z))
          for γ in (T(0), one(T), T(Inf), 0.5)
            @test sameall(prox(s, z, γ), ref)
            y = similar(z)
            @test sameall(prox!(y, s, z, γ), ref)
            y = similar(z)
            v = value_and_prox!(y, s, z, γ)
            @test isequal(v, s(y)) && sameall(y, ref) && isequal(v, any(isnan, z) ? T(NaN) : zero(T))
          end
          for steps in (fill(one(T), length(z)), fill(T(Inf), length(z)), fill(zero(T), length(z)))
            @test sameall(prox(s, z, steps), ref)
            y = similar(z)
            @test sameall(prox!(y, s, z, steps), ref)
            y = similar(z)
            v = value_and_prox!(y, s, z, steps)
            @test isequal(v, s(y)) && sameall(y, ref) && isequal(v, any(isnan, z) ? T(NaN) : zero(T))
          end
          w = copy(z)
          v = value_and_prox!(w, s, w, one(T))
          @test isequal(v, s(w)) && sameall(w, ref)
        end
        # the value at an infeasible point is Inf under each factor of the list
        @test isequal(s(xneg), T(Inf)) && isequal(s(xfeas), zero(T))
      end
    end
    # an infinite factor at a zero step: the projection, for a scalar step and an array of steps
    sinf = Scaled(IND, Inf)
    @test sameall(prox(sinf, x, zero(T)), refproject(x))
    @test sameall(prox(sinf, x, zeros(T, length(x))), refproject(x))
    y = similar(xn)
    @test isequal(value_and_prox!(y, sinf, xn, zero(T)), zero(T)) && sameall(y, refproject(xn))
    @test isequal(value_and_prox!(y, sinf, xn, zeros(T, length(xn))), zero(T)) && sameall(y, refproject(xn))
    # sparse and symmetric iterates under Scaled
    s = Scaled(IND, 2)
    sp = SparseIterate(T, 5); fill_entries!(sp, [(2, -1.0), (4, 3.0)])
    @test isequal(s(sp), T(Inf)) && sameall(Vector(prox(s, sp, one(T))), refproject(Vector(sp)))
    zp = copy(sp)
    @test isequal(value_and_prox!(zp, s, zp, one(T)), zero(T)) && isequal(s(zp), zero(T))
    S = SymmetricSparseIterate(T[1 -2; -2 3])
    @test isequal(s(S), T(Inf)) && sameall(Matrix(prox(s, S, one(T))), refproject(Matrix(S)))
    zs = copy(S)
    @test isequal(value_and_prox!(zs, s, zs, one(T)), zero(T))
    @test PB.check_symmetric_parameters(s, S) === true
  end
end

@testset "Scaled kkt_violation" begin
  for T in TYPES
    tiny = nextfloat(zero(T))
    for (α, entry) in ((T(2), tiny), (T(0.25), floatmax(T)), (tiny, tiny), (T(Inf), T(0.25)))
      v = T[entry]
      @test kkt_violation(Scaled(IND, α), v, 1) === kkt_violation(IND, v, 1)
    end

    merged = Scaled(Scaled(IND, T(2)), T(3))
    nested = Scaled(Scaled(IND, tiny), tiny)
    inner = Scaled{AbstractFunction,T}(IND, T(2))
    retained = Scaled{typeof(inner),T}(inner, T(2))
    @test merged.f === IND && merged.α === T(6)
    @test nested.f isa Scaled && PB._chained(nested)
    @test retained.f === inner && !PB._chained(retained)
    v = specials(T)
    x = zeros(T, length(v))
    for s in (Scaled(IND, T(1.5)), Scaled(IND, tiny), Scaled(IND, T(Inf)), merged, nested, retained)
      for i in eachindex(v)
        @test same(kkt_violation(s, v, i), kkt_violation(IND, v, i))
      end
      @test_throws BoundsError kkt_violation(s, v, 0)
      @test_throws BoundsError kkt_violation(s, v, length(v) + 1)
      @test_throws ArgumentError kkt_violation(s, complex.(v), 1)
      @test_throws ArgumentError kkt_violation(s, OffsetVector(v, 0:(length(v) - 1)), 1)
      @test_throws MethodError kkt_violation(s, v, x, 1)
      @test_throws MethodError kkt_violation(s, v, x)
    end
    @test_throws ArgumentError Scaled(IND, -one(T))
    @test_throws ArgumentError Scaled(IND, T(NaN))
  end
  @test @inferred(kkt_violation(Scaled(IND, 2.0), Float32[0.25], 1)) === -0.25f0
end

@testset "kkt_violation, restrict, traits" begin
  for T in TYPES
    v = T[-0.5, 0.25]
    @test kkt_violation(IND, v, 1) === T(0.5)
    @test kkt_violation(IND, v, 2) === T(-0.25)
    @test kkt_violation(IND, T[-0.0, 0.0], 1) === T(0.0)
    @test kkt_violation(IND, T[NaN], 1) |> isnan
    @test kkt_violation(IND, [1, -2], 2) === 2
    @test kkt_violation(IND, view(T[9, -1, 9, 4], 2:3), 1) === T(1)
    @test_throws ArgumentError kkt_violation(IND, complex.(v), 1)
    # the forms with a point are not defined for this function
    x = T[0, 1]
    @test_throws MethodError kkt_violation(IND, v, x, 1)
    @test_throws MethodError kkt_violation(IND, v, x)
    b = BlockIterate(T[1, -2], [[1], [2]])
    @test_throws MethodError kkt_violation(IND, v, b, 1)
    @test_throws MethodError kkt_violation(IND, v, b)
    S = SymmetricSparseIterate(T[1 2; 2 0])
    G = T[-1 2; 2 -3]
    @test_throws MethodError kkt_violation(IND, G, S, 1)
    @test_throws MethodError kkt_violation(IND, G, S)
    sp = SparseIterate(T[0, 1])
    @test_throws MethodError kkt_violation(IND, v, sp, 1)
    sc = Scaled(IND, 2)
    @test_throws MethodError kkt_violation(sc, v, x, 1)
    @test_throws MethodError kkt_violation(sc, v, b, 1)
    @test_throws MethodError kkt_violation(sc, G, S, 1)
    @test_throws MethodError kkt_violation(sc, v, x)
  end
  @test restrict(IND, [1, 3]) === IND
  @test restrict(IND, 1:2) === IND
  @test isconvex(IND) && isconvex(IndNonnegative)
  @test iszeropreserving(IND)
  @test isseparable(IND) && isproximable(IND)
  @test !issmooth(IND)
  @test PB._isindicator(IND)
  @test IND isa SeparableFunction && IndNonnegative() === IND
  @test PB.entry_parameters(IND) == ()
  @test_throws ArgumentError lipschitz(IND)
end

@testset "inference" begin
  for T in TYPES
    x = T[1, -2, 3, 0]
    y = similar(x)
    g = T[-0.5, 0.25]
    @test @inferred(IND(x)) isa T
    @test @inferred(prox!(y, IND, x, one(T))) === y
    @test @inferred(value_and_prox!(y, IND, x, one(T))) isa T
    @test @inferred(kkt_violation(IND, g, 1)) isa T
    @test @inferred(value_entry(IND, x[1], 1)) isa T
    @test @inferred(prox_block(IND, x[1], 1, one(T))) isa T
    sp = SparseIterate(T[0, -1, 2])
    @test @inferred(IND(sp)) isa T
    @test @inferred(value_and_prox!(sp, IND, sp, one(T))) isa T
  end
end

@testset "allocations" begin
  for T in TYPES
    x = T[1, -2, 3, 0, NaN, -Inf, 5, -0.0]
    y = similar(x)
    one_ = one(T)
    @test allocs(IND, x) == 0
    @test allocs(prox!, y, IND, x, one_) == 0
    @test allocs(value_and_prox!, y, IND, x, one_) == 0
    @test allocs(prox!, y, IND, x, ones(T, 8)) == 0
    @test allocs(value_and_prox!, y, IND, x, ones(T, 8)) == 0
    xl = randn(Xoshiro(948), T, 4096); yl = similar(xl)
    @test allocs(IND, xl) == 0
    @test allocs(prox!, yl, IND, xl, one_) == 0
    @test allocs(value_and_prox!, yl, IND, xl, one_) == 0
    sp = SparseIterate(T, 100); fill_entries!(sp, [(3, -1.0), (50, 2.0), (99, -3.0)])
    @test allocs(IND, sp) == 0
    @test allocs(prox!, sp, IND, sp, one_) == 0
    @test allocs(value_and_prox!, sp, IND, sp, one_) == 0
  end
end

end
