module IterateWritesTest

# One working-set rule for writes into a SparseIterate or a BlockIterate (#424): every
# write through the array interface is a value write, which activates the coordinates it
# makes nonzero and removes none; a broadcast is the explicit loop; copyto! and copy! are
# a state copy only between iterates of the same kind.

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays

const PT = ProximalBase.Testing

ws(x) = collect(activecoordinates(x))
snapshot(x) = (Array(x), ws(x))

# the explicit loop over a materialized right-hand side
function loop!(z, r)
  for i in eachindex(z)
    z[i] = r[i]
  end
  z
end

const v = [3.0, 0, 0, 0]
const A = [0.0 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 0]
const b = [0.0, 1, 0, 0]
const M = [0.0 0; 1 1; 0 0; 0 0]
const F = lu(Matrix(1.0I, 4, 4))

# old working set [4, 2] (a nonzero at 4, an explicit zero at 2)
mkS() = (x = SparseIterate(4); x[4] = 7.0; activate!(x, 2); x)
# a source holding 3 at 1 and an explicit zero at 3
srcS() = (y = SparseIterate(4); y[1] = 3.0; activate!(y, 3); y)
# old working set [2] (group 2 holds 7 at entry 4)
mkA() = (a = BlockIterate(4, [1:2, 3:4]); a[4] = 7.0; a)
srcA() = (y = BlockIterate(4, [1:2, 3:4]); y[1] = 3.0; y)

# name, call (x, y), whether it is a broadcast into x itself
const CALLS = [
  ("x[1] = 3", (x, y) -> (x[1] = 3), false),
  ("x .= v", (x, y) -> (x .= v), true),
  ("x .= 0", (x, y) -> (x .= 0), true),
  ("x .= 0 .* x", (x, y) -> (x .= 0 .* x), true),
  ("x .= 1 .* v", (x, y) -> (x .= 1 .* v), true),
  ("x .= y", (x, y) -> (x .= y), true),
  ("fill!(x, 0)", (x, y) -> fill!(x, 0), false),
  ("fill!(x, 2)", (x, y) -> fill!(x, 2), false),
  ("copyto!(x, v)", (x, y) -> copyto!(x, v), false),
  ("copy!(x, v)", (x, y) -> copy!(x, v), false),
  ("x[:] = v", (x, y) -> (x[:] = v), false),
  ("copyto!(x, 1, v, 1, 4)", (x, y) -> copyto!(x, 1, v, 1, 4), false),
  ("view(x, 1:2) .= [3, 0]", (x, y) -> (view(x, 1:2) .= [3.0, 0]), false),
  ("vec(x) .= v", (x, y) -> (vec(x) .= v), true),
  ("rmul!(x, 0)", (x, y) -> rmul!(x, 0), false),
  ("x .*= 0", (x, y) -> (x .*= 0), true),
  ("map!(identity, x, v)", (x, y) -> map!(identity, x, v), false),
  ("ldiv!(x, F, v)", (x, y) -> ldiv!(x, F, v), false),
  ("permute!(x, p)", (x, y) -> permute!(x, [4, 1, 2, 3]), true),
  ("mul!(x, A, b)", (x, y) -> mul!(x, A, b), false),
  ("sum!(x, M)", (x, y) -> sum!(x, M), false),
  ("x .= sparse vector", (x, y) -> (x .= sparsevec([1, 3], [3.0, 0.0], 4)), true),
  ("mul!(x, sparse(A), b)", (x, y) -> mul!(x, sparse(A), b), false),
]

# the coordinates of x made nonzero, as IDs of x's coordinates (entries or groups)
ids(x::SparseIterate, entries) = entries
ids(x::BlockIterate, entries) = unique(k for i in entries for k in 1:ncoordinates(x) if i in blockindices(x, k))

function check_value_write(mk, src, name, f, isbroadcast)
  x0 = mk()
  old = ws(x0)
  x = mk(); y = src()
  r = Vector(x); f(r, Vector(y))
  f(x, y)
  @test Vector(x) == r
  @test ws(x)[1:length(old)] == old                      # the old slots stay, in order
  @test Set(ws(x)) == Set(old) ∪ Set(ids(x, findall(!iszero, r)))
  PT.test_iterate(x)
  if isbroadcast                                          # the loop over the unaliased rhs
    z = loop!(copy(x0), r)
    @test Vector(z) == Vector(x) && ws(z) == ws(x)
  end
end

@testset "value writes never remove (FT05)" begin
  @testset "SparseIterate: $name" for (name, f, isbc) in CALLS
    check_value_write(mkS, srcS, name, f, isbc)
  end
  @testset "BlockIterate: $name" for (name, f, isbc) in CALLS
    check_value_write(mkA, srcA, name, f, isbc)
  end
  @testset "BlockIterate: blockview(a, 1) .= [3, 0]" begin
    a = mkA()
    blockview(a, 1) .= [3.0, 0]
    @test Vector(a) == [3.0, 0, 0, 7] && ws(a) == [2, 1]
    PT.test_iterate(a)
  end

  @testset "x .= y does not import an active zero group" begin
    x = BlockIterate(6, 3); x[6] = 7.0
    y = BlockIterate(6, 3); y[1] = 3.0; activate!(y, 2)
    x .= y
    @test Vector(x) == [3.0, 0, 0, 0, 0, 0] && ws(x) == [3, 1]
    x = mkS()
    x .= srcS()
    @test ws(x) == [4, 2, 1]                              # the explicit zero at 3 does not join
  end

  @testset "x .= x changes nothing" begin
    for x in (mkS(), mkA())
      s = snapshot(x)
      x .= x
      @test snapshot(x) == s
    end
  end

  @testset "a sweep over the active coordinates keeps the handle" begin
    for x in (mkS(), mkA())
      h = activecoordinates(x)
      before = copy(h)
      for k in h
        x isa SparseIterate ? (x[k] = 0) : (blockview(x, k) .= 0)
      end
      @test h == before && ws(x) == before
      fill!(x, 0)
      @test h == before && ws(x) == before && all(iszero, Array(x))
    end
  end

  @testset "ldiv! with a same-kind source is a state copy" begin
    x = mkS(); y = srcS()
    ldiv!(x, F, y)
    @test Vector(x) == Vector(y) && ws(x) == ws(y)
    a = mkA(); ya = srcA()
    ldiv!(a, F, ya)
    @test Vector(a) == Vector(ya) && ws(a) == ws(ya) == [1]
  end

  @testset "a zero fill! writes the converted zero" begin
    for z in (-0.0, -0f0)
      x = mkS()
      fill!(x, z)
      @test Vector(x) == zeros(4) && ws(x) == [4, 2] && all(signbit, activevalues(x))
      a = mkA()
      fill!(a, z)
      @test Vector(a) == zeros(4) && ws(a) == [2] && signbit(a[3]) && signbit(a[4])
      @test !signbit(a[1]) && !signbit(a[2])              # inactive entries are not written
      x = mkS()
      x .= z
      @test ws(x) == [4, 2] && all(signbit, activevalues(x))
    end
    # converted first: nonzero as a Float64, zero as a Float32
    x = SparseIterate(Float32, 4); x[4] = 7; activate!(x, 2)
    fill!(x, 1e-50)
    @test Vector(x) == zeros(Float32, 4) && ws(x) == [4, 2]
    fill!(x, -1e-50)
    @test all(signbit, activevalues(x)) && ws(x) == [4, 2]
  end

  @testset "a complex value is rejected before the zero write" begin
    for mk in (mkS, mkA), z in (0im, 0.0 + 0im, 1 + 0im)
      x = mk()
      s = snapshot(x)
      @test_throws ArgumentError fill!(x, z)
      @test snapshot(x) == s
      @test_throws ArgumentError (x .= z)
      @test snapshot(x) == s
    end
  end

  @testset "matrices: the loop in linear order" begin
    mk() = (x = SparseIterate(3, 2); x[3, 2] = 7.0; activate!(x, 2); x)
    ym = SparseIterate(3, 2); ym[1, 1] = 3.0; activate!(ym, 5)
    col = [1.0, 0, 2]
    row = [0.0, 5]
    Ms = sparse([1, 3, 2], [2, 1, 1], [1.0, 4.0, 0.0], 3, 2)   # a stored zero at (2, 1)
    @test nnz(Ms) == 3
    calls = [
      ("xm .= ym", (x, y) -> (x .= y)),
      ("xm .= column", (x, y) -> (x .= col)),
      ("xm .= row'", (x, y) -> (x .= row')),
      ("xm .= 0", (x, y) -> (x .= 0)),
      ("xm .= sparse(M)", (x, y) -> (x .= Ms)),
      ("fill!(xm, 0)", (x, y) -> fill!(x, 0)),
    ]
    @testset "$name" for (name, f) in calls
      x0 = mk()
      x = mk()
      r = Matrix(x); f(r, Matrix(ym))
      f(x, ym)
      @test Matrix(x) == r
      @test ws(x)[1:2] == [6, 2]
      @test Set(ws(x)) == Set([6, 2]) ∪ Set(findall(!iszero, vec(r)))
      z = loop!(copy(x0), r)
      @test Matrix(z) == Matrix(x) && ws(z) == ws(x)
      PT.test_iterate(x)
    end
    x = mk()
    x .= col
    @test ws(x) == [6, 2, 1, 3, 4]
    x = mk()
    x .= row'
    @test ws(x) == [6, 2, 4, 5]
  end
end

@testset "imports and lengths (FT07)" begin
  @testset "another shape of the same length is a value write" begin
    for cp! in (copyto!, copy!)
      x = SparseIterate(2, 2); x[1, 1] = 5.0
      s = SparseIterate(4); s[4] = 1.0; activate!(s, 2)
      @test cp!(x, s) === x
      @test Matrix(x) == [0.0 0; 0 1] && ws(x) == [1, 4]
      PT.test_iterate(x)
      # and back, against the loop
      t = SparseIterate(4); t[3] = 2.0
      m = SparseIterate(2, 2); m[2, 1] = 6.0; activate!(m, 4)
      z = loop!(copy(t), m)
      cp!(t, m)
      @test Vector(t) == Vector(z) == [0.0, 6, 0, 0] && ws(t) == ws(z) == [3, 2]
    end
  end

  @testset "different lengths throw before writing" begin
    pairs = [
      () -> ((x = SparseIterate(6); x[6] = 1.0; x), SparseIterate(4)),
      () -> ((x = BlockIterate(6, 3); x[6] = 1.0; x), BlockIterate(4, 2)),
      () -> ((x = SparseIterate(6); x[6] = 1.0; x), (a = BlockIterate(4, 2); a[1] = 2.0; a)),
      () -> ((x = SparseIterate(4); x[4] = 1.0; x), (a = BlockIterate(6, 3); a[1] = 2.0; a)),
      () -> ((x = BlockIterate(6, 3); x[6] = 1.0; x), (s = SparseIterate(4); s[1] = 2.0; s)),
      () -> ((x = BlockIterate(4, 2); x[4] = 1.0; x), (s = SparseIterate(6); s[1] = 2.0; s)),
      () -> ((x = SparseIterate(2, 2); x[4] = 1.0; x), (s = SparseIterate(6); s[1] = 2.0; s)),
    ]
    for mk in pairs, cp! in (copyto!, copy!)
      x, y = mk()
      s = snapshot(x)
      @test_throws DimensionMismatch cp!(x, y)
      @test snapshot(x) == s
      PT.test_iterate(x)
    end
  end

  @testset "a SparseIterate and a BlockIterate of the same length" begin
    for cp! in (copyto!, copy!)
      x = mkS(); y = srcA()
      z = loop!(copy(x), y)
      cp!(x, y)
      @test Vector(x) == Vector(z) == v && ws(x) == ws(z) == [4, 2, 1]
      a = mkA(); s = srcS()
      z = loop!(copy(a), s)
      cp!(a, s)
      @test Vector(a) == Vector(z) == v && ws(a) == ws(z) == [2, 1]
      PT.test_iterate(x); PT.test_iterate(a)
    end
  end

  @testset "BlockIterates of different layouts" begin
    for cp! in (copyto!, copy!)
      d = BlockIterate(4, [[1, 3], [2, 4]]); d[2] = 7.0
      src = BlockIterate(4, [1:2, 3:4]); src[3] = 5.0; activate!(src, 1)
      cp!(d, src)
      @test Vector(d) == [0.0, 0, 5, 0] && ws(d) == [2, 1]
      PT.test_iterate(d)
    end
  end

  @testset "same kind is still a state copy" begin
    for cp! in (copyto!, copy!)
      x = mkS(); y = srcS()
      cp!(x, y)
      @test Vector(x) == Vector(y) && ws(x) == ws(y) == [1, 3]
      a = mkA(); ya = BlockIterate(4, [1:2, 3:4]); ya[1] = 3.0; activate!(ya, 2)
      cp!(a, ya)
      @test Vector(a) == Vector(ya) && ws(a) == ws(ya) == [1, 2]
    end
  end
end

# The replace recipes of the contract, with their checks first.
function check_replace_source(x, src)
  length(src) == length(x) ||
    throw(DimensionMismatch("source has length $(length(src)), destination $(length(x))"))
  eltype(src) <: Real || throw(ArgumentError("the source must have a real element type"))
end
cheap_replace!(x, src) = (check_replace_source(x, src); copyto!(dropzeros!(fill!(x, 0)), src))
function staged_replace!(x, src)
  check_replace_source(x, src)
  t = similar(x)
  for (i, j) in zip(eachindex(t), eachindex(src))
    t[i] = src[j]
  end
  copyto!(x, t)
end

@testset "replace and clear (FT08, array iterates)" begin
  @testset "SparseIterate" begin
    mk() = (x = SparseIterate(10); x[5] = 1.0; x[2] = 2.0; x[9] = 3.0; x)
    @test ws(mk()) == [5, 2, 9]
    src = zeros(10); src[[2, 3, 9]] = [1.0, 2.0, 3.0]
    for replace! in (cheap_replace!, staged_replace!)
      x = replace!(mk(), src)
      @test Vector(x) == src && ws(x) == [2, 3, 9]
    end
    @test ws(dropzeros!(fill!(mk(), 0))) == Int[]
    @test ws(dropzeros!(copyto!(mk(), src))) == [3, 2, 9]   # the shorter form, another order
  end

  @testset "BlockIterate" begin
    mk() = (a = BlockIterate(8, 4); a[7] = 1.0; a[3] = 1.0; a[1] = 1.0; a)
    @test ws(mk()) == [4, 2, 1]
    src = zeros(8); src[[3, 5]] = [1.0, 2.0]
    for replace! in (cheap_replace!, staged_replace!)
      a = replace!(mk(), src)
      @test Vector(a) == src && ws(a) == [2, 3]
    end
    @test ws(dropzeros!(fill!(mk(), 0))) == Int[]
    # groups met in the order of their first nonzero entry, not in group order
    g = BlockIterate(4, [[2, 4], [1, 3]]); g[2] = 5.0
    @test ws(cheap_replace!(g, [1.0, 1, 0, 0])) == [2, 1]
    @test ws(BlockIterate([1.0, 1, 0, 0], [[2, 4], [1, 3]])) == [1, 2]
    # the strided multi-response layout of a 3 × 2 coefficient matrix
    groups = [[1, 4], [2, 5], [3, 6]]
    B = [0.0 1; 1 0; 0 0]
    h = BlockIterate(6, groups); h[3] = 1.0
    @test ws(cheap_replace!(h, vec(B))) == [2, 1]
    @test ws(BlockIterate(vec(B), groups)) == [1, 2]
  end

  @testset "same-kind source" begin
    y = SparseIterate(4); y[4] = 2.0; activate!(y, 1)
    x4 = SparseIterate(4); x4[2] = 9.0
    staged_replace!(x4, y)
    @test Vector(x4) == [0.0, 0, 0, 2] && ws(x4) == [4]
    x4 = SparseIterate(4); x4[2] = 9.0
    @test ws(cheap_replace!(x4, y)) == [4, 1]              # the shorthand state-copies the zero
    ya = BlockIterate(4, [1:2, 3:4]); ya[3] = 2.0; activate!(ya, 1)
    xa = BlockIterate(4, [1:2, 3:4]); xa[1] = 9.0
    staged_replace!(xa, ya)
    @test Vector(xa) == [0.0, 0, 2, 0] && ws(xa) == [2]
    xa = BlockIterate(4, [1:2, 3:4]); xa[1] = 9.0
    @test ws(cheap_replace!(xa, ya)) == [2, 1]
  end

  @testset "overlapping source" begin
    w = SparseIterate([1.0, 2.0])
    staged_replace!(w, view(w, 2:-1:1))
    @test Vector(w) == [2.0, 1.0] && ws(w) == [1, 2]
    w = SparseIterate([1.0, 2.0])
    @test Vector(cheap_replace!(w, view(w, 2:-1:1))) == [0.0, 0.0]
  end

  @testset "fallible conversion and length" begin
    xj = SparseIterate(Int, 2); xj[2] = 5
    @test_throws InexactError staged_replace!(xj, [1.0, 1.5])
    @test Vector(xj) == [0, 5] && ws(xj) == [2]
    xj = SparseIterate(Int, 2); xj[2] = 5
    @test_throws InexactError cheap_replace!(xj, [1.0, 1.5])
    @test Vector(xj) == [1, 0] && ws(xj) == [1]            # the shorthand loses the old state
    for replace! in (cheap_replace!, staged_replace!)
      xm = SparseIterate(3); xm[2] = 4.0
      @test_throws DimensionMismatch replace!(xm, [1.0, 2.0])
      @test Vector(xm) == [0.0, 4, 0] && ws(xm) == [2]
    end
  end

  @testset "Cartesian source" begin
    src = view(reshape(1:4, 2, 2), [2, 1], :)
    x5 = SparseIterate(4)
    @test_throws DimensionMismatch eachindex(similar(x5), src)
    staged_replace!(x5, src)
    @test Vector(x5) == [2.0, 1.0, 4.0, 3.0] && ws(x5) == [1, 2, 3, 4]
  end

  @testset "complex-typed source" begin
    for replace! in (cheap_replace!, staged_replace!)
      xc = SparseIterate(4); xc[3] = 9.0
      @test_throws ArgumentError replace!(xc, ComplexF64[1, 2, 3, 4])
      @test Vector(xc) == [0.0, 0, 9, 0] && ws(xc) == [3]
    end
  end
end

@testset "coordinates(x) is an ordinary vector that writes values (FP-7)" begin
  mk(T) = (x = SparseIterate(T, 5); x[2] = T(3); activate!(x, 4); x)
  stored(x) = x.nzval[1:x.nnz]
  @testset "fill! with a zero writes into the stored entries" begin
    for T in (Float64, Float32), zero_value in (0, 0.0, 0im, T(0))
      x = mk(T); c = coordinates(x)
      r = fill!(c, zero_value)
      @test r === c
      @test Array(x) == zeros(T, 5)
      @test ws(x) == [2, 4]
    end
    x = mk(Float64)
    fill!(coordinates(x), -0.0)
    @test all(signbit, stored(x)) && ws(x) == [2, 4]
    @test Array(x) == zeros(5)
    x = mk(Float32)
    fill!(coordinates(x), 1e-50)     # rounds to zero in Float32, as in a plain vector
    @test stored(x) == Float32[0, 0] && ws(x) == [2, 4]
    x = mk(Float64)
    coordinates(x) .= 0
    @test Array(x) == zeros(5) && ws(x) == [2, 4]
    x = SparseIterate(Float64, 2, 3); x[2, 2] = 1.0
    fill!(coordinates(x), 0)
    @test Array(x) == zeros(2, 3) && ws(x) == [4]
    @test isempty(ws(dropzeros!(x)))
  end
  @testset "fill! with a nonzero value" begin
    x = mk(Float64); c = coordinates(x)
    r = fill!(c, 1 + 0im)
    @test r === c && Array(x) == ones(5) && sort(ws(x)) == 1:5
    x = mk(Float64)
    fill!(coordinates(x), NaN)
    @test all(isnan, Array(x)) && sort(ws(x)) == 1:5
    x = SparseIterate(Float64, 2, 3)
    fill!(coordinates(x), 2)
    @test Array(x) == fill(2.0, 2, 3) && sort(ws(x)) == 1:6
  end
  @testset "fill! converts like a plain vector, before writing" begin
    x = mk(Float64); before = snapshot(x)
    @test_throws InexactError fill!(coordinates(x), 2im)
    @test snapshot(x) == before
    xi = SparseIterate(Int, 5); xi[2] = 3; before = snapshot(xi)
    @test_throws InexactError fill!(coordinates(xi), 0.5)
    @test snapshot(xi) == before
    fill!(coordinates(xi), 2.0)
    @test Array(xi) == fill(2, 5)
  end
  @testset "the view as a source and as a destination is a value write" begin
    for (x, y) in ((SparseIterate(Float64, 2, 3), SparseIterate(Float64, 2, 3)),
                   (SparseIterate(6), SparseIterate(6)))
      x[2] = 5.0; activate!(x, 6)
      y[1] = 3.0
      copyto!(x, coordinates(y))
      @test ws(x) == [2, 6, 1] && Array(x)[1] == 3.0 && Array(x)[2] == 0.0
      @test ws(y) == [1]
    end
    v = SparseIterate(4); v[2] = 9.0
    coordinates(v) .= SparseIterate([1.0, 0, 0, 0])
    @test ws(v) == [2, 1] && Vector(v) == [1.0, 0, 0, 0]
    xm = SparseIterate(Float64, 2, 3); xm[2, 1] = 1.0
    ym = SparseIterate(Float64, 2, 3); ym[1, 1] = 3.0
    coordinates(xm) .= vec(ym)
    @test ws(xm) == [2, 1] && Array(xm) == [3.0 0 0; 0 0 0]
    @test_throws DimensionMismatch (coordinates(xm) .= ym)
  end
end

lower_positions(p) = [(r, c) for c in 1:p for r in c:p]

@testset "the coordinates of an owner are an ordinary vector (CRD-4)" begin
  @testset "$T, p = $p" for T in (Float32, Float64), p in (0, 1, 3)
    q = p * (p + 1) ÷ 2
    vals = T.(1:q)
    function reference(c)
      D = zeros(T, p, p)
      for (k, (r, s)) in enumerate(lower_positions(p))
        D[r, s] = D[s, r] = c[k]
      end
      D
    end
    x = SymmetricSparseIterate(T, p)
    cx = coordinates(x)
    @test (cx .= vals) === cx
    @test collect(cx) == vals && nactive(x) == q
    cx .*= 2
    @test collect(cx) == 2 .* vals && cx ≈ 2 .* vals
    @test sum(abs, cx) == sum(2 .* vals)
    # the doubled coordinates are the doubled matrix, both triangles
    @test Matrix(x) == reference(2 .* vals) && matrixview(x) == reference(2 .* vals)
    if q >= 2
      view(cx, 1:2) .= zero(T)                       # a zero written to an active coordinate keeps it active
      @test cx[1] == 0 && cx[2] == 0 && nactive(x) == q
      reshape(cx, q, 1)[1, 1] = T(9)                 # a reshape of the coordinates writes through
      @test cx[1] == 9 && Matrix(x) == reference(collect(cx))
    end
    @test (cx .= vals .- vals) === cx && all(iszero, Matrix(x)) && nactive(x) == q
    dropzeros!(x)
    @test nactive(x) == 0 && all(iszero, Matrix(x))
  end
end

@testset "a matrix SparseIterate has the array writers that a SymmetricSparseIterate does not" begin
  Aa = [1.0 3; 2 4]
  Mx = SparseIterate(zeros(2, 2))
  @test (view(Mx, :, :) .= Aa) == Aa
  @test setindex!(Mx, [5.0, 6.0], 1, :) === Mx
  @test Array(Mx) == [5 6; 2 4]
  @test (vec(Mx) .= [1.0, 2.0, 3.0, 4.0]) == vec(Aa)
  @test Array(Mx) == Aa
  @test transpose!(Mx, Aa) === Mx
  @test Array(Mx) == transpose(Aa)
  @test copyto!(Mx, (1.0, 2.0, 3.0, 4.0)) === Mx
  @test Array(Mx) == Aa
  @test mul!(Mx, [1.0, 2.0, 3.0, 4.0], 2.0) === Mx
  @test Array(Mx) == 2Aa
end

zero_fill!(x) = (fill!(x, 0); nothing)
assign!(x, y) = (x .= y; nothing)

@testset "writes do not allocate" begin
  n = 1000
  x = SparseIterate(n); x[4] = 7.0; activate!(x, 9)
  y = SparseIterate(n); y[1] = 3.0
  d = zeros(n); d[9] = 1.0
  xm = SparseIterate(30, 20); xm[7] = 1.0
  ym = SparseIterate(30, 20); ym[5] = 2.0
  dm = zeros(30, 20); dm[11] = 3.0
  a = BlockIterate(n, 100); a[4] = 7.0
  ay = BlockIterate(n, 100); ay[1] = 3.0
  for z in (x, xm, a)
    @test PT.allocations(zero_fill!, z) == 0
  end
  for (z, s) in ((x, y), (x, d), (xm, ym), (xm, dm), (a, ay), (a, d))
    @test PT.allocations(assign!, z, s) == 0
  end
end

end
