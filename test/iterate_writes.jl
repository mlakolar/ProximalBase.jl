module IterateWritesTest

# One working-set rule for writes into a SparseIterate or an AtomIterate (#424): every
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
mkA() = (a = AtomIterate(4, [1:2, 3:4]); a[4] = 7.0; a)
srcA() = (y = AtomIterate(4, [1:2, 3:4]); y[1] = 3.0; y)

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
ids(x::AtomIterate, entries) = unique(k for i in entries for k in 1:ncoordinates(x) if i in atomindices(x, k))

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
  @testset "AtomIterate: $name" for (name, f, isbc) in CALLS
    check_value_write(mkA, srcA, name, f, isbc)
  end
  @testset "AtomIterate: atom(a, 1) .= [3, 0]" begin
    a = mkA()
    atom(a, 1) .= [3.0, 0]
    @test Vector(a) == [3.0, 0, 0, 7] && ws(a) == [2, 1]
    PT.test_iterate(a)
  end

  @testset "x .= y does not import an active zero group" begin
    x = AtomIterate(6, 3); x[6] = 7.0
    y = AtomIterate(6, 3); y[1] = 3.0; activate!(y, 2)
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
        x isa SparseIterate ? (x[k] = 0) : (atom(x, k) .= 0)
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
      () -> ((x = AtomIterate(6, 3); x[6] = 1.0; x), AtomIterate(4, 2)),
      () -> ((x = SparseIterate(6); x[6] = 1.0; x), (a = AtomIterate(4, 2); a[1] = 2.0; a)),
      () -> ((x = SparseIterate(4); x[4] = 1.0; x), (a = AtomIterate(6, 3); a[1] = 2.0; a)),
      () -> ((x = AtomIterate(6, 3); x[6] = 1.0; x), (s = SparseIterate(4); s[1] = 2.0; s)),
      () -> ((x = AtomIterate(4, 2); x[4] = 1.0; x), (s = SparseIterate(6); s[1] = 2.0; s)),
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

  @testset "a SparseIterate and an AtomIterate of the same length" begin
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

  @testset "AtomIterates of different layouts" begin
    for cp! in (copyto!, copy!)
      d = AtomIterate(4, [[1, 3], [2, 4]]); d[2] = 7.0
      src = AtomIterate(4, [1:2, 3:4]); src[3] = 5.0; activate!(src, 1)
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
      a = mkA(); ya = AtomIterate(4, [1:2, 3:4]); ya[1] = 3.0; activate!(ya, 2)
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

  @testset "AtomIterate" begin
    mk() = (a = AtomIterate(8, 4); a[7] = 1.0; a[3] = 1.0; a[1] = 1.0; a)
    @test ws(mk()) == [4, 2, 1]
    src = zeros(8); src[[3, 5]] = [1.0, 2.0]
    for replace! in (cheap_replace!, staged_replace!)
      a = replace!(mk(), src)
      @test Vector(a) == src && ws(a) == [2, 3]
    end
    @test ws(dropzeros!(fill!(mk(), 0))) == Int[]
    # groups met in the order of their first nonzero entry, not in group order
    g = AtomIterate(4, [[2, 4], [1, 3]]); g[2] = 5.0
    @test ws(cheap_replace!(g, [1.0, 1, 0, 0])) == [2, 1]
    @test ws(AtomIterate([1.0, 1, 0, 0], [[2, 4], [1, 3]])) == [1, 2]
    # the strided multi-response layout of a 3 × 2 coefficient matrix
    groups = [[1, 4], [2, 5], [3, 6]]
    B = [0.0 1; 1 0; 0 0]
    h = AtomIterate(6, groups); h[3] = 1.0
    @test ws(cheap_replace!(h, vec(B))) == [2, 1]
    @test ws(AtomIterate(vec(B), groups)) == [1, 2]
  end

  @testset "same-kind source" begin
    y = SparseIterate(4); y[4] = 2.0; activate!(y, 1)
    x4 = SparseIterate(4); x4[2] = 9.0
    staged_replace!(x4, y)
    @test Vector(x4) == [0.0, 0, 0, 2] && ws(x4) == [4]
    x4 = SparseIterate(4); x4[2] = 9.0
    @test ws(cheap_replace!(x4, y)) == [4, 1]              # the shorthand state-copies the zero
    ya = AtomIterate(4, [1:2, 3:4]); ya[3] = 2.0; activate!(ya, 1)
    xa = AtomIterate(4, [1:2, 3:4]); xa[1] = 9.0
    staged_replace!(xa, ya)
    @test Vector(xa) == [0.0, 0, 2, 0] && ws(xa) == [2]
    xa = AtomIterate(4, [1:2, 3:4]); xa[1] = 9.0
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
  a = AtomIterate(n, 100); a[4] = 7.0
  ay = AtomIterate(n, 100); ay[1] = 3.0
  for z in (x, xm, a)
    @test PT.allocations(zero_fill!, z) == 0
  end
  for (z, s) in ((x, y), (x, d), (xm, ym), (xm, dm), (a, ay), (a, d))
    @test PT.allocations(assign!, z, s) == 0
  end
end

end
