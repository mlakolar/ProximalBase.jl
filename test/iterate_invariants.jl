module IterateInvariantsTest

# Invariants that every iterate keeps, checked the same way on SparseIterate (vector and
# matrix), SymmetricSparseIterate and AtomIterate after each operation of one sequence: the
# active list, the reverse map and the values agree, and inactive coordinates are zero; a zero
# write neither activates nor deactivates; activate! stores an explicit zero and dropzeros!
# removes it without changing an entry; a copy is independent; copyto! replaces the whole state
# rather than merging; a write or copy whose conversion throws leaves the destination
# unchanged. Quick runs it for every change; the iterate files' own tests go into the details.

using Test
using ProximalBase
using SparseArrays: dropzeros!

# How to read and write coordinate k of each kind (for an AtomIterate, the first entry of
# group k).
const KINDS = (
  (name = "SparseIterate vector", make = T -> SparseIterate(T, 7),
   get = (x, k) -> x[k], set! = (x, k, v) -> (x[k] = v)),
  (name = "SparseIterate matrix", make = T -> SparseIterate(T, 3, 3),
   get = (x, k) -> x[k], set! = (x, k, v) -> (x[k] = v)),
  (name = "SymmetricSparseIterate", make = T -> SymmetricSparseIterate(T, 3),
   get = (x, k) -> coordinates(x)[k], set! = (x, k, v) -> (coordinates(x)[k] = v)),
  (name = "AtomIterate", make = T -> AtomIterate(T, 7, [1:2, [3, 6], 4:5, [7]]),
   get = (x, k) -> atom(x, k)[1], set! = (x, k, v) -> (atom(x, k)[1] = v)),
)

iszerocoordinate(x::AtomIterate, k) = all(iszero, atom(x, k))
iszerocoordinate(x, k) = iszero(coordinates(x)[k])

# The active list, the reverse map and the values agree, and inactive coordinates are zero.
function consistent(x)
  ids = collect(activecoordinates(x))
  n = ncoordinates(x)
  length(ids) == nactive(x) && allunique(ids) && all(k -> 1 <= k <= n, ids) || return false
  count(k -> isactive(x, k), 1:n) == nactive(x) && all(k -> isactive(x, k), ids) || return false
  all(k -> isactive(x, k) || iszerocoordinate(x, k), 1:n) || return false
  x isa AtomIterate && return true
  return activevalues(x) == [coordinates(x)[k] for k in ids]
end

snapshot(x) = (copy(activecoordinates(x)), copy(Array(x)))

@testset "iterate invariants: $(kind.name), $T" for kind in KINDS, T in (Float64, Int)
  x = kind.make(T)
  n = ncoordinates(x)
  @test nactive(x) == 0 && consistent(x)
  # a zero write to an inactive coordinate stores nothing
  kind.set!(x, 2, 0)
  @test nactive(x) == 0 && consistent(x)
  # nonzero writes activate, in the order of the writes
  kind.set!(x, 2, 3)
  kind.set!(x, n, 5)
  @test collect(activecoordinates(x)) == [2, n] && consistent(x)
  # a zero write to an active coordinate keeps it active, as a stored zero
  kind.set!(x, 2, 0)
  @test isactive(x, 2) && nactive(x) == 2 && consistent(x)
  # activate! stores an explicit zero, once
  activate!(x, 1)
  activate!(x, 1)
  @test isactive(x, 1) && nactive(x) == 3 && kind.get(x, 1) == 0 && consistent(x)
  # dropzeros! removes the coordinates that are zero and changes no entry
  before = copy(Array(x))
  dropzeros!(x)
  @test collect(activecoordinates(x)) == [n] && Array(x) == before && consistent(x)
  # a copy has the same state and is independent of the original
  y = copy(x)
  @test Array(y) == Array(x) && collect(activecoordinates(y)) == collect(activecoordinates(x))
  kind.set!(y, 1, 7)
  @test kind.get(x, 1) == 0 && !isactive(x, 1) && consistent(x) && consistent(y)
  # copyto! replaces the whole state: a coordinate active only in the destination becomes
  # inactive and zero
  z = kind.make(T)
  kind.set!(z, 3, 4)
  copyto!(x, z)
  @test collect(activecoordinates(x)) == [3] && Array(x) == Array(z) && !isactive(x, n) && consistent(x)
  # a write or a copy whose conversion throws leaves the destination unchanged
  if T == Int
    snap = snapshot(x)
    @test_throws InexactError kind.set!(x, 1, 0.5)
    @test snapshot(x) == snap && consistent(x)
    w = kind.make(Float64)
    kind.set!(w, 1, 0.5)
    @test_throws InexactError copyto!(x, w)
    @test snapshot(x) == snap && consistent(x)
  end
end

@testset "iterate invariants: one SymmetricSparseIterate coordinate per pair" begin
  x = SymmetricSparseIterate(Float64, 3)
  @test ncoordinates(x) == 6 == length(coordinates(x))
  x[3, 1] = 2
  @test x[1, 3] == 2 && nactive(x) == 1 && consistent(x)
  x[1, 3] = 5
  @test x[3, 1] == 5 && nactive(x) == 1 && count(!iszero, Matrix(x)) == 2 && consistent(x)
  # the coordinate of the pair is one entry of the lower triangle
  k = only(activecoordinates(x))
  @test coordinates(x)[k] == 5
  coordinates(x)[k] = 0
  @test isactive(x, k) && iszero(Matrix(x)) && consistent(x)
  dropzeros!(x)
  @test nactive(x) == 0 && consistent(x)
end

end
