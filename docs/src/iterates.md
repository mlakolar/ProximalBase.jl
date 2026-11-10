# Sparse iterates

ProximalBase provides three iterate types for coordinate descent and active-set
procedures. We separate the choice of storage from the update formulas,
which are derived in [Coordinate descent](@ref).

These are storage and coordinate-space choices for algorithms assembled from
[function operations](operations.md). Read
[Building optimization algorithms](optimization_algorithms.md) first if you
are new to the separation between functions, iterates and solvers.


| Iterate | Array represented | One coordinate |
|:--------|:------------------|:---------------|
| [`SparseIterate`](@ref) | A sparse vector or matrix | One entry |
| [`SymmetricSparseIterate`](@ref) | A sparse symmetric matrix | One lower-triangular entry and its symmetric counterpart |
| [`BlockIterate`](@ref) | A vector partitioned into groups | One group of entries |

## Choosing an iterate

Choose the representation by the **variable your algorithm updates** rather than
by whether the mathematical solution happens to contain zeros.

| Problem / update | Use | Index spaces and behavior |
|:--|:--|:--|
| Entrywise coordinate descent on vectors or general matrices | [`SparseIterate`](sparse_iterate.md) | One coordinate per entry; `x[k]` and `coordinates(x)[k]` address the same value; ordinary array behavior |
| Optimization over real symmetric matrices | [`SymmetricSparseIterate`](symmetric_sparse_iterate.md) | One coordinate per independent lower-triangular variable; `X[i,j]` is symmetric, `X[k]` is intentionally unsupported |
| Group or block coordinate descent on a partitioned vector | [`BlockIterate`](block_iterate.md) | One coordinate per group; `x[j]` still indexes an entry; `blockview(x,k)` accesses the group |

All three keep a **working set** distinct from the mathematical nonzero support.
Construction reserves storage proportional to the number of logical entries or
groups; these types optimize inexpensive activation and active-set traversal,
**not** compressed-sparse memory usage. Choose Julia's
`SparseVector`/`SparseMatrixCSC` when compact storage is the priority.
For matrix functions and Frobenius objectives, do not replace a symmetric
owner with its independent-coordinate vector without accounting for the two
matrix entries represented by an off-diagonal variable.

```@example choose_iterate
using ProximalBase

v = SparseIterate(4)
v[2] = 2.0
@assert ncoordinates(v) == 4 && nactive(v) == 1

X = SymmetricSparseIterate(3)
coordinates(X)[2] = 3.0
@assert X[2, 1] == X[1, 2] == 3.0 && ncoordinates(X) == 6

g = BlockIterate(6, [1:2, 3:4, 5:6])
g[2] = 1.0
@assert ncoordinates(g) == 3 && nactive(g) == 1
@assert collect(blockview(g, 1)) == [0.0, 1.0]
```

## Where to go next

| Goal | Guide |
|:--|:--|
| Pick an iterate representation | [SparseIterate](sparse_iterate.md), [SymmetricSparseIterate](symmetric_sparse_iterate.md), or [BlockIterate](block_iterate.md) |
| Understand active coordinates, stored zeros, and counts | [Tracking an active set](@ref) below |
| Apply one-coordinate proximal steps and KKT screens | [Coordinate oracles](iterate_oracles.md) |
| Work with symmetric matrix parameter arrays | [Symmetric parameters](symmetric_parameters.md) |
| Extend the storage implementations or check 0.6 obligations | [Developer Documentation](developer/index.md) |

The iterate determines the *optimization coordinate*, not the mathematical function being
optimized. The same function may support multiple iterate representations, with different
index spaces and active-set semantics. For quick examples and operation costs, use the
type-specific guides before writing a solver.

## Real-valued inputs and conversion

The entries of an iterate are real. A complex element type, or a complex operand
of an iterate product, raises an `ArgumentError`; see [Precision](@ref). Writing
a complex source or value into a `SparseIterate` or a `BlockIterate` with the
constructor, `copyto!`, `copy!`, a broadcast or `fill!` raises it too, before
anything is written, even when the imaginary part is zero. A broadcast is
checked for its leaves and for its result: a result known to be complex from the
types is rejected, and a result of another non-real type (a `Union` with a
complex member, `Any`, `Number`) is evaluated before writing and rejected if an
element is complex. A write into either
of them through a view, through `vec` of a matrix or a `reshape` to another
shape, or through [`blockview`](@ref) converts each value as a plain array would, so
a complex value raises an `InexactError`, possibly after part of the iterate has
been written. A [`SymmetricSparseIterate`](@ref) refuses a complex source or element
type with an `ArgumentError`, and a complex value written to one of its entries
raises an `InexactError`.

## Entries and coordinates

[`prox_block`](@ref)`(g, u, k, γ)` computes one mathematical penalty block
without writing an iterate. A separable function takes a scalar `u` and a linear
entry index; a group penalty takes a compact vector and a penalty group index.
A solver writes the result through indexed assignment, `coordinates(x)`, or
`copyto!(blockview(x,k), out)`, then updates its residual cache.

An iterate coordinate identifies an independent storage update unit: one entry
for `SparseIterate`, a lower-triangle variable for `SymmetricSparseIterate`, or
one group for `BlockIterate`. A symmetric coordinate ID must be converted to
`i = (c-1)p+r` for the scalar penalty. Validate symmetric parameters explicitly.
Penalty groups and iterate groups must have matching indices and order before
a solver uses the same group ID in both APIs.

## Tracking an active set

The active coordinates form a stored working set, which can contain entries
or groups that have returned to zero. We keep them during a sweep so that
the set being traversed does not change.

| Operation | Meaning |
|:----------|:--------|
| [`ncoordinates`](@ref)`(x)` | Number of possible coordinates |
| [`nactive`](@ref)`(x)` | Number of stored active coordinates |
| [`activecoordinates`](@ref)`(x)` | Read-only list of the indices of the active coordinates |
| [`activevalues`](@ref)`(x)` | Writable view of their values, in the same order (`SparseIterate` and `SymmetricSparseIterate`) |
| [`isactive`](@ref)`(x, k)` | Whether coordinate `k` is active |
| [`activate!`](@ref)`(x, k)` | Add a coordinate to the active set |
| `dropzeros!(x)` | Remove stored zero entries or zero groups |

Scaling a `SparseIterate` by a number is a value write like `x .*= s`:
with a finite multiplier or a finite nonzero divisor it touches only the stored
entries. When the scalar maps zero to a nonzero value, every entry is written,
and entries that become nonzero join the working set.

Use [`kkt_violation`](@ref) to test excluded zero coordinates and
[`restrict`](@ref) to construct a supported restricted problem. The
[Coordinate descent](@ref) chapter explains the full optimality checks that
must accompany these storage operations.

### What an oracle leaves in an iterate

After [`prox!`](@ref), [`value_and_prox!`](@ref), [`gradient!`](@ref), `mul!` or
indexed assignment writes a `SparseIterate` or a `BlockIterate`, the
values of the iterate are the result, and every coordinate with a nonzero
result is active. Which zero coordinates stay active is unspecified: it can
depend on the function and on the type of the argument. Call `dropzeros!` to
make the working set the support. Indexed writes follow the rules
of [Steps on one coordinate](@ref).

In this example, group 1 of the destination is inactive and becomes nonzero,
and group 2 is active and becomes zero.

```@example oracle_destinations
using ProximalBase, LinearAlgebra

v = [3.0, 0.0, 0.0, 0.0]
oracles = [
    (y -> prox!(y, NormL1(1.0), v, 1.0), [2.0, 0.0, 0.0, 0.0]),
    (y -> value_and_prox!(y, NormL1(1.0), v, 1.0), [2.0, 0.0, 0.0, 0.0]),
    (y -> gradient!(y, SqrNormL2(1.0), v), [3.0, 0.0, 0.0, 0.0]),
    (y -> mul!(y, Matrix(2.0I, 4, 4), v), [6.0, 0.0, 0.0, 0.0]),
]
for (oracle!, result) in oracles
    a = BlockIterate(4, [1:2, 3:4])
    a[3] = 1.0                                      # group 2 active, group 1 inactive
    oracle!(a)
    @assert Vector(a) == result
    @assert isactive(a, 1)                          # the group with a nonzero result
    dropzeros!(a)
    @assert collect(activecoordinates(a)) == [1]    # the support
end
```

### Counting coordinates

Four counts describe an iterate, and each has its own spelling.

| Count | `SparseIterate` | `BlockIterate` |
|:------|:----------------|:--------------|
| Coordinates | `ncoordinates(x) == length(x)` | `ncoordinates(x)`, the number of groups |
| Active coordinates, stored zeros included | `nactive(x)` | `nactive(x)`, the active groups |
| Nonzeros | `count(!iszero, activevalues(x))` | entries: `sum(k -> count(!iszero, blockview(x, k)), activecoordinates(x); init = 0)`; groups: `count(k -> any(!iszero, blockview(x, k)), activecoordinates(x))` |
| Capacity | a slot for each of the `length(x)` entries | `length(x)` values and a slot for each group |

In 0.6, `SparseIterate` intentionally does **not** specialize `nnz`, `nonzeros`, `findnz` or `issparse`. Use `nactive(x)` to count stored entries (including explicit zeros), and pair `activecoordinates(x)` with `activevalues(x)` to traverse them in active-slot order. Sort a copied list of IDs if sorted output is needed. The size of a model is the number of nonzeros,
`count(!iszero, activevalues(x))`, or `nactive(x)` after `dropzeros!`.

A `BlockIterate` has no `nnz`, `nonzeros`, `findnz` or `activevalues`; calling
them throws a `MethodError`. Loop over its active entries with
`for k in activecoordinates(x), j in blockindices(x, k)`, reading or writing
`x[j]`.

The slots of both iterates are reserved at construction, so activating a
coordinate does not change the size of the arrays that the iterate stores.

A `SymmetricSparseIterate` has no `nnz`. `nactive(x)` counts the active variables of
the lower triangle, active zeros included; `ncoordinates(x)` is ``p(p+1)/2``. Because
an off-diagonal variable counts once, and an active zero counts too, `nactive(x)` can
differ from the number of nonzero entries of `Matrix(x)`; for that number count
`Matrix(x)` or `sparse(matrixview(x))`. Its other counts are tracked in
[#391](https://github.com/mlakolar/ProximalBase.jl/issues/391).

```@example counting
using ProximalBase, SparseArrays

y = SparseIterate(5)
y[4] = 1.0
activate!(y, 2)                                     # an explicit zero
@assert nactive(y) == length(activevalues(y)) == 2
@assert collect(activecoordinates(y)) == [4, 2] && activevalues(y) == [1.0, 0.0]
@assert sort(collect(zip(activecoordinates(y), activevalues(y))); by = first) == [(2, 0.0), (4, 1.0)]
@assert count(!iszero, activevalues(y)) == 1        # the model size
dropzeros!(y)
@assert nactive(y) == 1

a = BlockIterate(6, [1:2, 3:4, 5:6])
a[1] = 1.0
a[4] = -2.0
activate!(a, 3)                                     # an active group of zeros
entries = sum(k -> count(!iszero, blockview(a, k)), activecoordinates(a); init = 0)
groups = count(k -> any(!iszero, blockview(a, k)), activecoordinates(a))
@assert (ncoordinates(a), nactive(a), groups, entries) == (3, 3, 2, 2)
@assert sum(a[j]^2 for k in activecoordinates(a) for j in blockindices(a, k)) == 5.0
try
    nnz(a)
    error("expected a MethodError")
catch e
    @assert e isa MethodError
end

m = SymmetricSparseIterate(3)
m[2, 1] = 3.0
@assert nactive(m) == 1 && count(!iszero, Matrix(m)) == 2
@assert nnz(sparse(matrixview(m))) == 2             # the entries of the matrix, counted by position
try
    nnz(m)
    error("expected a MethodError")
catch e
    @assert e isa MethodError
end
```
