# Sparse iterates

ProximalBase provides three iterate types for coordinate descent and active-set
procedures. We separate the choice of storage from the update formulas,
which are derived in [Coordinate descent](@ref).

| Iterate | Array represented | One coordinate |
|:--------|:------------------|:---------------|
| [`SparseIterate`](@ref) | A sparse vector or matrix | One entry |
| [`SymmetricSparseIterate`](@ref) | A sparse symmetric matrix | One lower-triangular entry and its symmetric counterpart |
| [`AtomIterate`](@ref) | A vector partitioned into groups | One group of entries |

The entries of an iterate are real. A complex element type, or a complex operand
of an iterate product, raises an `ArgumentError`; see [Precision](@ref). Writing
a complex source or value into a `SparseIterate` or an `AtomIterate` with the
constructor, `copyto!`, `copy!`, a broadcast or `fill!` raises it too, before
anything is written, even when the imaginary part is zero. A write into either
of them through a view, through `vec` of a matrix or a `reshape` to another
shape, or through [`atom`](@ref) converts each value as a plain array would, so
a complex value raises an `InexactError`, possibly after part of the iterate has
been written. A [`SymmetricSparseIterate`](@ref) refuses a complex source or element
type with an `ArgumentError`, and a complex value written to one of its entries
raises an `InexactError`.

## Entries and coordinates

[`prox_entry`](@ref)`(g, v, i, γ)` computes the scalar proximal operator of
entry ``i`` without writing the iterate. In contrast,
[`prox_coordinate!`](@ref)`(x, g, k, γ)` reads coordinate ``k`` of the iterate,
applies its proximal operator and writes the result. The calling procedure
must supply the gradient step before applying this proximal operation.

The two calls index different spaces:
- `prox_entry` takes an entry ``i`` of the array, the index of the weights and
  bounds.
- `prox_coordinate!` takes a coordinate ``k`` of the iterate.

The two coincide for a `SparseIterate`. A coordinate is a variable, an entry of the
lower triangle, for a `SymmetricSparseIterate`, and a group for an `AtomIterate`.

## SparseIterate

A [`SparseIterate`](@ref) is a vector or matrix that stores its nonzero
entries in a list. Three arrays hold them: the values, their indices, and for
every index the position of its value in the list (zero when it is not stored).
Reading or writing any entry is therefore ``O(1)``, and a product ``Xx`` costs
``O(n\cdot\mathrm{nnz})``. The stored entries are the active coordinates
([`activecoordinates`](@ref), [`nactive`](@ref)). Setting an entry to zero keeps
it stored, so that an active set does not change during a sweep. `dropzeros!`
removes the zeros between sweeps. `nnz(x)` counts stored zeros too;
[Counting coordinates](@ref) lists the other counts.
`nonzeros(x)` gives a view of the stored values, paired with
`activecoordinates(x)` in storage order; the view is invalidated when the
stored set changes. `findnz(x)` returns copies of the stored indices and
values in sorted order, including stored zeros.

### Writing from an orthogonal factor

For a matrix `x`, `copyto!(x, Q)` accepts an orthogonal or unitary factor `Q` such
as `qr(A).Q`, `qr(A, ColumnNorm()).Q`, `lq(A).Q` or `hessenberg(A).Q` (the types
`QRCompactWYQ`, `QRPackedQ`, `LQPackedQ` and `HessenbergQ`), or the adjoint of one.
Such a `Q` is a lazy factorization object, not a dense matrix. `copyto!` materializes
it into a dense temporary of the element type of `x` and writes that matrix, so
a complex factor, a size other than `size(x)`, a failure to materialize `Q` and a
failing conversion (an `InexactError` when `x` has integer entries) all throw
before anything is written and leave `x` unchanged.

The write is an ordinary value write, like `copyto!(x, Matrix(Q))`, not a copy of
a working-set state: an active coordinate whose new value is zero stays active, and
a nonzero value activates its coordinate. The dense temporary and its allocation are
deliberate. A caller who cares about performance can form the dense matrix once and
pass it to `copyto!`.

### Matrix coefficients

`mul!(out, A, x, α, β)` with a matrix `x` walks the stored entries of `x` when
at most 1/32 of its entries are stored, and its columns in order otherwise. The
first reads one column of `A` per stored entry, in storage order; the second
reads `A` sequentially. Which is faster depends mostly on the length ``m`` of a
column of `A`. With entries stored in random order (as an active-set solver
creates them), the density at which the two walks take equal time (median over
shapes of `x`) falls with ``m``. For `Float64` it is 7–16% for 64 to 512 rows,
3–5% for 1024, about 1.5–2% for 2048 and 1–5% for 4096. The values are larger
for `Float32`. They were measured on Apple M3 Pro, AMD EPYC (Zen 3) and Intel
Xeon (Skylake-SP), and differ by a factor of two or more between these CPUs.

The fixed 1/32 is a compromise. Averaged over densities from 0.5% to 50% (log
scale, equal weight) it costs 2.5–5% of the time of the faster walk, 11–32% for
64-row `Float32` products. The best constant, 1/16, costs 1–3%. A cutoff
`min(1/8, 256/(m*sizeof(T)))` for ``m`` rows costs 0.6–1.5%. That gain over
1/16 was under the 2 points required, so no rule that depends on the size of
`A` was added (the margin was fixed after the first data, not before).

## SymmetricSparseIterate

A [`SymmetricSparseIterate`](@ref), the *owner* below, holds the
``q = p(p+1)/2`` variables of a real symmetric ``p \times p`` matrix: the
entries of its lower triangle, in the compact list of a `SparseIterate`. **The
owner is not an array.** It has a size, `size(x) == (p, p)`, an element type and
the operations named below; it has no `length`, `iterate`, `eachindex`,
`nnz`, whole-owner broadcasting, `fill!` or one-index access, and a function of
Julia or of a package that treats its argument as an array, such as `sum`, `map`,
`sort!`, `view`, `reshape` or `lu`, fails for it: with a `MethodError`, or, when it
asks for `axes(x)` first, with the `ArgumentError` described below. One matrix entry
is not one variable: writing the position ``(r, c)`` also writes ``(c, r)``, so a
loop that writes every position would write each off-diagonal variable twice, and
the owner has no array method that could do so for you. Read it through a handle
instead, or copy it.

| Handle | What it is |
|:-------|:-----------|
| `coordinates(x)` | a live, writable vector of the ``q`` variables, in the column-major order of the lower triangle |
| `matrixview(x)` | a live, read-only ``p \times p`` matrix: ordinary matrix indexing, both triangles |
| `activecoordinates(x)`, `activevalues(x)` | the ids of the active variables, and a writable view of their values in the same order |
| `Matrix(x)`, `Array(x)` | an independent dense copy, which is writable |

### Three index spaces

A pair, a variable and an active slot are indexed differently. Mixing them is the
usual mistake of code written for 0.5, where `x[k]` could be either of the first two.

| Space | Index | Meaning | Range |
|:------|:------|:--------|:------|
| pair | `x[i, j]` | the variable of the position ``(i, j)``, from either triangle | `i`, `j` in `1:p` |
| variable id | `coordinates(x)[k]` | variable ``k`` of the lower triangle, numbered by [`tril_index`](@ref) | `1:ncoordinates(x)` |
| active slot | `activecoordinates(x)[a]` and `activevalues(x)[a]` | the id and the value of the ``a``-th active variable | `1:nactive(x)` |

`matrixview(x)[i]` with one index is a *matrix* linear index in column-major
order, from `1` to ``p^2``, not a variable id; and `x[k]` with one integer is an
`ArgumentError` that says so, because the two readings differ.

```@example owner_index_spaces
using ProximalBase

p = 3
x = SymmetricSparseIterate(p)
x[2, 2] = 7.0                                        # the pair (2, 2) is variable 4
x[3, 1] = 5.0                                        # the pairs (3, 1) and (1, 3) are variable 3
@assert x[1, 3] == 5.0 && coordinates(x)[3] == 5.0
@assert tril_index(p, 2, 2) == 4 && coordinates(x)[4] == 7.0
@assert collect(activecoordinates(x)) == [4, 3]      # variable ids, in the order of activation
@assert collect(activevalues(x)) == [7.0, 5.0]       # their values, slot by slot
@assert matrixview(x)[4] == 0.0                      # linear index 4 is the position (1, 2), a different variable
@assert matrixview(x)[5] == 7.0                      # and linear index 5 is (2, 2)
try
    x[3]
    error("expected an ArgumentError")
catch e
    @assert e isa ArgumentError
    showerror(stdout, e)
    println()
end
```

A write `x[i, j] = v` converts `v` to the element type first. It stores a nonzero
converted value (`NaN` included) as an active variable, does not activate a variable
for a zero, and keeps an active variable active when it writes a zero. Use
[`activate!`](@ref) for an explicit active zero, and `dropzeros!` to remove active
zeros. A pair outside `1:p` is a `BoundsError`. A complex scalar converts when its
imaginary part is zero (`x[1, 1] = 1 + 0im` stores `1`); a nonzero imaginary part is an
`InexactError`. A complex matrix source and a complex element type are refused with an
`ArgumentError`, before anything is written: the entries of an owner are real.

`axes(x)` throws an `ArgumentError`. Julia derives `axes` from `size` for any object
that has a size, and generic code that asks for the axes of the owner would then
treat it as an array; the owner defines this one method so that it fails at once.
Use `size(x)`, `coordinates(x)` for its variables, or `axes(matrixview(x))`.

### Copies, imports and equality

- `copy(x)` is an independent state: the values, the active zeros and the order of
  activation. `similar(x)` and `similar(x, S)` are empty owners of the same size.
- `copyto!(y, x)` between owners replaces the state of `y` by a copy of that of `x`,
  working set included; the sizes must agree.
- `copyto!(x, M)` and `SymmetricSparseIterate(M)` *import* a matrix `M` and replace the
  working set: `M` must be real, of size `(p, p)`, exactly symmetric (`NaN` matches `NaN`,
  `0.0` matches `-0.0`, compared before conversion to the element type) and must not share
  memory with `x`. The checks run before anything is written. The working set becomes the
  variables whose converted value is nonzero, in increasing order of id, so an active zero
  of `x` is not carried by `Matrix(x)`. `copyto!(x, matrixview(x))` is refused for sharing
  memory, while `prox!(x, f, matrixview(x), γ)` is supported: a proximal map reads its
  argument before it writes.
- An asymmetric matrix is not repaired. Pass `Symmetric(M, :L)`, or `Symmetric(M, :U)`, to
  choose a triangle yourself: the caller selects it, and ProximalBase does not guess. A
  *proximal* call is different: `prox!` into an owner reads its point through its symmetric
  part ``(V + V^\top)/2`` ([The argument of a proximal map](@ref)), because that is the
  proximal point over symmetric matrices.

```@example owner_copies
using ProximalBase, LinearAlgebra

x = SymmetricSparseIterate(3)
x[2, 1] = 1.0
activate!(x, 6)                                      # an explicit zero at (3, 3)
y = copy(x)
@assert collect(activecoordinates(y)) == [2, 6] && isactive(y, 6)

w = SymmetricSparseIterate(3)
copyto!(w, Matrix(x))                                # an import: the explicit zero is not carried
@assert collect(activecoordinates(w)) == [2]

A = [1.0 2.0 0.0; 2.5 1.0 0.0; 0.0 0.0 1.0]          # not symmetric
try
    copyto!(w, A)
    error("expected an ArgumentError")
catch e
    @assert e isa ArgumentError
end
@assert collect(activecoordinates(w)) == [2]         # nothing was written
copyto!(w, Symmetric(A, :L))                         # the lower triangle, chosen by the caller
@assert w[1, 2] == 2.5

try
    copyto!(w, matrixview(w))                        # the source shares memory with w
    error("expected an ArgumentError")
catch e
    @assert e isa ArgumentError
end
prox!(w, NormL1(0.1), matrixview(w), 1.0)            # a proximal map from its own view is fine
@assert w[2, 1] ≈ 2.4
```

Two owners are `==` and `isequal` when their matrices are: `isequal(x, y)` is
`isequal(Matrix(x), Matrix(y))`, whatever the active sets and their order. An active
`-0.0` is therefore different under `isequal` from an absent coordinate, which is
`+0.0`, and is equal to it under `==`; two `NaN`s are `isequal` and not `==`. Equal
matrices have equal hashes. There is no comparison of an owner with an array, and
there is no `isapprox` for owners: compare `Matrix(x)` or `matrixview(x)`, which are
ordinary matrices.

```@example owner_equality
using ProximalBase

x = SymmetricSparseIterate(3)
x[2, 1] = 1.0
activate!(x, 6)                                      # an active zero at (3, 3)
y = SymmetricSparseIterate(3)
y[2, 1] = 1.0
@assert isequal(x, y) && x == y && hash(x) == hash(y) # an active zero is an absent coordinate
y[2, 1] = 1.0 + 1e-9
@assert !isequal(x, y)
@assert isapprox(matrixview(x), matrixview(y); atol = 1e-6)   # an approximation compares the matrices
try
    isapprox(x, y)
    error("expected a MethodError")
catch e
    @assert e isa MethodError
end
```

### The multiplicity of an off-diagonal variable

A variable ``(r, c)`` with ``r \ne c`` stands for two entries of the matrix.
Everything that is a function of the matrix counts both: `norm(x)` is the Frobenius norm
of `Matrix(x)`, `dot(A, x)` and `dot(a, x, b)` are the inner products of the matrix, a
penalty such as `NormL1` takes the sum over both triangles, `IndBallL0` counts the entries of the
matrix, so a nonzero off-diagonal pair counts two, the gradient and the curvature
along a coordinate are those of the direction ``E_{rc} + E_{cr}``, and a proximal step is
the Frobenius one. The values of the variables themselves, `x[r, c]`, are not rescaled.
`kkt_violation` screens a coordinate at zero: a positive value says the coordinate should
enter the working set, and a value that is not positive does not certify that a returned
solution is stationary.

```@example owner_multiplicity
using ProximalBase, LinearAlgebra

x = SymmetricSparseIterate(3)
x[2, 1] = 3.0
x[3, 3] = 4.0
@assert norm(x) ≈ sqrt(2 * 3.0^2 + 4.0^2)           # (2, 1) counts twice
@assert norm(x) ≈ norm(Matrix(x))
@assert NormL1(1.0)(x) == 2 * 3.0 + 4.0
@assert dot(ones(3, 3), x) == 2 * 3.0 + 4.0
@assert x[1, 2] == 3.0                               # the variable itself is not rescaled
```

An owner as the output of a proximal map takes a *scalar* Frobenius step ``\gamma``. An array
of steps, or one step per group, is not supported on an owner: it throws an `ArgumentError`
before anything is written, and the dense and other iterate outputs are unchanged. Use a
scalar step, or `Matrix(x)` as the output.

### Array operations and the calls to use instead

An array function is not defined for the owner. Each of the following says what to
call for the question that the array function would have answered.

| To get | Use |
|:-------|:----|
| a variable by its id | `coordinates(x)[k]` |
| a matrix read or iteration: `x[:, j]`, `sum`, `map`, `view`, broadcasting | the same call on `matrixview(x)` |
| ``p`` | `size(x, 1)` |
| ``q`` | `ncoordinates(x)`, or `length(coordinates(x))` |
| the size of the working set | `nactive(x)` |
| the number of nonzeros of the matrix | `count(!iszero, Matrix(x))`, or `nnz(sparse(matrixview(x)))` |
| to clear the owner | `fill!(coordinates(x), 0); dropzeros!(x)` |
| the stored values | `activevalues(x)` |
| writable dense storage | `Matrix(x)` |
| the matrix as a sparse matrix, or a triangle of it | `sparse(matrixview(x))`, `triu(matrixview(x), k)`, `tril(matrixview(x), k)` |
| the product of two matrices stored in the owner | compute `C = A * B` into a dense matrix, and `copyto!(x, C)` if a symmetric import is meant |

[Compatibility and migration](@ref) lists what each of these was in 0.5.

Its coordinates are the
``p(p+1)/2`` entries ``(r, c)``, ``r \ge c``, of the lower triangle, numbered
column by column:

```math
k = \texttt{tril\_index}(p, r, c) = \frac{(c - 1)(2p - c + 2)}{2} + (r - c + 1).
```

[`tril_cartesian`](@ref) inverts this by solving a quadratic equation. Its
floating square root is exact for ``p < 2^{24}``, far beyond any matrix that
fits in memory, and it switches to an integer square root beyond that.
[`coordinates`](@ref)`(x)` is a vector view of the coordinates.

**The coordinate step on a symmetric matrix.** A coordinate ``(r, c)`` with
``r \ne c`` moves two entries of ``\Theta`` together. Take a penalty that
counts both triangles, ``\lambda\sum_{i,j}|\Theta_{ij}|``, as `NormL1` does on
the matrix. Let ``G`` be the gradient of ``f`` over symmetric matrices, and
``a`` the curvature along the direction ``E_{rc} + E_{cr}``. Along the
coordinate,

```math
\varphi(t) = 2G_{rc}(t - t_0) + \frac{a}{2}(t - t_0)^2 + 2\lambda|t| + \text{const},
\qquad
t^+ = S_{2\lambda/a}\Big(t_0 - \frac{2G_{rc}}{a}\Big).
```

That is `prox_entry(NormL1(λ), t₀ - (2/a) G[r, c], i, 2/a)`. On the diagonal
the factors of 2 disappear and the step is ``1/a``. So, as the
[`KronQuadratic`](@ref) example of the D-trace loss shows, off-diagonal
coordinates take the step ``2/a`` where the diagonal takes ``1/a``.

### A read-only matrix view

[`matrixview`](@ref)`(x)` returns a live, read-only ``p \times p`` view of a
`SymmetricSparseIterate`: ordinary matrix indices, both triangles, no copy. Pass it
to code that only reads a matrix. Writes to `x` show through it, and nothing can be
written through it or through a wrapper derived from it (`view`, `reshape`,
`transpose`, `Symmetric`): the write is refused and `x` is unchanged. A generic
allocating operation on the view (`A .+ 1`, `similar(A)`, `collect(A)`) returns an
ordinary writable dense array. `mul!(y, matrixview(x), v)` costs
``O(p + \mathrm{nactive}(x))``; other generic matrix operations read all ``p^2``
entries. `sparse(matrixview(x))`, `triu(matrixview(x), k)` and `tril(matrixview(x), k)`
return a `SparseMatrixCSC` built from the active variables in
``O(\mathrm{nactive}(x))``; an active zero is not an entry of the result.

Its linear indices are matrix indices, not coordinates: for ``p = 3``,
`coordinates(x)[4]` is the variable at ``(2, 2)`` and `matrixview(x)[4]` is the
entry at ``(1, 2)``. [`activevalues`](@ref)`(x)` gives the values of the active
coordinates in the order of [`activecoordinates`](@ref)`(x)`.

A variable that becomes active, `dropzeros!`, and a `copyto!` that replaces the state
of `x` change its working set, and a handle taken earlier does not follow: an
`activecoordinates` or `activevalues` handle keeps its length when a variable becomes
active, and after `dropzeros!` or a state replacement it may show other slots, though
reading it stays in bounds. Take it again after any of them. The views
`matrixview(x)` and `coordinates(x)` read the live state and need no refresh.
An active zero stays active when a zero is written to it, so that the set being swept
does not change: `fill!(coordinates(x), 0)` writes zeros to the active variables only,
in ``O(\mathrm{nactive}(x))``, and `dropzeros!(x)` then empties the owner.

```@example owner_view
using ProximalBase, SparseArrays, LinearAlgebra

x = SymmetricSparseIterate(4)
x[3, 1] = 2.0
x[4, 4] = 5.0
activate!(x, 2)                                      # an active zero at (2, 1)
A = matrixview(x)
S = sparse(A)
@assert S isa SparseMatrixCSC && S == sparse(Matrix(x))
@assert nnz(S) == 3                                  # (3, 1), (1, 3) and (4, 4); the active zero is no entry
@assert triu(A) == triu(Matrix(x)) && tril(A, -1) == tril(Matrix(x), -1)
@assert nnz(triu(A, 1)) == 1 && nnz(tril(A, -1)) == 1
try
    A[1, 1] = 1.0
    error("expected the write to be refused")
catch e
    @assert occursin(r"setindex!|read-only|not defined", sprint(showerror, e))
end
@assert x[1, 1] == 0.0

ids = activecoordinates(x)
@assert length(ids) == nactive(x) == 3
x[4, 3] = 1.0                                        # a new active variable
@assert length(ids) == 3 && length(activecoordinates(x)) == 4   # the old handle did not follow
fill!(coordinates(x), 0)                             # writes the active variables only
@assert nactive(x) == 4 && all(iszero, Matrix(x))
dropzeros!(x)
@assert nactive(x) == 0
```

### Symmetric parameters

Pass exactly symmetric parameter arrays when you use a `SymmetricSparseIterate`.
It stores only the lower triangle, the positions ``(r, c)`` with ``r \ge c``,
so different weights at ``(r, c)`` and ``(c, r)`` would make the answer depend
on which triangle is read. A check that finds different values throws an
`ArgumentError`.

A *per-entry parameter*, such as `W[r, c]`, assigns a weight or bound to
`x[r, c]`. Positions `(r, c)` and `(c, r)` form a mirrored pair.
The package symmetrizes computed results, not your parameters.

This weighted ℓ1 step stores an off-diagonal entry explicitly so that the
proximal call checks its weight.

```@example symmetric_parameters
using ProximalBase, LinearAlgebra

p = 3
x = SymmetricSparseIterate(p)
x[2, 1] = 4.0
W = [1.0 2.0 1.0; 2.0 1.0 1.0; 1.0 1.0 1.0]
y = prox(NormL1(1.0, W), x, 1.0)
@assert y[2, 1] == y[1, 2] == 2.0
@assert Matrix(y) == [0.0 2.0 0.0; 2.0 0.0 0.0; 0.0 0.0 0.0]
Matrix(y)
```

Changing just the upper weight breaks that agreement.

```@example symmetric_parameters
Wbad = copy(W)
Wbad[1, 2] = 6.0
try
    prox(NormL1(1.0, Wbad), x, 1.0)
    error("expected asymmetric weights to be rejected")
catch e
    @assert e isa ArgumentError
    showerror(stdout, e)
    println()
end
```

The repairs have different meanings. `Symmetric(Wbad, :L)` keeps the lower
weight of 2 at both positions. Averaging sets both weights to 4, changing the
penalty.

```@example symmetric_parameters
Wlower = Symmetric(Wbad, :L)
Wmean = (Wbad + Wbad') / 2
@assert Wlower[1, 2] == Wlower[2, 1] == 2.0
@assert Wmean[1, 2] == Wmean[2, 1] == 4.0
ylower = prox(NormL1(1.0, Wlower), x, 1.0)
ymean = prox(NormL1(1.0, Wmean), x, 1.0)
@assert Matrix(ylower) == Matrix(y)
@assert iszero(Matrix(ymean))
(Matrix(ylower), Matrix(ymean))
```

Choose the repair to match your model. Averaging is not a general repair for
patterns, bounds or labels: their meaning and domain must remain valid.
The sum in `(W + W') / 2` can also overflow even when both entries are finite.

Equality is exact, with no tolerance: `NaN` matches `NaN`, `0.0` matches
`-0.0`, and `1` matches `1.0`. Constructors separately check the parameter's
domain, such as real element types and nonnegative weights.

The table lists arrays indexed by matrix entry. Where a function accepts a
vector of length ``p^2``, it uses column-major order: `(c - 1)p + r` for
`(r, c)`. Usual size and axes requirements still apply.

| Function or argument | Per-entry parameter |
|:---|:---|
| `NormL1`, `NormL1Nonneg`, `ElasticNet`, `CappedL1`, `LogPenalty`, `NormLq`, `Berhu`, `CheckLoss`, `EpsInsensitiveLoss` | weights `w` |
| `HingeLoss`, `SqrHingeLoss` | weights `w`, labels `y` |
| `BernoulliLoss` | weights `w`, data `b` |
| `Tilt`, `LinearFunction`, `Regularize` | coefficients or center `c` |
| `IndBox`, `IndBinary` | bounds `l`, `u` |
| `IndPattern` | pattern `P` |
| `SqrNormL2` | center `y` |
| `KLDivergence` | reference `q` |
| `Translate` | shift `b` |
| array `Precompose` | scale `a` |
| `IndHyperplane`, `IndHalfspace`, `IndHyperslab` | normal `a` |
| `KronQuadratic` | linear coefficient `C` |

Design matrices and image data have different indexing and are not covered
by this rule. Neither are arrays indexed by groups, ranks or eigenvalues.

#### What each call checks

A stored entry is a lower-triangle coordinate retained by the iterate,
including one whose value has returned to zero. Calls that check only these
entries cannot detect asymmetry elsewhere, so success does not certify the
whole parameter array.

This proximal call succeeds while `(3, 1)` is absent, then fails once that
entry is stored.

```@example symmetric_parameters
Wlater = copy(W)
Wlater[1, 3] = 3.0
f = NormL1(1.0, Wlater)
z = SymmetricSparseIterate(3)
z[2, 1] = 4.0
@assert Matrix(prox(f, z, 1.0)) == Matrix(y)
z[3, 1] = 4.0
try
    prox(f, z, 1.0)
    error("expected the newly stored pair to be checked")
catch e
    @assert e isa ArgumentError
    showerror(stdout, e)
    println()
end
```

Built-in separable functions such as weighted `NormL1` can compute values
and proximal steps using only stored entries when they preserve zero:
`fᵢ(0) = 0` and the scalar proximal operator maps zero to zero.
Some wrappers retain this behavior; others compare every pair.

Some calls compare every mirrored pair, including positions the iterate does
not store. Others compare only pairs at stored entries because those are all
they read, as with weighted `NormL1` above. The table identifies which
argument must be a `SymmetricSparseIterate` for these checks to apply.

| Operation | Checked when this argument is a `SymmetricSparseIterate` | Which pairs are compared |
|:---|:---|:---|
| `f(x)` | Input `x` | Every pair, or only stored pairs as with weighted `NormL1` |
| [`prox!`](@ref), [`value_and_prox!`](@ref) | Output, or the point `x` with a dense output | Every pair, or pairs stored in either input or output as with weighted `NormL1` |
| [`gradient!`](@ref), [`value_and_gradient!`](@ref) | Output, or the point `x` with a dense output | Every pair |
| Both forms of [`prox_coordinate!`](@ref) | Iterate `x` | The pair at the chosen coordinate |
| Four-argument [`kkt_violation`](@ref) | Iterate `x` | The pair at the chosen coordinate |
| [`hessian_vec!`](@ref) | Point `x`, even with dense output; the output, even at a dense point | Every pair |
| [`quadratic_model`](@ref), [`quadratic_model!`](@ref) | Point `x` | Every pair |
| [`coordinate_lipschitz!`](@ref) | Output `L` | Every pair |
| Separable [`lla_weights!`](@ref), [`lla_weights`](@ref) | Input `x`, even with dense output | Every pair |
| [`ProximalBase.check_symmetric_parameters`](@ref) | Iterate `x` | Every declared pair, also inside built-in wrappers and `SeparableSum` blocks |

Comparing only stored pairs in a proximal call requires both input and output
to be symmetric iterates. For tuple arguments, these rules apply to each symmetric
block. A `SymmetricSparseIterate` output takes a scalar step only; an array step is an
`ArgumentError` (see [The multiplicity of an off-diagonal variable](@ref)), and
`gradient_mapping` and `gradient_mapping!` have no method for an owner: pass `Matrix(x)`.

For a `SymmetricSparseIterate` input `x`:

- `prox(f, x)` returns a `SymmetricSparseIterate`, so it checks like `prox!`.
- `gradient(f, x)` returns a dense matrix; the owner `x` is checked all the same, and
  an asymmetric parameter is an `ArgumentError` (`LinearFunction([1.0 2; 3 1])` at an
  owner point throws).
- `value_and_gradient(f, x)` returns a value and a dense gradient matrix, and checks
  the owner `x` in the same way.
- `lla_weights(f, x)` returns a dense matrix and checks every parameter pair
  because it checks `x`.

`kkt_violation(f, g, x, k)` uses the symmetric part of the gradient at the
chosen coordinate. It keeps equal entries exactly; otherwise it computes
`g[r, c] / 2 + g[c, r] / 2`. Dividing first avoids overflow to `Inf` when
adding two large finite entries. `+Inf` at one position and `-Inf` at its
mirror have no symmetric part and cause an `ArgumentError` at the chosen
coordinate. The call does not search the rest of `g` for such pairs.

To validate every declared parameter pair before a solver starts, call
`ProximalBase.check_symmetric_parameters(f, x)`. It returns `true` on success.
It does not replace arrays, remember earlier checks, or check separately supplied
steps. Ordinary calls still check; later mutation can invalidate validation. See
[Separable functions](@ref extending-separable) for declarations and solver use.

#### The argument of a proximal map

The argument `x` of `prox!` and `value_and_prox!` is not a parameter, and it is
not checked. For a symmetric ``Z`` and any ``V``,

```math
\|Z - V\|_F^2 = \|Z - (V + V^\top)/2\|_F^2 + \|(V - V^\top)/2\|_F^2,
```

so the proximal point over symmetric matrices at ``V`` is the proximal point at its
symmetric part, for every function. When the output is a `SymmetricSparseIterate`
and `x` is a square matrix of the same size that is not symmetric by type
(`Matrix`, `SparseMatrixCSC`, `SparseIterate`, ...), every built-in `prox!` and
`value_and_prox!` reads `x` through its symmetric part. Each pair is averaged as the correctly rounded `(a + b) / 2`
when the sum is finite (this keeps subnormal digits), and as `a / 2 + b / 2` when
the sum overflows; equal entries are kept exactly, and the result is exactly
symmetric. A `SymmetricSparseIterate`, a `Symmetric`, a `Hermitian`, a `Diagonal`
or a `SymTridiagonal` is used as it is, at no extra cost, and `Symmetric(x, :L)`
still selects the lower triangle. `+Inf` at one position and `-Inf` at its mirror give
`NaN`, as their sum does. For a tuple output, as in a `SeparableSum`, each block is
treated this way at every depth of nesting, so no block fails before another is written.
The rule holds for an argument on a device too, which is read on the host.

```@example symmetric_parameters
V = [0.0 1.0; 3.0 0.0]                   # not symmetric
y = SymmetricSparseIterate(2)
prox!(y, NormL1(0.5), V, 1.0)
@assert y[2, 1] == 1.5                   # the symmetric part has 2 off the diagonal
prox!(y, NormL1(0.5), Symmetric(V, :L), 1.0)
@assert y[2, 1] == 2.5                   # the lower triangle has 3
```

The spectral functions (`Spectral`, `IndPSD`, `TracePSD`) are defined on symmetric
matrices, so they decompose the symmetric part of a matrix that is not symmetric by
type for a dense output as well, instead of reading its upper triangle.
`gradient!`, `value_and_gradient!` and `hessian_vec!` of `KronQuadratic` write the
symmetric part of the result into a `SymmetricSparseIterate` output, also when
the point is a dense matrix that is not symmetric.

Weights and groups that are not symmetric are parameters, not the argument, and still
cause an `ArgumentError`. `Symmetric(x, :L)` is a choice of the caller, who selects the
triangle to read; the package never picks one for a parameter.

#### When no check happens

- Raw entry calls (`value_entry`, `prox_entry`, `lla_weight`, scalar
  `coordinate_lipschitz` and three-argument `kkt_violation`) and `restrict`
  have no symmetric iterate from which to determine mirrored positions.
  Validate the original function before using them in a symmetric solver.
- Scalars and `nothing` have no pair of array positions to compare.
  Diagonal entries mirror themselves and need no comparison.
- Arrays symmetric by construction, such as `Symmetric(W)`, skip comparison
  of their entries. The wrapper defines equal mirrored values by reading one
  triangle; it does not verify agreement between the triangles of `W`.
- Dense arrays, `SparseIterate` and `AtomIterate` do not trigger this
  symmetric-parameter check. For example, `gradient!` into a dense matrix at a
  dense point does not check. A call checks as soon as one of its iterate arguments
  is a `SymmetricSparseIterate`, as the point (`gradient!` into a dense matrix at
  an owner point throws for an asymmetric parameter) or as the output.
- An override of a public oracle replaces the generic check. Its author must
  call `ProximalBase.check_symmetric_parameters`; declaring parameters alone
  does not run the check.

#### Cost and remaining limitations

Checking a dense parameter array adds work. An exploratory measurement of
`prox!` with dense weights at ``p = 1000`` took 1.6868 ms against 1.3059 ms
before checking, about 1.3 times as long
([measurement](https://github.com/mlakolar/ProximalBase.jl/pull/329)).
`Symmetric(W)` skips pair comparisons, but still has overhead. Performance
work is tracked in
[#374](https://github.com/mlakolar/ProximalBase.jl/issues/374).

Parameter validation does not guarantee every symmetric operation works.
`KronQuadratic` checks `C`; its `A`, `B`, and `LogDetLoss`'s `S` have separate
symmetry policies under [#269](https://github.com/mlakolar/ProximalBase.jl/issues/269).
`SphereConstrained` checks its inner arrays and measures the sphere in the
Frobenius norm: an off-diagonal pair that carries the proximal point of the
fallback takes ``r/\sqrt 2`` in each of its entries (see
[Separable functions with constraints](@ref)).
Group, cone and slice policies remain under
[#342](https://github.com/mlakolar/ProximalBase.jl/issues/342): group proximal
operators already require mirrored positions to belong to the same group,
and group KKT calls reject symmetric coordinates.

## AtomIterate

An [`AtomIterate`](@ref) is a vector whose coordinates are groups of entries,
partitioning ``1{:}p``. Its active groups play the role of the stored entries
of a `SparseIterate`, and [`atom`](@ref)`(x, k)` is a view of `x` at group
``k``. Writes through it follow the rule of `x[i] = v`: a nonzero entry
activates the group, and a zero never deactivates it.
Block coordinate descent for the group lasso computes the trial point
``u = x_{G_k} - \nabla_{G_k} f(x)/L_k`` for group ``k``, with
``L_k = \|X_{G_k}\|_2^2`` for least squares. It then takes the block step
`prox_coordinate!(x, g, k, u, 1/L_k)`, which is block soft thresholding and
activates the group only when it ends nonzero. [Block trials](@ref) gives the
rules of this form. `AtomIterate(p, g)` builds an
iterate that shares the groups of a group penalty `g`.

### The groups of an iterate

[`atomindices`](@ref)`(x, k)` returns the indices of the entries of group `k`, in the order in
which `x` stores them, and `ProximalBase.samegroups(a, b)` tells whether two layouts are equal.
Use them instead of reading the fields of `x`, which are private.

`atomindices(x, k)` is a read-only `AbstractVector{<:Integer}`. A group that `x` stores as a
range comes back as that range, with its own index type: a group stored as a
`UnitRange{Int32}` is a `UnitRange{Int32}`, and `AtomIterate(p, K)` stores `UnitRange{Int}`s.
Every other group is stored as a `Vector{Int}`, and comes back as a read-only wrapper whose
element type is `Int`, so `Int32` vectors, mixed groups and collections of other integer types
all give `Int`. Writing through the wrapper (`h[i] = v`, `fill!`, `copyto!`, `h .= v`, a `sort!`
that moves an entry) throws an `ArgumentError` and leaves the layout unchanged; `copy` and
`collect` give an ordinary array. The call takes constant time, does not allocate, and throws a
`BoundsError` unless `k` is in `1:ncoordinates(x)`. The result stays valid for the lifetime
of `x`: the groups of an iterate do not change, and activating groups, `dropzeros!`, `copyto!`
and writes of values do not affect it. It indexes other arrays like any integer vector, for
instance `view(g, atomindices(x, k))`. A vector group is wrapped, so such a view is not
indexed by a `Vector{Int}`, and its norms and dot products take the generic paths.

`atomindices` is the supported route to the layout. `parentindices(atom(x, k))` returns the
private group array itself, and changing that array is unsupported: the groups and the map from
entries to groups would disagree. The same holds for the fields of the wrapper that
`atomindices` returns (`h.ids`) and for the fields of the iterate: they are internal, reading or
writing them is not covered by this contract, and ProximalBase does not block `getproperty`.
What the contract does cover is every documented operation on the result, and the exceptions
that `atomindices(x, k)` and its wrapper throw: they name the range of valid groups or the
wrapper, and do not hold a stored group. (`atom(x, k)` is unchanged: its `BoundsError` still names
the stored groups.)

`samegroups(a, b)` is `true` when `a` and `b` have the same number of groups and the same
indices in the same order in every group, and `false` otherwise. `a` and `b` are two iterates,
or an iterate and a group penalty or [`IndBallGroupL0`](@ref), in either order. Ranges and vectors
with the same indices are equal, whatever their index types, and reordering the groups or the
indices inside a group gives `false`. For a group penalty, it is the check that [`prox_coordinate!`](@ref) makes
before it writes and that the four-argument [`kkt_violation`](@ref)`(f, g, x, k)` makes before it
computes; each throws a `DimensionMismatch` when it is `false`. The whole-array oracles do not
require it: they compute the function of the vector for any layout, and use `samegroups` only to
choose the faster active-group paths.

An iterate and a group penalty whose groups are equal but were built separately
also take the active-group paths. Each iterate remembers one group array that it found equal to
its own:

- When `a` and `b` share their group array, `samegroups` takes constant time.
- When two iterates remember the same array, or one remembers the other's, it also takes
  constant time and writes nothing.
- Otherwise it compares the two layouts in full, which takes time proportional to the total
  size of the groups. If they are equal, the iterate (the first argument when both are iterates)
  remembers the other's group array, so that asking again about that array takes constant time.
  If they differ, it returns `false` and the iterate keeps what it remembered.
- The slot holds one array. Alternating between two equal group arrays that are different
  objects compares at every call, and an iterate that is also used with a penalty remembers
  only the one it was last compared with.

`copy` and `similar` keep what the iterate remembers, `copyto!` between two iterates
keeps the destination's, and an iterate built from a restricted penalty starts with nothing
remembered. Comparisons use no table or other storage.

Group arrays and the indices inside them must not be mutated after a penalty
or an iterate is built. This also applies to arrays retained by inner
constructors.

## Coordinate oracles

[`prox_coordinate!`](@ref) and the four-argument [`kkt_violation`](@ref) act on
one coordinate ``k`` of an iterate: an entry of a `SparseIterate`, or a group of
an `AtomIterate` (see [Entries and coordinates](@ref)). This section states what
they do on these two iterates. For a `SymmetricSparseIterate`, see also
[What each call checks](@ref).

### Steps on one coordinate

`prox_coordinate!(x, f, k, γ)` applies the proximal operator of ``\gamma f`` to
the current value of coordinate ``k``. `prox_coordinate!(x, f, k, u, γ)` applies
it to a trial value `u` instead, without writing `u` into `x` first. The trial
is a number for a `SparseIterate`, and a vector of the group's length for an
`AtomIterate`. Both forms write the result into coordinate ``k`` and return
it: the new value of the entry for a `SparseIterate`, and `atom(x, k)` for an
`AtomIterate`.

Only coordinate ``k`` can join the working set:

- A coordinate that is absent and whose result is zero stays absent.
- A coordinate whose result is nonzero becomes active.
- An active coordinate stays active, even when the step sets it to zero. No
  coordinate is removed, and no other coordinate changes.

Both forms support separable functions on both iterates. On an `AtomIterate`
they also support the group penalties of the package (`GroupNormL2`,
`GroupNormLinf`, `SparseGroupNormL2`, `GroupMCP` and `GroupSCAD`) whose groups
are those of `x`, and their `Scaled` and `Regularize` versions. The arguments
are checked before anything is written, and after each of these errors `x` is
unchanged:

| Error | Cause |
|:------|:------|
| `BoundsError` | `k` is not a coordinate of `x` |
| `DimensionMismatch` | the groups of a group penalty differ from those of `x` (see [The groups of an iterate](@ref)) |
| `ArgumentError` | any other function, such as a group penalty on a `SparseIterate`, `IndBallGroupL0`, or `Precompose` of a group penalty with an array |
| `DimensionMismatch` | a parameter array, such as the weights of `NormL1` or the center of `Regularize`, whose length is not `length(x)` |
| `ArgumentError` | a trial `u` whose element type is not real |
| `DimensionMismatch` | for an `AtomIterate`, a trial `u` whose length is not the size of group ``k`` |

On an `AtomIterate`, an error outside this list, such as an entry of `u` whose
reading throws or a value `1.5` for an iterate of integers, can leave entries of
the group written. A group with a nonzero entry is then active.

```@example coordinate_steps
using ProximalBase

x = SparseIterate(4)
x[1] = 0.5
v = prox_coordinate!(x, NormL1(1.0), 1, 1.0)        # soft thresholding of 0.5 at 1
@assert v == 0 && isactive(x, 1)                    # an active entry stays active at zero
v = prox_coordinate!(x, NormL1(1.0), 2, 3.0, 1.0)   # the trial value 3.0 is not written first
@assert v == 2.0 == x[2] && isactive(x, 2)
prox_coordinate!(x, NormL1(1.0), 3, 0.5, 1.0)       # a zero result
@assert !isactive(x, 3)                             # leaves an absent entry absent
try
    prox_coordinate!(x, NormL1(1.0, ones(3)), 4, 1.0)   # three weights for four entries
    error("expected a DimensionMismatch")
catch e
    @assert e isa DimensionMismatch
end
@assert Vector(x) == [0.0, 2.0, 0.0, 0.0]
Vector(x)
```

A trial `u` that shares memory with `x`, such as `atom(x, j)` or a view of `x`
with its entries permuted, is copied before anything is written, so the result
does not depend on the order of the writes. Here the trial is group 1 itself in
reverse order. Writing its entries one at a time without the copy would give
the trial ``(4, 4)`` and the result ``(3, 3)``.

```@example coordinate_steps
f = GroupNormL2(1.0, [1:2, 3:4])                    # weights √2
a = AtomIterate([3.0, 4.0, 0.0, 0.0], f)
r = prox_coordinate!(a, f, 1, view(a, [2, 1]), 1.0) # the trial (4, 3)
@assert collect(r) ≈ (1 - sqrt(2) / 5) .* [4.0, 3.0]
@assert r == atom(a, 1)
```

### Block trials

Block coordinate descent computes a trial point ``u`` for group ``k`` (see
[AtomIterate](@ref)). The supported way to take the group step at ``u`` is
`prox_coordinate!(x, f, k, u, γ)`. It gives the values that writing `u` into
group ``k`` and then calling `prox_coordinate!(x, f, k, γ)` gives. The working
sets differ: writing a nonzero `u` activates the group, while the five-argument
form activates it only when the result is nonzero. A group whose step ends at
zero therefore leaves no trace in the working set. After an error of the list
in [Steps on one coordinate](@ref), `x` is unchanged.

```@example block_trials
using ProximalBase

f = GroupNormL2(1.0, [1:2, 3:4])
x = AtomIterate(4, f)
r = prox_coordinate!(x, f, 1, [3.0, 4.0], 1.0)
@assert collect(r) ≈ (1 - sqrt(2) / 5) .* [3.0, 4.0]
@assert collect(activecoordinates(x)) == [1]
prox_coordinate!(x, f, 2, [0.5, 0.5], 1.0)          # the norm of u is below √2
@assert iszero(atom(x, 2)) && !isactive(x, 2)

z = AtomIterate(4, f)                               # writing the trial first
z[3] = 0.5
z[4] = 0.5
prox_coordinate!(z, f, 2, 1.0)
@assert iszero(atom(z, 2)) && isactive(z, 2)        # same values, but the write activated group 2

before = (Vector(x), collect(activecoordinates(x)))
try
    prox_coordinate!(x, GroupNormL2(1.0, [[1, 3], [2, 4]]), 1, [9.0, 9.0], 1.0)
    error("expected a DimensionMismatch")
catch e
    @assert e isa DimensionMismatch
end
@assert (Vector(x), collect(activecoordinates(x))) == before
```

Writing the trial into the internal storage of `x` (its fields) and then
calling the four-argument form is not supported. While such a write is
pending, the iterate disagrees with its working set if group ``k`` was
inactive before the write: the group is missing from it, and `nactive(x)` does
not count it, although `Vector(x)` is not zero in that group. In that case,
functions that visit only the active groups ignore the written entries. If the
next call fails one of the checks listed in [Steps on one coordinate](@ref),
nothing is written and the iterate stays in that state.

### Screening a zero coordinate

`kkt_violation(f, g, x, k)`, with `g` the gradient of the smooth part at `x`,
screens coordinate ``k`` at zero. A positive value means that the coordinate,
if it is zero, should enter the working set, and a value ``\le 0`` that zero
satisfies its optimality condition. It is not a stationarity residual of a
nonzero coordinate.

On an `AtomIterate`, ``k`` is a group. For a group penalty, the screen uses the
penalty's own dual norm and weights, and it does not read the values of `x` in
group ``k``:

| Function | Screen of group ``k`` |
|:---------|:----------------------|
| `GroupNormL2(λ0, groups, w)` | ``\lVert g_{G_k}\rVert_2 - \lambda_0 w_k`` |
| `GroupNormLinf(λ0, groups, w)` | ``\lVert g_{G_k}\rVert_1 - \lambda_0 w_k`` |
| `SparseGroupNormL2(λ1, λ2, groups, w)` | ``\lVert S_{\lambda_1}(g_{G_k})\rVert_2 - \lambda_2 w_k``, with ``S_{\lambda_1}`` soft thresholding at ``\lambda_1`` |
| `GroupMCP(λ, groups, w)`, `GroupSCAD(λ, groups, w)` | ``\lVert g_{G_k}\rVert_2 - \lambda w_k`` |
| `Scaled(f, α)` with ``\alpha > 0`` | ``\alpha`` times the screen of `f` at ``g/\alpha`` |
| `Regularize(f, ρ, c)` | the screen of `f` at ``g - \rho c``, or at ``g`` without `c` |
| a separable `f` | the largest `kkt_violation(f, g, i)` over the entries ``i`` of group ``k`` |

The weights ``w_k`` are those of the penalty, with the defaults of its
docstring: the square root of the group size, and 1 for `GroupNormLinf`.
`Scaled(f, 0)` is the function that [`Scaled`](@ref) builds at zero scale. For
a group penalty with finite levels, a finite shape parameter (`a` of `GroupMCP`
and `GroupSCAD`) and finite weights this is the zero function, whose screen on
an `AtomIterate` is `maximum(abs, view(g, atomindices(x, k)))`; a level or shape
parameter that is not finite makes `Scaled(f, 0)` throw an `ArgumentError`
when it is built. Otherwise, as
with an infinite weight or an indicator, the screen is that of the function
`Scaled` builds.

The table is not exhaustive. `IndBallGroupL0` has no screen and throws a
`MethodError`, and so do some functions without a screen of their own, such as
`NormL2`. Other functions behave differently: the array `Precompose` of a group
penalty has a screen on an `AtomIterate` (see [Entrywise scaling](@ref)), and
`MoreauEnvelope` of a function that is not separable throws an
`ArgumentError`. On an `AtomIterate`, the call throws a
`DimensionMismatch` when the groups of a group penalty differ from those of
`x`, and a `BoundsError` when ``k`` is not a group.

The screens differ in sign where it matters. At ``g = (5/8, 5/8)`` with
``\lambda_0 = 1``, the screen of `GroupNormLinf` is 0.25, so the group must
enter, while the ℓ₂ formula with weight 1 is about ``-0.116`` and would screen
the group out.

```@example screening
using ProximalBase

x2 = AtomIterate(2, [1:2])
g = [5 / 8, 5 / 8]
v = kkt_violation(GroupNormLinf(1.0, [1:2]), g, x2, 1)
@assert v == 0.25                                   # ‖g‖₁ − 1 > 0: the group must enter
@assert sqrt(sum(abs2, g)) - 1 ≈ -0.11611652351681556   # the ℓ₂ formula would screen it out
@assert kkt_violation(GroupNormL2(1.0, [1:2]), g, x2, 1) ≈ sqrt(sum(abs2, g)) - sqrt(2)
@assert kkt_violation(Scaled(GroupNormLinf(1.0, [1:2]), 2.0), g, x2, 1) == 2 * (sum(g) / 2 - 1)
@assert kkt_violation(Regularize(GroupNormL2(1.0, [1:2]), 1.0, [1.0, 1.0]), [3.0, 4.0], x2, 1) ≈
        sqrt(13) - sqrt(2)                          # the screen at (3, 4) − (1, 1)
v
```

On a `SparseIterate`, ``k`` is an entry. For a separable `f` the call returns
`kkt_violation(f, g, k)`. It checks neither ``k`` nor the length of `g` against
`x`. A group penalty screens
each entry with its group. When every entry of the group is zero in `x`, the
call returns the screen of the group in the table. Otherwise it returns the
screen of an entry at zero inside a nonzero group, `abs(g[k])`, or
`abs(g[k]) - λ1` for `SparseGroupNormL2`. An entry in no group gives
`abs(g[k])`.

```@example screening
f = GroupNormL2(1.0, [1:2, 3:4])
y = SparseIterate(4)
gs = [3.0, 4.0, 0.0, 0.0]
@assert kkt_violation(f, gs, y, 1) ≈ 5 - sqrt(2)    # group 1 of y is zero: the group screen
y[2] = 1.0
@assert kkt_violation(f, gs, y, 1) == 3.0           # entry 1 of a nonzero group: abs(gs[1])
```

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
[`prox_coordinate!`](@ref) writes a `SparseIterate` or an `AtomIterate`, the
values of the iterate are the result, and every coordinate with a nonzero
result is active. Which zero coordinates stay active is unspecified: it can
depend on the function and on the type of the argument. Call `dropzeros!` to
make the working set the support. `prox_coordinate!` follows the stricter rules
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
    a = AtomIterate(4, [1:2, 3:4])
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

| Count | `SparseIterate` | `AtomIterate` |
|:------|:----------------|:--------------|
| Coordinates | `ncoordinates(x) == length(x)` | `ncoordinates(x)`, the number of groups |
| Active coordinates, stored zeros included | `nactive(x) == nnz(x)` | `nactive(x)`, the active groups |
| Nonzeros | `count(!iszero, activevalues(x))` | entries: `sum(k -> count(!iszero, atom(x, k)), activecoordinates(x); init = 0)`; groups: `count(k -> any(!iszero, atom(x, k)), activecoordinates(x))` |
| Capacity | a slot for each of the `length(x)` entries | `length(x)` values and a slot for each group |

On a `SparseIterate`, `nnz`, `nonzeros` and `findnz` keep their SparseArrays
meaning and include stored zeros: `nnz(x) == nactive(x)`, `nonzeros(x)` is the
view `activevalues(x)`, in the order of `activecoordinates(x)`, and `findnz(x)`
returns copies sorted by index. The size of a model is the number of nonzeros,
`count(!iszero, activevalues(x))`, or `nactive(x)` after `dropzeros!`.

An `AtomIterate` has no `nnz`, `nonzeros`, `findnz` or `activevalues`; calling
them throws a `MethodError`. Loop over its active entries with
`for k in activecoordinates(x), j in atomindices(x, k)`, reading or writing
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
@assert nnz(y) == nactive(y) == length(nonzeros(y)) == 2
@assert collect(activecoordinates(y)) == [4, 2] && nonzeros(y) == [1.0, 0.0]
@assert findnz(y) == ([2, 4], [0.0, 1.0])           # sorted by index
@assert count(!iszero, activevalues(y)) == 1        # the model size
dropzeros!(y)
@assert nactive(y) == nnz(y) == 1

a = AtomIterate(6, [1:2, 3:4, 5:6])
a[1] = 1.0
a[4] = -2.0
activate!(a, 3)                                     # an active group of zeros
entries = sum(k -> count(!iszero, atom(a, k)), activecoordinates(a); init = 0)
groups = count(k -> any(!iszero, atom(a, k)), activecoordinates(a))
@assert (ncoordinates(a), nactive(a), groups, entries) == (3, 3, 2, 2)
@assert sum(a[j]^2 for k in activecoordinates(a) for j in atomindices(a, k)) == 5.0
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
