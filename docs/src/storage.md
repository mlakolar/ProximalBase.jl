# Storage

`SparseIterate` stores array entries and an active-coordinate set. Assigning a
nonzero entry activates its coordinate; assigning zero to an active entry
preserves deliberate activity until `dropzeros!`. `coordinates`,
`activecoordinates`, `activevalues`, `ncoordinates`, `nactive`, `isactive` and
`activate!` expose working-set storage independently of mathematical functions.

`BlockIterate(x,groups)` stores whole blocks with arbitrary ordered index lists,
including noncontiguous groups such as `[[3,1],[4,2]]`. `blockindices(x,k)`
returns the group's indices and `blockview(x,k)` aliases the corresponding
values. Writes through views follow activation and zero rules. Layouts must
satisfy the constructors' coverage and disjointness checks.

`SymmetricSparseIterate` owns independent lower-triangular coordinates and is
not an AbstractMatrix. Use `coordinates(x)` for independent coordinates and
`matrixview(x)` for an aliasing matrix view. A value at an off-diagonal
coordinate appears twice in the represented matrix. `Matrix(x)` and `copy(x)`
produce independent data; active-zero state is retained by storage copies.

The storage types retain their array access, copying, broadcasting, scalar
scaling and supported matrix products. Scalar `prox_block` and custom compact
block proximal functions return mathematical results; callers choose when and
how to write those results through storage views.
