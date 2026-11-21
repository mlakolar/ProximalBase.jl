# [Sparse products](@id routine-sparse-products)

The `SparseIterate` and `SymmetricSparseIterate` product kernels
described below compute using stored coordinates. The [sparse product semantics](../../numerics.md#Sparse-product-semantics)
section owns the zero and scaling behavior, and
[supported products](../../iterates.md#Supported-products) lists the
operand combinations.

## Sparse coefficients

For a strided design matrix times a vector `SparseIterate`, five-argument
`mul!` first clears the output when ``\beta=0`` or scales it by
``\beta`` otherwise. It returns after this step when ``\alpha=0``.
The accumulation visits the stored entries, skips a zero coefficient
before applying ``\alpha``, and uses the coefficient's global index to
select a column of the design. For matrix coefficients, there is a
stored-slot path and a path that looks up stored slots in a column
using a reverse map. The transpose vector product reduces over the
stored entries and masks terms for zero coefficients.
[Vector and matrix traversals](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/sparse_iterate.jl#L351-L411),
[output scaling](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/sparse_iterate.jl#L431),
and [transpose path](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/sparse_iterate.jl#L492-L515).

## Symmetric coefficients

A stored off-diagonal pair ``(r,c)`` represents both matrix positions
``(r,c)`` and ``(c,r)``. In the multiplication routes below with a
`SymmetricSparseIterate` on the left, a nonzero diagonal slot adds one
term to the output row. A nonzero off-diagonal slot adds a term to each
of rows ``r`` and ``c`` using the corresponding entries or rows of the
right operand. Zero slots are skipped. The paired indexing lets the dot
product with a dense matrix read both represented positions. See
[off-diagonal multiplicity](../../symmetric_sparse_iterate.md#The-multiplicity-of-an-off-diagonal-variable).
[Symmetric multiplication](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/symmetric_sparse_iterate.jl#L769-L821)
and [symmetric dot products](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/symmetric_sparse_iterate.jl#L689-L715).

## Tests

- [Stored-zero product tests](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/sparse_iterate.jl#L260-L294) and
  [scaling tests](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/sparse_iterate.jl#L323-L341) distinguish a
  stored zero from a nonzero coefficient whose scaling underflows.
- [Symmetric product tests](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/sparse_iterate.jl#L421-L459)
  compare vector, matrix, and dot products with their matrix equivalents.
