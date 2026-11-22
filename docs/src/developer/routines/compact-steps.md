# [Scalar, array and compact proximal steps](@id routine-compact-steps)

A proximal step uses a trial value and a step size to compute an
updated value. The array and compact forms select their parameters
using different indices. The [array-step guide](../../operations.md#Different-steps-for-different-entries)
and [compact block contract](../../iterate_oracles.md#block-proximal-oracle)
own the supported routes.

## Per-entry dispatch

In the generic separable path, the scalar and array-step methods
check the public arguments and call the same traversal. A dense
traversal calls `prox_block(f, v, i, step)` with the global linear
entry index `i`. The step accessor returns the common scalar or
reads the array element at `i`. The scalar hook uses that same
index to select function parameters. See the
[scalar extension contract](../contracts/block-prox-0.6.md#Scalar-entries).
[Scalar entry points](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/separable.jl#L65-L75),
[array entry points](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/array_steps.jl#L29-L40),
[step lookup](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/utils.jl#L49-L50),
and [dense traversal](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/separable.jl#L146-L155).

## Compact group dispatch

`prox_block!(out, f, u, k, γ)` takes the ordered trial for group
`k` in a compact vector. The five built-in group functions share
checks for group bounds, real one-based vectors, length, buffer
overlap, and the step. They use the group's parameters and run
their proximal kernels over compact indices `1:m`. The
`GroupNormL2` route applies Euclidean shrinkage. The
`SparseGroupNormL2` route computes an entry threshold for the
sparse group kernel. The `GroupNormLinf` route chooses the output
as its threshold workspace when the buffer predicate permits it,
and otherwise uses the function's scratch space. The `GroupMCP`
route applies its proximal map using the selected group weight, and
the `GroupSCAD` route does the same for its radial map.
[Common compact validation](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/groupnorml2.jl#L120-L128),
[GroupNormL2 kernel call](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/groupnorml2.jl#L236-L240),
[SparseGroupNormL2 kernel call](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/sparsegroupnorml2.jl#L223-L228),
[GroupNormLinf buffer selection](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/groupnormlinf.jl#L406-L416),
[GroupMCP compact dispatch](https://github.com/mlakolar/ProximalBase.jl/blob/ae693e0af9c8a09e08a90de98d0e6b23b9a518cb/src/functions/groupmcp.jl#L474-L490),
and [GroupSCAD compact dispatch](https://github.com/mlakolar/ProximalBase.jl/blob/405c212dc69e99b1f34cec12825fd53e93cacea8/src/functions/groupscad.jl#L486-L502).

The allocating wrapper creates a floating vector of the trial length
and delegates to the compact hook. To use the result with an iterate,
compute in independent scratch and then apply a storage writer such
as `copyto!(blockview(x, k), out)`.
[Allocating wrapper](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/block_prox.jl#L2-L5)
and [buffer ownership rule](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/interface.jl#L413-L431).

## Tests

- [Entry-step test](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/core_functions.jl#L45-L47) checks the
  `NormL1` array-step result against the entry formula.
- [Compact aliasing tests](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/groupnorml2.jl#L174-L184) check
  exact in-place use and rejection of partial overlap.
  [Compact storage tests](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/groupnorml2.jl#L388-L412)
  exercise the separate calculation and iterate write.
- [GroupMCP compact tests](https://github.com/mlakolar/ProximalBase.jl/blob/ae693e0af9c8a09e08a90de98d0e6b23b9a518cb/test/groupmcp.jl#L498-L509)
  compare the compact result with the full-array result and check exact in-place use.
- [GroupSCAD compact tests](https://github.com/mlakolar/ProximalBase.jl/blob/405c212dc69e99b1f34cec12825fd53e93cacea8/test/groupscad.jl#L903-L928)
  compare the compact result in place (`out === u`) with the result out of place, bit for bit.
