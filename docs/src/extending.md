# Extending

Subtype `SeparableFunction` and define mathematical entry hooks and parameter
metadata. The checked executable example `examples/custom_function.jl` defines
`WeightedSquare`, whose value is `sum(w .* x.^2)/2` and scalar proximal point
is `v/(1+γ*w[i])`. Generic kernels handle ordinary arrays, sparse iterates and
symmetric owners when parameters satisfy the symmetry contract.

The implementation defines `value_entry`, `prox_block`, `iszeropreserving`,
`isconvex`, `check_parameters` and `entry_parameters`. No package extension or
storage mutation hook is required. A custom public-oracle override accepting
symmetric owners must call `check_symmetric_parameters` at its boundary.

The pure compact-vector block hook can be implemented separately by defining
`prox_block!(out,f,u,k,γ)`. The function receives mathematical buffers and an
index; it does not activate coordinates or own caller iterate storage.
