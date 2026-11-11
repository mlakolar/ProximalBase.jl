# Executable examples

Each script uses only ProximalBase and the Julia standard library, deterministic
small CPU fixtures and explicit assertions. Run one from the repository root:

```sh
julia --project=. examples/lasso.jl
```

Or run every script in a fresh process with `julia --project=. docs/run_examples.jl`.
The strict docs build runs that isolation check, generates Literate pages and
executes their code again in independent page sandboxes.

| Script | Teaches / independent checks |
|:--|:--|
| lasso.jl | Mean least-squares scaling, lasso value/gradient/soft threshold, direct loss prox solve, matrix response |
| proximal_gradient.jl | Caller iteration, descent, orthogonal-design closed form and explicit stationarity |
| coordinate_operations.jl | Scalar trials, residual/curvature updates and KKT conditions |
| sparse_working_sets.jl | Explicit active zeros, state/value copying, KKT selection and restriction |
| block_storage.jl | Ordered noncontiguous groups, ownership, active-zero groups and global penalty indices |
| symmetric_storage.jl | Independent IDs, matrix view, Frobenius multiplicity, import/state-copy distinctions |
| custom_function.jl | Complete separable hooks, parameter metadata, dense/sparse/owner checks and preflight failure |
| block_prox_extension.jl | Compact-vector extension, independent buffers, exact alias and rejected partial overlap |

These are mathematical building blocks and educational caller loops. They do
not define a general solver API. Historical full-library examples remain at
`eff21b6d7d33c622313122f46c6ee71693c101ee` / `legacy/full`.
