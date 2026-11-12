# Pure block proximal contract

This page documents the current **mathematical extension hooks** for individual
entries and compact blocks. These hooks compute numerical results. They do
**not** update iterate storage, activate coordinates, or maintain solver caches.
The older, broader design is preserved in the [historical #844 contract](https://github.com/mlakolar/ProximalBase.jl/blob/eff21b6d7d33c622313122f46c6ee71693c101ee/docs/design/block-prox-0.6.md).

## Scalar entries

For an entry-separable function
``g(x)=\sum_{i=1}^{p}g_i(x_i)``, the call

```julia
prox_block(g, v, i, γ)
```

computes the scalar proximal point at a trial value `v`:

```math
\operatorname{prox}_{\gamma g_i}(v)
=\operatorname*{argmin}_z\left\{g_i(z)+\frac{(z-v)^2}{2\gamma}\right\}.
```

The integer `i` is the **global array-entry index** used to select the
function's parameters. It is not a packed active slot or a `BlockIterate`
group ID. For example, `NormL1` uses soft thresholding:

```julia
g = NormL1(0.5)
v = 2.0
result = prox_block(g, v, 1, 1.0)  # 1.5; does not write to an iterate
```

Current scalar `prox_block` methods include `NormL1`, `SqrNormL2`,
`LinearFunction`, `ZeroFunction`, and supported `Scaled` functions.
A custom `SeparableFunction` implements `prox_block(f, v, i, γ)`
and `value_entry(f, v, i)`; generic kernels can then provide full-array
function values and proximal updates.

Mathematical separability alone does **not** create those Julia methods.
For example, the existence of `isseparable(f)` does not imply
that `value_entry(f, v, i)` is implemented.

### Symmetric matrix coordinates

A `SymmetricSparseIterate` stores one independent variable for each
lower-triangular matrix pair. For a coordinate ID `k`, recover its
matrix indices and corresponding **full-matrix entry index**:

```julia
r, c = tril_cartesian(p, k)
i = (c - 1) * p + r
```

Use `i`, not `k`, for scalar function parameters. Validate the
original function's mirrored parameter pairs first, using
`ProximalBase.check_symmetric_parameters(f, x)` when appropriate:
a raw scalar hook has no owner object to inspect. Coordinate curvature
and gradients must respect the represented matrix's Frobenius geometry.

## Compact-vector block extensions

A custom group function ``h_k:\mathbb R^{m_k}\to\mathbb R`` may implement

```julia
prox_block!(out, f, u, k, γ)
```

where `u` is the **trial vector** for the ordered entries of group
`k`. Mathematically, the result is

```math
\operatorname{prox}_{\gamma h_k}(u)
=\operatorname*{argmin}_z
\left\{h_k(z)+\frac{\|z-u\|_2^2}{2\gamma}\right\}.
```

The method must **write its result into `out`** and return the identical
`out` object. It does not write to the `BlockIterate`. The generic
allocating `prox_block(f, u, k, γ)` constructs a floating-point
`Vector` of `length(u)` and delegates to this hook; it does not
validate the custom function's group layout or parameters.

### Required validation

A conforming custom block implementation checks predictable errors
**before modifying `out`**:

- `out` and `u` are one-based real vectors.
- `k` identifies a valid group, and the compact buffer lengths agree.
- Group parameters and the step `γ` are valid for the mathematical domain.
- Exact input/output identity (`out === u`) is permitted. Other overlapping
  views must be rejected.
- The returned object is exactly `out`.

Failures during arbitrary element reads, conversions or user arithmetic may
still leave a partially written numerical output; document any narrower
failure guarantees. Built-in extreme-range recovery is **not** automatically
inherited by custom Julia methods.

### Separating mathematical work from storage writes

**Caller precondition:** use an output buffer independent of iterate
storage and solver/model caches. The public views `coordinates(x)`,
`activevalues(x)` and `blockview(x, k)` are live aliases, even when
their parent is an ordinary vector.

A solver's sequence is:

1. Check that the custom function's ordered groups match
   `blockindices(x, k)`.
2. Construct the numerical trial `u`, and call
   `prox_block!(out, f, u, k, γ)` using independent scratch.
3. Write with `copyto!(blockview(x, k), out)`, then update residuals
   and other solver caches.

The write is what changes iterate activity. A zero numerical result that
has not been written leaves an absent block inactive. For an entry-separable
function, use scalar `prox_block` with each **global entry ID**, not the
group ID.

## Operations that are not synthesized automatically

A compact block hook alone does **not** supply `f(x)`, a full-array
`prox!`, `kkt_violation`, `restrict`, capability traits, scaling of
compact block methods, symmetric block geometry, or storage mutation.
Implement each separately if the mathematics and checked Julia dispatch
support it.

`ProximalBase.samegroups` compares two `BlockIterate` partitions;
it is not a generic query for the parameters of a custom penalty.

The earlier storage-aware names `prox_entry!`, `prox_coordinate!`
and `cdprox!` are **not** compatibility aliases. They appear here only
to clarify the historical migration.

## Evidence and examples

The Julia files `examples/custom_function.jl` and
`examples/block_prox_extension.jl` contain complete extensions.
The documentation executes those examples using Literate.
`test/core_functions.jl` covers the generic extension and allocating
block result; the block example also checks buffer validation and
overlap. The archived #844 design includes older group kernels and
downstream plans, but it is not a requirement to reintroduce them.
