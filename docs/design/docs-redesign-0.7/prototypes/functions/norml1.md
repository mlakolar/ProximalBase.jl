# Sparsity penalties (prototype: `NormL1` entry)

*In the implementation this entry opens `penalties.md`, followed by `ElasticNet`,
`NormL1Nonneg`, `MCP`, `SCAD` and `NormL0` in the same format.*

![Scalar proximal maps at step one: soft, hard and firm thresholding.](../../figures/thresholding.svg)

## [Weighted ℓ₁ norm: `NormL1`](@id entry-norml1)

The lasso penalty. It encourages coefficients that are exactly zero.

```math
g(x)=\lambda\sum_{i}w_i\,|x_i|,\qquad \lambda\ge 0,\ w_i\ge 0.
```

For a matrix the sum runs over all entries. A weight of zero leaves its entry
unpenalized. With finite positive weights and a finite ``\lambda>0``, ``g`` is a norm.

| Constructor | Meaning |
|:--|:--|
| `NormL1(λ = 1)` | all weights equal to one |
| `NormL1(λ, w)` | one weight per entry of ``x``, matched by linear index |

`λ` and the weights must be nonnegative and not `NaN`; `Inf` is allowed. This is
checked when the object is constructed. The weight array is kept by reference and is
not checked again: if you change it later, it must stay valid, and the function
changes with it.

```@example entry-norml1
using ProximalBase

g = NormL1(0.5)
x = [2.0, -1.0, 0.25]
g(x)
```

```@example entry-norml1
prox(g, x, 1.0)
```

```@example entry-norml1
h = NormL1(0.5, [1.0, 0.0, 2.0])   # the second entry is not penalized
prox(h, x, 1.0)
```

### Proximal operator: soft thresholding

For finite entries, finite parameters and a finite step ``\gamma>0``,

```math
\operatorname{prox}_{\gamma g}(v)_i=\operatorname{sign}(v_i)\,
\max\bigl(|v_i|-\gamma\lambda w_i,\ 0\bigr).
```

Each entry moves toward zero by its own threshold ``\gamma\lambda w_i`` and stops at
zero. Infinite levels or weights, zero steps and non-finite entries follow the
[endpoint rules](@ref spec-norml1-endpoints).

| Operation | Call | Supported |
|:--|:--|:--|
| value | `g(x)` | ✓ |
| proximal step | `prox(g, x, γ)`, `prox!(out, g, x, γ)` | ✓ a scalar step, or an array with one step per entry in linear order |
| value at the new point | `value_and_prox!(out, g, x, γ)` | ✓ returns `g(out)`, the documented value; the computed value can differ from `g(out)` in the last digit ([recorded discrepancy](../../audit.md#defects-and-gaps-recorded-separately)) |
| one coordinate | `prox_block(g, xᵢ, i, γ)` | ✓ |
| optimality score | `kkt_violation(g, v, i)`, `(g, v, x, i)`, `(g, v, x)` | ✓ |
| restriction | `restrict(g, S)` | ✓ |
| gradient | `gradient(g, x)` | — not differentiable at zero |

`prox!(x, g, x, γ)` may overwrite its input; an output that partly overlaps the input
is not supported.

### Inputs and storage

✓ vectors, matrices and strided views of them ·
✓ `SparseIterate`: the value, and prox into another `SparseIterate`, visit stored entries only ·
✓ `BlockIterate` ·
◐ `SymmetricSparseIterate`: the step must be a scalar and the optimality score is per
coordinate; if you supply weights, give ``p^2`` of them in linear order with exactly
equal mirrored entries ([symmetric parameters](@ref symmetric-parameter-guide)) ·
◐ `Symmetric` matrices: `prox` returns a dense `Matrix`.
Element types: `Float64` or `Float32`, the same type for data, parameters, step and
output ([precision rules](@ref numerical-usage)).

### Good to know

- Steps are not checked. A negative or `NaN` step is unsupported, and the result of
  such a call has no mathematical meaning ([shared step rule](@ref numerics)).
- Off-diagonal entries of a `SymmetricSparseIterate` count twice in the value,
  because each stands for two matrix entries.
- An infinite weight fixes a finite entry at zero. For infinite or `NaN` inputs, and
  for the overflow limitation of the threshold, see the
  [guarantee page](@ref spec-norml1-limitations).

**See also:** [`ElasticNet`](@ref) (adds a ridge term),
[`NormL1Nonneg`](@ref) (nonnegative coefficients),
[`MCP`](@ref) and [`SCAD`](@ref) (less shrinkage of large coefficients),
[`IndBallL1`](@ref) (an ℓ₁ constraint instead of a penalty) ·
[guarantees](@ref spec-norml1) · [API](@ref NormL1)
