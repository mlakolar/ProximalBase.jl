# Architecture views

The existing [overview Page](https://chatgpt.com/space/page_19d0374c8de481918344267237028f43)
and its five child Pages retain their native Markdown and Mermaid format.
The [recovered original Page source](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/architecture/original-pages.json)
is retained with the initial audit edition. The `.mmd` files here mirror the
coordinated views. Rendered PNGs and raw execution receipts are local artifacts; the linked initial renders remain
available at their immutable audit commit.

## Legend

- Thin arrow `-->`: inheritance, from subtype to parent, in type/storage views.
- Circle end `--o`: composition, from owner to the object it contains.
- Dotted arrow `-.->`: dispatch or selection; the edge label states the boundary.
- Thick arrow `==>`: runtime call, transformation or calculation.
- The `same binding` edge in the type view is an alias, not inheritance.

Boxes containing several names are conceptual groups of related concrete
types, not additional Julia types. Their members and the operation/field
relationships shown are verified from source. Solver boxes summarize callers;
ProximalBase does not supply those optimization loops. The complete declaration
inventory is in [type-inventory.md](../type-inventory.md).

| View | Source | Render | Representative implementation |
| --- | --- | --- | --- |
| Abstract and concrete type families | [01-types.mmd](01-types.mmd) | [PNG](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/architecture/rendered/01-types.png) | `src/interface.jl`, `src/functions/separable.jl`, `src/functions/norms.jl`, `src/functions/glm.jl` |
| Public operations and internal kernels | [02-oracles.mmd](02-oracles.mmd) | [PNG](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/architecture/rendered/02-oracles.png) | `src/interface.jl`, `src/second_order.jl`, `src/functions/separable.jl` |
| Wrapper composition and dispatch | [03-composition.mmd](03-composition.mmd) | [PNG](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/architecture/rendered/03-composition.png) | `src/calculus.jl`, `src/constrained.jl`, `src/composition.jl`, `src/functions/fibers.jl` |
| Iterate and coordinate storage | [04-storage-device.mmd](04-storage-device.mmd) | [PNG](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/architecture/rendered/04-storage-device.png) | `src/sparse_iterate.jl`, `src/atom_iterate.jl` |
| Full and partial matrix operations | [05-spectral.mmd](05-spectral.mmd) | [PNG](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/architecture/rendered/05-spectral.png) | `src/functions/spectral.jl`, `src/partial.jl`, `src/lapack.jl`, `src/spectralfunction.jl`, `src/functions/singular.jl` |
| Caches and optional extensions | [06-state-device.mmd](06-state-device.mmd) | [PNG](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/architecture/rendered/06-state-device.png) | `src/caches.jl`, `src/testing.jl`, both `ext/*.jl` files |

## Computational paths

`gradient!` has a default delegation to `value_and_gradient!`; concrete losses
may override it. GLM gradients form a predictor, call scalar loss derivatives,
and multiply by the transpose design. Hessian products call `deriv2`; GLM
quadratic models floor observation curvature and therefore differ from the
exact Hessian where the floor is active.

Array separable proximal operations loop over `prox_entry`; coordinate methods
translate storage coordinates to entry indices or call matching group kernels.
Box-constrained scalar methods clamp the inner proximal point. The caller must
ensure a nonempty intersection between the function domain and the box; there
is no separate symbolic domain-inference layer.

`Conjugate` calls an inner proximal operation through Moreau decomposition.
Its combined value/prox operation uses direct supported values or Fenchel
equality; the latter does not create a standalone value solver. `Spectral`
and `SingularValue` delegate vector operations after eigen/SVD decomposition.
The opt-in partial path checks represented subspaces and does not certify
that all leading directions were captured.

Function objects may own mutable buffers even when their outer structs are
immutable. Cache reset traverses supported wrappers; bounds/normals with
immutable cached properties require reconstruction after parameter mutation.
Device dispatch uses broadcasts, reductions and matrix products. Grouped
`prox!` skips function-value reductions; `value_and_prox!` requests them through
the same proximal kernel. A zero group term skips grouped norm calculations. JLArrays
validation is distinct from physical-device compiler/performance validation.

## Recreation

The inspected rendering used `@mermaid-js/mermaid-cli` 12.0.0 outside the
package environment, with the installed Chrome executable selected through
`puppeteer.json`. The two corrected views were rerendered with the same size
and white background:

```sh
mmdc -p puppeteer.json -i VIEW.mmd -o VIEW.png --size 1300 -b white
```

The Puppeteer configuration sets `executablePath` to the installed Chrome
binary and `args` to `["--no-sandbox"]`. No renderer dependency is added to
ProximalBase. The durable audit validation folder retains the renderer's npm
lockfile and configuration for recreation.
