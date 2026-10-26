# Architecture views

The existing [overview Page](https://chatgpt.com/space/page_19d0374c8de481918344267237028f43)
and its five child Pages retain their native Markdown and Mermaid format.
`original-pages.json` preserves the exact prior source recovered from those
Pages. The `.mmd` files here mirror the revised coordinated views; the rendered
PNGs were inspected for legibility and clipping.

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
| Abstract and concrete type families | [01-types.mmd](01-types.mmd) | [PNG](rendered/01-types.png) | `src/interface.jl`, `src/functions/separable.jl`, `src/functions/norms.jl`, `src/functions/glm.jl` |
| Public operations and internal kernels | [02-oracles.mmd](02-oracles.mmd) | [PNG](rendered/02-oracles.png) | `src/interface.jl`, `src/second_order.jl`, `src/functions/separable.jl` |
| Wrapper composition and dispatch | [03-composition.mmd](03-composition.mmd) | [PNG](rendered/03-composition.png) | `src/calculus.jl`, `src/constrained.jl`, `src/composition.jl`, `src/functions/fibers.jl` |
| Iterate and coordinate storage | [04-storage-device.mmd](04-storage-device.mmd) | [PNG](rendered/04-storage-device.png) | `src/sparse_iterate.jl`, `src/atom_iterate.jl` |
| Full and partial matrix operations | [05-spectral.mmd](05-spectral.mmd) | [PNG](rendered/05-spectral.png) | `src/functions/spectral.jl`, `src/partial.jl`, `src/lapack.jl`, `src/spectralfunction.jl`, `src/functions/singular.jl` |
| Caches and optional extensions | [06-state-device.mmd](06-state-device.mmd) | [PNG](rendered/06-state-device.png) | `src/caches.jl`, `src/testing.jl`, both `ext/*.jl` files |

## Computational paths

`gradient!` has a default delegation to `value_and_gradient!`; concrete losses
may override it. GLM gradients form a predictor, call scalar loss derivatives,
and multiply by the transpose design. Hessian products call `deriv2`; GLM
quadratic models floor observation curvature and therefore differ from the
exact Hessian where the floor is active.

Array separable proximal operations loop over `prox_entry`; coordinate methods
translate storage coordinates to entry indices or call matching group kernels.
Box-constrained scalar methods clamp the inner proximal point, check the
original built-in interval intersection and compare the result with endpoints
in working precision. User-defined domains retain
the documented proper-intersection precondition.

`Conjugate` calls an inner proximal operation through Moreau decomposition.
Its combined value/prox operation uses direct supported values or Fenchel
equality; the latter does not create a standalone value solver. `Spectral`
and `SingularValue` delegate vector operations after eigen/SVD decomposition.
The opt-in partial path checks represented subspaces and does not certify
that all leading directions were captured.

Function objects may own mutable buffers even when their outer structs are
immutable. Cache reset traverses supported wrappers; bounds/normals with
immutable cached properties require reconstruction after parameter mutation.
Device dispatch uses broadcasts, reductions and matrix products. JLArrays
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
