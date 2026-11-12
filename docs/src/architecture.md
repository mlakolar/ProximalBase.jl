# [Package structure](@id architecture)

The implementation separates mathematical function dispatch from iterate storage.
An algorithm asks a function for a numerical operation and decides when to write
the resulting values to an iterate. Private scratch/decomposition state belongs
to functions; residuals, stopping policy and working sets belong to the caller.

## Source and ownership map

| Source | Responsibility |
|:--|:--|
| `src/ProximalBase.jl` | Module, exports, qualified public names and include order |
| `src/compat.jl` | Version-dependent Base/SparseArrays compatibility types |
| `src/utils.jl` | Shape/type checks, scratch buffers, stored-value scans, scaled-value protocol hooks and triangular indices |
| `src/numerical_kernels.jl` | Shared numerical kernels: soft thresholding, exponent-scaled products, compensated and power-of-two sums, underflow and squared-residual recovery, norms and Lanczos norm estimate |
| `src/sparse_iterate.jl` | `SparseIterate`, symmetric owner, active maps, views, copies, products and alias integration |
| `src/block_iterate.jl` | Ordered partitions, dense values, group activity, block views and layout comparison |
| `src/interface.jl` | `AbstractFunction`, traits, generic allocating/mutating interfaces and coordinate KKT dispatch |
| `src/functions/zero.jl` | Zero function and its value/gradient/proximal operations |
| `src/functions/linear.jl` | Linear functions and their value/gradient/proximal operations |
| `src/functions/sqrnorml2.jl` | Squared Euclidean norm and centered ridge operations |
| `src/functions/quadratic.jl` | Quadratic function and cached eigendecomposition-based proximal map |
| `src/functions/leastsquares.jl` | Least-squares residual scratch and lazy thin SVD prox |
| `src/functions/scalar_loss.jl` | Qualified `ScalarLoss`/`Squared` and scalar derivative hooks |
| `src/functions/separable.jl` | Generic separable entry kernels and array/sparse traversal |
| `src/functions/norml1.jl` | Weighted ℓ₁ family, value and specialized proximal operations |
| `src/functions/scaled.jl` | Nonnegative scaling and exceptional exponent retention |
| `src/array_steps.jl` | Diagonal steps for supported separable/zero array methods |
| `src/functions/scaled_array.jl` | Diagonal array-step methods and scaling recovery for `Scaled` functions |
| `src/symmetric_parameters.jl` | Per-entry metadata, checked pair domains, explicit owner oracle integration |
| `src/block_prox.jl` | Allocating compact-vector wrapper; custom mutating hook remains external |
| `src/caches.jl` | Public cache invalidation, recursing through `Scaled` |
| `src/precompile.jl` | Small retained precompile workloads |

The include order puts storage before the mathematical interface because
explicit owner methods need the owner type. Generic interface methods then
combine with concrete implementations and final symmetric/array-step routes.
There is no automatic device registration or removed-family plugin layer.

## Representative call paths

| Public call | Principal retained path |
|:--|:--|
| `prox!(y,NormL1(λ,w),x,γ)` | Checked separable boundary → dense/stored traversal → scalar `prox_block` → shrink |
| `f(x)` for custom `SeparableFunction` | Parameter/symmetric validation → entry values; skip inactive zeros only under declared promise |
| `gradient!(g,LeastSquaresLoss(Y,X),β)` | Checked output → residual Xβ−Y → transpose-design product |
| `prox!(y,QuadraticFunction(A,b),x,γ)` | Shape/type preflight → cached eigenbasis → diagonal inverse factors → output conversion |
| `prox_block(f,u::AbstractVector,k,γ)` | Allocate plain floating Vector → user `prox_block!` |
| `prox!(owner,g,V,γ)` | Symmetric argument where needed → parameter-pair preflight → each independent coordinate written once |
| `kkt_violation(g,grad,block,k)` | Retained scalar screen at each global member → maximum |

Traits describe mathematical properties and capabilities; Julia concrete
methods determine supported signatures. `isseparable=true` alone neither
supplies entry hooks nor turns a full-vector function into a compact block
oracle. [Dispatch](developer/oracle_dispatch.md) describes extension boundaries.

## Public contracts and implementation details

Stable public contracts describe coordinate meanings, active-zero lifecycle,
return values, alias restrictions and checked failure behavior. Fields, private
wrapper names, scratch layouts, loop order/crossover thresholds and the layout
cache are implementation details. They explain why the current implementation
has its behavior; consumers must not depend on their representation.

Historical architecture SVGs included removed families, second-order layers
and devices. They are retained in the immutable historical snapshot rather
than displayed as current diagrams. The source map here matches the retained
include graph. Current contract documents are authored in `docs/design` and
rendered into [Developer Documentation](developer/index.md) at build time.
