# API coverage matrix

Every retained exported binding plus the supported qualified public extension
hooks appears below. The module binding itself is documented in
[the interface reference](interface.md). Julia 1.11+ marks the qualified names
public; they remain unexported and available qualified on Julia 1.10.

The build checks each live exported/public name against both explicit @docs
entries and this matrix. A row records the supported interface, not a promise
that every storage/type combination dispatches. Concrete guides document those
limits. Source links track master; validation evidence records the PR commit.

| Binding | Scope and signature | Reference / guide | Source | Test evidence |
|:--|:--|:--|:--|:--|
| `AbstractFunction` | abstract mathematical function | [Function abstraction](interface.md) / [guide](../function_objects.md) | [interface.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/interface.jl) | core_functions |
| `SeparableFunction` | `value_entry` + scalar `prox_block` extension | [Function abstraction](interface.md) / [guide](../function_objects.md) | [interface.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/interface.jl) | core_functions |
| `issmooth` | f or Type → Bool | [Function abstraction](interface.md) / [guide](../function_objects.md) | [interface.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/interface.jl) | core_functions |
| `isproximable` | f or Type → Bool | [Function abstraction](interface.md) / [guide](../function_objects.md) | [interface.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/interface.jl) | core_functions |
| `isseparable` | f or Type → Bool | [Function abstraction](interface.md) / [guide](../function_objects.md) | [interface.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/interface.jl) | core_functions |
| `isconvex` | f or Type → Bool | [Function abstraction](interface.md) / [guide](../function_objects.md) | [interface.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/interface.jl) | core_functions |
| `iszeropreserving` | f → Bool; instance promise | [Function abstraction](interface.md) / [guide](../function_objects.md) | [interface.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/interface.jl) | core_functions |
| `gradient` | (f,x) → dense gradient | [Mathematical operations](interface.md) / [guide](../operations.md) | [interface.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/interface.jl) | core_functions / minimal_math |
| `gradient!` | (out,f,x) → out | [Mathematical operations](interface.md) / [guide](../operations.md) | [interface.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/interface.jl) | core_functions / minimal_math |
| `value_and_gradient` | (f,x) → (value,gradient) | [Mathematical operations](interface.md) / [guide](../operations.md) | [interface.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/interface.jl) | core_functions / minimal_math |
| `value_and_gradient!` | (out,f,x) → f(x) | [Mathematical operations](interface.md) / [guide](../operations.md) | [interface.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/interface.jl) | core_functions / minimal_math |
| `prox` | (f,x,γ=1) → new point | [Mathematical operations](interface.md) / [guide](../operations.md) | [interface.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/interface.jl) | core_functions / minimal_math |
| `prox!` | (out,f,x,γ=1) → out | [Mathematical operations](interface.md) / [guide](../operations.md) | [interface.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/interface.jl) | core_functions / minimal_math |
| `value_and_prox!` | (out,f,x,γ=1) → f(stored out) | [Mathematical operations](interface.md) / [guide](../operations.md) | [interface.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/interface.jl) | core_functions / minimal_math |
| `prox_block` | (f,v,i,γ) scalar; vector allocates/delegates | [Mathematical operations](interface.md) / [guide](../operations.md) | [interface.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/interface.jl) | core_functions / minimal_math |
| `prox_block!` | (out,f,u,k,γ) → out; custom vector extension | [Mathematical operations](interface.md) / [guide](../operations.md) | [interface.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/interface.jl) | core_functions / minimal_math |
| `kkt_violation` | (f,grad,i) or (f,grad,x,k); zero screen | [Mathematical operations](interface.md) / [guide](../operations.md) | [interface.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/interface.jl) | core_functions / minimal_math |
| `restrict` | (f,S); supported family slices | [Mathematical operations](interface.md) / [guide](../operations.md) | [interface.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/interface.jl) | core_functions / minimal_math |
| `lipschitz` | (f) → gradient bound or ArgumentError | [Mathematical operations](interface.md) / [guide](../operations.md) | [interface.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/interface.jl) | core_functions / minimal_math |
| `value_entry` | (f,v,i); `NormL1`/custom separable scalar value | [Entry hook](interface.md) / [guide](../extending.md) | [functions/separable.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/separable.jl) | core_functions |
| `reset_cache!` | (f) → f; same-shape data invalidation | [Cache lifecycle](interface.md) / [guide](../numerical_usage.md) | [caches.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/caches.jl) | cache_and_blocks |
| `ZeroFunction` | () | [Smooth functions](functions.md) / [guide](../smooth.md) | [functions/smooth.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/smooth.jl) | core_functions / ridge_precision |
| `LinearFunction` | (c,d=0) | [Smooth functions](functions.md) / [guide](../smooth.md) | [functions/smooth.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/smooth.jl) | core_functions / ridge_precision |
| `SqrNormL2` | (λ=1,y=nothing) | [Smooth functions](functions.md) / [guide](../smooth.md) | [functions/smooth.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/smooth.jl) | core_functions / ridge_precision |
| `QuadraticFunction` | (A,b=nothing,c=0) | [Smooth functions](functions.md) / [guide](../smooth.md) | [functions/smooth.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/smooth.jl) | core_functions / ridge_precision |
| `LeastSquaresLoss` | (Y,X); half residual sum of squares | [Least squares](functions.md) / [guide](../smooth.md) | [functions/leastsquares.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/leastsquares.jl) | core_functions / minimal_math |
| `NormL1` | (λ0=1,w=nothing) | [Penalty](functions.md) / [guide](../penalties.md) | [functions/separable.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/separable.jl) | core_functions / minimal_math |
| `Scaled` | (f,α), α*f, f*α; may simplify at zero | [Wrapper](calculus.md) / [guide](../calculus.md) | [scaled.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/scaled.jl) | minimal_math / ridge_precision |
| `SparseIterate` | ([T,]n[,m]) or dense/sparse array | [Storage](iterates.md) / [guide](../iterates.md) | [sparse_iterate.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/sparse_iterate.jl) | iterate_invariants / iterate_queries |
| `SymmetricSparseIterate` | ([T,]p) or symmetric matrix | [Storage](iterates.md) / [guide](../iterates.md) | [sparse_iterate.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/sparse_iterate.jl) | iterate_invariants / iterate_queries |
| `ncoordinates` | (x) → total coordinate count | [Storage](iterates.md) / [guide](../iterates.md) | [sparse_iterate.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/sparse_iterate.jl) | iterate_invariants / iterate_queries |
| `nactive` | (x) → stored coordinate/group count | [Storage](iterates.md) / [guide](../iterates.md) | [sparse_iterate.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/sparse_iterate.jl) | iterate_invariants / iterate_queries |
| `coordinates` | (sparse/owner) → writable full coordinate vector | [Storage](iterates.md) / [guide](../iterates.md) | [sparse_iterate.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/sparse_iterate.jl) | iterate_invariants / iterate_queries |
| `activecoordinates` | (x) → read-only active IDs | [Storage](iterates.md) / [guide](../iterates.md) | [sparse_iterate.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/sparse_iterate.jl) | iterate_invariants / iterate_queries |
| `activevalues` | (sparse/owner) → writable packed values | [Storage](iterates.md) / [guide](../iterates.md) | [sparse_iterate.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/sparse_iterate.jl) | iterate_invariants / iterate_queries |
| `isactive` | (x,k) → Bool | [Storage](iterates.md) / [guide](../iterates.md) | [sparse_iterate.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/sparse_iterate.jl) | iterate_invariants / iterate_queries |
| `activate!` | (x,k) → x | [Storage](iterates.md) / [guide](../iterates.md) | [sparse_iterate.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/sparse_iterate.jl) | iterate_invariants / iterate_queries |
| `matrixview` | (owner) → read-only live matrix | [Storage](iterates.md) / [guide](../iterates.md) | [sparse_iterate.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/sparse_iterate.jl) | iterate_invariants / iterate_queries |
| `dropzeros!` | (x) → x; compact zeros | [Storage](iterates.md) / [guide](../iterates.md) | [sparse_iterate.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/sparse_iterate.jl) | iterate_invariants / iterate_queries |
| `nnz` | re-export for standard sparse arrays, no iterate method | [Storage](iterates.md) / [guide](../iterates.md) | [sparse_iterate.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/sparse_iterate.jl) | iterate_invariants / iterate_queries |
| `BlockIterate` | ([T,]p,groups/K) or (vector,groups/K) | [Grouped storage](iterates.md) / [guide](../block_iterate.md) | [block_iterate.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/block_iterate.jl) | block_copy / iterate_queries |
| `blockview` | (x,k) → writable ordered entry view | [Grouped storage](iterates.md) / [guide](../block_iterate.md) | [block_iterate.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/block_iterate.jl) | block_copy / iterate_queries |
| `blockindices` | (x,k) → read-only ordered global IDs | [Grouped storage](iterates.md) / [guide](../block_iterate.md) | [block_iterate.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/block_iterate.jl) | block_copy / iterate_queries |
| `shrink` | (v,c); scalar soft threshold | [Utilities](iterates.md) / [guide](../iterates.md) | [utils.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/utils.jl) | sparse_iterate_basics / minimal_math |
| `norm_diff` | (A,B,p=2); p=2 or Inf | [Utilities](iterates.md) / [guide](../iterates.md) | [utils.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/utils.jl) | sparse_iterate_basics / minimal_math |
| `tril_index` | (p,r,c); valid lower-triangle pair → k | [Utilities](iterates.md) / [guide](../iterates.md) | [utils.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/utils.jl) | sparse_iterate_basics / minimal_math |
| `tril_cartesian` | (p,k); valid independent ID → (r,c) | [Utilities](iterates.md) / [guide](../iterates.md) | [utils.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/utils.jl) | sparse_iterate_basics / minimal_math |
| `opnorm2` | (X); squared spectral norm estimate, not bound | [Norm estimate](interface.md) / [guide](../operations.md) | [utils.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/utils.jl) | minimal_math |
| `ProximalBase.check_parameters` | (f,x); default nothing, custom shape validation | [Qualified public extension](interface.md) / [guide](../extending.md) | [interface.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/interface.jl) | core_functions |
| `ProximalBase.entry_parameters` | (f); tuple metadata, custom separable default undeclared | [Qualified public extension](interface.md) / [guide](../extending.md) | [interface.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/interface.jl) | core_functions |
| `ProximalBase.check_symmetric_parameters` | (f,x) → true or throws; implementation symmetric_parameters.jl | [Qualified public extension](interface.md) / [guide](../extending.md) | [interface.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/interface.jl) | core_functions |
| `ProximalBase.samegroups` | (a::`BlockIterate`,b::`BlockIterate`) → Bool | [Qualified public layout](iterates.md) / [guide](../block_iterate.md) | [block_iterate.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/block_iterate.jl) | iterate_queries |
| `ProximalBase.ScalarLoss` | abstract scalar observation-loss interface | [Qualified scalar loss](functions.md) / [guide](../smooth.md) | [functions/scalar_loss.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/scalar_loss.jl) | core_functions |
| `ProximalBase.Squared` | (); (η,y) → (η-y)²/2 | [Qualified scalar loss](functions.md) / [guide](../smooth.md) | [functions/scalar_loss.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/scalar_loss.jl) | core_functions |
| `ProximalBase.deriv` | (ℓ,η,y) → scalar derivative | [Qualified scalar loss](functions.md) / [guide](../smooth.md) | [functions/scalar_loss.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/scalar_loss.jl) | core_functions |
| `ProximalBase.deriv2` | (ℓ,η,y) → scalar second derivative | [Qualified scalar loss](functions.md) / [guide](../smooth.md) | [functions/scalar_loss.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/scalar_loss.jl) | core_functions |
| `ProximalBase.curvature` | (ℓ) → global second-derivative bound | [Qualified scalar loss](functions.md) / [guide](../smooth.md) | [functions/scalar_loss.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/scalar_loss.jl) | core_functions |
| `ProximalBase.value_and_deriv` | (ℓ,η,y) → (value,derivative) | [Qualified scalar loss](functions.md) / [guide](../smooth.md) | [functions/scalar_loss.jl](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/scalar_loss.jl) | core_functions |

The eight [executable examples](../examples/lasso.md) complement these suites:
lasso values/gradients/prox, PG stationarity, scalar coordinate updates, sparse
working sets, ordered blocks, symmetric geometry, custom separable hooks and
compact-vector block extension. The extension examples validate their own
checked contract; the allocating generic wrapper alone does not validate a
custom block implementation. Test names identify evidence domains, not complete
coverage of every parameter endpoint.
