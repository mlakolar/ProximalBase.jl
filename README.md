# ProximalBase.jl

Mathematical function objects and sparse working-set storage for optimization
problems `min_x f(x)+g(x)`, with smooth loss `f` and penalty `g`.

Read the [manual](https://mlakolar.github.io/ProximalBase.jl/dev/), starting with
[getting started](docs/src/getting_started.md), or browse the
[complete retained function library](docs/src/function_library.md). ProximalBase
supplies values, gradients, proximal maps and storage; ProximalSolvers implements
optimization algorithms.

The minimal core retains `SparseIterate`, `BlockIterate`,
`SymmetricSparseIterate`; values, gradients and proximal interfaces;
`LeastSquaresLoss`, `SqrNormL2`, `QuadraticFunction`, `ProximalBase.Squared`;
`NormL1`, `Scaled`, `ZeroFunction` and `LinearFunction`.

```julia
using ProximalBase
X = [1.0 2; 3 1; 0 2]
y = [1.0, 2, 3]
loss = (1/size(X,1)) * LeastSquaresLoss(y,X)
penalty = NormL1(0.2)
β = [0.3, -0.4]
loss(β) + penalty(β)
gradient(loss,β)
prox(penalty,β,0.1)
```

`prox!(out,f,x,γ)` writes the mathematical proximal point. Scalar
`prox_block(f,v,i,γ)` and custom compact-vector `prox_block!` hooks are pure;
callers write results through iterate coordinates or block views.

Run `julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'`.
The test runner validates its complete supported inventory. Current CI runs
Linux on current Julia 1.x. See [CONTRIBUTING.md](CONTRIBUTING.md) for focused
checks, docs and benchmarks.

This is a development reduction, with no registration or release implied.
The full post-#848 implementation is preserved at commit
`eff21b6d7d33c622313122f46c6ee71693c101ee`, tree
`c76123a0109c0a0f683cb75085371696dffe1c8c`, `legacy/full`, archive branches and
annotated tag `preservation/post-848-full-2026-11-10`. Archive branch protection
still requires administrator action. [The audit](docs/audit/minimal-core-849/README.md)
records retained declarations and removal boundaries. ProximalSolvers migration
is separate.
