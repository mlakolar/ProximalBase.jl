# ProximalBase.jl

ProximalBase provides sparse iterate storage and mathematical building blocks
for proximal algorithms. The minimal core supports `SparseIterate`,
`BlockIterate`, `SymmetricSparseIterate`, `LeastSquaresLoss`, `QuadraticFunction`,
`SqrNormL2`, `NormL1`, `Scaled`, `ZeroFunction` and `LinearFunction`.
`ProximalBase.Squared` supplies the scalar squared-loss contract.

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

The executable `examples/lasso.jl` verifies these mathematical pieces against
dense formulas. Algorithms and solver state belong in a solver package.
The historical full implementation and its test catalog are preserved at
`eff21b6d7d33c622313122f46c6ee71693c101ee` and `legacy/full`.
