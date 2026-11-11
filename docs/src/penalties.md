# Penalties

A penalty is the ``g`` in ``\min_x f(x)+g(x)``. The nonsmooth penalty provided here is `NormL1`. For a differentiable ridge penalty use
[SqrNormL2](smooth.md#Squared-Euclidean-norm); nonnegative multiples are
[Scaled](calculus.md) functions.

## Weighted ℓ1 norm

```math
g(x)=\lambda_0\sum_i w_i|x_i|,\qquad
\operatorname{prox}_{\gamma g}(v)_i=
\operatorname{sign}(v_i)\max(|v_i|-\gamma\lambda_0w_i,0)
```

for ordinary finite nonnegative parameters and positive finite step.
Construct `NormL1(λ0=1)` for unit weights or `NormL1(λ0,w)` for real array
weights. The constructor checks ``\lambda_0\ge0`` and every ``w_i\ge0``;
negative or NaN parameters throw `ArgumentError`. `Inf` is accepted with the
special semantics below. A scalar weight argument is not a constructor form.
Use the level for an overall scalar multiplier.

The ordinary domain is all real vectors/matrices, with weights of the same
number of entries. Shapes of weights can differ if their linear indexing
matches. Symmetric owners use full ``p^2`` entry weights, with exact equality
at mirrored pairs read by the operation. An array mismatch is
`DimensionMismatch`; non-real data/outputs/steps are `ArgumentError`.
Weights are retained by reference; keep them nonnegative if changing entries.

NormL1 is a convex, proximable, zero-preserving `SeparableFunction`, and is not
smooth. It supports value, `value_entry(g,v,i)`, scalar `prox_block(g,v,i,γ)`,
scalar/array-step full `prox[!]`, `value_and_prox!`, `restrict` and zero KKT
screens. It has no gradient. A raw entry call's index is the **global linear
entry index**, not a BlockIterate group or symmetric independent-coordinate ID.

```@example library_l1
using ProximalBase
w = [1.0, 0, 2]
g = NormL1(0.5,w)
x = [2.0, -3, 0.25]
@assert g(x) == 1.25
@assert value_entry(g,x[3],3) == 0.25
@assert prox(g,x,1.0) == [1.5, -3, 0]
@assert prox_block(g,x[3],3,1.0) == 0
@assert prox(g,x,[1.0,2,0.5]) == [1.5, -3, 0]
@assert kkt_violation(g,[0.1,-4,0.8],3) ≈ -0.2
@assert restrict(g,[3,1])([1.0,-2]) == 2
prox(g,x)
```

## Mutation, activity and screening

`prox!(out,g,x,γ)` returns `out`; `value_and_prox!` returns `g(out)` at the
stored, rounded output. `out === x` is supported; other shared-memory
input/output views are unsupported. Use a destination independent of the
weights. A fresh allocating prox does not promise to preserve the source's
explicit active zeros. A mutating proximal value write keeps existing
active zeros in its destination; only `dropzeros!` removes them.

Zero-preserving sparse paths skip inactive zeros. A symmetric off-diagonal
value contributes twice to the matrix penalty, but the proximal threshold
is still ``\gamma\lambda_0w_{rc}``: the Frobenius squared distance also has
multiplicity two, so the factor cancels. See
[symmetric geometry](symmetric_sparse_iterate.md).

At a zero coordinate, the signed KKT violation is
``|\mathrm{grad}_i|-\lambda_0w_i``. A positive value suggests adding that
coordinate to the working set. On BlockIterate the four-argument screen takes
the maximum scalar violation over that block. The screen is not a residual
for a nonzero coefficient; use the full KKT conditions in
[coordinate descent](coordinate_descent.md).

## Zero and infinite parameters

A zero weight leaves its entry unpenalized at every level. An infinite weight
thresholds a **finite trial** to zero, including a zero level/step under the implemented
arithmetic; its value is infinite away from zero. At a NaN/Inf trial with a
finite level it currently returns NaN (infinite threshold arithmetic), rather
than enforcing a singleton projection. This contract question is tracked in
[#853](https://github.com/mlakolar/ProximalBase.jl/issues/853). An infinite level at a
positive step fixes every positive-weight entry to zero, including NaN/Inf
trial entries. A zero level on an unweighted dense array returns zero value,
even with NaN entries. Do not infer a uniform cancellation policy for weighted
or non-finite cases; [the numerical contract](numerics.md) gives checked
examples and distinguishes infinite parameters from finite-product overflow.

[`shrink(v,c)`](@ref shrink) is the raw scalar soft-threshold utility for
``c\ge0``; it does not validate a threshold's sign. A NaN argument propagates.
A threshold that overflows from finite parameters does not mean the model has
an infinite level; built-in coefficient recovery distinguishes them.
