# Function traits, dispatch, and coordinate oracles

This is a contributor guide to the interaction between **function types**,
**iterate types**, and the methods dispatched on their combination.
The user-visible calls are in [The interface](../interface.md) and
[Coordinate oracles](../iterate_oracles.md). The
[numerical contract](@ref numerics) governs supported result semantics.

A conceptual, user-facing catalog of function operations is maintained
in [Operations on functions](../operations.md); this page explains why
and how their Julia methods dispatch on the underlying types.


## Traits describe capabilities; types select methods

Every built-in function is an `AbstractFunction` subtype. Traits such as
`issmooth`, `isproximable`, `isseparable`, `isconvex`, `hashessian`, and
`iszeropreserving` answer *questions about the function*; they are not
replacement type hierarchies and they do not dispatch a scalar or group
kernel by themselves. Julia's multiple dispatch selects a method from
the concrete function and operand types. This allows a function to be both
smooth and proximable without inventing a class for every trait combination.

For example, `SqrNormL2` supports both a gradient and a proximal map.
A solver can query a trait to decide which algorithm is applicable, then
call `gradient!` or `prox!`; the corresponding Julia methods implement the
operation. The trait is not a promise that every arbitrary storage backend
has a specialized fast path.

## Three levels of operation

| Level | Example | Index meaning | Ownership |
|:--|:--|:--|:--|
| Mathematical entry | `value_entry(f, v, i)`, `prox_entry(f, v, i, γ)` | Entry of a function's parameter/weight arrays | Function interface |
| Optimization coordinate | `prox_coordinate!(x, f, k, γ)` and `prox_coordinate!(x, f, k, u, γ)` | Entry, lower-triangle variable, or group, depending on `x` | Function × iterate dispatch |
| Whole value | `f(x)`, `prox!`, `gradient!`, `value_and_prox!`, `value_and_gradient!` | Entire vector, matrix, or supported owner | Function/oracle interface |

The entry and coordinate indexes happen to agree for `SparseIterate`.
They **do not** agree in general: for a symmetric owner a coordinate
represents an independent matrix variable (and possibly two entries), and
for an `AtomIterate` a coordinate is a group. Therefore do not silently
replace entry methods with coordinate methods, or vice versa, merely because
a current sparse-vector call takes the same integer. Consolidation of the
API requires an explicit contract change and matching tests.

A coordinate method must preserve the active-set rules of its iterate.
With a trial `u`, a group that shrinks to zero can remain inactive without
first writing `u`. For scalar separable functions the coordinate kernel can
reuse an entry proximal operator. Group norms use group-specific methods;
the iterate's data structure alone cannot decide the function's shrinkage.

## Validation and symmetric geometry

A symmetric owner does not participate in generic `AbstractArray` methods.
Supported oracles explicitly accept it. The Frobenius geometry of a
symmetric matrix is different from the ordinary Euclidean geometry of its
independent-coordinate vector: off-diagonal pairs count twice for matrix
functions. See [Symmetric parameters](../symmetric_parameters.md) for
currently implemented checks and how a nonsymmetric point is read through
its symmetric part by a proximal map.

`restrict`, `kkt_violation`, coordinate curvature and screening are
algorithmic or function oracles; adding a new iterate does not automatically
define them. Preflight checks, aliasing, non-finite operands and the order
of decisions belong to their respective oracle contracts. In particular,
"strong zero" skips an exactly zero stored coefficient in specified sparse
products even against non-finite entries of the other operand; it is not
a license to suppress all IEEE exceptions in every generic array product.

## Implementation map

| Concern | Source |
|:--|:--|
| Traits and base oracle interface | `src/interface.jl` |
| Scalar and grouped function implementations | `src/functions/` |
| Function algebra and wrappers | `src/calculus.jl`, `src/composition.jl`, `src/constrained.jl` |
| Coordinate and restriction oracles | `src/partial.jl`, specialized functions |
| Iterate storage | `src/sparse_iterate.jl`, `src/atom_iterate.jl` |
| Parameter and owner integration | `src/symmetric_parameters.jl` |
| Numerical conformance | `test/conformance/`, `docs/src/numerics.md` |

When introducing a new method, document the exact supported operand domain,
whether it traverses active coordinates or full entries, what the output
means, and what happens on validation or conversion failure. Include tests
for entry-versus-coordinate differences and operation-specific cost
claims rather than assuming generic array methods are sparse-aware.
