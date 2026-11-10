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

## Mathematical operations and storage coordinates

| Level | Example | Index meaning | Ownership |
|:--|:--|:--|:--|
| Mathematical entry | `value_entry(f, v, i)`, `prox_block(f, v, i, γ)` | Entry of a function's parameter/weight arrays | Function interface |
| Mathematical group | `prox_block!(out, f, u, k, γ)`, `prox_block(f, u, k, γ)` | Penalty group; compact buffers follow its entry order | Function interface |
| Optimization coordinate | `coordinates(x)[k] = v`, `copyto!(atom(x,k), out)`, `kkt_violation(f,g,x,k)` | Entry, lower-triangle variable, or group, depending on `x` | Storage writer or function × iterate screen |
| Whole value | `f(x)`, `prox!`, `gradient!`, `value_and_prox!`, `value_and_gradient!` | Entire vector, matrix, or supported owner | Function/oracle interface |

The entry and coordinate indices agree for `SparseIterate`. For a symmetric
owner, a coordinate represents an independent matrix variable, so the solver
maps it to the full-matrix linear index before calling the scalar oracle and
validates mirrored parameters explicitly. For `AtomIterate`, the solver checks
that the iterate and penalty have the same ordered groups before using one
group ID in both interfaces. A separable penalty has no group partition;
apply its scalar kernel to each group's global entry indices.

A solver owns trial construction, index conversion, public storage writes,
cache updates and active-set policy. A compact group trial that shrinks to
zero can remain inactive because the trial is never written to the iterate.
The block oracle supplies only the mathematical result. Its allocating form
returns a plain `Vector`; its mutating form supports exact in-place buffers
and rejects other overlap and iterate-backed outputs.

The private `_block_owner` preflight recognizes built-in iterate owners and
their coordinate/group views so that a numerical output cannot accidentally
activate storage. This representation dependency is intentional safety
validation; the mathematical kernels do not read or write iterate state.
Adding or renaming an iterate or storage-backed wrapper requires updating
that guard and its rejection tests. Arbitrary arrays exposing private storage
remain unsupported, as specified by the block buffer contract.
See [Update vocabulary](@ref update-vocabulary) for the public terminology.

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
| Coordinate and restriction oracles | `src/block_prox.jl`, `src/partial.jl`, specialized functions |
| Iterate storage | `src/sparse_iterate.jl`, `src/atom_iterate.jl` |
| Parameter and owner integration | `src/symmetric_parameters.jl` |
| Numerical conformance | `test/conformance/`, `docs/src/numerics.md` |

When introducing a new method, document the exact supported operand domain,
whether it traverses active coordinates or full entries, what the output
means, and what happens on validation or conversion failure. Include tests
for entry-versus-coordinate differences and operation-specific cost
claims rather than assuming generic array methods are sparse-aware.
