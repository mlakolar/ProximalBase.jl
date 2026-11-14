# Glossary

*Assumes: gradients, convexity, and basic Julia syntax · You will learn: the optimization and Julia terminology used in this manual.*

These short definitions introduce ideas when they first appear. Follow the links
for mathematical derivations, code examples, and exact behavior.

## Proximal operator

For a function ``g`` and step ``\gamma>0``, the **proximal operator**
returns a minimizer of ``g(z)+\|z-v\|^2/(2\gamma)`` at input ``v``.
It balances reducing the function value against staying near ``v``;
see [proximability](function_objects.md#Proximability-and-proximal-operators)
and [proximal operations](operations.md#Computing-a-proximal-point).

## Proximable

A function is **proximable** when its proximal operator can be computed
efficiently by the supported implementation. This need not mean its gradient
exists or that the proximal method supports every array representation;
see [function properties](function_objects.md).

## Projection and indicator function

The **indicator function** of a nonempty closed convex set ``C`` is
``I_C(z)=0`` for ``z\in C`` and ``+\infty`` otherwise. Its mathematical
proximal operator is the Euclidean projection ``\Pi_C(v)``, so proximal
gradient generalizes projected gradient descent; see
[proximal-gradient methods](proximal_gradient.md). This is an illustrative
mathematical identity, not a claim that the package includes an indicator
function object for arbitrary sets.

## Soft thresholding

**Soft thresholding** with threshold ``t\geq0`` subtracts ``t`` from the
magnitude of a scalar and clips the result at zero:
``S_t(v)=\operatorname{sign}(v)\max(|v|-t,0)``. It is the proximal operator
of a scaled absolute-value function; see [the ℓ₁ function](penalties.md).

## Subgradient and subdifferential

For convex ``g``, a vector ``s`` is a **subgradient** at ``x`` if
``g(z)\geq g(x)+\langle s,z-x\rangle`` for every ``z``. The set of all such
vectors is the **subdifferential** ``\partial g(x)``; for ``g(t)=|t|``,
``\partial g(0)=[-1,1]``. See [lasso optimality](getting_started.md#Optimality-and-storage).

## KKT conditions for nonsmooth objectives

The **Karush–Kuhn–Tucker (KKT) optimality condition** for a convex
differentiable ``f`` plus a convex, possibly nonsmooth ``g`` takes the form
``0\in\nabla f(x)+\partial g(x)``. It generalizes stationarity from
differentiable constrained optimization; see
[lasso conditions](getting_started.md#Optimality-and-storage) and
[coordinate screening](coordinate_descent.md#Optimality-and-working-sets).

## Proper closed convex function

A **proper** extended-real function is finite somewhere and never equals
``-\infty``. **Closed** means lower semicontinuous (equivalently, its
epigraph is closed), and **convex** has its usual meaning; these assumptions
are useful for proximal-point existence and uniqueness statements.
See [proximal-gradient methods](proximal_gradient.md).

## Gradient mapping

For differentiable ``f`` and proximable ``g``, the **gradient mapping**
is

```math
G_\gamma(x)=\frac{x-\operatorname{prox}_{\gamma g}(x-\gamma\nabla f(x))}{\gamma}.
```

It measures how far one proximal-gradient step moves the point; zero
indicates the corresponding first-order
stationarity condition. See
[gradient mapping and optimality](proximal_gradient.md#Gradient-mapping-and-optimality).

## Separable function

A function is **separable** if it can be written as a sum of independent
coordinate terms, ``g(x)=\sum_i g_i(x_i)``. That structure permits
coordinate-wise computations; see
[separability](function_objects.md#Separability-and-coordinate-operations).

## Zero-preserving function

A separable function is **zero preserving** here if each scalar contribution
vanishes at zero and its scalar proximal map sends zero to zero. This permits
some sparse computations to skip stored and unstored zero coordinates; see
[zero preservation](function_objects.md#Zero-preservation-and-sparse-iterates).

## Iterate

An **iterate** is the current optimization variable, represented as a
Julia array or specialized working-set storage object. The type changes
storage and access behavior, not the mathematical objective; see
[iterates](iterates.md).

## Working set

A **working set** is the group of coordinates or blocks a solver has chosen
to visit or update. It may include values equal to zero and is a solver
choice, not a set of active inequality constraints; see
[iterates and working sets](iterates.md).

## Active coordinate

An **active coordinate** in ProximalBase is a coordinate explicitly retained
in an iterate's working-set storage, even when its numerical value is zero.
This differs from the familiar *active constraint* meaning an inequality
constraint holding with equality; see
[active coordinates](iterates.md#Active-coordinates-are-deliberate-state).

## Active slot

An **active slot** is the position of a stored coordinate in an internal
packed active list, not the coordinate's numerical index. It may change
after storage compaction; see
[entries and active slots](iterates.md#Entries,-coordinates-and-active-slots).

## Compaction

**Compaction**, via `dropzeros!(x)`, removes stored zero-valued coordinates
or all-zero blocks from the working-set representation. It does not
change the represented mathematical point; see
[working-set operations](iterates.md#Active-coordinates-are-deliberate-state).

## Spectral norm and Frobenius norm

The **spectral norm** of a matrix is its largest singular value, and
the **Frobenius norm** is the square root of the sum of squared entries.
They differ: least-squares step-size bounds involve a spectral norm,
while a matrix squared-distance loss uses the Frobenius norm;
see [the function library](function_library.md).

## In-place and mutating operation

A Julia function ending in `!` conventionally **mutates** an argument;
ProximalBase's mutating mathematical operations place the destination
first. Some proximal operations accept the same input and destination
but gradient operations require separate buffers; see
[calling conventions](operations.md#Calling-conventions,-results-and-supported-inputs).

## Aliasing

Two inputs **alias** when they share underlying writable memory, even if
they are different views or array objects. Such overlap matters for
mutating operations; see [array and precision conventions](numerical_usage.md).

## Exported and qualified names

An **exported name** is available directly after `using ProximalBase`,
while an **unexported name** is called with the module prefix, such as
`ProximalBase.Squared()` or `ProximalBase.deriv(...)`. See
[scalar losses](scalar_losses.md) and the [API reference](api/interface.md).
