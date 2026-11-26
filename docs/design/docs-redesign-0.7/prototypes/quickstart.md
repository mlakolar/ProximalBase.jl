# [Five-minute quick start](@id quickstart)

This page uses ordinary `Vector{Float64}` and `Matrix{Float64}` arrays and the
handful of calls most programs need. Every block runs as written.

## Install

```julia
using Pkg
Pkg.add(url="https://github.com/mlakolar/ProximalBase.jl", rev="master")
```

This installs the development version that this manual describes. The registered
release is older and has a different interface; see
[Which version should I install?](@ref install-versions).

## A function is an object

Construct a function once and call it like a Julia function. The weighted ℓ₁
norm ``g(x)=\lambda\sum_i|x_i|`` with ``\lambda=0.5`` is

```@example quickstart
using ProximalBase

g = NormL1(0.5)
x = [2.0, -1.0, 0.25]
g(x)                      # 0.5 * (2 + 1 + 0.25)
```

The object stores the parameters (here ``\lambda``). The point ``x`` is supplied
at each call, so the same `g` can be evaluated anywhere.

## Take a proximal step

The proximal operator of ``g`` with step ``\gamma>0`` returns the point that
balances a small value of ``g`` against staying close to the input ``v``:

```math
\operatorname{prox}_{\gamma g}(v)=\operatorname*{argmin}_z
\Bigl\{\,g(z)+\tfrac{1}{2\gamma}\|z-v\|_2^2\Bigr\}.
```

For the ℓ₁ norm this is *soft thresholding*: every entry moves toward zero by
``\gamma\lambda`` and stops at zero.

```@example quickstart
prox(g, x, 1.0)           # threshold γλ = 0.5
```

The third entry, ``0.25``, was within ``0.5`` of zero, so it became exactly zero.
This is how ℓ₁ regularization produces sparse estimates.

## Evaluate a loss and its gradient

[`LeastSquaresLoss(y, X)`](@ref LeastSquaresLoss) is
``f(\beta)=\tfrac12\|y-X\beta\|_2^2``. Note the factor ``\tfrac12`` and the
absence of ``1/n``.

```@example quickstart
X = [1.0 2.0; 3.0 1.0; 0.0 2.0; 2.0 -1.0]
y = [1.0, 2.0, 3.0, -0.5]
β = [0.3, -0.4]

f = LeastSquaresLoss(y, X)
f(β), gradient(f, β)      # gradient is X' * (X * β - y)
```

To use the mean loss ``\tfrac{1}{2n}\|y-X\beta\|_2^2``, multiply by a
nonnegative number. The result is again a function object:

```@example quickstart
n = size(X, 1)
f_mean = (1 / n) * f
f_mean(β)
```

## Combine them: one proximal-gradient step

A lasso estimate minimizes ``f(\beta)+\lambda\|\beta\|_1``. One step of the
proximal-gradient method takes a gradient step on the smooth loss and a proximal
step on the penalty. For this dense ``X``, [`lipschitz`](@ref) computes
``L=\|X\|_2^2`` in floating point, a positive bound on how fast the gradient changes, and ``1/L`` is a
standard step size. (`lipschitz` can return `0` for an all-zero design, so check the
value before dividing in your own code.)

```@example quickstart
penalty = NormL1(0.2)
γ = 1 / lipschitz(f)
β_next = prox(penalty, β - γ * gradient(f, β), γ)
```

This is one update. The [proximal-gradient tutorial](proximal_gradient.md) explains
how the update is repeated and how to measure optimality with the gradient mapping. To fit models without writing the loop,
see [ProximalSolvers.jl](@ref with-proximalsolvers).

## Reuse memory with `!` functions

`prox` and `gradient` have in-place versions, `prox!` and `gradient!`, that write
into an output you provide. The output comes first, as in Julia's `mul!`:

```@example quickstart
out = similar(x)
prox!(out, g, x, 1.0)     # writes the proximal point into out
grad = similar(β)
gradient!(grad, f, β)     # writes the gradient into grad
out, grad
```

[`value_and_prox!`](@ref) and [`value_and_gradient!`](@ref) also return a
function value. Note which point each value belongs to:

```@example quickstart
value_and_prox!(out, g, x, 1.0)       # g(out): value at the new point
```

```@example quickstart
value_and_gradient!(grad, f, β)       # f(β): value at the input
```

For these four calls the output must have the shape of the input. `prox!(x, g, x, γ)`
may overwrite its input; `gradient!` needs an output that does not share memory with
the input; other partly overlapping arrays are not supported by either. The `!`
means "writes its first argument", not "never allocates".
[Array conventions](@ref numerical-usage) gives the precise rules.

## Where to go next

- **Choose a function** for your model: the [function catalog](@ref function-catalog).
- **Check what a function supports** (gradient, prox, coordinate updates,
  sparse or block storage): the [capability tables](@ref capability-tables).
- **Write an algorithm**: the [proximal gradient](proximal_gradient.md) and
  [coordinate descent](coordinate_descent.md) tutorials, then [iterates](@ref Sparse-iterates)
  for storing only the coordinates an algorithm is updating.
- **Solve a model** without writing an algorithm: [ProximalSolvers.jl](@ref with-proximalsolvers).
