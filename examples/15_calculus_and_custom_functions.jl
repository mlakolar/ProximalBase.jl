# Functions built from functions, and a function of your own.
#
# `Scaled`, `Translate`, `Tilt` and `Conjugate` carry the proximal operator of a
# function over to the new one; `Sum` adds smooth functions and their
# gradients; `SeparableSum` acts on a tuple of blocks. A new function is any
# subtype of `AbstractFunction` that declares its traits and implements the
# oracles they promise. A separable one subtypes `SeparableFunction` and
# implements two scalar functions, `value_entry` and `prox_entry`; the value,
# `prox!`, `value_and_prox!` and `prox_coordinate!` then come with the type.

using ProximalBase, LinearAlgebra, Random

rng = Xoshiro(15)
x = randn(rng, 8)
γ = 0.7
soft(v, t) = sign(v) * max(abs(v) - t, 0)

# --- Scaled: α f, also written α * f ----------------------------------------
@assert prox(0.5 * NormL1(2.0), x, γ) ≈ prox(NormL1(1.0), x, γ)
@assert (3 * NormL2(1.0))(x) ≈ 3 * norm(x)

# --- Translate: f(x - b) ----------------------------------------------------
b = randn(rng, 8)
h = Translate(NormL1(1.0), b)                         # ‖x - b‖₁
@assert h(x) ≈ norm(x - b, 1)
@assert prox(h, x, γ) ≈ b .+ soft.(x .- b, γ)

# --- Tilt: f(x) + ⟨c, x⟩ + d ------------------------------------------------
c = randn(rng, 8)
h = Tilt(SqrNormL2(1.0), c, 2.0)                      # ‖x‖²/2 + ⟨c, x⟩ + 2
@assert h(x) ≈ norm(x)^2 / 2 + dot(c, x) + 2
@assert prox(h, x, γ) ≈ (x .- γ .* c) ./ (1 + γ)

# --- Conjugate: f*(y) = supₓ ⟨x, y⟩ - f(x), by Moreau's identity -------------
# The conjugate of a norm is the indicator of the unit ball of the dual norm:
# for ‖⋅‖₂, the unit ℓ2 ball, whose proximal operator is the projection.
h = Conjugate(NormL2(1.0))
@assert prox(h, 3x, γ) ≈ prox(IndBallL2(1.0), 3x, γ)
@assert h(x / (2norm(x))) == 0 && h(2x / norm(x)) == Inf
# Moreau's identity: x = prox_{γf}(x) + γ prox_{f*/γ}(x/γ)
f = NormL1(1.0)
@assert prox(f, x, γ) + γ * prox(Conjugate(f), x / γ, 1 / γ) ≈ x
println("Scaled, Translate, Tilt and Conjugate agree with their closed forms")

# --- Sum of smooth functions: ridge regression --------------------------------
n, p = 50, 8
X, y = randn(rng, n, p), randn(rng, n)
μ = 3.0
ridge = Sum(LeastSquaresLoss(y, X), SqrNormL2(μ))     # ‖y - Xβ‖²/2 + (μ/2)‖β‖²
β, grad = zeros(p), zeros(p)
L = opnorm(X)^2 + μ
for _ in 1:2000
    gradient!(grad, ridge, β)                         # the sum of the two gradients
    β .-= grad ./ L
end
@assert β ≈ (X'X + μ * I) \ (X'y)
println("Sum: gradient descent on the ridge loss reaches (X'X + μI)⁻¹X'y")

# --- SeparableSum on a pair of blocks ----------------------------------------
x1, x2 = randn(rng, 5), randn(rng, 3)
h = SeparableSum(NormL1(0.5), IndNonnegative())
y1, y2 = prox(h, (x1, x2), γ)
@assert y1 ≈ soft.(x1, 0.5γ) && y2 == max.(x2, 0)
@assert h((x1, abs.(x2))) ≈ 0.5 * norm(x1, 1)

# --- a function of your own: the Huber penalty ------------------------------
# λ Σᵢ h_δ(xᵢ), with h_δ(t) = t²/(2δ) for |t| ≤ δ and |t| - δ/2 beyond: a
# smoothed ℓ1 norm. Its scalar proximal operator, the minimizer of
# λ h_δ(z) + (z - v)²/(2γ), is v / (1 + γλ/δ) when |v| ≤ δ + γλ, and
# v - γλ sign(v) beyond. It maps 0 to 0, and h_δ(0) = 0, so declaring
# `iszeropreserving` lets it touch only the stored entries of a sparse iterate.
struct HuberPenalty <: SeparableFunction
    λ::Float64
    δ::Float64
end
huber(t, δ) = abs(t) <= δ ? t^2 / (2δ) : abs(t) - δ / 2

ProximalBase.value_entry(f::HuberPenalty, v, i) = f.λ * huber(v, f.δ)
function ProximalBase.prox_entry(f::HuberPenalty, v, i, γ)
    τ = γ * f.λ
    abs(v) <= f.δ + τ ? v / (1 + τ / f.δ) : v - τ * sign(v)
end
ProximalBase.iszeropreserving(::HuberPenalty) = true
ProximalBase.isconvex(::Type{HuberPenalty}) = true

hp = HuberPenalty(1.0, 0.5)
# the proximal operator against a brute-force minimization on a fine grid
for v in (-2.0, -0.8, 0.3, 1.1, 3.0)
    grid = range(v - 3, v + 3, length = 600_001)
    zbest = grid[argmin([hp([z]) + (z - v)^2 / (2γ) for z in grid])]
    @assert abs(prox_entry(hp, v, 1, γ) - zbest) < 1e-4
end
# the value and prox! come from value_entry and prox_entry, also on sparse iterates
@assert hp(x) ≈ sum(t -> huber(t, hp.δ), x) && prox(hp, x, γ) ≈ [prox_entry(hp, v, 1, γ) for v in x]
xs = SparseIterate(5)
xs[2] = 1.5
@assert prox_coordinate!(xs, hp, 2, γ) ≈ prox_entry(hp, 1.5, 2, γ)
prox_coordinate!(xs, hp, 4, γ)                        # a zero stays zero, and unstored
@assert nnz(xs) == 1

# and in a proximal gradient method: least squares with the Huber penalty, a
# smooth problem whose solution has ∇f(β) + λ h_δ'(β) = 0
f = LeastSquaresLoss(y, X)
β = zeros(p)
L = lipschitz(f)
for _ in 1:5000
    gradient!(grad, f, β)
    β .-= grad ./ L
    prox!(β, hp, β, 1 / L)
end
dhuber(t, δ) = abs(t) <= δ ? t / δ : sign(t)
@assert norm(gradient(f, β) .+ hp.λ .* dhuber.(β, hp.δ)) < 1e-8
println("HuberPenalty: prox_entry, prox! and prox_coordinate! work like the built-in functions")
