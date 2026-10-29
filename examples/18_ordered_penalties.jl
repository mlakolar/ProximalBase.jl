# Penalties that depend on the order of the entries.
#
# Total variation, monotonicity and the sorted ℓ1 norm pool adjacent entries
# into blocks of equal values. Three problems:
#
# - a piecewise-constant signal by total-variation denoising and the fused
#   lasso, whose solutions are proximal operators;
# - isotonic regression of a monotone trend, the projection onto the
#   nondecreasing vectors;
# - SLOPE regression, which gives correlated predictors equal coefficients
#   where the lasso picks among them.

using ProximalBase, LinearAlgebra, Random

rng = Xoshiro(18)

# --- a piecewise-constant signal ----------------------------------------------
# Total-variation denoising minimizes ‖y - z‖²/2 + λ Σᵢ |zᵢ₊₁ - zᵢ|: its
# solution is the proximal operator of TotalVariation1D(λ) at y with step 1.
n = 200
signal = [zeros(50); fill(2.0, 40); fill(-1.0, 60); fill(1.0, 50)]
y = signal .+ 0.5 .* randn(rng, n)
λ = 4.0
z = prox(TotalVariation1D(λ), y)
jumps = abs.(diff(z))
largest = sort(partialsortperm(jumps, 1:3; rev = true))
println("total variation: ", count(>(0), jumps), " jumps, the largest three after ", largest, "; error ",
        round(norm(z - signal) / sqrt(n), digits = 3), " against ", round(norm(y - signal) / sqrt(n), digits = 3),
        " for the data")
@assert largest == [50, 90, 150]                 # the change points
@assert norm(z - signal) < norm(y - signal) / 2
# the optimality conditions: the cumulative sums of the residual stay within
# ±λ and end at zero
r = cumsum(y - z)
@assert all(abs.(r) .<= λ + 1e-9) && abs(r[end]) < 1e-9

# the fused lasso adds λ1‖z‖₁, which soft-thresholds the denoised signal: the
# first segment, whose level is 0, becomes exactly zero up to the step at 49
zf = prox(FusedLasso(0.3, λ), y)
@assert zf ≈ shrink.(z, 0.3)
println("fused lasso: ", count(iszero, zf[1:50]), " of the first 50 entries are zero")
@assert all(iszero, zf[1:49])

# --- isotonic regression ---------------------------------------------------------
# The nondecreasing fit nearest to the data, the projection onto the monotone
# vectors, pools the entries that violate the order into blocks at their means.
t = sort(rand(rng, 100))
y = log1p.(5 .* t) .+ 0.2 .* randn(rng, 100)
z = prox(IndIsotonic(), y)
println("isotonic regression: ", length(unique(z)), " blocks for 100 points")
@assert issorted(z)
# the optimality conditions: the cumulative sums of the residual are
# nonnegative, and zero where a block ends
r = cumsum(y - z)
ends = [findall(!iszero, diff(z)); 100]
@assert all(r .>= -1e-10) && all(abs.(r[ends]) .< 1e-10)
# See example 19 for weighted isotonic regression.

# --- SLOPE regression -------------------------------------------------------------
# Three columns of X are nearly equal. The lasso splits their effect unevenly;
# SLOPE, whose sorted ℓ1 norm pools coefficients of equal magnitude, gives them
# one coefficient.
n, p = 100, 30
Z = randn(rng, n, p)
X = copy(Z)
X[:, 2:3] .= Z[:, 1] .+ 0.05 .* Z[:, 2:3]
β_true = [1.0; 1.0; 1.0; zeros(p - 6); -1.5; 0.0; 0.0]
y = X * β_true + 0.5 * randn(rng, n)
f = (1 / n) * LeastSquaresLoss(y, X)

function fista(f, g, p; iters = 20000)
    L = lipschitz(f)
    β, βold, z, grad = zeros(p), zeros(p), zeros(p), zeros(p)
    t = 1.0
    for _ in 1:iters
        gradient!(grad, f, z)
        copyto!(βold, β)
        β .= z .- grad ./ L
        prox!(β, g, β, 1 / L)
        tnew = (1 + sqrt(1 + 4t^2)) / 2
        z .= β .+ ((t - 1) / tnew) .* (β .- βold)
        t = tnew
    end
    β
end

βl = fista(f, NormL1(0.1), p)
g = SLOPE(0.1 .* collect(range(2.0, 0.2; length = p)))   # a decreasing sequence
βs = fista(f, g, p)
println("lasso: β[1:3] = ", round.(βl[1:3], digits = 3), "; SLOPE: β[1:3] = ", round.(βs[1:3], digits = 3))
@assert βs[1] ≈ βs[2] ≈ βs[3] && maximum(βl[1:3]) > 2minimum(βl[1:3])
@assert findall(!iszero, βs) == [1, 2, 3, p - 2]
@assert norm(gradient_mapping(f, g, βs, 1 / lipschitz(f))) < 1e-9
