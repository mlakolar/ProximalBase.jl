# Weighted isotonic regression of binned data.
#
# Monotone regression with unequal bin counts uses weighted least squares.
# Pool adjacent violators pools blocks at their weighted means in O(K) time.
# With weights w = 1 ./ γ, the call is prox(IndIsotonic(), ybar, 1 ./ w),
# or prox! into a buffer. Here the weights are the counts in each bin.

using ProximalBase, LinearAlgebra, Random

rng = Xoshiro(19)

# --- binned data --------------------------------------------------------------
# One fitted value θ_b per bin gives the least-squares identity
# Σ_{b,j} (a_bj − θ_b)² = Σ_{b,j} (a_bj − ā_b)² + Σ_b n_b (ā_b − θ_b)².
# The first term is constant in θ, so the raw-data monotone fit is weighted
# isotonic regression of the bin means with weights equal to the bin counts.
K = 40
t = ((1:K) .- 0.5) ./ K
curve = log1p.(5 .* t)
n = rand(rng, 1:30, K)
a = [curve[b] .+ 0.3 .* randn(rng, n[b]) for b in 1:K]
ā = [sum(a[b]) / n[b] for b in 1:K]
γ = 1 ./ n
f = IndIsotonic()
θw = prox(f, ā, γ)
θu = prox(f, ā)
rss(θ) = sum(sum(abs2, a[b] .- θ[b]) for b in 1:K)
# By the identity above, θw minimizes the raw residual sum of squares among
# monotone fits with one value per bin; θu treats a bin of 1 observation like
# a bin of 30.
@assert rss(θw) <= rss(θu)
println("binned data: ", sum(n), " observations in ", K, " bins")
println("raw residual sum of squares: ", round(rss(θw); digits = 4),
        " for the weighted fit, ", round(rss(θu); digits = 4), " for the unweighted fit")

# --- optimality checks -------------------------------------------------------
# The cumulative weighted residuals are nonnegative and vanish at block ends.
@assert issorted(θw)
λ = cumsum(n .* (ā .- θw))
tol = 1e-9 * sum(n .* abs.(ā))
ends = [findall(!iszero, diff(θw)); K]
@assert all(λ .>= -tol)
@assert all(abs.(λ[ends]) .<= tol)

# Isotonic regression never splits a run of equal data values: averaging two
# adjacent fitted entries in the run keeps the fit feasible and lowers the objective
# unless they are equal. On these runs the error is Σ_b n_b(ā_b − θ_b)².
# Repeating bin means therefore gives an independent unweighted check.
expanded = vcat(fill.(ā, n)...)
repeated_fit = vcat(fill.(θw, n)...)
@assert prox(IndIsotonic(), expanded) ≈ repeated_fit
println("optimality: ", length(ends), " blocks; KKT and repeated-data checks passed")

# --- a decreasing trend ------------------------------------------------------
# A decreasing dose-response curve uses the same counts and weighted pooling.
descending = exp.(-3 .* t)
ad = [descending[b] .+ 0.3 .* randn(rng, n[b]) for b in 1:K]
ād = [sum(ad[b]) / n[b] for b in 1:K]
fd = IndIsotonic(decreasing = true)
θd = prox(fd, ād, γ)
@assert issorted(θd; rev = true)
@assert prox(IndIsotonic(decreasing = true), vcat(fill.(ād, n)...)) ≈ vcat(fill.(θd, n)...)
println("decreasing trend: ", length(unique(θd)), " blocks; repeated-data check passed")

# --- weights in place --------------------------------------------------------
# Reuse the function and a preallocated output across repeated fits, as along
# a sequence of smoothing bandwidths or in a bootstrap. The first call sizes
# the work buffers; subsequent prox! calls of the same size and types allocate
# nothing. Generating the input data is separate from this projection.
y = similar(ā)
prox!(y, f, ā, γ)
@assert y ≈ θw
ybar = similar(ā)
for _ in 1:5
  ybar .= ā .+ 0.05 .* randn(rng, K)
  prox!(y, f, ybar, γ)
  @assert issorted(y)
end

# Only weight ratios matter, so a common scaling of the steps changes no fit.
@assert prox(f, ā, 1 ./ n) ≈ prox(f, ā, 10 ./ n)
# Steps must be positive and finite, with a representable ratio. Drop bins
# with no observations before fitting, since their inverse counts are infinite.
println("in place: reused the output and pooling buffers for five further fits")
