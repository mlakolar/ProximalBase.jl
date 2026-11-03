# A minimal Frank–Wolfe prototype that consumes `lmo_atom`, for matrix completion (issue #518,
# consumer: ProximalSolvers #145). It is a benchmark script, not a solver.
#
#   julia --project=benchmark benchmark/frank_wolfe_prototype.jl <out.csv> <m> <n> <rank> <p> [reps] [seed]
#
# Constrained problem   min ½‖P_Ω(X) − y‖²  s.t. ‖X‖_* ≤ δ,
# penalized comparator  min ½‖P_Ω(X) − y‖² + λ‖X‖_*  by accelerated proximal gradient (FISTA with
# restart) whose prox is `NuclearNorm(λ; partial = true)`, i.e. soft-impute with momentum.
#
# The two formulations share their solution: X_λ solves the constrained problem for δ = ‖X_λ‖_*.
# X_λ is computed first, to high accuracy, so that f* = f(X_λ) is known to both methods. FW is
# stopped on f − f* ≤ t (f(0) − f*), and the comparator on F − F* ≤ t (f(0) − f*): the same absolute
# threshold. For a feasible X, F − F* = (f − f*) + λ(‖X‖_* − δ) ≤ f − f*, so the FW criterion is the
# stricter per iterate, but FISTA iterates are infeasible for the constrained problem: the two
# criteria answer different questions, and this is stated with every result. Times exclude the cost
# of the stopping measurement. f*, F* and δ come from FISTA itself (3000 iterations with an inexact
# partial prox); the reference is checked by its proximal-gradient residual, which is printed. A
# tolerance that is not reached is written as a censored row (time NaN), never dropped.
#
# FW keeps the iterate as `Xo`, its values on the observed entries (all that the loss and the
# gradient read), and the factors of its atoms, X = Σⱼ cⱼ uⱼ vⱼᵀ: no m × n matrix is formed.
using ProximalBase, LinearAlgebra, SparseArrays, Random, Printf

out = ARGS[1]
m, n, rnk = parse(Int, ARGS[2]), parse(Int, ARGS[3]), parse(Int, ARGS[4])
p = parse(Float64, ARGS[5])
reps = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 5
seed = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 145
tols = (1e-2, 1e-3, 1e-4)

rng = Xoshiro(seed)
pattern = sprand(rng, m, n, p)
I, J, _ = findnz(pattern)                                  # column-major, the order of a CSC matrix
A, B = randn(rng, m, rnk) / rnk^0.25, randn(rng, n, rnk) / rnk^0.25
y = [dot(view(A, i, :), view(B, j, :)) + 0.05 * randn(rng) for (i, j) in zip(I, J)]
nobs = length(y)
G = sparse(I, J, y, m, n)                                   # the gradient reuses this structure
@assert nnz(G) == nobs

f(Xo) = 0.5 * sum(abs2, Xo .- y)
function dense_obs(Z)
  v = Vector{Float64}(undef, nobs)
  @inbounds for q in 1:nobs
    v[q] = Z[I[q], J[q]]
  end
  v
end

# ---- the penalized reference solution X_λ, to high accuracy ----------------------------------------
λ = 0.2 * opnorm(Matrix(G))
function fista(λ; maxiter, target = nothing, f0F0 = nothing, record = false)
  g = NuclearNorm(λ; partial = true)
  X, Xprev, Y, V = zeros(m, n), zeros(m, n), zeros(m, n), zeros(m, n)
  t = 1.0
  trace = Tuple{Int, Float64, Float64}[]               # (iteration, seconds of iteration work, F)
  elapsed = 0.0
  Fold = Inf
  for it in 1:maxiter
    val = 0.0
    elapsed += @elapsed begin
      copyto!(V, Y)
      @inbounds for q in eachindex(y)
        V[I[q], J[q]] = y[q]                           # V = Y − P_Ω(Y − y): the gradient step of size 1
      end
      copyto!(Xprev, X)
      val = value_and_prox!(X, g, V, 1.0)              # soft-impute: shrink the singular values by λ
    end
    Fnew = f(dense_obs(X)) + val
    elapsed += @elapsed begin                          # momentum with a function-value restart
      if Fnew > Fold
        t = 1.0
        copyto!(Y, X)
      else
        tn = (1 + sqrt(1 + 4t^2)) / 2
        @. Y = X + ((t - 1) / tn) * (X - Xprev)
        t = tn
      end
    end
    Fold = Fnew
    push!(trace, (it, elapsed, Fnew))
    target !== nothing && Fnew - target <= f0F0 && break
  end
  X, trace
end
t_ref = @elapsed ((Xλ, _) = fista(λ; maxiter = 3000)); @printf("reference solve %.1fs\n", t_ref); flush(stdout)
Xo_λ = dense_obs(Xλ)
δ = sum(svdvals(Xλ))
fstar = f(Xo_λ)
Fstar = fstar + λ * δ
f0 = f(zeros(nobs))
F0 = f0
# the reference is checked by its proximal-gradient residual ‖X − prox(X − ∇f(X))‖ / ‖X‖
let g = NuclearNorm(λ; partial = true), V = copy(Xλ), Z = similar(Xλ)
  @inbounds for q in 1:nobs
    V[I[q], J[q]] = y[q]                                 # X − ∇f(X)
  end
  prox!(Z, g, V, 1.0)
  global ref_residual = norm(Z - Xλ) / norm(Xλ)
end
@printf("reference residual ‖X − prox(X − ∇f)‖/‖X‖ = %.2e\n", ref_residual)
@printf("m=%d n=%d rank=%d obs=%d  λ=%.4g δ=‖X_λ‖_*=%.4g  rank(X_λ)=%d  f*=%.6g f(0)=%.6g\n", m, n, rnk, nobs, λ, δ,
        count(>(1e-6 * δ), svdvals(Xλ)), fstar, f0)

# ---- Frank–Wolfe with a factored iterate -------------------------------------------------------------
verbose = get(ENV, "FW_VERBOSE", "false") == "true"
function frank_wolfe(; method, maxiter, tol, exactgap = false)
  C = SingularValue(IndBallL1(δ))
  Xo = zeros(nobs)
  coefs, Us, Vs = Float64[], Vector{Float64}[], Vector{Float64}[]
  R = similar(Xo)
  Gk = copy(G)
  result = Dict{Float64, Tuple{Float64, Int, Int}}()   # tol → (time, iterations, atoms)
  tmain = 0.0
  gaps = Tuple{Int, Float64, Float64, Float64}[]       # (iteration, approximate gap, exact gap, f − f*)
  for k in 0:maxiter-1
    tw = @elapsed begin
      @. R = Xo - y                                    # the gradient is P_Ω(X) − y on the pattern
      copyto!(Gk.nzval, R)
      a = lmo_atom(C, Gk; method)                      # S = −δ u vᵀ, two vectors
      u, v = a.U, a.V                                  # a.U already carries the factor −δ
      gap_part = 0.0
      @inbounds for q in 1:nobs
        gap_part += R[q] * u[I[q], 1] * v[J[q], 1]     # ⟨R, S_Ω⟩
      end
      d_dot_R = gap_part - dot(R, Xo)                   # ⟨R, S_Ω − X_Ω⟩ = −(Frank–Wolfe gap)
      dd = 0.0
      @inbounds for q in 1:nobs
        s = u[I[q], 1] * v[J[q], 1]
        dd += (s - Xo[q])^2
      end
      γ = dd == 0 ? 0.0 : clamp(-d_dot_R / dd, 0.0, 1.0)  # exact line search for a quadratic
      @inbounds for q in 1:nobs
        Xo[q] = (1 - γ) * Xo[q] + γ * u[I[q], 1] * v[J[q], 1]
      end
      # the factored iterate: scale the old coefficients, append the atom's factors
      coefs .*= (1 - γ)
      push!(coefs, γ); push!(Us, vec(u)); push!(Vs, vec(v))   # shared columns of the atom, not copies of a matrix
    end
    tmain += tw
    fk = f(Xo) - fstar
    verbose && (k + 1) % 100 == 0 && (@printf("  fw it %d  t=%.2fs  (f-f*)/(f0-f*)=%.2e  gap=%.3e\n", k + 1, tmain, fk / (f0 - fstar), -d_dot_R); flush(stdout))
    if exactgap && (k + 1) % 10 == 0
      # the exact gap needs the exact oracle: a dense SVD of the sparse gradient (small sizes only)
      @. R = Xo - y
      copyto!(Gk.nzval, R)
      σ1 = svdvals(Matrix(Gk))[1]
      exact = dot(R, Xo) + δ * σ1
      push!(gaps, (k + 1, -d_dot_R, exact, fk))
    end
    for t in tols
      haskey(result, t) || fk <= t * (f0 - fstar) && (result[t] = (tmain, k + 1, length(coefs)))
    end
    length(result) == length(tols) && !exactgap && break
  end
  result, gaps, Base.summarysize((coefs, Us, Vs, Xo, R, Gk))
end

# warm up (compilation) on a few iterations
frank_wolfe(; method = :krylov, maxiter = 3, tol = 0.0)
fista(λ; maxiter = 2)

rows = ["m,n,rank,obs,seed,rep,method,tol,time_s,iterations,atoms_or_rank,state_bytes,dense_bytes,reference_residual"]
push!(rows, join((m, n, rnk, nobs, seed, 0, "reference", 0, "NaN", 0, 0, 0, 8 * m * n, @sprintf("%.3e", ref_residual)), ","))
times = Dict{Tuple{String, Float64}, Vector{Float64}}()
row(rep, method, t, tm, it, atoms, bytes) = push!(rows, join((m, n, rnk, nobs, seed, rep, method, t, tm, it, atoms, bytes, 8 * m * n,
                                                           @sprintf("%.3e", ref_residual)), ","))
function record_fw(rep, method, res, bytes)
  for t in tols
    if haskey(res, t)
      row(rep, "fw-" * method, t, @sprintf("%.4f", res[t][1]), res[t][2], res[t][3], bytes)
      push!(get!(times, ("fw-" * method, t), Float64[]), res[t][1])
    else
      row(rep, "fw-" * method, t, "NaN", -1, -1, bytes)         # censored: not reached
      push!(get!(times, ("fw-" * method, t), Float64[]), NaN)
    end
  end
end
for rep in 1:reps
  # interleaved within a repetition: FW with the Krylov oracle, FW with the full oracle, the comparator
  res, _, bytes = frank_wolfe(; method = :krylov, maxiter = 5_000, tol = 0.0)
  record_fw(rep, "krylov", res, bytes)
  if m * n <= 4_000_000
    res, _, bytes = frank_wolfe(; method = :full, maxiter = 5_000, tol = 0.0)
    record_fw(rep, "full", res, bytes)
  end
  Z, trace = fista(λ; maxiter = 5_000, target = Fstar, f0F0 = minimum(tols) * (f0 - fstar))
  sv = svdvals(Z)
  r = count(>(1e-8 * sv[1]), sv)
  state = 4 * sizeof(Float64) * m * n                   # X, Xprev, Y, V
  for t in tols
    k = findfirst(x -> x[3] - Fstar <= t * (f0 - fstar), trace)
    if k === nothing
      row(rep, "fista-nuclear", t, "NaN", -1, r, state)
      push!(get!(times, ("fista-nuclear", t), Float64[]), NaN)
    else
      row(rep, "fista-nuclear", t, @sprintf("%.4f", trace[k][2]), trace[k][1], r, state)
      push!(get!(times, ("fista-nuclear", t), Float64[]), trace[k][2])
    end
  end
  @printf("rep %d done\n", rep); flush(stdout)
end
# the paired decision: the median over repetitions of the ratio FW time / FISTA time, per tolerance
for method in ("fw-krylov", "fw-full"), t in tols
  haskey(times, (method, t)) || continue
  ratios = times[(method, t)] ./ times[("fista-nuclear", t)]
  ok = filter(!isnan, ratios)
  @printf("%-10s tol %.0e: reached %d/%d, median time FW/FISTA = %s (a ratio ≤ 0.8 meets the 20%% criterion)\n",
          method, t, length(ok), length(ratios), isempty(ok) ? "n/a" : @sprintf("%.3f", sort(ok)[(length(ok) + 1) ÷ 2]))
end

# honest gap: the approximate-oracle gap against the exact one, on a short run
_, gaps, _ = frank_wolfe(; method = :krylov, maxiter = 100, tol = 0.0, exactgap = m * n <= 4_000_000)
for (k, g_approx, g_exact, sub) in gaps
  push!(rows, join((m, n, rnk, nobs, seed, 0, "gap-check", k, @sprintf("%.6e", g_approx), @sprintf("%.6e", g_exact), @sprintf("%.6e", sub), "", "", ""), ","))
end
open(out, "w") do io
  foreach(r -> println(io, r), rows)
end
