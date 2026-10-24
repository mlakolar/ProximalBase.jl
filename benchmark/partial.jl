# Partial decompositions for the low-rank functions (`partial = true`):
# NuclearNorm, IndBallRank, IndPSD and TracePSD, cold and warm started, against
# the full LAPACK decompositions and against the partial SVDs and eigensolvers
# of registered packages (KrylovKit, Arpack, TSVD, LowRankApprox,
# RandomizedLinAlg), on low-rank matrices plus noise. Each case reports the
# fastest of a few calls, the bytes one call allocates, and the relative error
# of the result against the full decomposition.
#
# "cold" restarts the subspace iteration from a random block at every call;
# "warm" calls the function on a sequence of nearby matrices, as a solver does.
# The packages are told the exact rank, which favors them.
#
# Run from the repository root, with one BLAS thread:
#
#   julia --project=benchmark/partial -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=benchmark/partial benchmark/partial.jl [p ...]

using ProximalBase, LinearAlgebra, Random, Printf
using KrylovKit, Arpack, TSVD, LowRankApprox, RandomizedLinAlg

BLAS.set_num_threads(1)

const PB = ProximalBase
const SIZES = isempty(ARGS) ? [200, 500, 1000, 2000, 5000] : parse.(Int, ARGS)
const RANKS = [2, 10, 50]
const c = 1.0                      # the threshold: between the noise and the signal

orth(m, k) = Matrix(qr(randn(m, k)).Q)

# a rank-k signal with singular values in [10, 20] plus noise of spectral norm about 0.2
function lowrank(p, k)
  U, V = orth(p, k), orth(p, k)
  U * Diagonal(range(20, 10; length = k)) * V' + 0.1 * randn(p, p) / sqrt(p)
end
function symlowrank(p, k; δ = 0.0)
  V = orth(p, k)
  G = randn(p, p) / sqrt(2p)
  V * Diagonal(range(20, 10; length = k)) * V' + 0.1 * (G + G') - δ * I
end

reps(p) = p <= 500 ? 5 : p <= 1000 ? 3 : p <= 2000 ? 2 : 1

# the fastest of n calls of f(), and the bytes of the last one
function timed(f, n)
  best, bytes = Inf, 0
  for _ in 1:n
    s = @timed f()
    best = min(best, s.time)
    bytes = s.bytes
  end
  best, bytes
end

rel(A, B) = norm(A - B) / norm(B)

const ROWS = Vector{Any}[]
function report(p, k, func, method, t, bytes, err, note = "")
  push!(ROWS, Any[p, k, func, method, t, bytes, err, note])
  @printf "%5d %3d  %-16s %-22s %10.2f ms %12d B  %8.1e  %s\n" p k func method 1e3t bytes err note
  flush(stdout)
end

# the proximal operators from the triplets or eigenpairs of the packages
svt(U, s, V, c) = (j = s .> c; U[:, j] * Diagonal(s[j] .- c) * V[:, j]')
psdk(vecs, vals) = (j = vals .> 0; vecs[:, j] * Diagonal(vals[j]) * vecs[:, j]')

function packages_svd(p, k, X, ref, n)
  Y = similar(X)
  for (name, run) in (
      "KrylovKit svdsolve" => () -> begin
        vals, lv, rv, _ = svdsolve(X, k, :LR)
        svt(hcat(lv[1:k]...), vals[1:k], hcat(rv[1:k]...), c)
      end,
      "Arpack svds" => () -> (F = svds(X; nsv = k)[1]; svt(F.U, F.S, F.V, c)),
      "TSVD tsvd" => () -> ((U, s, V) = tsvd(X, k); svt(U, s, V, c)),
      "LowRankApprox psvd" => () -> ((U, s, V) = psvd(X; rank = k); svt(U, s, V, c)),
      "RandomizedLinAlg rsvd" => () -> (F = rsvd(X, k, 10); svt(F.U, F.S, F.V, c)))
    try
      Z = run()                                            # compiles at the first size
      t, bytes = timed(run, n)
      report(p, k, "NuclearNorm", name, t, bytes, rel(Z, ref))
    catch e
      report(p, k, "NuclearNorm", name, NaN, 0, NaN, "error: " * first(sprint(showerror, e), 60))
    end
  end
end

function packages_eig(p, k, S, ref, n)
  for (name, run) in (
      "KrylovKit eigsolve" => () -> begin
        vals, vecs, _ = eigsolve(Symmetric(S), k, :LR; ishermitian = true)
        psdk(hcat(vecs[1:k]...), vals[1:k])
      end,
      "Arpack eigs" => () -> ((d, v) = eigs(Symmetric(S); nev = k, which = :LR); psdk(v, d)),
      "LowRankApprox pheig" => () -> begin
        F = pheigfact(Hermitian(S); rank = 2k)
        j = sortperm(F.values; rev = true)[1:k]
        psdk(F.vectors[:, j], F.values[j])
      end)
    try
      Z = run()
      t, bytes = timed(run, n)
      report(p, k, "IndPSD(rank)", name, t, bytes, rel(Z, ref))
    catch e
      report(p, k, "IndPSD(rank)", name, NaN, 0, NaN, "error: " * first(sprint(showerror, e), 60))
    end
  end
end

# full, cold and warm partial calls of the proximal operator of make(partial)
# at X, with the nearby inputs Xs for the warm calls; the warm result is checked
# against a full decomposition only where that is cheap (p ≤ 2000)
function ours(p, k, name, make, X, Xs, ref, n; full = true)
  Y = similar(X)
  if full
    f = make(false)
    t, bytes = timed(() -> prox!(Y, f, X, 1.0), n)
    report(p, k, name, "full", t, bytes, rel(Y, ref))
  end
  f = make(true)
  prox!(Y, f, X, 1.0)                                      # sizes the workspace
  _forget_full!(f)
  t, bytes = timed(n) do
    f.pws isa Union{PB.PartialSVDWorkspace, PB.PartialEigenWorkspace} && (f.pws.warm = 0)
    prox!(Y, f, X, 1.0)
  end
  report(p, k, name, "partial, cold", t, bytes, rel(Y, ref), _fellback(f) ? "fell back" : "")
  f = make(true)
  prox!(Y, f, Xs[1], 1.0)
  _forget_full!(f)
  i = 1
  t, bytes = timed(n) do
    i += 1
    prox!(Y, f, Xs[i], 1.0)
  end
  err = p <= 2000 ? rel(Y, prox(make(false), Xs[i])) : NaN
  report(p, k, name, "partial, warm", t, bytes, err, _fellback(f) ? "fell back" : "")
end
# whether f ran a full decomposition since _forget_full!
const FULL_FIELDS = (:ws, :eig, :top)
_forget_full!(f) = foreach(s -> hasproperty(f, s) && setproperty!(f, s, nothing), FULL_FIELDS)
_fellback(f) = any(s -> hasproperty(f, s) && getproperty(f, s) !== nothing, FULL_FIELDS)

function run_size(p)
  n = reps(p)
  for k in RANKS
    2 * (k + 3 + max(10, (k + 3) ÷ 4)) > p && continue     # partial does not apply
    Random.seed!(p + k)
    full = p < 5000 || k == 10                             # the full SVD at p = 5000 once
    X = lowrank(p, k)
    E = randn(p, p) / sqrt(p)
    Xs = [X + 1e-3 * t * E for t in 1:n+2]
    F = svd(X)                                             # the references
    ref = svt(F.U, F.S, F.V, c)
    ours(p, k, "NuclearNorm", partial -> NuclearNorm(c; partial), X, Xs, ref, n; full)
    packages_svd(p, k, X, ref, n)
    ref = F.U[:, 1:k] * Diagonal(F.S[1:k]) * F.Vt[1:k, :]
    ours(p, k, "IndBallRank", partial -> IndBallRank(k; partial), X, Xs, ref, n; full = full && p < 5000)
    F = nothing

    S = symlowrank(p, k)
    G = randn(p, p) / sqrt(2p)
    Ss = [S + 1e-3 * t * (G + G') for t in 1:n+2]
    D = eigen(Symmetric(S))
    top = p-k+1:p
    ref = psdk(D.vectors[:, top], D.values[top])
    ours(p, k, "IndPSD(rank)", partial -> IndPSD(rank = k, partial = partial), S, Ss, ref, n)
    packages_eig(p, k, S, ref, n)
    ref = psdk(D.vectors, D.values .- c)
    ours(p, k, "TracePSD", partial -> TracePSD(c; partial), S, Ss, ref, n; full)
    Sm = S - 0.5I                                          # few positive eigenvalues
    Sms = [M - 0.5I for M in Ss]
    ref = psdk(D.vectors, D.values .- 0.5)
    ours(p, k, "IndPSD", partial -> IndPSD(partial = partial), Sm, Sms, ref, n; full = full && p < 5000)
    D = nothing
    GC.gc()
  end
end

println("ProximalBase partial decompositions; Julia ", VERSION, ", ", Sys.cpu_info()[1].model, ", BLAS threads ",
        BLAS.get_num_threads())
@printf "%5s %3s  %-16s %-22s %13s %14s  %8s\n" "p" "k" "function" "method" "time" "bytes" "rel err"
for p in SIZES
  run_size(p)
end
