# Benchmarks of the atoms of the linear minimization oracles (issue #518).
#
#   julia --project=benchmark benchmark/lmo_atoms.jl <label> <out.csv> [quick|standard|large|memgate]
#
# Correctness references (full svd/eigen) are computed outside the timed regions. Each case is
# called once before it is timed (that first call is also recorded, with its compilation, as
# `first_call_s_cold_only_for_first_case`); timings are medians of BenchmarkTools samples on one thread. The `large`
# suite adds sparse matrices of order 10^4 and is meant for a separate run; `memgate` runs only the order-10^4 sparse
# Krylov cases and the storage rows, with no dense matrix allocated, so that `maxrss_mb` is the peak of those cases alone.
#
# Memory columns, four different quantities (none of them is another's substitute):
#   `bytes`/`allocs`  Julia heap bytes allocated by one warmed call (BenchmarkTools). It is NOT a peak: memory freed during
#                     the call counts, and memory of native BLAS/LAPACK (the workspace of gesdd/syevr) is not counted.
#   `atom_bytes`      `Base.summarysize` of the returned atom, the storage that remains.
#   `retained_bytes`  `Base.summarysize` of the set object after the warm-up call (a cached SVD/eigen workspace or stored
#                     subspace block would show here; empty for the stateless oracles, nonzero for the `partial` baseline).
#   `maxrss_mb`       `Sys.maxrss()` of the whole process when the row was written: a monotone high-water mark of the
#                     resident set, shared by all cases run so far (the dense reference matrices of earlier cases
#                     included). It bounds the peak of a case only from above and is informative only for the `large` suite,
#                     which has no dense matrices of order 10^4; run the memory-gate cases in a separate process for a
#                     per-case peak. Krylov rows run with `fallback = false`: a call that does not converge is reported
# (`converged = false`, not timed) and never replaced by the dense decomposition, so a Krylov row is a measurement
# of the sparse algorithm only. The Krylov convergence probe runs before the first-call timing, so the first-call
# column of a Krylov row is warm. The `full` rows of a sparse gradient include the densification, as stated in `note`.
using ProximalBase, BenchmarkTools, Random, LinearAlgebra, SparseArrays, Printf

BLAS.set_num_threads(1)           # one thread, recorded in every row's `note`
label, out = ARGS[1], ARGS[2]
suite = length(ARGS) >= 3 ? ARGS[3] : "standard"
BenchmarkTools.DEFAULT_PARAMETERS.seconds = suite == "quick" ? 0.2 : 1.0
BenchmarkTools.DEFAULT_PARAMETERS.samples = 10_000

rows = String[]
header = "label,family,case,method,T,shape,median_ms,min_ms,allocs,bytes,atom_bytes,retained_bytes,maxrss_mb,first_call_s_cold_only_for_first_case,steps,fallback,converged,rel_obj_err,note"

function record!(family, case, method, T, shape, t; first = NaN, steps = "", fallback = "", converged = "", atom_bytes = "", retained = "", err = NaN, note = "")
  note = replace(string(note, note == "" ? "" : "; ", "blas=", BLAS.get_num_threads(), " julia=", Threads.nthreads()), "," => ";")   # the CSV has no quoting
  push!(rows, join((label, family, case, method, T, shape, @sprintf("%.4f", median(t).time / 1e6),
                    @sprintf("%.4f", minimum(t).time / 1e6), t.allocs, t.memory, atom_bytes, retained, @sprintf("%.1f", Sys.maxrss() / 2^20), @sprintf("%.3f", first),
                    steps, fallback, converged, @sprintf("%.2e", err), note), ","))
  @printf("%-10s %-22s %-8s %-8s %-12s %10.3f ms %8d allocs %12d B  err %.1e %s\n",
          family, case, method, T, shape, median(t).time / 1e6, t.allocs, t.memory, err, note)
end

# time f() after a first call; returns the benchmark and the first-call time
function measure(f)
  first = @elapsed f()
  t = @benchmark $f()
  t, first
end

σmax(G) = svdvals(Matrix(G))[1]

# ---- vertex atoms ------------------------------------------------------------------------
for T in (Float64, Float32), n in (suite == "memgate" ? Int[] : [1_000, 1_000_000])
  g = randn(Xoshiro(1), T, n)
  s = zeros(T, n)
  for (name, C) in (("simplex", IndSimplex(T(1))), ("l1ball", IndBallL1(T(1))))
    t, f1 = measure(() -> lmo!(s, C, g))
    record!("vertex", name, "lmo!", T, "n=$n", t; first = f1)
    t, f1 = measure(() -> lmo_atom(C, g))
    record!("vertex", name, "atom", T, "n=$n", t; first = f1, atom_bytes = string(Base.summarysize(lmo_atom(C, g))))
  end
end

# ---- matrix families -----------------------------------------------------------------------
# a gradient of a matrix-completion loss at X = 0: -P_Ω(M) for a low-rank M plus noise
function completion_gradient(rng, m, n, p; rank = 5)
  A, B = randn(rng, m, rank), randn(rng, n, rank)
  nnz_target = round(Int, p * m * n)
  I = rand(rng, 1:m, nnz_target); J = rand(rng, 1:n, nnz_target)
  vals = [-(dot(view(A, i, :), view(B, j, :)) + 0.1 * randn(rng)) for (i, j) in zip(I, J)]
  sparse(I, J, vals, m, n)
end

# the lower bound of the oracle, for the objective error of a Krylov atom
function nuclear_case(family, name, G, r; methods, T = Float64, reference = true)
  C = SingularValue(IndBallL1(T(r)))
  ref = reference ? σmax(G) : NaN
  shape = "$(size(G, 1))x$(size(G, 2))" * (G isa SparseMatrixCSC ? " nnz=$(nnz(G))" : "")
  for method in methods
    f = if method === :full
      () -> lmo_atom(C, G)
    elseif method === :krylov
      () -> lmo_atom(C, G; method = :krylov, rtol = sqrt(eps(T)), maxiter = 150, fallback = false)
    elseif method === :partial        # the existing randomized subspace iteration of #45 on a dense copy;
      P = IndBallRank(1; partial = true)   # warm: the same input again, so the stored block is accepted at once
      Y = zeros(T, size(G))
      X = Matrix(G)
      () -> prox!(Y, P, X, 1.0)
    elseif method === :partialcold    # a new set object per call: no stored block, the workspace is allocated
      Y = zeros(T, size(G))
      X = Matrix(G)
      () -> prox!(Y, IndBallRank(1; partial = true), X, 1.0)
    end
    if method === :krylov                 # fallback = false: a call that does not converge is reported, not densified
      converged = try
        f(); true
      catch ex
        ex isa ErrorException || rethrow()
        false
      end
      if !converged
        push!(rows, join((label, family, name, "krylov", T, shape, "", "", "", "", "", "", "", "", "", "disabled", "false", "",
                          "did not reach rtol within maxiter = 150 with fallback = false: not timed, no dense fallback used; blas=$(BLAS.get_num_threads())"), ","))
        @printf("%-10s %-22s krylov   %s: NOT CONVERGED (fallback = false)\n", family, name, shape)
        continue
      end
    end
    t, f1 = measure(f)
    err, steps, fb, conv, abytes = NaN, "", "", "", ""
    retained = method in (:partial, :partialcold) ? "" : string(Base.summarysize(C))
    method === :partial && (retained = string(Base.summarysize(P)))
    if method === :krylov
      a = lmo_atom(C, G; method = :krylov, rtol = sqrt(eps(T)), maxiter = 150, fallback = false)
      err = reference ? abs(dot(G, a) + r * ref) / (r * ref) : NaN
      e = ProximalBase._scale_exponent(ProximalBase._max_abs(G, T))
      _, _, _, _, st, ok = ProximalBase._gkl_top(G, T, e, sqrt(eps(T)), 150)
      steps, fb, conv, abytes = string(st), "disabled", string(ok), string(Base.summarysize(a))
    elseif method === :full
      a = lmo_atom(C, G)
      err = reference ? abs(dot(G, a) + r * ref) / (r * ref) : NaN
      abytes = string(Base.summarysize(a))
    end
    record!(method in (:partial, :partialcold) ? "projection-baseline" : family, name, string(method), T, shape, t; first = f1, steps, fallback = fb, converged = conv, atom_bytes = abytes, err,
            note = method in (:partial, :partialcold) ? "rank-1 projection of #45, not an LMO; dense copy made outside the timing; warm = same input again" : (G isa SparseMatrixCSC && method === :full ? "includes densification" : ""))
  end
end

sizes = suite == "memgate" ? Tuple{Int,Int}[] : suite == "quick" ? [(100, 100), (500, 500)] : [(100, 100), (500, 500), (1000, 1000), (2000, 500)]
for T in (Float64, Float32), (m, n) in sizes
  G = randn(Xoshiro(2), T, m, n)
  nuclear_case("nuclear-dense", "gaussian", G, 2.0; methods = (:full, :krylov, :partial, :partialcold), T)
end
for (m, n, p) in (suite == "memgate" ? Tuple{Int,Int,Float64}[] : suite == "quick" ? [(2000, 2000, 0.01)] : [(2000, 2000, 0.01), (3000, 1000, 0.02)])
  Gs = completion_gradient(Xoshiro(3), m, n, p)
  nuclear_case("nuclear-sparse", "completion", Gs, 10.0; methods = (:full, :krylov, :partial, :partialcold))
  Gr = sprandn(Xoshiro(4), m, n, p)
  nuclear_case("nuclear-sparse", "gaussian-sparse", Gr, 10.0; methods = (:full, :krylov))
end
if suite in ("large", "memgate")
  for (m, n, p) in ((10_000, 10_000, 0.01), (10_000, 3_000, 0.02))
    Gs = completion_gradient(Xoshiro(5), m, n, p)
    # no dense methods at this size: the reference is the Krylov result itself with a tight tolerance
    nuclear_case("nuclear-large", "completion", Gs, 10.0; methods = (:krylov,), reference = false)
  end
end

# ---- spectral sets ---------------------------------------------------------------------------
for T in (Float64, Float32), n in (suite == "memgate" ? Int[] : suite == "quick" ? [100, 500] : [100, 500, 1000])
  A = randn(Xoshiro(6), T, n, n); A = (A + A') / 2
  λ = eigvals(Symmetric(A))
  D = Spectral(IndSimplex(T(1)))
  for method in (:full, :krylov)
    f = () -> lmo_atom(D, A; method)
    t, f1 = measure(f)
    a = lmo_atom(D, A; method)
    record!("density", "gaussian", string(method), T, "n=$n", t; first = f1, atom_bytes = string(Base.summarysize(a)), err = abs(dot(A, a) - λ[1]) / abs(λ[1]),
            fallback = method === :krylov ? "enabled" : "", note = method === :krylov ? "dense input: fallback = true, whether it was used is not recorded" : "")
  end
  for k in (1, 5, 20, n ÷ 2, n - 1)
    F = Spectral(Constrained(IndBox(0, 1), IndHyperplane(1, k)))
    f = () -> lmo_atom(F, A)
    t, f1 = measure(f)
    a = lmo_atom(F, A)
    ref = sum(λ[1:k])
    record!("fantope", "gaussian", "full", T, "n=$n;k=$k", t; first = f1, atom_bytes = string(Base.summarysize(a)), err = abs(dot(A, a) - ref) / max(abs(ref), 1),
            note = "factor storage $(sizeof(a.U)) B")
  end
  # the dense oracle and the support value, without and with materialization
  S = zeros(T, n, n)
  t, f1 = measure(() -> lmo!(S, D, A))
  record!("density", "dense-lmo!", "full", T, "n=$n", t; first = f1)
  t, f1 = measure(() -> Conjugate(D)(A))
  record!("density", "conjugate", "eigvals", T, "n=$n", t; first = f1, err = abs(Conjugate(D)(A) - λ[end]) / abs(λ[end]))
end

# ---- storage ----------------------------------------------------------------------------------
for n in (1_000, 10_000)
  G = completion_gradient(Xoshiro(7), n, n, 0.002)
  a = lmo_atom(SingularValue(IndBallL1(1.0)), G; method = :krylov, fallback = false)
  push!(rows, join((label, "storage", "nuclear atom", "krylov", "Float64", "$(n)x$n", "", "", "", "", Base.summarysize(a), "", @sprintf("%.1f", Sys.maxrss() / 2^20), "", "", "disabled", "", "",
                    "atom size only (atom_bytes); the Julia heap bytes allocated by a call are in the bytes column of the timing rows (not a peak; native workspace not counted); dense matrix would be $(8n * n) B"), ","))
  @printf("storage: %dx%d atom %d B, dense %d B\n", n, n, Base.summarysize(a), 8n * n)
end

bad = [r for r in rows if count(==(','), r) != count(==(','), header)]   # no quoting: a comma in a field would shift the columns
isempty(bad) || error("$(length(bad)) rows do not have $(count(==(','), header) + 1) fields, first: $(first(bad))")
open(out, "w") do io
  println(io, header)
  foreach(r -> println(io, r), rows)
end
