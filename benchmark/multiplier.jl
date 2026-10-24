# The search for the multiplier of Constrained on a hyperplane (issue #77), and
# the threshold of the projection onto the simplex. Run from the repository root:
#
#   julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=benchmark benchmark/multiplier.jl
#
# 1. One call from zero, over independent draws: the first step of master (the
#    lower bound on the distance to the root, which a function without known
#    slopes still takes) against the first step from the exact slope.
# 2. The calls of a solver on slowly changing points: master, which searches
#    from zero on every call, against this branch, which starts from the last
#    multiplier.
# 3. FISTA on the zero-sum lasso, master against this branch.
# 4. One median selection over the 2n breakpoints of the zero-sum lasso, the step
#    of the breakpoint search that #77 proposed, against the whole search.
# 5. The threshold of the projection onto the simplex: master's Condat pass
#    against the pass that hands over to Michelot's algorithm.
# Every comparison runs its sides on the same data, interleaved over rounds,
# and keeps the fastest round of each: other jobs on a shared node slow down
# phases of a run, and interleaving spreads that over both sides. Passes are
# calls of the scalar proximal operator divided by n, the last pass, which
# writes the result, included.

using BenchmarkTools, LinearAlgebra, Random, Printf
import ProximalBase as PB

Random.seed!(1)
BLAS.set_num_threads(1)

const CALLS = Ref(0)
struct Counted{F} <: PB.SeparableFunction       # counts the passes; keeps the slopes
  f::F
end
struct Slopeless{F} <: PB.SeparableFunction     # hides the slopes: master's first step
  f::F
end
for W in (:Counted, :Slopeless)
  @eval PB.value_entry(g::$W, v, i) = PB.value_entry(g.f, v, i)
  @eval PB.isconvex(::Type{<:$W}) = true
end
PB.prox_entry(g::Counted, v, i, γ) = (CALLS[] += 1; PB.prox_entry(g.f, v, i, γ))
PB._has_prox_slope(::Type{<:Counted{F}}) where {F} = PB._has_prox_slope(F)
PB._prox_entry_slope(g::Counted, v, i, γ) = (CALLS[] += 1; PB._prox_entry_slope(g.f, v, i, γ))
PB.prox_entry(g::Slopeless, v, i, γ) = PB.prox_entry(g.f, v, i, γ)

cold!(c) = (c.last[] = NaN; c)
# 20 slowly changing points, made before the timing
points(x, noise) = [x .+ (0.002 * t) .* noise for t in 1:20]
calls_cold(c, y, xs) = (for xt in xs; cold!(c); PB.prox!(y, c, xt, 0.7); end)
calls_warm(c, y, xs) = (for xt in xs; PB.prox!(y, c, xt, 0.7); end)
# the passes of each call, from zero or, after a first round, from the last multiplier
function passes(c, y, xs; warm = false)
  warm && calls_warm(c, y, xs)
  map(xs) do xt
    warm || cold!(c)
    CALLS[] = 0
    PB.prox!(y, c, xt, 0.7)
    CALLS[] / length(xt)
  end
end
avg(v) = sum(v) / length(v)
# the fastest of 7 interleaved rounds of each side, per call
function interleaved(sides...; rounds = 7)
  foreach(s -> s(), sides)
  best = fill(Inf, length(sides))
  for _ in 1:rounds, (k, s) in enumerate(sides)
    best[k] = min(best[k], @elapsed s())
  end
  best
end

# the cases, on vectors of length n
cases(n) = [("simplex r = 1", PB.IndNonnegative(), PB.IndHyperplane(1, 1)),
            ("simplex r = n/10", PB.IndNonnegative(), PB.IndHyperplane(1, n / 10)),
            ("capped simplex k = 10", PB.IndBox(0, 1), PB.IndHyperplane(1, 10)),
            ("capped simplex k = n/2", PB.IndBox(0, 1), PB.IndHyperplane(1, n / 2)),
            ("zero-sum lasso λ = 0.5", PB.NormL1(0.5), PB.IndHyperplane(1, 0)),
            ("ℓ1 on [-1, 1], a general normal", PB.Constrained(PB.NormL1(0.3), PB.IndBox(-1, 1)), PB.IndHyperplane(randn(n), 3.)),
            ("elastic net on the simplex", PB.Constrained(PB.ElasticNet(0.1, 1.), PB.IndNonnegative()), PB.IndHyperplane(1, 1))]
label(n) = n == 1_000 ? "10³" : "10⁵"

println("## 1. One call from zero, over independent draws")
println()
println("200 draws of x at n = 10³ and 20 at n = 10⁵, the same for both sides: the mean time and passes per call, ",
        "and the share of draws on which the exact first step takes more passes than master's.")
println()
println("| case | master's first step (μs, passes) | exact first step (μs, passes) | more passes |")
println("|:-----|---:|---:|---:|")
for n in (1_000, 100_000)
  xs, y = [randn(n) for _ in 1:(n == 1_000 ? 200 : 20)], zeros(n)
  for (name, f, H) in cases(n)
    cm, cn = PB.Constrained(Slopeless(f), H), PB.Constrained(f, H)
    tm, tn = interleaved(() -> calls_cold(cm, y, xs), () -> calls_cold(cn, y, xs)) ./ length(xs)
    pm = passes(PB.Constrained(Slopeless(Counted(f)), H), y, xs)
    pn = passes(PB.Constrained(Counted(f), H), y, xs)
    @printf("| %s, n = %s | %.1f, %.1f | %.1f, %.1f | %.0f%% |\n", name, label(n), 1e6tm, avg(pm), 1e6tn, avg(pn),
            100avg(pn .> pm))
  end
end

println()
println("## 2. The calls of a solver")
println()
println("The same 20 slowly changing points for both sides: the time and passes per call.")
println()
println("| case | master, each call from zero (μs, passes) | this branch, from the last multiplier (μs, passes) | speedup |")
println("|:-----|---:|---:|---:|")
for n in (1_000, 100_000)
  x, noise, y = randn(n), randn(n), zeros(n)
  xs = points(x, noise)
  for (name, f, H) in cases(n)
    cm, cw = PB.Constrained(Slopeless(f), H), PB.Constrained(f, H)
    tm, tw = interleaved(() -> calls_cold(cm, y, xs), () -> calls_warm(cw, y, xs)) ./ length(xs)
    pm = passes(PB.Constrained(Slopeless(Counted(f)), H), y, xs)
    pw = passes(PB.Constrained(Counted(f), H), y, xs; warm = true)
    @printf("| %s, n = %s | %.1f, %.1f | %.1f, %.1f | %.2f× |\n", name, label(n), 1e6tm, avg(pm), 1e6tw, avg(pw), tm / tw)
  end
end

println()
println("## 3. FISTA on the zero-sum lasso")
println()
function fista!(β, f, g, iters; reset = false)
  L = PB.lipschitz(f)
  βold, z, grad = similar(β), copy(β), similar(β)
  t = 1.0
  for _ in 1:iters
    PB.gradient!(grad, f, z)
    copyto!(βold, β)
    β .= z .- grad ./ L
    reset && cold!(g)
    PB.prox!(β, g, β, 1 / L)
    tnew = (1 + sqrt(1 + 4t^2)) / 2
    z .= β .+ ((t - 1) / tnew) .* (β .- βold)
    t = tnew
  end
  β
end
println("| problem | master, each search from zero (ms, passes per call) | this branch, from the last multiplier (ms, passes per call) |")
println("|:-----|---:|---:|")
for (n, p) in ((100, 20_000), (50, 100_000))         # the proximal step a large part of each iteration
  W = exp.(randn(n, p))
  W ./= sum(W; dims = 2)
  Z = log.(W)
  Z .-= sum(Z; dims = 1) ./ n
  y = Z[:, 1:5] * [1.0, -0.8, 0.6, -0.5, -0.3] .+ 0.2 .* randn(n)
  y .-= sum(y) / n
  f = (1 / n) * PB.LeastSquaresLoss(y, Z)
  λ = 0.3 * maximum(abs.(Z' * y)) / n
  g0, g1 = PB.Constrained(Slopeless(PB.NormL1(λ)), PB.IndHyperplane(1, 0)), PB.Constrained(PB.NormL1(λ), PB.IndHyperplane(1, 0))
  tc, tw = interleaved(() -> fista!(zeros(p), f, g0, 500; reset = true), () -> fista!(zeros(p), f, g1, 500); rounds = 3)
  pc, pw = map((true, false)) do reset
    h = Counted(PB.NormL1(λ))
    gc = PB.Constrained(reset ? Slopeless(h) : h, PB.IndHyperplane(1, 0))
    CALLS[] = 0
    fista!(zeros(p), f, gc, 500; reset)
    CALLS[] / p / 500
  end
  @printf("| n = %d, p = %d, 500 iterations | %.1f, %.1f | %.1f, %.1f |\n", n, p, 1e3tc, pc, 1e3tw, pw)
end

println()
println("## 4. One median selection over the breakpoints, against the whole search")
println()
function breakpoints!(bp, x, t)
  @inbounds for i in eachindex(x)
    bp[2i - 1] = x[i] - t
    bp[2i] = x[i] + t
  end
  bp
end
println("| n | fill 2n breakpoints and select their median (μs) | the whole search from zero (μs) |")
println("|---:|---:|---:|")
for n in (1_000, 100_000)
  x, y = randn(n), zeros(n)
  bp = zeros(2n)
  c = PB.Constrained(PB.NormL1(0.5), PB.IndHyperplane(1, 0))
  reps = n == 1_000 ? 2000 : 20
  ts, tc = interleaved(() -> (for _ in 1:reps; PB._select_desc!(breakpoints!(bp, x, 0.35), 1, 2n, n); end),
                       () -> (for _ in 1:reps; cold!(c); PB.prox!(y, c, x, 0.7); end)) ./ reps
  @printf("| %d | %.1f | %.1f |\n", n, 1e6ts, 1e6tc)
end

println()
println("## 5. The threshold of the projection onto the simplex")
println()
# master's _simplex_threshold! (f5a7ec4), Condat's algorithm with a division per update
function master_threshold!(buf::AbstractVector{T}, x::AbstractVector, a::T) where {T}
  n = length(x)
  buf[1] = x[1]
  nt, nv = 0, 1
  ρ = x[1] - a
  @inbounds for j in 2:n
    y = x[j]
    if y > ρ
      ρ += (y - ρ) / (nv + 1)
      if ρ > y - a
        nv += 1
        buf[nt + nv] = y
      else
        nt += nv
        nv = 1
        buf[nt + 1] = y
        ρ = y - a
      end
    end
  end
  w = 0
  @inbounds for i in 1:nt
    y = buf[i]
    if y > ρ
      w += 1
      buf[w] = y
      ρ += (y - ρ) / (w + nv)
    end
  end
  @inbounds for i in 1:nv
    buf[w + i] = buf[nt + i]
  end
  len = w + nv
  changed = true
  @inbounds while changed
    changed = false
    i = 1
    while i <= len
      y = buf[i]
      if y <= ρ
        buf[i] = buf[len]
        len -= 1
        len == 0 && return ρ
        ρ += (ρ - y) / len
        changed = true
      else
        i += 1
      end
    end
  end
  ρ
end
println("| case | master (μs) | this branch (μs) | speedup | difference in τ |")
println("|:-----|---:|---:|---:|---:|")
for n in (1_000, 100_000)
  x, buf = randn(n), zeros(n)
  s = n == 1_000 ? "10³" : "10⁵"
  for (rs, r) in (("1", 1.0), ("100", 100.0), ("n/100", n / 100), ("n/10", n / 10), ("n/2", n / 2))
    reps = n == 1_000 ? 2000 : 50
    tm, tn = interleaved(() -> (for _ in 1:reps; master_threshold!(buf, x, r); end),
                         () -> (for _ in 1:reps; PB._simplex_threshold!(buf, x, eachindex(x), r); end)) ./ reps
    d = abs(master_threshold!(buf, x, r) - PB._simplex_threshold!(buf, x, eachindex(x), r))
    @printf("| r = %s, n = %s | %.1f | %.1f | %.2f× | %.0e |\n", rs, s, 1e6tm, 1e6tn, tm / tn, d)
  end
end
