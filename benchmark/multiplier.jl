# The search for the multiplier of Constrained on a hyperplane (issue #77), and
# the threshold of the projection onto the simplex. Run from the repository root:
#
#   julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=benchmark benchmark/multiplier.jl
#
# 1. One call from zero: the first step of master (the lower bound on the
#    distance to the root, which a function without known slopes still takes)
#    against the first step from the exact slope; and the calls of a solver on
#    slowly changing points, which start from the last multiplier.
# 2. FISTA on the zero-sum lasso, each search from zero against each from the
#    last multiplier.
# 3. One median selection over the 2n breakpoints of the zero-sum lasso, the step
#    of the breakpoint search that #77 proposed, against the whole search.
# 4. The threshold of the projection onto the simplex: master's Condat pass
#    against the pass that hands over to Michelot's algorithm.
# Times are medians from BenchmarkTools, or the best of five runs of a sequence;
# passes are calls of the scalar proximal operator divided by n, the last pass,
# which writes the result, included.

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
t_cold(c, y, x) = (cold!(c); PB.prox!(y, c, x, 0.7); @belapsed PB.prox!($y, $c, $x, 0.7) setup = cold!($c) evals = 1)
function passes_cold(c, y, x)
  cold!(c)
  CALLS[] = 0
  PB.prox!(y, c, x, 0.7)
  CALLS[] / length(x)
end
sequence(c, y, x, noise) = (for t in 1:20; PB.prox!(y, c, x .+ (0.002 * t) .* noise, 0.7); end)
function warm(f, H, y, x, noise)
  c = PB.Constrained(f, H)
  sequence(c, y, x, noise)
  t = minimum(@elapsed(sequence(c, y, x, noise)) for _ in 1:5) / 20
  cc = PB.Constrained(Counted(f), H)
  sequence(cc, y, x, noise)
  CALLS[] = 0
  sequence(cc, y, x, noise)
  t, CALLS[] / length(x) / 20
end

println("## 1. The search, one call from zero and the calls of a solver")
println()
println("| case | master's first step (μs, passes) | exact first step (μs, passes) | from the last multiplier (μs, passes) |")
println("|:-----|---:|---:|---:|")
for n in (1_000, 100_000)
  x, noise, y = randn(n), randn(n), zeros(n)
  s = n == 1_000 ? "10³" : "10⁵"
  cases = [("simplex r = 1", PB.IndNonnegative(), PB.IndHyperplane(1, 1)),
           ("simplex r = n/10", PB.IndNonnegative(), PB.IndHyperplane(1, n / 10)),
           ("capped simplex k = 10", PB.IndBox(0, 1), PB.IndHyperplane(1, 10)),
           ("capped simplex k = n/2", PB.IndBox(0, 1), PB.IndHyperplane(1, n / 2)),
           ("zero-sum lasso λ = 0.5", PB.NormL1(0.5), PB.IndHyperplane(1, 0)),
           ("ℓ1 on [-1, 1], a general normal", PB.Constrained(PB.NormL1(0.3), PB.IndBox(-1, 1)), PB.IndHyperplane(randn(n), 3.)),
           ("elastic net on the simplex", PB.Constrained(PB.ElasticNet(0.1, 1.), PB.IndNonnegative()), PB.IndHyperplane(1, 1))]
  for (name, f, H) in cases
    tm = t_cold(PB.Constrained(Slopeless(f), H), y, x)
    pm = passes_cold(PB.Constrained(Slopeless(Counted(f)), H), y, x)
    tn = t_cold(PB.Constrained(f, H), y, x)
    pn = passes_cold(PB.Constrained(Counted(f), H), y, x)
    tw, pw = warm(f, H, y, x, noise)
    @printf("| %s, n = %s | %.1f, %.0f | %.1f, %.0f | %.1f, %.1f |\n", name, s, 1e6tm, pm, 1e6tn, pn, 1e6tw, pw)
  end
end

println()
println("## 2. FISTA on the zero-sum lasso")
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
println("| problem | searches from zero (ms, passes per call) | from the last multiplier (ms, passes per call) |")
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
  res = Float64[]
  for reset in (true, false)
    g = PB.Constrained(PB.NormL1(λ), PB.IndHyperplane(1, 0))
    fista!(zeros(p), f, g, 500; reset)
    push!(res, minimum(@elapsed(fista!(zeros(p), f, g, 500; reset)) for _ in 1:3))
    gc = PB.Constrained(Counted(PB.NormL1(λ)), PB.IndHyperplane(1, 0))
    CALLS[] = 0
    fista!(zeros(p), f, gc, 500; reset)
    push!(res, CALLS[] / p / 500)
  end
  @printf("| n = %d, p = %d, 500 iterations | %.1f, %.1f | %.1f, %.1f |\n", n, p, 1e3res[1], res[2], 1e3res[3], res[4])
end

println()
println("## 3. One median selection over the breakpoints, against the whole search")
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
  ts = @belapsed PB._select_desc!(breakpoints!($bp, $x, 0.35), 1, $(2n), $n)
  c = PB.Constrained(PB.NormL1(0.5), PB.IndHyperplane(1, 0))
  @printf("| %d | %.1f | %.1f |\n", n, 1e6ts, 1e6t_cold(c, y, x))
end

println()
println("## 4. The threshold of the projection onto the simplex")
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
    tm = @belapsed master_threshold!($buf, $x, $r)
    tn = @belapsed PB._simplex_threshold!($buf, $x, $(eachindex(x)), $r)
    d = abs(master_threshold!(buf, x, r) - PB._simplex_threshold!(buf, x, eachindex(x), r))
    @printf("| r = %s, n = %s | %.1f | %.1f | %.2f× | %.0e |\n", rs, s, 1e6tm, 1e6tn, tm / tn, d)
  end
end
