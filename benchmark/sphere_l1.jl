# Benchmarks of the sphere proximal map of the ℓ1 norm (issue #459).
#
#   julia --project=<env with ProximalBase and BenchmarkTools> benchmark/sphere_l1.jl <label> <out.csv> [quick]
#
# The same script runs against any ProximalBase checkout (it uses only the public API), so a
# checkout before #441 and one after give comparable rows. Inputs are drawn with a fixed seed
# before timing; each case is compiled by a first call before it is measured; aliased calls
# copy the argument in the BenchmarkTools setup, outside the timed region. One thread.
using ProximalBase, BenchmarkTools, Random, LinearAlgebra, Printf

label, out = ARGS[1], ARGS[2]
quick = length(ARGS) >= 3 && ARGS[3] == "quick"
BenchmarkTools.DEFAULT_PARAMETERS.seconds = quick ? 0.1 : 0.4
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100_000

rows = String[]
header = "label,output,T,weights,branch,scale,n,op,alias,median_ns,min_ns,allocs,bytes"

function record!(rows, label, output, T, weights, branch, scale, n, op, alias, t)
  push!(rows, join((label, output, T, weights, branch, scale, n, op, alias,
                    @sprintf("%.1f", median(t).time), @sprintf("%.1f", minimum(t).time), t.allocs, t.memory), ","))
  @printf("%-8s %-22s %-8s %-6s %-4s %-8s %6d %-16s %-5s %10.1f ns  %d allocs\n",
          label, output, T, weights, branch, scale, n, op, alias, median(t).time, t.allocs)
end

opname(op) = op === prox! ? "prox!" : "value_and_prox!"

# λ, γ and the input scale for a branch: "pos" has margins above the thresholds (the shrink),
# "fb" has none (the fallback); "extreme" puts the thresholds and entries near the bottom of
# the range, where the threshold product is formed by the exceptional path and the scaling
# splits mantissas and exponents.
function params(::Type{T}, branch, scale) where {T}
  λ = branch == "pos" ? T(1) : T(20)
  scale == "ordinary" && return (λ, T(1), T(1))
  T === Float64 ? (λ * T(2)^-500, T(2)^-600, T(2)^-1100) : (λ * T(2)^-60, T(2)^-80, T(2)^-140)
end

function dense_cases!(rows, label; sizes, types, outputs, scales = ("ordinary",))
  for T in types, n in sizes, wt in ("none", "vector"), branch in ("pos", "fb"), scale in scales
    rng = Xoshiro(459 + n)
    λ, γ, s = params(T, branch, scale)
    x = T.(randn(rng, n)) .* s
    f = wt == "none" ? NormL1(λ) : NormL1(λ, T.(0.5 .+ rand(rng, n)))
    c = Constrained(f, IndSphereL2(T(1)))
    for output in outputs, op in (prox!, value_and_prox!)
      y = output == "Vector" ? zeros(T, n) : SparseIterate(T, n)
      op(y, c, x, γ)
      t = @benchmark $op($y, $c, $x, $γ)
      record!(rows, label, output, T, wt, branch, scale, n, opname(op), false, t)
      output == "Vector" || op === prox! || continue
      # aliased: y === x, with a fresh copy of the argument per evaluation
      mk = output == "Vector" ? copy : SparseIterate
      t = @benchmark $op(z, $c, z, $γ) setup = (z = $mk($x)) evals = 1
      record!(rows, label, output, T, wt, branch, scale, n, opname(op), true, t)
    end
  end
end

function symmetric_cases!(rows, label; sizes, types)
  for T in types, p in sizes, branch in ("pos", "fb")
    rng = Xoshiro(4590 + p)
    λ, γ, _ = params(T, branch, "ordinary")
    A = T.(randn(rng, p, p))
    X = (A + A') / 2
    c = Constrained(NormL1(λ), IndSphereL2(T(1)))
    for (output, mk) in (("SymmetricSparseIterate", () -> SymmetricSparseIterate(T, p)),
                         ("Matrix(symmetric)", () -> zeros(T, p, p)))
      for op in (prox!, value_and_prox!)
        y = mk()
        op(y, c, X, γ)
        t = @benchmark $op($y, $c, $X, $γ)
        record!(rows, label, output, T, "none", branch, "ordinary", p, opname(op), false, t)
      end
    end
  end
end

if quick
  dense_cases!(rows, label; sizes = (1024, 65536), types = (Float64,), outputs = ("Vector",))
else
  dense_cases!(rows, label; sizes = (16, 256, 1024, 65536), types = (Float32, Float64), outputs = ("Vector", "SparseIterate"))
  dense_cases!(rows, label; sizes = (1024,), types = (Float32, Float64), outputs = ("Vector",), scales = ("extreme",))
  symmetric_cases!(rows, label; sizes = (32, 256), types = (Float32, Float64))
end

open(out, "w") do io
  println(io, header)
  foreach(r -> println(io, r), rows)
end
