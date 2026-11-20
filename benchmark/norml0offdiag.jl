# Paired in-process timing of NormL0OffDiag against the all-entry NormL0 on the same symmetric
# matrix: prox!, value_and_prox!, the value and the in-place forms of the first two (each copies the
# input into the buffer first, for both functions), for p = 8, 32, 64 and Float32 and Float64. About
# half of the off-diagonal entries lie above the threshold. The two functions are timed in
# alternating samples of one loop each, and the table gives the medians of the samples.
#
#   julia --threads=1 --project=. benchmark/norml0offdiag.jl
#
# It loads ProximalBase, LinearAlgebra and Random. PB_SAMPLES sets the number of samples.
using ProximalBase, LinearAlgebra, Random
BLAS.set_num_threads(1)

const SAMPLES = parse(Int, get(ENV, "PB_SAMPLES", "41"))
const TARGET_NS = 2.0e6
const SINK = Ref(0.0)

function sample_ns(op::F, reps::Int) where {F}
  s = 0.0
  t0 = time_ns()
  for _ in 1:reps
    s += op()
  end
  t1 = time_ns()
  SINK[] = s
  (t1 - t0) / reps
end

# The medians of alternating samples of the two operations, in nanoseconds per call.
function paired(a::A, b::B) where {A, B}
  for _ in 1:200
    a(); b()
  end
  reps_a = max(1, round(Int, TARGET_NS / max(sample_ns(a, 50), 1.0)))
  reps_b = max(1, round(Int, TARGET_NS / max(sample_ns(b, 50), 1.0)))
  ta, tb = Float64[], Float64[]
  for k in 1:SAMPLES
    if isodd(k)
      push!(ta, sample_ns(a, reps_a)); push!(tb, sample_ns(b, reps_b))
    else
      push!(tb, sample_ns(b, reps_b)); push!(ta, sample_ns(a, reps_a))
    end
  end
  sort!(ta)[(SAMPLES + 1) ÷ 2], sort!(tb)[(SAMPLES + 1) ÷ 2]
end

function allocated(op::F) where {F}
  op(); op()
  @allocated op()
end

function symmetric_matrix(rng, ::Type{T}, p) where {T}
  A = randn(rng, T, p, p)
  A .+ A'
end

function run()
  println("Julia $VERSION, samples per case $SAMPLES")
  println(rpad("T", 8), rpad("p", 5), rpad("kept", 7), rpad("operation", 16), rpad("offdiag ns", 12),
          rpad("NormL0 ns", 12), rpad("ratio", 8), rpad("vs NormL0 prox!", 17), "allocated (offdiag)")
  for T in (Float32, Float64), p in (8, 32, 64)
    rng = Xoshiro(1058 + p)
    X = symmetric_matrix(rng, T, p)
    Y = similar(X)
    W = similar(X)
    Zf = similar(X)
    Zg = similar(X)
    λ, γ = T(0.3), T(1)
    f, g = NormL0OffDiag(λ), NormL0(λ)
    prox!(Y, f, X, γ)
    prox!(W, g, X, γ)
    for j in 1:p, i in 1:p
      i == j || isequal(Y[i, j], W[i, j]) || error("the maps differ at ($i, $j)")
    end
    kept = round((count(!iszero, Y) - count(!iszero, diag(Y))) / (p * (p - 1)); digits = 2)
    ops = (
      ("prox!", () -> (prox!(Y, f, X, γ); Float64(Y[1, 2])), () -> (prox!(W, g, X, γ); Float64(W[1, 2]))),
      ("value_and_prox!", () -> Float64(value_and_prox!(Y, f, X, γ)), () -> Float64(value_and_prox!(W, g, X, γ))),
      ("value", () -> Float64(f(X)), () -> Float64(g(X))),
      ("prox! in place", () -> (copyto!(Zf, X); prox!(Zf, f, Zf, γ); Float64(Zf[1, 2])),
                         () -> (copyto!(Zg, X); prox!(Zg, g, Zg, γ); Float64(Zg[1, 2]))),
      ("v&p! in place", () -> (copyto!(Zf, X); Float64(value_and_prox!(Zf, f, Zf, γ))),
                        () -> (copyto!(Zg, X); Float64(value_and_prox!(Zg, g, Zg, γ)))),
    )
    base = nothing
    for (name, a, b) in ops
      ta, tb = paired(a, b)
      name == "prox!" && (base = tb)
      println(rpad(T, 8), rpad(p, 5), rpad(kept, 7), rpad(name, 16), rpad(round(ta; digits = 1), 12),
              rpad(round(tb; digits = 1), 12), rpad(round(ta / tb; digits = 2), 8),
              rpad(round(ta / base; digits = 2), 17), allocated(a))
    end
  end
end

run()
