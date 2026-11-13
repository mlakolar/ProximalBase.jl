# PB #911 contract probes (disposable): aliasing, f.tmp isolation, concurrent calls on a shared loss,
# and reuse of the B2 vector layer by a second, non-squared ScalarLoss (design sketch only).
# Run with threads:  julia -t 4 --startup-file=no <this file>
include(joinpath(@__DIR__, "envsetup.jl"))
using Random, Printf
using ProximalBase: LeastSquaresLoss
print_provenance()
function loadvariant(name)
  dst = joinpath(mktempdir(), "src")
  cp(joinpath(@__DIR__, "variants", name, "src"), dst)
  w = Module(Symbol(name))
  Base.include(w, joinpath(dst, "ProximalSolvers.jl"))
  Base.invokelatest(getfield, w, :ProximalSolvers)
end
const VS = ["B1", "B2", "B3", "H"]
const V = Dict(nm => loadvariant(nm) for nm in VS)
rng = Xoshiro(1)
n, p = 50, 7
for T in (Float64, Float32)
  X = randn(rng, T, n, p); Y = randn(rng, T, n); x = randn(rng, T, p); z = X * x
  f = (one(T) / T(n)) * LeastSquaresLoss(Y, X)
  for nm in VS
    O = V[nm].PBOracle
    # aliasing: exact and partial overlap of the writable buffers with the inputs and with each other
    buf = zeros(T, 2n + p)
    cases = [("d === z", () -> (zz = copy(z); O.value_and_gradient_at_image!(zeros(T, p), zz, f, zz))),
             ("d partial-overlaps z (views of one buffer)", () -> (zv = view(buf, 1:n); copyto!(zv, z); O.value_and_gradient_at_image!(zeros(T, p), view(buf, 2:n+1), f, zv))),
             ("g overlaps d (views of one buffer)", () -> O.value_and_gradient_at_image!(view(buf, n+1:n+p), view(buf, n+1:2n), f, z)),
             ("d === Y", () -> O.value_and_gradient_at_image!(zeros(T, p), f.f.Y, f, z)),
             ("g aliases X (vec view of X)", () -> O.value_and_gradient_at_image!(view(vec(f.f.X), 1:p), zeros(T, n), f, z)),
             ("d === f.tmp (PB storage)", () -> O.value_and_gradient_at_image!(zeros(T, p), f.f.tmp, f, z)),
             ("g partial-overlaps f.tmp (view)", () -> O.value_and_gradient_at_image!(view(f.f.tmp, 1:p), zeros(T, n), f, z)),
             ("distinct buffers", () -> O.value_and_gradient_at_image!(zeros(T, p), zeros(T, n), f, z))]
    for (lab, th) in cases
      r = try
        th(); "accepted"
      catch e
        "refused: " * sprint(showerror, e)
      end
      @printf "%-7s %-3s alias %-44s %s\n" T nm lab r
    end
    # f.tmp is never read or written: poison it and compare
    g1 = zeros(T, p); d1 = zeros(T, n); g2 = zeros(T, p); d2 = zeros(T, n)
    v1 = (O.value_at_image(f, z), O.value_and_gradient_at_image!(g1, d1, f, z), O.value_at_residual(f, Y .- z))
    fill!(f.f.tmp, T(NaN)); before = copy(f.f.tmp)
    v2 = (O.value_at_image(f, z), O.value_and_gradient_at_image!(g2, d2, f, z), O.value_at_residual(f, Y .- z))
    ok = isequal(v1, v2) && g1 == g2 && isequal(f.f.tmp, before)
    @printf "%-7s %-3s f.tmp poisoned with NaN: results identical and f.tmp untouched = %s\n" T nm ok
    # concurrent calls sharing f (each task with its own buffers)
    nt = Threads.nthreads()
    gref = zeros(T, p); dref = zeros(T, n)
    ref = O.value_and_gradient_at_image!(gref, dref, f, z); vref = O.value_at_image(f, z)
    rres = Y .- z; resref = O.value_at_residual(f, rres)
    res = fill(true, 4nt)
    Threads.@threads for k in 1:4nt
      gk = zeros(T, p); dk = zeros(T, n)
      for _ in 1:200
        fill!(gk, T(NaN)); fill!(dk, T(NaN))
        vk = O.value_and_gradient_at_image!(gk, dk, f, z)
        res[k] &= isequal(vk, ref) && isequal(gk, gref) && isequal(dk, dref) &&
                  isequal(O.value_at_image(f, z), vref) && isequal(O.value_at_residual(f, rres), resref)
      end
    end
    @printf "%-7s %-3s concurrent calls on a shared f with %d threads x 4 tasks each x 200 (value, g, d, value_at_image, value_at_residual compared): all equal = %s\n" T nm nt all(res)
  end
end
# The native Xᵀd product before the scale (review r1 finding 3): finite inputs and residual, but an
# intermediate that leaves the range. Reference: α Xᵀ(z - Y) in BigFloat.
for nm in VS
  O = V[nm].PBOracle
  for (lab, X, Y, z, α) in (("Xᵀd overflows before a tiny α", fill(2.0^1023, 1, 1), [0.0], [4.0], 2.0^-1022),
                            ("Xᵀd underflows before a large α", fill(2.0^-600, 1, 1), [0.0], [2.0^-600], 2.0^600))
    g = zeros(1); d = zeros(1)
    v = O.value_and_gradient_at_image!(g, d, α * LeastSquaresLoss(Y, X), z)
    ref = big(α) * (big(X[1]) * (big(z[1]) - big(Y[1])))
    @printf "Float64 %-3s gradient range: %-34s g = %s, reference %s\n" nm lab repr(g[1]) string(Float64(ref))
  end
end
# B2 reuse by a second ScalarLoss with no new numerical code: the documented example of
# docs/src/scalar_losses.md, through the generic vector layer (no recovery hook: the ordinary sum)
struct DoubleSquared <: ProximalBase.ScalarLoss end
(::DoubleSquared)(η::Real, y::Real) = (η - y)^2
ProximalBase.deriv(::DoubleSquared, η::Real, y::Real) = 2 * (η - y)
ProximalBase.deriv2(::DoubleSquared, η::Real, y::Real) = 2.0
ProximalBase.curvature(::DoubleSquared) = 2.0
O2 = V["B2"].PBOracle
for T in (Float64, Float32)
  η = randn(Xoshiro(2), T, 1000); y = randn(Xoshiro(3), T, 1000); d = similar(η)
  v = O2.loss_value(DoubleSquared(), η, y); w = O2.loss_value_and_deriv!(d, DoubleSquared(), η, y)
  ref = sum(abs2, big.(η) .- big.(y)); refd = 2 .* (η .- y)
  @printf "%-7s B2 layer with DoubleSquared: rel err value %.2e, value+deriv %.2e, deriv exact = %s, inferred %s\n" T Float64(abs(big(v) - ref) / ref) Float64(abs(big(w) - ref) / ref) (d == refd) string(Base.return_types(O2.loss_value_and_deriv!, (Vector{T}, DoubleSquared, Vector{T}, Vector{T})))
  # specialization: calls left in the optimized loop
  ci = Base.code_typed(O2._loss_sum_deriv!, (Vector{T}, DoubleSquared, Vector{T}, Vector{T}); optimize = true)[1][1]
  inv = [string(st.args[1] isa Core.MethodInstance ? st.args[1].def.name : (st.args[1] isa Core.CodeInstance ? st.args[1].def.def.name : st.args[1])) for st in ci.code if Meta.isexpr(st, :invoke)]
  @printf "%-7s B2 _loss_sum_deriv!(DoubleSquared) non-inlined calls: %d %s\n" T length(inv) join(unique(inv), ";")
  big1 = fill(T(1e30), 100)
  @printf "%-7s B2 DoubleSquared, terms that overflow (no recovery hook): %s (exact %s)\n" T repr(O2.loss_value(DoubleSquared(), big1 .* 3, -big1)) string(Float64(sum(abs2, big.(big1 .* 3) .- big.(-big1))))
end
