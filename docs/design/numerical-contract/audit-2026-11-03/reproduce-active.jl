# Re-run the reproducers of the active entries of known-limitations.md.
#
#   julia --project=. docs/design/numerical-contract/audit-2026-11-03/reproduce-active.jl
#
# Each line prints PRESENT (the limitation still shows), ABSENT (it no longer
# shows: update the register) or ERROR (the probe itself failed: read it). The
# device entries (LIM-060 to LIM-063) run only when JLArrays and Adapt can be
# loaded, for example from an environment that has ProximalBase dev'd plus
# JLArrays; JLArrays runs device kernels on the CPU, so it is evidence about
# dispatch and kernel logic, not about real GPUs. This is an evidence script for
# reviewers, not part of the test suite.
using ProximalBase, LinearAlgebra, Random, Test
const PB = ProximalBase
const HAVE_JL = try
  @eval using JLArrays, Adapt
  true
catch
  false
end
const SHA = try
  strip(read(`git -C $(pkgdir(ProximalBase)) rev-parse HEAD`, String))
catch
  "unknown"
end
println("# ProximalBase ", pkgversion(ProximalBase), " at ", SHA, ", Julia ", VERSION, ", JLArrays: ", HAVE_JL)

const COUNT = Dict("PRESENT" => 0, "ABSENT" => 0, "ERROR" => 0, "SKIPPED" => 0)
short(x) = (s = replace(repr(x; context = :compact => true), '\n' => ' '); length(s) > 110 ? s[1:110] * "…" : s)
shorterr(e) = (s = replace(sprint(showerror, e), '\n' => ' '); length(s) > 110 ? s[1:110] * "…" : s)

"""check(f, id, what): f() returns (present::Bool, observed::String)."""
function check(f, id, what)
  status, obs = try
    present, o = f()
    (present ? "PRESENT" : "ABSENT", o)
  catch e
    ("ERROR", "probe error: " * shorterr(e))
  end
  COUNT[status] += 1
  println(rpad(id, 9), rpad(status, 8), what, " | ", obs)
end
skip(id, what, why) = (COUNT["SKIPPED"] += 1; println(rpad(id, 9), rpad("SKIPPED", 8), what, " | ", why))

"Run g(); return (threw::Bool, exception or result)."
attempt(g) = try
  (false, g())
catch e
  (true, e)
end
"Run a call that writes `y`; return (threw, exception or result, y changed)."
function writes(g, y)
  before = deepcopy(y)
  t, r = attempt(g)
  (t, r, !isequal(before, y))
end
const M = floatmax(Float64)
blk(f, x) = (SeparableSum(SqrNormL2(2.0), f), ([3.0], x), ([7.0], fill(7.0, size(x))))

# ---------------------------------------------------------------- failures (O6)
check("LIM-001", "Conjugate finite-result check at a valid step, later block") do
  s, pt, y = blk(Conjugate(SqrNormL2()), [1e300])
  t, r, ch = writes(() -> prox!(y, s, pt, 1e-10), y)
  (t && ch, "throws=$t $(t ? typeof(r) : "") earlier block written=$ch")
end
check("LIM-002", "Conjugate block at an invalid step caught after writing") do
  s, pt, y = blk(Conjugate(SqrNormL2()), [1.0])
  t, r, ch = writes(() -> prox!(y, s, pt, -1.0), y)
  (t && ch, "γ=-1: throws=$t earlier block written=$ch")
end
check("LIM-003", "search-found infeasibility in a later block writes earlier blocks") do
  s, pt, y = blk(Constrained(IndBox(0.0, 1.0), IndHyperplane(ones(2), 5.0)), [0.5, 0.5])
  t, r, ch = writes(() -> prox!(y, s, pt, 1.0), y)
  (t && ch, "throws=$t earlier block written=$ch")
end
check("LIM-004", "empty LinearConstrained set returns NaN at a NaN input / γ=Inf") do
  f = Constrained(IndBox(0.0, 1.0), IndHyperplane(1.0, 5.0))
  t1, r1 = attempt(() -> prox(f, [NaN, 0.0], 1.0))
  t2, r2 = attempt(() -> prox(f, [0.5, 0.5], Inf))
  t3, _ = attempt(() -> prox(f, [0.5, 0.5], 1.0))
  (!t1 || !t2, "NaN input: $(t1 ? "throws" : short(r1)); γ=Inf: $(t2 ? "throws" : short(r2)); control γ=1 throws=$t3")
end
check("LIM-005", "non-proximable later block: MethodError after writing") do
  s, pt, y = blk(LogisticLoss([1.0, 0.0], [1.0 0.0; 0.0 1.0]), [0.1, 0.2])
  t, r, ch = writes(() -> prox!(y, s, pt, 1.0), y)
  (t && r isa MethodError && ch, "$(t ? typeof(r) : "no error") earlier block written=$ch")
end
struct _Unknown{T} <: AbstractVector{T}
  data::Vector{T}
end
Base.size(x::_Unknown) = size(x.data)
Base.IndexStyle(::Type{<:_Unknown}) = IndexLinear()
Base.getindex(x::_Unknown, i::Int) = x.data[i]
Base.setindex!(x::_Unknown, v, i::Int) = (x.data[i] = v)
check("LIM-006", "tuple Conjugate destination conversion fails after writing") do
  c = Conjugate(SeparableSum(NormLinf(), NormLinf()))
  y = ([7.0], _Unknown(Int[7, 7]))
  t, r, ch = writes(() -> prox!(y, c, ([2.0], [1.0, 1.0]), 1.0), y)
  (t && ch, "$(t ? typeof(r) : "no error") block 1 written=$ch")
end
check("LIM-007", "quadratic_model! nested mismatch updates an earlier sibling") do
  Random.seed!(1)
  X = randn(3, 2); yv = [1.0, 0.0, 1.0]; β = zeros(2)
  f1, f2, f3 = LogisticLoss(yv, X), LogisticLoss(yv, X), LogisticLoss(yv, X)
  top = SeparableSum(f1, SeparableSum(f2, f3))
  bad = SeparableSum(quadratic_model(f1, β), SeparableSum(quadratic_model(f2, β)))
  bad.fs[1].y .= 99.0
  t, r = attempt(() -> quadratic_model!(bad, top, (β, (β, β))))
  ch = any(!=(99.0), bad.fs[1].y)
  (t && ch, "$(t ? typeof(r) : "no error"); first sibling updated=$ch")
end
check("LIM-008", "dense Symmetric destination: Base error after a partial write") do
  S = Symmetric(zeros(3, 3))
  t, r, ch = writes(() -> prox!(S, NormL1(1.0), ones(3, 3), 0.5), S)
  (t && ch, "$(t ? typeof(r) : "no error") written=$ch")
end
check("LIM-009", "fractional result into an Int SparseIterate: InexactError after writing") do
  si = SparseIterate(Int, 3)
  t, r, ch = writes(() -> prox!(si, NormL1(1.0), [2.0, 3.5, 4.0], 1.0), si)
  (t && (ch || !(r isa ArgumentError)), "$(t ? typeof(r) : "no error") written=$ch")
end
check("LIM-010", "complex source into a SparseIterate bypasses the real-only check") do
  s = SparseIterate(4)
  t, r, ch = writes(() -> copyto!(s, ComplexF64[1, 2im, 0, 0]), s)
  (t && !(r isa ArgumentError), "$(t ? typeof(r) : "no error") written=$ch")
end
check("LIM-011", "complex step raises MethodError, not ArgumentError") do
  t, r = attempt(() -> prox!(zeros(2), NormL1(1.0), [1.0, 2.0], 1.0 + 0im))
  (t && !(r isa ArgumentError), t ? string(typeof(r)) : "no error")
end
check("LIM-012", "complex adjoint product with a SparseIterate returns a result") do
  xs = SparseIterate(4); xs[2] = 1.0
  A = ComplexF64[1 2im 0 0; 0 1 0 0; 3 0 1 0; 0 0 0 1+im]
  t, r = attempt(() -> A' * xs)
  (!t, t ? "throws $(typeof(r))" : "returns $(eltype(r)) vector")
end
# ---------------------------------------------------------------- steps (P1-P3)
check("LIM-013", "negative and NaN steps accepted (NormL1, IndBox, prox_entry)") do
  a = prox(NormL1(1.0), [0.5, 0.5], -1.0)
  t2, b = attempt(() -> prox(IndBox(-1.0, 1.0), [3.0, 0.0], NaN))
  c = prox_entry(NormL1(1.0), 0.5, 1, -1.0)
  (true, "NormL1 γ=-1 -> $(short(a)); IndBox γ=NaN -> $(t2 ? "throws" : short(b)); prox_entry γ=-1 -> $c")
end
check("LIM-014", "value-dependent / wrong-class step rejections") do
  s, pt, y = blk(BernoulliLoss([0.0, 1.0]), [0.2, 0.7])
  t, r, ch = writes(() -> prox!(y, s, pt, -1.0), y)
  t2, r2 = attempt(() -> prox(NegLog(), [1.0], -1.0))
  (t && ch || (t2 && !(r2 isa ArgumentError)), "BernoulliLoss block: $(t ? typeof(r) : "no error") written=$ch; NegLog γ=-1: $(t2 ? typeof(r2) : "no error")")
end
check("LIM-015", "unnamed endpoint steps give NaN") do
  a = prox(SqrHingeLoss(), [0.0, 2.0], Inf)
  b = prox(SparseGroupNormL2(0.0, 0.0, [[1], [2]]), [1.0, 2.0], Inf)
  (any(isnan, a) || any(isnan, b), "SqrHingeLoss γ=Inf -> $(short(a)); SparseGroupNormL2(0,0) γ=Inf -> $(short(b))")
end
check("LIM-016", "subnormal / non-representable diagonal steps: infeasible or -Inf") do
  a = prox(Constrained(IndBox(-1.0, 1.0), IndHyperplane(1.0, 1.0)), [0.3, -1.2, 2.0], 1e-310)
  b = prox(IndHyperplane([1.0, 1.0], 1.0), [3.0, 3.0], [1e308, 1e308])
  c = prox(IndHyperplane(Float32[1, 1], 1f0), Float32[3, 3], [1e-50, 2e-50])
  bad = abs(sum(a) - 1) > 1e-12 || abs(sum(b) - 1) > 1e-12 || !all(isfinite, c)
  (bad, "box+hyperplane γ=1e-310 -> $(short(a)) (sum $(sum(a))); γ=[1e308,1e308] -> $(short(b)); Float32 with [1e-50,2e-50] -> $(short(c))")
end
# ---------------------------------------------------------------- searches (A3)
check("LIM-017", "Float32 multiplier search not certified, no PrecisionError") do
  Random.seed!(7)
  n = 20000
  a64 = randn(n); a64[1:n÷2] .= 0.0
  a64[n÷2+1:end] .= rand(n - n ÷ 2) .* (rand(n - n ÷ 2) .> 0.5) .- rand(n - n ÷ 2) .* (rand(n - n ÷ 2) .> 0.5)
  a = Float32.(a64); x = Float32.(randn(n)); b = Float32(sum(min.(Float64.(a), 0)) + 1)
  t, y = attempt(() -> prox(Constrained(IndBox(0f0, 1f0), IndHyperplane(a, b)), x, 0.7f0))
  t && return (!(y isa PrecisionError), "throws $(typeof(y))")
  y64 = prox(Constrained(IndBox(0.0, 1.0), IndHyperplane(Float64.(a), Float64(b))), Float64.(x), Float64(0.7f0))
  rel = maximum(abs.(Float64.(y) .- y64)) / maximum(abs.(y64))
  (rel > 100 * eps(Float32), "Float32 result vs Float64: rel $(round(rel; sigdigits = 3)) = $(round(rel / eps(Float32); sigdigits = 3)) eps32, no exception")
end
check("LIM-018", "'misses the domain' for a feasible problem") do
  t, r = attempt(() -> prox(Constrained(IndBox(0.0, 1.0), IndHyperplane([1.0, 1.0], 1.0)), [1e300, 1e300], 1.0))
  (t && r isa ArgumentError, t ? shorterr(r) : "solved $(short(r))")
end
check("LIM-019", "ridge search: spurious 'misses the domain' at large γλ") do
  t, r = attempt(() -> prox(Constrained(SqrNormL2(2f0^30), IndHyperplane(1f0, 2f0^-29)), zeros(Float32, 2), 2f0^100))
  (t, t ? shorterr(r) : "solved $(short(r))")
end
check("LIM-020", "hyperplane coefficient rounded to a narrow output") do
  y = prox!(zeros(Float32, 1), IndHyperplane([2.0^100], 0.0), Float32[2.0^-100], 1.0)
  (y[1] != 0, "Float32 output $(short(y)) (exact 0)")
end
check("LIM-021", "halfspace projection -Inf when <a,x> overflows") do
  y = prox(IndHalfspace([1.0, 1.0], 0.0), [M, M], 1.0)
  (any(isinf, y), "$(short(y)) (exact [0,0])")
end
# ---------------------------------------------------------------- range (R1-R4)
check("LIM-022", "scalar kernels lose range (MCP value)") do
  v = MCP(1e154, 3.0)([2e154]); (!isfinite(v), "MCP(1e154,3)([2e154]) = $v (exact ≈1.5e308)")
end
check("LIM-023", "Translate of a norm: no homogeneity recovery (numerics R1 example)") do
  v = Translate(NormL1(0.25), [-M])([M]); (!isfinite(v), "= $v (contract: M/2)")
end
check("LIM-024", "Scaled without a foldable level loses range") do
  v = Scaled(SqrNormL2(), 1e-200)([1e200]); (!isfinite(v), "Scaled(SqrNormL2(),1e-200)([1e200]) = $v (exact 5e199)")
end
check("LIM-025", "value_and_prox! of Scaled returns 0 for an underflowing inner value") do
  f = Scaled(NormL1(2.0^-1000), 2.0^1000); y = zeros(1)
  v = value_and_prox!(y, f, [2.0^-100], 2.0^-1000)
  (v == 0 && f(y) > 0, "returned $v, f(y) = $(f(y))")
end
check("LIM-026", "NormL2 rounds a subnormal norm before a large weight") do
  s = nextfloat(0.0); v = NormL2(1e300)([s, s]); exact = Float64(1e300 * sqrt(big(2)) * big(s))
  (abs(v - exact) / exact > 1e-10, "= $v (exact $exact)")
end
check("LIM-027", "proximal outputs overflow (HingeLoss at floatmax)") do
  y = prox(HingeLoss([2.0]), [M], 1.0); (!all(isfinite, y), "$(short(y)) (exact [M])")
end
check("LIM-028", "exponential-cone projection when an output overflows") do
  y = prox(IndExpDual(), [M, -M, M], 1.0); (y[1] == 0 && y[2] == 0, "$(short(y))")
end
check("LIM-029", "weighted NormL1 value loses underflowing products (Float32)") do
  x = fill(Float32(2.0^-9 + 2.0^-24), 512); w = fill(2f0^-126, 512)
  v = NormL1(1f0, w)(x); exact = Float32(sum(big.(w) .* big.(x)))
  ulps = abs(v - exact) / eps(exact)
  (ulps > 4, "= $v, exact $exact ($(round(ulps)) ulps)")
end
check("LIM-030", "SingularValue(NormL1) value overflows where NuclearNorm does not") do
  A = fill(M, 2, 2); a = SingularValue(NormL1(0.25))(A); b = NuclearNorm(0.25)(A)
  (!isfinite(a) && isfinite(b), "SingularValue = $a, NuclearNorm = $b")
end
check("LIM-031", "QuadraticFunction value when (A x)_i underflows") do
  a = 1.37 * 2.0^-970; A = [0 a; a 0]; x = [1.23 * 2.0^1000, 1.11 * 2.0^-60]
  v = QuadraticFunction(A)(x); ref = big(a) * big(x[1]) * big(x[2])
  rel = Float64(abs(big(v) - ref) / ref)
  (rel > 4eps(), "rel err $(round(rel; sigdigits = 3)) = $(round(rel / eps(); sigdigits = 3)) eps")
end
check("LIM-032", "GLMLoss value with an out-of-range observation weight") do
  v = PoissonLoss([0.0], ones(1, 1); weights = [1e-300])([1000.0]); (!isfinite(v), "= $v (exact 1.97e134)")
end
check("LIM-033", "GroupMCP prox NaN when the group norm overflows") do
  y = prox!(zeros(3), GroupMCP(1.0, [1:2, 3:3], [1.0, 1.0]; a = 3), Float32[3e38, 3e38, 3e38], 1.0)
  (any(isnan, y), short(y))
end
check("LIM-034", "nested Scaled underflows to ZeroFunction") do
  f = Scaled(Scaled(NormL1(), 1e-200), 1e-200); (f isa ZeroFunction, "typeof = $(nameof(typeof(f))), value at [1e300] = $(f([1e300]))")
end
check("LIM-035", "Poisson quadratic model at η = 710") do
  q = quadratic_model(PoissonLoss([1.0], ones(1, 1); weights = [1 / 8]), [710.0])
  (!all(isfinite, q.weights) || !all(isfinite, q.y), "weights = $(q.weights), y = $(q.y)")
end
check("LIM-036", "Constrained(NormL1, IndSphereL2) throws when γλw overflows") do
  t, r = attempt(() -> prox(Constrained(NormL1(1e300), IndSphereL2(1.0)), [1.0, 2.0], 1e300))
  (t, t ? shorterr(r) : short(r))
end
check("LIM-037", "Smoothed deriv2 is a finite difference") do
  f = PB.Smoothed(NegEntropy(1.0), 0.5)
  d = PB.deriv2(f, 0.7, 0.0)
  dref = PB.deriv2(PB.Smoothed(NegEntropy(big(1.0)), big(0.5)), big(0.7), big(0.0))
  rel = Float64(abs(big(d) - dref) / abs(dref))
  (rel > 1e3 * eps(), "Float64 deriv2 rel err vs BigFloat $(round(rel; sigdigits = 3)) = $(round(rel / eps(); sigdigits = 3)) eps")
end
# ---------------------------------------------------------------- precision (T3/T4/O2)
check("LIM-038", "narrow data at a wide argument: numerics T3 gradient example") do
  g = gradient!(zeros(Float32, 1), QuadraticFunction(Float32[1;;], Float32[-1]), [1 + 2.0^-28])
  (g[1] != Float32(2.0^-28), "gradient $(short(g)) (contract: 2^-28)")
end
check("LIM-039", "numerics V10 row") do
  z = prox!(zeros(1), NormL1(1 + 2f0^-23), Float32[1], 1 - 2f0^-23); (z[1] != 2.0^-46, "$(short(z)) (contract: 2^-46)")
end
check("LIM-040", "threshold products in the factors' type (TotalVariation1D)") do
  y = prox(TotalVariation1D(2f0^80), [2.0^161, 1.0], 2f0^80); (any(isnan, y), short(y))
end
check("LIM-041", "AtomIterate prox_coordinate! rounds a wider argument") do
  y = AtomIterate(Float32, 2, [1:1, 2:2])
  prox_coordinate!(y, GroupNormL2(1f0, [1:1, 2:2], ones(Float32, 2)), 1, [1.0 + 2.0^-28], 1.0 - 2.0^-28)
  (Vector(y)[1] == 0, "stored $(short(Vector(y))) (T3: 2^-27)")
end
check("LIM-042", "group value_and_prox! is not f(y) (Float32 arg, Float64 dest)") do
  f = GroupNormL2(1.0, [1:3], [1.0]); y = zeros(3); v = value_and_prox!(y, f, Float32[1.1, 2.3, -0.7], 0.5)
  rel = abs(v - f(y)) / f(y); (rel > 4eps(), "rel gap $(round(rel; sigdigits = 3))")
end
check("LIM-043", "group norm in the narrower argument type") do
  x32 = Float32[1.1, 2.3, -0.7, 0.3, -1.9, 2.2]; f = GroupNormL2(0.3, [1:3, 4:6], ones(2))
  a = prox!(zeros(6), f, x32, 0.5); b = prox!(zeros(6), f, Float64.(x32), 0.5)
  rel = maximum(abs.(a .- b) ./ abs.(b)); (rel > 4eps(), "max rel diff $(round(rel; sigdigits = 3))")
end
check("LIM-044", "Precompose of a value-less inner function: value in the image precision") do
  a = Float32[2, -0.5, 1.5]; P = Precompose(Conjugate(ElasticNet(1f0, 0.5f0)), a); y = zeros(3)
  v = value_and_prox!(y, P, Float32[1, 2, 3], 0.5)
  ref = sum(max.(abs.(big.(a) .* big.(y)) .- 1, 0) .^ 2) / (2 * big(0.5))
  rel = Float64(abs(big(v) - ref) / ref); (rel > 1e3 * eps(), "rel gap $(round(rel; sigdigits = 3))")
end
check("LIM-045", "Float16 argument and destination accepted") do
  t1, _ = attempt(() -> prox!(zeros(Float32, 3), NormL1(1.0), Float16[1, -2, 3], 0.3))
  t2, _ = attempt(() -> prox!(zeros(Float16, 3), NormL1(1.0), Float32[1, -2, 3], 0.3))
  (!t1 || !t2, "Float16 argument accepted=$(!t1); Float16 destination accepted=$(!t2)")
end
check("LIM-046", "LAPACK function accepts a BigFloat argument") do
  t, r = attempt(() -> prox!(zeros(2, 2), IndPSD(), BigFloat[1 0; 0 1], 1.0)); (!t, t ? shorterr(r) : "accepted")
end
# ---------------------------------------------------------------- NaN, Inf, degenerate (E1-E6)
check("LIM-047", "NormL2 prox turns NaN into zeros") do
  y = prox(NormL2(1.0), [NaN, 4.0], 1.0); (!any(isnan, y), "$(short(y)) (E2: [NaN,NaN])")
end
check("LIM-048", "other proximal maps lose NaN (NormL0, IndBallL0, NormL1Fused)") do
  a = prox_entry(NormL0(1.0), NaN, 1, 1.0); b = IndBallL0(1)([NaN, 0.0])
  c = prox(NormL1Fused(1.0, 1.0), [1.0, NaN, 0.5], 1.0)
  (!isnan(a) || !isnan(b) || all(isfinite, c[[1, 3]]), "NormL0 entry $a; IndBallL0 value $b; NormL1Fused $(short(c))")
end
check("LIM-049", "standalone values at NaN are Inf") do
  a = IndBox(-1.0, 1.0)([NaN]); b = NegEntropy()([NaN]); (a == Inf || b == Inf, "IndBox $a, NegEntropy $b (E2: NaN)")
end
check("LIM-050", "value_and_prox! of an indicator returns 0 at a stored NaN") do
  y = zeros(2); v = value_and_prox!(y, IndBox(-1.0, 1.0), [NaN, 0.0], 1.0); (v == 0 && isnan(y[1]), "value $v, y = $(short(y))")
end
check("LIM-051", "Precompose around an indicator returns 0 at a stored NaN") do
  y = fill(7f0, 2); v = value_and_prox!(y, Precompose(IndBallL2(1.0), 2.0), [NaN, 0.0], 1.0)
  (v == 0 && any(isnan, y), "value $v, y = $(short(y))")
end
check("LIM-052", "infinite-weight fixed coordinate is NaN at a NaN argument") do
  y = prox(NormL1(1.0, [Inf, 1.0]), [NaN, 2.0], 1.0); (isnan(y[1]), "$(short(y)) (E2: [0,1])")
end
check("LIM-053", "zero level not the zero function at non-finite arguments") do
  a = NuclearNorm(0.0)([NaN 0; 0 1]); g = gradient!(zeros(2), SqrNormL2(0.0), [NaN, 4.0])
  (a == Inf || any(isnan, g), "NuclearNorm(0) value $a; SqrNormL2(0) gradient $(short(g))")
end
check("LIM-054", "MoreauEnvelope value at an infinite entry") do
  v = MoreauEnvelope(NormL1(1.0), 1.0)([Inf]); (isnan(v), "= $v (E3: +Inf)")
end
check("LIM-055", "decision 48 limits for remaining families") do
  a = prox(Constrained(IndBallL0(1), IndSphereL2(1.0)), [Inf, 0.0], 1.0)
  b = prox(Conjugate(NormLinf(1.0)), [Inf, 0.5, 0.2], 1.0)
  (any(isnan, a) || any(isnan, b), "Constrained(IndBallL0,IndSphereL2) $(short(a)); Conjugate(NormLinf) $(short(b))")
end
check("LIM-056", "γ = Inf endpoint with a NaN argument") do
  y = prox(NormL1(1.0), [NaN, 1.0], Inf); (isnan(y[1]), "$(short(y)) (E2: [0,0])")
end
check("LIM-057", "Scaled(indicator, Inf) value is NaN") do
  v = Scaled(IndBallL2(1.0), Inf)([0.3, 0.4]); (isnan(v), "value inside the ball = $v")
end
check("LIM-058", "Scaled(f, 0) rejected for a total-domain function") do
  t, r = attempt(() -> Scaled(HingeLoss(ones(3)), 0.0)); (t, t ? shorterr(r) : "accepted: $(nameof(typeof(r)))")
end
check("LIM-059", "fused spectral value disagrees with the stored output") do
  Y = zeros(3, 3); f = SingularValue(NormL0(1.0)); v = value_and_prox!(Y, f, Matrix(Diagonal([2.0^60, 2.0, 0.0])), 1.0)
  (v != f(Y), "returned $v, f(Y) = $(f(Y))")
end
# ---------------------------------------------------------------- devices (JLArrays only)
if HAVE_JL
  check("LIM-060", "device IndBallL1 at an infinite entry and a negative step") do
    y = Array(prox(IndBallL1(1.0), JLArray([Inf, 0.5, 0.2]), 1.0)); cpu = prox(IndBallL1(1.0), [Inf, 0.5, 0.2], 1.0)
    t, _ = attempt(() -> prox!(JLArray(zeros(3)), IndBallL1(1.0), JLArray([0.3, 0.5, 0.2]), -0.5))
    (!isequal(y, cpu) || !t, "device $(short(y)) vs CPU $(short(cpu)); γ=-0.5 rejected on device=$t")
  end
  check("LIM-061", "device Conjugate(NormLinf) cancels") do
    y = Array(prox(Conjugate(NormLinf()), JLArray([1e300, 0.25, -0.5]), 1.0)); (y[1] == 0, "device $(short(y)) (CPU [1,0,-0])")
  end
  check("LIM-062", "device kernel narrows a host level") do
    f = ElasticNet(1e40, 1.0, Float32[1e-30])
    a = prox(Conjugate(f), Float32[2e10], 1f0); b = Array(prox(Conjugate(adapt(JLArray, f)), JLArray(Float32[2e10]), 1f0))
    (a != b, "CPU $(short(a)) device $(short(b))")
  end
  check("LIM-063", "wrappers index scalars on device arrays") do
    JLArrays.allowscalar(false)
    t, r = attempt(() -> MoreauEnvelope(NormL1(1.0), 0.5)(JLArray([1.0, -2.0, 0.3])))
    (t, t ? shorterr(r) : "value $(r)")
  end
else
  foreach(i -> skip("LIM-0$i", "device entry", "JLArrays/Adapt not loadable in this environment"), 60:63)
end
# ---------------------------------------------------------------- wrappers and others
check("LIM-064", "Constrained(f, IndBox) value_and_prox! throws for a value-less f") do
  t, r = attempt(() -> value_and_prox!(zeros(2), Constrained(Conjugate(ElasticNet(1.0, 1.0)), IndBox(-1.0, 1.0)), [0.5, 2.0], 1.0))
  (t, t ? shorterr(r) : "value $r")
end
check("LIM-065", "parameter shape rules differ between Translate and Tilt") do
  x2 = [1.0 2; 3 4]
  t1, r1 = attempt(() -> Translate(NormL1(1.0), [1.0, 2, 3, 4])(x2)); t2, r2 = attempt(() -> Tilt(NormL1(1.0), [1.0, 2, 3, 4])(x2))
  (t1 != t2, "Translate $(t1 ? typeof(r1) : r1); Tilt $(t2 ? typeof(r2) : r2)")
end
check("LIM-066", "restrict(IndBinary) with a feasible dropped coordinate") do
  t, r = attempt(() -> restrict(IndBinary([0.0, 1.0, 0.0], [1.0, 2.0, 1.0]), [1, 2])); (t, t ? shorterr(r) : "restricted")
end
check("LIM-067", "Conjugate(SingularValue) into a SymmetricSparseIterate") do
  t, r = attempt(() -> prox!(SymmetricSparseIterate(2), Conjugate(SingularValue(NormL1(1.0))), [2.0 0.5; 0.5 1.0], 1.0))
  (t, t ? shorterr(r) : "ok")
end
check("LIM-068", "fixed default tolerance independent of precision") do
  f = IndIntersection(IndBox(0f0, 1f0), IndBallL2(1f0)); (f.tol === 1.0e-10, "Float32 IndIntersection tol = $(f.tol) ($(typeof(f.tol)))")
end
check("LIM-069", "Conjugate on the Moreau route throws at valid extreme steps") do
  t1, _ = attempt(() -> prox(Conjugate(IndHalfspace([1.0, 1.0], 1.0)), [0.0, 1.0], 1e-310))
  t2, _ = attempt(() -> prox(Conjugate(SqrNormL2()), [1e300], 1e-10))
  t3, _ = attempt(() -> prox(Conjugate(IndBallL1(1.0)), [0.5, 1.0], 1e-310))  # [0, 1] passes only through #529's one-infinity limit
  e = prox_entry(Conjugate(2 * IndBox(0.0, Inf)), 3.0, 1, 1e-310)
  (t1 || t2 || t3 || e != 0, "IndHalfspace γ=1e-310 throws=$t1; SqrNormL2 x=1e300 γ=1e-10 throws=$t2; IndBallL1 at [0.5,1] γ=1e-310 throws=$t3; prox_entry boxed cone = $e (exact 0)")
end
check("LIM-070", "Conjugate on the Moreau route cancels (SLOPE)") do
  y = prox(Conjugate(SLOPE([1.0, 0.5, 0.25])), [1e300, 0.25, -0.5], 1.0); (y[1] == 0, "$(short(y)) (exact [1,.25,-.5])")
end
# ---------------------------------------------------------------- iterates
check("LIM-071", "strong zeros: a stored zero changes dot with Inf") do
  s = SparseIterate(Float64, 3); s[2] = 1.0; a = [1.0, 2.0, Inf]
  d1 = dot(a, s); activate!(s, 3); d2 = dot(a, s); (!isequal(d1, d2), "unstored: $d1, stored zero: $d2")
end
skip("LIM-072", "sparse value oracles read every entry", "evidence: the @test_broken rows test/sparse_value_cost.jl:162-173 are still broken in the test suite at this SHA")
check("LIM-073", "SymmetricSparseIterate rmul! by Inf differs from broadcast") do
  X = SymmetricSparseIterate(Float64, 3); X[1, 1] = 1.0; X[2, 1] = 2.0
  Y = copy(X); rmul!(X, Inf); Y .*= Inf; (!isequal(Matrix(X), Matrix(Y)), "rmul! $(short(vec(Matrix(X)))) vs .*= $(short(vec(Matrix(Y))))")
end
check("LIM-074", "SymmetricSparseIterate: a loop over positions doubles off-diagonals") do
  x = SymmetricSparseIterate(Float64, 3); x[2, 1] = 1.0
  for i in 1:3, j in 1:3
    x[i, j] *= 2
  end
  (x[2, 1] == 4.0, "x[2,1] = $(x[2, 1]) after doubling each position (x .*= 2 gives 2)")
end
check("LIM-075", "copyto! into vec of a SymmetricSparseIterate is unchecked") do
  x = SymmetricSparseIterate(Float64, 3); t, r = attempt(() -> copyto!(vec(x), collect(1.0:9.0)))
  (!t, t ? shorterr(r) : "no error, Matrix = $(short(Matrix(x)))")
end
check("LIM-076", "fill!(x, 0) empties the working set") do
  x = SparseIterate(4); x[1] = 1.0; x[2] = 2.0; fill!(x, 0.0); (isempty(activecoordinates(x)), "active after fill! = $(activecoordinates(x))")
end
check("LIM-077", "coordinates(::SparseIterate) returns x itself") do
  m = SparseIterate(Float64, 2, 2); (coordinates(m) === m, "coordinates(M) === M: $(coordinates(m) === m)")
end
check("LIM-078", "AtomIterate group vectors are shared mutable state") do
  x = AtomIterate(Float64, 4, [[1, 2], [3, 4]]); parentindices(atom(x, 1))[1][1] = 3
  (x.groups[1][1] == 3, "x.groups after writing through parentindices = $(x.groups)")
end
check("LIM-079", "asymmetry message for BigFloat beyond the Float64 range") do
  t, r = attempt(() -> copyto!(SymmetricSparseIterate(BigFloat, 2), [big"1e400" big"2e400"; big"3e400" big"1e400"]))
  (t && occursin("NaN", sprint(showerror, r)), t ? shorterr(r) : "no error")
end
struct _BadGrad{F} <: PB.AbstractFunction
  f::F
end
(b::_BadGrad)(x) = b.f(x)
PB.gradient!(g, b::_BadGrad, x) = PB.gradient!(g, b.f, x)
PB.value_and_gradient!(g, b::_BadGrad, x) = PB.value_and_gradient!(g, b.f, x)
PB.gradient(b::_BadGrad, x::Tuple) = map(a -> Float64.(a), PB.gradient(b.f, x))
PB.value_and_gradient(b::_BadGrad, x::Tuple) = (b.f(x), PB.gradient(b, x))
check("LIM-080", "Testing.test_gradient accepts Float64 leaves for a Float32 tuple") do
  # a failing check inside @testset raises TestSetException at its end: that is the fixed state
  t, r = attempt(() -> @testset "LIM-080 probe" begin
    PB.Testing.test_gradient(_BadGrad(SeparableSum(SqrNormL2(), SqrNormL2())), (Float32[1, 2], Float32[3, 4]))
  end)
  (!t, t ? "test_gradient fails ($(typeof(r)))" : "test_gradient passes: $(r.n_passed) checks, none failed")
end
println("# summary: ", join(["$k=$(COUNT[k])" for k in ("PRESENT", "ABSENT", "ERROR", "SKIPPED")], ", "))
