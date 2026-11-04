# Re-run the reproducers of the active entries of known-limitations.md, one
# case per behavior.
#
#   julia --project=. docs/design/numerical-contract/audit-2026-11-03/reproduce-active.jl
#
# Every case is one behavior of a register entry, tagged with the entry (LIM-)
# and the behavior of the audit index (clusters.tsv) it comes from. A case
# decides two exact predicates, and prints
#
#   PRESENT  the limitation shows exactly as the register describes it
#   FIXED    the behavior the contract requires shows instead
#   CHANGED  neither (or both): the behavior moved, but not to the contract
#   ERROR    the probe itself failed, for example because an API changed
#   SKIPPED  not runnable here; the reason is printed
#
# Anything but PRESENT means the register entry needs a look. An entry is
# resolved only when all its cases print FIXED and a regression test pins the
# behavior. The device cases (LIM-060 to LIM-063) run only when JLArrays and
# Adapt load, for example from an environment with ProximalBase dev'd and
# JLArrays added; JLArrays runs device kernels on the CPU, so these cases are
# evidence about dispatch and kernel logic, not about GPUs. This is an
# evidence script for reviewers, not part of the test suite.
using ProximalBase, LinearAlgebra, Random, SparseArrays, Test
const PB = ProximalBase
const HAVE_JL = try
  @eval using JLArrays, Adapt
  true
catch
  false
end
# device code must not index scalars, as on a GPU
HAVE_JL && JLArrays.allowscalar(false)
const SHA = try
  strip(read(`git -C $(pkgdir(ProximalBase)) rev-parse HEAD`, String))
catch
  "unknown"
end
println("# ProximalBase ", pkgversion(ProximalBase), " at ", SHA, ", Julia ", VERSION, ", JLArrays: ", HAVE_JL)

const STATUSES = ("PRESENT", "FIXED", "CHANGED", "ERROR", "SKIPPED")
const COUNT = Dict(s => 0 for s in STATUSES)
const BYLIM = Dict{String, Dict{String, Int}}()
short(x) = (s = replace(repr(x; context = :compact => true), '\n' => ' '); length(s) > 120 ? s[1:120] * "…" : s)
shorterr(e) = (s = replace(sprint(showerror, e), '\n' => ' '); length(s) > 120 ? s[1:120] * "…" : s)
# exception class and a short message, or the value
show_outcome(t, r) = t ? "throws $(nameof(typeof(r)))" : short(r)

function record(lim, cluster, status, what, obs)
  COUNT[status] += 1
  d = get!(() -> Dict(s => 0 for s in STATUSES), BYLIM, lim)
  d[status] += 1
  println(rpad(lim, 9), rpad(cluster, 7), rpad(status, 8), what, " | ", obs)
end

"""
    verdict(present, fixed)

`:present` when only the limitation's predicate holds, `:fixed` when only the
contract's holds, `:changed` otherwise.
"""
verdict(present::Bool, fixed::Bool) = present == fixed ? :changed : present ? :present : :fixed

"""
    case(f, lim, cluster, what)

`f()` returns `(verdict(present, fixed), observed::String)`.
"""
function case(f, lim, cluster, what)
  status, obs = try
    v, o = f()
    (v === :present ? "PRESENT" : v === :fixed ? "FIXED" : "CHANGED", o)
  catch e
    ("ERROR", "probe error: " * shorterr(e))
  end
  record(lim, cluster, status, what, obs)
end
skip(lim, cluster, what, why) = record(lim, cluster, "SKIPPED", what, why)

"Run `g()`; return `(threw, exception or result)`."
attempt(g) = try
  (false, g())
catch e
  (true, e)
end
"Run a call that may write `y`; return `(threw, exception or result, y changed)`."
function writes(g, y)
  before = deepcopy(y)
  t, r = attempt(g)
  (t, r, !isequal(before, y))
end
"The call throws `E` and leaves `y` as it was."
rejects(E, t, r, changed) = t && r isa E && !changed
const M = floatmax(Float64)
"A two-block `SeparableSum` whose first block is ordinary: `(s, x, y)`."
blk(f, x) = (SeparableSum(SqrNormL2(2.0), f), ([3.0], x), ([7.0], fill(7.0, size(x))))

# ================================================================ failures, steps and searches (LIM-001 to LIM-021)
# Group A: LIM-001 to LIM-021 (failures and partial writes O6, steps P1-P3, searches A3)

# ---- helpers (names prefixed _A_)

# a destination array type that the package does not recognise (#371)
struct _A_Unknown{T} <: AbstractVector{T}
  data::Vector{T}
end
Base.size(x::_A_Unknown) = size(x.data)
Base.IndexStyle(::Type{<:_A_Unknown}) = IndexLinear()
Base.getindex(x::_A_Unknown, i::Int) = x.data[i]
Base.setindex!(x::_A_Unknown, v, i::Int) = (x.data[i] = v)

# prox! and value_and_prox! of f into fresh destinations filled with 7:
# (outcome of prox!, y, outcome of value_and_prox!, y), an outcome being (threw, result, changed)
function _A_ops(f, x, γ)
  y1 = fill!(similar(x, float(eltype(x))), 7)
  y2 = fill!(similar(x, float(eltype(x))), 7)
  o1 = writes(() -> prox!(y1, f, x, γ), y1)
  o2 = writes(() -> value_and_prox!(y2, f, x, γ), y2)
  (o1, y1, o2, y2)
end
_A_msg(r) = shorterr(r)
_A_out(t, r) = t ? show_outcome(t, r) : "returns a result"
# (threw, result or exception, false, nothing) for a call that writes nothing
_A_try(g) = (o = attempt(g); (o[1], o[2], false, nothing))
# (threw, result, y changed, y) for a call that writes y
_A_wr(g, y) = (o = writes(g, y); (o[1], o[2], o[3], y))
# a search: (threw, result or exception, y changed, y) for prox!(y, f, x, γ), y filled with 7
function _A_search(f, x, γ)
  y = fill!(similar(x, float(eltype(x))), 7)
  t, r, ch = writes(() -> prox!(y, f, x, γ), y)
  (t, r, ch, y)
end
# A3 and O6.3: PrecisionError with y unchanged, or a result accepted by `ok`
_A_a3(t, r, ch, y, ok) = rejects(PrecisionError, t, r, ch) || (!t && ok(y))
_A_missesdomain(t, r) = t && r isa ArgumentError && occursin("misses the domain", _A_msg(r))

# deterministic data without Random: splitmix64 and Box-Muller
mutable struct _A_SM
  s::UInt64
end
function _A_next!(g::_A_SM)
  g.s += 0x9e3779b97f4a7c15
  z = g.s
  z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
  z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
  z ⊻ (z >> 31)
end
_A_unif!(g) = (_A_next!(g) >> 11) * 2.0^-53
_A_gauss!(g) = sqrt(-2 * log1p(-_A_unif!(g))) * cos(2π * _A_unif!(g))
# #151's table case: a signed normal that is zero in its first half, b = Σ min(a, 0) + 1
function _A_data151(n, seed)
  g = _A_SM(UInt64(seed))
  a = [_A_gauss!(g) for _ in 1:n]
  a[1:n÷2] .= 0.0
  for i in n÷2+1:n
    a[i] = _A_unif!(g) * (_A_unif!(g) > 0.5) - _A_unif!(g) * (_A_unif!(g) > 0.5)
  end
  a32 = Float32.(a)
  x32 = Float32[_A_gauss!(g) for _ in 1:n]
  a32, x32, Float32(sum(min.(Float64.(a32), 0)) + 1)
end
# the projection onto {a·z = b} in the metric diag(γ) of z - x - μγa..., for SqrNormL2(λ), in BigFloat
function _A_ridge_ref(x, a, b, γ, λ)
  X, A, G, L = BigFloat.(x), BigFloat.(a), BigFloat.(γ), BigFloat(λ)
  d = 1 .+ G .* L
  μ = (sum(A .* X ./ d) - BigFloat(b)) / sum(G .* A .^ 2 ./ d)
  (X .- G .* A .* μ) ./ d
end

# ---- LIM-001: Conjugate finite-result check at a valid step, later block
for op in (prox!, value_and_prox!)
  case("LIM-001", "P1-09", "SeparableSum(SqrNormL2(2), Conjugate(SqrNormL2())) block [1e300], γ = 1e-10, $(nameof(op)): ArgumentError after block 1 was written") do
    s, x, y = blk(Conjugate(SqrNormL2()), [1e300])
    t, r, ch = writes(() -> op(y, s, x, 1e-10), y)
    present = t && r isa ArgumentError && ch
    # O6.3 + decision 42: PrecisionError before writing; or the finite result 1e300 / (1 + 1e-10) (R1/R2)
    fixed = rejects(PrecisionError, t, r, ch) || (!t && isapprox(y[2][1], 1e300 / (1 + 1e-10); rtol = 1e-14))
    verdict(present, fixed), "$(show_outcome(t, r)), block 1 written = $ch, y = $(short(y))"
  end
end
case("LIM-001", "P1-09", "same block at x = [1.0], γ = 1e-310 (valid step; 1/γ overflows): ArgumentError after block 1 was written") do
  s, x, y = blk(Conjugate(SqrNormL2()), [1.0])
  t, r, ch = writes(() -> prox!(y, s, x, 1e-310), y)
  present = t && r isa ArgumentError && ch
  fixed = rejects(PrecisionError, t, r, ch) || (!t && y[2][1] == 1.0)    # x / (1 + 1e-310) = 1
  verdict(present, fixed), "$(show_outcome(t, r)), block 1 written = $ch, y = $(short(y))"
end
case("LIM-001", "P1-09", "SliceSum(SqrNormL2(2), Conjugate(SqrNormL2())) on [3 1e300], γ = 1e-10: ArgumentError after slice 1 was written") do
  s = SliceSum(SqrNormL2(2.0), Conjugate(SqrNormL2()))
  x = [3.0 1e300]; y = [7.0 7.0]
  t, r, ch = writes(() -> prox!(y, s, x, 1e-10), y)
  present = t && r isa ArgumentError && ch
  fixed = rejects(PrecisionError, t, r, ch) || (!t && isapprox(y[1, 2], 1e300 / (1 + 1e-10); rtol = 1e-14))
  verdict(present, fixed), "$(show_outcome(t, r)), slice 1 written = $ch, y = $(short(y))"
end

# ---- LIM-002: Conjugate block at an invalid or unnamed endpoint step
for γ in (-1.0, NaN, 0.0, Inf)
  case("LIM-002", "P1-10", "Conjugate(SqrNormL2()) block at γ = $γ: ArgumentError only after block 1 was written") do
    s, x, y = blk(Conjugate(SqrNormL2()), [1.0])
    t, r, ch = writes(() -> prox!(y, s, x, γ), y)
    present = t && r isa ArgumentError && ch
    fixed = rejects(ArgumentError, t, r, ch)        # P1, O6.1: rejected before writing
    verdict(present, fixed), "$(show_outcome(t, r)), block 1 written = $ch, y = $(short(y))"
  end
end
for γ in (0.0, Inf)
  case("LIM-002", "P1-10", "Conjugate(IndBallL1(1)) block at x = [0, 1], γ = $γ: ArgumentError only after block 1 was written") do
    s, x, y = blk(Conjugate(IndBallL1(1.0)), [0.0, 1.0])
    t, r, ch = writes(() -> prox!(y, s, x, γ), y)
    present = t && r isa ArgumentError && ch
    fixed = rejects(ArgumentError, t, r, ch)
    verdict(present, fixed), "$(show_outcome(t, r)), block 1 written = $ch, y = $(short(y))"
  end
end
case("LIM-002", "P1-10", "Conjugate(HingeLoss()) block at γ = -1: accepted and written") do
  s, x, y = blk(Conjugate(HingeLoss()), [1.0])
  t, r, ch = writes(() -> prox!(y, s, x, -1.0), y)
  present = !t && ch
  fixed = rejects(ArgumentError, t, r, ch)
  verdict(present, fixed), "$(show_outcome(t, r)), y = $(short(y))"
end

# ---- LIM-003: search-found infeasibility in a later LinearConstrained block
for (nm, f) in (
    ("IndBox(0,1) + IndHyperplane(ones(2), 5) (array normal)", Constrained(IndBox(0.0, 1.0), IndHyperplane(ones(2), 5.0))),
    ("IndBox(0,1) + IndHyperslab(ones(2), 3, 4)", Constrained(IndBox(0.0, 1.0), IndHyperslab(ones(2), 3.0, 4.0))),
    ("IndBox(0,1) + IndHalfspace(ones(2), -5)", Constrained(IndBox(0.0, 1.0), IndHalfspace(ones(2), -5.0))),
    ("BoxConstrained inner (SqrNormL2, IndBox) + IndHyperplane(ones(2), 5)", Constrained(Constrained(SqrNormL2(), IndBox(0.0, 1.0)), IndHyperplane(ones(2), 5.0))),
    ("overflowing box IndBox(-1e308, 1) + IndHyperplane(1, 5)", Constrained(IndBox(-1e308, 1.0), IndHyperplane(1.0, 5.0))))
  case("LIM-003", "P1-11", "$nm as block 2, γ = 1: infeasibility raised after block 1 was written") do
    s, x, y = blk(f, [0.5, 0.5])
    t, r, ch = writes(() -> prox!(y, s, x, 1.0), y)
    present = _A_missesdomain(t, r) && ch
    fixed = rejects(ArgumentError, t, r, ch)         # O6.2, decision 42: ArgumentError naming it, before writing
    verdict(present, fixed), "$(show_outcome(t, r)) ($(t ? _A_msg(r) : "")), block 1 written = $ch"
  end
end

# ---- LIM-004: empty LinearConstrained set gives NaN
let f = Constrained(IndBox(0.0, 1.0), IndHyperplane(1.0, 5.0))
  for (nm, x, γ) in (("a NaN argument entry [NaN, 0], γ = 1", [NaN, 0.0], 1.0), ("γ = Inf, x = [0.5, 0.5]", [0.5, 0.5], Inf))
    case("LIM-004", "P0-01", "empty IndBox(0,1) ∩ {x₁ + x₂ = 5} with $nm: NaN result, no error") do
      (t1, r1, c1), y1, (t2, r2, c2), y2 = _A_ops(f, x, γ)
      present = !t1 && all(isnan, y1) && !t2 && all(isnan, y2)
      fixed = rejects(ArgumentError, t1, r1, c1) && rejects(ArgumentError, t2, r2, c2)
      verdict(present, fixed), "prox! $(show_outcome(t1, r1)) y = $(short(y1)); value_and_prox! $(show_outcome(t2, r2)) y = $(short(y2))"
    end
  end
end

# ---- LIM-005: non-proximable later block
for op in (prox!, value_and_prox!)
  case("LIM-005", "P1-34", "SeparableSum(SqrNormL2(2), LogisticLoss) $(nameof(op)): MethodError after block 1 was written") do
    f = LogisticLoss([1.0, 0.0], [1.0 0.0; 0.0 1.0])
    s, x, y = blk(f, [0.1, 0.2])
    t, r, ch = writes(() -> op(y, s, x, 1.0), y)
    present = t && r isa MethodError && ch
    # O6.1: an unsupported call fails before any block is written (MethodError or ArgumentError)
    fixed = t && (r isa MethodError || r isa ArgumentError) && !ch
    verdict(present, fixed), "$(show_outcome(t, r)), block 1 written = $ch, isproximable = $(isproximable(s))"
  end
end

# ---- LIM-006: tuple Conjugate destination conversion after a partial write
let c = Conjugate(SeparableSum(NormLinf(), NormLinf()))
  for op in (prox!, value_and_prox!)
    case("LIM-006", "P1-35", "#371 case 1, $(nameof(op)): Conjugate(SeparableSum(NormLinf, NormLinf)), block 2 destination Unknown(Int[7, 7]): InexactError after block 1 was written") do
      y = ([7.0], _A_Unknown(Int[7, 7]))
      t, r, ch = writes(() -> op(y, c, ([2.0], [1.0, 1.0]), 1.0), y)
      present = t && r isa InexactError && ch
      fixed = rejects(ArgumentError, t, r, ch)       # O6.1 / T1: integer destination rejected before writing
      verdict(present, fixed), "$(show_outcome(t, r)), block 1 written = $ch, y = $(short(y))"
    end
  end
end
case("LIM-006", "P1-35", "#371 case 2, value_and_prox!: Conjugate(SeparableSum(Translate(GroupNormLinf), SqrNormL2)), block 2 destination Int[7]: InexactError after block 1 was written") do
  g = Translate(GroupNormLinf(1.0, [1:2]), zeros(2))
  c = Conjugate(SeparableSum(g, SqrNormL2()))
  y = ([7.0, 7.0], Int[7])
  t, r, ch = writes(() -> value_and_prox!(y, c, (_A_Unknown([3.0, 2.0]), [3.0]), 1.0), y)
  present = t && r isa InexactError && ch
  fixed = rejects(ArgumentError, t, r, ch)
  verdict(present, fixed), "$(show_outcome(t, r)), block 1 written = $ch, y = $(short(y))"
end

# ---- LIM-007: quadratic_model! nested mismatch updates earlier siblings
let X = [1.0 0.5; -0.5 2.0; 1.5 -1.0], yv = [1.0, 0.0, 1.0], β = [0.1, -0.2], β2 = [2.0, 2.0]
  f1, f2, f3 = LogisticLoss(yv, X), LogisticLoss(yv, X), LogisticLoss(yv, X)
  top = SeparableSum(f1, SeparableSum(f2, f3))
  snap(q) = (copy(q.y), copy(q.weights))
  case("LIM-007", "P1-43", "nested model with 1 block where the function has 2: DimensionMismatch after the first sibling model was updated") do
    q1 = quadratic_model(f1, β2)                       # a model at a different point, so an update changes it
    bad = SeparableSum(q1, SeparableSum(quadratic_model(f2, β)))
    before = snap(q1)
    t, r = attempt(() -> quadratic_model!(bad, top, (β, (β, β))))
    ch = snap(q1) != before
    present = t && r isa DimensionMismatch && ch
    fixed = t && r isa DimensionMismatch && !ch      # E1, O6.1: sizes raise DimensionMismatch before any model block is written
    verdict(present, fixed), "$(show_outcome(t, r)): $(t ? _A_msg(r) : ""); first sibling updated = $ch"
  end
  case("LIM-007", "P1-43", "nested model block of the wrong type (SqrNormL2 for a LogisticLoss): ArgumentError 'no Hessian is known' after the first sibling model was updated") do
    q1 = quadratic_model(f1, β2)
    bad = SeparableSum(q1, SeparableSum(SqrNormL2(1.0), quadratic_model(f3, β)))
    before = snap(q1)
    t, r = attempt(() -> quadratic_model!(bad, top, (β, (β, β))))
    ch = snap(q1) != before
    present = t && r isa ArgumentError && ch
    fixed = t && r isa ArgumentError && !ch          # E1, O6.1: ArgumentError (types) before any model block is written
    verdict(present, fixed), "$(show_outcome(t, r)): $(t ? _A_msg(r) : ""); first sibling updated = $ch"
  end
end

# ---- LIM-008: dense Symmetric destination
for (nm, f) in (("NormL1(1)", NormL1(1.0)), ("Conjugate(IndBallL1(1))", Conjugate(IndBallL1(1.0))), ("Conjugate(IndBox(-1, 1))", Conjugate(IndBox(-1.0, 1.0)))),
    op in (prox!, value_and_prox!)
  case("LIM-008", "P5-97", "$nm $(nameof(op)) into Symmetric(zeros(3, 3)): Base error after the diagonal was written") do
    S = Symmetric(zeros(3, 3))
    t, r, ch = writes(() -> op(S, f, ones(3, 3), 0.5), S)
    present = t && r isa ArgumentError && occursin("non-diagonal", _A_msg(r)) && ch
    fixed = rejects(ArgumentError, t, r, ch)       # O6.1: rejected before writing
    verdict(present, fixed), "$(show_outcome(t, r)): $(t ? _A_msg(r) : ""); S changed = $ch, diagonal = $(short(diag(S)))"
  end
end

case("LIM-008", "P5-97", "NormL1(1) prox! into Hermitian(zeros(3, 3)): Base error after the diagonal was written") do
  S = Hermitian(zeros(3, 3))
  t, r, ch = writes(() -> prox!(S, NormL1(1.0), ones(3, 3), 0.5), S)
  present = t && r isa ArgumentError && ch
  fixed = rejects(ArgumentError, t, r, ch)
  verdict(present, fixed), "$(show_outcome(t, r)): $(t ? _A_msg(r) : ""); S changed = $ch"
end

# ---- LIM-009: fractional result into an integer destination
case("LIM-009", "P6-36", "prox!(SparseIterate{Int}, NormL1(1), [2, 3.5, 4]): InexactError after entry 1 was written") do
  si = SparseIterate(Int, 3)
  t, r, ch = writes(() -> prox!(si, NormL1(1.0), [2.0, 3.5, 4.0], 1.0), si)
  present = t && r isa InexactError && ch && Vector(si) == [1, 0, 0]
  fixed = rejects(ArgumentError, t, r, ch)           # T1, decision 8, O6.1
  verdict(present, fixed), "$(show_outcome(t, r)), si = $(short(Vector(si)))"
end
case("LIM-009", "P6-36", "prox!(fill(7, 3), NormL1(1), [3, 2.5, 0]): InexactError after entry 1 was written, y = [2, 7, 7]") do
  y = fill(7, 3)
  t, r, ch = writes(() -> prox!(y, NormL1(1.0), [3.0, 2.5, 0.0], 1.0), y)
  present = t && r isa InexactError && y == [2, 7, 7]
  fixed = rejects(ArgumentError, t, r, ch)
  verdict(present, fixed), "$(show_outcome(t, r)), y = $(short(y))"
end
case("LIM-009", "P6-36", "prox!(fill(7, 3), NormL1(1), [2.5, 3, 3.5]): InexactError (not ArgumentError), y unchanged") do
  y = fill(7, 3)
  t, r, ch = writes(() -> prox!(y, NormL1(1.0), [2.5, 3.0, 3.5], 1.0), y)
  present = t && r isa InexactError && !ch
  fixed = rejects(ArgumentError, t, r, ch)
  verdict(present, fixed), "$(show_outcome(t, r)), y = $(short(y))"
end
case("LIM-009", "P6-36", "prox_coordinate!(SparseIterate{Int}, NormL1(1), 1, 3.5, 1): InexactError (not ArgumentError)") do
  si = SparseIterate(Int, 3)
  t, r, ch = writes(() -> prox_coordinate!(si, NormL1(1.0), 1, 3.5, 1.0), si)
  present = t && r isa InexactError && !ch
  fixed = rejects(ArgumentError, t, r, ch)
  verdict(present, fixed), "$(show_outcome(t, r)), si = $(short(Vector(si)))"
end

# ---- LIM-010: complex sources into iterates bypass the real-only check
let c = ComplexF64[1, 2im, 0, 0]
  mk = Dict("SparseIterate" => () -> SparseIterate(4), "AtomIterate" => () -> AtomIterate(4, [1:2, 3:4]))
  routes = (
    ("SparseIterate", "copyto!(x, [1, 2i, 0, 0])", x -> copyto!(x, c)),
    ("SparseIterate", "view(x, :) .= [1, 2i, 0, 0]", x -> (view(x, :) .= c)),
    ("SparseIterate", "reshape(x, 2, 2) .= [1 0; 2i 0]", x -> (reshape(x, 2, 2) .= reshape(c, 2, 2))),
    ("AtomIterate", "copyto!(x, [1, 2i, 0, 0])", x -> copyto!(x, c)),
    ("AtomIterate", "view(x, :) .= [1, 2i, 0, 0]", x -> (view(x, :) .= c)))
  for (T, nm, g) in routes
    case("LIM-010", "P1-58", "$T $nm: InexactError after entry 1 was written") do
      x = mk[T]()
      t, r, ch = writes(() -> g(x), x)
      present = t && r isa InexactError && ch && Vector(x) == [1.0, 0.0, 0.0, 0.0]
      fixed = rejects(ArgumentError, t, r, ch)        # S1, real-only rule: before writing
      verdict(present, fixed), "$(_A_out(t, r)), x = $(short(Vector(x)))"
    end
  end
  case("LIM-010", "P5-10", "view(SparseIterate, 1:2) .= [1+2i, 0]: InexactError (nothing written)") do
    x = SparseIterate(4)
    t, r, ch = writes(() -> (view(x, 1:2) .= ComplexF64[1+2im, 0]), x)
    present = t && r isa InexactError && !ch
    fixed = rejects(ArgumentError, t, r, ch)
    verdict(present, fixed), "$(_A_out(t, r)), x = $(short(Vector(x)))"
  end
  case("LIM-010", "P5-10", "atom(AtomIterate, 1) .= [1+2i, 0]: InexactError") do
    x = AtomIterate(4, [1:2, 3:4])
    t, r, ch = writes(() -> (atom(x, 1) .= ComplexF64[1+2im, 0]), x)
    present = t && r isa InexactError && !ch
    fixed = rejects(ArgumentError, t, r, ch)
    verdict(present, fixed), "$(_A_out(t, r)), x = $(short(Vector(x)))"
  end
  case("LIM-010", "P5-10", "view(SparseIterate, 1:2) .= [1+0i, 0] (zero imaginary part): accepted and stored as 1.0") do
    x = SparseIterate(4)
    t, r, ch = writes(() -> (view(x, 1:2) .= ComplexF64[1, 0]), x)
    present = !t && x[1] == 1.0
    fixed = rejects(ArgumentError, t, r, ch)
    verdict(present, fixed), "$(_A_out(t, r)), x = $(short(Vector(x)))"
  end
  for T in ("SparseIterate", "AtomIterate")
    case("LIM-010", "P6-84", "copyto!($T, ComplexF64[1, 2, 3, 4]) (zero imaginary parts): accepted") do
      x = mk[T]()
      t, r, ch = writes(() -> copyto!(x, ComplexF64[1, 2, 3, 4]), x)
      present = !t && Vector(x) == [1.0, 2.0, 3.0, 4.0]
      fixed = rejects(ArgumentError, t, r, ch)
      verdict(present, fixed), "$(_A_out(t, r)), x = $(short(Vector(x)))"
    end
  end
  case("LIM-010", "P6-84", "copyto!(SparseIterate, [1, 2, 3, 4+1i]): InexactError after entries 1-3 were written") do
    x = SparseIterate(4)
    t, r, ch = writes(() -> copyto!(x, ComplexF64[1, 2, 3, 4+1im]), x)
    present = t && r isa InexactError && Vector(x) == [1.0, 2.0, 3.0, 0.0]
    fixed = rejects(ArgumentError, t, r, ch)
    verdict(present, fixed), "$(_A_out(t, r)), x = $(short(Vector(x)))"
  end
end

# ---- LIM-011: complex steps and scalars raise MethodError
let xs = [1.0, 2.0]
  complexcalls = (
    ("prox!(y, NormL1, x, 1+0i)", "P1-30", () -> (y = zeros(2); _A_wr(() -> prox!(y, NormL1(1.0), xs, 1.0 + 0im), y))),
    ("value_and_prox!(y, NormL1, x, 1+0i)", "P1-30", () -> (y = zeros(2); _A_wr(() -> value_and_prox!(y, NormL1(1.0), xs, 1.0 + 0im), y))),
    ("prox(NormL1, x, 1.0im)", "P5-71", () -> _A_try(() -> prox(NormL1(1.0), ones(2), 1.0im))),
    ("prox_entry(NormL1, 1.0, 1, 1.0im)", "P1-30", () -> _A_try(() -> prox_entry(NormL1(1.0), 1.0, 1, 1.0im))),
    ("prox_entry(NormL1, 1+1im, 1, 0.3) (complex scalar)", "P5-71", () -> _A_try(() -> prox_entry(NormL1(1.0), 1.0 + 1im, 1, 0.3))),
    ("value_entry(NormL1, 1+1im, 1) (complex scalar)", "P5-71", () -> _A_try(() -> value_entry(NormL1(1.0), 1.0 + 1im, 1))),
    ("prox_coordinate!(SparseIterate, NormL1, 1, 0.5, 1.0im)", "P1-30", () -> (x = SparseIterate(3); _A_wr(() -> prox_coordinate!(x, NormL1(1.0), 1, 0.5, 1.0im), x))),
    ("prox(NormL1Fused, (1+1im, 2.0), 0.3) (complex scalar pair)", "P5-71", () -> _A_try(() -> prox(NormL1Fused(1.0, 1.0), (1.0 + 1im, 2.0), 0.3))))
  for (nm, cl, g) in complexcalls
    case("LIM-011", cl, "$nm: MethodError, not ArgumentError") do
      t, r, ch, _ = g()
      present = t && r isa MethodError && !ch
      fixed = rejects(ArgumentError, t, r, ch)        # interface.md (Precision): ArgumentError("<Name> requires real entries")
      verdict(present, fixed), "$(show_outcome(t, r)): $(t ? _A_msg(r) : "")"
    end
  end
end

# ---- LIM-012: complex adjoint product with a SparseIterate
let A = ComplexF64[1 2im 0 0; 0 1 0 0; 3 0 1 0; 0 0 0 1+im], xs = (x = SparseIterate(4); x[1] = 1.0; x)
  case("LIM-012", "P5-02", "A' * xs with complex A, SparseIterate xs: returns a complex vector") do
    t, r = attempt(() -> A' * xs)
    present = !t && eltype(r) <: Complex
    fixed = t && r isa ArgumentError
    verdict(present, fixed), show_outcome(t, r)
  end
  case("LIM-012", "P5-02", "mul!(complex out, A', xs): returns a complex vector") do
    out = zeros(ComplexF64, 4)
    t, r, ch = writes(() -> mul!(out, A', xs), out)
    present = !t && ch && out == conj.(A[1, :])        # the first column of A'
    fixed = rejects(ArgumentError, t, r, ch)
    verdict(present, fixed), "$(show_outcome(t, r)), out = $(short(out))"
  end
  case("LIM-012", "P5-02", "mul!(real out, A', xs): InexactError") do
    out = zeros(4)
    t, r, ch = writes(() -> mul!(out, A', xs), out)
    present = t && r isa InexactError
    fixed = rejects(ArgumentError, t, r, ch)
    verdict(present, fixed), "$(show_outcome(t, r)), out = $(short(out))"
  end
end

# ---- LIM-013: negative and NaN steps accepted
let leaves = (
    ("NormL1", NormL1(1.0), [0.5, 0.5]), ("SqrNormL2", SqrNormL2(1.0), [0.5, 0.5]),
    ("IndBox(-1, 1)", IndBox(-1.0, 1.0), [3.0, 0.0]), ("IndHyperplane(ones(2), 1)", IndHyperplane(ones(2), 1.0), [0.5, 2.5]),
    ("IndHalfspace(ones(2), 1)", IndHalfspace(ones(2), 1.0), [2.0, 2.5]), ("IndHyperslab(ones(2), 1, 2)", IndHyperslab(ones(2), 1.0, 2.0), [3.0, 2.5]),
    ("IndIsotonic", IndIsotonic(), [3.0, 1.0, 2.0]), ("IndSOC", IndSOC(), [1.0, 0.5, 0.5]),
    ("NuclearNorm(1)", NuclearNorm(1.0), [1.0 0.0; 0.0 2.0]), ("SLOPE([2, 1])", SLOPE([2.0, 1.0]), [0.5, 0.4]),
    ("TotalVariation1D(1)", TotalVariation1D(1.0), [0.5, 0.4, 0.3]))
  for (nm, f, x) in leaves, (γ, gs) in ((-1.0, "γ = -1"), (NaN, "γ = NaN"))
    case("LIM-013", "P1-16", "$nm prox! and value_and_prox! at $gs: accepted and written") do
      (t1, r1, c1), y1, (t2, r2, c2), y2 = _A_ops(f, x, γ)
      present = !t1 && c1 && !t2 && c2
      fixed = rejects(ArgumentError, t1, r1, c1) && rejects(ArgumentError, t2, r2, c2)   # P1, decision 25
      verdict(present, fixed), "prox! $(show_outcome(t1, r1)) y = $(short(vec(y1))); value_and_prox! $(show_outcome(t2, r2))"
    end
  end
  for (nm, f) in (("NormL1(1)", NormL1(1.0)), ("IndBox(-1, 1)", IndBox(-1.0, 1.0))), (γ, gs) in ((-1.0, "γ = -1"), (NaN, "γ = NaN"))
    case("LIM-013", "P1-16", "$nm as block 2 of a SeparableSum at $gs: accepted, block 1 written") do
      s, xb, y = blk(f, [3.0, 0.0])
      t, r, ch = writes(() -> prox!(y, s, xb, γ), y)
      present = !t && ch
      fixed = rejects(ArgumentError, t, r, ch)
      verdict(present, fixed), "$(show_outcome(t, r)), y = $(short(y))"
    end
  end
end
# prox_entry
for (nm, f, v, exs) in (("NormL1(1)", NormL1(1.0), 0.5, (1.0, NaN)), ("SqrNormL2(1)", SqrNormL2(1.0), 0.5, (-Inf, NaN))), (γ, ex) in zip((-1.0, NaN), exs)
  case("LIM-013", "P1-17", "prox_entry($nm, 0.5, 1, $γ): accepted, returns $ex") do
    t, r = attempt(() -> prox_entry(f, v, 1, γ))
    present = !t && isequal(r, ex)
    fixed = rejects(ArgumentError, t, r, false)
    verdict(present, fixed), show_outcome(t, r)
  end
end
# prox_coordinate!
for (γ, gs) in ((-1.0, "γ = -1"), (NaN, "γ = NaN"))
  case("LIM-013", "P1-17", "prox_coordinate!(SparseIterate, NormL1, 1, 0.5, γ) at $gs: accepted and written") do
    x = SparseIterate(3); x[2] = 5.0
    t, r, ch = writes(() -> prox_coordinate!(x, NormL1(1.0), 1, 0.5, γ), x)
    present = !t && ch
    fixed = rejects(ArgumentError, t, r, ch)
    verdict(present, fixed), "$(show_outcome(t, r)), x = $(short(Vector(x)))"
  end
  case("LIM-013", "P1-17", "prox_coordinate!(SparseIterate, NormL1, 1, γ) at $gs: accepted and written") do
    x = SparseIterate(3); x[1] = 2.0
    t, r, ch = writes(() -> prox_coordinate!(x, NormL1(1.0), 1, γ), x)
    present = !t && ch
    fixed = rejects(ArgumentError, t, r, ch)
    verdict(present, fixed), "$(show_outcome(t, r)), x = $(short(Vector(x)))"
  end
  case("LIM-013", "P1-17", "prox_coordinate!(AtomIterate, GroupNormL2, 1, u, γ) at $gs: accepted and written") do
    x = AtomIterate(2, [1:2]); x .= [5.0, 5.0]
    t, r, ch = writes(() -> prox_coordinate!(x, GroupNormL2(1.0, [1:2]), 1, [1.0, 2.0], γ), x)
    present = !t && ch
    fixed = rejects(ArgumentError, t, r, ch)
    verdict(present, fixed), "$(show_outcome(t, r)), x = $(short(Vector(x)))"
  end
  case("LIM-013", "P1-17", "prox_coordinate!(AtomIterate, GroupNormL2, 1, γ) at $gs: accepted and written") do
    x = AtomIterate(2, [1:2]); x .= [1.0, 2.0]
    t, r, ch = writes(() -> prox_coordinate!(x, GroupNormL2(1.0, [1:2]), 1, γ), x)
    present = !t && ch
    fixed = rejects(ArgumentError, t, r, ch)
    verdict(present, fixed), "$(show_outcome(t, r)), x = $(short(Vector(x)))"
  end
end
# IndBallGroupL0
for (γ, gs) in ((-1.0, "γ = -1"), (NaN, "γ = NaN"))
  f = IndBallGroupL0(1, [[1], [2]])
  case("LIM-013", "P1-18", "IndBallGroupL0(1, [[1], [2]]) prox! and value_and_prox! at $gs: accepted and written") do
    (t1, r1, c1), y1, (t2, r2, c2), y2 = _A_ops(f, [1.0, 2.0], γ)
    present = !t1 && c1 && !t2 && c2
    fixed = rejects(ArgumentError, t1, r1, c1) && rejects(ArgumentError, t2, r2, c2)
    verdict(present, fixed), "prox! $(show_outcome(t1, r1)) y = $(short(y1)); value_and_prox! $(show_outcome(t2, r2))"
  end
  case("LIM-013", "P1-18", "IndBallGroupL0 AtomIterate route at $gs: accepted and written") do
    x = AtomIterate(2, [1:1, 2:2]); x .= [1.0, 2.0]
    y = AtomIterate(2, [1:1, 2:2]); y .= [5.0, 5.0]
    t, r, ch = writes(() -> prox!(y, f, x, γ), y)
    present = !t && ch
    fixed = rejects(ArgumentError, t, r, ch)
    verdict(present, fixed), "$(show_outcome(t, r)), y = $(short(Vector(y)))"
  end
  case("LIM-013", "P1-18", "IndBallGroupL0 as block 2 of a SeparableSum at $gs: accepted, block 1 written") do
    s, xb, y = blk(f, [1.0, 2.0])
    t, r, ch = writes(() -> prox!(y, s, xb, γ), y)
    present = !t && ch
    fixed = rejects(ArgumentError, t, r, ch)
    verdict(present, fixed), "$(show_outcome(t, r)), y = $(short(y))"
  end
end

# ---- LIM-014: value-dependent or wrong-class step rejections
for (nm, f, x) in (("BernoulliLoss([0, 1])", BernoulliLoss([0.0, 1.0]), [0.2, 0.7]), ("CubeNormL2(1)", CubeNormL2(1.0), [1.0, 2.0]))
  case("LIM-014", "P1-15", "$nm in a block at γ = -1: DomainError after block 1 was written") do
    s, xb, y = blk(f, x)
    t, r, ch = writes(() -> prox!(y, s, xb, -1.0), y)
    present = t && r isa DomainError && ch
    fixed = rejects(ArgumentError, t, r, ch)
    verdict(present, fixed), "$(show_outcome(t, r)), y = $(short(y))"
  end
  case("LIM-014", "P1-15", "$nm standalone at γ = -1: DomainError (sqrt)") do
    (t1, r1, c1), y1, (t2, r2, c2), y2 = _A_ops(f, x, -1.0)
    present = t1 && r1 isa DomainError && t2 && r2 isa DomainError
    fixed = rejects(ArgumentError, t1, r1, c1) && rejects(ArgumentError, t2, r2, c2)
    verdict(present, fixed), "prox! $(show_outcome(t1, r1)); value_and_prox! $(show_outcome(t2, r2))"
  end
  case("LIM-014", "P1-15", "$nm standalone at γ = NaN: NaN output, no error") do
    (t1, r1, c1), y1, (t2, r2, c2), y2 = _A_ops(f, x, NaN)
    present = !t1 && all(isnan, y1) && !t2 && all(isnan, y2)
    fixed = rejects(ArgumentError, t1, r1, c1) && rejects(ArgumentError, t2, r2, c2)
    verdict(present, fixed), "prox! $(show_outcome(t1, r1)) y = $(short(y1)); value_and_prox! $(show_outcome(t2, r2))"
  end
end
case("LIM-014", "P1-15", "CubeNormL2(1) at x = [0, 0], γ = -1: accepted (the rejection depends on the argument)") do
  (t1, r1, c1), y1, (t2, r2, c2), y2 = _A_ops(CubeNormL2(1.0), [0.0, 0.0], -1.0)
  present = !t1 && c1 && !t2 && c2
  fixed = rejects(ArgumentError, t1, r1, c1) && rejects(ArgumentError, t2, r2, c2)
  verdict(present, fixed), "prox! $(show_outcome(t1, r1)) y = $(short(y1)); value_and_prox! $(show_outcome(t2, r2))"
end
case("LIM-014", "P1-15", "BernoulliLoss([0, 1], w = [0, 1]) at x = [0.2, 0.7], γ = -1: entry 1 (zero weight) is written, then DomainError") do
  y = fill(7.0, 2)
  t, r, ch = writes(() -> prox!(y, BernoulliLoss([0.0, 1.0], [0.0, 1.0]), [0.2, 0.7], -1.0), y)
  present = t && r isa DomainError && y == [0.2, 7.0]
  fixed = rejects(ArgumentError, t, r, ch)
  verdict(present, fixed), "$(show_outcome(t, r)), y = $(short(y))"
end
for (nm, f, x) in (("NegLog()", NegLog(), [1.0, 2.0]), ("NegEntropy()", NegEntropy(), [1.0, 2.0]),
                   ("KLDivergence([1, 2])", KLDivergence([1.0, 2.0]), [1.0, 2.0]), ("LogDetLoss(diag(2, 3))", LogDetLoss([2.0 0.0; 0.0 3.0]), [2.0 0.0; 0.0 3.0]))
  case("LIM-014", "P1-19", "$nm prox! and value_and_prox! at γ = -1: DomainError, outputs unchanged") do
    (t1, r1, c1), y1, (t2, r2, c2), y2 = _A_ops(f, x, -1.0)
    present = t1 && r1 isa DomainError && !c1 && t2 && r2 isa DomainError && !c2
    fixed = rejects(ArgumentError, t1, r1, c1) && rejects(ArgumentError, t2, r2, c2)
    verdict(present, fixed), "prox! $(show_outcome(t1, r1)); value_and_prox! $(show_outcome(t2, r2))"
  end
end
for (nm, f, x) in (("NegLog()", NegLog(), [1.0, 2.0]), ("NegEntropy()", NegEntropy(), [1.0, 2.0]), ("KLDivergence([1, 2])", KLDivergence([1.0, 2.0]), [1.0, 2.0]))
  case("LIM-014", "P1-19", "$nm prox! and value_and_prox! at γ = NaN: NaN output, no error") do
    (t1, r1, c1), y1, (t2, r2, c2), y2 = _A_ops(f, x, NaN)
    present = !t1 && all(isnan, y1) && !t2 && all(isnan, y2)
    fixed = rejects(ArgumentError, t1, r1, c1) && rejects(ArgumentError, t2, r2, c2)
    verdict(present, fixed), "prox! $(show_outcome(t1, r1)) y = $(short(y1)); value_and_prox! $(show_outcome(t2, r2))"
  end
end
case("LIM-014", "P1-19", "LogDetLoss(diag(2, 3)) at γ = NaN: DomainError, not ArgumentError") do
  (t1, r1, c1), y1, (t2, r2, c2), y2 = _A_ops(LogDetLoss([2.0 0.0; 0.0 3.0]), [2.0 0.0; 0.0 3.0], NaN)
  present = t1 && r1 isa DomainError && t2 && r2 isa DomainError
  fixed = rejects(ArgumentError, t1, r1, c1) && rejects(ArgumentError, t2, r2, c2)
  verdict(present, fixed), "prox! $(show_outcome(t1, r1)) $(t1 ? _A_msg(r1) : ""); value_and_prox! $(show_outcome(t2, r2))"
end
case("LIM-014", "P1-19", "NegLog() as block 2 at γ = -1: DomainError ('the step of NegLog must not be negative'), block 1 unchanged") do
  s, xb, y = blk(NegLog(), [1.0, 2.0])
  t, r, ch = writes(() -> prox!(y, s, xb, -1.0), y)
  present = t && r isa DomainError && !ch
  fixed = rejects(ArgumentError, t, r, ch)
  verdict(present, fixed), "$(show_outcome(t, r)): $(t ? _A_msg(r) : ""), changed = $ch"
end
let f = PointwiseMinimum(NegLog(1.0), SumLargest(1))
  case("LIM-014", "P6-45", "PointwiseMinimum(NegLog, SumLargest(1)) standalone at γ = -1: DomainError (sqrt)") do
    y = zeros(2)
    t, r, ch = writes(() -> prox!(y, f, [1.0, 2.0], -1.0), y)
    present = t && r isa DomainError && !ch
    fixed = rejects(ArgumentError, t, r, ch)
    verdict(present, fixed), "$(show_outcome(t, r)), changed = $ch"
  end
  case("LIM-014", "P6-45", "PointwiseMinimum(NegLog, SumLargest(1)) as block 2 at γ = -1: DomainError, blocks unchanged") do
    s, xb, y = blk(f, [1.0, 2.0])
    t, r, ch = writes(() -> prox!(y, s, xb, -1.0), y)
    present = t && r isa DomainError && !ch
    fixed = rejects(ArgumentError, t, r, ch)
    verdict(present, fixed), "$(show_outcome(t, r)): $(t ? _A_msg(r) : ""), changed = $ch"
  end
end

# ---- LIM-015: unnamed endpoint steps give NaN
# SqrNormL2(0) at γ = Inf is a zero-level case: LIM-053
for (nm, f, x) in (("SqrHingeLoss()", SqrHingeLoss(), [0.0, 2.0]),
                   ("SparseGroupNormL2(0, 0, [[1], [2]])", SparseGroupNormL2(0.0, 0.0, [[1], [2]]), [1.0, 2.0]))
  case("LIM-015", nm == "SqrHingeLoss()" ? "P1-21" : "P1-22", "$nm prox! at γ = Inf, x = $(short(x)): NaN entries") do
    y = fill(7.0, 2)
    t, r, ch = writes(() -> prox!(y, f, x, Inf), y)
    present = !t && any(isnan, y)
    # P1, decision 26: the named limit (the projection onto the minimizers: [1, 2] here), or ArgumentError before writing
    fixed = rejects(ArgumentError, t, r, ch) || (!t && y == [1.0, 2.0])
    verdict(present, fixed), "$(show_outcome(t, r)), y = $(short(y)), expected [1.0, 2.0] or ArgumentError"
  end
end

# ---- LIM-016: subnormal and non-representable diagonal steps
let x3 = [0.3, -1.2, 2.0]
  for γ in (1e-310, 5e-324)
    case("LIM-016", "P1-27", "Constrained(IndBox(-1,1), IndHyperplane(1,1)) at x = [0.3, -1.2, 2], scalar γ = $γ: point off the hyperplane") do
      t, r = attempt(() -> prox(Constrained(IndBox(-1.0, 1.0), IndHyperplane(1.0, 1.0)), x3, γ))
      present = !t && all(isfinite, r) && abs(sum(r) - 1) > 1e-6
      fixed = !t && isapprox(r, [0.75, -0.75, 1.0]; atol = 1e-9)     # the projection, independent of the scale (P3, decision 27); exact [3/4, -3/4, 1]
      verdict(present, fixed), "$(show_outcome(t, r)), sum = $(t ? "-" : sum(r)) (exact [0.75, -0.75, 1.0])"
    end
    case("LIM-016", "P1-27", "Constrained(NormL1(1), IndHyperplane(ones(3), 1)) at x = [0.3, -1.2, 2], scalar γ = $γ: -Inf entries") do
      t, r = attempt(() -> prox(Constrained(NormL1(1.0), IndHyperplane(ones(3), 1.0)), x3, γ))
      present = !t && all(==(-Inf), r)
      fixed = !t && isapprox(r, x3 .- (sum(x3) - 1) / 3; atol = 1e-9)  # soft threshold γ is below 1e-300: the projection onto the hyperplane
      verdict(present, fixed), "$(show_outcome(t, r)) (exact $(short(x3 .- (sum(x3) - 1) / 3)))"
    end
  end
end
# found while rechecking (review of #558): huge scalar steps on the same set
case("LIM-016", "P1-27", "Constrained(IndBox(-1,1), IndHyperplane(1,1)) at x = [0.3, -1.2, 2], scalar γ = 1e300 after a call at γ = 1 (the search starts from the last multiplier): point off the hyperplane") do
  f = Constrained(IndBox(-1.0, 1.0), IndHyperplane(1.0, 1.0))
  prox(f, [0.3, -1.2, 2.0], 1.0)
  t, r = attempt(() -> prox(f, [0.3, -1.2, 2.0], 1e300))
  present = !t && all(isfinite, r) && abs(sum(r) - 1) > 1e-6
  fixed = !t && isapprox(r, [0.75, -0.75, 1.0]; atol = 1e-9)     # an indicator's projection does not depend on γ
  verdict(present, fixed), "$(show_outcome(t, r)) (exact [0.75, -0.75, 1.0])"
end
case("LIM-016", "P1-27", "Constrained(IndBox(-1,1), IndHyperplane(1,1)) at x = [0.3, -1.2, 2], scalar γ = 1e308: 'misses the domain' for a feasible set") do
  y = fill(7.0, 3)
  t, r, ch = writes(() -> prox!(y, Constrained(IndBox(-1.0, 1.0), IndHyperplane(1.0, 1.0)), [0.3, -1.2, 2.0], 1e308), y)
  present = t && r isa ArgumentError && occursin("misses the domain", sprint(showerror, r))
  fixed = !t && isapprox(y, [0.75, -0.75, 1.0]; atol = 1e-9)
  verdict(present, fixed), "$(show_outcome(t, r)), y = $(short(y)) (exact [0.75, -0.75, 1.0])"
end
for (nm, f, ex) in (("IndHyperplane([1, 1], 1)", IndHyperplane([1.0, 1.0], 1.0), [0.5, 0.5]),
                    ("IndHalfspace([1, 1], 1)", IndHalfspace([1.0, 1.0], 1.0), [0.5, 0.5]),
                    ("IndHyperslab([1, 1], 0, 1.5)", IndHyperslab([1.0, 1.0], 0.0, 1.5), [0.75, 0.75])),
    (s, ss) in ((1e-310, "[1e-310, 1e-310]"), (1e308, "[1e308, 1e308]"))
  case("LIM-016", "P2-20", "$nm at x = [3, 3], equal array steps $ss: " * (s < 1 ? "-Inf entries" : "x returned unchanged")) do
    t, r = attempt(() -> prox(f, [3.0, 3.0], [s, s]))
    present = !t && (s < 1 ? all(==(-Inf), r) : r == [3.0, 3.0])
    fixed = !t && isapprox(r, ex; rtol = 1e-12)       # P3, decision 27: scale times ratios (all ratios 1)
    verdict(present, fixed), "$(show_outcome(t, r)) (exact $(short(ex)))"
  end
end
case("LIM-016", "P2-20", "IndHyperplane([1, 1], 1) at x = [3, 3], equal array steps [1e-320, 1e-320]: -Inf entries") do
  t, r = attempt(() -> prox(IndHyperplane([1.0, 1.0], 1.0), [3.0, 3.0], [1e-320, 1e-320]))
  present = !t && all(==(-Inf), r)
  fixed = !t && isapprox(r, [0.5, 0.5]; rtol = 1e-12)
  verdict(present, fixed), "$(show_outcome(t, r)) (exact [0.5, 0.5])"
end
# IndHyperslab returns the projection in this example (it fails at the equal steps above)
for (nm, f) in (("IndHyperplane", IndHyperplane(Float32[1, 1], 1f0)), ("IndHalfspace", IndHalfspace(Float32[1, 1], 1f0)))
  case("LIM-016", "P2-20", "$nm Float32, x = Float32[3, 3], Float64 steps [1e-50, 2e-50] (the P3 example): Float32[-Inf, -Inf]") do
    t, r = attempt(() -> prox(f, Float32[3, 3], [1e-50, 2e-50]))
    present = !t && all(==(-Float32(Inf)), r)
    # P3, decision 27: the projection of the steps [1, 2]; exact [4/3, -1/3]
    fixed = !t && eltype(r) === Float32 && isapprox(r, Float32[4/3, -1/3]; rtol = 1e-5)
    verdict(present, fixed), "$(show_outcome(t, r)) (exact [4/3, -1/3])"
  end
end

# ---- LIM-017: Float32 multiplier search is not certified
case("LIM-017", "P2-05", "#151 table case, deterministic data (n = 20000, Float32, step 0.7f0): differs from the Float64 solve by more than 100 eps(Float32), no PrecisionError") do
  a, x, b = _A_data151(20000, 1)
  f = Constrained(IndBox(0f0, 1f0), IndHyperplane(a, b))
  t, r, ch, y = _A_search(f, x, 0.7f0)
  y64 = prox(Constrained(IndBox(0.0, 1.0), IndHyperplane(Float64.(a), Float64(b))), Float64.(x), Float64(0.7f0))   # same effective problem (T3), solved in Float64
  rel = t ? NaN : maximum(abs.(Float64.(y) .- y64)) / maximum(abs.(y64))
  present = !t && rel > 100 * eps(Float32)
  # A3, O6.3: PrecisionError before writing, or a certified result; no bound is stated, 16 eps(Float32) relative to the Float64 solve is used
  fixed = rejects(PrecisionError, t, r, ch) || (!t && rel <= 16 * eps(Float32))
  verdict(present, fixed), "$(_A_out(t, r)), max|y32 - y64| / max|y64| = $(round(rel / eps(Float32); sigdigits = 3)) eps(Float32)"
end

# ---- LIM-018: 'misses the domain' for a feasible problem
case("LIM-018", "P2-23", "Constrained(IndBox(0,1), IndHyperplane([1,1], 1)) at x = [1e300, 1e300], γ = 1: ArgumentError 'misses the domain'") do
  t, r, ch, y = _A_search(Constrained(IndBox(0.0, 1.0), IndHyperplane([1.0, 1.0], 1.0)), [1e300, 1e300], 1.0)
  present = _A_missesdomain(t, r)
  fixed = _A_a3(t, r, ch, y, y -> isapprox(y, [0.5, 0.5]; atol = 1e-9))     # the projection is [1/2, 1/2]
  verdict(present, fixed), "$(show_outcome(t, r)): $(t ? _A_msg(r) : "") y = $(short(y))"
end
case("LIM-018", "P2-23", "Constrained(NormL1(1), IndHyperplane([1,1], 1)) at x = [0.3, 0.7], steps [1e300, 1]: ArgumentError 'misses the domain'") do
  t, r, ch, y = _A_search(Constrained(NormL1(1.0), IndHyperplane([1.0, 1.0], 1.0)), [0.3, 0.7], [1e300, 1.0])
  present = _A_missesdomain(t, r)
  fixed = _A_a3(t, r, ch, y, y -> all(isfinite, y) && abs(sum(y) - 1) <= 1e-9)    # a point with ⟨a, z⟩ = b (resolution criteria)
  verdict(present, fixed), "$(show_outcome(t, r)): $(t ? _A_msg(r) : "") y = $(short(y))"
end
for (nm, mk) in (("IndHyperplane(a, 1)", a -> IndHyperplane(a, 1.0)), ("IndHyperslab(a, 1, 2)", a -> IndHyperslab(a, 1.0, 2.0))),
    k in (120, 200, 1021)
  case("LIM-018", "P3-58", "Constrained(SqrNormL2(1), $nm), x = [0, -2, .5, 3], a = [.25, .5, 2, 1], steps [.3, .6, .9, 2^$k]: ArgumentError 'misses the domain'") do
    x = [0.0, -2.0, 0.5, 3.0]; a = [0.25, 0.5, 2.0, 1.0]; γ = [0.3, 0.6, 0.9, ldexp(1.0, k)]
    t, r, ch, y = _A_search(Constrained(SqrNormL2(1.0), mk(a)), x, γ)
    ref = _A_ridge_ref(x, a, 1.0, γ, 1.0)
    present = _A_missesdomain(t, r)
    fixed = _A_a3(t, r, ch, y, y -> maximum(abs.(BigFloat.(y) .- ref)) <= 1e-9 * maximum(abs.(ref)))   # BigFloat reference of the same problem
    verdict(present, fixed), "$(show_outcome(t, r)): $(t ? _A_msg(r) : "") y = $(short(y)), exact $(short(Float64.(ref)))"
  end
end
for s in (1e20, 1e30)
  case("LIM-018", "P4-33", "Constrained(IndBox(0,1), IndHyperplane(ones(4), 2)) at x = fill($s, 4), γ = 1: ArgumentError 'misses the domain'") do
    t, r, ch, y = _A_search(Constrained(IndBox(0.0, 1.0), IndHyperplane(ones(4), 2.0)), fill(s, 4), 1.0)
    present = _A_missesdomain(t, r)
    fixed = _A_a3(t, r, ch, y, y -> isapprox(y, fill(0.5, 4); atol = 1e-9))
    verdict(present, fixed), "$(show_outcome(t, r)): $(t ? _A_msg(r) : "") y = $(short(y))"
  end
end

# ---- LIM-019: ridge-plus-hyperplane search, spurious 'misses the domain'
case("LIM-019", "P6-82", "#510: Constrained(SqrNormL2(2^30), IndHyperplane(1, 2^-29)), Float32, γ = 2^100: ArgumentError 'misses the domain'") do
  t, r, ch, y = _A_search(Constrained(SqrNormL2(2f0^30), IndHyperplane(1f0, 2f0^-29)), zeros(Float32, 2), 2f0^100)
  present = _A_missesdomain(t, r)
  fixed = _A_a3(t, r, ch, y, y -> eltype(y) === Float32 && isapprox(y, fill(2f0^-30, 2); rtol = 1e-5))
  verdict(present, fixed), "$(show_outcome(t, r)): $(t ? _A_msg(r) : "") y = $(short(y)), exact 2^-30 = $(2f0^-30)"
end
case("LIM-019", "P6-82", "Float64 Constrained(SqrNormL2(1e17), IndHyperplane(1, 1)), γ = 1e17 (γλ = 1e34): ArgumentError 'misses the domain'") do
  t, r, ch, y = _A_search(Constrained(SqrNormL2(1e17), IndHyperplane(1.0, 1.0)), zeros(2), 1e17)
  present = _A_missesdomain(t, r)
  fixed = _A_a3(t, r, ch, y, y -> isapprox(y, [0.5, 0.5]; rtol = 1e-9))
  verdict(present, fixed), "$(show_outcome(t, r)): $(t ? _A_msg(r) : "") y = $(short(y))"
end
case("LIM-019", "P6-82", "Float32 Constrained(SqrNormL2(1e14), IndHyperplane(1, 1)), γ = 1e14 (γλ = 1e28): ArgumentError 'misses the domain'") do
  t, r, ch, y = _A_search(Constrained(SqrNormL2(1f14), IndHyperplane(1f0, 1f0)), zeros(Float32, 2), 1f14)
  present = _A_missesdomain(t, r)
  fixed = _A_a3(t, r, ch, y, y -> eltype(y) === Float32 && isapprox(y, Float32[0.5, 0.5]; rtol = 1e-5))
  verdict(present, fixed), "$(show_outcome(t, r)): $(t ? _A_msg(r) : "") y = $(short(y))"
end
case("LIM-019", "P6-82", "Float64 Constrained(Regularize(NormL1(1), 1e17), IndHyperplane(1, 1)), γ = 1e17: ArgumentError 'misses the domain'") do
  t, r, ch, y = _A_search(Constrained(Regularize(NormL1(1.0), 1e17), IndHyperplane(1.0, 1.0)), zeros(2), 1e17)
  present = _A_missesdomain(t, r)
  fixed = _A_a3(t, r, ch, y, y -> isapprox(y, [0.5, 0.5]; rtol = 1e-9))     # strictly convex and symmetric: [1/2, 1/2]
  verdict(present, fixed), "$(show_outcome(t, r)): $(t ? _A_msg(r) : "") y = $(short(y))"
end

# ---- LIM-020: hyperplane/halfspace coefficient rounded to a narrow output
for (nm, f) in (("IndHyperplane", IndHyperplane([2.0^100], 0.0)), ("IndHalfspace", IndHalfspace([2.0^100], 0.0))),
    (γ, gs) in ((1.0, "scalar step 1"), ([1.0], "array step [1.0]"))
  case("LIM-020", "P2-21", "$nm([2^100], 0) into Float32 at Float32[2^-100], $gs: argument returned unchanged") do
    y = zeros(Float32, 1)
    t, r, ch = writes(() -> prox!(y, f, Float32[2.0^-100], γ), y)
    present = !t && y == Float32[2.0^-100]
    fixed = !t && y == Float32[0]                      # T3, T4, R2: the exact result is 0
    verdict(present, fixed), "$(show_outcome(t, r)), y = $(short(y)) (exact Float32[0])"
  end
end

# ---- LIM-021: hyperplane/halfspace projection returns -Inf when ⟨a, x⟩ overflows
for (nm, f, x, ex) in (("IndHalfspace([1, 1], 0) at [M, M]", IndHalfspace([1.0, 1.0], 0.0), [M, M], [0.0, 0.0]),
                       ("IndHyperplane([1, 1], 0) at [M, M]", IndHyperplane([1.0, 1.0], 0.0), [M, M], [0.0, 0.0]),
                       ("IndHyperplane([1, 1], 1e308) at [M, M]", IndHyperplane([1.0, 1.0], 1e308), [M, M], [5e307, 5e307]),
                       ("IndHyperplane([1, 1], 0) at [M, M/2]", IndHyperplane([1.0, 1.0], 0.0), [M, M / 2], [M / 4, -M / 4]))
  case("LIM-021", "P2-19", "$nm: -Inf entries (exact $(short(ex)))") do
    y = zeros(2)
    t, r, ch = writes(() -> prox!(y, f, x, 1.0), y)
    present = !t && any(==(-Inf), y)
    fixed = !t && all(isfinite, y) && isapprox(y, ex; rtol = 1e-9, atol = 0)        # R2: representable closed-form entries are returned
    verdict(present, fixed), "$(show_outcome(t, r)), y = $(short(y))"
  end
end

# ================================================================ range and precision (LIM-022 to LIM-046)
# ===================== group B: LIM-022 .. LIM-046 =====================
# Helpers (prefixed _B_).
# relative error of a computed value `v` against a BigFloat reference `r` (Inf unless `v` is finite)
_B_rel(v, r) = (v isa Real && isfinite(v)) ? Float64(abs(big(v) - r) / abs(r)) : Inf
# `v` is finite and within `k` eps of the type of `v` of `r`: R1/R2 "the accuracy of the ordinary computation"
_B_near(v, r, k = 16) = v isa Real && isfinite(v) && _B_rel(v, r) <= k * eps(float(typeof(v)))
# a stated error `e` is within a factor 2 of the register's number `e0`
_B_band(e, e0) = e0 / 2 <= e <= 2 * e0
# "Float32-level" error for a Float64 result: far above eps(Float64) (2e-16), below 1e3 eps(Float32)
_B_f32(e) = 1e-10 < e < 1e-4
_B_isinf(v) = v isa Real && v == Inf
_B_isnan(v) = v isa Real && isnan(v)
_B_iszero(v) = v isa Real && iszero(v)
# R1 value case: `g()` is the value, `r` the BigFloat exact value, `bad(v)` the register's wrong value
function _B_value(lim, cl, what, g, r, bad; k = 16)
  case(lim, cl, what) do
    v = g()
    verdict(bad(v), _B_near(v, r, k)), "= $(short(v)); exact $(short(Float64(r)))"
  end
end
# deterministic data (no RNG): entries in [-5/7, 5/7], and vectors
_B_mat(T, m, n, s) = T[((3i + 5j + s) % 11 - 5) / 7 for i in 1:m, j in 1:n]
_B_vec(T, n, s) = T[((4i + s) % 9 - 4) / 5 + 1e-9 * i for i in 1:n]
_B_relmax(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))
function _B_mcp(λ, a, t)
  λ, a, t = big(λ), big(a), big(abs(t))
  t <= a * λ ? λ * t - t^2 / (2a) : a * λ^2 / 2
end
function _B_scad(λ, a, t)
  λ, a, t = big(λ), big(a), big(abs(t))
  t <= λ ? λ * t : t <= a * λ ? (2a * λ * t - t^2 - λ^2) / (2 * (a - 1)) : λ^2 * (a + 1) / 2
end

# ---- LIM-022: scalar kernels and losses lose range before the weight (R1)
_B_value("LIM-022", "P3-14", "Berhu(1e-300, 1)([1e200]) is Inf (exact 5e99)",
  () -> Berhu(1e-300, 1.0)([1e200]), big(1e-300) * (big(1e200)^2 + 1) / 2, _B_isinf)
_B_value("LIM-022", "P3-14", "CappedL1(1e-200, 1e300, [1e-200])([1e300]) is 0 (exact 1e-100)",
  () -> CappedL1(1e-200, 1e300, [1e-200])([1e300]), big(1e-200) * big(1e-200) * big(1e300), _B_iszero)
_B_value("LIM-022", "P3-14", "HingeLoss([2], [0.25])([-M]) is Inf (exact M/2 + 0.25)",
  () -> HingeLoss([2.0], [0.25])([-M]), big(0.25) * (1 + 2 * big(M)), _B_isinf)
_B_value("LIM-022", "P3-14", "CheckLoss(1e-200, [1e300])([1e-200]) is 0 (exact 1e-100)",
  () -> CheckLoss(1e-200, [1e300])([1e-200]), big(1e300) * big(1e-200) * big(1e-200), _B_iszero)
_B_value("LIM-022", "P3-14", "LogPenalty(1, 1e-300)([1e300]) is Inf (exact log(1 + 1e600) = 1381.55)",
  () -> LogPenalty(1.0, 1e-300)([1e300]), log1p(big(1e300) / big(1e-300)), _B_isinf)
_B_value("LIM-022", "P3-14", "MCP(1e154, 3)([2e154]) is NaN (exact 1.33e308)",
  () -> MCP(1e154, 3.0)([2e154]), _B_mcp(1e154, 3.0, 2e154), _B_isnan)
_B_value("LIM-022", "P3-14", "SCAD(8e153, 3)([1.6e154]) is NaN (exact 1.12e308)",
  () -> SCAD(8e153, 3.0)([1.6e154]), _B_scad(8e153, 3.0, 1.6e154), _B_isnan)
_B_value("LIM-022", "P3-14", "Float32 NegLog(1e40)([nextfloat(1f0)]) is -Inf (exact -1.19e33)",
  () -> NegLog(1e40)(Float32[nextfloat(1f0)]), -big(1e40) * log(big(nextfloat(1f0))), v -> v isa Real && v == -Inf)
_B_value("LIM-022", "P3-14", "Squared()(1.5e154, 0) is Inf (exact 1.125e308)",
  () -> PB.Squared()(1.5e154, 0.0), big(1.5e154)^2 / 2, _B_isinf)
_B_value("LIM-022", "P3-14", "Huber(2e154)(1.5e154, 0) is Inf (exact 1.125e308)",
  () -> PB.Huber(2e154)(1.5e154, 0.0), big(1.5e154)^2 / 2, _B_isinf)
_B_value("LIM-022", "P3-14", "Poisson()(710, 1e305) is Inf (exact exp(710) - 710e305 = 1.52e308)",
  () -> PB.Poisson()(710.0, 1e305), exp(big(710)) - 710 * big(1e305), _B_isinf)
_B_value("LIM-022", "P3-29", "SingularValue(MCP(1e154, 3))(diag(2e154, 0)) is NaN (exact 1.33e308)",
  () -> SingularValue(MCP(1e154, 3.0))(Matrix(Diagonal([2e154, 0.0]))), _B_mcp(1e154, 3.0, 2e154), _B_isnan)
_B_value("LIM-022", "P3-29", "Spectral(SCAD(8e153, 3.7))(diag(1.6e154, 0)) is NaN (exact 1.16e308)",
  () -> Spectral(SCAD(8e153, 3.7))(Matrix(Diagonal([1.6e154, 0.0]))), _B_scad(8e153, 3.7, 1.6e154), _B_isnan)

# ---- LIM-023: Translate/Precompose of a norm do not recover through homogeneity (R1, W1)
_B_value("LIM-023", "P3-15", "W1: Translate(NormL1(0.25), [-M])([M]) is Inf (contract M/2)",
  () -> Translate(NormL1(0.25), [-M])([M]), big(M) / 2, _B_isinf, k = 4)
_B_value("LIM-023", "P3-15", "Precompose(NormL1(0.25), 2.0)([M]) is Inf (exact M/2)",
  () -> Precompose(NormL1(0.25), 2.0)([M]), big(M) / 2, _B_isinf, k = 4)
_B_value("LIM-023", "P3-15", "Precompose(NormL1(0.25), [2.0])([M]) (array scale) is Inf (exact M/2)",
  () -> Precompose(NormL1(0.25), [2.0])([M]), big(M) / 2, _B_isinf, k = 4)
_B_value("LIM-023", "P6-46", "Translate(NormL2(0.25), [-M])([M]) is Inf (exact M/2)",
  () -> Translate(NormL2(0.25), [-M])([M]), big(M) / 2, _B_isinf, k = 4)
_B_value("LIM-023", "P6-46", "Translate(NormLinf(0.25), [-M])([M]) is Inf (exact M/2)",
  () -> Translate(NormLinf(0.25), [-M])([M]), big(M) / 2, _B_isinf, k = 4)
_B_value("LIM-023", "P6-46", "Precompose(NormL1(0.25), 4.0)([M]) is Inf (exact M)",
  () -> Precompose(NormL1(0.25), 4.0)([M]), big(M), _B_isinf, k = 4)
_B_value("LIM-023", "P6-46", "Precompose(NormL1(0.25), [2 0; 0 2], 4.0)([M, M]) (matrix) is Inf (exact M)",
  () -> Precompose(NormL1(0.25), [2.0 0.0; 0.0 2.0], 4.0)([M, M]), big(M), _B_isinf, k = 4)

# ---- LIM-024: values of sums and calculus wrappers lose range (R1, decision 45)
_B_value("LIM-024", "P3-17", "Scaled(SqrNormL2(), 1e-200)([1e200]) is Inf (exact 5e199)",
  () -> Scaled(SqrNormL2(), 1e-200)([1e200]), big(1e-200) * big(1e200)^2 / 2, _B_isinf)
_B_value("LIM-024", "P2-42", "Scaled(SqrNormL2(1), 2^-1000)([2^600, 2^600]) is Inf (exact 2^200)",
  () -> Scaled(SqrNormL2(1.0), 2.0^-1000)([2.0^600, 2.0^600]), big(2.0)^200, _B_isinf)
_B_value("LIM-024", "P2-42", "Scaled(SqrNormL2(2^-1000), 2^1000)([2^-100]) is 0 (exact 2^-201)",
  () -> Scaled(SqrNormL2(2.0^-1000), 2.0^1000)([2.0^-100]), big(2.0)^-201, _B_iszero)
_B_value("LIM-024", "P6-48", "Scaled(SqrNormL2(1), 1e-300)([1e200]) is Inf (exact 5e99)",
  () -> Scaled(SqrNormL2(1.0), 1e-300)([1e200]), big(1e-300) * big(1e200)^2 / 2, _B_isinf)
_B_lf(c) = LinearFunction([0.0], c)
_B_value("LIM-024", "P3-17", "Sum of LinearFunctions with constants M, M, -M at [0] is Inf (exact M)",
  () -> Sum(_B_lf(M), _B_lf(M), _B_lf(-M))([0.0]), big(M), _B_isinf)
_B_value("LIM-024", "P3-17", "SeparableSum of LinearFunctions with constants M, M, -M is Inf (exact M)",
  () -> SeparableSum(_B_lf(M), _B_lf(M), _B_lf(-M))(([0.0], [0.0], [0.0])), big(M), _B_isinf)
_B_value("LIM-024", "P3-17", "SliceSum of LinearFunctions with constants M, M, -M is Inf (exact M)",
  () -> SliceSum([_B_lf(M), _B_lf(M), _B_lf(-M)])(zeros(1, 3)), big(M), _B_isinf)
_B_value("LIM-024", "P3-17", "Tilt(NormL1(), [1], -M)([M]) is Inf (exact M + M - M = M)",
  () -> Tilt(NormL1(), [1.0], -M)([M]), big(M), _B_isinf)
_B_value("LIM-024", "P2-43", "Conjugate(IndBox(1, 2))([0.75M, -M]) is Inf (exact 0.5M)",
  () -> Conjugate(IndBox(1.0, 2.0))([0.75 * M, -M]), 2 * big(0.75 * M) - big(M), _B_isinf)
_B_value("LIM-024", "P2-43", "Conjugate(IndBox(1, 2))([-M, 0.75M]) is Inf (exact 0.5M)",
  () -> Conjugate(IndBox(1.0, 2.0))([-M, 0.75 * M]), -big(M) + 2 * big(0.75 * M), _B_isinf)
case("LIM-024", "P3-17", "MoreauEnvelope(LinearFunction([2e154]), 1)([1e154]) is NaN (exact 0, not a normal value)") do
  v = MoreauEnvelope(LinearFunction([2e154]), 1.0)([1e154])
  # exact c x - c^2/2 = 0; R1 does not cover it, but a NaN from finite input is not allowed (E2) and 0 is the exact value
  verdict(_B_isnan(v), v isa Real && iszero(v)), "= $(short(v)); exact 0"
end
_B_value("LIM-024", "P3-17", "MoreauEnvelope(LinearFunction([2e154]), 1)([1.5e154]) normal-valued variant is Inf (exact 1e308)",
  () -> MoreauEnvelope(LinearFunction([2e154]), 1.0)([1.5e154]), big(2e154) * big(1.5e154) - big(2e154)^2 / 2, _B_isinf)
case("LIM-024", "P3-17", "Conjugate(Scaled(IndBallL2(), s))([s]), s = nextfloat(0.0), is Inf (exact s, subnormal)") do
  s = nextfloat(0.0)
  v = Conjugate(Scaled(IndBallL2(), s))([s])
  # R4: no overflow where the exact value is s; a subnormal exact value may be off by about floatmin
  verdict(_B_isinf(v), v isa Real && isfinite(v) && abs(v - s) <= floatmin(Float64)), "= $(short(v)); exact $(short(s))"
end
_B_value("LIM-024", "P3-17", "GroupMCP(1e200, [[1]], [1e200])([1e-200]) is Inf (exact 1e200)",
  () -> GroupMCP(1e200, [[1]], [1e200])([1e-200]), big(1e200) * _B_mcp(1e200, 3.0, 1e-200), _B_isinf)
_B_value("LIM-024", "P3-15", "LinearImage(NormL1(1e-300), [0.5;;], 0.25)([M]) is Inf (exact 1e-300 * 2M = 3.6e8)",
  () -> LinearImage(NormL1(1e-300), [0.5;;], 0.25)([M]), big(1e-300) * 2 * big(M), _B_isinf)
_B_value("LIM-024", "P3-17", "LinearImage(QuadraticFunction([2;;], [-1e154]), [1;;])([1.5e154]) is Inf (exact 7.5e307)",
  () -> LinearImage(QuadraticFunction([2.0;;], [-1e154]), ones(1, 1))([1.5e154]), big(1.5e154)^2 - big(1e154) * big(1.5e154), _B_isinf)

# ---- LIM-025: value_and_prox! of Scaled returns 0 when the inner value underflows (O2, R1)
case("LIM-025", "P2-44", "value_and_prox! of Scaled(NormL1(2^-1000), 2^1000) at [2^-100], γ = 2^-1000: returns 0.0 while f(y) = 2^-100") do
  f = Scaled(NormL1(2.0^-1000), 2.0^1000); y = zeros(1)
  v = value_and_prox!(y, f, [2.0^-100], 2.0^-1000)
  ref = big(2.0)^1000 * big(2.0)^-1000 * big(y[1])            # the value at the stored output
  verdict(v == 0.0 && f(y) == 2.0^-100, _B_near(v, ref)), "returned $(short(v)), f(y) = $(short(f(y))), y = $(short(y))"
end

# ---- LIM-026: norms rounded before a large weight; signed reductions overflow (R1)
_B_s = nextfloat(0.0)
_B_l2ref = big(1e300) * sqrt(big(2)) * big(_B_s)
_B_value("LIM-026", "P3-18", "NormL2(1e300)([s, s]), s = nextfloat(0.0), is 4.94e-24 (exact 6.99e-24)",
  () -> NormL2(1e300)([_B_s, _B_s]), _B_l2ref, v -> v isa Real && v == 1e300 * _B_s)
_B_value("LIM-026", "P3-18", "GroupNormL2(1e300, [[1, 2]], [1])([s, s]) is 4.94e-24 (exact 6.99e-24)",
  () -> GroupNormL2(1e300, [[1, 2]], [1.0])([_B_s, _B_s]), _B_l2ref, v -> v isa Real && v == 1e300 * _B_s)
_B_value("LIM-026", "P3-18", "DistL2(IndBox(0, 0), 1e300)([s, s]) is 4.94e-24 (exact 6.99e-24)",
  () -> DistL2(IndBox(0.0, 0.0), 1e300)([_B_s, _B_s]), _B_l2ref, v -> v isa Real && v == 1e300 * _B_s)
let t = [exp(-1.0), exp(-1.0), exp(1.0)]
  _B_value("LIM-026", "P3-18", "NegLog(1e308)([e^-1, e^-1, e]) is Inf (exact 1e308)",
    () -> NegLog(1e308)(t), -big(1e308) * sum(log.(big.(t))), _B_isinf)
end
_B_value("LIM-026", "P3-18", "Conjugate(SqrHingeLoss(y, w))([-M, -M, -M]) is Inf (exact M)",
  () -> Conjugate(SqrHingeLoss([1.0, 1.0, 0.5], [M / 8, M / 8, M]))([-M, -M, -M]),
  sum(big(s) / big(yy) + big(s)^2 / (4 * big(ww) * big(yy)^2) for (s, yy, ww) in zip(fill(-M, 3), [1.0, 1.0, 0.5], [M / 8, M / 8, M])),
  _B_isinf)

# ---- LIM-027: proximal outputs of Berhu, HingeLoss and NormL1Fused overflow (R2)
case("LIM-027", "P3-19", "prox(Berhu(1, 1e200), [1e300], 1) is [Inf] (exact 1e300)") do
  y = prox(Berhu(1.0, 1e200), [1e300], 1.0)
  ref = big(1e300) * big(1e200) / (big(1e200) + 1)
  verdict(y == [Inf], _B_near(y[1], ref)), "$(short(y)); exact $(short(Float64(ref)))"
end
case("LIM-027", "P3-19", "prox(HingeLoss([2]), [M], 1) is [Inf] (exact [M])") do
  y = prox(HingeLoss([2.0]), [M], 1.0)
  verdict(y == [Inf], _B_near(y[1], big(M))), "$(short(y)); exact [M]"
end
case("LIM-027", "P3-19", "prox(NormL1Fused(0, 1), [M, M, M], 1e308) is all NaN (exact [M, M, M])") do
  y = prox(NormL1Fused(0.0, 1.0), [M, M, M], 1e308)
  verdict(all(isnan, y), all(v -> _B_near(v, big(M)), y)), "$(short(y)); exact [M, M, M]"
end
case("LIM-027", "P3-23", "value_and_prox!(NormL1Fused(0, 1), [M, M, M], 1e308): preprocessing a - t(2k-n-1) overflows, NaN value and output (exact 0, [M, M, M])") do
  y = zeros(3); v = value_and_prox!(y, NormL1Fused(0.0, 1.0), [M, M, M], 1e308)
  verdict(isnan(v) && all(isnan, y), iszero(v) && all(w -> _B_near(w, big(M)), y)), "value $(short(v)), $(short(y)); exact 0, [M, M, M]"
end

# ---- LIM-028: exponential-cone projections when an output component overflows (R4, R2)
function _B_expcone(lim, cl, what, f, x)
  case(lim, cl, what) do
    y = zeros(3)
    t, r, ch = writes(() -> prox!(y, f, x, 1.0), y)
    ref = prox(f, x ./ 2.0^20, 1.0) .* 2.0^20       # cones are homogeneous: the scale-consistent reference
    present = !t && y[1] == 0 && y[2] == 0 && y[3] == M
    fixed = rejects(PrecisionError, t, r, ch) || (!t && isapprox(y, ref; rtol = 1e-6))
    verdict(present, fixed), "$(show_outcome(t, r)) $(t ? "" : short(y)); reference $(short(ref))"
  end
end
_B_expcone("LIM-028", "P2-40", "prox(IndExpDual(), [M, -M, M]) is [0, 0, M]", IndExpDual(), [M, -M, M])
_B_expcone("LIM-028", "P2-40", "prox(IndExpPrimal(), [M, M, M]) is [0, 0, M]", IndExpPrimal(), [M, M, M])

# ---- LIM-029: weighted NormL1 value loses underflowing products (R1)
case("LIM-029", "P3-27", "Float32 NormL1(1, fill(2^-126, 512))(fill(2^-9 + 2^-24, 512)) is floatmin (exact 1.1755302e-38, 256 ulps low)") do
  w = fill(2f0^-126, 512); x = fill(Float32(2.0^-9 + 2.0^-24), 512)
  v = NormL1(1f0, w)(x); ref = sum(big.(w) .* big.(x))
  verdict(v == floatmin(Float32), isfinite(v) && abs(big(v) - ref) <= 4 * eps(Float32(ref))), "= $(short(v)); exact $(short(Float32(ref))), error $(round(Float64(abs(big(v) - ref) / eps(Float32(ref))))) ulps"
end

# ---- LIM-030: spectral wrappers overflow at a finite matrix whose spectrum overflows (R1, decision 45)
let X = fill(M, 2, 2)
  _B_value("LIM-030", "P2-27", "SingularValue(NormL1(0.25))(fill(M, 2, 2)) is Inf (NuclearNorm(0.25) = M/2)",
    () -> SingularValue(NormL1(0.25))(X), big(M) / 2, _B_isinf, k = 4)
  _B_value("LIM-030", "P2-27", "Spectral(NormL1(0.25))(fill(M, 2, 2)) is Inf (exact M/2)",
    () -> Spectral(NormL1(0.25))(X), big(M) / 2, _B_isinf, k = 4)
  _B_value("LIM-030", "P2-27", "SingularValue(NormL2(0.25))(fill(M, 2, 2)) is Inf (exact M/2)",
    () -> SingularValue(NormL2(0.25))(X), big(M) / 2, _B_isinf, k = 4)
  _B_value("LIM-030", "P2-27", "SingularValue(NormLinf(0.25))(fill(M, 2, 2)) is Inf (exact M/2)",
    () -> SingularValue(NormLinf(0.25))(X), big(M) / 2, _B_isinf, k = 4)
  _B_value("LIM-030", "P2-27", "SingularValue(ElasticNet(0.25, 0))(fill(M, 2, 2)) is Inf (exact M/2)",
    () -> SingularValue(ElasticNet(0.25, 0.0))(X), big(M) / 2, _B_isinf, k = 4)
  _B_value("LIM-030", "P2-27", "Conjugate(SingularValue(IndBallL2(0.25)))(fill(M, 2, 2)) is Inf (exact M/2)",
    () -> Conjugate(SingularValue(IndBallL2(0.25)))(X), big(M) / 2, _B_isinf, k = 4)
end
_B_value("LIM-030", "P3-28", "SingularValue(Scaled(SqrNormL2(), 1e-300))(diag(1e200, 1e200)) is Inf (exact 1e100)",
  () -> SingularValue(Scaled(SqrNormL2(), 1e-300))(Matrix(Diagonal([1e200, 1e200]))), big(1e-300) * 2 * big(1e200)^2 / 2, _B_isinf)
_B_value("LIM-030", "P3-28", "Spectral(Scaled(SqrNormL2(), 1e-300))(diag(1e200, 1e200)) is Inf (exact 1e100)",
  () -> Spectral(Scaled(SqrNormL2(), 1e-300))(Matrix(Diagonal([1e200, 1e200]))), big(1e-300) * 2 * big(1e200)^2 / 2, _B_isinf)

# ---- LIM-031: QuadraticFunction and KronQuadratic values lose accuracy when (Ax)_i underflows (R1)
let a = 1.37 * 2.0^-970, x = [1.23 * 2.0^1000, 1.11 * 2.0^-60]
  ref = big(a) * big(x[1]) * big(x[2])
  case("LIM-031", "P3-47", "QuadraticFunction([0 a; a 0])(x), a = 1.37 * 2^-970: relative error 7.8e-15 (about 35 eps)") do
    v = QuadraticFunction([0 a; a 0])(x); e = _B_rel(v, ref)
    verdict(_B_band(e, 7.8e-15), e <= 4 * eps()), "relative error $(round(e; sigdigits = 3)) = $(round(e / eps(); sigdigits = 3)) eps"
  end
  case("LIM-031", "P3-47", "KronQuadratic([0 a; a 0], [1;;]) at x (the same product): relative error 7.8e-15") do
    v = KronQuadratic([0 a; a 0], ones(1, 1))(reshape(x, 2, 1)); e = _B_rel(v, ref)
    verdict(_B_band(e, 7.8e-15), e <= 4 * eps()), "relative error $(round(e; sigdigits = 3)) = $(round(e / eps(); sigdigits = 3)) eps"
  end
end

# ---- LIM-032: GLMLoss values do not recover weights and predictors that leave the range (R1)
_B_value("LIM-032", "P3-48", "PoissonLoss([0], [1;;]; weights = [1e-300])([1000]) is Inf (exact 1.97e134)",
  () -> PoissonLoss([0.0], ones(1, 1); weights = [1e-300])([1000.0]), big(1e-300) * exp(big(1000)), _B_isinf)
_B_value("LIM-032", "P3-48", "LogisticLoss([0], [1;;]; weights = [1e300])([-1000]) is 0.0 (exact 5.08e-135)",
  () -> LogisticLoss([0.0], ones(1, 1); weights = [1e300])([-1000.0]), big(1e300) * log1p(exp(big(-1000))), _B_iszero)
_B_value("LIM-032", "P3-48", "GLMLoss(SquaredHinge(), [1], [1;;]; weights = [1e-300])([-1e200]) is Inf (exact 1e100)",
  () -> GLMLoss(PB.SquaredHinge(), [1.0], ones(1, 1); weights = [1e-300])([-1e200]), big(1e-300) * (1 + big(1e200))^2, _B_isinf)
_B_value("LIM-032", "P3-48", "PoissonLoss([0], [M -M])([2, 2]), predictor 2M - 2M = 0, is NaN (exact 1)",
  () -> PoissonLoss([0.0], reshape([M, -M], 1, 2))([2.0, 2.0]), big(1), _B_isnan)

# ---- LIM-033: GroupMCP/GroupSCAD prox NaN when the group norm overflows the argument type (R2)
function _B_groupcap(lim, cl, what, f, x, D, nanidx)
  case(lim, cl, what) do
    y = zeros(D, length(x)); prox!(y, f, x, 1.0)
    rest = setdiff(eachindex(y), nanidx)
    present = all(i -> isnan(y[i]), nanidx) && all(i -> y[i] == D(x[i]), rest)
    # beyond the cap the proximal map is the identity: every entry is x (within 4 eps of D)
    fixed = all(i -> _B_near(y[i], big(x[i]), 4), eachindex(y))
    verdict(present, fixed), "$(short(y)); exact identity $(short(Float64.(x)))"
  end
end
_B_x32 = Float32[3e38, 3e38, 3e38]
_B_groupcap("LIM-033", "P3-46", "Float32 [3e38]*3, GroupMCP([1:2, 3:3], a = 3), Float64 destination: [NaN, NaN, 3e38] (exact identity)",
  GroupMCP(1.0, [1:2, 3:3], [1.0, 1.0]; a = 3), _B_x32, Float64, 1:2)
_B_groupcap("LIM-033", "P3-46", "Float32 [3e38]*3, GroupMCP([1:2, 3:3], a = 3), Float32 destination: [NaN, NaN, 3e38]",
  GroupMCP(1.0, [1:2, 3:3], [1.0, 1.0]; a = 3), _B_x32, Float32, 1:2)
_B_groupcap("LIM-033", "P3-46", "Float32 [3e38]*3, GroupSCAD([1:2, 3:3], a = 3.7), Float64 destination: [NaN, NaN, 3e38]",
  GroupSCAD(1.0, [1:2, 3:3], [1.0, 1.0]; a = 3.7), _B_x32, Float64, 1:2)
_B_groupcap("LIM-033", "P6-50", "Float64 [M, M], GroupMCP([[1, 2]], a = 2): [NaN, NaN] (exact [M, M])",
  GroupMCP(1.0, [[1, 2]]; a = 2.0), [M, M], Float64, 1:2)
_B_groupcap("LIM-033", "P6-50", "Float64 [M, M], GroupSCAD([[1, 2]], a = 3): [NaN, NaN] (exact [M, M])",
  GroupSCAD(1.0, [[1, 2]]; a = 3.0), [M, M], Float64, 1:2)

# ---- LIM-034: resolved by #549 (see "Resolved since the audit" in known-limitations.md)

# ---- LIM-035: Poisson quadratic model holds an infinite weight and a NaN working response at η = 710 (S2, O6.3)
_B_pq() = PoissonLoss([1.0], ones(1, 1); weights = [1 / 8])
case("LIM-035", "P3-03", "quadratic_model(PoissonLoss(weights = 1/8), [710]): weights == [Inf], y == [NaN], no exception") do
  t, r = attempt(() -> quadratic_model(_B_pq(), [710.0]))
  present = !t && r.weights == [Inf] && all(isnan, r.y)
  ref = exp(big(710)) / 8
  fixed = (t && r isa PrecisionError) || (!t && _B_near(r.weights[1], ref, 4) && all(isfinite, r.y))
  verdict(present, fixed), t ? "throws $(nameof(typeof(r)))" : "weights $(short(r.weights)), y $(short(r.y)); exact weight $(short(Float64(ref)))"
end
case("LIM-035", "P3-03", "quadratic_model!(q, PoissonLoss(weights = 1/8), [710]): q written with weights [Inf], y [NaN]") do
  f = _B_pq(); q = quadratic_model(f, [0.0]); w0, y0 = copy(q.weights), copy(q.y)
  t, r = attempt(() -> quadratic_model!(q, f, [710.0]))
  unchanged = q.weights == w0 && isequal(q.y, y0)
  ref = exp(big(710)) / 8
  present = !t && q.weights == [Inf] && all(isnan, q.y)
  fixed = (t && r isa PrecisionError && unchanged) || (!t && _B_near(q.weights[1], ref, 4) && all(isfinite, q.y))
  verdict(present, fixed), t ? "throws $(nameof(typeof(r))), q unchanged = $unchanged" : "weights $(short(q.weights)), y $(short(q.y)); exact weight $(short(Float64(ref)))"
end
case("LIM-035", "P4-66", "quadratic_model(PoissonLoss([0], [1;;]) with default weight, [710]): weights [Inf], y [NaN] (exp(710) > floatmax: only PrecisionError is correct)") do
  t, r = attempt(() -> quadratic_model(PoissonLoss([0.0], ones(1, 1)), [710.0]))
  verdict(!t && r.weights == [Inf] && all(isnan, r.y), t && r isa PrecisionError), t ? "throws $(nameof(typeof(r)))" : "weights $(short(r.weights)), y $(short(r.y))"
end

# ---- LIM-036: Constrained(NormL1, IndSphereL2) raises when γλw overflows (R3)
case("LIM-036", "P3-55", "prox!(Constrained(NormL1(1e300), IndSphereL2(1)), [1, 2], 1e300): ArgumentError 'sphere thresholds must be finite' (R3 result [0, 1])") do
  f = Constrained(NormL1(1e300), IndSphereL2(1.0)); y = [7.0, 7.0]
  t, r, ch = writes(() -> prox!(y, f, [1.0, 2.0], 1e300), y)
  present = t && r isa ArgumentError && occursin("sphere thresholds must be finite", sprint(showerror, r))
  fixed = !t && isapprox(y, [0.0, 1.0]; atol = 1e-12)            # R3: as if γλw were not rounded: the best unit vector
  verdict(present, fixed), "$(show_outcome(t, r)) ($(t ? shorterr(r) : "")), y written = $ch, y = $(short(y))"
end

# docstring text of Smoothed (top level: @doc returns a DocStr on Julia 1.13 and Markdown.MD on 1.10)
const _B_smoothed_doc = let dd = @doc(PB.Smoothed)
  dd isa Base.Docs.DocStr ? join(dd.text) : string(dd)
end
# ---- LIM-037: deriv2 of Smoothed is a central finite difference of about √eps accuracy (A1)
case("LIM-037", "P2-12", "deriv2(Smoothed(NegEntropy(1), 0.5), 0.7, 0): relative error 6.3e-9 (2.8e7 eps), docstring silent") do
  d = PB.deriv2(PB.Smoothed(NegEntropy(1.0), 0.5), 0.7, 0.0)
  ref = let z = big(0.5), η = big(0.7), γ = big(0.5)       # closed form: φ''(z)/(1 + γ φ''(z)) at z = prox, φ'' = 1/z
    for _ in 1:200; z -= (z + γ * (log(z) + 1) - η) / (1 + γ / z); end
    (1 / z) / (1 + γ / z)
  end
  e = _B_rel(d, ref)
  doc = _B_smoothed_doc
  stated = occursin(r"finite.difference|error scale"i, doc)    # A1 allows a stated error scale
  verdict(_B_band(e, 6.3e-9) && !stated, e <= 16 * eps() || stated), "relative error $(round(e; sigdigits = 3)) = $(round(e / eps(); sigdigits = 3)) eps; docstring states a scale = $stated"
end

# ---- LIM-038: narrow data at a wider argument computes in the data's precision (T3, T4, decision 6)
case("LIM-038", "P3-30", "T3 example: gradient!(Float32 g, QuadraticFunction(Float32[1;;], Float32[-1]), [1 + 2^-28]) is Float32[0] (contract 2^-28)") do
  g = zeros(Float32, 1); gradient!(g, QuadraticFunction(Float32[1;;], Float32[-1]), [1 + 2.0^-28])
  verdict(g == Float32[0], g == Float32[2.0^-28]), "$(short(g)); contract Float32[2^-28]"
end
let A32 = (A = _B_mat(Float64, 6, 6, 1); Matrix{Float32}(A' * A + 2I)), c32 = _B_vec(Float32, 6, 2), x = _B_vec(Float64, 6, 3) .+ 1e-9
  A32 = (A32 + A32') / 2
  f32, f64 = QuadraticFunction(A32, c32), QuadraticFunction(Float64.(A32), Float64.(c32))
  case("LIM-038", "P3-30", "QuadraticFunction(Float32 data) prox at a Float64 argument vs Float64 data: Float32-level difference") do
    e = _B_relmax(prox(f32, x, 0.7), prox(f64, x, 0.7))
    verdict(_B_f32(e), e <= 64 * eps()), "max relative difference $(round(e; sigdigits = 3)) ($(round(e / eps(); sigdigits = 3)) eps)"
  end
end
let q32 = QuadraticFunction(Float32[2 0.3; 0.3 2], zeros(Float32, 2)), q64 = QuadraticFunction(Float64.(Float32[2 0.3; 0.3 2]), zeros(2)), x = [1 + 1e-9, 2 + 1e-9]
  case("LIM-038", "P2-34", "QuadraticFunction(Float32[2 0.3; 0.3 2], 0) value at [1 + 1e-9, 2 + 1e-9]: differs from the Float64-data function by 2.35e-7") do
    d = q32(x) - q64(x)
    verdict(_B_band(abs(d), 2.35e-7), abs(d) <= 8 * eps() * abs(q64(x))), "difference $(short(d)) (value $(short(q64(x))))"
  end
  case("LIM-038", "P2-34", "same function: standalone value and value_and_gradient! disagree (vag agrees with Float64 data, difference 0)") do
    v, vg, ref = q32(x), value_and_gradient!(zeros(2), q32, x), q64(x)
    verdict(_B_band(abs(v - ref), 2.35e-7) && abs(vg - ref) <= 8 * eps() * abs(ref), abs(v - ref) <= 8 * eps() * abs(ref) && abs(vg - ref) <= 8 * eps() * abs(ref)),
      "standalone - Float64 data = $(short(v - ref)), value_and_gradient! - Float64 data = $(short(vg - ref))"
  end
end
let X32 = _B_mat(Float32, 3, 2, 3), y32 = Float32[1, 2, 3.3], b = [1 + 1e-9, 2 + 1e-9]
  l32, l64 = LeastSquaresLoss(y32, X32), LeastSquaresLoss(Float64.(y32), Float64.(X32))
  case("LIM-038", "P2-34", "LeastSquaresLoss(Float32 data) value at a Float64 argument: Float32-level difference") do
    e = abs(l32(b) - l64(b)) / abs(l64(b))
    verdict(_B_f32(e), e <= 8 * eps()), "relative difference $(round(e; sigdigits = 3))"
  end
end
let X32 = _B_mat(Float32, 8, 5, 2), y32 = _B_vec(Float32, 8, 1), β = _B_vec(Float64, 5, 6)
  l32, l64 = LeastSquaresLoss(y32, X32), LeastSquaresLoss(Float64.(y32), Float64.(X32))
  case("LIM-038", "P3-30", "LeastSquaresLoss(Float32 data) gradient (Float64 destination) at a Float64 argument: Float32-level difference") do
    e = _B_relmax(gradient(l32, β), gradient(l64, β))
    verdict(_B_f32(e), e <= 64 * eps()), "max relative difference $(round(e; sigdigits = 3))"
  end
end
let K1 = (K = Float32.(_B_mat(Float64, 3, 3, 4)); (K + K') / 2), K2 = (K = Float32.(_B_mat(Float64, 4, 4, 6)); (K + K') / 2), X = reshape(_B_vec(Float64, 12, 5), 3, 4)
  k32, k64 = KronQuadratic(Matrix{Float32}(K1), Matrix{Float32}(K2)), KronQuadratic(Float64.(K1), Float64.(K2))
  case("LIM-038", "P3-30", "KronQuadratic(Float32 data) value at a Float64 argument: Float32-level difference") do
    e = abs(k32(X) - k64(X)) / abs(k64(X))
    verdict(_B_f32(e), e <= 8 * eps()), "relative difference $(round(e; sigdigits = 3))"
  end
  case("LIM-038", "P3-30", "KronQuadratic(Float32 data) gradient at a Float64 argument: Float32-level difference") do
    e = _B_relmax(gradient(k32, X), gradient(k64, X))
    verdict(_B_f32(e), e <= 64 * eps()), "max relative difference $(round(e; sigdigits = 3))"
  end
end
let Aa = _B_mat(Float32, 2, 5, 3), ba = _B_vec(Float32, 2, 2), xa = _B_vec(Float64, 5, 1)
  case("LIM-038", "P3-30", "IndAffine(Float32 data) prox at a Float64 argument: the residual is Float32-level (Float64 data: 1e-16)") do
    y = prox(IndAffine(Aa, ba), xa); res = norm(big.(Aa) * big.(y) - big.(ba))
    verdict(_B_f32(Float64(res)), res <= 1e-14), "residual $(short(Float64(res)))"
  end
end
let S = Float32[2 0.1; 0.1 3], X = [2.0 0.5; 0.5 3.0]
  case("LIM-038", "P2-32", "LogDetLoss(Float32 S) value_and_prox! at a Float64 argument and destination: max |Y - Y_ref| = 1.6e-7") do
    Y1, Y2 = zeros(2, 2), zeros(2, 2)
    value_and_prox!(Y1, LogDetLoss(S, 1f0), X, 0.5); value_and_prox!(Y2, LogDetLoss(Float64.(S), 1.0), X, 0.5)
    e = maximum(abs.(Y1 - Y2))
    verdict(_B_band(e, 1.6e-7), e <= 64 * eps() * maximum(abs.(Y2))), "max |Y - Y_ref| = $(short(e))"
  end
  case("LIM-038", "P2-32", "LogDetLoss(Float32 S) value_and_prox! value at a Float64 argument: differs from Float64 S by 3.6e-7") do
    v1 = value_and_prox!(zeros(2, 2), LogDetLoss(S, 1f0), X, 0.5); v2 = value_and_prox!(zeros(2, 2), LogDetLoss(Float64.(S), 1.0), X, 0.5)
    verdict(_B_band(abs(v1 - v2), 3.6e-7), abs(v1 - v2) <= 64 * eps() * abs(v2)), "difference $(short(abs(v1 - v2)))"
  end
  case("LIM-038", "P2-32", "LogDetLoss(Float32 S), Float32 argument, Float64 destination: max |Y - Y_ref| = 1.6e-7") do
    X32 = Float32.(X); Y1, Y2 = zeros(2, 2), zeros(2, 2)
    value_and_prox!(Y1, LogDetLoss(S, 1f0), X32, 0.5); value_and_prox!(Y2, LogDetLoss(Float64.(S), 1.0), Float64.(X32), 0.5)
    e = maximum(abs.(Y1 - Y2))
    verdict(_B_band(e, 1.6e-7), e <= 64 * eps() * maximum(abs.(Y2))), "max |Y - Y_ref| = $(short(e))"
  end
end

# ---- LIM-039: separable kernels with narrow parameters and a wider destination (T4, V10)
case("LIM-039", "P3-31", "V10: prox!(Float64 z, NormL1(1 + 2^-23), Float32[1], 1 - 2^-23) is [0.0] (contract [2^-46])") do
  z = zeros(1); prox!(z, NormL1(1 + 2f0^-23), Float32[1], 1 - 2f0^-23)
  verdict(z == [0.0], z == [2.0^-46]), "$(short(z)); contract [2^-46]"
end

# ---- LIM-040: threshold products formed in the factors' type (T3)
let x = [2.0^161, 1.0], g = 2f0^80, G = Float64(2f0^80)
  case("LIM-040", "P3-43", "prox(TotalVariation1D(2f0^80), [2^161, 1], 2f0^80) is [NaN, NaN] (widened factors: [1.46e48, 1.46e48])") do
    y = prox(TotalVariation1D(g), x, g); ref = prox(TotalVariation1D(G), x, G)
    verdict(all(isnan, y), isapprox(y, ref; rtol = 1e-12, atol = 0)), "$(short(y)); widened factors $(short(ref))"
  end
  case("LIM-040", "P3-43", "prox(FusedLasso(2f0^80, 2f0^80), [2^161, 1], 2f0^80) is [NaN, NaN] (widened factors: [0, 0])") do
    y = prox(FusedLasso(g, g), x, g); ref = prox(FusedLasso(G, G), x, G)
    verdict(all(isnan, y), isapprox(y, ref; rtol = 1e-12, atol = 0)), "$(short(y)); widened factors $(short(ref))"
  end
  case("LIM-040", "P3-43", "prox(CheckLoss(0.5f0, fill(2f0^80, 2)), [2^161, 1], 2f0^80) is [0, 0] (widened factors: [2.19e48, 0])") do
    y = prox(CheckLoss(0.5f0, fill(g, 2)), x, g); ref = prox(CheckLoss(0.5, fill(G, 2)), x, G)
    verdict(y == [0.0, 0.0], isapprox(y, ref; rtol = 1e-12, atol = 0)), "$(short(y)); widened factors $(short(ref))"
  end
end

# ---- LIM-041: AtomIterate coordinate and block steps round a wider argument (T3)
case("LIM-041", "P3-45", "prox_coordinate!(AtomIterate{Float32}, GroupNormL2, 1, [1 + 2^-28], 1 - 2^-28) stores 0 (T3: 2^-27)") do
  y = AtomIterate(Float32, 2, [1:1, 2:2])
  prox_coordinate!(y, GroupNormL2(1f0, [1:1, 2:2], ones(Float32, 2)), 1, [1.0 + 2.0^-28], 1.0 - 2.0^-28)
  v = Vector(y)
  verdict(v == Float32[0, 0], v == Float32[2.0^-27, 0]), "stored $(short(v)); T3 Float32[2^-27, 0]"
end

# ---- LIM-042: group value_and_prox! is not f(y) for a narrower argument and wider destination (O2, T3)
let x = Float32[1.1, 2.3, -0.7]
  case("LIM-042", "P3-05", "value_and_prox!(Float64 y, GroupNormL2(1, [1:3], [1]), Float32 x, 0.5): relative gap to f(y) is 1.45e-8 (same reproducer as P6-51)") do
    f = GroupNormL2(1.0, [1:3], [1.0]); y = zeros(3); v = value_and_prox!(y, f, x, 0.5)
    e = abs(v - f(y)) / f(y)
    verdict(_B_band(e, 1.45e-8), e <= 8 * eps()), "returned $(short(v)), f(y) $(short(f(y))), gap $(round(e; sigdigits = 3))"
  end
  case("LIM-042", "P6-51", "value_and_prox!(Float64 y, GroupMCP(1, [[1, 2, 3]]; a = 2), Float32 x, 0.5): relative gap to f(y) is 6.97e-9") do
    f = GroupMCP(1.0, [[1, 2, 3]]; a = 2.0); y = zeros(3); v = value_and_prox!(y, f, x, 0.5)
    e = abs(v - f(y)) / f(y)
    verdict(_B_band(e, 6.97e-9), e <= 8 * eps()), "returned $(short(v)), f(y) $(short(f(y))), gap $(round(e; sigdigits = 3))"
  end
end

# ---- LIM-043: group norms computed in the narrower argument type (T3)
case("LIM-043", "P3-06", "prox!(Float64 y, GroupNormL2(0.3, [1:3, 4:6], ones(2)), Float32 x, 0.5) differs from the call with Float64.(x) at Float32 level (register 5.4e-10)") do
  f = GroupNormL2(0.3, [1:3, 4:6], ones(2)); x = Float32[1.1, 2.3, -0.7, 0.3, -1.9, 2.2]
  a = prox!(zeros(6), f, x, 0.5); b = prox!(zeros(6), f, Float64.(x), 0.5)
  e = maximum(abs.(a .- b) ./ abs.(b))
  verdict(_B_f32(e), e <= 16 * eps()), "max relative difference $(round(e; sigdigits = 3)) (register: up to 5.4e-10 on audit data)"
end

# ---- LIM-044: Precompose around a value-less inner function returns the value in the image precision (O2, T3)
_B_en(z) = sum(t -> max(abs(t) - 1, 0)^2, z) / (2 * big(0.5))
case("LIM-044", "P3-04", "value_and_prox! of Precompose(Conjugate(ElasticNet(1f0, 0.5f0)), Float32[2, -0.5, 1.5]) at Float32[1, 2, 3], γ = 0.5: relative gap 2e-7 to the BigFloat value at y") do
  a = Float32[2, -0.5, 1.5]; y = zeros(3)
  v = value_and_prox!(y, Precompose(Conjugate(ElasticNet(1f0, 0.5f0)), a), Float32[1, 2, 3], 0.5)
  ref = _B_en(big.(a) .* big.(y)); e = _B_rel(v, ref)
  verdict(_B_band(e, 2e-7), e <= 16 * eps()), "v = $(short(v)), exact at y $(short(Float64(ref))), gap $(round(e; sigdigits = 3))"
end
case("LIM-044", "P3-04", "scalar scale: Precompose(Conjugate(ElasticNet(1f0, 0.5f0)), 2f0) at Float32[1, 2, 3.3], γ = 0.5: relative gap 1.35e-7") do
  y = zeros(3)
  v = value_and_prox!(y, Precompose(Conjugate(ElasticNet(1f0, 0.5f0)), 2f0), Float32[1, 2, 3.3], 0.5)
  ref = _B_en(2 .* big.(y)); e = _B_rel(v, ref)
  verdict(_B_band(e, 1.35e-7), e <= 16 * eps()), "v = $(short(v)), exact at y $(short(Float64(ref))), gap $(round(e; sigdigits = 3))"
end

# ---- LIM-045: Float16 is not rejected by role (T6, decision 7): fixed = ArgumentError before writing
_B_x32 = Float32[3, -0.5, 0.2, -2, 0, 1.5]; _B_x16 = Float16.(_B_x32)
_B_G = [1:2, 3:4, 5:6]
_B_fam = Pair{String, Any}[
  "NormL1" => NormL1(1.0), "SqrNormL2" => SqrNormL2(1.0), "NormL2" => NormL2(1.0), "NormLinf" => NormLinf(1.0),
  "IndBox" => IndBox(-1.0, 1.0), "IndBallL1" => IndBallL1(1.0), "IndSimplex" => IndSimplex(1.0),
  "GroupNormL2" => GroupNormL2(1.0, _B_G), "ElasticNet" => ElasticNet(1.0, 0.5), "MCP" => MCP(1.0, 2.0),
  "CheckLoss" => CheckLoss(0.3), "Constrained(NormL1, IndHyperplane)" => Constrained(NormL1(1.0), IndHyperplane(ones(6), 1.0)),
  "Constrained(NormL1, IndBallL2)" => Constrained(NormL1(1.0), IndBallL2(1.0)), "Conjugate(NormL1)" => Conjugate(NormL1(1.0)),
  "Translate(NormL1)" => Translate(NormL1(1.0), ones(6)), "Regularize(NormL1)" => Regularize(NormL1(1.0), 0.3),
  "Scaled(NormL1)" => Scaled(NormL1(1.0), 2.0), "SLOPE" => SLOPE(collect(6.0:-1:1)), "TotalVariation1D" => TotalVariation1D(1.0),
  "IndIsotonic" => IndIsotonic(), "IndSOC" => IndSOC(), "LinearFunction" => LinearFunction(ones(6)),
]
for (nm, f) in _B_fam
  case("LIM-045", "P5-85", "$nm: value at a Float16 argument is computed") do
    c, _ = attempt(() -> f(_B_x32)); c && error("control (Float32 argument) failed")
    t, r = attempt(() -> f(_B_x16))
    verdict(!t, t && r isa ArgumentError), show_outcome(t, r)
  end
  case("LIM-045", "P5-85", "$nm: prox!(Float32 destination, Float16 argument) is computed and written") do
    c, _ = attempt(() -> prox!(fill(7f0, 6), f, _B_x32, 0.3)); c && error("control (Float32 argument) failed")
    y = fill(7f0, 6); t, r, ch = writes(() -> prox!(y, f, _B_x16, 0.3), y)
    verdict(!t && ch, rejects(ArgumentError, t, r, ch)), "$(show_outcome(t, r)), destination written = $ch"
  end
  case("LIM-045", "P5-85", "$nm: prox!(Float16 destination, Float32 argument) is computed and written") do
    c, _ = attempt(() -> prox!(fill(7f0, 6), f, _B_x32, 0.3)); c && error("control (Float32 destination) failed")
    y = fill(Float16(7), 6); t, r, ch = writes(() -> prox!(y, f, _B_x32, 0.3), y)
    verdict(!t && ch, rejects(ArgumentError, t, r, ch)), "$(show_outcome(t, r)), destination written = $ch"
  end
end
case("LIM-045", "P5-85", "value_and_prox!(Float32 destination, NormL1, Float16 argument) is computed and written") do
  y = fill(7f0, 6); t, r, ch = writes(() -> value_and_prox!(y, NormL1(1.0), _B_x16, 0.3), y)
  verdict(!t && ch, rejects(ArgumentError, t, r, ch)), "$(show_outcome(t, r)), destination written = $ch"
end
case("LIM-045", "P5-85", "value_and_prox!(Float16 destination, NormL1, Float32 argument) is computed and written") do
  y = fill(Float16(7), 6); t, r, ch = writes(() -> value_and_prox!(y, NormL1(1.0), _B_x32, 0.3), y)
  verdict(!t && ch, rejects(ArgumentError, t, r, ch)), "$(show_outcome(t, r)), destination written = $ch"
end
case("LIM-045", "P5-85", "gradient!(Float16 destination, SqrNormL2, Float32 argument) is computed and written") do
  g = fill(Float16(7), 6); t, r, ch = writes(() -> gradient!(g, SqrNormL2(1.0), _B_x32), g)
  verdict(!t && ch, rejects(ArgumentError, t, r, ch)), "$(show_outcome(t, r)), destination written = $ch"
end
case("LIM-045", "P5-85", "gradient!(Float32 destination, SqrNormL2, Float16 argument) is computed and written") do
  g = fill(7f0, 6); t, r, ch = writes(() -> gradient!(g, SqrNormL2(1.0), _B_x16), g)
  verdict(!t && ch, rejects(ArgumentError, t, r, ch)), "$(show_outcome(t, r)), destination written = $ch"
end
case("LIM-045", "P5-85", "value_and_gradient!(Float16 destination, SqrNormL2, Float32 argument) is computed and written") do
  g = fill(Float16(7), 6); t, r, ch = writes(() -> value_and_gradient!(g, SqrNormL2(1.0), _B_x32), g)
  verdict(!t && ch, rejects(ArgumentError, t, r, ch)), "$(show_outcome(t, r)), destination written = $ch"
end
case("LIM-045", "P5-85", "hessian_vec!(Float32 destination, SqrNormL2, Float32 argument, Float16 direction) is computed and written") do
  g = fill(7f0, 6); t, r, ch = writes(() -> hessian_vec!(g, SqrNormL2(1.0), _B_x32, _B_x16), g)
  verdict(!t && ch, rejects(ArgumentError, t, r, ch)), "$(show_outcome(t, r)), destination written = $ch"
end
case("LIM-045", "P5-85", "hessian_vec!(Float16 destination, SqrNormL2, Float32 argument and direction) is computed and written") do
  g = fill(Float16(7), 6); t, r, ch = writes(() -> hessian_vec!(g, SqrNormL2(1.0), _B_x32, _B_x32), g)
  verdict(!t && ch, rejects(ArgumentError, t, r, ch)), "$(show_outcome(t, r)), destination written = $ch"
end
# data: rejected when the function is built or at its first use, before anything is written
function _B_data16(cl, what, mk, run)
  case("LIM-045", cl, what) do
    t0, f = attempt(mk)
    if t0
      verdict(false, f isa ArgumentError), "constructor throws $(nameof(typeof(f)))"
    else
      g = fill(7f0, 6); t, r, ch = writes(() -> run(f, g), g)
      verdict(!t && ch, rejects(ArgumentError, t, r, ch)), "constructed; $(show_outcome(t, r)), destination written = $ch"
    end
  end
end
_B_A = _B_mat(Float64, 8, 6, 2); _B_y = _B_vec(Float64, 8, 1)
_B_data16("P5-85", "LeastSquaresLoss with Float16 X: built and gradient! computed",
  () -> LeastSquaresLoss(_B_y, Float16.(_B_A)), (f, g) -> gradient!(g, f, _B_x32))
_B_data16("P5-85", "QuadraticFunction with Float16 matrix: built and gradient! computed",
  () -> QuadraticFunction(Float16.(_B_A' * _B_A)), (f, g) -> gradient!(g, f, _B_x32))
_B_data16("P5-85", "LogisticLoss with Float16 X: built and gradient! computed",
  () -> LogisticLoss(Float64.(_B_y .> 0), Float16.(_B_A)), (f, g) -> gradient!(g, f, _B_x32))

# ---- LIM-046: LAPACK functions do not check the argument's element type (A5, T6, T7)
let I2 = [1 0; 0 1], J2 = [2 1; 1 2]
  for (nm, A) in (("BigFloat", BigFloat.(I2)), ("Float16", Float16.(I2)), ("Int", I2))
    case("LIM-046", "P4-01", "prox!(Float64 output, IndPSD(), $nm argument) is accepted") do
      y = fill(7.0, 2, 2); t, r, ch = writes(() -> prox!(y, IndPSD(), A, 1.0), y)
      verdict(!t && ch, rejects(ArgumentError, t, r, ch)), "$(show_outcome(t, r)), destination written = $ch"
    end
  end
  for (nm, f) in (("NuclearNorm(0.25)", NuclearNorm(0.25)), ("Spectral(NormL1(0.25))", Spectral(NormL1(0.25))), ("SingularValue(NormL1(0.25))", SingularValue(NormL1(0.25))))
    case("LIM-046", "P4-01", "prox!(Float64 output, $nm, Float16 argument) is accepted") do
      c, _ = attempt(() -> prox!(fill(7.0, 2, 2), f, Float64.(J2), 0.5)); c && error("control (Float64 argument) failed")
      y = fill(7.0, 2, 2); t, r, ch = writes(() -> prox!(y, f, Float16.(J2), 0.5), y)
      verdict(!t && ch, rejects(ArgumentError, t, r, ch)), "$(show_outcome(t, r)), destination written = $ch"
    end
  end
  case("LIM-046", "P4-01", "LogDetLoss(I)(Float16 argument) value is computed") do
    t, r = attempt(() -> LogDetLoss([1.0 0; 0 1.0])(Float16.(J2)))
    verdict(!t, t && r isa ArgumentError), show_outcome(t, r)
  end
  for (nm, T) in (("Float16", Float16), ("BigFloat", BigFloat))
    case("LIM-046", "P4-01", "prox!($nm output, NuclearNorm(0), $nm argument) shortcut is accepted") do
      y = fill(T(7), 2, 2); t, r, ch = writes(() -> prox!(y, NuclearNorm(0.0), T.(J2), 1.0), y)
      verdict(!t && ch, rejects(ArgumentError, t, r, ch)), "$(show_outcome(t, r)), destination written = $ch"
    end
    case("LIM-046", "P4-01", "prox!($nm output, IndBallRank(2), $nm argument) full-rank shortcut is accepted") do
      y = fill(T(7), 2, 2); t, r, ch = writes(() -> prox!(y, IndBallRank(2), T.(J2), 1.0), y)
      verdict(!t && ch, rejects(ArgumentError, t, r, ch)), "$(show_outcome(t, r)), destination written = $ch"
    end
  end
end

# ================================================================ NaN, infinities, degenerate parameters, devices (LIM-047 to LIM-063)
# Group C: LIM-047 to LIM-063 (NaN, infinities and degenerate parameters E1-E6; devices G1-G6)

# ---- helpers (prefixed _C_)
_C_isnan(r) = r isa Number && isnan(r)
_C_allnan(y) = y isa AbstractArray && !isempty(y) && all(isnan, y)
_C_prox(f, x, γ = 1.0) = attempt(() -> prox(f, x, γ))
"value_and_prox! into a destination of element type T filled with 7: `(threw, value or exception, y)`"
function _C_vp(f, x, γ = 1.0, T = Float64)
  y = fill(T(7), size(x))
  t, r = attempt(() -> value_and_prox!(y, f, x, γ))
  (t, r, y)
end
"the call as the register describes the failure of an indicator: value 0 although the stored output has NaN"
_C_false_zero(t, v, y) = !t && v == 0 && any(isnan, y)
"the contract after a NaN argument: the stored output depends on the NaN, so the joint value is NaN"
_C_nan_value(t, v, y) = !t && any(isnan, y) && isnan(v)
"a copy of x as the zero function returns it"
_C_zerofn(g, x) = g(x) == 0 && isequal(prox(g, x, 0.7), x)
"device writes: run a call that may write the device array `yd`; `(threw, result or exception, changed)`"
function _C_dwrites(g, yd)
  before = Array(yd)
  t, r = attempt(g)
  (t, r, !isequal(before, Array(yd)))
end
"a device case runs only where JLArrays and Adapt load"
_C_devcase(f, lim, cl, what) = HAVE_JL ? case(f, lim, cl, what) :
  skip(lim, cl, what, "JLArrays/Adapt not loadable here; run with the JLArrays environment (Julia 1.13)")
_C_scalarindex(r) = r isa ErrorException && occursin("Scalar indexing", sprint(showerror, r))

# ---- LIM-047: NormL2/GroupNormL2 prox turn NaN into zeros
case("LIM-047", "P4-60", "NormL2 prox at [NaN, 4], γ = 1: [0, 0] (E2's own example: [NaN, NaN])") do
  t, y = _C_prox(NormL2(1.0), [NaN, 4.0])
  verdict(!t && y == [0.0, 0.0], !t && isequal(y, [NaN, NaN])), show_outcome(t, y)
end
case("LIM-047", "P4-60", "NormL2 value_and_prox! at [NaN, 4]: y = [0, 0], finite value (E2: y = [NaN, NaN], value NaN)") do
  t, v, y = _C_vp(NormL2(1.0), [NaN, 4.0])
  verdict(!t && y == [0.0, 0.0] && isfinite(v), _C_nan_value(t, v, y) && _C_allnan(y)), "$(show_outcome(t, v)), y = $(short(y))"
end
# the NaN group and an independent group: group 2 = [3, 4] has norm 5 and the default weight √2, so its output is [3, 4] (1 - √2/5)
_C_g2ref = [3.0, 4.0] .* (1 - sqrt(2) / 5)
case("LIM-047", "P4-60", "GroupNormL2 prox, groups 1:2 and 3:4, NaN in group 1: zeros in group 1 (E2: NaN there, group 2 = [3,4](1-√2/5))") do
  t, y = _C_prox(GroupNormL2(1.0, [1:2, 3:4]), [NaN, 4.0, 3.0, 4.0])
  ok2 = !t && isapprox(y[3:4], _C_g2ref; rtol = 16 * eps())
  verdict(!t && y[1:2] == [0.0, 0.0] && ok2, !t && isnan(y[1]) && isnan(y[2]) && ok2), show_outcome(t, y)
end
case("LIM-047", "P4-60", "GroupNormL2 value_and_prox!, NaN in group 1: zeros in group 1 and a finite value (E2: NaN group, value NaN)") do
  t, v, y = _C_vp(GroupNormL2(1.0, [1:2, 3:4]), [NaN, 4.0, 3.0, 4.0])
  ok2 = !t && isapprox(y[3:4], _C_g2ref; rtol = 16 * eps())
  verdict(!t && y[1:2] == [0.0, 0.0] && isfinite(v) && ok2, !t && isnan(y[1]) && isnan(y[2]) && isnan(v) && ok2), "$(show_outcome(t, v)), y = $(short(y))"
end
# named in the entry's Affected API (CPU and device); not in its Actual or Resolution criteria
case("LIM-047", "P4-60", "Constrained(GroupNormL2, IndBallL2) CPU prox at [NaN, 0]: [0, 0] (E2: the one group is NaN, so [NaN, NaN])") do
  t, y = _C_prox(Constrained(GroupNormL2(1.0, [1:2]), IndBallL2(1.0)), [NaN, 0.0])
  verdict(!t && y == [0.0, 0.0], !t && _C_allnan(y)), show_outcome(t, y)
end
_C_devcase("LIM-047", "P4-60", "device NormL2 prox at [NaN, 4] (JLArrays): [0, 0] (E2: [NaN, NaN])") do
  t, y = _C_prox(NormL2(1.0), JLArray([NaN, 4.0]))
  y = t ? y : Array(y)
  verdict(!t && y == [0.0, 0.0], !t && isequal(y, [NaN, NaN])), show_outcome(t, y)
end
_C_devcase("LIM-047", "P4-60", "device GroupNormL2 prox, NaN in group 1 (JLArrays): zeros in group 1 (E2: NaN there, group 2 finite)") do
  t, y = _C_prox(adapt(JLArray, GroupNormL2(1.0, [1:2, 3:4])), JLArray([NaN, 4.0, 3.0, 4.0]))
  y = t ? y : Array(y)
  ok2 = !t && isapprox(y[3:4], _C_g2ref; rtol = 16 * eps())
  verdict(!t && y[1:2] == [0.0, 0.0] && ok2, !t && isnan(y[1]) && isnan(y[2]) && ok2), show_outcome(t, y)
end
_C_devcase("LIM-047", "P4-60", "device Constrained(GroupNormL2, IndBallL2) prox at [NaN, 0] (JLArrays): [0, 0] (E2: [NaN, NaN])") do
  t, y = _C_prox(adapt(JLArray, Constrained(GroupNormL2(1.0, [1:2]), IndBallL2(1.0))), JLArray([NaN, 0.0]))
  y = t ? y : Array(y)
  verdict(!t && y == [0.0, 0.0], !t && _C_allnan(y)), show_outcome(t, y)
end

# ---- LIM-048a, LIM-048b, LIM-048c: proximal maps and values finite where they depend on a NaN (split of LIM-048)
# L0 subset (owner #560). Hard thresholding is separable, so only the NaN entry is NaN; IndBallL0's top-k ranking couples all entries.
case("LIM-048a", "P4-76", "NormL0 prox_entry at NaN: 0 (E2: NaN)") do
  t, r = attempt(() -> prox_entry(NormL0(1.0), NaN, 1, 1.0))
  verdict(!t && r == 0, !t && isnan(r)), show_outcome(t, r)
end
case("LIM-048a", "P4-76", "NormL0 prox at [NaN, 1.5], γ = 1: [0, 1.5] (E2: [NaN, 1.5], entry 2 is independent)") do
  t, y = _C_prox(NormL0(1.0), [NaN, 1.5])
  verdict(!t && y[1] == 0 && y[2] == 1.5, !t && isnan(y[1]) && y[2] == 1.5), show_outcome(t, y)
end
case("LIM-048a", "P4-76", "NormL0 value at [NaN, 1.5]: 2.0, NaN counted as nonzero (E2: NaN)") do
  t, v = attempt(() -> NormL0(1.0)([NaN, 1.5]))
  verdict(!t && v == 2.0, !t && isnan(v)), show_outcome(t, v)
end
case("LIM-048a", "P4-76", "IndBallL0(1) value at [NaN, 0]: 0, false feasibility (E2: NaN)") do
  t, v = attempt(() -> IndBallL0(1)([NaN, 0.0]))
  verdict(!t && v == 0, !t && isnan(v)), show_outcome(t, v)
end
# k = 1 keeps the largest entry, so each output depends on both entries (E2 quantifies over finite values of the others)
case("LIM-048a", "P4-76", "IndBallL0(1) prox at [NaN, 1.5]: [0, 1.5] (E2: [NaN, NaN], the ranking depends on the NaN)") do
  t, y = _C_prox(IndBallL0(1), [NaN, 1.5])
  verdict(!t && y == [0.0, 1.5], !t && _C_allnan(y)), show_outcome(t, y)
end
case("LIM-048a", "P4-76", "IndBallL0(1) value_and_prox! at [NaN, 1.5]: value 0, y = [0, 1.5] (E2: value NaN, y all NaN)") do
  t, v, y = _C_vp(IndBallL0(1), [NaN, 1.5])
  verdict(!t && v == 0 && y == [0.0, 1.5], !t && isnan(v) && _C_allnan(y)), "$(show_outcome(t, v)), y = $(short(y))"
end
# scalar penalties and losses subset (owner #561): separable, entry 2 is independent and keeps its finite value
for (nm, f) in (("LogPenalty(1, 1)", LogPenalty(1.0, 1.0)), ("CheckLoss(0.5)", CheckLoss(0.5)), ("HingeLoss([1, 1])", HingeLoss([1.0, 1.0])))
  case("LIM-048b", "P4-76", "$nm prox at [NaN, 1.5]: a finite entry 1 (E2: NaN there, entry 2 unchanged)") do
    ref = prox_entry(f, 1.5, 2, 1.0)
    t, y = _C_prox(f, [NaN, 1.5])
    verdict(!t && isfinite(y[1]) && y[2] == ref, !t && isnan(y[1]) && y[2] == ref), "$(show_outcome(t, y)), entry 2 reference $ref"
  end
end
case("LIM-048b", "P4-76", "MCP(1, 3) value at [NaN]: finite 1.5 (E2: NaN)") do
  t, v = attempt(() -> MCP(1.0, 3.0)([NaN]))
  verdict(!t && isfinite(v), !t && isnan(v)), show_outcome(t, v)
end
case("LIM-048b", "P4-76", "SCAD(1, 3.7) value at [NaN]: finite 2.35 (E2: NaN)") do
  t, v = attempt(() -> SCAD(1.0, 3.7)([NaN]))
  verdict(!t && isfinite(v), !t && isnan(v)), show_outcome(t, v)
end
# ordered, fused and isotonic subset (owner #562): every output depends on every entry (sorting rank, fused coupling, pooling), so all NaN
case("LIM-048c", "P4-77", "SLOPE([3,2,1]) prox at [1, NaN, 0.5]: finite entries 1 and 3 (E2: all NaN, the rank of every entry depends on the NaN)") do
  t, y = _C_prox(SLOPE([3.0, 2.0, 1.0]), [1.0, NaN, 0.5])
  verdict(!t && isnan(y[2]) && isfinite(y[1]) && isfinite(y[3]), !t && _C_allnan(y)), show_outcome(t, y)
end
case("LIM-048c", "P4-77", "NormL1Fused(1, 1) prox at [1, NaN, 0.5]: finite entries 1 and 3, entry 1 above the input (E2: all NaN, the total-variation term couples all entries)") do
  t, y = _C_prox(NormL1Fused(1.0, 1.0), [1.0, NaN, 0.5])
  verdict(!t && isnan(y[2]) && isfinite(y[1]) && isfinite(y[3]), !t && _C_allnan(y)), show_outcome(t, y)
end
case("LIM-048c", "P4-39", "IndIsotonic prox at [1, NaN, 0.5]: finite entries 1 > 3 violate the constraint (E2: all NaN, pooling couples all entries)") do
  t, y = _C_prox(IndIsotonic(), [1.0, NaN, 0.5])
  verdict(!t && isnan(y[2]) && isfinite(y[1]) && isfinite(y[3]) && y[1] > y[3], !t && _C_allnan(y)), show_outcome(t, y)
end

# ---- LIM-049: standalone values at a NaN argument are +Inf
_C_nanvals = (
  ("IndBox(-1, 1)", () -> IndBox(-1.0, 1.0), [NaN]),
  ("IndNonnegative", () -> IndNonnegative(), [NaN]),
  ("IndBallL2(1)", () -> IndBallL2(1.0), [NaN, 0.0]),
  ("IndBallL1(1)", () -> IndBallL1(1.0), [NaN, 0.0]),
  ("IndSimplex(1)", () -> IndSimplex(1.0), [NaN, 0.0]),
  ("IndBallLinf(1)", () -> IndBallLinf(1.0), [NaN]),
  ("IndBinary(0, 1)", () -> IndBinary(0.0, 1.0), [NaN]),
  ("IndSOC", () -> IndSOC(), [NaN, 0.0, 1.0]),
  ("IndHalfspace", () -> IndHalfspace([1.0, 1.0], 1.0), [NaN, 0.0]),
  ("IndHyperplane", () -> IndHyperplane([1.0, 1.0], 1.0), [NaN, 0.0]),
  ("IndHyperslab", () -> IndHyperslab([1.0, 1.0], -1.0, 1.0), [NaN, 0.0]),
  ("Conjugate(NormL1(1))", () -> Conjugate(NormL1(1.0)), [NaN]),
  ("Conjugate(SqrHingeLoss)", () -> Conjugate(SqrHingeLoss(ones(1))), [NaN]),
  ("NegEntropy", () -> NegEntropy(), [NaN]),
  ("NegLog", () -> NegLog(), [NaN]),
  ("NormL1Nonneg(1)", () -> NormL1Nonneg(1.0), [NaN]),
  ("IndIsotonic", () -> IndIsotonic(), [NaN, 0.0]),
  ("IndPSD", () -> IndPSD(), [NaN 0.0; 0.0 1.0]),
  ("NuclearNorm(1)", () -> NuclearNorm(1.0), [NaN 0.0; 0.0 1.0]),
  ("SingularValue(NormL1(1))", () -> SingularValue(NormL1(1.0)), [NaN 0.0; 0.0 1.0]),
  ("Spectral(NormL1(1))", () -> Spectral(NormL1(1.0)), [NaN 0.0; 0.0 1.0]),
  ("LogDetLoss(I)", () -> LogDetLoss(Matrix(1.0I, 2, 2)), [NaN 0.0; 0.0 1.0]),
)
for (nm, mk, x) in _C_nanvals
  case("LIM-049", "P4-27", "value of $nm at a NaN argument: +Inf (E2, decision 20: NaN)") do
    t, v = attempt(() -> mk()(x))
    verdict(!t && v == Inf, !t && isnan(v)), show_outcome(t, v)
  end
end

# ---- LIM-050: value_and_prox! of indicators returns 0 at a stored NaN
# NaN argument: the stored output depends on the NaN (E2), so the value is NaN. An Inf argument whose stored output is NaN:
# either the value is NaN, or the output is a finite limit (E3) with the ordinary value 0.
_C_vpnan = (
  ("IndBox(-1, 1)", () -> IndBox(-1.0, 1.0), [NaN, 0.0]),
  ("IndNonnegative", () -> IndNonnegative(), [NaN, 0.0]),
  ("IndBallLinf(1)", () -> IndBallLinf(1.0), [NaN, 0.0]),
  ("IndIsotonic", () -> IndIsotonic(), [NaN, 0.0]),
  ("Conjugate(NormL1(1))", () -> Conjugate(NormL1(1.0)), [NaN, 0.0]),
  ("Conjugate(NormL2(1))", () -> Conjugate(NormL2(1.0)), [NaN, 0.0]),
  ("Conjugate(NuclearNorm(1))", () -> Conjugate(NuclearNorm(1.0)), [NaN 0.0; 0.0 1.0]),
)
for (nm, mk, x) in _C_vpnan
  case("LIM-050", "P4-29", "$nm value_and_prox! at a NaN argument: value 0 with NaN in y (E2: value NaN)") do
    t, v, y = _C_vp(mk(), x)
    verdict(_C_false_zero(t, v, y), _C_nan_value(t, v, y)), "$(show_outcome(t, v)), y = $(short(y))"
  end
end
for (nm, mk, x) in (("Conjugate(NormL2(1))", () -> Conjugate(NormL2(1.0)), [Inf, 0.0]), ("Conjugate(NuclearNorm(1))", () -> Conjugate(NuclearNorm(1.0)), [Inf 0.0; 0.0 1.0]))
  case("LIM-050", "P4-29", "$nm value_and_prox! at an Inf entry: value 0 with NaN in y (contract: value NaN, or a finite limit y with value 0)") do
    t, v, y = _C_vp(mk(), x)
    verdict(_C_false_zero(t, v, y), _C_nan_value(t, v, y) || (!t && all(isfinite, y) && v == 0)), "$(show_outcome(t, v)), y = $(short(y))"
  end
end
case("LIM-050", "P4-29", "SeparableSum(IndBox, IndBox) value_and_prox! at ([NaN], [0]): value 0 with NaN in block 1 (E2: value NaN)") do
  s = SeparableSum(IndBox(-1.0, 1.0), IndBox(-1.0, 1.0)); y = ([7.0], [7.0])
  t, v = attempt(() -> value_and_prox!(y, s, ([NaN], [0.0]), 1.0))
  hasnan = any(isnan, y[1]) || any(isnan, y[2])
  verdict(!t && v == 0 && hasnan, !t && hasnan && isnan(v)), "$(show_outcome(t, v)), y = $(short(y))"
end
for (nm, f) in (("IndBallL1(1)", IndBallL1(1.0)), ("IndSimplex(1)", IndSimplex(1.0)))
  case("LIM-050", "P6-42", "$nm value_and_prox! with an array step at [NaN, 1, 1]: value 0 with NaN in y (E2: value NaN)") do
    t, v, y = _C_vp(f, [NaN, 1.0, 1.0], [1.0, 1.0, 1.0])
    verdict(_C_false_zero(t, v, y), _C_nan_value(t, v, y)), "$(show_outcome(t, v)), y = $(short(y))"
  end
end
for (nm, f) in (("Conjugate(SingularValue(IndBallL1(1)))", Conjugate(SingularValue(IndBallL1(1.0)))), ("Conjugate(SingularValue(ElasticNet(1, 1)))", Conjugate(SingularValue(ElasticNet(1.0, 1.0)))))
  case("LIM-050", "P4-29", "$nm value_and_prox! at a NaN matrix: ArgumentError 'not finite' (E2: a NaN block and value NaN, a prescribed NaN is a result)") do
    t, v, y = _C_vp(f, [NaN 0.0; 0.0 1.0])
    thrown = t && v isa ArgumentError && occursin("not finite", sprint(showerror, v))
    verdict(thrown, !t && isnan(v) && _C_allnan(y)), "$(show_outcome(t, v)), y = $(short(y))"
  end
end

# ---- LIM-051: Precompose around an indicator returns 0 at a stored NaN
case("LIM-051", "P4-30", "Precompose(IndBallL2(1), 2) Float64 -> Float32 at [NaN, 0]: value 0, y = [NaN, NaN] (E5, O2: value NaN)") do
  t, v, y = _C_vp(Precompose(IndBallL2(1.0), 2.0), [NaN, 0.0], 1.0, Float32)
  verdict(_C_false_zero(t, v, y), _C_nan_value(t, v, y)), "$(show_outcome(t, v)), y = $(short(y))"
end
case("LIM-051", "P4-30", "Precompose(IndBallL1(1), 2) Float32 -> Float64 at [NaN, 0]: value 0, y = [NaN, NaN] (E5, O2: value NaN)") do
  t, v, y = _C_vp(Precompose(IndBallL1(1.0), 2.0), Float32[NaN, 0.0])
  verdict(_C_false_zero(t, v, y), _C_nan_value(t, v, y)), "$(show_outcome(t, v)), y = $(short(y))"
end
case("LIM-051", "P4-30", "Precompose(IndBox(-1, 1), 2) at [NaN, 0]: value 0, y = [NaN, 0] (O2: value NaN)") do
  t, v, y = _C_vp(Precompose(IndBox(-1.0, 1.0), 2.0), [NaN, 0.0])
  verdict(_C_false_zero(t, v, y), _C_nan_value(t, v, y)), "$(show_outcome(t, v)), y = $(short(y))"
end
case("LIM-051", "P4-30", "Precompose(IndBox(-1, 1), array scale [1, 2]) at [NaN, 0]: value 0, y = [NaN, 0] (O2: value NaN)") do
  t, v, y = _C_vp(Precompose(IndBox(-1.0, 1.0), [1.0, 2.0]), [NaN, 0.0])
  verdict(_C_false_zero(t, v, y), _C_nan_value(t, v, y)), "$(show_outcome(t, v)), y = $(short(y))"
end
# the same call with a Float64 destination gives Inf today (the standalone value, LIM-049), NaN once LIM-049 is repaired:
# the Float32 destination must agree with it and must not be 0
case("LIM-051", "P6-43", "Precompose(IndPSD, 2) Float64 -> Float32 at [1 NaN; NaN 1]: value 0 (the Float64 destination gives the standalone class, not 0)") do
  x = [1.0 NaN; NaN 1.0]; f = Precompose(IndPSD(), 2.0)
  t, v, y = _C_vp(f, x, 1.0, Float32)
  t64, v64, _ = _C_vp(f, x)
  verdict(!t && v == 0, !t && !t64 && !isfinite(v) && isequal(v, v64)), "Float32 destination $(show_outcome(t, v)), Float64 destination $(show_outcome(t64, v64))"
end

# ---- LIM-052: coordinate fixed by an infinite weight is NaN at a NaN or infinite argument
# the fixed coordinate is 0 whatever x1 is; coordinate 2 (weight 1, argument 2) is independent and keeps its value
_C_infw = (
  ("NormL1", NormL1(1.0, [Inf, 1.0])),
  ("ElasticNet", ElasticNet(1.0, 1.0, [Inf, 1.0])),
  ("CappedL1", CappedL1(1.0, 1.0, [Inf, 1.0])),
  ("Berhu", Berhu(1.0, 1.0, [Inf, 1.0])),
)
for (nm, f) in (_C_infw..., ("LogPenalty", LogPenalty(1.0, 1.0, [Inf, 1.0])), ("NormLq", NormLq(1.0, 0.5, [Inf, 1.0])))
  case("LIM-052", "P4-20", "$nm with weights [Inf, 1], prox at [NaN, 2]: NaN in the fixed coordinate (E2, decision 48: 0)") do
    ref = prox_entry(f, 2.0, 2, 1.0)
    t, y = _C_prox(f, [NaN, 2.0])
    verdict(!t && isnan(y[1]) && y[2] == ref, !t && y[1] == 0 && y[2] == ref), "$(show_outcome(t, y)), entry 2 reference $ref"
  end
end
# EpsInsensitiveLoss is not a gap: an infinite weight confines its coordinate to the tube [-ε, ε]
for (nm, f) in _C_infw, (sg, v1) in (("+Inf", Inf), ("-Inf", -Inf))
  case("LIM-052", "P4-20", "$nm with weights [Inf, 1], prox at [$sg, 2]: NaN in the fixed coordinate (E2, decision 48: 0)") do
    ref = prox_entry(f, 2.0, 2, 1.0)
    t, y = _C_prox(f, [v1, 2.0])
    verdict(!t && isnan(y[1]) && y[2] == ref, !t && y[1] == 0 && y[2] == ref), "$(show_outcome(t, y)), entry 2 reference $ref"
  end
end

# ---- LIM-053: a zero level is not the zero function at non-finite arguments
case("LIM-053", "P4-26", "NuclearNorm(0) value at [NaN 0; 0 1]: Inf (E4, decision 22: 0)") do
  t, v = attempt(() -> NuclearNorm(0.0)([NaN 0.0; 0.0 1.0]))
  verdict(!t && v == Inf, !t && v == 0), show_outcome(t, v)
end
case("LIM-053", "P4-26", "NuclearNorm(0) value at [Inf 0; 0 1]: Inf (E4: the zero function, 0)") do
  t, v = attempt(() -> NuclearNorm(0.0)([Inf 0.0; 0.0 1.0]))
  verdict(!t && v == Inf, !t && v == 0), show_outcome(t, v)
end
case("LIM-053", "P4-26", "NuclearNorm(0) value_and_prox! at [NaN 0; 0 1]: value Inf, Y = X copied (E4: value 0, Y = X copied)") do
  X = [NaN 0.0; 0.0 1.0]; Y = zeros(2, 2)
  t, v = attempt(() -> value_and_prox!(Y, NuclearNorm(0.0), X, 1.0))
  verdict(!t && v == Inf && isequal(Y, X), !t && v == 0 && isequal(Y, X)), "$(show_outcome(t, v)), Y = $(short(Y))"
end
# a full-rank ball is the whole space; the register expects 0 (E4). E2/E3 would also allow reading it as an indicator (NaN or +Inf).
case("LIM-053", "P4-26", "IndBallRank(2) (full rank) value at [NaN 0; 0 1]: Inf (register, E4: 0)") do
  t, v = attempt(() -> IndBallRank(2)([NaN 0.0; 0.0 1.0]))
  verdict(!t && v == Inf, !t && v == 0), show_outcome(t, v)
end
case("LIM-053", "P4-26", "IndBallRank(2) (full rank) value_and_prox! at [NaN 0; 0 1]: value Inf, Y = X copied (register, E4: value 0, Y = X)") do
  X = [NaN 0.0; 0.0 1.0]; Y = zeros(2, 2)
  t, v = attempt(() -> value_and_prox!(Y, IndBallRank(2), X, 1.0))
  verdict(!t && v == Inf && isequal(Y, X), !t && v == 0 && isequal(Y, X)), "$(show_outcome(t, v)), Y = $(short(Y))"
end
case("LIM-053", "P4-62", "SqrNormL2(0) gradient! at [NaN, 4]: [NaN, 0] (E4: [0, 0])") do
  g = zeros(2)
  t, r = attempt(() -> gradient!(g, SqrNormL2(0.0), [NaN, 4.0]))
  verdict(!t && isnan(g[1]) && g[2] == 0, !t && all(iszero, g)), "$(show_outcome(t, r)), g = $(short(g))"
end
for (sg, v1) in (("+Inf", Inf), ("-Inf", -Inf))
  case("LIM-053", "P4-62", "SqrNormL2(0) gradient! at [$sg, 4]: [NaN, 0] (E4: [0, 0])") do
    g = zeros(2)
    t, r = attempt(() -> gradient!(g, SqrNormL2(0.0), [v1, 4.0]))
    verdict(!t && isnan(g[1]) && g[2] == 0, !t && all(iszero, g)), "$(show_outcome(t, r)), g = $(short(g))"
  end
end
case("LIM-053", "P4-62", "SqrNormL2(0) prox at [1, 2], γ = Inf: [NaN, NaN] (E4: the argument is copied, [1, 2])") do
  t, y = _C_prox(SqrNormL2(0.0), [1.0, 2.0], Inf)
  verdict(!t && _C_allnan(y), !t && y == [1.0, 2.0]), show_outcome(t, y)
end

# ---- LIM-054: MoreauEnvelope value at an infinite entry
case("LIM-054", "P4-28", "MoreauEnvelope(NormL1(1), 1) value at [Inf]: NaN (E3: +Inf, the Huber function)") do
  t, v = attempt(() -> MoreauEnvelope(NormL1(1.0), 1.0)([Inf]))
  verdict(!t && isnan(v), !t && v == Inf), show_outcome(t, v)
end
case("LIM-054", "P4-28", "MoreauEnvelope(NuclearNorm(1), 1) value at [Inf 0; 0 1]: NaN (E3: +Inf)") do
  t, v = attempt(() -> MoreauEnvelope(NuclearNorm(1.0), 1.0)([Inf 0.0; 0.0 1.0]))
  verdict(!t && isnan(v), !t && v == Inf), show_outcome(t, v)
end

# ---- LIM-055: decision 48 limits for the remaining constrained families and the NormLinf conjugates
# one infinite entry: the limit is [1, 0] (the unit radius on the infinite coordinate), all-NaN block today
_C_lim55 = (
  ("Constrained(IndBallL0(1), IndSphereL2(1))", Constrained(IndBallL0(1), IndSphereL2(1.0))),
  ("Constrained(IndBallL0(1), IndBallL2(1))", Constrained(IndBallL0(1), IndBallL2(1.0))),
  ("Constrained(ZeroFunction, IndBallL2(1))", Constrained(ZeroFunction(), IndBallL2(1.0))),
  ("Constrained(GroupNormL2(1, [1:2]), IndBallL2(1))", Constrained(GroupNormL2(1.0, [1:2]), IndBallL2(1.0))),
  ("Constrained(GroupNormL2(1, [1:2]), IndSphereL2(1))", Constrained(GroupNormL2(1.0, [1:2]), IndSphereL2(1.0))),
  ("Constrained(Scaled(NormL1(1), 2), IndBallL2(1))", Constrained(Scaled(NormL1(1.0), 2.0), IndBallL2(1.0))),
)
for (nm, f) in _C_lim55
  case("LIM-055", "P4-52", "$nm prox at [Inf, 0]: all-NaN block (E3, decision 48: the limit [1, 0])") do
    t, y = _C_prox(f, [Inf, 0.0])
    verdict(!t && _C_allnan(y), !t && isapprox(y, [1.0, 0.0]; atol = 8 * eps())), show_outcome(t, y)
  end
end
case("LIM-055", "P4-69", "Conjugate(NormLinf(1)) prox at [Inf, 0.5, 0.2]: [NaN, 0, 0] (decision 48: the ℓ1-ball limit [1, 0, 0])") do
  t, y = _C_prox(Conjugate(NormLinf(1.0)), [Inf, 0.5, 0.2])
  verdict(!t && isnan(y[1]) && y[2] == 0 && y[3] == 0, !t && isapprox(y, [1.0, 0.0, 0.0]; atol = 12 * eps())), show_outcome(t, y)
end
case("LIM-055", "P4-69", "Conjugate(GroupNormLinf(1, [1:2, 3:3])) prox at [Inf, 0.5, 0.2]: [NaN, 0, 0.2] (decision 48: limit [1, 0, 0.2])") do
  t, y = _C_prox(Conjugate(GroupNormLinf(1.0, [1:2, 3:3])), [Inf, 0.5, 0.2])
  verdict(!t && isnan(y[1]) && y[2] == 0 && y[3] == 0.2, !t && isapprox(y, [1.0, 0.0, 0.2]; atol = 12 * eps())), show_outcome(t, y)
end

# ---- LIM-056: gamma = Inf with a NaN argument
case("LIM-056", "P4-54", "NormL1(1) prox at [NaN, 1], γ = Inf: [NaN, 0] (E2, decision 48: the map is identically 0, [0, 0])") do
  t, y = _C_prox(NormL1(1.0), [NaN, 1.0], Inf)
  verdict(!t && isnan(y[1]) && y[2] == 0, !t && y == [0.0, 0.0]), show_outcome(t, y)
end

# ---- LIM-057: Scaled(indicator, Inf)
# #549 (nested Scaled factors) changed other Scaled paths; the infinite scale of an indicator is checked as stated
case("LIM-057", "P6-44", "Scaled(IndBallL2(1), Inf) value at [0.3, 0.4] (inside): NaN (O2, E4: 0)") do
  t, v = attempt(() -> Scaled(IndBallL2(1.0), Inf)([0.3, 0.4]))
  verdict(!t && isnan(v), !t && v == 0), show_outcome(t, v)
end
case("LIM-057", "P6-44", "Scaled(IndBallL2(1), Inf) value_and_prox! at [3, 4]: value NaN, y = [0.6, 0.8] correct (joint value 0)") do
  t, v, y = _C_vp(Scaled(IndBallL2(1.0), Inf), [3.0, 4.0])
  ok = !t && isapprox(y, [0.6, 0.8]; rtol = 8 * eps())
  verdict(ok && isnan(v), ok && v == 0), "$(show_outcome(t, v)), y = $(short(y))"
end
case("LIM-057", "P6-44", "Scaled(IndBallL2(1), Inf) value_and_prox! at [0.3, 0.4] (inside): value NaN, y = x (joint value 0)") do
  t, v, y = _C_vp(Scaled(IndBallL2(1.0), Inf), [0.3, 0.4])
  ok = !t && y == [0.3, 0.4]
  verdict(ok && isnan(v), ok && v == 0), "$(show_outcome(t, v)), y = $(short(y))"
end

# ---- LIM-058: Scaled(f, 0) rejected for total-domain functions
# a total-domain function becomes the zero function (value 0 and the prox copies the argument)
_C_lim58 = (
  ("HingeLoss", () -> HingeLoss(ones(3)), ones(3)),
  ("SqrHingeLoss", () -> SqrHingeLoss(ones(3)), ones(3)),
  ("CheckLoss", () -> CheckLoss(0.3), ones(3)),
  ("EpsInsensitiveLoss", () -> EpsInsensitiveLoss(0.1), ones(3)),
  ("LogisticLoss", () -> LogisticLoss([1.0, 0.0, 1.0, 0.0, 1.0], reshape(0.1 .* (1:15), 5, 3)), ones(3)),
  ("TotalVariation1D", () -> TotalVariation1D(1.0), ones(3)),
  ("FusedLasso", () -> FusedLasso(1.0, 1.0), ones(3)),
  ("SLOPE", () -> SLOPE([3.0, 2.0, 1.0]), ones(3)),
  ("NormL1Fused", () -> NormL1Fused(1.0, 1.0), ones(3)),
  ("NuclearNorm", () -> NuclearNorm(1.0), [1.0 0.0; 0.0 2.0]),
  ("MoreauEnvelope(NormL1)", () -> MoreauEnvelope(NormL1(1.0), 1.0), ones(3)),
  ("Translate(NormL1)", () -> Translate(NormL1(1.0), [1.0, 2.0, 3.0]), ones(3)),
)
for (nm, mk, x) in _C_lim58
  case("LIM-058", "P5-52", "Scaled($nm, 0): ArgumentError 'no supported domain-preserving representation' (E4: the zero function)") do
    t, g = attempt(() -> Scaled(mk(), 0.0))
    thrown = t && g isa ArgumentError && occursin("domain-preserving", sprint(showerror, g))
    verdict(thrown, !t && _C_zerofn(g, x)), show_outcome(t, g)
  end
end
# these two have the domain x >= 0 (value Inf at -1), unlike the register's 'total-domain' wording: E4 keeps the domain, an indicator of x >= 0
for (nm, mk) in (("KLDivergence", () -> KLDivergence(ones(3))), ("NegEntropy", () -> NegEntropy()))
  case("LIM-058", "P5-52", "Scaled($nm, 0) (domain x >= 0): ArgumentError 'no supported domain-preserving representation' (E4: keeps the domain, value Inf at a negative entry, 0 on it)") do
    t, g = attempt(() -> Scaled(mk(), 0.0))
    thrown = t && g isa ArgumentError && occursin("domain-preserving", sprint(showerror, g))
    verdict(thrown, !t && g([-1.0, 1.0, 1.0]) == Inf && g([0.5, 1.0, 1.0]) == 0), show_outcome(t, g)
  end
end
# Scaled(BernoulliLoss, 0) is not a gap: its domain is open at 0 (E4 allows the rejection, LIM-087)

# ---- LIM-059: fused spectral values disagree with the stored output
case("LIM-059", "P4-64", "SingularValue(NormL0(1)) value_and_prox! at Diagonal([2^60, 2, 0]): value 2, f(Y) = 1 (A5, O2: they agree)") do
  Y = zeros(3, 3); f = SingularValue(NormL0(1.0))
  t, v = attempt(() -> value_and_prox!(Y, f, Matrix(Diagonal([2.0^60, 2.0, 0.0])), 1.0))
  fy = f(Y)
  verdict(!t && v == 2 && fy == 1, !t && isequal(v, fy)), "$(show_outcome(t, v)), f(Y) = $fy"
end
case("LIM-059", "P4-64", "Spectral(NormL0(1)) value_and_prox! at Diagonal([2^60, 2, 0]): value 2, f(Y) = 1 (A5, O2: they agree)") do
  Y = zeros(3, 3); f = Spectral(NormL0(1.0))
  t, v = attempt(() -> value_and_prox!(Y, f, Matrix(Diagonal([2.0^60, 2.0, 0.0])), 1.0))
  fy = f(Y)
  verdict(!t && v == 2 && fy == 1, !t && isequal(v, fy)), "$(show_outcome(t, v)), f(Y) = $fy"
end
case("LIM-059", "P4-65", "IndBallRank(1) value_and_prox!, Float32 Y, rank-one X = nextfloat(0f0)[1 5/8; 5/8 25/64]: value 0, f(Y) = Inf (A5, O2: they agree)") do
  h = nextfloat(0f0); X = Float64(h) .* [1 5/8; 5/8 25/64]; Y = zeros(Float32, 2, 2); f = IndBallRank(1)
  t, v = attempt(() -> value_and_prox!(Y, f, X, 1.0))
  fy = f(Y)
  verdict(!t && v == 0 && fy == Inf, !t && isequal(v, fy)), "$(show_outcome(t, v)), f(Y) = $fy, Y = $(short(Y))"
end

# ---- LIM-060: device ball/simplex kernels do not classify NaN/Inf and accept bad steps (JLArrays evidence only)
_C_ball = (("IndBallL1(1)", () -> IndBallL1(1.0)), ("IndSimplex(1)", () -> IndSimplex(1.0)))
for (nm, mk) in _C_ball
  _C_devcase("LIM-060", "P4-78", "device $nm prox at [NaN, 0.5, 0.2]: partly NaN (G1, E2: all NaN as on the CPU)") do
    f = mk(); x = [NaN, 0.5, 0.2]
    t, yd = _C_prox(f, JLArray(x)); yd = t ? yd : Array(yd); cpu = prox(f, x)
    verdict(!t && isnan(yd[1]) && !all(isnan, yd), !t && isequal(yd, cpu) && all(isnan, yd)), "device $(show_outcome(t, yd)), CPU $(short(cpu))"
  end
  _C_devcase("LIM-060", "P5-99", "device $nm prox at [Inf, 0.5, 0.2]: [NaN, 0, 0] (G1, E3: the CPU limit [1, 0, 0])") do
    f = mk(); x = [Inf, 0.5, 0.2]
    t, yd = _C_prox(f, JLArray(x)); yd = t ? yd : Array(yd); cpu = prox(f, x)
    verdict(!t && isnan(yd[1]) && yd[2] == 0 && yd[3] == 0, !t && isequal(yd, cpu) && yd == [1.0, 0.0, 0.0]), "device $(show_outcome(t, yd)), CPU $(short(cpu))"
  end
  _C_devcase("LIM-060", "P5-99", "device $nm prox at [Inf, Inf, 0.2]: [NaN, NaN, 0] (G1, E3: all NaN as on the CPU)") do
    f = mk(); x = [Inf, Inf, 0.2]
    t, yd = _C_prox(f, JLArray(x)); yd = t ? yd : Array(yd); cpu = prox(f, x)
    verdict(!t && isnan(yd[1]) && isnan(yd[2]) && yd[3] == 0, !t && isequal(yd, cpu) && all(isnan, yd)), "device $(show_outcome(t, yd)), CPU $(short(cpu))"
  end
  _C_devcase("LIM-060", "P5-99", "device $nm value_and_prox! at [NaN, 0.5, 0.2]: value 0 (G1, O2: NaN as on the CPU)") do
    f = mk(); x = [NaN, 0.5, 0.2]; yd = JLArray(zeros(3)); ycpu = zeros(3)
    t, v = attempt(() -> value_and_prox!(yd, f, JLArray(x), 1.0)); vcpu = value_and_prox!(ycpu, f, x, 1.0)
    verdict(!t && v == 0 && isnan(vcpu), !t && isnan(v) && isnan(vcpu)), "device $(show_outcome(t, v)), CPU value $vcpu"
  end
  # device value_and_prox! rejects these steps before writing, as on the CPU; prox! does not
  for (sn, γ) in (("γ = -0.5", -0.5), ("γ = NaN", NaN)), (on, op) in (("prox!", prox!),)
    _C_devcase("LIM-060", "P5-99", "device $nm $on with $sn: accepted and written (P1: ArgumentError before writing, as on the CPU)") do
      yd = JLArray(fill(7.0, 3))
      t, r, ch = _C_dwrites(() -> op(yd, mk(), JLArray([0.3, 0.5, 0.2]), γ), yd)
      verdict(!t && ch, rejects(ArgumentError, t, r, ch)), "$(show_outcome(t, r)), y written = $ch, y = $(short(Array(yd)))"
    end
  end
end
for (tag, x) in (("Inf", [Inf, 0.5, 0.2]), ("NaN", [NaN, 0.5, 0.2]))
  _C_devcase("LIM-060", "P5-102", "device Constrained(NormL1, IndBallL2) prox at [$tag, 0.5, 0.2]: differs from the CPU (G1: equals the CPU, or G6: raises)") do
    f = Constrained(NormL1(1.0), IndBallL2(1.0)); cpu = prox(f, x)
    t, yd = _C_prox(adapt(JLArray, f), JLArray(x)); yd = t ? yd : Array(yd)
    verdict(!t && !isequal(yd, cpu), t || isequal(yd, cpu)), "device $(show_outcome(t, yd)), CPU $(short(cpu))"
  end
end

# ---- LIM-061: device Conjugate(NormLinf/GroupNormLinf) cancels through Moreau's identity
# the CPU result is the direct ℓ1-ball projection; the register's tolerance is 4n eps r (here r = 1, n = 3)
_C_devcase("LIM-061", "P5-86", "device Conjugate(NormLinf()) prox at [1e300, 0.25, -0.5]: [0, 0, 0] (G1, R2: the CPU [1, 0, -0])") do
  x = [1e300, 0.25, -0.5]; cpu = prox(Conjugate(NormLinf()), x)
  t, yd = _C_prox(Conjugate(NormLinf()), JLArray(x)); yd = t ? yd : Array(yd)
  verdict(!t && all(iszero, yd), !t && all(abs.(yd .- cpu) .<= 4 * 3 * eps())), "device $(show_outcome(t, yd)), CPU $(short(cpu))"
end
_C_devcase("LIM-061", "P3-11", "device Conjugate(GroupNormLinf(1, [1:2, 3:3])) prox at [1e300, 0.25, -0.5]: [0, 0, -0.5] (G1, R2: the CPU [1, 0, -0.5])") do
  x = [1e300, 0.25, -0.5]; g = GroupNormLinf(1.0, [1:2, 3:3]); cpu = prox(Conjugate(g), x)
  t, yd = _C_prox(Conjugate(adapt(JLArray, g)), JLArray(x)); yd = t ? yd : Array(yd)
  verdict(!t && yd[1] == 0 && yd[2] == 0 && yd[3] == -0.5, !t && all(abs.(yd .- cpu) .<= 4 * 3 * eps())), "device $(show_outcome(t, yd)), CPU $(short(cpu))"
end

# ---- LIM-062: device kernels narrow a host level before the weight
# ElasticNet(1e40, 1, w = 1e-30): f = 1e10 |x| + x²/2, prox_f(2e10) = shrink(2e10, 1e10)/2 = 5e9, so the conjugate prox is 2e10 - 5e9 = 1.5e10
_C_devcase("LIM-062", "P5-79", "device Conjugate(ElasticNet(1e40, 1, Float32[1e-30])) prox at Float32[2e10]: 2e10 (G5, decision 34: 1.5e10 as on the CPU)") do
  f = ElasticNet(1e40, 1.0, Float32[1e-30]); x = Float32[2e10]
  cpu = prox(Conjugate(f), x, 1f0)
  t, yd = _C_prox(Conjugate(adapt(JLArray, f)), JLArray(x), 1f0); yd = t ? yd : Array(yd)
  ref = big(x[1]) - max(big(x[1]) - big(1e10), 0) / 2
  verdict(!t && yd == Float32[2f10], !t && isapprox(big(yd[1]), ref; rtol = 4 * eps(Float32))), "device $(show_outcome(t, yd)), CPU $(short(cpu)), exact $(Float64(ref))"
end
_C_devcase("LIM-062", "P5-79", "device Conjugate(ElasticNet(big\"1e400\", 1, [1e-300])) prox at [2e100] (Float64): 2e100 (G5, decision 34: 1.5e100 as on the CPU)") do
  f = ElasticNet(big"1e400", 1.0, [1e-300]); x = [2e100]
  cpu = prox(Conjugate(f), x, 1.0)
  t, yd = _C_prox(Conjugate(adapt(JLArray, f)), JLArray(x), 1.0); yd = t ? yd : Array(yd)
  verdict(!t && yd == [2e100], !t && isapprox(yd[1], 1.5e100; rtol = 8 * eps())), "device $(show_outcome(t, yd)), CPU $(short(cpu))"
end

# ---- LIM-063: wrappers index scalars on device arrays (allowscalar(false))
# present: the scalar-indexing ErrorException, nothing written; fixed: the CPU result (rtol 1e-12: reductions are not bit-identical, G2)
_C_x6 = [3.0, -0.5, 0.2, -2.0, 0.0, 1.5]
_C_wrappers = (
  ("MoreauEnvelope(NormL1(1), 0.5)", () -> MoreauEnvelope(NormL1(1.0), 0.5)),
  ("SqrDistL2(IndBox(-1, 1))", () -> SqrDistL2(IndBox(-1.0, 1.0))),
  ("DistL2(IndBox(-1, 1))", () -> DistL2(IndBox(-1.0, 1.0))),
  ("Precompose(NormL1(1), 2.0)", () -> Precompose(NormL1(1.0), 2.0)),
  ("Precompose(NormL1(1), array scale)", () -> Precompose(NormL1(1.0), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
)
for (nm, mk) in _C_wrappers
  _C_devcase("LIM-063", "P5-60", "device $nm value: scalar-indexing ErrorException (G1: the CPU value)") do
    f = mk(); cpu = f(_C_x6)
    t, v = attempt(() -> adapt(JLArray, f)(JLArray(_C_x6)))
    verdict(t && _C_scalarindex(v), !t && isapprox(v, cpu; rtol = 1e-12)), "$(show_outcome(t, v)), CPU $cpu"
  end
  _C_devcase("LIM-063", "P5-60", "device $nm prox!: scalar-indexing ErrorException, y unchanged (G1: the CPU prox)") do
    f = mk(); cpu = prox(f, _C_x6, 0.3); yd = JLArray(fill(7.0, 6))
    t, r, ch = _C_dwrites(() -> prox!(yd, adapt(JLArray, f), JLArray(_C_x6), 0.3), yd)
    verdict(t && _C_scalarindex(r) && !ch, !t && isapprox(Array(yd), cpu; rtol = 1e-12)), "$(show_outcome(t, r)), y written = $ch, CPU $(short(cpu))"
  end
  _C_devcase("LIM-063", "P5-60", "device $nm value_and_prox!: scalar-indexing ErrorException, y unchanged (G1: the CPU value and prox)") do
    f = mk(); ycpu = zeros(6); vcpu = value_and_prox!(ycpu, f, _C_x6, 0.3); yd = JLArray(fill(7.0, 6))
    t, r, ch = _C_dwrites(() -> value_and_prox!(yd, adapt(JLArray, f), JLArray(_C_x6), 0.3), yd)
    verdict(t && _C_scalarindex(r) && !ch, !t && isapprox(r, vcpu; rtol = 1e-12) && isapprox(Array(yd), ycpu; rtol = 1e-12)), "$(show_outcome(t, r)), y written = $ch, CPU value $vcpu"
  end
end
_C_devcase("LIM-063", "P5-60", "device Translate(NormL1, a) value: scalar-indexing ErrorException (G1: the CPU value)") do
  f = Translate(NormL1(1.0), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); cpu = f(_C_x6)
  t, v = attempt(() -> adapt(JLArray, f)(JLArray(_C_x6)))
  verdict(t && _C_scalarindex(v), !t && isapprox(v, cpu; rtol = 1e-12)), "$(show_outcome(t, v)), CPU $cpu"
end

# ================================================================ wrappers, conjugates and iterates (LIM-064 to LIM-080)
# Group D: LIM-064 .. LIM-080 (wrappers, conjugates and other oracles; iterates)

# ---- helpers -----------------------------------------------------------------------
_D_msg(r) = sprint(showerror, r)
# the Moreau-route stopgap: ArgumentError "the proximal point of the conjugate is not finite ..."
_D_fin(r) = r isa ArgumentError && occursin("not finite at a finite input", _D_msg(r))
# O6.3 + R1-R3: the representable result (a few eps, relative, and absolute at the scale of x where
# x - γ·proj(x/γ) cancels) or PrecisionError before writing
function _D_moreau(f, x, γ, want)
  y = fill!(similar(x), 7)
  t, r, ch = writes(() -> prox!(y, f, x, γ), y)
  tol = 8 * eps(float(real(eltype(y))))
  present = t && _D_fin(r)
  fixed = rejects(PB.PrecisionError, t, r, ch) || (!t && isapprox(y, want; rtol = tol, atol = tol * maximum(abs, x)))
  verdict(present, fixed), "$(show_outcome(t, r)); y = $(short(y)), exact $(short(want))"
end
# the same for a spectral output, in the normwise scale of A5
function _D_moreau_spectral(f, x, γ, want)
  y = fill!(similar(x), 7)
  t, r, ch = writes(() -> prox!(y, f, x, γ), y)
  present = t && _D_fin(r)
  fixed = rejects(PB.PrecisionError, t, r, ch) ||
          (!t && maximum(abs, y .- want) <= 100 * eps() * maximum(abs, want))
  verdict(present, fixed), "$(show_outcome(t, r)); y = $(short(y)), exact $(short(want))"
end

# a delegating wrapper whose allocating gradient returns Float64 leaves (LIM-080)
struct _D_BadGrad{F} <: PB.AbstractFunction
  f::F
end
(b::_D_BadGrad)(x) = b.f(x)
PB.gradient!(g, b::_D_BadGrad, x) = PB.gradient!(g, b.f, x)
PB.value_and_gradient!(g, b::_D_BadGrad, x) = PB.value_and_gradient!(g, b.f, x)
PB.gradient(b::_D_BadGrad, x::Tuple) = map(a -> Float64.(a), PB.gradient(b.f, x))
PB.value_and_gradient(b::_D_BadGrad, x::Tuple) = (b.f(x), PB.gradient(b, x))

# a separable function with a prescribed KKT violation, to count the checks of test_kkt (LIM-068)
struct _D_Kkt <: PB.AbstractFunction
  v::Float64
end
PB.isproximable(::Type{_D_Kkt}) = true
PB.isseparable(::Type{_D_Kkt}) = true
PB.prox!(y::AbstractArray, ::_D_Kkt, x::AbstractArray, γ::Real) = (y .= 1; y)    # z is nonzero: a positive violation passes
PB.kkt_violation(f::_D_Kkt, g::AbstractArray, i::Integer) = f.v
function _D_kkt_checks(T, v)
  ts = redirect_stdout(devnull) do       # the test set prints its summary
    @testset "kkt count" begin
      PB.Testing.test_kkt(_D_Kkt(v), zeros(T, 3), 0.5; trials = 1)
    end
  end
  ts.n_passed
end

# the symmetric matrix of the stored lower triangle, to compare with a dense one
_D_ssi() = (x = SymmetricSparseIterate(Float64, 3); x[2, 1] = 1.0; x)

# ---- LIM-064: Constrained(f, IndBox) value_and_prox! throws for an inner function without a value
case("LIM-064", "P6-54", "value_and_prox!(Constrained(Conjugate(ElasticNet(1,1)), IndBox(-1,1))) at [0.5, 2], γ = 1: raises, y unchanged") do
  mk(b) = Constrained(Conjugate(ElasticNet(1.0, 1.0)), IndBox(-b, b))
  y = [9.0, 9.0]
  t, r, ch = writes(() -> value_and_prox!(y, mk(1.0), [0.5, 2.0], 1.0), y)
  present = t && r isa ArgumentError && occursin("not implemented", _D_msg(r)) && !ch
  # exact: y = clamp([0.5, 1.5]) = [0.5, 1.0], value f*(y) = Σ max(|y|-1, 0)²/2 = 0; box 2: y = [0.5, 1.5], value 0.125
  y2 = [9.0, 9.0]
  t2, r2 = attempt(() -> value_and_prox!(y2, mk(2.0), [0.5, 2.0], 1.0))
  fixed = !t && r == 0.0 && y == [0.5, 1.0] && !t2 && r2 ≈ 0.125 && y2 ≈ [0.5, 1.5]
  verdict(present, fixed), "$(show_outcome(t, r)), y unchanged = $(!ch); box 2: $(show_outcome(t2, r2)) (exact 0 and 0.125)"
end

# ---- LIM-065: parameter-array shape rules differ across wrappers
x2 = [1.0 2.0; 3.0 4.0]
case("LIM-065", "P6-56", "Tilt(NormL1, vec4)(2x2) accepts (39.0) while Translate(NormL1, vec4)(2x2) raises DimensionMismatch") do
  t1, r1 = attempt(() -> Tilt(NormL1(1.0), [1.0, 2, 3, 4])(x2))
  t2, r2 = attempt(() -> Translate(NormL1(1.0), [1.0, 2, 3, 4])(x2))
  present = !t1 && r1 == 39.0 && t2 && r2 isa DimensionMismatch
  # one rule: both accept (Tilt 39.0; Translate in linear order ‖x - b‖₁ = 2.0) or both raise DimensionMismatch
  fixed = (!t1 && !t2 && r1 == 39.0 && r2 == 2.0) || (t1 && t2 && r1 isa DimensionMismatch && r2 isa DimensionMismatch)
  verdict(present, fixed), "Tilt: $(show_outcome(t1, r1)); Translate: $(show_outcome(t2, r2))"
end
case("LIM-065", "P6-56", "LinearFunction(vec4)(2x2) accepts (29.0) while Translate(NormL1, vec4)(2x2) raises DimensionMismatch") do
  t1, r1 = attempt(() -> LinearFunction([1.0, 2, 3, 4])(x2))
  t2, r2 = attempt(() -> Translate(NormL1(1.0), [1.0, 2, 3, 4])(x2))
  present = !t1 && r1 == 29.0 && t2 && r2 isa DimensionMismatch
  fixed = (!t1 && !t2 && r1 == 29.0 && r2 == 2.0) || (t1 && t2 && r1 isa DimensionMismatch && r2 isa DimensionMismatch)
  verdict(present, fixed), "LinearFunction: $(show_outcome(t1, r1)); Translate: $(show_outcome(t2, r2))"
end
case("LIM-065", "P6-56", "SqrNormL2(1, ones(1,2)) on a length-2 vector: value raises DimensionMismatch, prox_entry accepts (1.5)") do
  f = SqrNormL2(1.0, ones(1, 2))
  t1, r1 = attempt(() -> f([1.0, 2.0]))
  t2, r2 = attempt(() -> prox_entry(f, 2.0, 2, 1.0))
  t3, r3 = attempt(() -> prox!(zeros(2), f, [1.0, 2.0], 1.0))
  present = t1 && r1 isa DimensionMismatch && !t2 && r2 == 1.5      # exact: (2 + 1·1)/(1 + 1)
  fixed = (t1 && t2 && r1 isa DimensionMismatch && r2 isa DimensionMismatch) || (!t1 && !t2)
  verdict(present, fixed), "value: $(show_outcome(t1, r1)); prox_entry: $(show_outcome(t2, r2)); prox!: $(show_outcome(t3, r3))"
end

# ---- LIM-066: restrict(IndBinary) requires every bound pair to contain zero
case("LIM-066", "P5-68", "restrict(IndBinary([0,1,0],[1,2,1]), [1,2]): raises 'zero-preserving' although the dropped pair {0,1} contains 0") do
  t, r = attempt(() -> restrict(IndBinary([0.0, 1.0, 0.0], [1.0, 2.0, 1.0]), [1, 2]))
  present = t && r isa ArgumentError && occursin("zero-preserving", _D_msg(r))
  # the restriction is IndBinary([0,1],[1,2]): 0 at a member, Inf otherwise
  fixed = !t && attempt(() -> (r([0.0, 1.0]), r([0.5, 1.0])))[2] == (0.0, Inf)
  verdict(present, fixed), show_outcome(t, r)
end
case("LIM-066", "P6-64", "restrict(Translate(IndBinary(-1,1), [1,1,0]), [3]): raises 'zero-preserving' although the dropped terms are finite at -b = -1") do
  t, r = attempt(() -> restrict(Translate(IndBinary(-1.0, 1.0), [1.0, 1.0, 0.0]), [3]))
  present = t && r isa ArgumentError && occursin("zero-preserving", _D_msg(r))
  # kept coordinate 3 has b = 0: the restriction is 0 at x = [±1] and Inf at [0]
  fixed = !t && attempt(() -> (r([1.0]), r([-1.0]), r([0.0])))[2] == (0.0, 0.0, Inf)
  verdict(present, fixed), show_outcome(t, r)
end

# ---- LIM-067: Conjugate(SingularValue) into a SymmetricSparseIterate
_D_E = eigen(Symmetric([2.0 0.5; 0.5 1.0]))
_D_Pref = _D_E.vectors * Diagonal(min.(_D_E.values, 1.0)) * _D_E.vectors'     # σ clipped at 1: the projection onto the dual ball
for (name, g) in (("Conjugate(SingularValue(NormL1(1)))", Conjugate(SingularValue(NormL1(1.0)))),
                  ("Conjugate(NuclearNorm(1))", Conjugate(NuclearNorm(1.0))))
  case("LIM-067", "P4-47", "prox!(SymmetricSparseIterate(2), $name, [2 .5; .5 1], 1): raises 'must be symmetric' for an exactly symmetric argument") do
    y = SymmetricSparseIterate(2)
    t, r = attempt(() -> prox!(y, g, [2.0 0.5; 0.5 1.0], 1.0))
    present = t && r isa ArgumentError && occursin("must be symmetric", _D_msg(r))
    fixed = !t && maximum(abs, Matrix(y) .- _D_Pref) <= 1e-12          # A5 absolute scale, ‖X‖ = 2.2
    verdict(present, fixed), t ? shorterr(r) : "y = $(short(Matrix(y))), exact $(short(_D_Pref))"
  end
end

# ---- LIM-068: fixed default tolerances independent of the precision
case("LIM-068", "P2-08", "IndIntersection default tol is the Float64 literal 1e-10 for Float32 and Float64 sets alike") do
  f32 = IndIntersection(IndBox(0f0, 1f0), IndBallL2(1f0))
  f64 = IndIntersection(IndBox(0.0, 1.0), IndBallL2(1.0))
  present = f32.tol === 1.0e-10 && f64.tol === 1.0e-10
  fixed = hasproperty(f32, :tol) && hasproperty(f64, :tol) && f32.tol !== f64.tol      # A4: derived from the precision
  verdict(present, fixed), "Float32 set tol = $(f32.tol) ($(typeof(f32.tol))), Float64 set tol = $(f64.tol)"
end
case("LIM-068", "P7-03", "_largest_eigenvalue default rtol equals the literal 1e-10 for Float32 and Float64 (same call counts as rtol = 1e-10)") do
  function calls(T; kw...)
    n = 60; d = T[1 + i / n for i in 1:n]; c = Ref(0)
    PB._largest_eigenvalue(n, T; kw...) do w, v
      c[] += 1; w .= d .* v
    end
    c[]
  end
  d32, e32, d64, e64 = calls(Float32), calls(Float32; rtol = 1e-10), calls(Float64), calls(Float64; rtol = 1e-10)
  sees = calls(Float64; rtol = 1e-5) != e64          # the probe can tell tolerances apart
  present = sees && d32 == e32 && d64 == e64
  fixed = sees && d32 != e32                           # a precision-aware Float32 default is not 1e-10
  verdict(present, fixed), "operator calls: Float32 default $d32 vs rtol=1e-10 $e32; Float64 default $d64 vs rtol=1e-10 $e64"
end
case("LIM-068", "P7-03", "Testing.test_kkt default tol is 1e-8 for Float32 and Float64 (checks run exactly when |violation| > 1e-8)") do
  vs = (5e-9, 9.9e-9, 1.01e-8, 2e-8, 1e-6)
  n32 = [_D_kkt_checks(Float32, v) for v in vs]; n64 = [_D_kkt_checks(Float64, v) for v in vs]
  present = n32 == n64 == [0, 0, 3, 3, 3]
  fixed = n32 != n64                                   # the default depends on the precision
  verdict(present, fixed), "checks run at violations $(vs): Float32 $n32, Float64 $n64"
end

# ---- LIM-069: Conjugate on the Moreau route throws or misreports at extreme valid steps
_D_a = [1.0, 1.0]
case("LIM-069", "P1-26", "prox(Conjugate(IndHalfspace([1,1],1)), [0,1], γ = 1e-310): raises 'not finite' (exact [0.5, 0.5])") do
  _D_moreau(Conjugate(IndHalfspace(_D_a, 1.0)), [0.0, 1.0], 1e-310, [0.5, 0.5])
end
case("LIM-069", "P1-26", "prox(Conjugate(IndHyperplane([1,1],1)), [0,1], γ = 1e-310): raises 'not finite' (exact [0.5, 0.5])") do
  _D_moreau(Conjugate(IndHyperplane(_D_a, 1.0)), [0.0, 1.0], 1e-310, [0.5, 0.5])
end
case("LIM-069", "P1-26", "prox(Conjugate(IndHyperslab([1,1],-1,1)), [0,1], γ = 1e-310): raises 'not finite' (exact [0.5, 0.5])") do
  _D_moreau(Conjugate(IndHyperslab(_D_a, -1.0, 1.0)), [0.0, 1.0], 1e-310, [0.5, 0.5])
end
case("LIM-069", "P1-26", "prox(Conjugate(IndAffine([1 1],[1])), [0,1], γ = 1e-310): raises 'not finite' (exact [0.5, 0.5])") do
  _D_moreau(Conjugate(IndAffine([1.0 1.0], [1.0])), [0.0, 1.0], 1e-310, [0.5, 0.5])
end
case("LIM-069", "P1-26", "prox(Conjugate(IndBallL1(1)), [0.5, 1], γ = 1e-310): raises 'not finite' (exact [0.5, 1])") do
  _D_moreau(Conjugate(IndBallL1(1.0)), [0.5, 1.0], 1e-310, [0.5, 1.0])      # x/γ = [5e309, 1e310] projects to [0, 1]
end
case("LIM-069", "P1-26", "prox(Conjugate(IndSimplex(1)), [0.5, 1], γ = 1e-310): raises 'not finite' (exact [0.5, 1])") do
  _D_moreau(Conjugate(IndSimplex(1.0)), [0.5, 1.0], 1e-310, [0.5, 1.0])
end
case("LIM-069", "P1-26", "prox(Conjugate(SqrNormL2()), [1e300], γ = 1e-10): raises 'not finite' (exact 1e300/(1 + 1e-10))") do
  _D_moreau(Conjugate(SqrNormL2()), [1e300], 1e-10, [Float64(big(1e300) / (1 + big(1e-10)))])
end
case("LIM-069", "P1-26", "prox(Conjugate(SqrNormL2()), [3], γ = 1e300) returns 4.4e-16, not 3/(1 + 1e300) = 3e-300 (a misreport; not in the register's Actual)") do
  x = [3.0]; y = [7.0]
  t, r, ch = writes(() -> prox!(y, Conjugate(SqrNormL2()), x, 1e300), y)
  want = [Float64(big(3.0) / (1 + big(1e300)))]
  present = !t && abs(y[1]) > 1e-20                          # 280 orders of magnitude above the exact 3e-300
  fixed = rejects(PB.PrecisionError, t, r, ch) || (!t && isapprox(y, want; rtol = 8 * eps()))
  verdict(present, fixed), "$(show_outcome(t, r)); y = $(short(y)), exact $(short(want))"
end
case("LIM-069", "P1-26", "prox(Conjugate(ElasticNet(1,1)), [0.5, 2], γ = 1e-310): raises 'not finite' (exact [0.5, 2])") do
  _D_moreau(Conjugate(ElasticNet(1.0, 1.0)), [0.5, 2.0], 1e-310, [0.5, 2.0])      # (x + γ)/(1 + γ) beyond the unit threshold
end
case("LIM-069", "P1-26", "prox(Conjugate(SqrHingeLoss()), [-3, 2], γ = 1e-310): raises 'not finite' (exact [-3, 0])") do
  _D_moreau(Conjugate(SqrHingeLoss()), [-3.0, 2.0], 1e-310, [-3.0, 0.0])          # min(0, (x - γ)/(1 + γ/2))
end
case("LIM-069", "P1-26", "prox!(Conjugate(SeparableSum(NormL1(1), SqrNormL2())), ([0.5],[3]), γ = 1e-310): raises 'not finite' (exact ([0.5],[3]))") do
  f = Conjugate(SeparableSum(NormL1(1.0), SqrNormL2()))
  x = ([0.5], [3.0]); y = ([7.0], [7.0])
  t, r, ch = writes(() -> prox!(y, f, x, 1e-310), y)
  present = t && _D_fin(r)
  fixed = rejects(PB.PrecisionError, t, r, ch) || (!t && y[1] ≈ [0.5] && y[2] ≈ [3.0])
  verdict(present, fixed), "$(show_outcome(t, r)); blocks written = $ch, y = $(short(y))"
end
case("LIM-069", "P1-26", "prox(Conjugate(Spectral(SqrNormL2())), [floatmax 0; 0 1], γ = 0.5): raises 'not finite' (exact X/1.5)") do
  X = [M 0.0; 0.0 1.0]
  _D_moreau_spectral(Conjugate(Spectral(SqrNormL2())), X, 0.5, [M / 1.5 0.0; 0.0 1 / 1.5])
end
case("LIM-069", "P1-26", "prox(Conjugate(SingularValue(SqrNormL2())), [floatmax 0; 0 1], γ = 0.5): raises 'not finite' (exact X/1.5)") do
  X = [M 0.0; 0.0 1.0]
  _D_moreau_spectral(Conjugate(SingularValue(SqrNormL2())), X, 0.5, [M / 1.5 0.0; 0.0 1 / 1.5])
end
case("LIM-069", "P1-26", "value_and_prox!(Conjugate(IndHalfspace([1,1],1)), [0,1], γ = 1e-310): raises 'not finite' (exact value 0.5 at [0.5, 0.5])") do
  f = Conjugate(IndHalfspace(_D_a, 1.0)); y = [7.0, 7.0]
  t, r, ch = writes(() -> value_and_prox!(y, f, [0.0, 1.0], 1e-310), y)
  present = t && _D_fin(r)
  fixed = rejects(PB.PrecisionError, t, r, ch) || (!t && r ≈ 0.5 && y ≈ [0.5, 0.5])     # σ_C([.5,.5]) = 0.5
  verdict(present, fixed), "$(show_outcome(t, r)); y = $(short(y))"
end
case("LIM-069", "P4-34", "prox!(Conjugate(2 * IndBox(0, Inf)), [3,3], γ = [1e-310, 1e-310]): array step raises 'not finite' (exact [0, 0])") do
  y = [7.0, 7.0]
  t, r, ch = writes(() -> prox!(y, Conjugate(2 * IndBox(0.0, Inf)), [3.0, 3.0], [1e-310, 1e-310]), y)
  present = t && _D_fin(r)
  fixed = rejects(PB.PrecisionError, t, r, ch) || (!t && all(<=(8 * eps() * 3.0), abs.(y)))      # exact 0, up to the cancellation scale 8eps·|x|
  verdict(present, fixed), "$(show_outcome(t, r)): $(t ? shorterr(r) : short(y))"
end
case("LIM-069", "P4-34", "prox_entry(Conjugate(2 * IndBox(0, Inf)), 3.0, 1, γ = 1e-310) is -Inf (exact 0)") do
  t, r = attempt(() -> prox_entry(Conjugate(2 * IndBox(0.0, Inf)), 3.0, 1, 1e-310))
  present = !t && r == -Inf
  fixed = !t && abs(r) <= 8 * eps() * 3.0                 # exact 0, up to the cancellation scale 8eps·|v|
  verdict(present, fixed), show_outcome(t, r)
end
case("LIM-069", "P4-34", "prox_entry(Conjugate(SqrNormL2()), 3.0, 1, γ = 1e-310) is NaN (exact 3.0); the register's Actual names only the boxed cone") do
  t, r = attempt(() -> prox_entry(Conjugate(SqrNormL2()), 3.0, 1, 1e-310))
  present = !t && isnan(r)
  fixed = !t && r == 3.0
  verdict(present, fixed), show_outcome(t, r)
end
case("LIM-069", "P2-36", "prox(Conjugate(IndBallL1(1f0)), Float32[0,1], γ = 1e-310): Float64 step underflows in Float32, raises 'not finite' (exact [0, 1])") do
  _D_moreau(Conjugate(IndBallL1(1f0)), Float32[0, 1], 1e-310, Float32[0, 1])
end
case("LIM-069", "P2-36", "prox(Conjugate(IndBallL1(1f0)), Float32[0,1], γ = 1e300): Float64 step overflows in Float32, raises 'not finite' (exact [0, 0])") do
  _D_moreau(Conjugate(IndBallL1(1f0)), Float32[0, 1], 1e300, Float32[0, 0])
end

# ---- LIM-070: Conjugate on the Moreau route loses results to cancellation
case("LIM-070", "P3-10", "prox(Conjugate(SLOPE([1,.5,.25])), [1e300,.25,-.5], 1) is [0,.25,-.5] (exact [1,.25,-.5])") do
  t, r = attempt(() -> prox(Conjugate(SLOPE([1.0, 0.5, 0.25])), [1e300, 0.25, -0.5], 1.0))
  want = [1.0, 0.25, -0.5]                     # projection onto the dual ball: |y|₍₁₎ ≤ 1, partial sums ≤ 1, 1.5, 1.75
  present = !t && r[1] == 0.0 && r[2] == 0.25 && r[3] == -0.5
  fixed = !t && isapprox(r, want; rtol = 1e-14)
  verdict(present, fixed), "$(show_outcome(t, r)), exact $(short(want))"
end
case("LIM-070", "P3-10", "prox(Conjugate(TotalVariation1D(1)), [1e300,.25,-.5], 1) is [0,-.125,-.875], sum -1 (dual ball needs sum 0; exact [1,-.125,-.875])") do
  t, r = attempt(() -> prox(Conjugate(TotalVariation1D(1.0)), [1e300, 0.25, -0.5], 1.0))
  want = [1.0, -0.125, -0.875]                 # s₁ = 1 at the cap, s₂ = 0.875 minimizes (s₂ - 1.25)² + (s₂ - 0.5)²
  present = !t && r[1] == 0.0 && sum(r) == -1.0
  fixed = !t && isapprox(r, want; rtol = 1e-14) && abs(sum(r)) <= 1e-14
  verdict(present, fixed), "$(show_outcome(t, r)), sum = $(t ? "-" : sum(r)), exact $(short(want))"
end

# ---- LIM-071: strong zeros (FD-9) not implemented in products
case("LIM-071", "P6-23", "SparseIterate: dot([1,2,Inf], s) is 2.0 unstored and NaN once coordinate 3 is activated with value 0 (exact 2.0 both)") do
  s = SparseIterate(Float64, 3); s[2] = 1.0; a = [1.0, 2.0, Inf]
  d1 = dot(a, s); activate!(s, 3); d2 = dot(a, s)
  present = d1 == 2.0 && isnan(d2)
  fixed = d1 == 2.0 && d2 == 2.0
  verdict(present, fixed), "unstored: $d1, stored zero: $d2"
end
case("LIM-071", "P4-43", "dot(a, b) with a stored NaN against an unstored zero of b is NaN, dot(b, a) is 0.0 (exact 0.0 both)") do
  a = SparseIterate(3); a[1] = NaN; b = SparseIterate(3); b[2] = 1.0
  d1, d2 = dot(a, b), dot(b, a)
  present = isnan(d1) && d2 == 0.0
  fixed = d1 == 0.0 && d2 == 0.0
  verdict(present, fixed), "dot(a,b) = $d1, dot(b,a) = $d2"
end
case("LIM-071", "P4-44", "AtomIterate with an active zero group: dot(v, a) and dot(a, v) are NaN for v = [1,NaN,1] (exact 1.0)") do
  a = AtomIterate(3, 3); activate!(a, 1); activate!(a, 2); atom(a, 1) .= 1; atom(a, 2) .= 0
  v = [1.0, NaN, 1.0]
  d1, d2 = dot(v, a), dot(a, v)
  present = isnan(d1) && isnan(d2)
  fixed = d1 == 1.0 && d2 == 1.0
  verdict(present, fixed), "dot(v,a) = $d1, dot(a,v) = $d2"
end
case("LIM-071", "P4-44", "SparseIterate: A*x with a stored zero against A[1,2] = Inf is [NaN,1,1], [1,1,1] after dropzeros! (exact [1,1,1] both)") do
  A = [1.0 Inf; 1.0 1.0; 1.0 1.0]
  x = SparseIterate(2); x[1] = 1.0; activate!(x, 2)
  before = A * x; dropzeros!(x); after = A * x
  present = isnan(before[1]) && before[2:3] == [1.0, 1.0] && after == [1.0, 1.0, 1.0]
  fixed = before == [1.0, 1.0, 1.0] && after == [1.0, 1.0, 1.0]
  verdict(present, fixed), "A*x stored zero: $(short(before)), after dropzeros!: $(short(after))"
end
case("LIM-071", "P6-23", "mul!(y, matrixview(X), [Inf,1,0]) with a stored zero at (3,1): y[3] NaN, 0.0 after dropzeros! (exact [1,Inf,0] both)") do
  X = SymmetricSparseIterate(Float64, 3); X[2, 1] = 1.0; activate!(X, tril_index(3, 3, 1))
  v = [Inf, 1.0, 0.0]
  before = mul!(zeros(3), matrixview(X), v); dropzeros!(X); after = mul!(zeros(3), matrixview(X), v)
  present = nactive(X) == 1 && isnan(before[3]) && after == [1.0, Inf, 0.0]
  fixed = before == [1.0, Inf, 0.0] && after == [1.0, Inf, 0.0]
  verdict(present, fixed), "stored zero: $(short(before)), after dropzeros!: $(short(after))"
end

# ---- LIM-072: sparse value oracles read every entry
skip("LIM-072", "P6-24", "sparse value oracles (NormL2 recovery, IndSimplex, IndBallL0, LinearFunction matrix) make O(p²) zeros on sparse iterates",
     "needs the zero-counting element type of test/sparse_value_cost.jl; evidence: the @test_broken rows test/sparse_value_cost.jl:162, 165, 169, 173 (testset \"Known dense paths\"; 6 broken results, rows 162 and 165 run for :sym and :mat) are still broken at c3454e25; zeros made on 1.13: 19994, 19994, 19996, 19996, 5048, 9998")

# ---- LIM-073: SymmetricSparseIterate scaling by Inf/NaN/0 depends on storage
function _D_scale(op!, bc!)
  X = SymmetricSparseIterate(Float64, 3); X[1, 1] = 1.0; X[2, 1] = 2.0
  Y = copy(X); op!(X); bc!(Y)
  mx, my = Matrix(X), Matrix(Y)
  present = mx[3, 3] == 0.0 && isnan(my[3, 3])               # an unstored position: 0 by the loop, NaN by the broadcast
  fixed = isequal(mx, my) && nactive(X) == nactive(Y)
  verdict(present, fixed), "loop: $(short(vec(mx))) nactive $(nactive(X)); broadcast: $(short(vec(my))) nactive $(nactive(Y))"
end
case("LIM-073", "P6-03", "rmul!(X, Inf) leaves unstored positions 0, X .*= Inf makes them NaN") do
  _D_scale(X -> rmul!(X, Inf), Y -> (Y .*= Inf))
end
case("LIM-073", "P6-03", "lmul!(NaN, X) leaves unstored positions 0, X .= NaN .* X makes them NaN") do
  _D_scale(X -> lmul!(NaN, X), Y -> (Y .= NaN .* Y))
end
case("LIM-073", "P6-03", "rdiv!(X, 0.0) leaves unstored positions 0, X ./= 0.0 makes them NaN") do
  _D_scale(X -> rdiv!(X, 0.0), Y -> (Y ./= 0.0))
end
case("LIM-073", "P6-03", "ldiv!(0.0, X) leaves unstored positions 0, X .= 0.0 .\\ X makes them NaN") do
  _D_scale(X -> ldiv!(0.0, X), Y -> (Y .= 0.0 .\ Y))
end

# ---- LIM-074: SymmetricSparseIterate is still an AbstractMatrix
# a loop over x[i, j] that visits both triangles also doubles, but IDX-2 keeps x[i, j]: documented, not a gap
case("LIM-074", "P6-02", "for k in eachindex(x); x[k] *= 2; end doubles an off-diagonal 1.0 to 4.0 (positions 2 and 4 are one variable)") do
  x = _D_ssi()
  t, r = attempt(() -> (for k in eachindex(x); x[k] *= 2; end))
  present = !t && x[2, 1] == 4.0
  fixed = t && (r isa MethodError || r isa ArgumentError)          # OWN-3: eachindex must not return values
  verdict(present, fixed), t ? show_outcome(t, r) : "x[2,1] = $(x[2, 1])"
end
case("LIM-074", "P6-02", "x[2] is the matrix position (2,1): returns 1.0 (T09: ArgumentError naming coordinates and matrixview)") do
  x = _D_ssi()
  t, r = attempt(() -> x[2])
  present = !t && r == 1.0
  fixed = t && r isa ArgumentError && occursin("coordinates", _D_msg(r)) && occursin("matrixview", _D_msg(r))
  verdict(present, fixed), show_outcome(t, r)
end
case("LIM-074", "P6-02", "length(x) is 9 for a 3x3 owner (T09: MethodError)") do
  x = _D_ssi()
  t, r = attempt(() -> length(x))
  present = !t && r == 9
  fixed = t && r isa MethodError
  verdict(present, fixed), show_outcome(t, r)
end
case("LIM-074", "P6-86", "nnz(x) is defined (1 for one stored off-diagonal pair); T09/ACT-2: MethodError") do
  x = _D_ssi()
  t, r = attempt(() -> nnz(x))
  present = !t && r == 1
  fixed = t && r isa MethodError
  verdict(present, fixed), show_outcome(t, r)
end
case("LIM-074", "P6-17", "Random.shuffle!(rng, x) runs silently on the owner (OWN-1: MethodError)") do
  x = SymmetricSparseIterate(Float64, 3); x[1, 1] = 1.0; x[2, 1] = 1.0; x[3, 3] = 2.0
  t, r = attempt(() -> shuffle!(MersenneTwister(1), x))
  present = !t && r === x
  fixed = t && r isa MethodError
  verdict(present, fixed), t ? show_outcome(t, r) : "no error, returns x, symmetric = $(issymmetric(Matrix(x)))"
end
case("LIM-074", "P6-17", "Random.rand!(rng, x) runs silently on the owner (OWN-1: MethodError)") do
  x = SymmetricSparseIterate(Float64, 3)
  t, r = attempt(() -> rand!(MersenneTwister(1), x))
  present = !t && r === x && issymmetric(Matrix(x))
  fixed = t && r isa MethodError
  verdict(present, fixed), t ? show_outcome(t, r) : "no error, returns x, symmetric = $(issymmetric(Matrix(x))), nactive $(nactive(x))"
end

# ---- LIM-075: copyto! into views and reshapes of a SymmetricSparseIterate is unchecked
function _D_unchecked(route, src, collapsed)
  x = SymmetricSparseIterate(Float64, 3)
  t, r = attempt(() -> copyto!(route(x), src))
  m = Matrix(x)
  ch = any(!iszero, m)
  present = !t && m == collapsed
  fixed = rejects(ArgumentError, t, r, ch)             # asymmetric source: ArgumentError before writing
  verdict(present, fixed), "$(show_outcome(t, r)), Matrix(x) = $(short(m))"
end
case("LIM-075", "P6-18", "copyto!(vec(x), 1:9): no error, leaves [1 4 7; 4 5 8; 7 8 9] (upper positions win)") do
  _D_unchecked(vec, collect(1.0:9.0), [1.0 4.0 7.0; 4.0 5.0 8.0; 7.0 8.0 9.0])
end
case("LIM-075", "P6-18", "copyto!(view(x, 1:2, 1:2), [1 2; 3 4]): no error, stores [1 2; 2 4] (3 is discarded)") do
  _D_unchecked(x -> view(x, 1:2, 1:2), [1.0 2.0; 3.0 4.0], [1.0 2.0 0.0; 2.0 4.0 0.0; 0.0 0.0 0.0])
end
case("LIM-075", "P6-18", "copyto!(reshape(x, 9), 1:9): no error, leaves [1 4 7; 4 5 8; 7 8 9]") do
  _D_unchecked(x -> reshape(x, 9), collect(1.0:9.0), [1.0 4.0 7.0; 4.0 5.0 8.0; 7.0 8.0 9.0])
end
case("LIM-075", "P6-18", "copyto!(view(vec(x), 1:9), 1:9) (nested wrappers): no error, leaves [1 4 7; 4 5 8; 7 8 9]") do
  _D_unchecked(x -> view(vec(x), 1:9), collect(1.0:9.0), [1.0 4.0 7.0; 4.0 5.0 8.0; 7.0 8.0 9.0])
end

# ---- LIM-076: fill!(x, 0) and x .= 0 empty the working set
case("LIM-076", "P6-26", "SparseIterate: fill!(x, 0.0) empties the working set [1, 2] (exact [1, 2])") do
  x = SparseIterate(4); x[1] = 1.0; x[2] = 2.0; fill!(x, 0.0)
  a = collect(activecoordinates(x))
  verdict(isempty(a), a == [1, 2]), "active after fill! = $(short(a))"
end
case("LIM-076", "P6-26", "SparseIterate: x .= 0 empties the working set [1, 2] (exact [1, 2])") do
  x = SparseIterate(4); x[1] = 1.0; x[2] = 2.0; x .= 0
  a = collect(activecoordinates(x))
  verdict(isempty(a), a == [1, 2]), "active after x .= 0: $(short(a))"
end
case("LIM-076", "P6-26", "AtomIterate: fill!(x, 0.0) empties the working set [1] (exact [1])") do
  x = AtomIterate(Float64, 4, [[1, 2], [3, 4]]); activate!(x, 1); atom(x, 1) .= 1.0; fill!(x, 0.0)
  a = collect(activecoordinates(x))
  verdict(isempty(a), a == [1]), "active after fill! = $(short(a))"
end
case("LIM-076", "P6-26", "AtomIterate: x .= 0 empties the working set [1] (exact [1])") do
  x = AtomIterate(Float64, 4, [[1, 2], [3, 4]]); activate!(x, 1); atom(x, 1) .= 1.0; x .= 0
  a = collect(activecoordinates(x))
  verdict(isempty(a), a == [1]), "active after x .= 0: $(short(a))"
end

# ---- LIM-077: coordinates(::SparseIterate) returns x itself
case("LIM-077", "P6-31", "coordinates(M) === M for a 2x2 SparseIterate (FC-3: an AbstractVector of length ncoordinates)") do
  M2 = SparseIterate(Float64, 2, 2); c = coordinates(M2)
  verdict(c === M2, !(c === M2) && c isa AbstractVector && length(c) == ncoordinates(M2)), "coordinates(M) === M: $(c === M2), $(typeof(c))"
end
case("LIM-077", "P6-31", "coordinates(v) === v for a SparseIterate vector (FC-3: an AbstractVector of length ncoordinates)") do
  v = SparseIterate(Float64, 4); c = coordinates(v)
  verdict(c === v, !(c === v) && c isa AbstractVector && length(c) == ncoordinates(v)), "coordinates(v) === v: $(c === v), $(typeof(c))"
end

# ---- LIM-078: AtomIterate group vectors are shared mutable state
case("LIM-078", "P6-34", "writing parentindices(atom(x, 1))[1][1] = 3 changes x.groups to [[3,2],[3,4]] while group_of stays [1,1,2,2]") do
  x = AtomIterate(Float64, 4, [[1, 2], [3, 4]])
  t, r = attempt(() -> (parentindices(atom(x, 1))[1][1] = 3))
  present = !t && x.groups == [[3, 2], [3, 4]] && x.group_of == [1, 1, 2, 2]
  fixed = x.groups == [[1, 2], [3, 4]]                 # the layout is read-only: the write fails or does not reach it
  verdict(present, fixed), "write $(show_outcome(t, r)); groups = $(short(x.groups)), group_of = $(short(x.group_of))"
end

# ---- LIM-079: asymmetry diagnostic prints NaN for BigFloat beyond the Float64 range
case("LIM-079", "P4-70", "copyto!(SymmetricSparseIterate(BigFloat,2), [1e400 2e400; 3e400 1e400]): message says (relative asymmetry NaN), exact 0.33") do
  t, r = attempt(() -> copyto!(SymmetricSparseIterate(BigFloat, 2), [big"1e400" big"2e400"; big"3e400" big"1e400"]))
  m = t ? _D_msg(r) : ""
  present = t && r isa ArgumentError && occursin("(relative asymmetry NaN)", m)
  mm = match(r"\(relative asymmetry ([^),]*)", m)
  v = mm === nothing ? nothing : tryparse(Float64, mm.captures[1])
  fixed = t && r isa ArgumentError && v !== nothing && isapprox(v, 0.33; atol = 0.005)     # scale invariant: as for [1 2; 3 1]
  verdict(present, fixed), t ? "ArgumentError, printed asymmetry: $(mm === nothing ? "none" : mm.captures[1])" : "no error"
end

# ---- LIM-080: Testing.test_gradient does not check leaf element types
case("LIM-080", "P7-08", "test_gradient accepts Float64 gradient leaves for a Float32 tuple (a failed check is the fixed state)") do
  x = (Float32[1, 2], Float32[3, 4])
  run(f) = attempt(() -> redirect_stdout(devnull) do      # the test set prints its summary
    @testset "LIM-080 probe" begin
      PB.Testing.test_gradient(f, x)
    end
  end)
  tb, rb = run(_D_BadGrad(SeparableSum(SqrNormL2(), SqrNormL2())))
  tg, rg = run(SeparableSum(SqrNormL2(), SqrNormL2()))          # control: the correct Float32 gradient must keep passing
  control = !tg && rg.n_passed > 0
  present = control && !tb && rb.n_passed > 0
  fixed = control && tb && rb isa Test.TestSetException && rb.fail >= 1 && rb.error == 0
  verdict(present, fixed), tb ? "bad gradient: $(rb isa Test.TestSetException ? "TestSetException pass=$(rb.pass) fail=$(rb.fail) error=$(rb.error)" : nameof(typeof(rb)))" :
                              "bad gradient passes: $(rb.n_passed) checks, none failed; control passes: $control"
end

# ================================================================ fixed claims rechecked
# Behaviors the audit classified FIXED whose index row cited no test or fixing
# pull request (review of #558). These print FIXED while the fix holds. One of
# them, P7-26, was only half fixed: its GroupNormLinf case prints PRESENT and is
# tagged "index", because the allocation it shows is in clusters.tsv, not in the
# register (performance).
_F_src(path) = read(joinpath(pkgdir(ProximalBase), path), String)
function _F_alloc(f, y, x, γ)
  prox!(y, f, x, γ); value_and_prox!(y, f, x, γ)
  a = @allocated prox!(y, f, x, γ)
  b = @allocated value_and_prox!(y, f, x, γ)
  (a, b)
end

case("fixed", "P4-07", "quadratic_model docstring states the GLMLoss curvature floor") do
  d = string(@doc quadratic_model)
  floor_stated = occursin("floored at `√eps`", d) && occursin("regularized", d) && occursin("curvature model where the floor is active", d)
  verdict(!floor_stated, floor_stated), "floor and regularized model stated in the docstring: $floor_stated"
end
case("fixed", "P4-67", "no bitwise promise left in docs/src or src") do
  hits = String[]
  for dir in ("docs/src", "src"), (root, _, files) in walkdir(joinpath(pkgdir(ProximalBase), dir)), f in files
    endswith(f, r"\.(md|jl)") || continue
    for (i, l) in enumerate(eachline(joinpath(root, f)))
      occursin(r"(?i)bitwise|same bits|identical bits", l) && push!(hits, "$(relpath(joinpath(root, f), pkgdir(ProximalBase))):$i")
    end
  end
  only_s3 = hits == ["docs/src/numerics.md:$(findfirst(l -> startswith(l, "**S3."), readlines(joinpath(pkgdir(ProximalBase), "docs/src/numerics.md"))))"]
  verdict(!only_s3, only_s3), "matches: $(join(hits, ", ")) (expected only S3's 'Not promised')"
end
case("fixed", "P5-07", "active-set accessors are exported") do
  names_ = names(ProximalBase)
  want = (:ncoordinates, :nactive, :activecoordinates, :activevalues, :isactive)
  ok = all(in(names_), want)
  verdict(!ok, ok), join(["$n=$(n in names_)" for n in want], " ")
end
if HAVE_JL
  case("fixed", "P5-59", "device gradient! and value_and_gradient! of MoreauEnvelope(NormL1)") do
    f = MoreauEnvelope(NormL1(1.0), 0.5); x = [0.5, -2.0, 1.5]
    g = Array(gradient!(JLArray(zeros(3)), f, JLArray(x))); gc = gradient!(zeros(3), f, x)
    v = value_and_gradient!(JLArray(zeros(3)), f, JLArray(x)); vc = value_and_gradient!(zeros(3), f, x)
    ok = g == gc && v ≈ vc
    verdict(!ok, ok), "device $(short(g)) value $v; CPU $(short(gc)) value $vc"
  end
  case("fixed", "P5-78", "device CappedL1, LogPenalty, NormLq, Berhu: value and prox! match the CPU") do
    x = [0.3, -2.5, 1.7, 0.0]; bad = String[]
    for f in (CappedL1(1.0, 2.0), LogPenalty(1.0, 0.5), NormLq(1.0, 0.5), Berhu(1.0, 1.0))
      yd = Array(prox!(JLArray(zeros(4)), f, JLArray(x), 0.7)); yc = prox!(zeros(4), f, x, 0.7)
      vd = f(JLArray(x)); vc = f(x)
      (isequal(yd, yc) && vd ≈ vc) || push!(bad, string(nameof(typeof(f))))
    end
    verdict(!isempty(bad), isempty(bad)), isempty(bad) ? "all four agree" : "differ: $(join(bad, ", "))"
  end
else
  skip("fixed", "P5-59", "device gradient of MoreauEnvelope", "JLArrays/Adapt not loadable in this environment")
  skip("fixed", "P5-78", "device CappedL1, LogPenalty, NormLq, Berhu", "JLArrays/Adapt not loadable in this environment")
end
case("fixed", "P6-71", "Constrained(NegLog, hyperplane or halfspace through the closure only): ArgumentError before writing") do
  y1 = fill(7.0, 2); t1, r1, c1 = writes(() -> prox!(y1, Constrained(NegLog(1.0), IndHyperplane([1.0, 1.0], 0.0)), [0.5, 0.5], 1.0), y1)
  y2 = fill(7.0, 2); t2, r2, c2 = writes(() -> prox!(y2, Constrained(NegLog(1.0), IndHalfspace([1.0, 1.0], 0.0)), [0.5, 0.5], 1.0), y2)
  ok = rejects(ArgumentError, t1, r1, c1) && rejects(ArgumentError, t2, r2, c2)
  bad = (!t1 && any(isinf, y1)) || (!t2 && any(isinf, y2))
  verdict(bad, ok), "hyperplane: $(show_outcome(t1, r1)), written=$c1; halfspace: $(show_outcome(t2, r2)), written=$c2"
end
case("fixed", "P6-71", "Constrained(NegLog, tiny-coefficient feasible hyperplane): the projection [1, 1]") do
  y = zeros(2); t, v = attempt(() -> value_and_prox!(y, Constrained(NegLog(1.0), IndHyperplane([1e-300, 1e-300], 2e-300)), [0.5, 0.5], 1.0))
  ok = !t && isapprox(y, [1.0, 1.0]; rtol = 1e-12) && abs(v) <= 1e-12
  verdict(t, ok), "$(show_outcome(t, v)), y = $(short(y)) (exact [1, 1], value 0)"
end
for (name, f) in (("NormL2", Constrained(NormL2(1.0), IndBallL2(M))),
                  ("GroupNormLinf", Constrained(GroupNormLinf(1.0, [1:2]), IndBallL2(M))))
  # the audit probed only NormL2; GroupNormLinf still allocates (reclassified CONFIRMED in clusters.tsv)
  case(name == "NormL2" ? "fixed" : "index", "P7-26", "Constrained($name, IndBallL2) at an overflowing norm: no allocation") do
    a, b = _F_alloc(f, zeros(2), [0.6M, 0.6M], 1.0)
    verdict(a > 0 || b > 0, a == 0 && b == 0), "prox! $a B, value_and_prox! $b B"
  end
end
case("fixed", "P7-82", "iterate acceptance plan header says implementation is in progress") do
  head = join(first(readlines(joinpath(pkgdir(ProximalBase), "docs/design/iterate-family-0.6-acceptance.md")), 12), " ")
  ok = occursin("Implementation is in progress on `master`", head)
  verdict(!ok, ok), "header mentions implementation in progress: $ok"
end

println("# summary: ", join(["$s=$(COUNT[s])" for s in STATUSES], ", "))
println("# by entry (a case that is not PRESENT needs a look):")
for lim in sort(collect(keys(BYLIM)))
  d = BYLIM[lim]
  println("#   ", rpad(lim, 9), join(["$(s)=$(d[s])" for s in STATUSES if d[s] > 0], " "))
end
