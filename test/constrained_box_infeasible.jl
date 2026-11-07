module ConstrainedBoxInfeasibleTest

# #666, item 2 (O6, decision 49): Constrained(NormL1(Inf), IndBox(l, u)) fixes every entry with a
# positive weight at 0 at a positive step, so a box that excludes 0 at such an entry has no feasible
# point. prox!, prox and value_and_prox! raise ArgumentError("the IndBox misses the domain of NormL1")
# before the first write, instead of returning the clamp of the hard zero into the box. Entries of
# weight zero are free and are not checked. Rows marked "regression guard" pass at base and must keep
# their results: a box that contains 0, finite levels, other families, the standalone value, the
# hyperplane form and the wrappers whose check this change does not decide.
# Outputs are compared with isequal.

using Test, ProximalBase, Random

const PB = ProximalBase
const MSG = "the IndBox misses the domain of NormL1"

# the message of the ArgumentError that the call raises, or what it raised or returned instead
msg_of(f) = try f(); nothing catch e; e isa ArgumentError ? e.msg : e end

# The call raises the message, and a sentinel output or an aliased argument is exactly as it was.
function raises(op, f, x, γ)
  T = float(eltype(x))
  ok = true
  for out in (:fresh, :alias)
    y = out === :fresh ? fill(T(-7), size(x)) : copy(x)
    xin = out === :fresh ? x : y
    before = copy(y)
    r = try
      op === prox ? prox(f, xin, γ) : op(y, f, xin, γ)
      nothing
    catch e
      e
    end
    ok &= r isa ArgumentError && r.msg == MSG
    op === prox || (ok &= isequal(y, before))
    op === prox && out === :alias && (ok &= isequal(xin, before))
  end
  ok
end
raises_all(f, x, γ) = all(op -> raises(op, f, x, γ), (prox, prox!, value_and_prox!))

# prox, prox! and value_and_prox! into a fresh output, a sentinel output and the argument itself,
# all equal to the expected output (isequal), the value being the value at the stored output
function check(f, x, γ, expected; value = nothing)
  T = float(eltype(x))
  @test isequal(prox(f, x, γ), expected)
  for op in (prox!, value_and_prox!)
    fresh = similar(x, T)
    v1 = op(fresh, f, x, γ)
    sentinel = fill(T(-7), size(x))
    op(sentinel, f, x, γ)
    alias = copy(x)
    v3 = op(alias, f, alias, γ)
    @test isequal(fresh, expected) && isequal(sentinel, expected) && isequal(alias, expected)
    if op === value_and_prox! && value !== nothing
      @test isequal(v1, value) && isequal(v3, value)
    end
  end
end

####################################
#  the rows that fail without the check
####################################

@testset "the box misses the domain: raises before writing" begin
  for T in (Float32, Float64)
    inf = T(Inf)
    for (l, u) in ((T(1), T(2)), (T(-2), T(-1)), (T(1e-30), T(1)), (T(-1), -floatmin(T)), (-inf, T(-1)), (T(1), inf))
      f = Constrained(NormL1(inf), IndBox(l, u))
      for x in (T[1, 2], T[-1, 5, 0], T[0, 0], T[NaN, 1], T[Inf, -Inf], T[3.5])
        @test raises_all(f, x, T(1))
      end
      # a positive step of any size, finite or Inf, and a step of the other precision
      for γ in (T(0.5), T(1e30), floatmin(T), inf, 2, 1.0, 1e300)
        @test raises_all(f, T[1, 2], γ)
      end
    end
    # matrices
    @test raises_all(Constrained(NormL1(inf), IndBox(T(1), T(2))), T[1 2; 3 4], T(1))
  end
  # the table of the issue
  @test msg_of(() -> prox(Constrained(NormL1(Inf), IndBox(1.0, 2.0)), [1.0, 2.0], 1.0)) == MSG
  @test msg_of(() -> prox(Constrained(NormL1(Inf, [1.0, 0.0]), IndBox(1.0, 2.0)), [1.0, 2.0], 1.0)) == MSG
  # Int bounds and an infinite level of another precision
  @test raises_all(Constrained(NormL1(Inf), IndBox(1, 2)), [1.0, 2.0], 1.0)
  @test raises_all(Constrained(NormL1(Inf32), IndBox(1.0, 2.0)), [1.0, 2.0], 1.0)
  @test raises_all(Constrained(NormL1(Inf), IndBox(1f0, 2f0)), Float32[1, 2], 1f0)
end

@testset "array bounds, weights and array steps" begin
  for T in (Float32, Float64)
    inf = T(Inf)
    x = T[1, 2, 3]
    l = T[-1, 1, -3]
    u = T[1, 2, 3]
    # the box excludes 0 at entry 2 only
    f = Constrained(NormL1(inf), IndBox(l, u))
    @test raises_all(f, x, T(1))
    # a weight of zero leaves entry 2 free: the output is the clamp of the free entry, the others are 0
    w0 = T[1, 0, 1]
    g = Constrained(NormL1(inf, w0), IndBox(l, u))
    check(g, x, T(1), T[0, 2, 0])
    check(g, T[5, 0.25, 5], T(1), T[0, 1, 0])
    # a positive weight fixes it again, an infinite weight too (a hard weight with an infinite level)
    @test raises_all(Constrained(NormL1(inf, T[1, 1, 1]), IndBox(l, u)), x, T(1))
    @test raises_all(Constrained(NormL1(inf, T[1, 1e-30, 1]), IndBox(l, u)), x, T(1))
    @test raises_all(Constrained(NormL1(inf, T[0, inf, 0]), IndBox(l, u)), x, T(1))
    @test raises_all(Constrained(NormL1(inf, T[0, 2, 0]), IndBox(l, u)), x, T(1))
    # every excluded entry has weight zero
    check(Constrained(NormL1(inf, T[1, 0, 1]), IndBox(l, u)), T[NaN, NaN, NaN], T(1), T[0, NaN, 0])
    # mixed hard and free entries: entries 1 and 3 are excluded, 2 is free and 1 is hard
    l2, u2 = T[1, -1, -5], T[2, 1, -4]
    @test raises_all(Constrained(NormL1(inf, T[1, 0, 0]), IndBox(l2, u2)), x, T(1))
    @test raises_all(Constrained(NormL1(inf, T[0, 0, 1]), IndBox(l2, u2)), x, T(1))
    check(Constrained(NormL1(inf, T[0, 1, 0]), IndBox(l2, u2)), x, T(1), T[1, 0, -4])
    # an array step with a positive entry at the excluded coordinate
    @test raises_all(f, x, T[1, 1, 1])
    @test raises_all(f, x, T[0.5, 2, 1e30])
    @test raises_all(f, x, T[1, inf, 1])
    # scalar bounds against array weights, and array bounds against no weights
    @test raises_all(Constrained(NormL1(inf, T[1, 0, 0]), IndBox(T(1), T(2))), x, T(1))
    @test raises_all(Constrained(NormL1(inf), IndBox(T[1, -1, -1], T[2, 1, 1])), x, T(1))
    check(Constrained(NormL1(inf, T[0, 1, 1]), IndBox(T[1, -1, -1], T[2, 1, 1])), x, T(1), T[1, 0, 0])
  end
  # array bounds shaped like a matrix argument
  l = [-1.0 1.0; -1.0 -1.0]
  u = [1.0 2.0; 1.0 1.0]
  @test raises_all(Constrained(NormL1(Inf), IndBox(l, u)), [1.0 2.0; 3.0 4.0], 1.0)
  check(Constrained(NormL1(Inf, [1.0 0.0; 1.0 1.0]), IndBox(l, u)), [1.0 2.0; 3.0 4.0], 1.0, [0.0 2.0; 0.0 0.0])
end

@testset "boundary cases: the box contains 0" begin
  # 0 on an endpoint, a singleton at 0 and infinite bounds: feasible, the output is the hard zero
  boxes = (IndBox(0.0, 1.0), IndBox(-1.0, 0.0), IndBox(-0.0, 1.0), IndBox(-1.0, -0.0), IndBox(0.0, 0.0),
           IndBox(-0.0, 0.0), IndBox(0.0, -0.0), IndBox(-0.0, -0.0), IndBox(-Inf, Inf), IndBox(-Inf, 0.0),
           IndBox(0.0, Inf), IndBox(-Inf, 1.0), IndBox(-1.0, Inf), IndBox(0, 1), IndBox(-1, 0),
           IndBox([0.0, -1.0], [1.0, 0.0]), IndBox([-0.0, -2.0], [-0.0, 2.0]))
  for b in boxes, x in ([1.0, -2.0], [NaN, Inf], [0.0, 0.0]), γ in (1.0, 1e300, Inf, [1.0, 2.0])
    y = prox(Constrained(NormL1(Inf), b), x, γ)
    @test all(iszero, y) && length(y) == 2
    @test all(iszero, prox(Constrained(NormL1(Inf, [1.0, 1.0]), b), x, γ))
  end
  @test all(iszero, prox(Constrained(NormL1(Inf), IndNonnegative()), [1.0, -2.0], 1.0))
  # the smallest excluded intervals
  @test raises_all(Constrained(NormL1(Inf), IndBox(nextfloat(0.0), 1.0)), [1.0, -2.0], 1.0)
  @test raises_all(Constrained(NormL1(Inf), IndBox(-1.0, prevfloat(-0.0))), [1.0, -2.0], 1.0)
  @test raises_all(Constrained(NormL1(Inf), IndBox([0.0, nextfloat(0.0)], [1.0, 1.0])), [1.0, -2.0], 1.0)
  # a single excluded entry among many that contain 0
  l = fill(-1.0, 50); u = fill(1.0, 50); l[37] = 0.5; u[37] = 1.0
  @test raises_all(Constrained(NormL1(Inf), IndBox(l, u)), randn(MersenneTwister(1), 50), 1.0)
  # the same entry with weight zero, the box contains 0 elsewhere
  wf = ones(50); wf[37] = 0.0
  y = prox(Constrained(NormL1(Inf, wf), IndBox(l, u)), fill(3.0, 50), 1.0)
  @test y[37] == 1.0 && all(iszero, y[[1:36; 38:50]])
end

####################################
#  nothing is written, and the value
####################################

@testset "no partial write; value and empty arguments" begin
  # the infeasible entry is the last one: the entries before it are not written either
  f = Constrained(NormL1(Inf), IndBox([-1.0, -1.0, -1.0, 1.0], [1.0, 1.0, 1.0, 2.0]))
  x = [3.0, -3.0, 0.5, 0.0]
  for op in (prox!, value_and_prox!)
    y = fill(-7.0, 4)
    @test msg_of(() -> op(y, f, x, 1.0)) == MSG
    @test y == fill(-7.0, 4)
    a = copy(x)
    @test msg_of(() -> op(a, f, a, 1.0)) == MSG
    @test a == x
    @test msg_of(() -> op(y, f, x, fill(1.0, 4))) == MSG
    @test y == fill(-7.0, 4)
  end
  # the standalone value of the infeasible function is Inf, with or without a throw in the prox
  g = Constrained(NormL1(Inf), IndBox(1.0, 2.0))
  @test g([1.0, 2.0]) == Inf && g([0.0, 0.0]) == Inf
  @test Constrained(NormL1(Inf), IndBox(-1.0, 2.0))([0.0, 0.0]) == 0.0
  # an empty argument has no entry to check
  @test prox(g, Float64[], 1.0) == Float64[]
  @test value_and_prox!(Float64[], g, Float64[], 1.0) == 0.0
  # a hard-fixed coordinate in a box that contains 0 keeps the value 0 at the stored output
  h = Constrained(NormL1(Inf), IndBox(-1.0, 2.0))
  check(h, [1.0, 2.0], 1.0, [0.0, 0.0]; value = 0.0)
  check(h, [NaN, Inf], 1.0, [0.0, 0.0]; value = 0.0)
end

####################################
#  regression guards: unchanged results
####################################

@testset "regression guard: finite levels, other families, other constraints" begin
  rng = MersenneTwister(666)
  # finite levels and boxes that exclude 0: the clamp of the soft threshold, as composed by hand
  for T in (Float32, Float64), n in (1, 2, 7, 50), _ in 1:20
    x = randn(rng, T, n) .* 5
    γ = T(rand(rng) * 3 + 0.1)
    λ = T(rand(rng) * 2 + 0.1)
    l = T(1) .+ rand(rng, T, n)
    u = l .+ rand(rng, T, n)
    for w in (nothing, rand(rng, T, n), [rand(rng, T, n - 1); T(0)])
      for b in (IndBox(T(1), T(2)), IndBox(l, u), IndBox(-u, -l), IndBox(-l, u))
        f = NormL1(λ, w)
        expected = prox(b, prox(f, x, γ), γ)
        @test isequal(prox(Constrained(f, b), x, γ), expected)
        @test isequal(prox(Constrained(f, b), x, fill(γ, n)), expected)
        y = similar(x)
        v = value_and_prox!(y, Constrained(f, b), x, γ)
        @test isequal(y, expected)
        # infinite level, box that contains 0: the hard zero where the weight is positive, the clamp elsewhere
        g = Constrained(NormL1(T(Inf), w), IndBox(-l, u))
        yg = prox(g, x, γ)
        @test all(i -> iszero(yg[i]) || (w !== nothing && iszero(w[i])), eachindex(yg))
      end
    end
  end
  # a finite level that merely overflows keeps its finite meaning (R3)
  @test prox(Constrained(NormL1(1e308), IndBox(1.0, 2.0)), [1.0, 2.0], 1e308) == [1.0, 1.0]
  @test prox(Constrained(NormL1(1e300), IndBox(1.0, 2.0)), [1.0, 2.0], 1e10) == [1.0, 1.0]
  @test prox(Constrained(NormL1(1.0), IndBox(1.0, 2.0)), [1.5, 3.0], 1.0) == [1.0, 2.0]
  # families that are not decided here keep their results
  @test prox(Constrained(NormL1Nonneg(Inf), IndBox(1.0, 2.0)), [1.0, 2.0], 1.0) == [1.0, 1.0]
  @test prox(Constrained(NormL1Nonneg(1.0), IndBox(1.0, 2.0)), [1.5, 3.0], 1.0) == [1.0, 2.0]
  # the hyperplane form keeps its own message
  e = try prox(Constrained(NormL1(Inf), IndHyperplane([1.0, 1.0], 1.0)), [1.0, 2.0], 1.0); nothing catch e; e end
  @test e isa ArgumentError && e.msg == "the IndHyperplane misses the domain of NormL1"
  # the other box arguments: nonpositive-bounded and sets that are not boxes keep their constructor errors
  @test_throws ArgumentError IndBox(2.0, 1.0)
  @test_throws ArgumentError IndBox(Inf, Inf)
end

####################################
#  wrappers
####################################

@testset "wrappers" begin
  g = Constrained(NormL1(Inf), IndBox(1.0, 2.0))
  # a positive Scaled step reaches the box with the product as its step: the check runs
  for α in (2.0, 0.5, 1e-3, 1e10)
    @test raises_all(Scaled(g, α), [1.0, 2.0], 1.0)
    @test raises_all(Scaled(g, α), [1.0, 2.0], [1.0, 3.0])
  end
  @test raises_all(Scaled(Scaled(g, 2.0), 3.0), [1.0, 2.0], 1.0)
  ok = Constrained(NormL1(Inf), IndBox(-1.0, 2.0))
  check(Scaled(ok, 2.0), [1.0, 2.0], 1.0, [0.0, 0.0]; value = 0.0)
  check(Scaled(Scaled(ok, 2.0), 3.0), [NaN, 2.0], 1.0, [0.0, 0.0]; value = 0.0)
  # a block of a separable sum and a translated function: nothing is written in any block
  s = SeparableSum(g, NormL1(1.0))
  for op in (prox!, value_and_prox!)
    y = ([7.0, 7.0], [7.0])
    @test msg_of(() -> op(y, s, ([1.0, 2.0], [3.0]), 1.0)) == MSG
    @test y == ([7.0, 7.0], [7.0])
  end
  @test raises_all(Translate(g, [0.0, 0.0]), [1.0, 2.0], 1.0)
  # a feasible block keeps the hard zero
  s2 = SeparableSum(ok, NormL1(1.0))
  y = ([7.0, 7.0], [7.0])
  prox!(y, s2, ([1.0, 2.0], [3.0]), 1.0)
  @test y == ([0.0, 0.0], [2.0])
end

####################################
#  steps that do not fix an entry
####################################

@testset "a zero step does not fix an entry: the check does not decide" begin
  # γ = 0 keeps the earlier result until the maintainer rules on the zero step (decision 49, excluded
  # cases): the identity clamped into the box, and no error from this check, also per entry in an array step
  g = Constrained(NormL1(Inf), IndBox(1.0, 2.0))
  @test msg_of(() -> prox(g, [1.0, 2.0], 0.0)) != MSG
  @test msg_of(() -> prox(g, [1.0, 2.0], [0.0, 0.0])) != MSG
  @test msg_of(() -> prox(g, [1.0, 2.0], [0.0, 1.0])) == MSG
end

end # module
