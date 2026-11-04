module ConstrainedNormL1InfiniteTest

# #469, Stage 3A (decision 48, E2, E3, P1, M2): NormL1 constrained on an ℓ2 ball or sphere at NaN and
# infinite arguments. The expected points are stated by hand and checked against finite BigFloat sequences
# x(M) → ∞ projected by references that share no code with the package: the soft threshold and the scaling
# for the ball, and the maximizer of ⟨z, x⟩ - γf(z) on the sphere, from the margins a_k = |x_k| - γλw_k and
# the multiplicities m_k (1 on the diagonal, 2 for a stored off-diagonal pair, M2). Where two sequences give
# different limits, the package returns NaN over the coordinates that the limit governs.

using Test
using ProximalBase
using LinearAlgebra

const BF = BigFloat

# --- references in BigFloat -----------------------------------------------------------------------

# the threshold γλw of a coordinate: infinite for a hard-fixed (infinite) weight, zero without a penalty
thr(γ, λ, w) = w == Inf ? BF(Inf) : (iszero(λ) || iszero(w) || iszero(γ)) ? BF(0) : BF(γ) * BF(λ) * BF(w)

# the proximal point of γ NormL1(λ, w) scaled onto the ball of radius r, coordinates x with multiplicities m
function ref_ball(x, λ, w, γ, r; m = ones(Int, length(x)))
  t = [thr(γ, λ, w[i]) for i in eachindex(x)]
  z = [isinf(t[i]) ? BF(0) : sign(x[i]) * max(abs(x[i]) - t[i], BF(0)) for i in eachindex(x)]
  n = sqrt(sum(m[i] * z[i]^2 for i in eachindex(z)))
  n <= r ? z : z .* (BF(r) / n)
end

# the maximizer on the sphere of radius r: u_k ∝ a_k over the positive margins, or the single coordinate
# k with the largest √m_k a_k when none is positive (the free coordinates only; hard-fixed ones are zero)
function ref_sphere(x, λ, w, γ, r; m = ones(Int, length(x)))
  t = [thr(γ, λ, w[i]) for i in eachindex(x)]
  free = [i for i in eachindex(x) if !isinf(t[i])]
  a = Dict(i => abs(x[i]) - t[i] for i in free)
  y = zeros(BF, length(x))
  pos = [i for i in free if a[i] > 0]
  if !isempty(pos)
    N = sqrt(sum(m[i] * a[i]^2 for i in pos))
    for i in pos
      y[i] = (x[i] < 0 ? -1 : 1) * BF(r) * a[i] / N
    end
  else
    k = argmax(i -> sqrt(BF(m[i])) * a[i], free)
    y[k] = (x[k] < 0 ? -1 : 1) * BF(r) / sqrt(BF(m[k]))
  end
  y
end

# the lower-triangle coordinates of a symmetric p × p matrix, and the multiplicity of each
lower(p) = [(r, c) for c in 1:p for r in c:p]
mult(r, c) = r == c ? 1 : 2

# the reference for a symmetric argument X with weights W, as a full symmetric matrix
function ref_sym(X, W, λ, γ, r, kind)
  p = size(X, 1)
  L = lower(p)
  x = [BF(X[r, c]) for (r, c) in L]
  w = [W[r, c] for (r, c) in L]
  m = [mult(r, c) for (r, c) in L]
  z = kind === :ball ? ref_ball(x, λ, w, γ, r; m) : ref_sphere(x, λ, w, γ, r; m)
  Y = zeros(BF, p, p)
  for (k, (r, c)) in enumerate(L)
    Y[r, c] = Y[c, r] = z[k]
  end
  Y
end

# the sequence point: ±Inf entries become ±M, finite entries stay
point(x, M) = BF[isinf(v) ? sign(v) * M : BF(v) for v in x]

# equal up to rounding, and NaN exactly where the other has NaN
function close_to(a, b; rtol)
  size(a) == size(b) || return false
  all(eachindex(a)) do i
    isnan(a[i]) || isnan(b[i]) ? (isnan(a[i]) && isnan(b[i])) : isapprox(a[i], b[i]; rtol, atol = rtol)
  end
end

# prox!, prox and value_and_prox! with separate and aliased outputs, for Float32 and Float64 in each role and
# at each step in γs. The value is NaN exactly for a NaN block, and equals `value` otherwise when it is given.
function check_routes(c, x, want; γs = (1.0, 1.5), value = nothing)
  wnan = any(isnan, want)
  for S in (Float64, Float32), T in (Float64, Float32), γ in γs
    rtol = S == Float32 || T == Float32 ? 1e-6 : 1e-12
    xs = S.(x)
    y = fill(T(7), length(want))
    @test prox!(y, c, xs, γ) === y
    @test close_to(y, want; rtol)
    y = fill(T(7), length(want))
    v = value_and_prox!(y, c, xs, γ)
    @test close_to(y, want; rtol)
    @test isnan(v) == wnan && v isa T
    if value !== nothing && !wnan
      @test isapprox(v, value; rtol)
    end
    @test isequal(xs, S.(x))                              # the argument is not written
    if S == T
      z = copy(xs)
      @test prox!(z, c, z, γ) === z
      @test close_to(z, want; rtol)
      z = copy(xs)
      v = value_and_prox!(z, c, z, γ)
      @test close_to(z, want; rtol)
      @test isnan(v) == wnan
    end
  end
  @test close_to(prox(c, x, γs[1]), want; rtol = 1e-12)
end

# the output is unchanged by a call that throws an ArgumentError
function unchanged_on_throw(op, c, x, γ, y)
  before = deepcopy(y)
  @test_throws ArgumentError op(y, c, x, γ)
  @test isequal(y, before)
end

# sequences are evaluated in BigFloat with 512 bits: M up to 1e50 and the sums of squares of them
setprecision(BF, 512) do

@testset "constrained NormL1 at NaN and infinite arguments (#469, Stage 3A)" begin

# the ball and the sphere of radius one, with unit or given weights
ball(λ, w = nothing) = Constrained(w === nothing ? NormL1(λ) : NormL1(λ, w), IndBallL2(1.0))
sphere(λ, w = nothing) = Constrained(w === nothing ? NormL1(λ) : NormL1(λ, w), IndSphereL2(1.0))

@testset "examples stated in the task" begin
  c = ball(1.0)
  check_routes(c, [Inf, 2.0], [1.0, 0.0]; value = 1.0)
  check_routes(c, [-Inf, 2.0], [-1.0, 0.0]; value = 1.0)
  s = sphere(1.0)
  check_routes(s, [Inf, 2.0], [1.0, 0.0]; value = 1.0)
  check_routes(s, [-Inf, 2.0], [-1.0, 0.0]; value = 1.0)
  h = sphere(1.0, [Inf, 1.0])
  check_routes(h, [Inf, 3.0], [0.0, 1.0]; value = 1.0)
  check_routes(h, [NaN, 3.0], [0.0, 1.0]; value = 1.0)
  check_routes(h, [3.0, NaN], [0.0, NaN])
  check_routes(h, [3.0, Inf], [0.0, 1.0]; value = 1.0)
  # endpoint steps
  check_routes(c, [3.0, 0.0], [0.0, 0.0]; γs = (Inf,), value = 0.0)
  check_routes(c, [NaN, 0.0], [0.0, 0.0]; γs = (Inf,), value = 0.0)
  check_routes(c, [Inf, 0.0], [NaN, NaN]; γs = (Inf,))
end

@testset "one infinite free coordinate: ±r there, at every finite step, sign and weight" begin
  for (λ, w) in ((1.0, nothing), (2.0, [1.0, 1.0, 1.0]), (1.0, [0.0, 1.0, 2.0]), (1.0, [0.0, 0.0, 0.0]))
    for k in 1:3, σ in (-1, 1)
      x = [0.5, -2.0, 3.0]
      x[k] = σ * Inf
      want = zeros(3)
      want[k] = σ * 1.0
      wk = w === nothing ? 1.0 : w[k]
      for c in (ball(λ, w), sphere(λ, w))
        check_routes(c, x, want; γs = (0.0, 1.5, 1e6), value = λ * wk)
      end
      # the finite sequence x(M) with M = 1e40 has the same limit (BigFloat reference, the sphere and the ball)
      ww = w === nothing ? ones(3) : w
      for z in (ref_ball(point(x, BF(10)^40), λ, ww, 1.5, 1), ref_sphere(point(x, BF(10)^40), λ, ww, 1.5, 1))
        @test maximum(abs, Float64.(z) .- want) < 1e-30
      end
    end
  end
end

@testset "two infinite free coordinates, or a NaN one, give NaN in the free block; sequences witness it" begin
  for c in (ball(1.0), sphere(1.0)), x in ([Inf, Inf, 0.0], [Inf, -Inf, 0.0], [Inf, 2.0, -Inf],
                                              [NaN, 0.0, 0.0], [0.0, NaN, 0.0], [Inf, NaN, 0.0])
    check_routes(c, x, fill(NaN, 3))
  end
  # the direction of two infinite entries depends on their relative growth, for the sphere and the ball alike
  Mb = BF(10)^50
  for ref in (ref_sphere, ref_ball)
    a = ref(BF[Mb, Mb], 1.0, [1.0, 1.0], 1.0, 1.0)            # x(M) = (M, M)
    b = ref(BF[Mb, 2Mb], 1.0, [1.0, 1.0], 1.0, 1.0)           # x(M) = (M, 2M)
    @test abs(a[1] - 1 / sqrt(BF(2))) < 1e-40 && abs(a[2] - 1 / sqrt(BF(2))) < 1e-40
    @test abs(b[1] - 1 / sqrt(BF(5))) < 1e-40 && abs(b[2] - 2 / sqrt(BF(5))) < 1e-40
  end
  # so the limit is not unique: the package gives the NaN block over the free outputs
  check_routes(sphere(1.0), [Inf, Inf], [NaN, NaN])
  check_routes(ball(1.0), [Inf, Inf], [NaN, NaN])
end

@testset "hard-fixed (infinite-weight) coordinates are excluded before anything else" begin
  h = sphere(1.0, [Inf, 1.0])
  # a hard-fixed coordinate is zero whatever its argument, and the free ones project as the sphere does
  check_routes(h, [Inf, 3.0], [0.0, 1.0]; value = 1.0)
  check_routes(h, [-Inf, 3.0], [0.0, 1.0]; value = 1.0)
  check_routes(h, [NaN, 3.0], [0.0, 1.0]; value = 1.0)
  check_routes(h, [3.0, -Inf], [0.0, -1.0]; value = 1.0)
  # a hard-fixed, a penalized and a free coordinate: a NaN in the free one is NaN in every free output
  hb = ball(1.0, [Inf, 1.0, 0.0])
  check_routes(hb, [NaN, NaN, 1.0], [0.0, NaN, NaN]; γs = (1.0,))
  check_routes(hb, [Inf, 0.5, 2.0], [0.0, 0.0, 1.0]; γs = (1.0,))
  # the ball with every coordinate hard-fixed is {0}
  allhard = ball(1.0, [Inf, Inf])
  check_routes(allhard, [NaN, Inf], [0.0, 0.0]; value = 0.0)
  check_routes(allhard, [-Inf, 2.0], [0.0, 0.0]; value = 0.0)
  # a sphere with every coordinate hard-fixed is infeasible: it throws, and writes nothing
  inf_sphere = sphere(1.0, [Inf, Inf])
  for x in ([1.0, 2.0], [NaN, 2.0], [Inf, 2.0])
    unchanged_on_throw(value_and_prox!, inf_sphere, x, 1.0, fill(7.0, 2))
    unchanged_on_throw(prox!, inf_sphere, x, 1.0, fill(7.0, 2))
  end
end

@testset "zero weights and zero levels leave a coordinate free" begin
  # a zero weight is free: an infinite one is the single limit, and a NaN one makes the free block NaN
  z = sphere(1.0, [0.0, 1.0])
  check_routes(z, [Inf, 3.0], [1.0, 0.0]; value = 0.0)
  check_routes(z, [NaN, 3.0], [NaN, NaN])
  check_routes(z, [3.0, NaN], [NaN, NaN])
  # a zero level: the hard-fixed coordinate stays outside the domain, the other one is free
  zl = sphere(0.0, [Inf, 1.0])
  check_routes(zl, [Inf, 3.0], [0.0, 1.0]; value = 0.0)
  check_routes(zl, [NaN, 3.0], [0.0, 1.0]; value = 0.0)
  zlb = ball(0.0, [Inf, 1.0])
  check_routes(zlb, [3.0, Inf], [0.0, 1.0]; value = 0.0)
  # a zero level without hard coordinates: no penalty, a single infinite entry is the limit
  check_routes(sphere(0.0), [Inf, 3.0], [1.0, 0.0]; value = 0.0)
end

@testset "endpoint γ = Inf: the zero-penalty part projects, a penalized NaN does not matter" begin
  s = sphere(1.0, [0.0, 1.0])
  # the penalized coordinate is zero, and the zero-penalty one projects alone onto the sphere
  check_routes(s, [3.0, 4.0], [1.0, 0.0]; γs = (Inf,), value = 0.0)
  check_routes(s, [Inf, 3.0], [1.0, 0.0]; γs = (Inf,), value = 0.0)
  check_routes(s, [NaN, 3.0], [NaN, 0.0]; γs = (Inf,))
  check_routes(s, [3.0, NaN], [1.0, 0.0]; γs = (Inf,), value = 0.0)
  # a penalized infinite argument at an infinite step is a joint limit: NaN over the free block
  check_routes(s, [3.0, Inf], [NaN, NaN]; γs = (Inf,))
  # with every free coordinate penalized there is no zero-penalty sphere: rejected before anything is written
  pen = sphere(1.0)
  for x in ([3.0, 4.0], [NaN, 4.0], [Inf, 4.0])
    unchanged_on_throw(value_and_prox!, pen, x, Inf, fill(7.0, 2))
    unchanged_on_throw(prox!, pen, x, Inf, fill(7.0, 2))
  end
  # the ball is zero there, and NaN at the joint limit
  b = ball(1.0, [1.0, 1.0])
  check_routes(b, [NaN, 0.0], [0.0, 0.0]; γs = (Inf,), value = 0.0)
  check_routes(b, [3.0, 0.0], [0.0, 0.0]; γs = (Inf,), value = 0.0)
  check_routes(b, [Inf, 0.0], [NaN, NaN]; γs = (Inf,))
end

@testset "joint limit as x and γ diverge: the sequences disagree, so the block is NaN" begin
  # ball, unit weights, x(M) = (M, 0): with γ(M) = M/2 the first coordinate survives and the limit is (1, 0);
  # with γ(M) = 2M it does not and the limit is (0, 0). The two sequences disagree.
  M = BF(10)^40
  ref_a = ref_ball(BF[M, 0], 1.0, [1.0, 1.0], M / 2, 1)
  ref_b = ref_ball(BF[M, 0], 1.0, [1.0, 1.0], 2M, 1)
  @test abs(ref_a[1] - 1) < 1e-30 && abs(ref_b[1]) < 1e-30
  b = ball(1.0, [1.0, 1.0])
  check_routes(b, [Inf, 0.0], [NaN, NaN]; γs = (Inf,))
  # at a fixed finite step the same argument has the single limit (1, 0)
  check_routes(b, [Inf, 0.0], [1.0, 0.0]; γs = (1.0,))
end

@testset "finite sequences approach the stated limits" begin
  for (c, x, want) in ((ball(1.0), [1e8, 2.0], [1.0, 0.0]),
                       (sphere(1.0, [Inf, 1.0]), [1e8, 3.0], [0.0, 1.0]),
                       (sphere(1.0, [0.0, 1.0]), [-1e8, 3.0], [-1.0, 0.0]))
    @test isapprox(prox(c, x, 1.0), want; atol = 1e-6)
  end
end

@testset "symmetric storage: a mirrored pair is one coordinate, counted twice (M2)" begin
  for kind in (:ball, :sphere)
    cc = kind === :ball ? ball(1.0) : sphere(1.0)
    # one off-diagonal pair infinite: ±r/√2 in each of its entries (the Frobenius norm counts the pair twice)
    X = zeros(3, 3)
    X[2, 1] = X[1, 2] = Inf
    want = zeros(3, 3)
    want[2, 1] = want[1, 2] = 1 / sqrt(2)
    for op in (prox!, value_and_prox!)
      y = SymmetricSparseIterate(3)
      v = op(y, cc, SymmetricSparseIterate(X), 1.0)
      @test Matrix(y) ≈ want
      op === value_and_prox! && @test v ≈ √2                 # the stored pair, its two entries counted in f
    end
    # a diagonal infinite entry stands for itself
    X = zeros(3, 3)
    X[3, 3] = -Inf
    y = SymmetricSparseIterate(3)
    value_and_prox!(y, cc, SymmetricSparseIterate(X), 1.0)
    @test Matrix(y)[3, 3] ≈ -1 && count(!iszero, Matrix(y)) == 1
  end
  # two independent infinite coordinates: NaN in every free entry
  X = zeros(3, 3)
  X[2, 1] = X[1, 2] = Inf
  X[3, 3] = Inf
  y = SymmetricSparseIterate(3)
  v = value_and_prox!(y, sphere(1.0), SymmetricSparseIterate(X), 1.0)
  @test all(isnan, Matrix(y)) && isnan(v)
  # a hard-fixed pair is excluded: zero there, and the free diagonal entry is the single limit
  W = ones(3, 3)
  W[2, 1] = W[1, 2] = Inf
  X = zeros(3, 3)
  X[2, 1] = X[1, 2] = -Inf
  X[3, 3] = 2.0
  y = SymmetricSparseIterate(3)
  value_and_prox!(y, Constrained(NormL1(1.0, W), IndSphereL2(1.0)), SymmetricSparseIterate(X), 1.0)
  @test Matrix(y) ≈ [0.0 0 0; 0 0 0; 0 0 1.0]
  # an asymmetric argument: the symmetric part of the pair (Inf, 0) is Inf, one infinite coordinate
  A = zeros(3, 3)
  A[1, 3] = Inf
  y = SymmetricSparseIterate(3)
  value_and_prox!(y, ball(1.0), A, 1.0)
  @test Matrix(y)[1, 3] ≈ 1 / sqrt(2) && Matrix(y)[3, 1] ≈ 1 / sqrt(2)
  # the BigFloat reference for a pair sequence has ±r/√2 as its limit
  M = BF(10)^30
  X = zeros(BF, 3, 3)
  X[2, 1] = X[1, 2] = M
  X[3, 3] = 1
  for kind in (:ball, :sphere)
    Y = ref_sym(X, ones(3, 3), 1.0, 1.0, 1.0, kind)
    @test abs(Y[2, 1] - 1 / sqrt(BF(2))) < 1e-20
  end
end

@testset "a NaN in a stored pair is NaN in the free block, on every storage" begin
  c = sphere(1.0, ones(3, 3))
  X = zeros(3, 3)
  X[2, 1] = X[1, 2] = NaN
  X[2, 2] = 5.0
  y = SymmetricSparseIterate(3)
  v = value_and_prox!(y, c, SymmetricSparseIterate(X), 1.0)
  @test all(isnan, Matrix(y)) && isnan(v)
  yd = zeros(3, 3)
  v = value_and_prox!(yd, c, X, 1.0)
  @test all(isnan, yd) && isnan(v)
end

@testset "storage roles and aliasing" begin
  c = ball(1.0, [1.0, 1.0, 0.0])
  x = [Inf, 2.0, 3.0]
  want = [1.0, 0.0, 0.0]
  check_routes(c, x, want; γs = (1.0,), value = 1.0)
  y = copy(x)
  v = value_and_prox!(y, c, y, 1.0)
  @test y ≈ want && v ≈ 1.0
  y = copy(x)
  prox!(y, c, y, 1.0)
  @test y ≈ want
  # sparse iterates as the output, and as an aliased argument
  y = SparseIterate(Float64, 3)
  value_and_prox!(y, c, x, 1.0)
  @test collect(y) ≈ want
  z = SparseIterate(copy(x))
  value_and_prox!(z, c, z, 1.0)
  @test collect(z) ≈ want
end

@testset "invalid steps write nothing, with a NaN or infinite argument" begin
  c = ball(1.0)
  for γ in (NaN, -1.0), x in ([Inf, 2.0], [NaN, 2.0])
    unchanged_on_throw(value_and_prox!, c, x, γ, fill(7.0, 2))
    unchanged_on_throw(prox!, c, x, γ, fill(7.0, 2))
  end
  unchanged_on_throw(prox!, sphere(1.0), [Inf, 2.0], -1.0, fill(7.0, 2))
end

@testset "a sum: an infeasible later block writes nothing into the earlier ones" begin
  s = SeparableSum(sphere(1.0, [0.0, 1.0]), sphere(1.0, [Inf, Inf]))
  ys = (fill(7.0, 2), fill(7.0, 2))
  @test_throws ArgumentError prox!(ys, s, ([Inf, 3.0], [1.0, 2.0]), 1.0)
  @test ys == (fill(7.0, 2), fill(7.0, 2))
  # when every block is feasible, each block takes its own projection
  s2 = SeparableSum(ball(1.0, [0.0, 1.0]), sphere(1.0, [1.0, 1.0]))
  ys2 = (fill(7.0, 2), fill(7.0, 2))
  prox!(ys2, s2, ([Inf, 3.0], [1.0, 2.0]), 1.0)
  @test ys2[1] ≈ [1.0, 0.0] && ys2[2] ≈ [0.0, 1.0]
end

@testset "finite arguments are unaffected, overflow included" begin
  # a finite argument whose norm overflows is not an infinite entry: the finite result stands
  c = ball(1.0)
  y = zeros(2)
  value_and_prox!(y, c, [1e200, 1e200], 1.0)
  @test y ≈ [1 / sqrt(2), 1 / sqrt(2)]
  y = zeros(Float32, 2)
  value_and_prox!(y, c, Float32[1e30, 1e30], 1.0)
  @test isapprox(y, Float32[1 / sqrt(2), 1 / sqrt(2)]; rtol = 1e-6)
  # a finite argument agrees with the BigFloat reference, on the ball and on the sphere
  x = [0.7, -1.9, 0.0, 2.4]
  w = [1.0, 0.0, 2.0, 1.0]
  for kind in (:ball, :sphere)
    cc = kind === :ball ? ball(1.0, w) : sphere(1.0, w)
    Y = kind === :ball ? ref_ball(BF.(x), 1.0, w, 1.0, 1) : ref_sphere(BF.(x), 1.0, w, 1.0, 1)
    @test isapprox(prox(cc, x, 1.0), Float64.(Y); rtol = 1e-12, atol = 1e-12)
  end
end

end # testset
end # setprecision

end # module
