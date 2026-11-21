using Test, ProximalBase, LinearAlgebra, OffsetArrays
using .TestUtils: allocations

const poisson_value = ProximalBase.value_from_predictor
const poisson_fused! = ProximalBase.value_and_gradient_from_predictor!

# The largest predictor with a finite exponential.
const POISSON_L = log(floatmax(Float64))

# A deterministic design with entries in [-1, 1].
poisson_design(n, p) = [sin(1.3i + 0.7j * i + j) for i in 1:n, j in 1:p]

# Ratios of the computed results to the bounds of the accuracy statement, relative
# to the predictor η that the loop read. The reference is a 2048-bit BigFloat
# evaluation at that predictor: a high-precision rounded value, not an exact one.
# A computed ratio of at most one passes the comparison; the reference and the
# ratio are rounded. The cases have finite intermediates, as the statement requires.
#   term:    |Δvᵢ| ≤ 4 eps (exp(ηᵢ) + |yᵢηᵢ|) + 2^-1074
#   deriv:   |Δdᵢ| ≤ 4 eps (exp(ηᵢ) + yᵢ) + 2^-1074
#   value:   the term bounds plus recursive summation, γ_{n-1} (Σᵢ|reference vᵢ| +
#            Σᵢ term boundᵢ), which bounds Σᵢ|computed vᵢ| from above
#   gradient: Σᵢ|X_ij| (derivative bound) + γ_n Σᵢ|X_ij||computed dᵢ|, plus the
#            allowance n 2^-1075. Each product X_ij*dᵢ that is subnormal or rounds to
#            zero adds at most 2^-1075, so n 2^-1075 is a conservative count that
#            the code adds in every case, also when no product is subnormal.
function poisson_ratios(X, η, y, vterms, v, d, g)
  n = length(y)
  setprecision(BigFloat, 2048) do
    u = big(eps(Float64)) / 2
    γ(k) = k * u / (1 - k * u)
    p, Y, Xb = big.(η), big.(y), big.(X)
    e = exp.(p)
    exactv, exactd = e .- Y .* p, e .- Y
    tiny = big(2.0)^-1074
    termtol = 4 * big(eps(Float64)) .* (e .+ abs.(Y .* p)) .+ tiny
    dtol = 4 * big(eps(Float64)) .* (e .+ Y) .+ tiny
    vtol = sum(termtol) + γ(max(n - 1, 0)) * (sum(abs, exactv) + sum(termtol))
    gtol = transpose(abs.(Xb)) * dtol .+ γ(n) .* (transpose(abs.(Xb)) * abs.(big.(d))) .+ n * tiny / 2
    ratio(computed, exact, tol) = Float64(maximum(abs.(big.(computed) .- exact) ./ tol; init=big(0)))
    (term=ratio(vterms, exactv, termtol), deriv=ratio(d, exactd, dtol),
     value=ratio([v], [sum(exactv)], [vtol]), gradient=ratio(g, transpose(Xb) * exactd, gtol))
  end
end

# The computed value terms, one observation at a time.
poisson_terms(y, η) = [poisson_value(PoissonLoss([y[i]], ones(1, 1)), [η[i]]) for i in eachindex(y)]

# op(args...) throws E and leaves each array of state and each array argument unchanged.
function poisson_refuses(E, state, op, args...)
  arrays = Any[a for a in args if a isa AbstractArray]
  before = deepcopy((state, arrays))
  @test_throws E op(args...)
  @test isequal((state, arrays), before)
end

function poisson_check(X, η, y)
  f = PoissonLoss(y, X)
  d, g = fill(NaN, length(y)), fill(NaN, size(X, 2))
  v = poisson_fused!(g, d, f, η)
  @test v == poisson_value(f, η)
  r = poisson_ratios(X, η, y, poisson_terms(y, η), v, d, g)
  @test r.term <= 1 && r.deriv <= 1 && r.value <= 1 && r.gradient <= 1
  r
end

@testset "Poisson regression" begin
  y6 = [0.0, 1.0, 2.0, 3.0, 2.5, 0.25]
  @testset "Agreement with the BigFloat reference" begin
    for (η, y) in (
        (zeros(6), y6),
        ([-1.5, -0.3, 0.2, 0.9, 1.7, 2.2], y6),
        ([-4.0, -2.5, -1.0, 0.0, 1.0, 3.5], [0.0, 0.0, 1.0, 4.0, 0.5, 7.0]),
        ([700.0, 650.5, -20.0, 5.0, 0.0, -700.0], [1.0, 0.0, 3.0, 2.5, 0.0, 4.0]),
        ([-300.0, -700.0, -745.0, -745.0, -745.2, -745.2, -800.0, -1000.0],
         [0.0, 2.0, 0.5, 0.0, 1.0, 0.0, 0.0, 3.0]),
        ([-1.0, 1.0, 20.0, 30.0, -30.0, 0.5], [1e-3, 1e3, 1e8, 1e12, 0.0, 1e-300]))
      check = poisson_check(poisson_design(length(y), 3), η, y)
      @test all(isfinite, values(check))
    end
    # A zero response at -745.0 has a subnormal derivative that passes through the
    # gradient: exp(-745.0) is subnormal, and the written gradient equals it.
    f = PoissonLoss([0.0], ones(1, 1))
    g, d = [NaN], [NaN]
    @test poisson_fused!(g, d, f, [-745.0]) == d[1]
    @test 0 < d[1] < floatmin(Float64) && g[1] == d[1]
    @test gradient(f, [-745.0]) == [d[1]]

    # The product X_ij*d_i is exactly 2^-1075, halfway between zero and the smallest
    # subnormal: it rounds to zero, within the bound of the accuracy statement.
    X = fill(nextfloat(0.0), 1, 1)
    f = PoissonLoss([0.5], X)
    g, d = [NaN], [NaN]
    v = poisson_fused!(g, d, f, [0.0])
    @test d == [0.5] && g == [0.0]
    r = poisson_ratios(X, [0.0], [0.5], [v], v, d, g)
    @test r.gradient <= 1 && r.deriv <= 1 && r.value <= 1

    # The last finite exponential, alone so that the value stays finite.
    @test isfinite(exp(POISSON_L))
    r = poisson_check([1.0 -0.5], [POISSON_L], [1.0])
    @test all(isfinite, values(r))

    # An offset formed by the caller enters through the predictor oracles.
    X, y, β = poisson_design(4, 2), [0.0, 2.0, 1.0, 3.0], [0.4, -0.3]
    exposure = [1.0, 2.5, 0.5, 4.0]
    offset = log.(exposure)
    f = PoissonLoss(y, X)
    η = X * β + offset
    d, g = zeros(4), zeros(2)
    v = poisson_fused!(g, d, f, η)
    r = poisson_ratios(X, η, y, poisson_terms(y, η), v, d, g)
    @test r.term <= 1 && r.deriv <= 1 && r.value <= 1 && r.gradient <= 1
    @test d ≈ exposure .* exp.(X * β) .- y rtol=1e-13
    @test g ≈ transpose(X) * d rtol=1e-13
    @test poisson_value(f, η) == v
    @test v != f(β)
  end

  @testset "Ordinary path" begin
    X, β = poisson_design(8, 3), [0.4, -0.7, 0.25]
    y = [0.0, 1.0, 2.0, 0.0, 5.0, 3.0, 0.5, 1.0]
    f = PoissonLoss(y, X)
    @test f.y === y && f.X === X
    @test issmooth(f) && isconvex(f) && !isproximable(f) && !isseparable(f)
    @test issmooth(PoissonLoss) && isconvex(PoissonLoss)
    @test_throws ArgumentError lipschitz(f)
    v = @inferred f(β)
    η = copy(f.tmp)
    @test η ≈ X * β rtol=8eps()
    gradient_only = @inferred gradient(f, β)
    @test f.tmp ≈ exp.(η) .- y rtol=1e-13
    vg, g = @inferred value_and_gradient(f, β)
    @test vg == v
    d = zeros(8)
    gcached = zeros(3)
    @test (@inferred poisson_fused!(gcached, d, f, η)) == v
    @test (@inferred poisson_value(f, η)) == v
    r = poisson_ratios(X, η, y, poisson_terms(y, η), v, d, gcached)
    @test r.term <= 1 && r.deriv <= 1 && r.value <= 1 && r.gradient <= 1
    for gcomputed in (gradient_only, g)
      r = poisson_ratios(X, η, y, poisson_terms(y, η), v, d, gcomputed)
      @test r.gradient <= 1
    end
    @test gradient!(similar(β), f, β) isa Vector{Float64}
    out = fill(NaN, 3)
    @test gradient!(out, f, β) === out
    @test value_and_gradient!(out, f, β) == v

    # Predictor calls leave the scratch and the predictor alone.
    fill!(f.tmp, NaN)
    ηcopy = copy(η)
    poisson_value(f, η)
    poisson_fused!(gcached, d, f, η)
    @test all(isnan, f.tmp) && η == ηcopy

    # Buffers reused across calls give the results of fresh buffers.
    other = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8]
    dfresh, gfresh = zeros(8), zeros(3)
    vfresh = poisson_fused!(gfresh, dfresh, f, other)
    @test poisson_fused!(gcached, d, f, other) == vfresh
    @test d == dfresh && gcached == gfresh
    @test poisson_fused!(gcached, d, f, η) == v

    @test allocations(f, β) == 0
    @test allocations(gradient!, out, f, β) == 0
    @test allocations(value_and_gradient!, out, f, β) == 0
    @test allocations(poisson_value, f, η) == 0
    @test allocations(poisson_fused!, gcached, d, f, η) == 0
  end

  @testset "Finite differences" begin
    X, β = poisson_design(10, 3), [0.3, -0.5, 0.2]
    y = [0.0, 1.0, 2.0, 0.0, 4.0, 3.0, 0.5, 1.0, 0.0, 2.0]
    f = PoissonLoss(y, X)
    f(β)
    η = copy(f.tmp)
    g = gradient(f, β)
    d = exp.(η) .- y
    h = 1e-5
    n, p = size(X)
    for j in 1:p
      step = zeros(p)
      step[j] = h
      difference = (f(β + step) - f(β - step)) / (2h)
      # Leading-order truncation h²/6 Σ exp(ηᵢ)|X_ij|³, rounding of the two values and
      # of the two predictors divided by h, and the error bound of the gradient. The factor two
      # covers exp(ηᵢ ± h X_ij) against exp(ηᵢ) and first-order treatment of the sums.
      tolerance = setprecision(BigFloat, 256) do
        u = big(eps(Float64)) / 2
        γ(k) = k * u / (1 - k * u)
        e = exp.(big.(η))
        Xj = abs.(big.(X[:, j]))
        truncation = big(h)^2 / 6 * sum(e .* Xj .^ 3)
        values = (4 + big(n) / 2) * big(eps(Float64)) * sum(e .+ abs.(big.(y) .* big.(η))) / h
        predictors = γ(p + 1) * sum((e .+ big.(y)) .* (abs.(big.(X)) * (abs.(big.(β)) .+ h))) / h
        derivative = sum(Xj .* (4 * big(eps(Float64)) .* (e .+ big.(y)))) +
                     γ(n) * sum(Xj .* abs.(big.(d)))
        Float64(2 * (truncation + values + predictors + derivative))
      end
      @test abs(g[j] - difference) <= tolerance
      @test tolerance < 1e-3
    end
  end

  @testset "Endpoint and overflow cases" begin
    @test isfinite(exp(POISSON_L)) && isinf(exp(nextfloat(POISSON_L)))
    @test isfinite(exp(709.0)) && isinf(2.6e305 * 709.0) && isfinite(2.6e305)
    @test isfinite(exp(-2.0)) && isinf(floatmax(Float64) * -2.0)
    @test isinf(exp(800.0)) && isinf(floatmax(Float64) * 800.0)
    M = ones(1, 1)
    for (η, y, v, d) in (
        (-Inf, 0.0, 0.0, 0.0), (-Inf, -0.0, 0.0, 0.0),
        (Inf, 0.0, Inf, Inf), (Inf, -0.0, Inf, Inf), (Inf, 2.0, Inf, Inf),
        (-Inf, 2.0, Inf, -2.0),
        (NaN, 0.0, NaN, NaN), (NaN, 2.0, NaN, NaN),
        (nextfloat(POISSON_L), 1.0, Inf, Inf), (800.0, 1.0, Inf, Inf),
        (709.0, 2.6e305, -Inf, exp(709.0) - 2.6e305),
        # y*η overflows to -Inf while exp(η) is finite: the value term is +Inf.
        (-2.0, floatmax(Float64), Inf, exp(-2.0) - floatmax(Float64)),
        # Both exp(η) and y*η overflow: the exponential takes precedence.
        (800.0, floatmax(Float64), Inf, Inf),
        # exp(η) overflows and y*η is finite.
        (800.0, 1e300, Inf, Inf))
      f = PoissonLoss([y], M)
      g, dd = [NaN], [NaN]
      @test isequal(poisson_value(f, [η]), v)
      @test isequal(poisson_fused!(g, dd, f, [η]), v)
      @test isequal(dd[1], d) && isequal(g[1], d)
      @test isequal(f([η]), v)
      @test isequal(value_and_gradient(f, [η]), (v, [d]))
      @test isequal(gradient(f, [η]), [d])
    end
    f = PoissonLoss([1.0], M)
    v, g = value_and_gradient(f, [POISSON_L])
    @test isfinite(v) && isfinite(g[1]) && g[1] == exp(POISSON_L) - 1.0

    # Aggregates of finite inputs that are not finite.
    f = PoissonLoss([1.0, 2.6e305], ones(2, 1))
    @test isnan(poisson_value(f, [800.0, 709.0]))
    f = PoissonLoss([1.0], [1.0 0.0])
    g, d = [0.0, 0.0], [0.0]
    poisson_fused!(g, d, f, [800.0])
    @test d == [Inf] && g[1] == Inf && isnan(g[2])
    # Infinite derivatives meet design entries of both signs in one column.
    f = PoissonLoss([1.0, 1.0], reshape([1.0, -1.0], 2, 1))
    g, d = [0.0], zeros(2)
    @test poisson_fused!(g, d, f, [800.0, 800.0]) == Inf
    @test d == [Inf, Inf] && isnan(g[1])
    # The exact sum of these terms is about 1e308, but the first partial sum of two
    # terms overflows.
    f = PoissonLoss([1e308, 1e308, 1e308], ones(3, 1))
    @test poisson_value(f, [-1.0, -1.0, 1.0]) == Inf
    # The transposed product: finite derivatives and a zero exact gradient, but each
    # product with an entry of X overflows.
    f = PoissonLoss(fill(8e307, 2), reshape([3.0, -3.0], 2, 1))
    g, d = [0.0], zeros(2)
    @test poisson_fused!(g, d, f, zeros(2)) == 2.0
    @test d == fill(-8e307, 2) && !isfinite(g[1])
    # A finite design and coefficients with an infinite product.
    f = PoissonLoss([1.0], [1e308 1e308])
    @test f([10.0, 10.0]) == Inf
  end

  @testset "Empty shapes" for (n, p) in ((0, 3), (2, 0), (0, 0))
    f = PoissonLoss(fill(2.0, n), zeros(n, p))
    @test f(zeros(p)) == n * 1.0
    @test value_and_gradient(f, zeros(p)) == (n * 1.0, zeros(p))
    g, d = ones(p), zeros(n)
    @test poisson_fused!(g, d, f, ones(n)) ≈ n * (exp(1) - 2) rtol=8eps()
    @test iszero(g)
    @test d ≈ fill(exp(1) - 2, n) rtol=8eps()
    @test poisson_value(f, ones(n)) ≈ n * (exp(1) - 2) rtol=8eps()
    @test gradient(f, zeros(p)) == zeros(p)
  end

  @testset "Validation before writes" begin
    X, y, β = poisson_design(4, 2), [0.0, 1.0, 2.5, 3.0], [0.75, -0.25]
    @test_throws DimensionMismatch PoissonLoss(ones(3), X)
    @test_throws MethodError PoissonLoss(y, X; weights=ones(4))
    @test_throws MethodError PoissonLoss(y, X; offset=ones(4))
    @test_throws MethodError PoissonLoss(Float32.(y), X)
    @test_throws MethodError PoissonLoss(y, Float32.(X))
    @test_throws MethodError PoissonLoss(Int.(round.(y)), X)
    @test_throws MethodError PoissonLoss(y, Matrix{Int}(round.(X)))
    @test_throws MethodError PoissonLoss(complex.(y), X)
    @test_throws MethodError PoissonLoss(view(y, 1:4), X)
    @test_throws MethodError PoissonLoss(y, view(X, :, 1:2))
    @test_throws Union{ArgumentError,MethodError} PoissonLoss(OffsetArray(y, 0:3), X)
    @test_throws Union{ArgumentError,MethodError} PoissonLoss(y, OffsetArray(X, 0:3, 1:2))
    for bad in (NaN, -Inf, Inf, -0.25, -1e-300)
      @test_throws ArgumentError PoissonLoss([1.0, 0.0, 0.5, bad], X)
    end
    @test PoissonLoss([-0.0, 0.0, 1e300, 2.5], X) isa PoissonLoss

    η = X * β
    f = PoissonLoss(y, X)
    g, d, scratch = fill(2.0, 2), fill(3.0, 4), fill(7.0, 4)
    copyto!(f.tmp, scratch)
    s = 2.0 * f
    # Each refusal leaves the outputs, the scratch, y, X, β, η and the array
    # arguments of the call unchanged. Each argument is built before the call.
    state = (g, d, f.tmp, y, X, β, η)
    refuse(E, op, args...) = poisson_refuses(E, state, op, args...)
    refuse(DimensionMismatch, f, [1.0])
    refuse(DimensionMismatch, s, [1.0])
    refuse(ArgumentError, s, view(f.tmp, 1:2))
    refuse(ArgumentError, f, view(f.tmp, 1:2))
    refuse(DimensionMismatch, gradient!, g, f, [1.0])
    refuse(DimensionMismatch, gradient!, d, f, β)
    refuse(DimensionMismatch, value_and_gradient!, d, f, β)
    refuse(MethodError, value_and_gradient!, [1, 2], f, β)
    refuse(MethodError, f, Float32.(β))
    refuse(MethodError, poisson_value, f, Float32.(η))
    refuse(ArgumentError, poisson_value, f, view(f.tmp, 1:4))
    refuse(DimensionMismatch, poisson_value, f, β)
    refuse(DimensionMismatch, poisson_fused!, d, d, f, η)
    refuse(DimensionMismatch, poisson_fused!, g, g, f, η)
    refuse(MethodError, poisson_fused!, g, d, f, [1, 2, 3, 4])
    refuse(MethodError, poisson_fused!, [1, 2], d, f, η)
    refuse(ArgumentError, poisson_value, f, f.tmp)
    refuse(ArgumentError, poisson_fused!, g, d, f, f.tmp)
    refuse(ArgumentError, poisson_fused!, g, η, f, η)
    refuse(ArgumentError, poisson_fused!, view(d, 2:3), d, f, η)
    refuse(ArgumentError, poisson_fused!, view(η, 1:2), d, f, η)
    refuse(ArgumentError, poisson_fused!, g, view(X, :, 1), f, η)
    refuse(ArgumentError, poisson_fused!, view(X, 1:2), d, f, η)
    refuse(ArgumentError, poisson_fused!, g, f.tmp, f, η)
    refuse(ArgumentError, poisson_fused!, view(f.tmp, 1:2), d, f, η)
    refuse(ArgumentError, poisson_fused!, g, y, f, η)
    refuse(ArgumentError, poisson_fused!, view(y, 1:2), d, f, η)
    refuse(ArgumentError, gradient!, β, f, β)
    refuse(ArgumentError, gradient!, view(y, 1:2), f, β)
    refuse(ArgumentError, gradient!, view(X, 1:2), f, β)
    refuse(ArgumentError, gradient!, view(f.tmp, 1:2), f, β)
    refuse(ArgumentError, value_and_gradient!, g, f, view(f.tmp, 1:2))
    refuse(ArgumentError, value_and_gradient!, view(y, 1:2), s, β)
    refuse(ArgumentError, gradient!, view(y, 1:2), s, β)
    refuse(ArgumentError, value_and_gradient!, view(f.tmp, 1:2), s, β)
    refuse(Union{ArgumentError,MethodError}, f, OffsetArray([0.75, -0.25], 0:1))
    refuse(Union{ArgumentError,MethodError}, gradient!, OffsetArray(zeros(2), 0:1), f, β)
    refuse(Union{ArgumentError,MethodError}, poisson_value, f, OffsetArray(η, 0:3))
    refuse(Union{ArgumentError,MethodError}, poisson_fused!, OffsetArray(zeros(2), 0:1), d, f, η)
    refuse(Union{ArgumentError,MethodError}, poisson_fused!, g, OffsetArray(zeros(4), 0:3), f, η)
    @test g == fill(2.0, 2) && d == fill(3.0, 4) && f.tmp == scratch
    @test y == [0.0, 1.0, 2.5, 3.0] && X == poisson_design(4, 2)
    @test η == X * β && β == [0.75, -0.25]
  end

  @testset "Scaled" begin
    X, β = poisson_design(5, 2), [0.3, -0.4]
    y = [0.0, 1.0, 2.0, 0.5, 3.0]
    f = PoissonLoss(y, X)
    v, g = value_and_gradient(f, β)
    s = 1.75 * f
    @test s(β) == 1.75 * v
    @test gradient(s, β) == 1.75 .* g
    @test value_and_gradient(s, β) == (1.75 * v, 1.75 .* g)
    mean = (1 / 5) * f
    @test mean(β) ≈ v / 5 rtol=8eps()
    @test gradient(mean, β) ≈ g ./ 5 rtol=8eps()
    η = X * β
    @test_throws MethodError poisson_value(s, η)
    @test_throws MethodError poisson_fused!(zeros(2), zeros(5), s, η)
  end

  @testset "Borrowed data and strided arguments" begin
    X, y, β = poisson_design(5, 2), [0.0, 1.0, 2.0, 0.5, 3.0], [0.3, -0.4]
    f = PoissonLoss(y, X)
    f(β)
    X[1] += 1.0
    y[2] = 0.125
    η = X * β
    d, g = zeros(5), zeros(2)
    v = poisson_fused!(g, d, f, η)
    r = poisson_ratios(X, η, y, poisson_terms(y, η), v, d, g)
    @test r.term <= 1 && r.deriv <= 1 && r.value <= 1 && r.gradient <= 1
    @test f(β) ≈ v rtol=8eps()
    # The edits above keep y finite and nonnegative, as the loss requires.
    ηview = view(repeat(η; inner=2), 1:2:10)
    βview = view(repeat(β; inner=2), 1:2:4)
    gview = view(fill(NaN, 4), 1:2:4)
    dview = view(fill(NaN, 10), 1:2:10)
    @test poisson_fused!(gview, dview, f, ηview) == v
    @test dview == d
    @test gview ≈ g rtol=64eps()
    @test poisson_value(f, ηview) == v
    @test f(βview) ≈ f(β) rtol=64eps()
    @test gradient!(gview, f, βview) === gview
    @test gview ≈ gradient(f, β) rtol=8eps()
    @test value_and_gradient!(gview, f, βview) ≈ f(β) rtol=64eps()
  end
end

# Include the same implementation under a local product counter.
module CountPoissonProducts
using LinearAlgebra: transpose
using ProximalBase: AbstractFunction, _check_one_based, _ls_checklength, _ls_overlap, _ls_alias_error
import ProximalBase: issmooth, isconvex
const forward = Ref(0)
const backward = Ref(0)
function mul!(dest, X, src)
  count = X isa LinearAlgebra.Transpose ? backward : forward
  count[] += 1
  LinearAlgebra.mul!(dest, X, src)
end
import LinearAlgebra
function gradient! end
function value_and_gradient! end
include(joinpath(@__DIR__, "..", "src", "functions", "poissonloss.jl"))
end

@testset "Poisson predictor product counts" begin
  C = CountPoissonProducts
  counts() = (C.forward[], C.backward[])
  X = [1.0 2.0; 3.0 4.0; 0.5 -1.0]
  C.forward[] = C.backward[] = 0
  f = C.PoissonLoss([1.0, 0.0, 2.0], X)
  β = [0.25, 0.5]
  η = X * β
  g, d = zeros(2), zeros(3)
  f(β)
  @test counts() == (1, 0)
  C.forward[] = C.backward[] = 0
  C.gradient!(g, f, β)
  @test counts() == (1, 1)
  C.forward[] = C.backward[] = 0
  v = C.value_and_gradient!(g, f, β)
  @test counts() == (1, 1)
  C.forward[] = C.backward[] = 0
  fill!(f.tmp, NaN)
  C.value_from_predictor(f, η)
  @test counts() == (0, 0)
  @test C.value_and_gradient_from_predictor!(g, d, f, η) == v
  @test counts() == (0, 1)
  @test all(isnan, f.tmp)
  for (n, p) in ((0, 2), (3, 0), (0, 0))
    C.forward[] = C.backward[] = 0
    none = C.PoissonLoss(ones(n), zeros(n, p))
    none(zeros(p))
    C.gradient!(zeros(p), none, zeros(p))
    C.value_and_gradient!(zeros(p), none, zeros(p))
    C.value_and_gradient_from_predictor!(zeros(p), zeros(n), none, zeros(n))
    @test counts() == (0, 0)
  end
end
