module LeastSquaresPredictorTest

# The least-squares oracles at a predictor or a residual supplied by the caller (issue #915):
# value_from_predictor, value_and_gradient_from_predictor!, value_from_residual and leastsquares_data,
# in Float32 and Float64, raw and Scaled. Values and gradients are compared with 256-bit BigFloat
# references formed from the exact bits of X, Y, z (or r) and the scale as supplied.

using Test, ProximalBase, Random, LinearAlgebra, SparseArrays
const PB = ProximalBase

with256(f) = setprecision(f, BigFloat, 256)
pow2(::Type{T}, k) where {T} = ldexp(one(T), k)

mkloss(Y, X, α) = α === nothing ? LeastSquaresLoss(Y, X) : Scaled(LeastSquaresLoss(Y, X), α)
baseloss(f::LeastSquaresLoss) = f
baseloss(f::Scaled) = f.f
scalebig(α) = α === nothing ? big(1) : big(α)

# α ‖z - Y‖² / 2 and α ‖r‖² / 2 from the exact bits
function ref_value(α, z, Y)
  with256() do
    scalebig(α) * sum(abs2, big.(z) .- big.(Y)) / 2
  end
end
function ref_residual(α, r)
  with256() do
    scalebig(α) * sum(abs2, big.(r)) / 2
  end
end

# A value v of n squares against its reference ref, with u = floatmax - prevfloat(floatmax), thr = floatmax + u/2
# the round-to-nearest overflow threshold and e = (n + 8) eps(T) the relative error bound:
#   (a) |ref| (1 + e) <= floatmax: finite and within e |ref| (plus two units of the smallest subnormal where
#       |ref| < floatmin);
#   (d) between that margin and thr: finite within e |ref|, or not finite (as before the exception below);
#   (b) thr <= |ref| <= thr (1 + e), the boundary region: not finite, or finite within e |ref|;
#   (c) above that region: not finite.
# Region (b) is the maintainer-approved exception to the rule that an overflowing reference must give a non-finite
# result: the recomputation shared with SqrNormL2 does not decide on which side of thr the value lies within the
# relative error bound of thr (docs/src/numerics.md), so a reference that rounds to Inf may give floatmax there. The
# other classes keep their assertions.
function value_ok(v, ref, ::Type{T}, n) where {T}
  with256() do
    m, fm, e = abs(ref), big(floatmax(T)), (n + 8) * big(eps(T))
    thr = fm + (fm - big(prevfloat(floatmax(T)))) / 2
    close_enough = !isfinite(v) ? false : abs(big(v) - ref) <= e * m
    if m * (1 + e) <= fm
      isfinite(v) && abs(big(v) - ref) <= e * m + (m < floatmin(T) ? 2 * big(nextfloat(zero(T))) : 0)
    elseif m < thr
      !isfinite(v) || close_enough
    elseif m <= thr * (1 + e)
      !isfinite(v) || close_enough
    else
      !isfinite(v)
    end
  end
end

# The gradient g against α Xᵀ(z - Y) componentwise within (n + 8) eps α (|X|ᵀ|z - Y|)ⱼ + 2 nextfloat(0), where
# no entry of |X|ᵀ|z - Y| is subnormal-ish or near the top of the range and α times it stays in range
# (outside, the documented range limitation applies). Gives :skipped outside that range.
function gradient_check(g, α, X, zy, ::Type{T}, n) where {T}
  with256() do
    a = scalebig(α)
    Xb = big.(X)
    S = transpose(abs.(Xb)) * abs.(zy)
    gref = transpose(Xb) * zy
    lo, hi = big(floatmin(T)) / big(eps(T)), big(floatmax(T)) / 4
    all(s -> iszero(s) || (lo <= s <= hi && a * s <= hi), S) || return :skipped
    ok = all(eachindex(g)) do j
      isfinite(g[j]) && abs(big(g[j]) - a * gref[j]) <= (n + 8) * big(eps(T)) * a * S[j] + 2 * big(nextfloat(zero(T)))
    end
    ok ? :pass : :fail
  end
end

# a case of data (X, Y, z) whose residuals have the named size; z is data, not a product Xβ
function make_case(rng, ::Type{T}, kind, n, p) where {T}
  X = randn(rng, T, n, p)
  Y = randn(rng, T, n)
  z = randn(rng, T, n)
  E, e0 = exponent(floatmax(T)), exponent(floatmin(T))
  if kind === :exact
    z = copy(Y)
  elseif kind === :near
    z = Y .+ eps(T) .* randn(rng, T, n)
  elseif kind === :cancel
    Y = T(1000) .+ Y
    z = Y .+ T(0.01) .* randn(rng, T, n)
  elseif kind === :under
    s = pow2(T, T === Float32 ? -70 : -540)
    Y .*= s
    z .*= s
  elseif kind === :over
    s = pow2(T, T === Float32 ? 60 : 500)
    Y .*= s
    z .*= s
  elseif kind === :huge                      # residuals near floatmax / 4: the squares overflow
    s = pow2(T, E - 2)
    Y .*= s
    z .*= s
  elseif kind === :subnormal
    z = T.(rand(rng, 1:50, n)) .* nextfloat(zero(T))
    Y .= zero(T)
  elseif kind === :mixed                     # one large residual, the others far below it
    z .*= pow2(T, e0 ÷ 2 - 8)
    Y .*= pow2(T, e0 ÷ 2 - 8)
    z[1] = pow2(T, E ÷ 2 + 1)
  end
  X, Y, z
end
const KINDS = (:ordinary, :exact, :near, :cancel, :under, :over, :huge, :subnormal, :mixed)

# the scales of the tests: nothing is the raw loss
scales(::Type{Float32}) = Any[nothing, 1f0 / 64, 1f30, 1f-30, pow2(Float32, -140), 1e-39, 1e-44, 1e-60, 1e30, 3, 16777217, 2^53]
scales(::Type{Float64}) = Any[nothing, 1 / 64, 1e200, 1e-200, 1e-300, pow2(Float64, -1030), pow2(Float64, 1000), 3, 2^53, 2^60]

# a design whose products by X and by Xᵀ are counted, and that is a strided dense matrix
mutable struct CountMat{T} <: DenseMatrix{T}
  data::Matrix{T}
  forward::Int
  adjoint::Int
end
CountMat(data::Matrix{T}) where {T} = CountMat{T}(data, 0, 0)
Base.size(A::CountMat) = size(A.data)
Base.getindex(A::CountMat, i::Int, j::Int) = A.data[i, j]
Base.strides(A::CountMat) = strides(A.data)
Base.elsize(::Type{CountMat{T}}) where {T} = sizeof(T)
Base.cconvert(::Type{Ptr{T}}, A::CountMat{T}) where {T} = A.data
Base.unsafe_convert(::Type{Ptr{T}}, A::CountMat{T}) where {T} = Base.unsafe_convert(Ptr{T}, A.data)
Base.dataids(A::CountMat) = Base.dataids(A.data)
function LinearAlgebra.mul!(y::AbstractVector, A::CountMat, x::AbstractVector)
  A.forward += 1
  mul!(y, A.data, x)
end
function LinearAlgebra.mul!(y::AbstractVector, At::Transpose{<:Any, <:CountMat}, x::AbstractVector)
  parent(At).adjoint += 1
  mul!(y, transpose(parent(At).data), x)
end
products(A::CountMat) = (A.forward, A.adjoint)
resetcount!(A::CountMat) = (A.forward = 0; A.adjoint = 0; A)

# equal to a few units in the last place: a view and an array can be summed in different orders
close(a::T, b::T) where {T} = abs(a - b) <= 8 * eps(T) * abs(b)
closevec(a, b) = length(a) == length(b) && all(i -> close(a[i], b[i]), eachindex(a))

# allocation barriers: a call after warm calls, with the result stored so that it is not discarded
alloc_value(f, z, sink) = @allocated sink[] = PB.value_from_predictor(f, z)
alloc_residual(f, r, sink) = @allocated sink[] = PB.value_from_residual(f, r)
alloc_vg(g, d, f, z, sink) = @allocated sink[] = PB.value_and_gradient_from_predictor!(g, d, f, z)
alloc_data(f) = @allocated PB.leastsquares_data(f)
alloc_call(f, β, sink) = @allocated sink[] = f(β)
alloc_ls_vg(g, f, β, sink) = @allocated sink[] = value_and_gradient!(g, f, β)
alloc_grad(g, f, β) = @allocated gradient!(g, f, β)

# the three oracles on one input, against the references; returns the number of gradient comparisons made
function check_oracles(f, α, X, Y, z, ::Type{T}) where {T}
  n, p = size(X)
  base = baseloss(f)
  ref = ref_value(α, z, Y)
  v = PB.value_from_predictor(f, z)
  @test v isa T
  ok = value_ok(v, ref, T, n)
  ok || @info "value_from_predictor" T α n v Float64(ref)
  @test ok
  g, d = fill(T(7), p), fill(T(7), n)
  vg = PB.value_and_gradient_from_predictor!(g, d, f, z)
  @test vg isa T
  @test all(i -> d[i] === z[i] - Y[i], eachindex(d))
  count = 0
  if all(isfinite, d)
    ok = value_ok(vg, ref, T, n)
    ok || @info "value_and_gradient_from_predictor!" T α n vg Float64(ref)
    @test ok
    zy = with256() do
      big.(z) .- big.(Y)
    end
    st = gradient_check(g, α, X, zy, T, n)
    st === :skipped || (count += 1)
    st === :fail && @info "gradient" T α n g
    @test st !== :fail
  else
    @test !isfinite(vg)                               # a z - Y that overflows is not recovered with the gradient
  end
  if all(isfinite, d)
    r = copy(d)
    rr = ref_residual(α, r)
    vr = PB.value_from_residual(f, r)
    @test vr isa T
    ok = value_ok(vr, rr, T, n)
    ok || @info "value_from_residual" T α n vr Float64(rr)
    @test ok
    @test PB.value_from_residual(f, -r) === vr
  end
  count
end

@testset "accuracy against 256-bit references" begin
  for T in (Float32, Float64)
    gradients = 0
    for kind in KINDS, n in (kind in (:ordinary, :cancel) ? (1, 7, 64, 300) : (1, 7, 64)), α in scales(T)
      rng = MersenneTwister(915 + n)
      p = 6
      X, Y, z = make_case(rng, T, kind, n, p)
      f = mkloss(Y, X, α)
      gradients += check_oracles(f, α, X, Y, z, T)
    end
    @test gradients > 40                              # the gradient bound was applied, not skipped
  end
end

@testset "exact fit, zero columns and the sign of the residual" begin
  for T in (Float32, Float64), α in (nothing, T(0.25), 7)
    rng = MersenneTwister(1)
    n, p = 64, 6
    X = randn(rng, T, n, p)
    X[:, 3] .= zero(T)
    Y = randn(rng, T, n)
    f = mkloss(Y, X, α)
    g, d = fill(T(7), p), fill(T(7), n)
    @test PB.value_from_predictor(f, copy(Y)) === zero(T)
    @test PB.value_from_residual(f, zeros(T, n)) === zero(T)
    @test PB.value_and_gradient_from_predictor!(g, d, f, copy(Y)) === zero(T)
    @test all(iszero, g) && all(iszero, d)
    z = randn(rng, T, n)
    PB.value_and_gradient_from_predictor!(g, d, f, z)
    @test iszero(g[3])
    @test !iszero(g[1])
    @test PB.value_from_residual(f, z .- Y) === PB.value_from_residual(f, Y .- z)
  end
end

@testset "the squared-residual kernel entry points where the level fits" begin
  for T in (Float32, Float64), kind in KINDS
    rng = MersenneTwister(5)
    _, Y, z = make_case(rng, T, kind, 64, 3)
    for λ in (one(T), T(0.3), floatmin(T), floatmax(T) / 8)
      # with a second operand both entry points form the sum over the differences and agree to the bit
      @test PB._square_value(λ, z, Y) === PB._square_value_wide(λ, z, Y)
      # without one, SqrNormL2 sums with sum(abs2, x) and the least-squares route in one vectorizing loop: the two
      # recursive sums of the squares can differ in the last bits, and give the same class
      a, b = PB._square_value(λ, z, nothing), PB._square_value_wide(λ, z, nothing)
      @test (isfinite(a) == isfinite(b)) && (!isfinite(a) || abs(a - b) <= 72 * eps(T) * abs(b) + 2 * nextfloat(zero(T)))
    end
  end
end

@testset "the #609 inputs that Scaled(f, α)(β) misses" begin
  # data of the base probe: squares that underflow (Float64) and overflow (Float32) with a representable value
  for (T, s, αs) in ((Float64, pow2(Float64, -540), Any[1e200]),
                     (Float32, pow2(Float32, 60), Any[1f0 / 64, 1f-30, 1e-60, 1e-44]))
    rng = MersenneTwister(915)
    n, p = 64, 6
    X = randn(rng, T, n, p) .* s
    β = randn(rng, T, p)
    Y = randn(rng, T, n) .* s
    z = similar(Y)
    mul!(z, X, β)
    for α in αs
      f = Scaled(LeastSquaresLoss(Y, X), α)
      ref = ref_value(α, z, Y)
      @test isfinite(ref) && T(ref) != 0
      v = PB.value_from_predictor(f, z)
      @test isfinite(v) && value_ok(v, ref, T, n)
      d, g = similar(Y), zeros(T, p)
      vg = PB.value_and_gradient_from_predictor!(g, d, f, z)
      @test isfinite(vg) && value_ok(vg, ref, T, n)
      vr = PB.value_from_residual(f, z .- Y)
      @test isfinite(vr) && value_ok(vr, ref_residual(α, z .- Y), T, n)
    end
  end
end

@testset "the overflow threshold region" begin
  # The reviewer's examples (n = 1, X = 1, Y = 0, z a single entry): the exact value is just above the
  # round-to-nearest threshold floatmax + u/2, within the boundary region, and the shared recomputation returns
  # floatmax or Inf. The 256-bit references, rounded to the type, are Inf:
  #   Float32, α = 3.0327533f38, z = 1.498015f0:
  #     3.40282357622190324004480821361991745536e38, floatmax = 3.4028235e38, thr = 3.40282356779733661637539395458142568448e38
  #   Float64, α = 9.52950562431221e307, z = 1.942395183255378:
  #     1.797693134862315847674172902245013323878960846173726629592701784996404053128842e308,
  #     thr = 1.797693134862315807937289714053034150799341327100378269361737789804449682927648e308
  for (T, α, z) in ((Float32, 3.0327533f38, 1.498015f0), (Float64, 9.52950562431221e307, 1.942395183255378))
    f = Scaled(LeastSquaresLoss(T[0], ones(T, 1, 1)), α)
    ref = ref_value(α, [z], T[0])
    e = 9 * big(eps(T))
    thr = with256() do
      big(floatmax(T)) + (big(floatmax(T)) - big(prevfloat(floatmax(T)))) / 2
    end
    @test T(ref) == T(Inf)
    @test thr <= ref <= thr * (1 + e)                         # the boundary region
    g, d = zeros(T, 1), zeros(T, 1)
    for v in (PB.value_from_predictor(f, T[z]), PB.value_from_residual(f, T[z]),
              PB.value_and_gradient_from_predictor!(g, d, f, T[z]))
      @test v isa T
      @test v == floatmax(T) || v == T(Inf)
      @test value_ok(v, ref, T, 1)
    end
  end
  # a value built from the same construction, α chosen so that α z²/2 = t floatmax before the rounding of α, and
  # classified against the reference inside the test: below the threshold with margin (finite within the bound),
  # and clearly beyond the region (not finite)
  for (T, z) in ((Float32, 1.498015f0), (Float64, 1.942395183255378))
    e = 9 * big(eps(T))
    thr, fm = with256() do
      big(floatmax(T)) + (big(floatmax(T)) - big(prevfloat(floatmax(T)))) / 2, big(floatmax(T))
    end
    for (t, kind) in ((1 - 4e, :margin), (1 + 100 * big(eps(T)), :beyond))
      α = with256() do
        T(2 * t * fm / big(z)^2)
      end
      ref = ref_value(α, [z], T[0])
      kind === :margin && @test ref * (1 + e) <= fm
      kind === :beyond && @test ref > thr * (1 + e)
      f = Scaled(LeastSquaresLoss(T[0], ones(T, 1, 1)), α)
      g, d = zeros(T, 1), zeros(T, 1)
      for v in (PB.value_from_predictor(f, T[z]), PB.value_from_residual(f, T[z]),
                PB.value_and_gradient_from_predictor!(g, d, f, T[z]))
        @test value_ok(v, ref, T, 1)
        kind === :margin && @test isfinite(v)
        kind === :beyond && @test v == T(Inf)
      end
    end
  end
  # the rounding of a value below the smallest subnormal: the exact value is 0.75 nextfloat(0.0)
  f = Scaled(LeastSquaresLoss([0.0], ones(1, 1)), 1.5)
  z = [pow2(Float64, -537)]
  @test ref_value(1.5, z, [0.0]) == 0.75 * big(nextfloat(0.0))
  @test value_ok(PB.value_from_predictor(f, z), ref_value(1.5, z, [0.0]), Float64, 1)
  # the result is 0.0 or nextfloat(0.0) (within the subnormal allowance): Julia 1.13.3 gives nextfloat(0.0), 1.10.12 gives 0.0
  for v in (PB.value_from_predictor(f, z), PB.value_from_residual(f, z),
            PB.value_and_gradient_from_predictor!(zeros(1), zeros(1), f, z))
    @test v === 0.0 || v === nextfloat(0.0)
  end
end

@testset "the scale as supplied" begin
  # Float32 data with an integer scale that Float32 does not hold: 9 α / 2 = 75497476.5 rounds up, while
  # the scale rounded to Float32 first would give 75497472
  f = LeastSquaresLoss(Float32[0], reshape(Float32[1], 1, 1))
  z = Float32[3]
  @test PB.value_from_predictor(Scaled(f, 16777217), z) === 75497480f0
  @test PB.value_from_residual(Scaled(f, 16777217), z) === 75497480f0
  @test 9 * Float32(16777217) / 2 === 75497472f0
  g, d = zeros(Float32, 1), zeros(Float32, 1)
  @test PB.value_and_gradient_from_predictor!(g, d, Scaled(f, 16777217), z) === 75497480f0
  @test g[1] === Float32(16777217.0 * 3.0)
  # a Float64 scale with Float32 data is applied in Float64 and the result rounded once
  for α in (1e-60, 1e-44, 1e-39)
    f32 = LeastSquaresLoss(Float32[0, 0], Float32[1 0; 0 1])
    z32 = Float32[3, 4]
    @test PB.value_from_predictor(Scaled(f32, α), z32) === Float32(α * 25 / 2)
  end
  # integers: exact in Float64 are accepted, the others are refused
  f64 = LeastSquaresLoss([0.0], reshape([1.0], 1, 1))
  z64 = [3.0]
  @test PB.value_from_predictor(Scaled(f64, 2^53), z64) === pow2(Float64, 53) * 9 / 2
  @test PB.value_from_predictor(Scaled(f64, 2^53 + 2), z64) === Float64(2^53 + 2) * 9 / 2
  @test PB.value_from_predictor(Scaled(f64, UInt8(3)), z64) === 13.5
  @test PB.value_from_predictor(Scaled(f64, true), z64) === 4.5
  @test PB.value_from_predictor(Scaled(f64, 3), z64) === 13.5
  @test_throws ArgumentError PB.value_from_predictor(Scaled(f64, 2^53 + 1), z64)
  @test_throws ArgumentError PB.value_from_predictor(Scaled(f64, Int128(2)^80 + 1), z64)
  @test PB.value_from_predictor(Scaled(f64, Int128(2)^80), z64) === pow2(Float64, 80) * 9 / 2
  # a Float32 scale with Float64 data is used as it is
  @test PB.value_from_predictor(Scaled(f64, 0.1f0), z64) === Float64(0.1f0) * 9 / 2
end

@testset "scales that the oracles refuse, before anything is written" begin
  for T in (Float32, Float64)
    rng = MersenneTwister(3)
    n, p = 5, 3
    X, Y, z = randn(rng, T, n, p), randn(rng, T, n), randn(rng, T, n)
    for α in (Inf, T(Inf), 1 // 2, big(0.5), Float16(0.5), 2^53 + 1, big(2)^53 + 1)
      f = Scaled(LeastSquaresLoss(Y, X), α)
      @test f isa Scaled{<:LeastSquaresLoss}
      base = f.f
      fill!(base.tmp, T(7))
      g, d = fill(T(7), p), fill(T(7), n)
      @test_throws ArgumentError PB.value_from_predictor(f, z)
      @test_throws ArgumentError PB.value_from_residual(f, z)
      @test_throws ArgumentError PB.value_and_gradient_from_predictor!(g, d, f, z)
      @test all(==(T(7)), g) && all(==(T(7)), d) && all(==(T(7)), base.tmp)
      @test PB.leastsquares_data(f).α === α             # the accessor returns the factor as stored
    end
  end
  # the message names the scale
  f = Scaled(LeastSquaresLoss([1.0], reshape([1.0], 1, 1)), 1 // 2)
  err = try PB.value_from_predictor(f, [1.0]); nothing catch e e end
  @test err isa ArgumentError
  @test occursin("1//2", err.msg) && occursin("Rational", err.msg)
end

@testset "non-finite entries" begin
  for T in (Float32, Float64), α in (nothing, T(1) / 8, T(1e-30), 3)
    rng = MersenneTwister(11)
    n, p = 8, 3
    X, Y, z0 = randn(rng, T, n, p), randn(rng, T, n), randn(rng, T, n)
    f = mkloss(Y, X, α)
    for bad in (T(NaN), T(Inf), -T(Inf)), i in (1, n)
      z = copy(z0)
      z[i] = bad
      @test !isfinite(PB.value_from_predictor(f, z))
      @test !isfinite(PB.value_from_residual(f, z))
      g, d = zeros(T, p), zeros(T, n)
      @test !isfinite(PB.value_and_gradient_from_predictor!(g, d, f, z))
      @test !all(isfinite, g)
    end
    # a non-finite entry of Y gives a non-finite value
    Yb = copy(Y)
    Yb[2] = T(NaN)
    @test isnan(PB.value_from_predictor(mkloss(Yb, X, α), z0))
    Yb[2] = T(Inf)
    @test !isfinite(PB.value_from_predictor(mkloss(Yb, X, α), z0))
    # a non-finite entry of the design reaches the gradient components it multiplies and not the value
    Xb = copy(X)
    Xb[3, 2] = T(NaN)
    fb = mkloss(Y, Xb, α)
    g, d = zeros(T, p), zeros(T, n)
    v = PB.value_and_gradient_from_predictor!(g, d, fb, z0)
    @test v === PB.value_and_gradient_from_predictor!(similar(g), similar(d), mkloss(Y, X, α), z0)
    @test isfinite(g[1]) && isnan(g[2]) && isfinite(g[3])
    @test PB.value_from_predictor(fb, z0) === PB.value_from_predictor(mkloss(Y, X, α), z0)
  end
end

@testset "range limits of the gradient, and an overflowing z - Y" begin
  # X'd overflows before the scale: the gradient is Inf where the exact one is 8
  X = fill(pow2(Float64, 1023), 1, 1)
  f = Scaled(LeastSquaresLoss([0.0], X), pow2(Float64, -1022))
  g, d = zeros(1), zeros(1)
  v = PB.value_and_gradient_from_predictor!(g, d, f, [4.0])
  @test g[1] == Inf
  @test v === pow2(Float64, -1019)
  @test PB.value_from_predictor(f, [4.0]) === pow2(Float64, -1019)
  # X'd underflows before the scale: the gradient is 0 where the exact one is about 2.4e-181
  X = fill(pow2(Float64, -600), 1, 1)
  f = Scaled(LeastSquaresLoss([0.0], X), pow2(Float64, 600))
  v = PB.value_and_gradient_from_predictor!(g, d, f, [pow2(Float64, -600)])
  @test g[1] === 0.0
  @test v === pow2(Float64, -601)                        # the value is recovered from d
  @test PB.value_from_residual(f, [pow2(Float64, -600)]) === pow2(Float64, -601)
  # z - Y overflows with finite entries: the value alone is recovered, the value with the gradient is not
  for (T, α) in ((Float64, pow2(Float64, -1030)), (Float32, pow2(Float32, -130)))
    f = Scaled(LeastSquaresLoss([-floatmax(T)], fill(one(T), 1, 1)), α)
    z = [floatmax(T)]
    ref = ref_value(α, z, [-floatmax(T)])
    v = PB.value_from_predictor(f, z)
    @test isfinite(v) && value_ok(v, ref, T, 1)
    g, d = zeros(T, 1), zeros(T, 1)
    @test !isfinite(PB.value_and_gradient_from_predictor!(g, d, f, z))
    @test isinf(d[1])
  end
  # the gradient after an overflowing z - Y: the entries are non-finite for p > 0 with the BLAS of the tests, a zero column
  # of X included (0 * Inf is NaN), and the gradient is empty for p = 0
  for T in (Float32, Float64), α in (nothing, T(2))
    Xz = T[1 0 3; 0 0 0]
    f = mkloss(T[-floatmax(T), 0], Xz, α)
    g, d = zeros(T, 3), zeros(T, 2)
    @test !isfinite(PB.value_and_gradient_from_predictor!(g, d, f, T[floatmax(T), 0]))
    @test isinf(d[1]) && d[2] == 0
    @test !any(isfinite, g)
    f0 = mkloss(T[-floatmax(T), 0], zeros(T, 2, 0), α)
    @test !isfinite(PB.value_and_gradient_from_predictor!(T[], zeros(T, 2), f0, T[floatmax(T), 0]))
  end
  # the value that is not representable is not finite, and the one that underflows is zero
  f = Scaled(LeastSquaresLoss([0.0], fill(1.0, 1, 1)), 1e300)
  @test PB.value_from_predictor(f, [1e10]) === Inf
  f = Scaled(LeastSquaresLoss([0.0], fill(1.0, 1, 1)), 1e-300)
  @test PB.value_from_predictor(f, [1e-30]) === 0.0
end

@testset "consistency with f(β), value_and_gradient! and gradient!" begin
  for T in (Float32, Float64)
    rng = MersenneTwister(21)
    n, p = 64, 6
    X, Y, β = randn(rng, T, n, p), randn(rng, T, n), randn(rng, T, p)
    z = similar(Y)
    mul!(z, X, β)
    f = LeastSquaresLoss(Y, X)
    @test f(β) === PB.value_from_predictor(f, z)
    g1, g2, d = similar(β), similar(β), similar(Y)
    v1 = value_and_gradient!(g1, f, β)
    v2 = PB.value_and_gradient_from_predictor!(g2, d, f, z)
    @test v1 === v2
    @test all(g1 .=== g2)
    @test all(gradient!(similar(β), f, β) .=== g1)
    @test PB.value_from_residual(f, d) === v1
    # a scaled loss: the values agree to the rounding of the product, the gradient to α
    for α in (T(0.3), 3, 1e-3)
      s = Scaled(f, α)
      ref = ref_value(α, z, Y)
      @test value_ok(s(β), ref, T, n) && value_ok(PB.value_from_predictor(s, z), ref, T, n)
      gs, ds = similar(β), similar(Y)
      vs = PB.value_and_gradient_from_predictor!(gs, ds, s, z)
      @test value_ok(vs, ref, T, n)
      gl = similar(β)
      value_and_gradient!(gl, s, β)
      @test all(j -> abs(gs[j] - gl[j]) <= 4 * eps(T) * abs(gl[j]) + floatmin(T), eachindex(gs))
    end
  end
  # where Xβ overflows, f(β) recomputes it from β and the predictor oracle does not
  for T in (Float32, Float64)
    h = exponent(floatmax(T)) ÷ 2
    big2 = ldexp(one(T), h + 4)
    X = T[big2 big2; 1 1]
    Y = T[3, -2]
    β = T[big2, -big2]
    f = LeastSquaresLoss(Y, X)
    z = similar(Y)
    mul!(z, X, β)
    @test !all(isfinite, z)
    @test f(β) == T(6.5)
    @test value_and_gradient!(similar(β), f, β) == T(6.5)
    @test !isfinite(PB.value_from_predictor(f, z))
    @test !isfinite(PB.value_and_gradient_from_predictor!(similar(β), similar(Y), f, z))
    @test !isfinite(PB.value_from_residual(f, z .- Y))
    s = Scaled(f, T(0.5))
    @test !isfinite(PB.value_from_predictor(s, z))
  end
end

@testset "the scratch of the loss is neither read nor written" begin
  for T in (Float32, Float64), α in (nothing, T(2), 5)
    rng = MersenneTwister(31)
    n, p = 20, 4
    X, Y, z = randn(rng, T, n, p), randn(rng, T, n), randn(rng, T, n)
    clean = mkloss(Y, X, α)
    f = mkloss(Y, X, α)
    base = baseloss(f)
    fill!(base.tmp, T(NaN))
    g, d = zeros(T, p), zeros(T, n)
    g0, d0 = zeros(T, p), zeros(T, n)
    @test PB.value_from_predictor(f, z) === PB.value_from_predictor(clean, z)
    @test PB.value_from_residual(f, z) === PB.value_from_residual(clean, z)
    @test PB.value_and_gradient_from_predictor!(g, d, f, z) === PB.value_and_gradient_from_predictor!(g0, d0, clean, z)
    @test all(g .=== g0) && all(d .=== d0)
    @test all(isnan, base.tmp)
    # z or r may be a view of the scratch, which the call then reads and does not write
    copyto!(base.tmp, z)
    zv = view(base.tmp, 1:n)
    @test close(PB.value_from_predictor(f, zv), PB.value_from_predictor(clean, z))
    @test close(PB.value_from_residual(f, zv), PB.value_from_residual(clean, z))
    @test close(PB.value_and_gradient_from_predictor!(g, d, f, zv), PB.value_and_gradient_from_predictor!(g0, d0, clean, z))
    @test closevec(g, g0) && d == d0
    @test base.tmp == z
    # a design full of NaN does not enter the values
    fnan = mkloss(Y, fill(T(NaN), n, p), α)
    @test PB.value_from_predictor(fnan, z) === PB.value_from_predictor(clean, z)
    @test PB.value_from_residual(fnan, z) === PB.value_from_residual(clean, z)
  end
end

@testset "products and reads of the design" begin
  for T in (Float32, Float64), α in (nothing, T(0.5))
    rng = MersenneTwister(41)
    n, p = 30, 5
    Xd, Y, z = randn(rng, T, n, p), randn(rng, T, n), randn(rng, T, n)
    X = CountMat(Xd)
    @test X isa StridedMatrix{T}
    f = mkloss(Y, X, α)
    clean = mkloss(Y, Xd, α)
    g, d = zeros(T, p), zeros(T, n)
    PB.value_from_predictor(f, z)
    PB.value_from_residual(f, z)
    @test products(X) == (0, 0)
    v = PB.value_and_gradient_from_predictor!(g, d, f, z)
    @test products(X) == (0, 1)
    g0, d0 = zeros(T, p), zeros(T, n)
    @test v === PB.value_and_gradient_from_predictor!(g0, d0, clean, z)
    @test all(g .=== g0)
    # the methods that form Xβ keep their products
    β = randn(rng, T, p)
    fl = LeastSquaresLoss(Y, X)
    resetcount!(X)
    fl(β)
    @test products(X) == (1, 0)
    resetcount!(X)
    value_and_gradient!(g, fl, β)
    @test products(X) == (1, 1)
    resetcount!(X)
    gradient!(g, fl, β)
    @test products(X) == (1, 1)
  end
end

@testset "dimension errors, and errors leave the buffers unchanged" begin
  for T in (Float32, Float64), α in (nothing, T(2))
    rng = MersenneTwister(51)
    n, p = 6, 3
    X, Y, z = randn(rng, T, n, p), randn(rng, T, n), randn(rng, T, n)
    f = mkloss(Y, X, α)
    base = baseloss(f)
    g, d = fill(T(7), p), fill(T(7), n)
    fill!(base.tmp, T(7))
    unchanged(bufs...) = all(b -> all(==(T(7)), b), bufs) && all(==(T(7)), base.tmp)
    @test_throws DimensionMismatch PB.value_from_predictor(f, zeros(T, n + 1))
    @test_throws DimensionMismatch PB.value_from_predictor(f, zeros(T, n - 1))
    # the predictor's and the residual's axes are checked against Y, also under an infinite scale (checked after the sum)
    finf = Scaled(LeastSquaresLoss(Y, X), T(Inf))
    @test_throws DimensionMismatch PB.value_from_predictor(finf, zeros(T, n + 1))
    @test_throws DimensionMismatch PB.value_from_residual(finf, zeros(T, n - 1))
    @test_throws DimensionMismatch PB.value_from_residual(f, zeros(T, n + 1))
    @test_throws DimensionMismatch PB.value_and_gradient_from_predictor!(g, d, f, zeros(T, n + 1))
    @test unchanged(g, d)
    gl, dl = fill(T(7), p + 1), fill(T(7), n)
    @test_throws DimensionMismatch PB.value_and_gradient_from_predictor!(gl, dl, f, z)
    @test unchanged(gl, dl)
    gs, ds = fill(T(7), p - 1), fill(T(7), n + 1)
    @test_throws DimensionMismatch PB.value_and_gradient_from_predictor!(g, ds, f, z)
    @test_throws DimensionMismatch PB.value_and_gradient_from_predictor!(gs, d, f, z)
    @test unchanged(g, d, gs, ds)
  end
end

# the call with views of one buffer accepted or refused; the outputs and the scratch are unchanged on an error
function refused(g, d, f, z; err = ArgumentError)
  base = baseloss(f)
  before = (copy(g), copy(d), copy(base.tmp), copy(base.X), copy(base.Y), copy(z))
  @test_throws err PB.value_and_gradient_from_predictor!(g, d, f, z)
  after = (g, d, base.tmp, base.X, base.Y, z)
  @test all(i -> before[i] == after[i], 1:6)
end
function accepted(g, d, f, z)
  base = baseloss(f)
  X, Y = copy(base.X), copy(base.Y)
  clean = mkloss(Y, X, f isa Scaled ? f.α : nothing)
  z0 = copy(z)
  g0, d0 = similar(g), similar(d)
  v0 = PB.value_and_gradient_from_predictor!(g0, d0, clean, z0)
  v = PB.value_and_gradient_from_predictor!(g, d, f, z)
  @test close(v, v0)
  @test closevec(g, g0) && d == d0
  @test z == z0 && base.X == X && base.Y == Y
end

@testset "overlapping buffers" begin
  for (T, α) in ((Float32, nothing), (Float64, 2.0))
    rng = MersenneTwister(61)
    # the outputs against each other (n = p)
    n = p = 4
    Xs, Ys, zs = randn(rng, T, n, p), randn(rng, T, n), randn(rng, T, n)
    f = mkloss(Ys, Xs, α)
    both = fill(T(7), n)
    refused(both, both, f, zs)
    buf = fill(T(7), 12)
    refused(view(buf, 1:4), view(buf, 4:7), f, zs)             # partial overlap
    refused(view(buf, 4:7), view(buf, 1:4), f, zs)
    refused(view(buf, 1:2:7), view(buf, 3:2:9), f, zs)          # strided, overlapping
    accepted(view(buf, 1:4), view(buf, 5:8), f, zs)             # disjoint ranges
    accepted(view(buf, 1:2:7), view(buf, 2:2:8), f, zs)         # interleaved
    mat = fill(T(7), 4, 2)
    accepted(view(mat, :, 1), view(mat, :, 2), f, zs)           # separate columns
    refused(view(mat, :, 1), view(mat, :, 1), f, zs)
    # conservative: a view of the memory against a view of a reshape of it is refused although disjoint
    buf8 = fill(T(7), 8)
    refused(view(reshape(buf8, 4, 2), :, 1), view(buf8, 5:8), f, zs)

    # against the design, a plain matrix and a view of one
    n, p = 5, 3
    X, Y, z = randn(rng, T, n, p), randn(rng, T, n), randn(rng, T, n)
    f = mkloss(Y, X, α)
    g, d = fill(T(7), p), fill(T(7), n)
    refused(g, view(vec(X), 1:n), f, z)
    refused(view(vec(X), 1:p), d, f, z)
    Xbuf = randn(rng, T, n, 8)
    Xv = view(Xbuf, :, 1:3)
    fv = mkloss(Y, Xv, α)
    refused(g, view(Xbuf, :, 2), fv, z)
    refused(view(Xbuf, 1:3, 2), d, fv, z)
    accepted(g, view(Xbuf, :, 5), fv, z)
    accepted(view(Xbuf, 1:3, 6), d, fv, z)
    # against the response
    Ybuf = randn(rng, T, 2n)
    Yv = view(Ybuf, 1:n)
    fy = mkloss(Yv, X, α)
    refused(g, view(Ybuf, 3:n+2), fy, z)
    refused(view(Ybuf, n:n+p-1), d, fy, z)
    accepted(g, view(Ybuf, n+1:2n), fy, z)
    accepted(view(Ybuf, n+1:n+p), d, fy, z)
    refused(g, Yv, fy, z)                                          # the response itself
    # against the predictor
    zbuf = randn(rng, T, 2n)
    zv = view(zbuf, 1:n)
    refused(g, view(zbuf, n:2n-1), f, zv)
    refused(view(zbuf, n:n+p-1), d, f, zv)
    accepted(g, view(zbuf, n+1:2n), f, zv)
    accepted(view(zbuf, n+1:n+p), d, f, zv)
    refused(g, zv, f, zv)
    zz = copy(z)
    refused(g, zz, f, zz)
    # against the scratch of the loss, which also a view of it as a whole refuses
    base = baseloss(f)
    refused(g, view(base.tmp, 1:n), f, z)
    refused(view(base.tmp, 1:p), d, f, z)
    refused(g, base.tmp, f, z)
  end
end

@testset "empty designs" begin
  for T in (Float32, Float64), α in (nothing, T(2), 3)
    # no observations
    f = mkloss(T[], zeros(T, 0, 3), α)
    g, d = fill(T(7), 3), T[]
    @test PB.value_from_predictor(f, T[]) === zero(T)
    @test PB.value_from_residual(f, T[]) === zero(T)
    @test PB.value_and_gradient_from_predictor!(g, d, f, T[]) === zero(T)
    @test all(iszero, g)
    # an empty view of the scratch of the loss is not counted as overlapping
    @test PB.value_and_gradient_from_predictor!(fill(T(7), 3), view(baseloss(f).tmp, :), f, T[]) === zero(T)
    # no coefficients
    Y = T[1, 2, 3, 4]
    f = mkloss(Y, zeros(T, 4, 0), α)
    a = α === nothing ? 1 : α
    z = zeros(T, 4)
    expected = T(a * 15)
    @test PB.value_from_predictor(f, z) === expected
    @test PB.value_from_residual(f, z .- Y) === expected
    d = similar(Y)
    @test PB.value_and_gradient_from_predictor!(T[], d, f, z) === expected
    @test d == -Y
  end
end

@testset "reentrancy with distinct buffers" begin
  for T in (Float32, Float64), α in (nothing, T(0.5), 3)
    rng = MersenneTwister(71)
    n, p = 64, 6
    X, Y = randn(rng, T, n, p), randn(rng, T, n)
    f = mkloss(Y, X, α)
    base = baseloss(f)
    zs = [randn(rng, T, n) for _ in 1:8]
    serial = map(zs) do z
      g, d = zeros(T, p), zeros(T, n)
      v = PB.value_and_gradient_from_predictor!(g, d, f, z)
      (v, g, d, PB.value_from_predictor(f, z), PB.value_from_residual(f, d))
    end
    fill!(base.tmp, T(NaN))
    results = Vector{Any}(undef, length(zs))
    @sync for k in eachindex(zs)
      Threads.@spawn begin
        g, d = zeros(T, p), zeros(T, n)
        out = nothing
        for _ in 1:20
          v = PB.value_and_gradient_from_predictor!(g, d, f, zs[k])
          out = (v, copy(g), copy(d), PB.value_from_predictor(f, zs[k]), PB.value_from_residual(f, d))
          yield()
        end
        results[k] = out
      end
    end
    for k in eachindex(zs)
      @test results[k][1] === serial[k][1] && results[k][4] === serial[k][4] && results[k][5] === serial[k][5]
      @test all(results[k][2] .=== serial[k][2]) && all(results[k][3] .=== serial[k][3])
    end
    @test all(isnan, base.tmp)
  end
end

@testset "the accessor" begin
  for T in (Float32, Float64)
    X, Y = randn(MersenneTwister(81), T, 5, 3), randn(MersenneTwister(82), T, 5)
    f = LeastSquaresLoss(Y, X)
    data = PB.leastsquares_data(f)
    @test keys(data) == (:Y, :X, :α)
    @test data.Y === Y && data.X === X
    @test data.α === one(T)
    for α in (T(0.5), 0.25, 3, 2^53 + 1, 1 // 2, Float16(0.5), Inf, big(0.5))
      s = Scaled(f, α)
      data = PB.leastsquares_data(s)
      @test data.Y === Y && data.X === X
      @test data.α === α && typeof(data.α) === typeof(α)
    end
    # any LeastSquaresLoss: a matrix response, a sparse design
    fm = LeastSquaresLoss(randn(MersenneTwister(83), T, 5, 2), X)
    @test PB.leastsquares_data(fm).Y === fm.Y && PB.leastsquares_data(fm).α === one(T)
    @test PB.leastsquares_data(Scaled(fm, 0.5)).α === 0.5
    @test_throws MethodError PB.leastsquares_data(SqrNormL2(T(2)))
    @test_throws MethodError PB.leastsquares_data(Scaled(SqrNormL2(T(2)), 0.5))
    nested = Scaled(Scaled(f, 1e200), 1e200)
    @test nested isa Scaled{<:Scaled}
    @test_throws MethodError PB.leastsquares_data(nested)
  end
end

@testset "functions outside the domain" begin
  for T in (Float32, Float64)
    rng = MersenneTwister(91)
    n, p = 5, 3
    X, Y, z = randn(rng, T, n, p), randn(rng, T, n), randn(rng, T, n)
    f = LeastSquaresLoss(Y, X)
    g, d = zeros(T, p), zeros(T, n)
    # a matrix response
    fm = LeastSquaresLoss(randn(rng, T, n, 2), X)
    @test_throws MethodError PB.value_from_predictor(fm, z)
    @test_throws MethodError PB.value_from_residual(fm, z)
    @test_throws MethodError PB.value_and_gradient_from_predictor!(g, d, fm, z)
    @test_throws MethodError PB.value_from_predictor(Scaled(fm, 0.5), z)
    # a nested Scaled, which exists where the product of the factors is not a normal number
    nested = Scaled(Scaled(f, 1e200), 1e200)
    @test nested isa Scaled{<:Scaled}
    @test_throws MethodError PB.value_from_predictor(nested, z)
    @test_throws MethodError PB.value_from_residual(nested, z)
    @test_throws MethodError PB.value_and_gradient_from_predictor!(g, d, nested, z)
    # a non-strided design, an integer response, another element type, mixed types
    @test_throws MethodError PB.value_from_predictor(LeastSquaresLoss(Y, sparse(X)), z)
    @test_throws MethodError PB.value_from_predictor(LeastSquaresLoss(Y, Diagonal(ones(T, n))), z)
    @test_throws MethodError PB.value_from_predictor(LeastSquaresLoss(round.(Int, 10Y), X), z)
    @test_throws MethodError PB.value_from_predictor(LeastSquaresLoss(Float16.(Y), Float16.(X)), Float16.(z))
    other = T === Float32 ? Float64 : Float32
    @test_throws MethodError PB.value_from_predictor(f, other.(z))
    @test_throws MethodError PB.value_from_residual(f, other.(z))
    @test_throws MethodError PB.value_and_gradient_from_predictor!(other.(g), d, f, z)
    @test_throws MethodError PB.value_and_gradient_from_predictor!(g, other.(d), f, z)
    # a function that is not a least-squares loss, and a non-strided predictor
    @test_throws MethodError PB.value_from_predictor(SqrNormL2(T(2)), z)
    @test_throws MethodError PB.value_from_predictor(Scaled(SqrNormL2(T(2)), 2), z)
    @test_throws MethodError PB.value_from_predictor(f, 1:n)
    @test_throws MethodError PB.value_from_predictor(f, view(z, [1, 3, 2, 5, 4]))
    @test_throws MethodError PB.value_from_predictor(f, [z z])
  end
end

@testset "names are public, not exported" begin
  for name in (:value_from_predictor, :value_and_gradient_from_predictor!, :value_from_residual, :leastsquares_data)
    @test !Base.isexported(PB, name)
    @test isdefined(PB, name)
    VERSION >= v"1.11" && @test Base.ispublic(PB, name)
    @test haskey(Base.Docs.meta(PB), Base.Docs.Binding(PB, name))              # a docstring is attached
  end
end

@testset "the ordinary path of the least-squares route at its acceptance boundary" begin
  # For strided Float32 and Float64 vectors the route keeps the ordinary product α s / 2 when K <= s <= floatmax(T),
  # K = 2^63 floatmin(T)/eps(T), and, for a Scaled, α s <= floatmax; otherwise it runs the kernel's test (s >= n
  # floatmin(T)/eps(T) and α s / 2 finite) and recomputation on s. The inputs below put s near n floatmin(T)/eps(T)
  # and K (squares of rounded square roots and their neighbours, which bracket the two bounds without hitting them
  # exactly in every case), near floatmax, and make it Inf or NaN. On each of them the oracles and the methods of the
  # loss must give the kernel's result on the same sum, and agree with the 256-bit reference. The test of the bounds
  # on s is checked directly at K, floatmax and their neighbours.
  for T in (Float32, Float64)
    E = exponent(floatmax(T))
    K = T(0x1p63) * (floatmin(T) / eps(T))
    @test K == pow2(T, 64 + (precision(T) - 1) - exponent(floatmax(T)))
    @test PB._ls_ordinary_sum(K) && PB._ls_ordinary_sum(nextfloat(K)) && !PB._ls_ordinary_sum(prevfloat(K))
    @test PB._ls_ordinary_sum(floatmax(T)) && PB._ls_ordinary_sum(prevfloat(floatmax(T)))
    @test !PB._ls_ordinary_sum(nextfloat(floatmax(T))) && !PB._ls_ordinary_sum(T(NaN)) && !PB._ls_ordinary_sum(-K)
    @test !PB._ls_ordinary_sum(zero(T)) && !PB._ls_ordinary_sum(nextfloat(zero(T))) && !PB._ls_ordinary_sum(floatmin(T))
    inputs = Vector{T}[]
    for n in (1, 64), t in (n * (floatmin(T) / eps(T)), K)
      c = sqrt(t / n)
      for u in (prevfloat(c, 2), prevfloat(c), c, nextfloat(c), nextfloat(c, 2))
        push!(inputs, fill(u, n))
      end
    end
    h = pow2(T, (E - 1) ÷ 2)                          # s = 2 h^2 = 2^E, below floatmax; 2 s and 4 s are not
    append!(inputs, (T[h, h], T[h, nextfloat(h)], T[floatmax(T)], T[floatmax(T), one(T)], T[one(T), T(NaN)],
                     T[one(T), T(Inf)], T[T(Inf), T(-Inf)], zeros(T, 3), T[nextfloat(zero(T)), zero(T)]))
    wide = T === Float32 ? Any[1e-60, 1e30, 2.0^-150] : Any[]
    for r in inputs, α in Any[nothing, T(0.5), T(2), T(4), pow2(T, -E), pow2(T, E), 3, wide...]
      n = length(r)
      Y = zeros(T, n)
      X = Matrix{T}(I, n, n)
      f = mkloss(Y, X, α)
      a = α === nothing ? one(T) : PB._ls_base_scale(f)[2]
      kr = PB._square_value_level(a, a, r, nothing, PB._sumabs2_loop(r), true)
      kp = PB._square_value_level(a, a, r, Y, PB._ls_sumabs2_diff(r, Y), true)
      vr, vp = PB.value_from_residual(f, r), PB.value_from_predictor(f, r)
      g, d = zeros(T, n), zeros(T, n)
      vg = PB.value_and_gradient_from_predictor!(g, d, f, r)
      @test isequal(vr, kr) && isequal(vp, kp) && isequal(vg, kr)
      ok = value_ok(vr, ref_residual(α, r), T, n) && value_ok(vp, ref_value(α, r, Y), T, n)
      ok || @info "boundary input" T α r vr vp
      @test ok
      if α === nothing && all(isfinite, r)
        # the methods of the loss: for finite r, X = I gives Xβ = r (products by one and zero, sums with zeros), so that
        # f(β) and value_and_gradient! see the same sums
        @test isequal(f(r), vp)
        @test isequal(value_and_gradient!(similar(r), f, r), vr)
      end
    end
    # an infinite scale: the value oracles check it on the exceptional path, which the call reaches for these sums
    # (α s is Inf or NaN, or s is below the bound of the ordinary path)
    for r in (ones(T, 4), zeros(T, 4), T[1, NaN], T[floatmax(T), 1])
      f = Scaled(LeastSquaresLoss(zeros(T, length(r)), Matrix{T}(I, length(r), length(r))), T(Inf))
      @test_throws ArgumentError PB.value_from_residual(f, r)
      @test_throws ArgumentError PB.value_from_predictor(f, r)
    end
    # below the overflow threshold, outside the margin: the result is the finite value, four units below floatmax
    f4 = Scaled(LeastSquaresLoss(T[0], ones(T, 1, 1)), prevfloat(floatmax(T) / 2, 4))
    @test PB.value_from_predictor(f4, T[2]) === prevfloat(floatmax(T), 4)
    @test PB.value_from_residual(f4, T[2]) === prevfloat(floatmax(T), 4)
  end
  # Float32 data with a Float64 scale: α s / 2 is formed in Float64 and rounds to 0f0 below the range of Float32
  f = Scaled(LeastSquaresLoss(zeros(Float32, 4), Matrix{Float32}(I, 4, 4)), 1e-60)
  @test PB.value_from_residual(f, ones(Float32, 4)) === 0.0f0
  @test PB.value_from_predictor(f, ones(Float32, 4)) === 0.0f0
end

@testset "offset axes of the predictor, the residual and the response" begin
  # A vector with other axes than the response is a DimensionMismatch, also where the lengths agree (an offset view of
  # length n); vectors with the same axes, offset or not, give the value of the same entries in ordinary vectors.
  for T in (Float32, Float64)
    buf = T[100, 1, 2, 3, 4]
    Yoff = view(buf, Base.IdentityUnitRange(2:5))            # axes 2:5, entries 1, 2, 3, 4
    @test Yoff isa StridedVector{T} && axes(Yoff, 1) == 2:5
    f = LeastSquaresLoss(Yoff, ones(T, 4, 1))
    @test_throws DimensionMismatch f(T[0])                    # the scratch Xβ is a 1-based vector
    @test_throws DimensionMismatch value_and_gradient!(zeros(T, 1), f, T[0])
    Y = T[1, 2, 3, 4]
    X = Matrix{T}(I, 4, 4)
    zpar = T[10, 2, 4, 6, 8, 20]
    zoff = view(zpar, Base.IdentityUnitRange(2:5))           # axes 2:5, entries 2, 4, 6, 8
    for α in (nothing, T(2))
      g = mkloss(Y, X, α)
      @test_throws DimensionMismatch PB.value_from_predictor(g, zoff)
      @test_throws DimensionMismatch PB.value_from_residual(g, zoff)
      gd, dd = fill(T(7), 4), fill(T(7), 4)
      @test_throws DimensionMismatch PB.value_and_gradient_from_predictor!(gd, dd, g, zoff)
      @test all(==(T(7)), gd) && all(==(T(7)), dd)
      # the same axes: a loss with the offset response and offset predictor and residual
      h = mkloss(Yoff, X, α)
      a = α === nothing ? 1 : 2
      z1 = T[2, 4, 6, 8]
      @test PB.value_from_predictor(h, zoff) == T(a * sum(abs2, z1 .- Y) / 2) == PB.value_from_predictor(g, z1)
      roff = view(T[0, 1, 2, 3, 4, 0], Base.IdentityUnitRange(2:5))
      @test PB.value_from_residual(h, roff) == T(a * 30 / 2)
      gd, dd = fill(T(7), 4), fill(T(7), 4)
      @test_throws DimensionMismatch PB.value_and_gradient_from_predictor!(gd, dd, h, zoff)
      @test all(==(T(7)), gd) && all(==(T(7)), dd)
      # offset outputs with the axes of the response: refused before d is written (the product with Xᵀ refuses them)
      doff = view(fill(T(7), 6), Base.IdentityUnitRange(2:5))
      @test_throws ArgumentError PB.value_and_gradient_from_predictor!(gd, doff, h, zoff)
      @test all(==(T(7)), gd) && all(==(T(7)), doff)
      # an offset design (rows or columns): the combined oracle refuses it before anything is written; the value
      # oracles do not read the design
      for Xo in (view(ones(T, 5, 4), Base.IdentityUnitRange(2:5), :), view(ones(T, 4, 5), :, Base.IdentityUnitRange(2:5)))
        @test Xo isa StridedMatrix{T} && size(Xo) == (4, 4)
        fo = mkloss(Y, Xo, α)
        bo = α === nothing ? fo : fo.f
        fill!(bo.tmp, T(7))
        gd, dd = fill(T(7), 4), fill(T(7), 4)
        @test_throws ArgumentError PB.value_and_gradient_from_predictor!(gd, dd, fo, z1)
        @test all(==(T(7)), gd) && all(==(T(7)), dd) && all(==(T(7)), bo.tmp)
        @test PB.value_from_predictor(fo, z1) == PB.value_from_predictor(g, z1)
      end
      # ordinary (1-based) strided views of longer arrays
      zv = view(zpar, 2:5)
      Yv = view(buf, 2:5)
      @test PB.value_from_predictor(mkloss(Yv, X, α), zv) == PB.value_from_predictor(g, z1)
    end
    # empty designs with offset axes (n = 0, p = 0, 0 × 0): the product with Xᵀ needs no BLAS call, and the combined
    # oracle gives the value, d and g of the same empty design without offset
    for α in (nothing, T(2))
      a = α === nothing ? 1 : 2
      cases = ((view(ones(T, 5, 0), Base.IdentityUnitRange(2:5), :), ones(T, 4), zeros(T, 4), T(a * 4 / 2)),
               (view(ones(T, 0, 4), :, Base.IdentityUnitRange(2:3)), T[], T[], zero(T)),
               (view(ones(T, 2, 2), Base.IdentityUnitRange(2:1), Base.IdentityUnitRange(2:1)), T[], T[], zero(T)))
      for (Xe, Ye, ze, v) in cases
        @test isempty(Xe) && Xe isa StridedMatrix{T}
        n, p = size(Xe)
        fe = mkloss(Ye, Xe, α)
        fp = mkloss(Ye, zeros(T, n, p), α)
        ge, de = fill(T(7), p), fill(T(7), n)
        gp, dp = fill(T(7), p), fill(T(7), n)
        @test PB.value_and_gradient_from_predictor!(ge, de, fe, ze) === v ===
              PB.value_and_gradient_from_predictor!(gp, dp, fp, ze)
        @test de == dp == ze .- Ye && ge == gp == zeros(T, p)
        @test PB.value_from_predictor(fe, ze) === v
      end
    end
  end
end

@testset "large-n accuracy of the sums of squares" begin
  # A constant vector of length 10^6: a sum in one loop of several lanes loses thousands of eps here, a pairwise sum a
  # few. The values must be within 64 eps(T) of the exact value (the documented bound (n + 8) eps(T) is far larger).
  for T in (Float32, Float64)
    n = 10^6
    c = T(1.1)
    exact = with256(() -> n * big(c)^2 / 2)
    rel(v) = Float64(with256(() -> abs(big(v) - exact) / exact / big(eps(T))))
    f = LeastSquaresLoss(zeros(T, n), ones(T, n, 1))
    r = fill(c, n)
    g, d = zeros(T, 1), zeros(T, n)
    vals = (PB.value_from_residual(f, r), PB.value_from_predictor(f, r), PB.value_and_gradient_from_predictor!(g, d, f, r),
            f(T[c]), value_and_gradient!(zeros(T, 1), f, T[c]))
    errs = map(rel, vals)
    all(<=(64), errs) || @info "large-n errors in eps" T errs
    @test all(<=(64), errs)
  end
end

@testset "classification at the top of the range with a long inexact sum" begin
  # Float32 data of length 2^20 + 1 whose sum of squares rounds well below its exact value: with a Float64 or an
  # integer scale the exact value overflows Float32 (by 2.5e-4 relative), and the oracles must give Inf.
  T = Float32
  n = 2^20 + 1
  z = fill(T(2)^42, n)
  z[1] = T(2)^55
  f = LeastSquaresLoss(zeros(T, n), ones(T, n, 1))
  exact = big(2)^110 + big(n - 1) * big(2)^84
  for α in (516349, 516349.0, 516349f0)
    @test T(big(α) * exact / 2) == T(Inf)
    s = Scaled(f, α)
    @test PB.value_from_predictor(s, z) === T(Inf)
    @test PB.value_from_residual(s, z) === T(Inf)
    @test PB.value_and_gradient_from_predictor!(zeros(T, 1), zeros(T, n), s, z) === T(Inf)
  end
  # sums that round above their exact value, and a Float64 scale for which the exact value is below floatmax: the
  # oracles must give a finite value within 2 eps(T) of it
  for m in (2^16, 2^20)
    z = fill(T(7) * T(2)^10, m)
    z[1:16] .= T(2)^24
    exact = 16 * big(2)^48 + big(m - 16) * big(49) * big(2)^20
    s32 = sum(Float64.(z) .^ 2; init = 0.0)                  # exact here: the squares and their sum fit Float64
    @test big(s32) == exact
    for ρ in (1e-4, 1e-3)
      α = Float64(2 * big(floatmax(T)) * (1 - big(ρ)) / exact)
      ref = big(α) * exact / 2
      @test ref < big(floatmax(T))
      g = Scaled(LeastSquaresLoss(zeros(T, m), ones(T, m, 1)), α)
      vals = (PB.value_from_predictor(g, z), PB.value_from_residual(g, z),
              PB.value_and_gradient_from_predictor!(zeros(T, 1), zeros(T, m), g, z))
      ok = all(v -> isfinite(v) && abs(big(v) - ref) <= 2 * big(eps(T)) * ref, vals)
      ok || @info "below the threshold" m ρ vals Float64(ref)
      @test ok
    end
  end
end

@testset "zero allocation and inference" begin
  for T in (Float32, Float64)
    rng = MersenneTwister(101)
    n, p = 64, 6
    X, Y, z, β = randn(rng, T, n, p), randn(rng, T, n), randn(rng, T, n), randn(rng, T, p)
    # a raw loss, a scale of the data type, a Float64 scale with Float32 data (Float64 data: a Float32 scale), an Int
    other = T === Float32 ? 0.3 : 0.3f0
    for α in (nothing, T(0.3), other, 3)
      f = mkloss(Y, X, α)
      g, d, sink = zeros(T, p), zeros(T, n), Ref(zero(T))
      @inferred PB.value_from_predictor(f, z)
      @inferred PB.value_from_residual(f, z)
      @inferred PB.value_and_gradient_from_predictor!(g, d, f, z)
      @inferred PB.leastsquares_data(f)
      alloc_value(f, z, sink); alloc_residual(f, z, sink); alloc_vg(g, d, f, z, sink); alloc_data(f)
      @test alloc_value(f, z, sink) == 0
      @test alloc_residual(f, z, sink) == 0
      @test alloc_vg(g, d, f, z, sink) == 0
      @test alloc_data(f) == 0
    end
    # the methods of the loss that were rerouted, and the kernel of SqrNormL2
    f = LeastSquaresLoss(Y, X)
    g, sink = zeros(T, p), Ref(zero(T))
    @inferred f(β)
    @inferred value_and_gradient!(g, f, β)
    @inferred gradient!(g, f, β)
    alloc_call(f, β, sink); alloc_ls_vg(g, f, β, sink); alloc_grad(g, f, β)
    @test alloc_call(f, β, sink) == 0
    @test alloc_ls_vg(g, f, β, sink) == 0
    @test alloc_grad(g, f, β) == 0
    s = Scaled(f, T(0.3))
    alloc_call(s, β, sink)
    @test alloc_call(s, β, sink) == 0
    r = SqrNormL2(T(0.3), Y)
    @inferred r(z)
    alloc_call(r, z, sink)
    @test alloc_call(r, z, sink) == 0
    rw = SqrNormL2(T === Float32 ? 0.3 : T(0.3), Y)
    @inferred rw(z)
  end
end

end
