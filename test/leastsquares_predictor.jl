module LeastSquaresPredictorTest

# The least-squares oracles at a predictor or a residual supplied by the caller (issue #915):
# value_from_predictor, value_and_gradient_from_predictor!, value_from_residual and leastsquares_data,
# in Float32 and Float64, raw and Scaled. Values and gradients are compared with 256-bit BigFloat
# references formed from the exact bits of X, Y, z (or r) and the scale as supplied. BigFloat is the
# test oracle only.

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

# A value v of n squares against its reference: finite and within (n + 8) eps |v*| (plus two units of
# the smallest subnormal where |v*| < floatmin) when v* is representable with that margin; not finite
# when v* is beyond the range by twice the margin; in between, either.
function value_ok(v, ref, ::Type{T}, n) where {T}
  with256() do
    m, fm, e = abs(ref), big(floatmax(T)), (n + 8) * big(eps(T))
    if m * (1 + e) <= fm
      isfinite(v) && abs(big(v) - ref) <= e * m + (m < floatmin(T) ? 2 * big(nextfloat(zero(T))) : 0)
    elseif m >= fm * (1 + 2e)
      !isfinite(v)
    else
      !isfinite(v) || abs(big(v) - ref) <= e * m
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

@testset "the squared-residual kernel entry points agree where the level fits" begin
  for T in (Float32, Float64), kind in KINDS
    rng = MersenneTwister(5)
    _, Y, z = make_case(rng, T, kind, 64, 3)
    for λ in (one(T), T(0.3), floatmin(T), floatmax(T) / 8)
      @test PB._square_value(λ, z, Y) === PB._square_value_wide(λ, z, Y)
      @test PB._square_value(λ, z, nothing) === PB._square_value_wide(λ, z, nothing)
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
    # z or r may be a view of the scratch, which is then only read
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
