module SqrtLeastSquaresTest

# SqrtLeastSquaresLoss (issue #960): the loss f(β) = ‖Y - Xβ‖₂, its value oracles value_from_predictor and
# value_from_residual (shared with LeastSquaresLoss) and the accessor leastsquares_data, in Float32 and Float64, raw and
# Scaled. Values are compared with 256-bit BigFloat references formed from the exact bits of Y, z (or r) and the scale
# as supplied, within the documented bound.

using Test, ProximalBase, Random, LinearAlgebra, SparseArrays
const PB = ProximalBase

with256(f) = setprecision(f, BigFloat, 256)
pow2(::Type{T}, k) where {T} = ldexp(one(T), k)

mkloss(Y, X, α) = α === nothing ? SqrtLeastSquaresLoss(Y, X) : Scaled(SqrtLeastSquaresLoss(Y, X), α)
baseloss(f::SqrtLeastSquaresLoss) = f
baseloss(f::Scaled) = f.f
scalebig(α) = α === nothing ? big(1) : big(α)

# α ‖z - Y‖₂ (α ‖z‖₂ for Y === nothing) from the exact bits, for finite entries
function ref_norm(α, z, Y)
  with256() do
    s = big(0)
    for i in eachindex(z)
      d = big(z[i]) - (Y === nothing ? big(0) : big(Y[i]))
      s += d^2
    end
    scalebig(α) * sqrt(s)
  end
end

# A value v of n entries against its reference ref, with u = floatmax - prevfloat(floatmax), thr = floatmax + u/2 the
# round-to-nearest overflow threshold and e = (n/2 + 4) eps(T) the relative error bound:
#   (a) |ref| (1 + e) <= floatmax: finite and within e |ref| (plus two units of the smallest subnormal where
#       |ref| < floatmin);
#   (d) between that margin and thr: finite within e |ref|, or +Inf;
#   (b) thr <= |ref| <= thr (1 + e), the boundary region: +Inf, or finite within e |ref| (the recomputation does not
#       decide on which side of thr the value lies within the relative error bound of thr);
#   (c) above that region: +Inf.
function value_ok(v, ref, ::Type{T}, n) where {T}
  # For the supported positive-scale losses, the reference is nonnegative.
  if isnan(ref)
    return isnan(v)
  elseif isinf(ref)
    return v == T(Inf)
  end
  v >= zero(T) || return false
  with256() do
    m, fm, e = abs(ref), big(floatmax(T)), (n / 2 + 4) * big(eps(T))
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

@testset "value_ok reference classes" begin
  with256() do
    for T in (Float32, Float64)
      nan, inf = T(NaN), T(Inf)
      @test value_ok(one(T), BigFloat(1), T, 1)
      for v in (nan, inf, -inf, -one(T))
        @test !value_ok(v, BigFloat(1), T, 1)
      end

      fm = BigFloat(floatmax(T))
      @test value_ok(inf, 2 * fm, T, 1)
      for v in (nan, -inf, floatmax(T))
        @test !value_ok(v, 2 * fm, T, 1)
      end

      @test value_ok(inf, BigFloat(Inf), T, 1)
      @test value_ok(nan, BigFloat(NaN), T, 1)
      for v in (nan, -inf, one(T))
        @test !value_ok(v, BigFloat(Inf), T, 1)
      end
      for v in (inf, -inf, one(T))
        @test !value_ok(v, BigFloat(NaN), T, 1)
      end

      below = BigFloat(prevfloat(floatmax(T)))
      thr = fm + (fm - below) / 2
      for ref in (below, thr)
        @test value_ok(floatmax(T), ref, T, 1)
        @test value_ok(inf, ref, T, 1)
        @test !value_ok(nan, ref, T, 1)
        @test !value_ok(-inf, ref, T, 1)
      end

      tiny = nextfloat(zero(T))
      @test value_ok(3 * tiny, BigFloat(tiny), T, 1)
      @test !value_ok(4 * tiny, BigFloat(tiny), T, 1)
    end
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
  elseif kind === :under                     # the squares underflow, the norm does not
    s = pow2(T, T === Float32 ? -80 : -600)
    Y .*= s
    z .*= s
  elseif kind === :over                      # the squares overflow, the norm does not
    s = pow2(T, T === Float32 ? 80 : 600)
    Y .*= s
    z .*= s
  elseif kind === :huge                      # entries near floatmax / 4: the differences may overflow
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
scales(::Type{Float32}) = Any[nothing, 1f0 / 64, 1f30, 1f-30, pow2(Float32, -140), nextfloat(0f0), 1e-39, 1e-44, 1e-60, 1e30, 3,
                              16777217, 2^53]
scales(::Type{Float64}) = Any[nothing, 1 / 64, 1e200, 1e-200, 1e-300, pow2(Float64, -1030), nextfloat(0.0), pow2(Float64, 1000), 3,
                              2^53, 2^60]

# a design whose products are counted, and that is a strided dense matrix (as in test/leastsquares_predictor.jl)
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

# equal values (isequal) and equal signs, which tells -0.0 from 0.0; NaN payloads are not compared
samebits(a, b) = length(a) == length(b) && all(i -> isequal(a[i], b[i]) && signbit(a[i]) == signbit(b[i]), eachindex(a))

# allocation barriers: a call after warm calls, with the result stored so that it is not discarded
alloc_value(f, z, sink) = @allocated sink[] = PB.value_from_predictor(f, z)
alloc_residual(f, r, sink) = @allocated sink[] = PB.value_from_residual(f, r)
alloc_data(f) = @allocated PB.leastsquares_data(f)
alloc_call(f, β, sink) = @allocated sink[] = f(β)

# the oracles on one input against the references; z - Y is the residual where it is finite
function check_oracles(f, α, Y, z, ::Type{T}) where {T}
  n = length(Y)
  v = PB.value_from_predictor(f, z)
  @test v isa T
  ref = ref_norm(α, z, Y)
  ok = value_ok(v, ref, T, n)
  ok || @info "value_from_predictor" T α n v Float64(ref)
  @test ok
  r = z .- Y
  if all(isfinite, r)
    vr = PB.value_from_residual(f, r)
    @test vr isa T
    rr = ref_norm(α, r, nothing)
    ok = value_ok(vr, rr, T, n)
    ok || @info "value_from_residual" T α n vr Float64(rr)
    @test ok
    @test PB.value_from_residual(f, -r) === vr            # the sign of the residual does not matter
  end
  v
end

@testset "construction, types and traits" begin
  for T in (Float32, Float64)
    X, Y = randn(MersenneTwister(1), T, 5, 3), randn(MersenneTwister(2), T, 5)
    f = SqrtLeastSquaresLoss(Y, X)
    @test f isa SqrtLeastSquaresLoss{T, Vector{T}, Matrix{T}, Vector{T}}
    @test f isa AbstractFunction
    @test f.Y === Y && f.X === X && f.tmp == zeros(T, 5) && axes(f.tmp) == axes(Y)
    @test !issmooth(f) && !isproximable(f) && !isseparable(f) && isconvex(f) && !iszeropreserving(f)
    @test !issmooth(typeof(f)) && !isproximable(typeof(f)) && isconvex(typeof(f))
    s = Scaled(f, T(0.5))
    @test !issmooth(s) && !isproximable(s) && isconvex(s)
    @test Base.isexported(PB, :SqrtLeastSquaresLoss)
    @test haskey(Base.Docs.meta(PB), Base.Docs.Binding(PB, :SqrtLeastSquaresLoss))
  end
  # promotion of the element types: the loss and its scratch are in float(promote_type(eltype(Y), eltype(X)))
  for (Ty, Tx, T) in ((Float32, Float64, Float64), (Float64, Float32, Float64), (Int, Int, Float64), (Int, Float32, Float32),
                      (Float32, Float32, Float32), (Bool, Float64, Float64))
    f = SqrtLeastSquaresLoss(ones(Ty, 4), ones(Tx, 4, 2))
    @test f isa SqrtLeastSquaresLoss{T}
    @test eltype(f.tmp) === T
  end
  # dimensions
  @test_throws DimensionMismatch SqrtLeastSquaresLoss(ones(4), ones(5, 2))
  @test_throws DimensionMismatch SqrtLeastSquaresLoss(ones(4), ones(3, 2))
  @test_throws DimensionMismatch SqrtLeastSquaresLoss(ones(4), ones(4, 2), ones(5))      # tmp has other axes than Y
  # a matrix response is not supported, and nor are complex entries
  @test_throws MethodError SqrtLeastSquaresLoss(ones(4, 2), ones(4, 3))
  @test_throws ArgumentError SqrtLeastSquaresLoss(ones(ComplexF64, 4), ones(4, 2))
  @test_throws ArgumentError SqrtLeastSquaresLoss(ones(4), ones(ComplexF64, 4, 2))
  @test_throws ArgumentError SqrtLeastSquaresLoss(ones(ComplexF64, 4), ones(4, 2), zeros(4))
  # one-based axes of Y, X and tmp, also through the positional constructor
  buf = [100.0, 1, 2, 3, 4]
  Yoff = view(buf, Base.IdentityUnitRange(2:5))
  Xoff = view(ones(5, 2), Base.IdentityUnitRange(2:5), :)
  @test_throws ArgumentError SqrtLeastSquaresLoss(Yoff, ones(4, 2))
  @test_throws ArgumentError SqrtLeastSquaresLoss(ones(4), Xoff)
  @test_throws ArgumentError SqrtLeastSquaresLoss(ones(4), view(ones(4, 5), :, Base.IdentityUnitRange(2:3)))
  @test_throws ArgumentError SqrtLeastSquaresLoss(ones(4), ones(4, 2), view(zeros(5), Base.IdentityUnitRange(2:5)))
  @test_throws ArgumentError SqrtLeastSquaresLoss(Yoff, ones(4, 2), zeros(4))
  @test_throws ArgumentError SqrtLeastSquaresLoss(ones(4), Xoff, zeros(4))
  @test SqrtLeastSquaresLoss(ones(4), ones(4, 2), zeros(4)) isa SqrtLeastSquaresLoss{Float64}
  # an empty loss: no observations, and no coefficients
  for T in (Float32, Float64)
    f = SqrtLeastSquaresLoss(T[], zeros(T, 0, 3))
    @test f(ones(T, 3)) === zero(T)
    @test PB.value_from_predictor(f, T[]) === zero(T)
    @test PB.value_from_residual(f, T[]) === zero(T)
    @test PB.value_from_predictor(Scaled(f, 0.3), T[]) === zero(T)
    f = SqrtLeastSquaresLoss(T[3, 4], zeros(T, 2, 0))
    @test f(T[]) === T(5)
    @test PB.value_from_predictor(f, zeros(T, 2)) === T(5)
  end
  # Scaled(f, 0) is the zero function for finite data, and an ArgumentError otherwise
  f = SqrtLeastSquaresLoss([1.0, 2.0], [1.0 0; 0 1])
  @test Scaled(f, 0) isa ZeroFunction
  @test_throws ArgumentError Scaled(SqrtLeastSquaresLoss([1.0, 2.0], [NaN 0; 0 1]), 0)
  @test_throws ArgumentError Scaled(SqrtLeastSquaresLoss([Inf, 2.0], [1.0 0; 0 1]), 0)
end

@testset "worked examples" begin
  Y, X, β = [3.0, 0.0], Matrix(1.0I, 2, 2), [0.0, 4.0]
  f = SqrtLeastSquaresLoss(Y, X)
  @test f(β) === 5.0
  @test PB.value_from_predictor(f, [0.0, 4.0]) === 5.0
  @test PB.value_from_residual(f, [3.0, -4.0]) === 5.0
  @test PB.value_from_residual(f, [-3.0, 4.0]) === 5.0
  s = Scaled(f, 0.5)
  @test s(β) === 2.5
  @test PB.value_from_predictor(s, [0.0, 4.0]) === 2.5
  @test PB.value_from_residual(s, [3.0, -4.0]) === 2.5
  @test PB.value_from_residual(s, [-3.0, 4.0]) === 2.5
  s = Scaled(f, 2)
  @test s(β) === 10.0
  @test PB.value_from_predictor(s, [0.0, 4.0]) === 10.0
  @test PB.value_from_residual(s, [3.0, -4.0]) === 10.0
  # the normalized objective's data term
  n = 2
  s = Scaled(f, inv(sqrt(n)))
  @test PB.value_from_predictor(s, [0.0, 4.0]) ≈ 5 / sqrt(2)
  # Float32
  f32 = SqrtLeastSquaresLoss(Float32[3, 0], Matrix(1.0f0I, 2, 2))
  @test PB.value_from_predictor(f32, Float32[0, 4]) === 5.0f0
  @test PB.value_from_residual(Scaled(f32, 0.5f0), Float32[3, -4]) === 2.5f0
  @test PB.value_from_residual(Scaled(f32, 0.5), Float32[3, -4]) === 2.5f0       # a Float64 factor, a Float32 result
  # the exact fit is +0 for entries of either zero sign
  @test PB.value_from_predictor(f, [3.0, -0.0]) === 0.0
  @test PB.value_from_predictor(f, [3.0, 0.0]) === 0.0
  @test PB.value_from_residual(f, [0.0, -0.0]) === 0.0
  @test PB.value_from_residual(f, [-0.0, -0.0]) === 0.0
  @test f([3.0, 0.0]) === 0.0
  @test PB.value_from_predictor(Scaled(f, 0.5), [3.0, 0.0]) === 0.0
  # the accessor
  data = PB.leastsquares_data(Scaled(f, 2))
  @test data.Y === Y && data.X === X && data.α === 2
end

@testset "f(β) against the oracles at X * β" begin
  for T in (Float32, Float64)
    rng = Random.Xoshiro(21)
    n, p = 64, 6
    X, Y, β = randn(rng, T, n, p), randn(rng, T, n), randn(rng, T, p)
    z = X * β
    f = SqrtLeastSquaresLoss(Y, X)
    # f(β) and the oracles sum the squares of the same entries in loops that may add in different orders (for example
    # when the oracles come from a package image compiled for another CPU): each value is checked against the
    # reference within the accuracy bound, not compared bit for bit with one another. The two residuals z - Y and Y - z
    # have the same squares and go through the same method, so their values are compared bit for bit.
    @test f(β) isa T
    ref = ref_norm(nothing, z, Y)
    vp = PB.value_from_predictor(f, z)
    @test vp isa T && value_ok(vp, ref, T, n)
    vr = PB.value_from_residual(f, z .- Y)
    @test vr isa T && value_ok(vr, ref_norm(nothing, z .- Y, nothing), T, n)
    @test vr === PB.value_from_residual(f, Y .- z)
    @test value_ok(f(β), ref, T, n)
    # the sparse and the block iterate: the value at their own product X * β against the reference at that product, and
    # that of the dense vector of their entries up to the order of the products
    βs = SparseIterate(β)
    zs = X * βs
    vs = PB.value_from_predictor(f, zs)
    @test f(βs) isa T && vs isa T
    @test value_ok(f(βs), ref_norm(nothing, zs, Y), T, n) && value_ok(vs, ref_norm(nothing, zs, Y), T, n)
    @test close(f(βs), f(Vector(βs)))
    βb = BlockIterate(copy(β), [Int[3, 1, 5], Int[6, 2, 4]])
    zb = X * βb
    vb = PB.value_from_predictor(f, zb)
    @test f(βb) isa T && vb isa T
    @test value_ok(f(βb), ref_norm(nothing, zb, Y), T, n) && value_ok(vb, ref_norm(nothing, zb, Y), T, n)
    @test close(f(βb), f(Vector(βb)))
    βz = SparseIterate(T[0, 2, 0, 0, -1, 0])
    zz = X * βz
    vz = PB.value_from_predictor(f, zz)
    @test f(βz) isa T && vz isa T
    @test value_ok(f(βz), ref_norm(nothing, zz, Y), T, n) && value_ok(vz, ref_norm(nothing, zz, Y), T, n)
    @test close(f(βz), f(Vector(βz)))
    # scaled: the generic wrapper and the oracles are each checked against α ‖z - Y‖₂ within the accuracy bound, for a
    # factor of the data type, an Int and a Float64 factor with Float32 data
    for α in (inv(sqrt(T(n))), T(0.5), T(3), 2, 3, 1e-3)
      s = Scaled(f, α)
      v, o = s(β), PB.value_from_predictor(s, z)
      @test v isa T && o isa T
      refα = ref_norm(α, z, Y)
      @test value_ok(v, refα, T, n) && value_ok(o, refα, T, n)
      vsr = PB.value_from_residual(s, z .- Y)
      @test vsr isa T && value_ok(vsr, ref_norm(α, z .- Y, nothing), T, n)
    end
    # a Float64 factor with Float32 data, and an inverse square root of n in Float64
    sw = Scaled(f, inv(sqrt(n)))
    refw = ref_norm(inv(sqrt(n)), z, Y)
    @test value_ok(sw(β), refw, T, n) && value_ok(PB.value_from_predictor(sw, z), refw, T, n)
    # a factor Inf: the wrapper gives Inf * f(β), which is NaN at an exact fit; the oracles refuse it
    si = Scaled(f, Inf)
    @test si(β) == T(Inf)
    βfit = randn(rng, T, p)
    fit = SqrtLeastSquaresLoss(X * βfit, X)                  # Y is the product X * βfit, which f(βfit) forms again
    @test fit(βfit) === zero(T)
    @test isnan(Scaled(fit, Inf)(βfit))
    @test_throws ArgumentError PB.value_from_predictor(si, z)
    @test_throws ArgumentError PB.value_from_predictor(Scaled(fit, Inf), X * βfit)
  end
  # a loss whose scratch and response have different types: f(β) takes the generic sum
  X, Y, β = randn(Random.Xoshiro(22), 20, 3), randn(Random.Xoshiro(23), Float32, 20), randn(Random.Xoshiro(24), 3)
  f = SqrtLeastSquaresLoss(Y, X)
  @test f isa SqrtLeastSquaresLoss{Float64, Vector{Float32}, Matrix{Float64}}
  ref = ref_norm(nothing, X * β, Y)
  @test value_ok(f(β), ref, Float64, 20)
  @test value_ok(Scaled(f, 0.25)(β), ref / 4, Float64, 20)
  @test_throws MethodError PB.value_from_predictor(f, X * β)
  # Int data, a non-strided design, Float16
  Yi, Xi = rand(Random.Xoshiro(25), -5:5, 10), rand(Random.Xoshiro(26), -3:3, 10, 2)
  βi = [1.5, -2.0]
  fi = SqrtLeastSquaresLoss(Yi, Xi)
  @test value_ok(fi(βi), ref_norm(nothing, Xi * βi, Yi), Float64, 10)
  Xsp = sparse(randn(Random.Xoshiro(27), 10, 2) .* (rand(Random.Xoshiro(28), 10, 2) .< 0.5))
  fsp = SqrtLeastSquaresLoss(randn(Random.Xoshiro(29), 10), Xsp)
  @test value_ok(fsp(βi), ref_norm(nothing, Xsp * βi, fsp.Y), Float64, 10)
  f16 = SqrtLeastSquaresLoss(Float16[3, 0], Float16[1 0; 0 1])
  @test f16(Float16[0, 4]) === Float16(5)
  # f(β) of the wrong length or with complex entries
  f = SqrtLeastSquaresLoss([1.0, 2.0], [1.0 0; 0 1])
  @test_throws DimensionMismatch f([1.0, 2.0, 3.0])
  @test_throws ArgumentError f([1.0 + 0im, 2.0])
  @test_throws MethodError f([1.0 2.0; 3.0 4.0])
end

@testset "accuracy against 256-bit references" begin
  for T in (Float32, Float64)
    for kind in KINDS, n in (kind in (:ordinary, :cancel) ? (1, 7, 64, 300) : (1, 7, 64)), α in scales(T)
      rng = Random.Xoshiro(960 + n)
      X, Y, z = make_case(rng, T, kind, n, 3)
      f = mkloss(Y, X, α)
      check_oracles(f, α, Y, z, T)
    end
    # the factor 1/√n, and factors at the edge of the range, with each kind of data in KINDS
    for kind in KINDS, n in (7, 64)
      rng = Random.Xoshiro(961 + n)
      X, Y, z = make_case(rng, T, kind, n, 3)
      for α in (inv(sqrt(T(n))), nextfloat(zero(T)), floatmin(T), floatmax(T), prevfloat(floatmax(T)))
        check_oracles(mkloss(Y, X, α), α, Y, z, T)
      end
    end
  end
end

@testset "exact fit, zero residual and the sign of the residual" begin
  for T in (Float32, Float64), α in (nothing, T(0.25), 7)
    rng = Random.Xoshiro(1)
    n, p = 64, 6
    X = randn(rng, T, n, p)
    Y = randn(rng, T, n)
    f = mkloss(Y, X, α)
    @test PB.value_from_predictor(f, copy(Y)) === zero(T)
    @test PB.value_from_residual(f, zeros(T, n)) === zero(T)
    @test PB.value_from_residual(f, fill(-zero(T), n)) === zero(T)
    z = randn(rng, T, n)
    @test PB.value_from_residual(f, z .- Y) === PB.value_from_residual(f, Y .- z)
    # zero where the squares of the entries underflow, and where one entry is subnormal and the others are zero
    @test PB.value_from_residual(f, fill(zero(T), n)) === zero(T)
    tiny = zeros(T, n)
    tiny[7] = nextfloat(zero(T))
    @test value_ok(PB.value_from_residual(f, tiny), ref_norm(α, tiny, nothing), T, n)
    @test PB.value_from_residual(f, tiny) >= zero(T)
  end
end

@testset "the ordinary path at the edges of its tests" begin
  # The ordinary value is kept when the sum s is finite, s >= n floatmin(T)/eps(T) (_squares_normal) and the product
  # is below floatmax(T)/2 (_l2_ordinary). The inputs put s just below and above n floatmin/eps, and the value just
  # below and at floatmax/2 and floatmax, with the factors that move the product across the tests.
  for T in (Float32, Float64)
    E = exponent(floatmax(T))
    inputs = Vector{T}[]
    for n in (1, 5, 64), t in (n * (floatmin(T) / eps(T)),)
      c = sqrt(t / n)
      for u in (prevfloat(c, 3), prevfloat(c), c, nextfloat(c), nextfloat(c, 3))
        push!(inputs, fill(u, n))
      end
    end
    half = floatmax(T) / 2
    for h in (prevfloat(half, 2), prevfloat(half), half, nextfloat(half), floatmax(T), prevfloat(floatmax(T)))
      push!(inputs, T[h])
      push!(inputs, T[-h])
    end
    push!(inputs, T[floatmax(T), floatmax(T)], T[floatmax(T), 1], T[pow2(T, E ÷ 2), pow2(T, E ÷ 2)],
          T[pow2(T, E ÷ 2 + 1)], T[pow2(T, E ÷ 2 + 1), pow2(T, E ÷ 2 + 1)], T[floatmin(T)], T[nextfloat(zero(T))])
    wide = T === Float32 ? Any[1e-60, 1e30, 2.0^-150] : Any[]
    for r in inputs, α in Any[nothing, T(0.5), T(2), T(4), pow2(T, -E), pow2(T, E ÷ 2), 3, wide...]
      n = length(r)
      Y = zeros(T, n)
      X = Matrix{T}(I, n, n)
      f = mkloss(Y, X, α)
      vr, vp = PB.value_from_residual(f, r), PB.value_from_predictor(f, r)
      ok = value_ok(vr, ref_norm(α, r, nothing), T, n) && value_ok(vp, ref_norm(α, r, Y), T, n)
      ok || @info "boundary input" T α r vr vp
      @test ok
      if α === nothing && all(isfinite, r)
        # X = I: Xβ = r. With at most two nonzero entries the sum has at most one addition that is not with +0, so the
        # order of the additions does not change it (for the two-entry inputs above it is also the same with a fused
        # multiply-add), and the two values are compared bit for bit; otherwise f(r) is checked against the reference
        if count(!iszero, r) <= 2
          @test f(r) === vp
        else
          @test f(r) isa T && value_ok(f(r), ref_norm(nothing, r, Y), T, n)
        end
      end
    end
  end
end

@testset "the overflow threshold region" begin
  # n = 1, X = 1, Y = 0, z one entry: the exact value is α |z|. α is chosen so that α |z| lies at the round-to-nearest
  # threshold thr = floatmax + u/2 times a factor near 1; the result must be classified against the reference
  # inside the test: below the threshold with margin (finite within the bound), within the boundary region (Inf, or
  # floatmax, or within the bound) and clearly beyond it (Inf).
  for (T, z) in ((Float32, 1.498015f0), (Float64, 1.942395183255378))
    e = 5 * big(eps(T))
    thr, fm = with256() do
      big(floatmax(T)) + (big(floatmax(T)) - big(prevfloat(floatmax(T)))) / 2, big(floatmax(T))
    end
    for (t, kind) in ((1 - 8e, :margin), (1 - 2e, :near_below), (1 - e / 2, :boundary_below), (1 + e / 2, :boundary_above),
                      (1 + 2e, :near_above), (1 + 100 * big(eps(T)), :beyond))
      α = with256() do
        T(t * thr / big(z))
      end
      for k in -2:2
        a = k < 0 ? prevfloat(α, -k) : nextfloat(α, k)
        ref = ref_norm(a, [z], T[0])
        f = SqrtLeastSquaresLoss(T[0], ones(T, 1, 1))
        for v in (PB.value_from_predictor(Scaled(f, a), T[z]), PB.value_from_residual(Scaled(f, a), T[z]),
                  PB.value_from_residual(Scaled(f, a), T[-z]))
          @test v isa T
          ok = value_ok(v, ref, T, 1)
          ok || @info "threshold" T kind k a v Float64(ref)
          @test ok
          kind === :margin && @test isfinite(v)
          kind === :beyond && @test v == T(Inf)
        end
      end
    end
  end
  # below the overflow threshold, outside the margin: four units below floatmax
  for T in (Float32, Float64)
    f4 = Scaled(SqrtLeastSquaresLoss(T[0], ones(T, 1, 1)), prevfloat(floatmax(T) / 2, 4))
    @test PB.value_from_predictor(f4, T[2]) === prevfloat(floatmax(T), 4)
    @test PB.value_from_residual(f4, T[2]) === prevfloat(floatmax(T), 4)
  end
  # the rounding of a value below the smallest subnormal: the exact value is 0.75 nextfloat(0.0), the result 0.0 or
  # nextfloat(0.0) (within the subnormal allowance)
  f = Scaled(SqrtLeastSquaresLoss([0.0], ones(1, 1)), 0.75)
  z = [nextfloat(0.0)]
  @test ref_norm(0.75, z, [0.0]) == 0.75 * big(nextfloat(0.0))
  for v in (PB.value_from_predictor(f, z), PB.value_from_residual(f, z))
    @test v === 0.0 || v === nextfloat(0.0)
    @test value_ok(v, ref_norm(0.75, z, [0.0]), Float64, 1)
  end
  # Float32 data with a Float64 factor: the product rounds to 0f0 below the range of Float32
  f = Scaled(SqrtLeastSquaresLoss(zeros(Float32, 4), Matrix{Float32}(I, 4, 4)), 1e-60)
  @test PB.value_from_residual(f, ones(Float32, 4)) === 0.0f0
  @test PB.value_from_predictor(f, ones(Float32, 4)) === 0.0f0
end

@testset "norms whose squares overflow or underflow" begin
  # the norm is finite where the sum of squares is not, and the other way round for the squares that underflow
  for (T, big_entry, small_entry) in ((Float64, 1e200, 1e-200), (Float32, 1f30, 1f-30))
    n = 3
    Y, X = zeros(T, n), Matrix{T}(I, n, n)
    f = SqrtLeastSquaresLoss(Y, X)
    z = fill(T(big_entry), n)
    ref = ref_norm(nothing, z, Y)
    @test isfinite(T(ref)) && !isfinite(sum(abs2, z))
    v = PB.value_from_predictor(f, z)
    @test isfinite(v) && value_ok(v, ref, T, n)
    @test value_ok(PB.value_from_residual(f, z), ref, T, n)
    @test value_ok(f(z), ref, T, n)                                      # X = I: Xβ = z
    z = fill(T(small_entry), n)
    ref = ref_norm(nothing, z, Y)
    @test iszero(sum(abs2, z)) && T(ref) > 0
    v = PB.value_from_predictor(f, z)
    @test v > 0 && value_ok(v, ref, T, n)
    # with a factor
    for α in (T(0.5), inv(sqrt(T(n))), T(2))
      s = Scaled(f, α)
      z = fill(T(big_entry), n)
      @test value_ok(PB.value_from_predictor(s, z), ref_norm(α, z, Y), T, n)
      @test value_ok(PB.value_from_residual(s, z), ref_norm(α, z, nothing), T, n)
    end
  end
  # a difference that overflows with finite entries: the entries are huge and of opposite signs
  for T in (Float32, Float64)
    Y, X = T[-floatmax(T), floatmax(T) / 2], Matrix{T}(I, 2, 2)
    z = T[floatmax(T), -floatmax(T) / 2]
    f = SqrtLeastSquaresLoss(Y, X)
    ref = ref_norm(nothing, z, Y)
    @test isinf(T(ref))
    @test PB.value_from_predictor(f, z) === T(Inf)
    s = Scaled(f, pow2(T, -3))
    ref = ref_norm(pow2(T, -3), z, Y)
    @test isfinite(T(ref))
    @test value_ok(PB.value_from_predictor(s, z), ref, T, 2)
  end
end

@testset "long vectors" begin
  # up to _LS_ONE_LOOP_MAX entries are summed in one loop, above it in the pairwise reduction
  for T in (Float32, Float64), n in (2^17, 2^17 + 1), kind in (:ordinary, :over)
    rng = Random.Xoshiro(33)
    X = ones(T, n, 1)
    Y = randn(rng, T, n)
    z = randn(rng, T, n)
    if kind !== :ordinary
      s = pow2(T, (kind === :over ? 1 : -1) * (T === Float32 ? 80 : 600))
      Y .*= s
      z .*= s
    end
    for α in (nothing, 3)
      f = mkloss(Y, X, α)
      check_oracles(f, α, Y, z, T)
    end
  end
  # a vector of 10^6 equal entries: a pairwise sum is within a few eps of the exact value, a loop in several lanes loses far more
  for T in (Float32, Float64)
    n = 10^6
    c = T(1.1)
    exact = with256(() -> sqrt(n * big(c)^2))
    rel(v) = Float64(with256(() -> abs(big(v) - exact) / exact / big(eps(T))))
    f = SqrtLeastSquaresLoss(zeros(T, n), ones(T, n, 1))
    r = fill(c, n)
    errs = map(rel, (PB.value_from_residual(f, r), PB.value_from_predictor(f, r), f(T[c])))
    all(<=(64), errs) || @info "large-n errors in eps" T errs
    @test all(<=(64), errs)
  end
end

@testset "non-finite entries" begin
  for T in (Float32, Float64), α in (nothing, T(1) / 8, T(1e-30), 3)
    rng = Random.Xoshiro(11)
    n, p = 8, 3
    X, Y, z0 = randn(rng, T, n, p), randn(rng, T, n), randn(rng, T, n)
    f = mkloss(Y, X, α)
    for i in (1, 4, n)
      z = copy(z0); z[i] = T(NaN)
      @test isnan(PB.value_from_predictor(f, z)) && isnan(PB.value_from_residual(f, z))
      z[i] = T(Inf)
      @test PB.value_from_predictor(f, z) === T(Inf) && PB.value_from_residual(f, z) === T(Inf)
      z[i] = -T(Inf)
      @test PB.value_from_predictor(f, z) === T(Inf) && PB.value_from_residual(f, z) === T(Inf)
      # a NaN and an Inf in one vector: NaN
      zz = copy(z0); zz[i] = T(Inf); zz[mod1(i + 1, n)] = T(NaN)
      @test isnan(PB.value_from_residual(f, zz)) && isnan(PB.value_from_predictor(f, zz))
      # an entry of Y that is not finite
      Yb = copy(Y); Yb[i] = T(NaN)
      @test isnan(PB.value_from_predictor(mkloss(Yb, X, α), z0))
      Yb[i] = T(Inf)
      @test PB.value_from_predictor(mkloss(Yb, X, α), z0) === T(Inf)
      Yb[i] = -T(Inf)
      @test PB.value_from_predictor(mkloss(Yb, X, α), z0) === T(Inf)
      # Inf of the same sign in z and Y: the difference is NaN; of opposite signs it is Inf
      Yb[i] = T(Inf); z = copy(z0); z[i] = T(Inf)
      @test isnan(PB.value_from_predictor(mkloss(Yb, X, α), z))
      z[i] = -T(Inf)
      @test PB.value_from_predictor(mkloss(Yb, X, α), z) === T(Inf)
      Yb[i] = -T(Inf); z[i] = -T(Inf)
      @test isnan(PB.value_from_predictor(mkloss(Yb, X, α), z))
      z[i] = T(Inf)
      @test PB.value_from_predictor(mkloss(Yb, X, α), z) === T(Inf)
    end
    # f(β) with a non-finite entry of β or of the design
    β = randn(rng, T, p)
    βn = copy(β); βn[2] = T(NaN)
    @test isnan(f(βn))
    Xb = copy(X); Xb[3, 2] = T(NaN)
    @test isnan(mkloss(Y, Xb, α)(β))
  end
end

@testset "domain: lengths, axes, types and factors" begin
  for T in (Float32, Float64), α in (nothing, T(2))
    rng = Random.Xoshiro(51)
    n, p = 6, 3
    X, Y, z = randn(rng, T, n, p), randn(rng, T, n), randn(rng, T, n)
    f = mkloss(Y, X, α)
    base = baseloss(f)
    @test_throws DimensionMismatch PB.value_from_predictor(f, zeros(T, n + 1))
    @test_throws DimensionMismatch PB.value_from_predictor(f, zeros(T, n - 1))
    @test_throws DimensionMismatch PB.value_from_residual(f, zeros(T, n + 1))
    @test_throws DimensionMismatch PB.value_from_residual(f, zeros(T, n - 1))
    # other axes of the same length: an offset view
    zpar = randn(rng, T, n + 2)
    zoff = view(zpar, Base.IdentityUnitRange(2:n+1))
    @test zoff isa StridedVector{T} && axes(zoff, 1) == 2:n+1
    @test_throws DimensionMismatch PB.value_from_predictor(f, zoff)
    @test_throws DimensionMismatch PB.value_from_residual(f, zoff)
    # views with the axes of Y are accepted and give the value of their entries
    @test close(PB.value_from_predictor(f, view(zpar, 2:n+1)), PB.value_from_predictor(f, zpar[2:n+1]))
    @test close(PB.value_from_residual(f, view(zpar, 2:n+1)), PB.value_from_residual(f, zpar[2:n+1]))
    # a view with a stride
    zbuf = randn(rng, T, 2n)
    @test close(PB.value_from_predictor(f, view(zbuf, 1:2:2n)), PB.value_from_predictor(f, zbuf[1:2:2n]))
    # element types and arrays outside the domain
    other = T === Float32 ? Float64 : Float32
    @test_throws MethodError PB.value_from_predictor(f, other.(z))
    @test_throws MethodError PB.value_from_residual(f, other.(z))
    @test_throws MethodError PB.value_from_predictor(f, 1:n)
    @test_throws MethodError PB.value_from_residual(f, collect(1:n))
    @test_throws MethodError PB.value_from_predictor(f, view(z, [1, 3, 2, 5, 4, 6]))
    @test_throws MethodError PB.value_from_residual(f, view(z, [1, 3, 2, 5, 4, 6]))
    @test_throws MethodError PB.value_from_predictor(f, [z z])
    @test_throws MethodError PB.value_from_residual(f, [z z])
    @test_throws MethodError PB.value_from_predictor(f, Float16.(z))
    @test_throws MethodError PB.value_from_predictor(f, sparsevec(z))
    # losses outside the domain: a sparse design, an integer response, Float16, mixed types
    @test_throws MethodError PB.value_from_predictor(SqrtLeastSquaresLoss(Y, sparse(X)), z)
    @test_throws MethodError PB.value_from_predictor(SqrtLeastSquaresLoss(Y, Diagonal(ones(T, n))), z)
    @test_throws MethodError PB.value_from_predictor(SqrtLeastSquaresLoss(round.(Int, 10Y), X), z)
    @test_throws MethodError PB.value_from_predictor(SqrtLeastSquaresLoss(Float16.(Y), Float16.(X)), Float16.(z))
    @test_throws MethodError PB.value_from_predictor(SqrtLeastSquaresLoss(other.(Y), X), z)
    # a function that is not a least-squares loss
    @test_throws MethodError PB.value_from_predictor(SqrNormL2(T(2)), z)
    @test_throws MethodError PB.value_from_residual(Scaled(SqrNormL2(T(2)), 2), z)
  end
  # a nested Scaled, which exists where the product of the factors is not a normal number
  f = SqrtLeastSquaresLoss([1.0, 2.0], [1.0 0; 0 1])
  z = [3.0, 4.0]
  nested = Scaled(Scaled(f, 1e200), 1e200)
  @test nested isa Scaled{<:Scaled}
  @test_throws MethodError PB.value_from_predictor(nested, z)
  @test_throws MethodError PB.value_from_residual(nested, z)
  @test_throws MethodError PB.leastsquares_data(nested)
  # a factor outside the domain is an ArgumentError before anything is read, also at an exact fit and with the Inf that
  # Scaled accepts (the factor is checked before the sum, so that Inf * 0 is not formed)
  for T in (Float32, Float64)
    rng = Random.Xoshiro(3)
    n, p = 5, 3
    X, Y, z = randn(rng, T, n, p), randn(rng, T, n), randn(rng, T, n)
    for α in (Inf, T(Inf), 1 // 2, big(0.5), Float16(0.5), 2^53 + 1, big(2)^53 + 1, Int128(2)^80 + 1)
      f = Scaled(SqrtLeastSquaresLoss(Y, X), α)
      @test f isa Scaled{<:SqrtLeastSquaresLoss}
      fill!(f.f.tmp, T(7))
      for zz in (z, copy(Y), zeros(T, n), fill(T(NaN), n))
        @test_throws ArgumentError PB.value_from_predictor(f, zz)
        @test_throws ArgumentError PB.value_from_residual(f, zz)
      end
      @test all(==(T(7)), f.f.tmp)
      @test PB.leastsquares_data(f).α === α                  # the accessor returns the factor as stored
    end
  end
  f = Scaled(SqrtLeastSquaresLoss([1.0], reshape([1.0], 1, 1)), 1 // 2)
  err = try PB.value_from_predictor(f, [1.0]); nothing catch e e end
  @test err isa ArgumentError && occursin("1//2", err.msg) && occursin("Rational", err.msg)
  # the factors that are accepted: an integer that Float64 holds, a Float32 factor with Float64 data
  f64 = SqrtLeastSquaresLoss([0.0], reshape([1.0], 1, 1))
  z64 = [3.0]
  @test PB.value_from_predictor(Scaled(f64, 2^53), z64) === pow2(Float64, 53) * 3
  @test PB.value_from_predictor(Scaled(f64, UInt8(3)), z64) === 9.0
  @test PB.value_from_predictor(Scaled(f64, true), z64) === 3.0
  @test PB.value_from_predictor(Scaled(f64, 0.1f0), z64) === Float64(0.1f0) * 3
  @test PB.value_from_predictor(Scaled(f64, Int128(2)^80), z64) === pow2(Float64, 80) * 3
  # Float32 data with an integer that Float32 does not hold: 3 α rounds up, while α rounded to Float32 first would not
  f32 = SqrtLeastSquaresLoss(Float32[0], reshape(Float32[1], 1, 1))
  @test PB.value_from_predictor(Scaled(f32, 16777217), Float32[3]) === Float32(3 * 16777217.0)
  @test PB.value_from_residual(Scaled(f32, 16777217), Float32[3]) === Float32(3 * 16777217.0)
  # a Float64 factor with Float32 data is applied in Float64 and the result rounded once
  for α in (1e-60, 1e-44, 1e-39, 0.1)
    @test PB.value_from_predictor(Scaled(f32, α), Float32[3]) === Float32(α * 3)
  end
end

@testset "the design and the scratch of the loss are not touched by the oracles" begin
  for T in (Float32, Float64), α in (nothing, T(2), 5)
    rng = Random.Xoshiro(31)
    n, p = 20, 4
    X, Y, z = randn(rng, T, n, p), randn(rng, T, n), randn(rng, T, n)
    clean = mkloss(Y, X, α)
    f = mkloss(Y, X, α)
    base = baseloss(f)
    fill!(base.tmp, T(7))
    z0, r0 = copy(z), z .- Y
    r = copy(r0)
    @test PB.value_from_predictor(f, z) === PB.value_from_predictor(clean, z)
    @test PB.value_from_residual(f, r) === PB.value_from_residual(clean, r)
    @test all(==(T(7)), base.tmp)                                    # the scratch keeps its sentinel
    @test samebits(z, z0) && samebits(r, r0)                         # the arguments are not modified
    # z or r may be a view of the scratch, which the call reads and does not write
    copyto!(base.tmp, z)
    zv = view(base.tmp, 1:n)
    @test close(PB.value_from_predictor(f, zv), PB.value_from_predictor(clean, z))
    @test close(PB.value_from_residual(f, zv), PB.value_from_residual(clean, z))
    @test samebits(base.tmp, z)
    # the whole scratch as the predictor
    @test PB.value_from_predictor(f, base.tmp) === PB.value_from_predictor(clean, z)
    # a design full of NaN does not enter the values
    fnan = mkloss(Y, fill(T(NaN), n, p), α)
    @test PB.value_from_predictor(fnan, z) === PB.value_from_predictor(clean, z)
    @test PB.value_from_residual(fnan, z) === PB.value_from_residual(clean, z)
    # and a loss whose scratch holds NaN
    fill!(baseloss(fnan).tmp, T(NaN))
    @test PB.value_from_predictor(fnan, z) === PB.value_from_predictor(clean, z)
    @test all(isnan, baseloss(fnan).tmp)
    # the oracles do not copy the response either: a change of Y after the construction is seen
    Y2 = copy(Y)
    g = mkloss(Y2, X, α)
    v = PB.value_from_predictor(g, z)
    Y2[1] += T(1)
    @test PB.value_from_predictor(g, z) !== v
  end
end

@testset "products with the design" begin
  for T in (Float32, Float64), α in (nothing, T(0.5))
    rng = Random.Xoshiro(41)
    n, p = 30, 5
    Xd, Y, z = randn(rng, T, n, p), randn(rng, T, n), randn(rng, T, n)
    X = CountMat(Xd)
    @test X isa StridedMatrix{T}
    f = mkloss(Y, X, α)
    PB.value_from_predictor(f, z)
    PB.value_from_residual(f, z)
    PB.leastsquares_data(f)
    @test products(X) == (0, 0)                                      # the oracles form no product
    # f(β) forms one product by X and none by Xᵀ
    β = randn(rng, T, p)
    fl = SqrtLeastSquaresLoss(Y, X)
    resetcount!(X)
    v = fl(β)
    @test products(X) == (1, 0)
    # the values of the two designs sum the same squares in loops that may add in different orders: each is checked
    # against the reference at the product Xd * β
    zd = Xd * β
    vd = PB.value_from_predictor(SqrtLeastSquaresLoss(Y, Xd), zd)
    @test v isa T && vd isa T
    @test value_ok(v, ref_norm(nothing, zd, Y), T, n) && value_ok(vd, ref_norm(nothing, zd, Y), T, n)
    resetcount!(X)
    Scaled(fl, T(0.3))(β)
    @test products(X) == (1, 0)
  end
end

@testset "functions that the loss does not provide" begin
  for T in (Float32, Float64)
    rng = Random.Xoshiro(91)
    n, p = 5, 3
    X, Y, β = randn(rng, T, n, p), randn(rng, T, n), randn(rng, T, p)
    f = SqrtLeastSquaresLoss(Y, X)
    for g in (f, Scaled(f, T(0.5)))
      gr, y = zeros(T, p), zeros(T, p)
      @test_throws MethodError gradient!(gr, g, β)
      @test_throws MethodError gradient(g, β)
      @test_throws MethodError value_and_gradient!(gr, g, β)
      @test_throws MethodError prox!(y, g, β, one(T))
      @test_throws MethodError prox(g, β, one(T))
      @test_throws MethodError value_and_prox!(y, g, β, one(T))
      @test_throws MethodError prox_block(g, β[1], 1, one(T))
      @test_throws MethodError kkt_violation(g, gr, 1)
      @test_throws MethodError kkt_violation(g, gr, β, 1)
      @test_throws MethodError restrict(g, [1, 2])
      @test_throws MethodError PB.value_and_gradient_from_predictor!(gr, zeros(T, n), g, X * β)
    end
    @test_throws ArgumentError lipschitz(f)
    @test reset_cache!(f) === f
    s = Scaled(f, T(0.5))
    @test reset_cache!(s) === s
    # the loss still gives a value after a failed call
    @test f(β) isa T
  end
end

@testset "the accessor" begin
  for T in (Float32, Float64)
    X, Y = randn(Random.Xoshiro(81), T, 5, 3), randn(Random.Xoshiro(82), T, 5)
    f = SqrtLeastSquaresLoss(Y, X)
    data = PB.leastsquares_data(f)
    @test keys(data) == (:Y, :X, :α)
    @test data.Y === Y && data.X === X
    @test data.α === one(T)
    for α in (T(0.5), 0.25, 3, 2^53 + 1, 1 // 2, Float16(0.5), Inf, big(0.5))
      data = PB.leastsquares_data(Scaled(f, α))
      @test data.Y === Y && data.X === X
      @test data.α === α && typeof(data.α) === typeof(α)
    end
    # the accessor is shared with LeastSquaresLoss and the losses are told apart by their types
    ls = LeastSquaresLoss(Y, X)
    @test PB.leastsquares_data(ls).Y === Y && PB.leastsquares_data(ls).X === X
    @test PB.leastsquares_data(ls) == PB.leastsquares_data(f)
    @test typeof(f) !== typeof(ls) && !(f isa LeastSquaresLoss) && !(ls isa SqrtLeastSquaresLoss)
    @test PB.leastsquares_data(Scaled(f, 0.5)).α === 0.5
    # a non-strided design or a mixed-type loss is accepted too
    @test PB.leastsquares_data(SqrtLeastSquaresLoss(Y, sparse(X))).Y === Y
    @test_throws MethodError PB.leastsquares_data(SqrNormL2(T(2)))
  end
end

@testset "zero allocation and inference" begin
  for T in (Float32, Float64), n in (64, 2^17 + 5)
    rng = Random.Xoshiro(101)
    p = 6
    X, Y, z, β = randn(rng, T, n, p), randn(rng, T, n), randn(rng, T, n), randn(rng, T, p)
    # a raw loss, a scale of the data type, a Float64 scale with Float32 data (Float64 data: a Float32 scale), an Int
    other = T === Float32 ? 0.3 : 0.3f0
    for α in (nothing, T(0.3), other, 3)
      f = mkloss(Y, X, α)
      sink = Ref(zero(T))
      @inferred PB.value_from_predictor(f, z)
      @inferred PB.value_from_residual(f, z)
      @inferred PB.leastsquares_data(f)
      alloc_value(f, z, sink); alloc_residual(f, z, sink); alloc_data(f)
      @test alloc_value(f, z, sink) == 0
      @test alloc_residual(f, z, sink) == 0
      @test alloc_data(f) == 0
      # a view and a strided view
      zv = view(z, 1:n)
      alloc_value(f, zv, sink); alloc_residual(f, zv, sink)
      @test alloc_value(f, zv, sink) == 0
      @test alloc_residual(f, zv, sink) == 0
    end
    # the method of the loss with a dense β: one product into the scratch and the same kernel
    f = SqrtLeastSquaresLoss(Y, X)
    sink = Ref(zero(T))
    @inferred f(β)
    alloc_call(f, β, sink)
    @test alloc_call(f, β, sink) == 0
    s = Scaled(f, T(0.3))
    alloc_call(s, β, sink)
    @test alloc_call(s, β, sink) == 0
  end
end

end
