module ConjugatePrecisionTest

# The reciprocal in Moreau's identity (numerics T3, R3, P2): `Conjugate` forms 1/γ from the stored
# step in the promotion of the model precision and the step's type, instead of rounding it to the
# step's type first (issue #576). Every reference below is built from the stored parameters in
# BigFloat, converted before any arithmetic.

using Test
using ProximalBase
using ProximalBase: prox_block
using LinearAlgebra
using JLArrays
using ..TestUtils

const PB = ProximalBase

relerr(got, ref) = Float64(abs(big(got) - ref) / abs(ref))
# prox_{γ(αf)*}(x) for f(t) = λ1|t| + (λ2/2)t², centered at b: x - γ prox_{(α/γ)f(·-b)}(x/γ)
function ref_en(x, γ; λ1 = 0.5, λ2 = 2.0, α = 1.0, b = 0.0)
  X, G, A, B, L1, L2 = big(x), big(γ), big(α), big(b), big(λ1), big(λ2)
  ρ = A / G
  u = X / G - B
  s = B + sign(u) * max(abs(u) - ρ * L1, 0) / (1 + ρ * L2)
  X - G * s
end
# the quadratic block SqrNormL2() of a conjugate: x / (1 + γ)
ref_sqr(x, γ) = big(x) / (1 + big(γ))

# an operator with its element type on the instance only (the interface of LinearImage, test/image.jl)
struct ReviewMap{M}
  A::M
end
Base.size(L::ReviewMap, d::Integer) = size(L.A, d)
Base.eltype(L::ReviewMap) = eltype(L.A)
Base.adjoint(L::ReviewMap) = ReviewMap(L.A')
LinearAlgebra.mul!(y, L::ReviewMap, x) = mul!(y, L.A, x)

flat(y::Tuple) = reduce(vcat, flat.(y))
flat(y::AbstractArray) = vec(copy(y))
function sentinel!(y::Tuple, v)
  foreach(b -> sentinel!(b, v), y)
  y
end
sentinel!(y::AbstractArray, v) = fill!(y, v)

# what a call does: its output and value, or the kind of its error with the output as it was left
function outcome(op, c, x, y, γ)
  sentinel!(y, 7)
  try
    v = op === prox! ? (prox!(y, c, x, γ); nothing) : value_and_prox!(y, c, x, γ)
    (:ok, flat(y), v)
  catch e
    (:err, typeof(e), flat(y))
  end
end

@testset "the reciprocal helper" begin
  # Float32 model, Float32 step: the plain reciprocal, in Float32
  @test PB._moreau_reciprocal(Float32, 3f0) === inv(3f0)
  # Float64 model, Float32 step: the reciprocal of the stored step, in Float64
  r = PB._moreau_reciprocal(Float64, 3f0)
  @test r === inv(3.0)
  @test r !== Float64(inv(3f0))
  @test PB._moreau_reciprocal(Float64, 0.1f0) === inv(Float64(0.1f0))
  # a step wider than the model is not rounded to the model: the promotion is the step's type
  @test PB._moreau_reciprocal(Float32, 0.1) === inv(0.1)
  # a power of two is exact, and a subnormal Float32 step has a finite reciprocal in Float64
  @test PB._moreau_reciprocal(Float64, 0.5f0) === 2.0
  @test PB._moreau_reciprocal(Float64, 1f-45) === inv(Float64(1f-45))
  @test isinf(PB._moreau_reciprocal(Float32, 1f-45))
  # an integer step is a Float64 step, as before
  @test PB._moreau_reciprocal(Float32, 2) === 0.5
  # the model of arrays and of tuples
  f0 = SqrNormL2()
  @test PB._moreau_reciprocal(f0, zeros(Float64, 2), zeros(Float32, 2), 3f0) === inv(3.0)
  @test PB._moreau_reciprocal(f0, zeros(Float32, 2), zeros(Float32, 2), 3f0) === inv(3f0)
  @test PB._moreau_reciprocal(f0, (zeros(Float32, 1), (zeros(Float64, 1),)), (zeros(Float32, 1), (zeros(Float32, 1),)), 3f0) === inv(3.0)
  @test PB._moreau_reciprocal(f0, (zeros(Float32, 1),), (zeros(Float32, 1),), 3f0) === inv(3f0)
  # the data of the function widen the model: Float64 data under Float32 arrays
  Q = QuadraticFunction(reshape([1.0], 1, 1), [1.0])
  Q32 = QuadraticFunction(reshape([1f0], 1, 1), [1f0])
  z32 = zeros(Float32, 1)
  @test PB._moreau_reciprocal(Q, z32, z32, 3f0) === inv(3.0)
  @test PB._moreau_reciprocal(Q32, z32, z32, 3f0) === inv(3f0)
  for g in (Scaled(Q, 2.0), Translate(Q, [1.0]), Tilt(Q, [1.0]), Conjugate(Q), SeparableSum(NormL1(), Q),
            SeparableSum(NormL1(), SeparableSum(SqrNormL2(), Scaled(Q, 2.0))), Sum(Q, SqrNormL2()))
    @test PB._moreau_reciprocal(g, z32, z32, 3f0) === inv(3.0)
  end
  for g in (Scaled(Q32, 2.0), SeparableSum(NormL1(), Q32), NormL1(), SqrNormL2(), ElasticNet(0.5, 2.0))
    @test PB._moreau_reciprocal(g, z32, z32, 3f0) === inv(3f0)
  end
  @test PB._data_type(typeof(Q)) === Float64 && PB._data_type(typeof(Q32)) === Float32
  @test PB._data_type(typeof(NormL1(1.0, Float64[1, 2]))) === Union{}
  @test PB._data_type(typeof(LeastSquaresLoss(Float64[1, 2], Float64[1 2; 3 4]))) === Float64
  @test PB._data_type(typeof(LeastSquaresLoss(Float32[1, 2], Float32[1 2; 3 4]))) === Float32
  @test PB._data_type(typeof(IndHyperplane(Float64[1, 1], 1.0))) === Float64
  @test PB._data_type(typeof(IndHyperplane(Float32[1, 1], 1f0))) === Float32
  @test @inferred(PB._moreau_reciprocal(SeparableSum(NormL1(), Q), (z32, z32), (z32, z32), 3f0)) isa Float64
  # the coefficient form divides once
  @test PB._moreau_ratio(Float64, 2f0, 3f0) === 2.0 / 3.0
  @test PB._moreau_ratio(Float32, 2f0, 3f0) === 2f0 / 3f0
  @test PB._moreau_ratio(Float64, 2f0, 0.1f0) === 2.0 / Float64(0.1f0)
  # inferred, so that the hot paths do not box
  @test @inferred(PB._moreau_reciprocal(Float64, 3f0)) isa Float64
  @test @inferred(PB._moreau_reciprocal(Float32, 3f0)) isa Float32
  @test @inferred(PB._moreau_reciprocal(f0, (zeros(Float32, 1), (zeros(Float64, 1),)), (zeros(Float32, 1), (zeros(Float32, 1),)), 3f0)) isa Float64
end

@testset "the lazy reciprocal array" begin
  γ = Float32[3, 0.1, 7]
  r64 = PB._ReciprocalArray(Float64, γ)
  @test r64 isa AbstractVector{Float64}
  @test size(r64) == size(γ) && axes(r64) == axes(γ) && length(r64) == 3
  @test collect(r64) == inv.(Float64.(γ))
  @test all(i -> r64[i] === inv(Float64(γ[i])), eachindex(γ))
  @test collect(r64) != Float64.(inv.(γ))
  r32 = PB._ReciprocalArray(Float32, γ)
  @test eltype(r32) === Float32 && collect(r32) == inv.(γ)
  # the promotion of the model and the steps, so that a wider step is not narrowed
  @test eltype(PB._ReciprocalArray(Float32, [3.0, 0.1])) === Float64
  @test eltype(PB._ReciprocalArray(NormL1(), (zeros(Float64, 2),), (zeros(Float32, 2),), γ[1:2])) === Float64
  @test eltype(PB._ReciprocalArray(NormL1(), zeros(Float32, 2), zeros(Float32, 2), γ[1:2])) === Float32
  @test eltype(PB._ReciprocalArray(QuadraticFunction(reshape([1.0], 1, 1), [1.0]), zeros(Float32, 2), zeros(Float32, 2), γ[1:2])) === Float64
  # a matrix of steps keeps its shape
  M = Float32[3 0.1; 7 1]
  rm = PB._ReciprocalArray(Float64, M)
  @test size(rm) == (2, 2) && axes(rm) == axes(M) && rm[2, 1] === inv(Float64(7f0)) && vec(collect(rm)) == vec(inv.(Float64.(M)))
  @test allocs(getindex, r64, 2) == 0
end

@testset "the original defect" begin
  c = Conjugate(SeparableSum(NormL1(), SeparableSum(SqrNormL2())))
  # 1f10 is ill-conditioned: x - γp cancels 10 digits, for the Float64 control as well, so it is
  # compared with the control only
  for γ in (3f0, 0.1f0, 7f-3, 7f0, 1f10)
    x = (Float64[2], (Float64[1],))
    for op in (prox!, value_and_prox!)
      y = (zeros(1), (zeros(1),))
      v = op === prox! ? (prox!(y, c, x, γ); nothing) : value_and_prox!(y, c, x, γ)
      ref = ref_sqr(1.0, γ)
      @test y[1][1] == 1.0
      γ < 1f9 && @test relerr(y[2][1][1], ref) <= 8 * eps(Float64)
      # Float64 controls: the stored step as a Float64 gives the same bits
      yc = (zeros(1), (zeros(1),))
      op === prox! ? prox!(yc, c, x, Float64(γ)) : value_and_prox!(yc, c, x, Float64(γ))
      @test y == yc
      if op === value_and_prox!
        # the value of the stored output (O2): the conjugate of the quadratic block is ½y²
        @test relerr(v, big(y[2][1][1])^2 / 2) <= 8 * eps(Float64)
      end
    end
  end
  # a step at which Moreau's identity cancels completely: the Float64 control does the same
  x, y, yc = (Float64[2], (Float64[1],)), (zeros(1), (zeros(1),)), (zeros(1), (zeros(1),))
  prox!(y, c, x, 1f30)
  prox!(yc, c, x, Float64(1f30))
  @test y == yc
end

@testset "a Float32 step is its stored Float64 value in a Float64 model" begin
  b = [0.5, 0.25]
  en = ElasticNet(0.5, 2.0)
  arrays = (("ElasticNet", Conjugate(en)), ("SqrNormL2", Conjugate(SqrNormL2())),
            ("Scaled", Conjugate(Scaled(en, 2.0))), ("Translate", Conjugate(Translate(en, b))),
            ("Scaled Translate", Conjugate(Scaled(Translate(SqrNormL2(), b), 3.0))),
            ("nested Scaled", Conjugate(Scaled(Scaled(en, 2.0^600), 2.0^-590))),
            ("exceptional Scaled", Conjugate(Scaled(Scaled(en, 2.0^900), 2.0^200))),
            ("exceptional Scaled below", Conjugate(Scaled(Scaled(SqrNormL2(), 2.0^-900), 2.0^-200))))
  tuples = (("nested quadratic", Conjugate(SeparableSum(NormL1(), SeparableSum(SqrNormL2()))), () -> (Float64[2], (Float64[1],))),
            ("elastic net", Conjugate(SeparableSum(en, NormL1())), () -> (Float64[2], Float64[1])),
            ("Scaled sum", Conjugate(Scaled(SeparableSum(SqrNormL2(), en), 2.0)), () -> (Float64[2], Float64[1])),
            ("nested tuple", Conjugate(SeparableSum(SeparableSum(en, SqrNormL2()), SqrNormL2())), () -> ((Float64[2], Float64[1]), Float64[3])),
            ("exceptional Scaled sum", Conjugate(Scaled(Scaled(SeparableSum(SqrNormL2(), en), 2.0^900), 2.0^200)), () -> (Float64[2], Float64[1])))
  # the special steps of P1 and the extreme ones, as Float32 values
  steps = Float32[0, -1, NaN, Inf, 1f-45, 1f-40, 1f-38, 0.5, 3, 0.1, 7, 1f30, 1f38, floatmax(Float32)]
  for γ in steps, op in (prox!, value_and_prox!)
    for (name, c) in arrays
      x = [2.0, 3.0]
      a = outcome(op, c, x, zeros(2), γ)
      b64 = outcome(op, c, x, zeros(2), Float64(γ))
      @test isequal(a, b64)
      a[1] === :err && @test a[3] == [7.0, 7.0]
    end
    for (name, c, mk) in tuples
      x = mk()
      y() = deepcopy(map(t -> t isa Tuple ? map(zero, t) : zero(t), x))
      a = outcome(op, c, x, y(), γ)
      b64 = outcome(op, c, x, y(), Float64(γ))
      @test isequal(a, b64)
      a[1] === :err && @test all(==(7.0), a[3])
    end
  end
  # the endpoints that are rejected stay rejected, before anything is written
  for γ in (0f0, NaN32, Inf32), op in (prox!, value_and_prox!), (name, c) in arrays
    r = outcome(op, c, [2.0, 3.0], zeros(2), γ)
    @test r[1] === :err && r[2] === ArgumentError && r[3] == [7.0, 7.0]
  end
  # a subnormal Float32 step has a finite reciprocal in a Float64 model: the point is not the one
  # of a step whose reciprocal overflowed to Inf
  x, y = (Float64[2], Float64[1]), (zeros(1), zeros(1))
  c = Conjugate(SeparableSum(en, NormL1()))
  v = value_and_prox!(y, c, x, 1f-40)
  @test isfinite(v) && v > 0
end

@testset "references for the array, entry and wrapper routes" begin
  b = [0.5, 0.25]
  x = [2.0, 3.0]
  for γ in (3f0, 0.1f0, 7f0, 1f-3)
    for (c, kw) in ((Conjugate(ElasticNet(0.5, 2.0)), (;)),
                    (Conjugate(Scaled(ElasticNet(0.5, 2.0), 2.0)), (; α = 2.0)),
                    (Conjugate(Scaled(Scaled(ElasticNet(0.5, 2.0), 2.0^600), 2.0^-590)), (; α = 2.0^10)),
                    (Conjugate(Translate(ElasticNet(0.5, 2.0), b)), (; b = b)))
      refs = [ref_en(x[i], γ; kw..., b = haskey(kw, :b) ? b[i] : 0.0) for i in 1:2]
      y = zeros(2)
      prox!(y, c, x, γ)
      @test all(i -> relerr(y[i], refs[i]) <= 8 * eps(Float64), 1:2)
      y .= 0
      value_and_prox!(y, c, x, γ)
      @test all(i -> relerr(y[i], refs[i]) <= 8 * eps(Float64), 1:2)
      # unaliased and aliased calls agree
      xa = copy(x)
      prox!(xa, c, xa, γ)
      @test xa == y
      xa = copy(x)
      value_and_prox!(xa, c, xa, γ)
      @test xa == y
      # prox_block is the same map, entry by entry
      @test all(i -> relerr(prox_block(c, x[i], i, γ), refs[i]) <= 8 * eps(Float64), 1:2)
      @test all(i -> prox_block(c, x[i], i, γ) == prox_block(c, x[i], i, Float64(γ)), 1:2)
    end
    # the entry of the plain quadratic
    @test relerr(prox_block(Conjugate(SqrNormL2()), 1.0, 1, γ), ref_sqr(1.0, γ)) <= 8 * eps(Float64)
  end
  # Float32 entries stay Float32
  @test prox_block(Conjugate(ElasticNet(0.5, 2.0)), 2f0, 1, 3f0) isa Float32
  @test relerr(prox_block(Conjugate(ElasticNet(0.5, 2.0)), 2f0, 1, 3f0), ref_en(2f0, 3f0)) <= 4 * eps(Float32)
end

@testset "step arrays of a union or abstract element type" begin
  # the declared element type has no concrete promotion, yet each stored Float32 step is a Float32
  # step in a Float64 model: the points are those of the Float64 conversion of the stored steps
  c = Conjugate(ElasticNet(0.5, 2.0))
  x = [2.0, 3.0]
  refs = [ref_en(x[i], Float32[3, 0.1][i]) for i in 1:2]
  for g in (Union{Float32, Float64}[3f0, 0.1f0], AbstractFloat[3f0, 0.1f0], Real[3f0, 0.1f0]), op in (prox!, value_and_prox!)
    y = zeros(2)
    v = op(y, c, x, g)
    @test all(i -> relerr(y[i], refs[i]) <= 8 * eps(Float64), 1:2)
    y64 = zeros(2)
    v64 = op(y64, c, x, Float64.(g))
    @test y == y64 && isequal(v, v64)
  end
  @test eltype(PB._ReciprocalArray(Float64, Union{Float32, Float64}[3f0])) === Float64
  @test PB._ReciprocalArray(Float64, Union{Float32, Float64}[3f0])[1] === inv(3.0)
  # a Float64 entry in a Float32 model is rounded once to the model
  @test PB._ReciprocalArray(Float32, Union{Float32, Float64}[0.1])[1] === Float32(inv(0.1))
  # concrete arrays are as before
  @test PB._ReciprocalArray(Float64, Float32[3])[1] === inv(3.0)
  @test PB._ReciprocalArray(Float32, Float32[3])[1] === inv(3f0)
end

@testset "diagonal metric: Float32 step arrays" begin
  en = ElasticNet(0.5, 2.0)
  c = Conjugate(en)
  γ = Float32[3, 0.1, 7, 1f-3]
  x = [2.0, 3.0, -1.5, 0.25]
  refs = [ref_en(x[i], γ[i]) for i in 1:4]
  y = zeros(4)
  prox!(y, c, x, γ)
  @test all(i -> relerr(y[i], refs[i]) <= 8 * eps(Float64), 1:4)
  yv = zeros(4)
  v = value_and_prox!(yv, c, x, γ)
  @test yv == y
  # the value is that of the stored output; the conjugate of the elastic net has no standalone
  # value oracle, so it is the Fenchel value at the stored point
  @test isfinite(v)
  # the stored Float64 steps give the same bits
  @test y == prox!(zeros(4), c, x, Float64.(γ))
  # aliased
  xa = copy(x)
  prox!(xa, c, xa, γ)
  @test xa == y
  # a matrix argument with a matrix of steps keeps its shape
  X = [2.0 3.0; -1.5 0.25]
  Γ = Float32[3 7; 0.1 1f-3]
  Y = zeros(2, 2)
  prox!(Y, c, X, Γ)
  @test size(Y) == (2, 2)
  @test all(i -> relerr(Y[i], ref_en(X[i], Γ[i])) <= 8 * eps(Float64), eachindex(Y))
  # Float32 arrays with Float32 steps stay Float32
  y32 = zeros(Float32, 4)
  prox!(y32, c, Float32.(x), γ)
  @test eltype(y32) === Float32
  @test all(i -> relerr(y32[i], ref_en(Float32(x[i]), γ[i])) <= 4 * eps(Float32), 1:4)
  # a bad entry is rejected before anything is written
  yb = fill(7.0, 4)
  @test_throws ArgumentError prox!(yb, c, x, Float32[3, 0.1, NaN32, 1])
  @test all(==(7.0), yb)
  @test_throws ArgumentError prox!(yb, c, x, Float32[3, 0.1, 0, 1])
  @test all(==(7.0), yb)
end

@testset "precision combinations" begin
  c = Conjugate(ElasticNet(0.5, 2.0))
  for γ in (3f0, 0.1f0), (Ty, Tx) in ((Float64, Float64), (Float64, Float32), (Float32, Float64), (Float32, Float32))
    x = Tx[2, 3]
    y = zeros(Ty, 2)
    prox!(y, c, x, γ)
    @test eltype(y) === Ty
    refs = [ref_en(x[i], γ) for i in 1:2]
    # a Float32 destination holds a Float32 result; a Float64 model holds Float64 digits
    tol = Ty === Float64 ? 8 * eps(Float64) : 4 * eps(Float32)
    @test all(i -> relerr(y[i], refs[i]) <= tol, 1:2)
    y2 = zeros(Ty, 2)
    v = value_and_prox!(y2, c, x, γ)
    @test y2 == y
    @test v isa Ty
  end
  # a Float64 step in a Float32 model is formed once from the stored step
  x = Float32[2, 3]
  y = zeros(Float32, 2)
  prox!(y, c, x, 0.1)
  @test all(i -> relerr(y[i], ref_en(x[i], 0.1)) <= 4 * eps(Float32), 1:2)
  # blocks of different precisions: each block keeps its type, and its accuracy
  ct = Conjugate(SeparableSum(NormL1(), SeparableSum(SqrNormL2())))
  for γ in (3f0, 0.1f0)
    x = (Float32[2], (Float64[1],))
    y = (zeros(Float32, 1), (zeros(1),))
    prox!(y, ct, x, γ)
    @test y[1] isa Vector{Float32} && y[2][1] isa Vector{Float64}
    @test y[1][1] == 1f0
    @test relerr(y[2][1][1], ref_sqr(1.0, γ)) <= 8 * eps(Float64)
    x = (Float64[2], (Float32[1],))
    y = (zeros(1), (zeros(Float32, 1),))
    prox!(y, ct, x, γ)
    @test y[2][1] isa Vector{Float32}
    @test relerr(y[2][1][1], ref_sqr(1f0, γ)) <= 4 * eps(Float32)
  end
  # a Float32 model keeps its precision and its plain reciprocal
  x = (Float32[2], (Float32[1],))
  y = (zeros(Float32, 1), (zeros(Float32, 1),))
  prox!(y, ct, x, 3f0)
  @test y[2][1] isa Vector{Float32}
  @test relerr(y[2][1][1], ref_sqr(1f0, 3f0)) <= 4 * eps(Float32)
  # a Float64 destination of a Float32 argument: the model is Float64
  x = (Float32[2], (Float32[1],))
  y = (zeros(1), (zeros(1),))
  prox!(y, ct, x, 0.1f0)
  @test relerr(y[2][1][1], ref_sqr(1f0, 0.1f0)) <= 8 * eps(Float64)
end

@testset "preflight uses the same reciprocal" begin
  # a step whose Float32 reciprocal overflows is accepted in a Float64 model, in the preflight
  # and in the execution alike, and rejected steps leave the output untouched
  en = ElasticNet(0.5, 2.0)
  c = Conjugate(SeparableSum(en, NormL1()))
  x = (Float64[2], Float64[1])
  y = (fill(7.0, 1), fill(7.0, 1))
  @test_throws ArgumentError prox!(y, c, x, NaN32)
  @test y == ([7.0], [7.0])
  @test_throws ArgumentError value_and_prox!(y, c, x, -1f0 * Inf32)
  @test y == ([7.0], [7.0])
  @test isfinite(value_and_prox!(y, c, x, 1f-40))
  @test isfinite(value_and_prox!(y, c, x, floatmax(Float32)))
  # the preflight of a block call agrees with its execution
  @test PB._checkstep(prox!, c, y, x, 1f-40)
  @test PB._checkstep(value_and_prox!, c, y, x, 1f-40)
  yarr, xarr = [7.0, 7.0], [2.0, 3.0]
  ca = Conjugate(en)
  @test PB._checkstep(prox!, ca, yarr, xarr, 1f-40)
  @test PB._checkstep(value_and_prox!, ca, yarr, xarr, 1f-40)
  # a function whose check looks at the step it receives: the Float32 step and its stored Float64
  # value are accepted, or rejected before any write, alike
  for g in (Constrained(NormL1(0.5), IndBallL2(2.0)),)
    cc = Conjugate(g)
    for γ in Float32[3, 1f-40, 1f-45, 1f30, 0, NaN, -1, Inf], op in (prox!, value_and_prox!)
      a = outcome(op, cc, [3.0, 1.0], zeros(2), γ)
      @test isequal(a, outcome(op, cc, [3.0, 1.0], zeros(2), Float64(γ)))
      a[1] === :err && @test a[3] == [7.0, 7.0]
    end
    for γ in (0f0, NaN32), op in (prox!, value_and_prox!)
      r = outcome(op, cc, [3.0, 1.0], zeros(2), γ)
      @test r[1] === :err && r[2] === ArgumentError && r[3] == [7.0, 7.0]
    end
  end
end

@testset "no partial write when a block of an enclosing sum is checked" begin
  # The conjugate's blocks have different precisions and the step is subnormal in Float32. The
  # check of the whole tuple and the blocks that execute must agree, so that the sum around the
  # conjugate never writes a block and then rejects: either the call succeeds with the point of the
  # stored Float64 step, or nothing is written (aliased or not).
  g = 1f-40
  c = Conjugate(SeparableSum(NormL1(), Scaled(MoreauEnvelope(NormL1(), 1f0), g)))
  f = SeparableSum(SqrNormL2(1e40), c)
  x = ([2.0], ([2.0], Float32[g]))
  for op in (prox!, value_and_prox!)
    ref = outcome(op, f, x, ([0.0], ([0.0], Float32[0])), Float64(g))
    @test ref[1] === :ok          # the stored Float64 step is accepted, and so is the Float32 step
    a = outcome(op, f, x, ([0.0], ([0.0], Float32[0])), g)
    @test a[1] === :ok && isequal(a, ref)
    xa = deepcopy(x)
    op(xa, f, xa, g)              # aliased
    @test isequal(flat(xa), ref[2])
  end
  # a step that is rejected is rejected before the sum writes its first block, aliased or not
  for bad in (NaN32,), op in (prox!, value_and_prox!)
    a = outcome(op, f, x, ([0.0], ([0.0], Float32[0])), bad)
    @test a[1] === :err && a[2] === ArgumentError && all(==(7), a[3])
    xa = deepcopy(x)
    @test_throws ArgumentError op(xa, f, xa, bad)
    @test xa == x
  end
  # all Float32: the Float32 reciprocal overflows, and both the check and the execution reject it
  # before the first block of the sum is written
  c32 = Conjugate(SeparableSum(NormL1(), Scaled(MoreauEnvelope(NormL1(), 1f0), g)))
  f32 = SeparableSum(SqrNormL2(1f30), c32)
  x32 = (Float32[2], (Float32[2], Float32[g]))
  for op in (prox!, value_and_prox!)
    a = outcome(op, f32, x32, (Float32[0], (Float32[0], Float32[0])), g)
    @test a[1] === :err && a[2] === ArgumentError && all(==(7), a[3])
    xa = deepcopy(x32)
    @test_throws ArgumentError op(xa, f32, xa, g)
    @test xa == x32
  end
  # the two oracles store the same point, also where the step leaves the Float32 range
  for γ in (1f-40, 3f0, 1f30)
    cs = Conjugate(Scaled(SeparableSum(NormL1(), SqrNormL2()), γ))
    xs = (Float32[γ], Float32[γ])
    p = outcome(prox!, cs, xs, (zeros(Float32, 1), zeros(Float32, 1)), γ)
    v = outcome(value_and_prox!, cs, xs, (zeros(Float32, 1), zeros(Float32, 1)), γ)
    @test p[1] === v[1]
    p[1] === :ok && @test p[2] == v[2]
  end
  # Known gap (R3), as on the base of #576: the Float32 reciprocal of 1f-40 overflows before the
  # scale 1f-40 of the sum is applied, so the quadratic block is g instead of g/2 although the
  # combined step g/g = 1 is normal. It is tracked on its own issue; the oracles agree on it.
  g2 = 1f-40
  cs = Conjugate(Scaled(SeparableSum(NormL1(), SqrNormL2()), g2))
  ys = (zeros(Float32, 1), zeros(Float32, 1))
  prox!(ys, cs, (Float32[g2], Float32[g2]), g2)
  @test_broken ys[2][1] == g2 / 2
end

@testset "the data of a function widen the model" begin
  # f(t) = t^2/2 + t + c with Float64 data under Float32 arrays: the dual proximal point at 0 is
  # γ/(1 + γ), representable and without cancellation, for every Float32 step including 1f-40 whose
  # Float32 reciprocal overflows; the dual value at the stored point is (y - 1)^2/2 - c
  Q = QuadraticFunction(reshape([1.0], 1, 1), [1.0])
  # the data in Float32 and the constant in Float64: the model of A, b and c is Float64
  Qc = QuadraticFunction(reshape(Float32[1], 1, 1), Float32[1], 0.5)
  for (name, f, c0) in (("Float64 data", Q, 0.0), ("Float64 constant only", Qc, 0.5))
    for γ in (3f0, 0.1f0, 1f-40), op in (prox!, value_and_prox!)
      tol = γ < 1f-30 ? 1e-4 : 4 * eps(Float32)
      ref = big(γ) / (1 + big(γ))
      y = Float32[7]
      v = op(y, Conjugate(f), Float32[0], γ)
      @test y[1] isa Float32
      @test relerr(y[1], ref) <= tol
      op === value_and_prox! && @test relerr(v, (big(y[1]) - 1)^2 / 2 - big(c0)) <= 16 * eps(Float32) + 1e-12 && v isa Float32
      # the stored Float64 step gives the same bits
      y64 = Float32[7]
      v64 = op(y64, Conjugate(f), Float32[0], Float64(γ))
      @test y64 == y && isequal(v64, v)
      # inside a tuple, next to a block without data
      ct = Conjugate(SeparableSum(NormL1(), f))
      yt, xt = (Float32[7], Float32[7]), (Float32[2], Float32[0])
      op(yt, ct, xt, γ)
      @test yt[1] == Float32[1] && relerr(yt[2][1], ref) <= tol
      # wrapped
      yw = Float32[7]
      op(yw, Conjugate(Translate(f, [0.0])), Float32[0], γ)
      @test relerr(yw[1], ref) <= tol
      # inside slices (a tuple, and a vector of functions)
      for sl in (SliceSum(f), SliceSum([f]))
        ys = fill(7f0, 1, 1)
        op(ys, Conjugate(sl), zeros(Float32, 1, 1), γ)
        @test relerr(ys[1], ref) <= tol
      end
    end
  end
  # an operator that is asked for its element type on the instance, as LinearImage and Precompose do
  # (the interface of test/image.jl), with Float64 entries under Float32 arrays
  Lm = ReviewMap(reshape([1.0], 1, 1))
  @test eltype(typeof(Lm)) === Any && eltype(Lm) === Float64
  fi = LinearImage(SqrNormL2(1f0, Float32[1]), Lm, 1f0)
  @test PB._data_of(fi) === Float64
  @test PB._data_of(Conjugate(Scaled(fi, 2.0))) === Float64
  @test PB._data_of(SeparableSum(NormL1(), fi)) === Float64
  @test PB._data_of(Precompose(SqrNormL2(), Lm, 1.0)) === Float64
  for γ in (3f0, 1f-40), op in (prox!, value_and_prox!)
    y = Float32[7]
    v = op(y, Conjugate(fi), Float32[0], γ)
    ref = -big(γ) / (1 + big(γ))
    @test relerr(y[1], ref) <= (γ < 1f-30 ? 1e-4 : 4 * eps(Float32))
    y64 = Float32[7]
    v64 = op(y64, Conjugate(fi), Float32[0], Float64(γ))
    @test y64 == y && isequal(v64, v)
  end
  # a vector of a union of concrete types has the data of every arm; an abstractly typed vector says
  # nothing from its type (a known gap, issue #597)
  Qd = QuadraticFunction(Diagonal([1.0]), [1.0])
  fu = SliceSum(Union{typeof(Q), typeof(Qd)}[Q, Qd])
  @test PB._data_of(fu) === Float64
  for γ in (3f0, 1f-40), op in (prox!, value_and_prox!)
    y = fill(7f0, 1, 2)
    op(y, Conjugate(fu), zeros(Float32, 1, 2), γ)
    ref = big(γ) / (1 + big(γ))
    @test all(i -> relerr(y[i], ref) <= (γ < 1f-30 ? 1e-4 : 4 * eps(Float32)), 1:2)
  end
  # a vector of one concrete type keeps the instance-level answer of its functions
  fv = SliceSum([fi])
  @test PB._data_of(fv) === Float64 && PB._data_of(SliceSum([Scaled(fi, 2.0)])) === Float64
  for γ in (3f0, 1f-40), op in (prox!, value_and_prox!)
    y = fill(7f0, 1, 1)
    v = op(y, Conjugate(fv), zeros(Float32, 1, 1), γ)
    y64 = fill(7f0, 1, 1)
    v64 = op(y64, Conjugate(fv), zeros(Float32, 1, 1), Float64(γ))
    @test relerr(y[1], -big(γ) / (1 + big(γ))) <= (γ < 1f-30 ? 1e-4 : 4 * eps(Float32))
    @test y == y64 && isequal(v, v64)
  end
  # Known gap (issue #597): an operator type whose element type differs between instances
  M6 = Union{Matrix{Float32}, Matrix{Float64}}
  fa32 = LinearImage(SqrNormL2(1f0, Float32[1]), ReviewMap{M6}(reshape(Float32[1], 1, 1)), 1f0)
  fa64 = LinearImage(SqrNormL2(1f0, Float32[1]), ReviewMap{M6}(reshape(Float64[1], 1, 1)), 1f0)
  @test typeof(fa32) === typeof(fa64)
  @test_broken PB._data_of(SliceSum([fa32, fa64])) === Float64
  fa = SliceSum(QuadraticFunction[Q])
  @test_broken PB._data_of(fa) === Float64
  # the selector
  @test PB._data_type(typeof(Qc)) === Float64
  @test PB._data_type(typeof(SliceSum(Q))) === Float64 && PB._data_type(typeof(SliceSum([Q]))) === Float64
  @test PB._data_type(typeof(SliceSum(NormL1(), SqrNormL2()))) === Union{}
  @test PB._data_type(typeof(SliceSum(NormL1(), Q))) === Float64
  @test PB._data_type(typeof(IndIntersection(IndBallL2(1.0), IndHyperplane(Float64[1, 1], 1.0)))) === Float64
  @test PB._data_type(typeof(Spectral(NormL1()))) === Union{}
  @test PB._data_float(Matrix{Float32}) === Float32 && PB._data_float(Nothing) === Union{} &&
        PB._data_float(typeof(identity)) === Union{} && PB._data_float(Float64) === Float64
  # Float32 data keep the Float32 model and its plain reciprocal
  Q32 = QuadraticFunction(reshape([1f0], 1, 1), [1f0])
  @test_throws ArgumentError prox!(Float32[7], Conjugate(Q32), Float32[0], 1f-40)
  # a Float32 model of data and arrays: allocations stay zero
  x32, y32 = Float32[0.5, 1], zeros(Float32, 2)
  Q2 = QuadraticFunction([2.0 0.5; 0.5 1.0], [1.0, -1.0])
  @test allocs(prox!, y32, Conjugate(Q2), x32, 0.5f0) == 0
  @test allocs(value_and_prox!, y32, Conjugate(Q2), x32, 0.5f0) == 0
end

@testset "tuples beyond 32 blocks keep the data selector inferred and allocation free" begin
  Q = QuadraticFunction(reshape([1.0], 1, 1), [1.0])
  for n in (33, 40)
    fs = ntuple(k -> iseven(k) ? SqrNormL2() : Translate(Scaled(Q, 2f0), [0.0]), n)
    f = SeparableSum(fs...)
    c = Conjugate(f)
    x = ntuple(_ -> Float32[0], n)
    y = ntuple(_ -> zeros(Float32, 1), n)
    @test @inferred(PB._data_of(f)) === Float64
    @test @inferred(PB._moreau_reciprocal(f, y, x, 3f0)) === inv(3.0)
    @test allocs(prox!, y, c, x, 3f0) == 0
    @test allocs(value_and_prox!, y, c, x, 3f0) == 0
  end
end

@testset "device arrays: the lazy reciprocal converts each step before inverting" begin
  # JLArrays dispatch to the extension of GPUArraysCore: Float64 argument and output, Float32 step array
  c = Conjugate(ElasticNet(0.5, 2.0))
  x = Float64[2, 3]
  γ = Float32[3, 0.1]
  refs = [ref_en(x[i], γ[i]) for i in 1:2]
  for op in (prox!, value_and_prox!)
    y = JLArray(zeros(2))
    op(y, c, JLArray(x), JLArray(γ))
    @test all(i -> relerr(Array(y)[i], refs[i]) <= 8 * eps(Float64), 1:2)
    yh = zeros(2)
    op(yh, c, x, γ)
    @test Array(y) == yh
    y64 = JLArray(zeros(2))
    op(y64, c, JLArray(x), JLArray(Float64.(γ)))
    @test Array(y64) == Array(y)
  end
end

@testset "unchanged shortcuts" begin
  # direct projections never form a reciprocal, and a step of any type gives the same point
  x = [0.5, -2.0, 1.5]
  for f in (NormL1(), NormL1(0.7, [1.0, 2.0, 3.0]), IndNonnegative(), NormL2(), IndBox(-1.0, 1.0), IndBallL2(2.0))
    c = Conjugate(f)
    @test prox!(zeros(3), c, x, 0.5f0) == prox!(zeros(3), c, x, Float64(0.5f0))
    @test prox!(zeros(3), c, x, 3f0) == prox!(zeros(3), c, x, 3.0)
    @test allocs(prox!, zeros(3), c, x, 3f0) == 0
  end
  @test prox!(zeros(3), Conjugate(Conjugate(NormL1(0.5))), x, 3f0) == prox!(zeros(3), NormL1(0.5), x, 3f0)
end

@testset "the closed form of the conjugate of the quadratic" begin
  # prox_{γf*}(x) = λ(x - γy)/(λ + γ) for f = SqrNormL2(λ, y), from the stored parameters
  ref(λ, x, γ, y = 0) = big(λ) * (big(x) - big(γ) * big(y)) / (big(λ) + big(γ))
  ulp_error(T, got, r) = Float64(abs(big(got) - r) / (abs(T(r)) < floatmin(T) ? big(nextfloat(zero(T))) : big(eps(T(r)))))
  setprecision(BigFloat, 400) do
    # Moreau's identity cancels where γ is much larger than λ; the closed form does not
    for (λ, γ) in ((1e-12, 1.0), (1.0, 1e10), (1e-300, 1.0), (1.0, 1e-12), (3.0, 3.0))
      x = [1.0, -2.0, 0.3, 1e10]
      y = zeros(4)
      prox!(y, Conjugate(SqrNormL2(λ)), x, γ)
      @test all(i -> ulp_error(Float64, y[i], ref(λ, x[i], γ)) <= 2, 1:4)
      c = [0.5, -1.0, 0.25, 3.0]
      prox!(y, Conjugate(SqrNormL2(λ, c)), x, γ)
      @test all(i -> ulp_error(Float64, y[i], ref(λ, x[i], γ, c[i])) <= 2, 1:4)
    end
    # a Float32 step and a Float64 level are parameters rounded once to the model
    for γ in (0.1f0, 3f0, 1f-40, 1f30)
      x = [2.0, 3.0]
      y = zeros(2)
      prox!(y, Conjugate(SqrNormL2(0.7)), x, γ)
      @test all(i -> ulp_error(Float64, y[i], ref(0.7, x[i], γ)) <= 2, 1:2)
      @test y == prox!(zeros(2), Conjugate(SqrNormL2(0.7)), x, Float64(γ))
      x32 = Float32[2, 3]
      y32 = zeros(Float32, 2)
      prox!(y32, Conjugate(SqrNormL2(0.7f0)), x32, γ)
      @test all(i -> ulp_error(Float32, y32[i], ref(0.7f0, x32[i], γ)) <= 2, 1:2)
      @test eltype(y32) === Float32
    end
    # the model precision is the wider of the argument's and the destination's
    for (Ty, Tx) in ((Float64, Float32), (Float32, Float64))
      x = Tx[2, -3, 1e-3]
      y = zeros(Ty, 3)
      prox!(y, Conjugate(SqrNormL2(0.7)), x, 0.1)
      @test all(i -> ulp_error(Ty, y[i], ref(0.7, x[i], 0.1)) <= 2, 1:3)
    end
    # a Float64 center outside the range of a Float32 model: γy = 3 although y is not a Float32
    c = Conjugate(SqrNormL2(1f0, [3 * 2.0^993]))
    y = zeros(Float32, 1)
    prox!(y, c, Float32[1], 2.0^-993)
    @test y == Float32[-2]
    # a level and a step whose ratio is outside the range of the model
    for (λ, γ, x) in ((1e-300, 1e300, 1e300), (1e300, 1e-300, 3.0), (2.0^-1000, 2.0^500, 2.0^900))
      y = zeros(1)
      prox!(y, Conjugate(SqrNormL2(λ)), [x], γ)
      @test ulp_error(Float64, y[1], ref(λ, x, γ)) <= 2
    end
    # prox_block is the same map
    for γ in (1e-10, 1.0, 1e10, 1e300), v in (1e300, -3.0, 1e-300)
      f = SqrNormL2(2.0, [0.25])
      e = prox_block(Conjugate(f), v, 1, γ)
      @test ulp_error(Float64, e, ref(2.0, v, γ, 0.25)) <= 2
      @test e == prox!(zeros(1), Conjugate(f), [v], γ)[1]
    end
  end
end

@testset "allocations" begin
  en = ElasticNet(0.5, 2.0)
  for (Ty, Tx, γ) in ((Float64, Float64, 3f0), (Float64, Float32, 3f0), (Float32, Float32, 3f0), (Float32, Float32, 3.0), (Float64, Float64, 0.1f0))
    x = Tx[2, 3, -1]
    y = zeros(Ty, 3)
    for c in (Conjugate(en), Conjugate(SqrNormL2()), Conjugate(Scaled(en, 2.0)))
      @test allocs(prox!, y, c, x, γ) == 0
      @test allocs(value_and_prox!, y, c, x, γ) == 0
    end
    γa = fill(Float32(γ), 3)
    @test allocs(prox!, y, Conjugate(en), x, γa) == 0
    @test allocs(value_and_prox!, y, Conjugate(en), x, γa) == 0
    for c in (Conjugate(SeparableSum(NormL1(), SeparableSum(SqrNormL2()))), Conjugate(SeparableSum(en, SqrNormL2())))
      bx = c.f.fs[1] isa NormL1 ? (x, (copy(x),)) : (x, copy(x))
      by = c.f.fs[1] isa NormL1 ? (y, (copy(y),)) : (y, copy(y))
      @test allocs(prox!, by, c, bx, γ) == 0
      @test allocs(value_and_prox!, by, c, bx, γ) == 0
    end
  end
  # prox_block in a loop
  function entries!(out, c, v, γ)
    @inbounds for i in eachindex(out, v)
      out[i] = prox_block(c, v[i], i, γ)
    end
    out
  end
  for (Tv, γ) in ((Float64, 3f0), (Float32, 3f0), (Float32, 3.0), (Float64, 0.1f0))
    v, out = Tv[2, 3, -1], zeros(Tv, 3)
    for c in (Conjugate(ElasticNet(0.5, 2.0)), Conjugate(Scaled(ElasticNet(0.5, 2.0), 2.0)),
              Conjugate(Translate(ElasticNet(0.5, 2.0), [0.5, 0.25, 0.1])))
      @test allocs(entries!, out, c, v, γ) == 0
    end
  end
end

end
