module RidgePrecisionTest

# SqrNormL2 under the numerical contract (T3, R2, R3, O1, O2, O7): the model
# precision of each result, range recovery, and agreement between the in-place,
# allocating, coordinate and fused methods. References are exact BigFloat
# evaluations of the formulas, not other package oracles.

using Test, ProximalBase, SparseArrays

pow2(T, k) = ldexp(one(T), k)
const BIG = 400   # bits: enough for every product below

# a is within n ulps of the exact reference r (a normal number in type T)
near(a::T, r, n) where {T} = abs(big(a) - r) <= n * eps(abs(T(r)))
function normal(T, r)
  a = abs(r)
  floatmin(T) <= a <= floatmax(T)
end

setprecision(BigFloat, BIG) do

  ref_prox(W, x, c, γ, λ) = (t = big(γ) * big(λ); (big(x) + t * big(c)) / (1 + t))
  ref_grad(W, x, c, λ) = big(λ) * (big(x) - big(c))

  @testset "cases from the contract" begin
    # A: a wider argument keeps its effect in a narrower gradient
    x = Float64[1 + pow2(Float64, -28)]
    g = zeros(Float32, 1)
    gradient!(g, SqrNormL2(1, Float32[1]), x)
    @test g[1] === pow2(Float32, -28)

    # B: the center's coefficient survives when γλ underflows
    z = zeros(Float32, 1)
    prox!(z, SqrNormL2(pow2(Float32, -100), Float32[pow2(Float32, 100)]), Float32[0], pow2(Float32, -60))
    @test z[1] === pow2(Float32, -60)
    @test prox_entry(SqrNormL2(pow2(Float32, -100), Float32[pow2(Float32, 100)]), 0f0, 1, pow2(Float32, -60)) === pow2(Float32, -60)

    # V3 to V6, uncentered and centered
    @test prox!(zeros(Float32, 1), SqrNormL2(pow2(Float64, 160)), Float32[1], pow2(Float64, -160)) == Float32[0.5]
    @test prox!(zeros(Float32, 1), SqrNormL2(pow2(Float64, 160), Float32[0]), Float32[1], pow2(Float64, -160)) == Float32[0.5]
    @test prox!(zeros(Float32, 1), SqrNormL2(pow2(Float64, 160)), [pow2(Float32, 100)], 1.0) == [pow2(Float32, -60)]
    @test prox!(zeros(1), SqrNormL2(pow2(Float64, 800)), [pow2(Float64, 900)], pow2(Float64, 800)) == [pow2(Float64, -700)]
    @test gradient!(zeros(Float32, 1), SqrNormL2(pow2(Float64, 160)), [pow2(Float32, -80)]) == [pow2(Float32, 80)]
  end

  @testset "prox: sweep against exact references" begin
    # (W, factor type, exponents of the factors, exponents of the data)
    sweeps = ((Float32, Float64, (-200, -100, -60, -1, 0, 1, 60, 100, 200), (0, 100, -100, 127)),
              (Float32, Float32, (-100, -60, -1, 0, 1, 60, 100), (0, 100, -100, 127)),
              (Float64, Float64, (-1000, -800, -400, -1, 0, 1, 400, 800, 1000), (0, 900, -900, 1022)))
    for (W, F, ks, ds) in sweeps, kγ in ks, kλ in ks, kx in ds, kc in ds, sx in (1, -1), sc in (1, -1)
      γ, λ = pow2(F, kγ), pow2(F, kλ)
      isfinite(γ) && isfinite(λ) || continue
      x = [W(sx) * pow2(W, kx)]
      c = [W(sc) * pow2(W, kc)]
      for centered in (false, true)
        r = ref_prox(W, x[1], centered ? c[1] : 0, γ, λ)
        normal(W, r) || continue
        f = SqrNormL2(λ, centered ? c : nothing)
        z = zeros(W, 1)
        prox!(z, f, x, γ)
        @test near(z[1], r, 3)
        @test prox_entry(f, x[1], 1, γ) === z[1]
        @test prox(f, x, γ)[1] === z[1]
        xa = copy(x)
        prox!(xa, f, xa, γ)                 # supported in-place call
        @test xa[1] === z[1]
      end
    end
  end

  @testset "prox: extremes of the data and a wide center" begin
    # t = 2^200 in Float32 with the largest coordinate: no overflow in the small weight
    for s in (1f0, -1f0)
      x = [s * floatmax(Float32)]
      z = prox!(zeros(Float32, 1), SqrNormL2(pow2(Float64, 200)), x, 1.0)
      @test near(z[1], ref_prox(Float32, x[1], 0, 1.0, pow2(Float64, 200)), 3)
    end
    # a Float64 center outside Float32's range, ordinary coefficient and scaled coefficient
    for (λ, γ) in ((pow2(Float64, -100), 1.0), (pow2(Float64, -160), 1.0), (pow2(Float64, -40), pow2(Float64, -100)))
      x, c = Float32[1], Float64[pow2(Float64, 200)]
      f = SqrNormL2(λ, c)
      r = ref_prox(Float32, x[1], c[1], γ, λ)
      normal(Float32, r) || continue
      z = prox!(zeros(Float32, 1), f, x, γ)
      @test near(z[1], r, 3)
      @test prox_entry(f, x[1], 1, γ) === z[1]
    end
    # Float32 data, Float64 center that rounds ordinarily: the converted problem is solved
    z = prox!(zeros(Float32, 1), SqrNormL2(1.0, [1 + pow2(Float64, -28)]), Float32[1], 1.0)
    @test z == Float32[1]
    # a wider argument into a narrower destination keeps its digits
    x = Float64[1 + pow2(Float64, -40)]
    z = prox!(zeros(Float32, 1), SqrNormL2(pow2(Float64, -60), Float64[0]), x, pow2(Float64, 60))
    @test near(z[1], ref_prox(Float64, x[1], 0, pow2(Float64, 60), pow2(Float64, -60)), 1)
  end

  @testset "gradient: sweep and overflow of the difference" begin
    sweeps = ((Float32, Float64, (-200, -100, -1, 0, 1, 100, 200), (0, 100, -100, 127)),
              (Float64, Float64, (-1022, -800, -1, 0, 1, 800, 1000), (0, 900, -900, 1023)))
    for (W, F, ks, ds) in sweeps, kλ in ks, kx in ds, kc in ds, sx in (1, -1), sc in (1, -1)
      λ = pow2(F, kλ)
      x = [W(sx) * pow2(W, kx)]
      c = [W(sc) * pow2(W, kc)]
      for centered in (false, true)
        r = ref_grad(W, x[1], centered ? c[1] : 0, λ)
        normal(W, r) || continue
        f = SqrNormL2(λ, centered ? c : nothing)
        g = zeros(W, 1)
        gradient!(g, f, x)
        @test near(g[1], r, 2)
        gs = fill(W(7), 1)
        v = value_and_gradient!(gs, f, x)
        @test gs[1] === g[1]
        @test gradient(f, x)[1] === g[1]
      end
    end
    # V7: the difference overflows, the gradient does not, in the same call
    f = SqrNormL2(pow2(Float64, -1022), [-pow2(Float64, 1023)])
    g = fill(7.0, 1)
    @test value_and_gradient!(g, f, [pow2(Float64, 1023)]) == Inf
    @test g == [4.0]
    f32 = SqrNormL2(pow2(Float32, -126), Float32[-pow2(Float32, 127)])
    g32 = fill(7f0, 1)
    value_and_gradient!(g32, f32, Float32[pow2(Float32, 127)])
    @test g32 == Float32[4.0]
    # a wider argument into a narrower gradient: λ rounds to the wider model
    g = zeros(Float32, 1)
    gradient!(g, SqrNormL2(1 + pow2(Float64, -40)), Float64[1])
    @test g[1] === Float32(1 + pow2(Float64, -40))
  end

  @testset "fused calls: both outputs from one invocation" begin
    # V11: value in the argument's model, gradient in the gradient's
    g = fill(7.0, 1)
    v = value_and_gradient!(g, SqrNormL2(1.0, [1 + pow2(Float64, -28)]), Float32[1])
    @test v === 0f0
    @test g == [-pow2(Float64, -28)]
    # V9: value at the stored point, destination written by the same call
    y = fill(7f0, 1)
    v = value_and_prox!(y, SqrNormL2(1.0, [1 + pow2(Float64, -28)]), Float32[1], 1.0)
    @test v === 0f0
    @test y == Float32[1]
    # matching precision: the fused call agrees with the standalone calls
    for (T, λ, γ) in ((Float32, 0.3f0, 0.7f0), (Float64, 0.3, 0.7), (Float32, 0.3, 0.7))
      x = T[1.5, -2, 0.25]
      c = T[0.5, 1, -3]
      for f in (SqrNormL2(λ), SqrNormL2(λ, c))
        gs = fill(T(7), 3)
        v = value_and_gradient!(gs, f, x)
        @test gs == gradient!(zeros(T, 3), f, x)
        @test v === f(x)
        ys = fill(T(7), 3)
        v = value_and_prox!(ys, f, x, γ)
        z = prox!(zeros(T, 3), f, x, γ)
        @test ys == z
        @test v === f(ys)
        @test [prox_entry(f, x[i], i, γ) for i in 1:3] == z
        xa = copy(x)
        prox!(xa, f, xa, γ)
        @test xa == z
      end
    end
  end

  @testset "value: a Float64 center does not widen a Float32 value" begin
    f = SqrNormL2(1.0, [1 + pow2(Float64, -28)])
    @test f(Float32[1]) === 0f0
    # a center outside Float32's range is used unrounded (R3)
    f = SqrNormL2(pow2(Float64, -400), [pow2(Float64, 200)])
    @test f(Float32[0]) === 0.5f0
    # entries that fit are still rounded when another entry does not
    f = SqrNormL2(pow2(Float64, -400), [pow2(Float64, 200), 1 + pow2(Float64, -28)])
    @test f(Float32[0, 1]) === 0.5f0
    # the centered value agrees with the dense reference for a sparse argument
    c = Float64[1.5, -2, 0.25, 4]
    xs = sparsevec([2, 4], Float32[1, 3], 4)
    @test SqrNormL2(0.5, c)(xs) === SqrNormL2(0.5, c)(Vector(xs))
    # ordinary value unchanged in the other mixes
    @test SqrNormL2(0.5f0, Float32[1])(Float64[2]) === 0.25
    @test SqrNormL2(0.5, Float32[1])(Float32[2]) === 0.25f0
  end
  @testset "dense paths: one center entry leaves Float32's range" begin
    n = 9
    x = Float32[0.5, -1, 2, 3, -4, 0.25, 8, -0.125, 1]
    for (k, label) in ((200, "overflows"), (-160, "underflows"))
      c = Float64[0.3, -1.5, 2, 1, 0.75, -2, 1.25, 0.5, -3]
      c[4] = pow2(Float64, k)
      cw = [Float64(Float32(ci)) for ci in c]
      cw[4] = c[4]                          # the entry that cannot be rounded is used as given
      # gradient with a level that fits Float32
      λ = k > 0 ? pow2(Float64, -100) : pow2(Float64, 100)
      g = fill(7f0, n)
      gradient!(g, SqrNormL2(λ, c), x)
      for i in 1:n
        r = big(λ) * (big(x[i]) - big(cw[i]))
        normal(Float32, r) ? @test(near(g[i], r, 2)) : nothing
      end
      # prox, aliased and separate
      γ = 1.0
      z = zeros(Float32, n)
      prox!(z, SqrNormL2(λ, c), x, γ)
      xa = copy(x)
      prox!(xa, SqrNormL2(λ, c), xa, γ)
      @test xa == z
      for i in 1:n
        r = ref_prox(Float32, x[i], cw[i], γ, λ)
        normal(Float32, r) ? @test(near(z[i], r, 3)) : nothing
      end
      # value
      v = SqrNormL2(λ, c)(x)
      r = big(λ) / 2 * sum((big(x[i]) - big(cw[i]))^2 for i in 1:n)
      normal(Float32, r) ? @test(near(v, r, 4)) : nothing
    end
  end
  @testset "a center that underflows on rounding is used unrounded" begin
    # x - c = -2^-150 exactly; rounding c to a Float32 subnormal would give -2^-149 or 0
    x = fill(1f0, 9); x[4] = pow2(Float32, -149)
    c = fill(1.0, 9); c[4] = 1.5 * pow2(Float64, -149)
    λ = pow2(Float64, 100)
    g = zeros(Float32, 9)
    gradient!(g, SqrNormL2(λ, c), x)
    @test g[4] === -pow2(Float32, -50)
    @test all(==(0f0), g[[1:3; 5:9]])
    @test gradient(SqrNormL2(λ, c[4:4]), x[4:4])[1] === -pow2(Float32, -50)
  end
  @testset "review regressions" begin
    # the gradient may be written over the center: the repair pass must not reread it
    fm = floatmax(Float64)
    c = [-fm, 1.0, 2.0]
    r = gradient(SqrNormL2(pow2(Float64, -1020), copy(c)), [fm, 2.0, 3.0])
    @test gradient!(c, SqrNormL2(pow2(Float64, -1020), c), [fm, 2.0, 3.0]) == r
    @test isapprox(r[1], 32.0; rtol = 1e-14)
    # a BigFloat center that underflows on rounding to the model precision is used as given
    setprecision(BigFloat, 200) do
      g = zeros(2)
      gradient!(g, SqrNormL2(pow2(Float64, 1000), [BigFloat(2)^-1100, BigFloat(0.5)]), [0.0, 1.0])
      @test g[1] == -ldexp(1.0, -100)
      x32 = Float32[0, 1]
      g32 = zeros(Float32, 2)
      gradient!(g32, SqrNormL2(pow2(Float64, 100), [BigFloat(2)^-200, BigFloat(0.5)]), x32)
      @test g32[1] === -pow2(Float32, -100)
    end
    # long Float32 argument with a Float64 center: the sum does not lose digits
    x = rand(Float32, 10^6); c = rand(Float64, 10^6)
    ref = sum(abs2, Float64.(x) .- Float64.(Float32.(c))) / 2
    @test isapprox(SqrNormL2(1.0, c)(x), ref; rtol = 1e-6)
  end
  @testset "review of #411" begin
    # the fused call reports the value at the original center even when g is the center
    c = [1.0]
    f = SqrNormL2(2.0, c)
    @test value_and_gradient!(c, f, [2.0]) == 1.0
    @test c == [2.0]
    # a BigFloat center is rounded to the argument's precision: the value is the Float32 problem's
    setprecision(BigFloat, 100) do
      @test SqrNormL2(1.0, [BigFloat(1) + BigFloat(2)^-28])(Float32[1]) === 0f0
      @test SqrNormL2(1.0, [BigFloat(1) + BigFloat(2)^-60])(Float64[1]) === 0.0
    end
    # t just inside and outside the band, with parameters that are not powers of two
    for (W, F, lo, hi) in ((Float32, Float64, -126, 125), (Float64, Float64, -1022, 1021), (Float32, Float32, -126, 125))
      for k in (lo - 1, lo, lo + 1, hi - 1, hi, hi + 1), m in (1.1, 1.7, 1.99), sgn in (1, -1)
        t = ldexp(m, k)
        for (γ, λ) in ((F(sqrt(t)), F(sqrt(t))), (F(t), one(F)), (one(F), F(t)))
          isfinite(γ) && isfinite(λ) && !iszero(γ) && !iszero(λ) || continue
          x = [W(sgn) * W(1.3) * W(2)^(k > 0 ? 20 : -20)]
          c = [W(-1.7) * W(2)^(k > 0 ? 20 : -20)]
          for f in (SqrNormL2(λ), SqrNormL2(λ, c))
            z = prox!(zeros(W, 1), f, x, γ)
            r = ref_prox(W, x[1], f.y === nothing ? 0 : c[1], γ, λ)
            normal(W, r) && @test near(z[1], r, 3)
          end
        end
      end
    end
  end
  @testset "BigFloat centers are rounded entry by entry" begin
    setprecision(BigFloat, 256) do
      x = Float32[1, 0]
      c = BigFloat[BigFloat(1) + BigFloat(2)^(-28), BigFloat(2)^(-200)]
      @test SqrNormL2(ldexp(1.0, 400), c)(x) === 0.5f0
    end
  end
end

end
