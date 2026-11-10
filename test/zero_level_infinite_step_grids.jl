module ZeroLevelInfiniteStepGridsTest

using ..TestUtils: take_block_step!

# The Float32 pass of Regularize with a zero level, for the functions whose Float64 pass stays in
# zero_level_infinite_step.jl: in the loop over functions NormL1 with weights, IndNonnegative,
# Constrained, the two CappedL1 and SqrNormL2(0); in array steps NormL1 with weights, IndNonnegative and
# CappedL1 with weights; in per-group steps and the coordinate step the three functions after the first.
# zero_level_infinite_step.jl runs Float32 for the other functions of those loops. The helpers are
# copied from it.

using Test, ProximalBase

const PB = ProximalBase

same(a, b) = isequal(a, b)

# the class of a value: NaN, an infinity of either sign, or finite
value_class(v) = isnan(v) ? :nan : isinf(v) ? (v > 0 ? :pinf : :ninf) : :finite

function check_regularize(f, x::AbstractVector{T}, γ; c = nothing, ρs = (0, T(0), 0.0, T(-0.0), -0.0)) where {T}
  want = prox(f, x, γ)
  yf = fill(T(7), length(x))
  vf = value_and_prox!(yf, f, x, γ)
  entry(s, i) = s isa AbstractArray ? s[i] : s
  for ρ in ρs
    r = Regularize(f, ρ, c)
    @test same(prox(r, x, γ), want)
    y = fill(T(7), length(x))
    @test prox!(y, r, x, γ) === y && same(y, want)
    z = copy(x)
    @test prox!(z, r, z, γ) === z && same(z, want)
    y = fill(T(7), length(x))
    v = value_and_prox!(y, r, x, γ)
    @test same(y, yf)
    all(isfinite, y) && @test same(v, vf)
    @test value_class(v) == value_class(r(y))            # the fused value is the value at the stored point
    z = copy(x)
    v = value_and_prox!(z, r, z, γ)
    @test same(z, yf)
    all(isfinite, z) && @test same(v, vf)
    if PB.isseparable(f)
      @test same([prox_block(r, x[i], i, entry(γ, i)) for i in eachindex(x)],
                 [prox_block(f, x[i], i, entry(γ, i)) for i in eachindex(x)])
    end
  end
end

@testset "Regularize with a zero level, $T" for T in (Float32,)
  Inf_ = T(Inf)
  x = T[2, -3, 0.5, 0, -0.0, 1, -1, 0.25]
  xn = T[2, NaN, Inf, -Inf, -0.0, 0, 1, NaN]
  xc = T[2, -3, 0.5, 0, 0, 1, -1, 0.25]                 # for a center: no -0
  c = T[1, -2, 0.5, 3, 0, -1, 1, 2]
  fs = (NormL1(T(0.5), T[0, 1, Inf, 2, 1, 1, 0.5, 4]), IndNonnegative(), Constrained(NormL1(T(1)), IndBallL2(T(1))),
        CappedL1(T(0), T(1)), CappedL1(T(1), T(1), T[0, 1, Inf, 2, 1, 1, 0.5, 4]), SqrNormL2(T(0)))
  @testset "$(nameof(typeof(f)))" for f in fs
    # a NaN or infinite entry is the argument of f itself, the weights (1, 0) leave it unchanged
    check_regularize(f, x, Inf_)
    check_regularize(f, xn, Inf_)
    check_regularize(f, xc, Inf_; c = c)
    check_regularize(f, x, one(T))               # finite steps, as before
    check_regularize(f, xc, T(0.5); c = c)
  end
  @testset "array steps" begin
    for f in (NormL1(T(0.5), T[0, 1, Inf, 2, 1, 1, 0.5, 4]), IndNonnegative(),
              CappedL1(T(1), T(1), T[0, 1, Inf, 2, 1, 1, 0.5, 4]))
      for γ in (T[Inf, 1, 0.5, Inf, 1, Inf, 2, 0.25], fill(Inf_, 8), T[Inf, 1, 0.5, 0, 1, 0, 2, 0.25])
        check_regularize(f, x, γ)
        check_regularize(f, xc, γ; c = c)
      end
    end
  end
  @testset "per-group steps and the coordinate step" begin
    groups = [1:2, 3:5, 6:8]
    for f in (GroupNormL2(T(0.5), groups, T[0, 1, Inf]),
              GroupNormLinf(T(0.5), groups), SparseGroupNormL2(T(0.2), T(0.4), groups))
      for γ in (T[Inf, 1, 0.5], T[Inf, Inf, Inf], T[1, Inf, 2], T[0.5, 1, 2])
        check_regularize(f, x, γ)
        check_regularize(f, xc, γ; c = c)
      end
      for k in 1:3, centered in (false, true)
        rc = centered ? c : nothing
        for start in (centered ? xc : x, T[0, 0, 0, 0, 0, 1, -1, 0.25])
          for γ in (Inf_, one(T))
            xf = BlockIterate(copy(start), groups)
            xr = BlockIterate(copy(start), groups)
            take_block_step!(xf, f, k, γ)
            take_block_step!(xr, Regularize(f, 0, rc), k, γ)
            @test same(Vector(xr), Vector(xf))
            @test activecoordinates(xr) == activecoordinates(xf)
            @test same(xr.storage, xf.storage)
          end
        end
      end
    end
  end
end

end
