module ScaledInfiniteLevelUnderflowGridsTest

# The cases of "the listed indicators are accepted" that scaled_infinite_level_underflow.jl does not
# run. Float32: the seventeen indicators other than IndBox, IndSimplex and IndPattern, alone and under
# Scaled and Translate(Scaled). Float64: the Scaled and Translate(Scaled) variants of the sixteen
# indicators other than IndBox, IndSimplex, IndSOC and IndPattern. scaled_infinite_level_underflow.jl
# keeps the rest, including all twenty indicators alone in Float64. The helper z0 is copied from it.

using Test, ProximalBase

const PB = ProximalBase

# adding a zero turns a stored -0 into +0, so two results are compared up to the sign of a zero
z0(a) = a .+ zero(eltype(a))

# An indicator of the list, alone, translated and under a positive finite factor, next to a hard block: the hard
# block is the hard zero and the call is that of the same function at an ordinary step.
@testset "the listed indicators are accepted, $T" for T in (Float32, Float64)
  tiny = T === Float64 ? T(1e-300) : T(1f-30)
  one3 = T[1, 1, 1]
  listed = Any[(IndBox(T(-1), T(1)), T[3, -4, 0.5]), (IndNonnegative(), T[3, -4, 0.5]), (IndBinary(T(0), T(1)), T[3, -4, 0.5]),
               (IndBallL0(1), T[3, -4, 0.5]), (IndBallL1(T(1)), T[3, -4, 0.5]), (IndBallL2(T(1)), T[3, -4, 0.5]), (IndSphereL2(T(1)), T[3, -4, 0.5]),
               (IndSimplex(T(1)), T[3, -4, 0.5]), (IndHyperplane(one3, T(1)), T[3, -4, 0.5]), (IndHalfspace(one3, T(1)), T[3, -4, 0.5]),
               (IndHyperslab(one3, T(0), T(1)), T[3, -4, 0.5]), (IndSOC(), T[3, -4, 0.5]), (IndRotatedSOC(), T[3, -4, 0.5]),
               (IndExpPrimal(), T[3, -4, 0.5]), (IndExpDual(), T[3, -4, 0.5]), (IndIsotonic(), T[3, -4, 0.5]), (IndBallGroupL0(1, [1:3]), T[3, -4, 0.5]),
               (IndPattern(BitArray([1 0; 0 1])), T[1 2; 3 4]), (IndBanded(0, 0, 2, 2), T[1 2; 3 4]), (IndToeplitz(), T[1 2; 3 4])]
  for (ind, x2) in listed
    # Float32 runs three indicators and Float64 the wrapper variants of four in the thorough file
    T === Float32 && ind isa Union{IndBox, IndSimplex, IndPattern} && continue
    T === Float64 && ind isa Union{IndBox, IndSimplex, IndSOC, IndPattern} && continue
    T === Float32 && @test PB._hard_function(ind) === ind
    shift = x2 isa AbstractMatrix ? zeros(T, 2, 2) : zeros(T, 3)
    variants = (Scaled(ind, T(2)), Translate(Scaled(ind, T(3)), shift))
    for blk in (T === Float32 ? (ind, variants...) : variants)
      inner = SeparableSum(NormL1(T(Inf)), blk)
      arg = (T[1, 2], x2)
      ref = try prox(inner, arg, one(T)) catch e; e end
      ref isa Exception && continue
      f = Scaled(inner, tiny)
      @test PB._hard_function(inner) !== nothing
      res = prox(f, arg, tiny)
      @test res[1] == T[0, 0]
      @test isequal(z0(res[1]), z0(ref[1])) && isequal(res[2], ref[2])
      y = (zeros(T, 2), similar(x2))
      value_and_prox!(y, f, arg, tiny)
      @test isequal(z0(y[1]), z0(ref[1])) && isequal(y[2], ref[2])
    end
  end
end

end
