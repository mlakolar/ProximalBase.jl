module SymmetricWrappersGridsTest

# The Float32 passes of two testsets of symmetric_wrappers.jl that it leaves to the Float64 pass: "a separable sum
# with a rejection-only owner block writes no block", all of it but its "supported blocks" testsets, and
# "the exact check applies to every other combination" for Conjugate(Translate(NuclearNorm)),
# Conjugate(Regularize(NuclearNorm)) and MoreauEnvelope(Conjugate(NuclearNorm)). symmetric_wrappers.jl keeps the
# Float64 pass of both, the Float32 supported blocks, and the Float32 entries of the other three combinations.
# `state` and the testsets are copied from it, and the combinations are the three that it filters out of its
# Float32 pass. The fixtures are the same; the generators of the combination testset are created for each trial from
# the trial number and sizeof(T), so the draws of the entries that stay there do not change.

using Test, ProximalBase, LinearAlgebra, Random

state(x) = (Matrix(x), nactive(x), copy(activecoordinates(x)), copy(x.nzval), copy(x.nzval2ind), copy(x.ind2nzval))

# The block preflight of a separable sum asks, after the parameters are validated and before any block is written,
# whether the owner role of each block has a method: the three rejection-only functions end in the MethodError of
# the operation, with every block, owner or dense, as it was.
@testset "a separable sum with a rejection-only owner block writes no block, $T" for T in (Float32,)
  W = ones(T, 2, 2); W[2, 1] = 2                       # not symmetric
  rejecting = ("PointwiseMinimum" => PointwiseMinimum(NormL1(T(0.2)), SqrNormL2(T(0.3))),
               "IndIntersection" => IndIntersection(IndBox(-T(1), T(1)), IndBallL2(T(1))),
               "LinearConstrained" => Constrained(NormL1(T(0.2)), IndHalfspace(ones(T, 2, 2), T(1))))
  invalid = ("PointwiseMinimum" => PointwiseMinimum(NormL1(T(0.2), W), SqrNormL2(T(0.3))),
             "IndIntersection" => IndIntersection(IndBox(-W, T(1)), IndBallL2(T(1))),
             "LinearConstrained" => Constrained(NormL1(T(0.2)), IndHalfspace(W, T(1))))
  mkowner(v) = SymmetricSparseIterate(fill(T(v), 2, 2))
  point() = ones(T, 2, 2)
  tstate(t::Tuple) = map(tstate, t)
  tstate(y::SymmetricSparseIterate) = state(y)
  tstate(y) = copy(y)
  # the outputs and the points of a call, as a tuple of two or of a block and a nested pair
  roles = ("owner outputs, dense points" => () -> ((mkowner(1), mkowner(2)), (point(), point())),
           "dense outputs, owner points" => () -> ((fill(T(3), 2, 2), fill(T(4), 2, 2)), (mkowner(1), mkowner(2))),
           "owner and dense outputs, dense points" => () -> ((mkowner(1), fill(T(4), 2, 2)), (point(), point())))
  @testset "$name, $role, $where, $op" for (name, bad) in rejecting, (role, make) in roles,
                                           where in ("second", "first", "nested"), op in (prox!, value_and_prox!)
    nl1 = NormL1(T(0.5))
    y, x = make()                                       # the outputs and the points of two blocks
    if where == "second"
      f, ry, rx = SeparableSum(nl1, bad), y[2], x[2]
    elseif where == "first"
      f, ry, rx = SeparableSum(bad, nl1), y[1], x[1]
    else                                                # (a, (b, c)) with the rejecting block last, of the kind of b
      ey = y[2] isa SymmetricSparseIterate ? mkowner(5) : fill(T(5), 2, 2)
      ex = x[2] isa SymmetricSparseIterate ? mkowner(6) : point()
      f, ry, rx = SeparableSum(nl1, SeparableSum(nl1, bad)), ey, ex
      y, x = (y[1], (y[2], ey)), (x[1], (x[2], ex))
    end
    before = tstate(y)
    err = try op(y, f, x, T(0.5)); nothing catch e; e end
    if ry isa SymmetricSparseIterate || rx isa SymmetricSparseIterate
      @test err isa MethodError && err.f === op
      @test isequal(tstate(y), before)
    else                                                # a role without an owner is a dense call, which is supported
      @test err === nothing
    end
  end
  # the wrappers that forward to a rejection-only function (Scaled, Translate, Tilt, nested) end in its MethodError too
  forwarding = (("Scaled", f -> Scaled(f, T(2))), ("Translate", f -> Translate(f, zeros(T, 2, 2))),
                ("Tilt", f -> Tilt(f, T[0.1 0.2; 0.2 0.3])), ("Scaled of Translate", f -> Scaled(Translate(f, zeros(T, 2, 2)), T(2))),
                ("Tilt of Scaled", f -> Tilt(Scaled(f, T(2)), T[0.1 0.2; 0.2 0.3])))
  @testset "$wname of $name, $role, $where, $op" for (wname, wrap) in forwarding, (name, bad) in rejecting,
                                                     (role, make) in roles[1:2], where in ("second", "first"), op in (prox!, value_and_prox!)
    nl1 = NormL1(T(0.5))
    f = where == "second" ? SeparableSum(nl1, wrap(bad)) : SeparableSum(wrap(bad), nl1)
    y, x = make()
    before = tstate(y)
    err = try op(y, f, x, T(0.5)); nothing catch e; e end
    @test err isa MethodError && err.f === op
    @test isequal(tstate(y), before)
  end
  # a scaled sum of blocks as a block: the preflight descends to its leaves, nested sums and scales included
  @testset "scaled sum of blocks, $name, $role, $op" for (name, bad) in rejecting, (role, make) in roles[1:2], op in (prox!, value_and_prox!)
    nl1 = NormL1(T(0.5))
    extra(t, v) = t isa SymmetricSparseIterate ? mkowner(v) : fill(T(v), 2, 2)
    for (f, depth) in ((SeparableSum(nl1, Scaled(SeparableSum(nl1, bad), T(2))), 1),
                       (SeparableSum(nl1, Scaled(Scaled(SeparableSum(nl1, bad), T(2)), T(3))), 1),
                       (SeparableSum(nl1, Scaled(SeparableSum(nl1, Scaled(SeparableSum(nl1, bad), T(2))), T(3))), 2))
      y, x = make()                                    # (a, b) and its point; the nested pair holds the rejecting block last
      y = depth == 1 ? (y[1], (y[2], extra(y[2], 5))) : (y[1], (y[2], (extra(y[2], 5), extra(y[2], 6))))
      x = depth == 1 ? (x[1], (x[2], extra(x[2], 5))) : (x[1], (x[2], (extra(x[2], 5), extra(x[2], 6))))
      before = tstate(y)
      err = try op(y, f, x, T(0.5)); nothing catch e; e end
      @test err isa MethodError && err.f === op
      @test isequal(tstate(y), before)
    end
  end
  # parameter validation wins over the MethodError
  @testset "invalid parameters under $wname, $op" for (wname, wrap) in forwarding, op in (prox!, value_and_prox!)
    for (_, bad) in invalid
      y = (mkowner(1), mkowner(2)); before = tstate(y)
      @test_throws ArgumentError op(y, SeparableSum(NormL1(T(0.5)), wrap(bad)), (point(), point()), T(0.5))
      @test isequal(tstate(y), before)
    end
  end
  # a block with invalid parameters is refused by the parameter check, whichever block is rejection-only
  @testset "invalid parameters, $name, $op" for ((name, bad), (_, good)) in zip(invalid, rejecting), op in (prox!, value_and_prox!)
    for f in (SeparableSum(NormL1(T(0.5), W), good), SeparableSum(NormL1(T(0.5)), bad), SeparableSum(bad, NormL1(T(0.5))))
      y = (mkowner(1), mkowner(2)); before = tstate(y)
      @test_throws ArgumentError op(y, f, (point(), point()), T(0.5))
      @test isequal(tstate(y), before)
    end
  end
end

# Nothing else is derived: the wrappers of a nuclear norm (Scaled, Translate, Regularize, a conjugate, nested
# inside another wrapper) and every other spectral function take the exact check. Each call is refused before the
# owner is written, or the inner result is exactly symmetric and the owner stores the lower triangle of the dense
# result; some of the calls are refused.
@testset "the exact check applies to every other combination, $T" for T in (Float32,)
  nuc(c) = NuclearNorm(T(c))
  combos = ("Conjugate(Translate(NuclearNorm))" => Conjugate(Translate(nuc(0.4), zeros(T, 3, 3))),
            "Conjugate(Regularize(NuclearNorm))" => Conjugate(Regularize(nuc(0.4), T(0.5))),
            "MoreauEnvelope(Conjugate(NuclearNorm))" => MoreauEnvelope(Conjugate(nuc(0.4)), T(0.5)))
  @testset "$name, $op" for (name, f) in combos, op in (prox!, value_and_prox!)
    refused = 0
    for trial in 1:12
      rng = MersenneTwister(1000trial + sizeof(T))
      A = randn(rng, T, 3, 3); A = (A + A') / 2
      y = SymmetricSparseIterate(T, 3)
      coordinates(y)[1] = T(5); coordinates(y)[3] = T(-2)
      before = state(y)
      ref = zeros(T, 3, 3)
      op(ref, f, A, T(0.5))
      stored = try op(y, f, A, T(0.5)); true catch e; e isa ArgumentError ? false : rethrow() end
      if stored
        @test Matrix(y) == Matrix(Symmetric(ref, :L))
      else
        refused += 1
        @test isequal(state(y), before)
      end
    end
    @test refused > 0
  end
end

end
