module BlockChecksWorkspaceTest

using Test, ProximalBase, LinearAlgebra, SparseArrays

const PB = ProximalBase
const proximal = (prox!, value_and_prox!)

function oracle!(op, y, f, x)
  @nospecialize op y f x
  op(y, f, x, 1.0)
end

function unchanged(op, f, point, E; output = point, modes = (false, true))
  @nospecialize op f point E output
  for alias in modes
    x = deepcopy(point)
    y = alias ? x : deepcopy(output)
    input, before = deepcopy(x), deepcopy(y)
    @test_throws E oracle!(op, y, f, x)
    @test isequal(x, input)
    @test isequal(y, before)
  end
end

function block_failure(f, x, E = ArgumentError)
  @nospecialize f x E
  for op in proximal
    unchanged(op, f, x, E; output = fill(eltype(x)(7), size(x)))
    s = SeparableSum(SqrNormL2(2.0), f)
    point, dest = ([3.0], x), ([7.0], fill(eltype(x)(7), size(x)))
    unchanged(op, s, point, E; output = dest)
    unchanged(op, Scaled(s, 2.0), point, E; output = dest)
    nested = SeparableSum(SqrNormL2(2.0), SeparableSum(f))
    unchanged(op, nested, ([3.0], (x,)), E; output = ([7.0], (dest[2],)))
  end
end

# Every branch (full, partial, rank-limited) raises the same element-type error,
# so the factory that would fail can no longer be observed; the message and the
# unwritten blocks are what remains.
function cases()
  (NuclearNorm(1.0), NuclearNorm(1.0; partial = true),
   IndBallRank(1), IndBallRank(1; partial = true),
   IndPSD(), IndPSD(; partial = true),
   IndPSD(; rank = 1), IndPSD(; rank = 1, partial = true),
   IndPSD(; rank = 2), IndPSD(; rank = 2, partial = true),
   TracePSD(1.0), TracePSD(1.0; partial = true),
   Spectral(NormL1(1.0)), SingularValue(NormL1(1.0)))
end

function type_error(f, x)
  @nospecialize f x
  name = nameof(typeof(f))
  for op in proximal, step in (0.0, 1.0)
    y = fill(eltype(x)(7), size(x))
    @test_throws ArgumentError op(y, f, x, step)
    @test_throws "$name needs Float32 or Float64 entries, got $(float(eltype(x)))" op(y, f, x, step)
    @test all(==(7), y)
  end
end

@testset "Workspace failures precede every block write" begin
  for T in (Float16, BigFloat), f in cases()
    x = T[2 1; 1 3]
    block_failure(f, x)
    type_error(f, x)
    # A cache for another precision must not change the error.
    prox!(zeros(2, 2), f, Float64.(x), 1.0)
    type_error(f, x)
  end
  for T in (Float16, BigFloat)
    block_failure(TracePSD(1.0), Symmetric(T[1 2; 2 4]))
    # Stiefel has an explicit output contract, checked before its SVD.
    block_failure(IndStiefel(), T[2 1; 1 3], ArgumentError)
    block_failure(IndStiefel(), zeros(T, 2, 0), ArgumentError)
    for f in (NuclearNorm(1.0), NuclearNorm(1.0; partial = true),
              IndPSD(), IndPSD(; partial = true),
              IndPSD(; rank = 1), IndPSD(; rank = 1, partial = true), TracePSD(0.0),
              TracePSD(0.0; partial = true), Spectral(ZeroFunction()),
              SingularValue(ZeroFunction()))
      block_failure(f, zeros(T, 0, 0))
    end
    # Zero weight does not bypass PSD projection or a wrapper's decomposition.
    for f in (TracePSD(0.0), TracePSD(0.0; partial = true),
              Spectral(ZeroFunction()), SingularValue(ZeroFunction()))
      block_failure(f, T[2 1; 1 3])
    end
  end
end

function accepted(f, x, output; identity = false)
  @nospecialize f x output
  for op in proximal, alias in (false, true)
    eltype(x) === eltype(output) || !alias || continue
    input = deepcopy(x)
    y = alias ? input : deepcopy(output)
    ref = zeros(Float64, size(x))
    rv = oracle!(op, ref, deepcopy(f), Float64.(x))
    v = oracle!(op, y, deepcopy(f), input)
    @test y ≈ eltype(y).(ref)
    identity && @test isequal(y, x)
    @test op === prox! ? v === y : v ≈ rv
    alias || @test isequal(input, x)
    for scaled in (false, true), nested in (false, true)
      # Scaled uses twice the step for both the reference and the block sum.
      g = scaled ? Scaled(deepcopy(f), 2.0) : deepcopy(f)
      ref = deepcopy(output)
      a = oracle!(op, [7.0], scaled ? Scaled(SqrNormL2(2.0), 2.0) : SqrNormL2(2.0), [3.0])
      b = oracle!(op, ref, deepcopy(g), deepcopy(x))
      s = nested ? SeparableSum(SqrNormL2(2.0), SeparableSum(deepcopy(f))) : SeparableSum(SqrNormL2(2.0), deepcopy(f))
      scaled && (s = Scaled(s, 2.0))
      bx = nested ? ([3.0], (deepcopy(x),)) : ([3.0], deepcopy(x))
      by = alias ? bx : nested ? ([7.0], (deepcopy(output),)) : ([7.0], deepcopy(output))
      saved = deepcopy(bx)
      result = oracle!(op, by, s, bx)
      @test by[1] ≈ [3 / (scaled ? 5 : 3)]
      @test (nested ? by[2][1] : by[2]) == ref
      @test op === prox! ? result === by : result ≈ a + b
      alias || @test isequal(bx, saved)
    end
  end
end

@testset "Shortcuts, mixed precision and supported controls" begin
  for T in (Float16, BigFloat), partial in (false, true)
    for x in (T[1 2; 3 4], zeros(T, 0, 0))
      accepted(NuclearNorm(0.0; partial), x, similar(x); identity = true)
      accepted(IndBallRank(2; partial), x, similar(x); identity = true)
    end
  end
  for T in (Float16, BigFloat), f in cases()
    accepted(f, T[2 1; 1 3], zeros(2, 2))
  end
  for T in (Float32, Float64), f in (cases()..., IndStiefel())
    accepted(f, T[2 1; 1 3], zeros(T, 2, 2))
  end
  accepted(IndStiefel(), Float64[2 1; 1 3], zeros(2, 2))
  accepted(IndStiefel(), Float16[2 1; 1 3], zeros(2, 2))
  accepted(IndStiefel(), zeros(2, 0), zeros(2, 0); identity = true)
end

function check_allocated(op::O, f::F, y::Y, x::X) where {O, F, Y, X}
  PB._checkwrite(op, f, y, x)
  @allocated PB._checkwrite(op, f, y, x)
end

@testset "Workspace checks allocate nothing" begin
  for f in (cases()..., NuclearNorm(0.0), IndBallRank(2), IndStiefel()), op in proximal
    x, y = [2.0 1.0; 1.0 3.0], zeros(2, 2)
    # Wrappers reuse spectrum buffers only after their first oracle call.
    oracle!(op, y, f, x)
    @test check_allocated(op, f, y, x) == 0
    s = SeparableSum(SqrNormL2(2.0), f)
    @test check_allocated(op, s, ([7.0], y), ([3.0], x)) == 0
  end
  for f in (NuclearNorm(0.0; partial = true), IndBallRank(2; partial = true))
    @test check_allocated(prox!, f, zeros(Float16, 2, 2), ones(Float16, 2, 2)) == 0
  end
end

end
