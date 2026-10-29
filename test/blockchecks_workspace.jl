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

function block_failure(f, x, E = MethodError)
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

# Fixed factory expectations distinguish partial dispatch from its full fallback.
function cases()
  ((NuclearNorm(1.0), PB._svd_workspace, (2, 2)),
   (NuclearNorm(1.0; partial = true), PB._partial_svd_workspace, (2, 2, PB._PARTIAL_GUARD)),
   (IndBallRank(1), PB._svd_workspace, (2, 2)),
   (IndBallRank(1; partial = true), PB._partial_svd_workspace, (2, 2, 1)),
   (IndPSD(), PB._eig_workspace, (2,)),
   (IndPSD(; partial = true), PB._partial_eigen_workspace, (2, PB._PARTIAL_GUARD)),
   (IndPSD(; rank = 1), PB._eigtop_workspace, (2, 1)),
   (IndPSD(; rank = 1, partial = true), PB._partial_eigen_workspace, (2, 1)),
   (IndPSD(; rank = 2), PB._eig_workspace, (2,)),
   (IndPSD(; rank = 2, partial = true), PB._eig_workspace, (2,)),
   (TracePSD(1.0), PB._eig_workspace, (2,)),
   (TracePSD(1.0; partial = true), PB._partial_eigen_workspace, (2, PB._PARTIAL_GUARD)),
   (Spectral(NormL1(1.0)), PB._eig_workspace, (2,)),
   (SingularValue(NormL1(1.0)), PB._svd_workspace, (2, 2)))
end

function factory_error(f, x, factory, dims)
  @nospecialize f x factory dims
  for op in proximal, step in (0.0, 1.0)
    err = try
      op(similar(x), f, x, step)
      nothing
    catch e
      e
    end
    @test err isa MethodError
    if err isa MethodError
      @test err.f === factory
      cache = factory === PB._svd_workspace ? f.ws :
              factory === PB._eig_workspace ? f.eig :
              factory === PB._eigtop_workspace ? f.top : nothing
      @test err.args[1] === cache
      @test isequal(Base.tail(err.args), (float(eltype(x)), dims...))
    end
  end
end

@testset "Workspace failures precede every block write" begin
  for T in (Float16, BigFloat), (f, factory, dims) in cases()
    x = T[2 1; 1 3]
    block_failure(f, x)
    factory_error(f, x, factory, dims)
    # A cache for another precision must not change the missing method.
    prox!(zeros(2, 2), f, Float64.(x), 1.0)
    factory_error(f, x, factory, dims)
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
  for T in (Float16, BigFloat), (f, _, _) in cases()
    accepted(f, T[2 1; 1 3], zeros(2, 2))
  end
  for T in (Float32, Float64), f in (map(first, cases())..., IndStiefel())
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
  for f in (map(first, cases())..., NuclearNorm(0.0), IndBallRank(2), IndStiefel()), op in proximal
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
