module BlockChecksValuesTest

using Test, ProximalBase, LinearAlgebra, SparseArrays
using ..TestUtils

const PB = ProximalBase
const proximal = (prox!, value_and_prox!)

function oracle!(op, y, f, x)
  @nospecialize op y f x
  op in proximal ? op(y, f, x, 1.0) : op(y, f, x)
end

function unchanged(op, f, point, E; output = point)
  @nospecialize op f point E output
  for alias in (false, true)
    x = deepcopy(point)
    y = alias ? x : deepcopy(output)
    input, before = deepcopy(x), deepcopy(y)
    @test_throws E oracle!(op, y, f, x)
    @test isequal(x, input)
    @test isequal(y, before)
  end
end

function block_failure(op, f, x, E)
  @nospecialize op f x E
  unchanged(op, f, x, E)
  s = SeparableSum(SqrNormL2(2.0), f)
  point = ([3.0], x)
  unchanged(op, s, point, E; output = ([7.0], x))
  unchanged(op, Scaled(s, 2.0), point, E; output = ([7.0], x))
  nested = SeparableSum(SqrNormL2(2.0), SeparableSum(f))
  unchanged(op, nested, ([3.0], (x,)), E; output = ([7.0], (x,)))
end

function success(f, x, expected, value)
  @nospecialize f x expected value
  for op in proximal, alias in (false, true), kind in 1:3
    input = kind == 3 ? ([3.0], (copy(x),)) : ([3.0], copy(x))
    y = alias ? input : deepcopy(input)
    sumf = kind == 3 ? SeparableSum(SqrNormL2(2.0), SeparableSum(f)) : SeparableSum(SqrNormL2(2.0), f)
    kind == 2 && (sumf = Scaled(sumf, 2.0))
    # Compute the scaled reference with its actual step.
    ref = copy(x)
    v = value_and_prox!(ref, deepcopy(f), copy(x), kind == 2 ? 2.0 : 1.0)
    kind != 2 && (@test ref ≈ expected; @test v ≈ value)
    result = oracle!(op, y, sumf, input)
    @test (kind == 3 ? y[2][1] : y[2]) ≈ ref
    firstpoint = kind == 2 ? 3/5 : 1.0
    @test y[1] ≈ [firstpoint]
    op === value_and_prox! && @test result ≈ (kind == 2 ? 2 : 1) * (firstpoint^2 + v)
  end
end

@testset "Conjugate standalone values before writes" begin
  x = [0.5, -0.5]
  for set in (IndBox(-1.0, 1.0), IndHyperplane([1.0, 1.0], 0.0))
    f = Constrained(Conjugate(ElasticNet(1.0, 1.0)), set)
    block_failure(value_and_prox!, f, x, ArgumentError)
    y = copy(x)
    @test prox!(y, f, x, 1.0) === y
    @test y == x
  end
  f = Constrained(Tilt(Conjugate(NormL1Nonneg(1.0)), zeros(2)), IndBox(-1.0, 1.0))
  block_failure(value_and_prox!, f, x, ArgumentError)
  y = copy(x)
  @test prox!(y, f, x, 1.0) === y
  @test y == x
  for f in (Conjugate(ElasticNet(1.0, 1.0)), Conjugate(NormL1Nonneg(1.0)))
    @test_throws ArgumentError PB._checkvalue(f, x)
    @test_throws ArgumentError f(x)
  end
  success(Constrained(Conjugate(NormL1Nonneg(1.0)), IndBox(-1.0, 1.0)), x, x, 0.0)
  # ElasticNet's prox at this input is zero, so its conjugate prox is x.
  success(Conjugate(ElasticNet(1.0, 1.0)), x, x, 0.0)
end

@testset "Centered SqrNormL2 controls" begin
  success(Constrained(SqrNormL2(1.0, ones(2)), IndBox(0, 1)), [0.5, 0.5], [0.75, 0.75], 1/16)
  for x in (sparse([0.5, 0.5]), view([0.5, 0.5, 2.0], 1:2), SymmetricSparseIterate([0.5 0.0; 0.0 0.5]))
    f = SqrNormL2(1.0, ones(size(x)))
    y = deepcopy(x)
    @test value_and_prox!(y, f, x, 1.0) ≈ sum(abs2, (Array(x) .- 1) ./ 2) / 2
  end
end

function outcome(f, args...)
  @nospecialize f args
  try
    (nothing, f(args...))
  catch e
    (typeof(e), nothing)
  end
end

function representations(dims)
  A = reshape(collect(1.0:16.0), dims)
  S = sparse(A)
  result = Any[A, S, view(sparse(hcat(A, A)), :, collect(1:size(A, 2))),
    adjoint(copy(transpose(A))), transpose(copy(transpose(A))), adjoint(sparse(transpose(A))),
    transpose(sparse(transpose(A))), vec(copy(A)), sparse(vec(A))]
  columns = view(sparse(transpose(A)), :, collect(1:size(A, 1)))
  append!(result, (adjoint(columns), transpose(columns)))
  if dims[1] == dims[2]
    for B in (A, S)
      append!(result, (Symmetric(B), Hermitian(B), UpperTriangular(B), LowerTriangular(B),
        UnitUpperTriangular(B), UnitLowerTriangular(B)))
    end
    append!(result, (Diagonal(diag(A)), Tridiagonal(diag(A,-1), diag(A), diag(A,1)),
      SymTridiagonal(diag(A), diag(A,1)), Bidiagonal(diag(A), diag(A,1), :U),
      Bidiagonal(diag(A), diag(A,-1), :L), UpperHessenberg(A), SymmetricSparseIterate(Symmetric(A))))
  end
  result
end

function dot_case(c, x, dest)
  @nospecialize c x dest
  f = LinearFunction(c)
  # A write can reject a structured destination before the value is evaluated.
  for (op, writeop) in ((value_and_gradient!, gradient!), (value_and_prox!, prox!))
    y, p = deepcopy(dest), deepcopy(dest)
    before, input = deepcopy(y), deepcopy(x)
    written = outcome(oracle!, writeop, p, f, x)
    expected = written[1] === nothing ? outcome(dot, c, op === value_and_prox! ? p : x) : written
    actual = outcome(oracle!, op, y, f, x)
    @test isequal(actual[1], expected[1])
    if expected[1] === nothing
      @test actual[2] ≈ expected[2]
      @test y ≈ p
    else
      @test isequal(y, before)
    end
    @test isequal(x, input)
  end
end

@testset "dot domains" begin
  square, rectangle = representations((4, 4)), representations((2, 8))
  grid = vcat(square, rectangle)
  for c in grid, x in grid
    @testset "$(typeof(c)) $(size(c)) vs $(typeof(x)) $(size(x))" begin
      expected = outcome(dot, c, x)[1]
      @test outcome(PB._checkvalue, LinearFunction(c), x)[1] === expected
    end
  end
  # Mixed sparse wrappers also specialize dot for column-subset parents.
  for cdims in ((4, 4), (2, 8)), xdims in ((4, 4), (2, 8))
    c = sparse(reshape(collect(1.0:16.0), cdims))
    x = sparse(reshape(collect(1.0:16.0), xdims))
    for cp in (c, view(c, :, collect(1:size(c, 2)))),
        xp in (x, view(x, :, collect(1:size(x, 2)))),
        (cw, xw) in ((adjoint(cp), transpose(xp)), (transpose(cp), adjoint(xp)))
      @testset "$(typeof(cw)) $(size(cw)) vs $(typeof(xw)) $(size(xw))" begin
        @test outcome(PB._checkvalue, LinearFunction(cw), xw)[1] === outcome(dot, cw, xw)[1]
      end
    end
  end
  A, B = reshape(collect(1.0:16.0), 4, 4), reshape(collect(1.0:16.0), 2, 8)
  @test dot(A, B) == dot(vec(A), B) == 1496.0
  @test PB._checkvalue(LinearFunction(A), B) === true
  @test_throws DimensionMismatch PB._checkvalue(LinearFunction(Diagonal(diag(A))), B)
  @test_throws DimensionMismatch PB._checkvalue(LinearFunction(sparse(A)'), sparse(B)')
  # Cross input and output representations independently on the dispatch boundaries.
  for c in (A, sparse(A)', Diagonal(diag(A)), vec(A)),
      x in (B, sparse(B), copy(B')', vec(B)),
      dest in (copy(B), sparse(B), copy(B')')
    dot_case(c, x, size(x) == size(dest) ? dest : copy(x))
  end
  for c in (B, sparse(B), vec(B)), x in (A, sparse(A)', SymmetricSparseIterate(Symmetric(A))),
      dest in (copy(A), sparse(A), SymmetricSparseIterate(Symmetric(A)))
    dot_case(c, x, dest)
  end
end

function measured_oracle!(op::O, y, f, x) where {O}
  op in proximal ? op(y, f, x, 1.0) : op(y, f, x)
end

function allocations(op::O, y, f, x) where {O}
  measured_oracle!(op, y, f, x)
  PB._checkwrite(op, f, y, x)
  check = @allocated PB._checkwrite(op, f, y, x)
  call = @allocated measured_oracle!(op, y, f, x)
  (check, call)
end

function check_allocations(op::O, y, f, x) where {O}
  PB._checkwrite(op, f, y, x)
  @allocated PB._checkwrite(op, f, y, x)
end

function value_check_allocations(f, x)
  PB._checkvalue(f, x)
  @allocated PB._checkvalue(f, x)
end

@testset "Allocation free value checks" begin
  A = reshape(collect(1.0:16.0), 4, 4)
  # Diagonal destinations avoid dot's allocating diagonal view of a dense matrix.
  for (f, x) in ((Conjugate(NormL1()), [0.5, -0.5]), (SqrNormL2(1.0, ones(2)), [0.5, -0.5]),
      (LinearFunction(Diagonal(diag(A))), Diagonal(diag(A))), (LinearFunction(sparse(A)'), copy(A)))
    for op in (f isa Conjugate ? proximal : f isa LinearFunction{<:Diagonal} ? (value_and_gradient!,) : (prox!, value_and_prox!, value_and_gradient!))
      y = copy(x)
      allocations(op, y, f, x)
      @test allocations(op, y, f, x) == (0, 0)
    end
  end
  for (c, x) in ((adjoint(sparse(A)), transpose(sparse(A))),
                 (transpose(sparse(A)), adjoint(sparse(A))))
    f, y = LinearFunction(c), copy(A)
    for op in (prox!, value_and_prox!, value_and_gradient!)
      allocations(op, y, f, x)
      @test allocations(op, y, f, x) == (0, 0)
    end
  end
  S = sparse(A)
  columns = view(S, :, collect(1:size(S, 2)))
  for a in (S, columns), b in (adjoint(columns), transpose(columns)), (c, x) in ((a, b), (b, a))
    f, y = LinearFunction(c), deepcopy(x)
    value_check_allocations(f, x)
    @test value_check_allocations(f, x) == 0
    for op in (value_and_prox!, value_and_gradient!)
      check_allocations(op, y, f, x)
      @test check_allocations(op, y, f, x) == 0
    end
  end
  f = LinearFunction(Diagonal(diag(A)))
  for op in proximal
    PB._checkwrite(op, f, copy(A), A)
    y = copy(A)
    @test (@allocated PB._checkwrite(op, f, y, A)) == 0
  end
end

@testset "Nested boxes check the prospective value before writes" begin
  for inner in (Conjugate(ElasticNet(1.0, 1.0)), SqrNormL2(1.0, ones(1, 2))),
      wrap in (identity, f -> Scaled(f, 2.0), f -> Tilt(f, zeros(2)), f -> Translate(f, zeros(2)))
    inside = wrap(Constrained(inner, IndBox(0.0, 1.0)))
    f = Constrained(inside, IndBox(-1.0, 2.0))
    E = inner isa Conjugate ? ArgumentError : DimensionMismatch
    x = [0.5, 0.5]
    block_failure(value_and_prox!, f, x, E)
    # The outer box moves the result outside the inner value's domain.
    outside = Constrained(inside, IndBox(2.0, 3.0))
    y = zeros(2)
    @test value_and_prox!(y, outside, x, 1.0) === Inf
    @test y == [2.0, 2.0]
    @test prox!(y, f, x, 1.0) === y

    point = hcat([3.0, 3.0], x)
    unchanged(value_and_prox!, SliceSum(SqrNormL2(2.0), f), point, E)
    unchanged(prox!, SliceSum(SqrNormL2(2.0), f), point, E)
    fibers = zeros(2, 2, 2)
    fibers[1, 1, :] .= 3
    fibers[2, 1, :] .= 0.5
    fb = FiberSum(f; diagonal = SqrNormL2(2.0), symmetric = true)
    unchanged(value_and_prox!, fb, fibers, E)
  end
  for inner in (SqrNormL2(1.0, ones(2)), SqrNormL2(1.0, ones(1, 2)), Conjugate(ElasticNet(1.0, 1.0)))
    f = Constrained(Constrained(inner, IndBox(0, 1)), IndBox(2, 3))
    x, y = [0.5, 0.5], zeros(2)
    @test allocs(value_and_prox!, y, f, x, 1.0) == 0
    s = SeparableSum(SqrNormL2(2.0), f)
    point = ([3.0], x)
    dest = ([7.0], y)
    @test allocs(value_and_prox!, dest, s, point, 1.0) == 0
  end
  # Membership uses the converted output and its precision's tolerance.
  f = Constrained(Constrained(SqrNormL2(1.0, ones(1, 2)), IndBox(0, 1)), IndBox(1 + 2e-8, 2))
  x = [0.5, 0.5]
  y = Float32[7, 9]
  @test_throws DimensionMismatch value_and_prox!(y, f, x, [0.5, 1.0])
  @test y == Float32[7, 9]
  y64 = zeros(2)
  @test value_and_prox!(y64, f, x, [0.5, 1.0]) === Inf
  @test y64 == fill(1 + 2e-8, 2)
  # The first ordinary fiber is outside, but a later active fiber needs the value.
  f = Constrained(Constrained(SqrNormL2(1.0, ones(1, 2)), IndBox(0, 1)), IndBox(-1, 2))
  x = fill(NaN, 3, 3, 2)
  x[3, 1, :] .= 0.5
  fb = FiberSum(f; diagonal = nothing, symmetric = true)
  unchanged(value_and_prox!, fb, x, DimensionMismatch)
  # Upper triangles and free diagonals do not evaluate the function.
  x[3, 1, :] .= NaN
  x[1, 3, :] .= 0.5
  for i in 1:3
    x[i, i, :] .= 0.5
  end
  y = copy(x)
  @test value_and_prox!(y, fb, x, 1.0) === Inf
  @test [y[i, i, 1] for i in 1:3] == fill(0.5, 3)
end

end
