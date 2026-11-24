using Test, ProximalBase, LinearAlgebra, SparseArrays, OffsetArrays, Random

BLAS.set_num_threads(1)

floatbits(x::Float32) = reinterpret(UInt32, x)
floatbits(x::Float64) = reinterpret(UInt64, x)

function check_reference(f, v, x, big_gradient)
  x_before, v_before = copy(collect(x)), copy(collect(v))
  d = similar(v)
  gradient!(d, f, x)
  expected = maximum(abs, v .+ d; init = zero(eltype(v)))
  score = kkt_violation(f, v, x)
  @test floatbits(score) == floatbits(expected)
  @test isequal(collect(x), x_before) && isequal(collect(v), v_before)
  reference = maximum(abs, BigFloat.(v) .+ big_gradient; init = big"0.0")
  @test isapprox(BigFloat(score), reference; rtol = 16eps(eltype(v)))
  # Allocation-free recipe for callers that own the gradient destination.
  d .+= v
  @test floatbits(kkt_violation(ZeroFunction(), d, x)) == floatbits(expected)
end

@testset "Analytic and independent gradient references ($T)" for T in (Float32, Float64)
  x = T[2, 0, -1]
  v = T[-1, 0.25, 2]
  A = T[2 1 0; 1 -1 0.25; 0 0.25 3]
  b = T[1, -1, 0.5]
  f = QuadraticFunction(A, b, zero(T))
  @test kkt_violation(f, T[-5, -0.75, 2.5], x) === zero(T)
  @test kkt_violation(f, v, x) === T(4)
  X = T[1 0 2; 0.25 -1 0; 2 0.5 1; 0 0 -0.5]
  Y = T[1, 0, 2, 1]
  f = LeastSquaresLoss(Y, X)
  @test kkt_violation(f, T[-1.125, 0, 0.75], x) === zero(T)
  @test kkt_violation(f, v, x) === T(1.25)

  setprecision(BigFloat, 256) do
    for A in (A, sparse(A), Diagonal(diag(A)), Symmetric(A)), b in (b, nothing)
      f = QuadraticFunction(A, b, zero(T))
      Ab = BigFloat.(Matrix(A))
      ref = Ab * BigFloat.(x)
      b === nothing || (ref .+= BigFloat.(b))
      for point in (x, view(x, :), SparseIterate(x), BlockIterate(x, [[1, 3], [2]]))
        check_reference(f, view(v, :), point, ref)
      end
    end
    for X in (X, sparse(X), Diagonal(T[1, 2, 3]), Symmetric(A))
      y = Y[1:size(X, 1)]
      f = LeastSquaresLoss(y, X)
      Xb = BigFloat.(Matrix(X))
      ref = transpose(Xb) * (Xb * BigFloat.(x) - BigFloat.(y))
      for point in (x, view(x, :), SparseIterate(x), BlockIterate(x, [[1, 3], [2]]))
        check_reference(f, v, point, ref)
      end
      Ymat = hcat(y, 2y)
      xmat = hcat(x, -x)
      vmat = hcat(v, 2v)
      fmat = LeastSquaresLoss(Ymat, X)
      refmat = transpose(Xb) * (Xb * BigFloat.(xmat) - BigFloat.(Ymat))
      check_reference(fmat, view(vmat, :, :), view(xmat, :, :), refmat)
      check_reference(fmat, vmat, SparseIterate(xmat), refmat)
    end

    rng = MersenneTwister(1051)
    A = Matrix(Symmetric(randn(rng, T, 65, 65)))
    x = randn(rng, T, 65)
    v = randn(rng, T, 65)
    b = randn(rng, T, 65)
    check_reference(QuadraticFunction(A, b, zero(T)), v, x,
                    BigFloat.(A) * BigFloat.(x) + BigFloat.(b))
    X = sprandn(rng, T, 100, 65, 0.12)
    y = randn(rng, T, 100)
    Xb = BigFloat.(Matrix(X))
    check_reference(LeastSquaresLoss(y, X), v, x,
                    transpose(Xb) * (Xb * BigFloat.(x) - BigFloat.(y)))
  end
end

@testset "Shape, type and exceptional values ($T)" for T in (Float32, Float64)
  x = T[0, -1, 2]
  v = T[1, 2, 3]
  X = sparse(T[1 0 0; 0 0 1])
  for f in (LeastSquaresLoss(zeros(T, 2), X),
            QuadraticFunction(Diagonal(T[1, 0, 1]), nothing, zero(T)))
    for i in eachindex(x)
      for (special, expected) in ((T(NaN), T(NaN)), (T(Inf), T(Inf)), (T(-Inf), T(Inf)))
        badv, badx = copy(v), copy(x)
        badv[i] = badx[i] = special
        @test isequal(kkt_violation(f, badv, x), expected)
        @test isequal(kkt_violation(f, v, badx), expected)
      end
      badv, badx = copy(v), copy(x)
      badv[i], badx[end] = T(Inf), T(NaN)
      @test isnan(kkt_violation(f, badv, badx))
      badv[i], badx[end] = T(NaN), T(-Inf)
      @test isnan(kkt_violation(f, badv, badx))
    end
    @test kkt_violation(f, T[-0.0, -0.0, -0.0], zeros(T, 3)) === zero(T)
    @test kkt_violation(f, T[-0.0, nextfloat(zero(T)), -0.0], zeros(T, 3)) === nextfloat(zero(T))
    huge = floatmax(T)
    @test kkt_violation(f, T[huge, 0, 0], zeros(T, 3)) === huge
    @test kkt_violation(f, T[huge, 0, 0], T[huge, 0, 0]) === T(Inf)
    @test kkt_violation(f, [1, 2, 3], x) isa T
    @test kkt_violation(f, BigFloat.(v), x) == big"5.0"
  end

  huge = floatmax(T)
  Q = QuadraticFunction(Diagonal(T[huge, 1]), nothing, zero(T))
  @test kkt_violation(Q, zeros(T, 2), T[2, 0]) === T(Inf)
  loss = LeastSquaresLoss(zeros(T, 2), T[huge 1; huge -1])
  @test isnan(kkt_violation(loss, zeros(T, 2), T[2, 0]))
  @test kkt_violation(loss, T[Inf, 0], T[2, 0]) === T(Inf)

  @test kkt_violation(QuadraticFunction(zeros(T, 0, 0), nothing, zero(T)), T[], T[]) === zero(T)
  for (response, design, point) in ((zeros(T, 3), zeros(T, 3, 0), T[]),
                                      (zeros(T, 3, 2), zeros(T, 3, 0), zeros(T, 0, 2)),
                                      (zeros(T, 3, 0), zeros(T, 3, 2), zeros(T, 2, 0)))
    loss = LeastSquaresLoss(response, design)
    @test kkt_violation(loss, copy(point), point) === zero(T)
    @test loss.output.buf === nothing
  end
  @test kkt_violation(LeastSquaresLoss(T[], zeros(T, 0, 2)), T[1, -2], T[0, 3]) === T(2)
end

@testset "The residual widens after the gradient is rounded" begin
  x = [1.0 + eps(Float64)]
  v = BigFloat[1]
  X = ones(Float32, 1, 1)
  for f in (LeastSquaresLoss(Float32[1], X),
            QuadraticFunction(X, Float32[-1], 0.0f0))
    d = Float32[NaN]
    gradient!(d, f, x)
    @test only(d) === 0.0f0
    @test kkt_violation(f, v, x) == big"1.0"
    @test kkt_violation(f, [1.0], x) === 1.0
  end
end

function scratch_state(f)
  f isa QuadraticFunction && return copy(f.tmp)
  buf = f.output.buf
  (copy(f.tmp), buf, buf === nothing ? nothing : copy(buf))
end

function rejects_unchanged(E, f, v, x, oracle = f)
  before = scratch_state(f)
  v_before, x_before = copy(v), copy(x)
  @test_throws E kkt_violation(oracle, v, x)
  @test isequal(scratch_state(f), before)
  @test isequal(v, v_before) && isequal(x, x_before)
  if f isa LeastSquaresLoss
    @test f.output.buf === before[2]
  end
end

@testset "Preflight and scratch aliasing ($T)" for T in (Float32, Float64)
  v, x = T[1, 2], T[-2, 3]
  Y, X = T[1, 0], T[2 1; 1 2]
  for f in (LeastSquaresLoss(Y, X), QuadraticFunction(X, nothing, zero(T)))
    fill!(f.tmp, T(7))
    rejects_unchanged(DimensionMismatch, f, v, x[1:1])
    rejects_unchanged(DimensionMismatch, f, repeat(v, 2), repeat(x, 2))
    rejects_unchanged(DimensionMismatch, f, reshape(v, 1, 2), x)
    rejects_unchanged(DimensionMismatch, f, reshape(v, 2, 1), reshape(x, 2, 1))
    rejects_unchanged(ArgumentError, f, OffsetArray(v, 0:1), x)
    rejects_unchanged(ArgumentError, f, v, OffsetArray(x, 0:1))
    rejects_unchanged(ArgumentError, f, complex.(v), x)
    rejects_unchanged(ArgumentError, f, v, complex.(x))
    for oracle in (f, Scaled(f, T(2)))
      rejects_unchanged(ArgumentError, f, view(f.tmp, :), x, oracle)
      rejects_unchanged(ArgumentError, f, v, view(f.tmp, :), oracle)
    end
    @test kkt_violation(f, x, x) == maximum(abs, x .+ gradient(f, x))
  end
  loss = LeastSquaresLoss(Y, X)
  @test loss.output.buf === nothing
  kkt_violation(loss, v, x)
  for oracle in (loss, Scaled(loss, T(2)))
    rejects_unchanged(ArgumentError, loss, view(loss.output.buf, :), x, oracle)
    rejects_unchanged(ArgumentError, loss, v, view(loss.output.buf, :), oracle)
  end
  input, output = copy(x), SparseIterate(zeros(T, length(x)))
  expected = prox(LeastSquaresLoss(Y, X), input, T(0.5))
  loss = LeastSquaresLoss(Y, X)
  prox!(output, loss, input, T(0.5))
  @test collect(output) == expected
  buf = loss.output.buf
  @test buf isa Vector{T}
  kkt_violation(loss, v, input)
  @test loss.output.buf === buf
  @test collect(prox!(output, loss, input, T(0.5))) == expected
  Ymat = hcat(Y, 2Y)
  loss = LeastSquaresLoss(Ymat, X)
  xmat = hcat(x, 2x)
  rejects_unchanged(DimensionMismatch, loss, v, x)
  rejects_unchanged(DimensionMismatch, loss, ones(T, 1, 4), ones(T, 1, 4))
  rejects_unchanged(ArgumentError, loss, Diagonal(view(loss.tmp, :, 1)), xmat)
  kkt_violation(loss, xmat, xmat)
  for oracle in (loss, Scaled(loss, T(2))), buf in (loss.tmp, loss.output.buf)
    d, e = view(buf, 1:2, 1), view(buf, 1:1, 2)
    for alias in (view(buf, :, :), Diagonal(d), view(Diagonal(d), :, :),
                  Symmetric(view(buf, :, :)), Bidiagonal(d, e, :U),
                  SymTridiagonal(d, e), Tridiagonal(e, d, e))
      rejects_unchanged(ArgumentError, loss, alias, xmat, oracle)
      rejects_unchanged(ArgumentError, loss, xmat, alias, oracle)
    end
  end
end

@testset "Value, gradient, prox and reset after a score ($T)" for T in (Float32, Float64)
  for make in (A -> QuadraticFunction(A, T[-1, 2], zero(T)),
               A -> LeastSquaresLoss(T[1, -2], A))
    A = T[2 1; 1 2]
    f, control = make(A), make(copy(A))
    x, v, step = T[1, -3], T[2, 0], T(0.25)
    expected = prox(control, x, step)
    @test prox(f, x, step) == expected
    factorization = f isa QuadraticFunction ? f.Q : f.V
    score = kkt_violation(f, v, x)
    @test score === maximum(abs, v .+ gradient(control, x))
    @test f(x) === control(x)
    @test gradient(f, x) == gradient(control, x)
    d = similar(x)
    @test value_and_gradient!(d, f, x) === control(x)
    @test d == gradient(control, x)
    @test prox(f, x, step) == expected
    @test (f isa QuadraticFunction ? f.Q : f.V) === factorization

    A[1, 1] += one(T)
    reset_cache!(f)
    updated = make(copy(A))
    @test kkt_violation(f, v, x) === maximum(abs, v .+ gradient(updated, x))
    @test prox(f, x, step) == prox(updated, x, step)
  end
end

@testset "Local, owner and Scaled routes ($T)" for T in (Float32, Float64)
  x, v = T[1, 2], T[-4, 2]
  X = T[1 0; 0 2]
  for f in (LeastSquaresLoss(T[0, 1], X), QuadraticFunction(X, nothing, zero(T)))
    s = Scaled(f, T(2))
    @test kkt_violation(s, v, x) === maximum(abs, v .+ 2 .* gradient(f, x))
    a = T(3)
    @test kkt_violation(Scaled(f, a), v, x) === a * maximum(abs, (inv(a) .* v) .+ gradient(f, x))
    z = Scaled(f, zero(T))
    @test z isa ZeroFunction
    @test kkt_violation(z, v, x) === T(4)
    @test_throws ArgumentError kkt_violation(Scaled(f, T(Inf)), v, x)
    for a in (floatmin(T), sqrt(floatmax(T)) * 2)
      nested = Scaled(Scaled(f, a), a)
      @test nested.f isa Scaled
      @test_throws ArgumentError kkt_violation(nested, v, x)
    end
    symmetric = SymmetricSparseIterate(T[1 0; 0 2])
    block = BlockIterate(x, [[1, 2]])
    for oracle in (f, s)
      @test_throws ArgumentError kkt_violation(oracle, X, symmetric)
      for args in ((v, 1), (v, x, 1), (v, block, 1), (X, symmetric, 1))
        err = try
          kkt_violation(oracle, args...)
        catch e
          e
        end
        @test err isa ArgumentError && occursin("kkt_violation(f, v, x)", sprint(showerror, err))
      end
    end
  end
  sq = SqrtLeastSquaresLoss(T[0, 1], X)
  for f in (sq, Scaled(sq, T(2)))
    @test_throws MethodError kkt_violation(f, v, x)
    @test_throws MethodError kkt_violation(f, v, x, 1)
    @test_throws MethodError kkt_violation(f, v, 1)
  end
end

struct SmoothTestScreen <: ProximalBase.AbstractFunction end
ProximalBase.kkt_violation(::SmoothTestScreen, v::AbstractArray, i::Integer) = abs(v[i]) - 1
@testset "Custom dispatch and ambiguities" begin
  f = Scaled(SmoothTestScreen(), 2.0)
  @test kkt_violation(f, [3.0], [0.0], 1) === 1.0
  @test_throws ArgumentError kkt_violation(f, [3.0], [1.0], 1)
  @test_throws MethodError kkt_violation(f, [3.0], [0.0])
  @test isempty(Test.detect_ambiguities(ProximalBase))
end

@noinline measure_score(f, v, x) = kkt_violation(f, v, x)
@noinline score_allocations(f, v, x) = @allocated measure_score(f, v, x)
@noinline measure_array!(ref, ::Type{T}, dims) where {T} = (ref[] = Array{T}(undef, dims); nothing)

function check_allocations(make, v, x, one_array)
  measure_score(make(), v, x)
  score_allocations(make(), v, x)
  fresh = make()
  cold = score_allocations(fresh, v, x)
  warm = score_allocations(fresh, v, x)
  expected = if one_array
    saved = Ref{Any}()
    measure_array!(saved, eltype(x), size(x))
    @allocated measure_array!(saved, eltype(x), size(x))
  else
    0
  end
  @info "Smooth KKT allocation bytes" julia=VERSION type=eltype(x) shape=size(x) function_type=typeof(fresh) cold warm expected
  @test cold == expected
  @test warm == 0
end

@testset "Cold-object and warmed allocations ($T)" for T in (Float32, Float64)
  A = Matrix{T}(I, 127, 127)
  v, x = fill(T(2), 127), ones(T, 127)
  check_allocations(() -> QuadraticFunction(A, v, zero(T)), v, x, false)
  check_allocations(() -> LeastSquaresLoss(v, A), v, x, true)
  # a smaller size for the two-column case: Julia allocates arrays above its small-object size with the system
  # allocator, and on Windows (Julia 1.13) the counted bytes of two allocations of the same such size differed;
  # arrays this small come from Julia's pools
  A2 = Matrix{T}(I, 63, 63)
  V, X = hcat(fill(T(2), 63), fill(T(2), 63)), hcat(ones(T, 63), ones(T, 63))
  check_allocations(() -> LeastSquaresLoss(V, A2), V, X, true)
end
