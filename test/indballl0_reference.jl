module IndBallL0ReferenceTest

using Test, LinearAlgebra, ProximalBase, Random

# Enumerate subsets of the nonzero counted coordinates. Form the objective in
# exact arithmetic over the full entries, with both entries of a pair included.
function reference(x, s, diagonal; symmetric = false)
  free(i) = ndims(x) == 2 && !diagonal && i[1] == i[2]
  counted = [i for i in CartesianIndices(x) if !free(i) && !iszero(x[i]) &&
             (!symmetric || i[1] >= i[2])]
  pairs = [symmetric && i[1] != i[2] for i in counted]
  squares = [Rational{BigInt}(x[i])^2 for i in counted]
  best, bestkey, selected = nothing, nothing, Int[]
  for mask in 0:(1 << length(counted))-1
    kept = [k for k in eachindex(counted) if isodd(mask >> (k - 1))]
    e = count(k -> pairs[k], kept)
    length(kept) + e <= s || continue
    loss = zero(Rational{BigInt})
    for k in eachindex(counted)
      if k ∉ kept
        loss += squares[k]
        pairs[k] && (loss += squares[k])
      end
    end
    # For symmetric inputs, ties first prefer fewer pairs and then earlier
    # coordinates in each of the two classes.
    key = (e, filter(k -> !pairs[k], kept), filter(k -> pairs[k], kept))
    if best === nothing || loss < best || (loss == best && isless(key, bestkey))
      best, bestkey, selected = loss, key, kept
    end
  end
  y = similar(x)
  fill!(y, zero(eltype(x)))
  for i in CartesianIndices(x)
    free(i) && (y[i] = x[i])
  end
  for k in selected
    i = counted[k]
    y[i] = x[i]
    pairs[k] && (y[i[2], i[1]] = x[i])
  end
  y, length(counted) + count(identity, pairs)
end

function make_owner(x, order, symmetric)
  a = symmetric ? SymmetricSparseIterate(eltype(x), size(x, 1)) :
                  SparseIterate(eltype(x), size(x)...)
  pts = [i for i in CartesianIndices(x) if !symmetric || i[1] >= i[2]]
  for k in order(eachindex(pts))
    i = pts[k]
    # Some zero coordinates are active and others absent.
    if !iszero(x[i]) || isodd(k)
      activate!(a, k)
      coordinates(a)[k] = x[i]
    end
  end
  a
end

function check_reference(x, symmetric)
  T = eltype(x)
  owners = [make_owner(x, order, symmetric) for order in (identity, reverse)]
  inputs = symmetric ? (Symmetric(x, :L), Symmetric(x, :U), owners...) : (x, owners...)
  for diagonal in (true, false), s in [0:length(x)+1; typemax(Int)]
    y, cost = reference(x, s, diagonal; symmetric)
    f = IndBallL0(s; diagonal)
    for input in inputs
      @test f(input) === (cost <= s ? zero(T) : T(Inf))
      out = prox(f, input)
      @test size(out) == size(x)
      @test Array(out) == y
      @test f(out) === zero(T)
      if input isa Union{SparseIterate, SymmetricSparseIterate}
        @test typeof(out) === typeof(input)
        @test collect(activecoordinates(out)) == collect(activecoordinates(input))
        @test Array(input) == x
        active = collect(activecoordinates(out))
        @test value_and_prox!(out, f, out, zero(T)) === f(out)
        @test Array(out) == y
        @test collect(activecoordinates(out)) == active
      end
    end
  end
end

@testset "Independent exhaustive support references" begin
  rng = Xoshiro(1084)
  for T in (Float32, Float64)
    for shape in ((0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,),
                  (2, 3), (3, 2), (2, 2), (0, 3), (3, 0))
      for _ in 1:3
        check_reference(T.(rand(rng, -3:3, shape...)), false)
      end
    end
    for p in 1:4, _ in 1:4
      check_reference(Matrix(Symmetric(T.(rand(rng, -4:4, p, p)), :L)), true)
    end
    # A tie between two diagonals and a pair. Ties within each group also
    # exercise the logical ID order under reversed sparse activation.
    check_reference(T[7 5 5; 5 1 0; 5 0 0], true)
    # A common huge diagonal must not erase the difference between two splits.
    check_reference(T[floatmax(T) 2 0; 2 1 0; 0 0 1], true)
    t = nextfloat(zero(T))
    check_reference(T[7t 5t; 5t t], true)
    check_reference(T[floatmax(T) -floatmax(T) t -t -zero(T)], false)
  end
end

@testset "Owner state replacement, alias and Scaled" begin
  for T in (Float32, Float64), symmetric in (false, true)
    # For symmetric projection the pair beats the two smaller diagonals.
    x = T[1 7 0; 7 -0.0 0; 0 0 4]
    source = make_owner(x, reverse, symmetric)
    ids = collect(activecoordinates(source))
    y = similar(source)
    coordinates(y)[2] = 42
    activate!(y, ncoordinates(y))
    for diagonal in (true, false)
      f = IndBallL0(2; diagonal)
      expected, _ = reference(x, 2, diagonal; symmetric)
      @test prox!(y, f, source) === y
      @test collect(activecoordinates(y)) == ids
      @test Array(y) == expected
      @test Array(source) == x
      scaled = Scaled(f, T(3))
      @test Array(prox(scaled, source)) == expected
      @test value_and_prox!(y, scaled, source, T(1)) === scaled(y)
      @test Array(y) == expected
      trial = copy(source)
      @test prox!(trial, f, trial) === trial
      @test Array(trial) == expected
      @test collect(activecoordinates(trial)) == ids
      dropzeros!(trial)
      @test Array(trial) == expected
      @test nactive(trial) == count(!iszero, activevalues(trial))

      snapshot = copy(y)
      for step in (T(Inf), T(NaN), T(-1))
        @test_throws ArgumentError prox!(y, f, source, step)
        @test_throws ArgumentError value_and_prox!(y, f, source, step)
      end
      @test isequal(Array(y), Array(snapshot))
      @test collect(activecoordinates(y)) == collect(activecoordinates(snapshot))
    end
    # Value classifies only counted entries. Free nonfinite diagonals can
    # pass through the copy without changing the stored indicator value.
    source[2, 2] = T(NaN)
    f = IndBallL0(2; diagonal = false)
    @test value_and_prox!(y, f, source) === f(y) === zero(T)
    @test isnan(y[2, 2])
    @test isnan(IndBallL0(typemax(Int))(source))
    @test_throws ArgumentError prox!(y, IndBallL0(2), source)
    @test isnan(y[2, 2])
  end
end

@testset "Sampled dense selection on structured orders" begin
  n = 4096
  for T in (Float32, Float64)
    for x in (T.(1:n), T.(n:-1:1), T[min(i, n-i+1) for i in 1:n],
              T[(-1)^i * (i % 11) for i in 1:n])
      order = sortperm(eachindex(x); by = i -> (-abs(x[i]), i))
      for s in (1, 17, n ÷ 2, n - 1)
        expected = zeros(T, n)
        for i in view(order, 1:s)
          expected[i] = x[i]
        end
        f = IndBallL0(s)
        @test prox(f, x) == expected
        y = copy(x)
        @test value_and_prox!(y, f, y) === zero(T)
        @test y == expected
      end
    end
  end
end

@testset "Dense value classification with exceptional sums" begin
  for T in (Float32, Float64)
    f = IndBallL0(2)
    @test f(T[-Inf, Inf]) === zero(T)
    @test IndBallL0(1)(T[-Inf, Inf]) === T(Inf)
    @test f(T[-floatmax(T), floatmax(T)]) === zero(T)
    @test isnan(f(T[NaN, Inf]))
    @test isnan(f(T[Inf, -NaN]))
    h = floatmax(T)
    x = repeat(T[h, h, -h, -h], 1024)
    @test IndBallL0(length(x))(x) === zero(T)
    @test IndBallL0(length(x) - 1)(x) === T(Inf)
    @test IndBallL0(0)(T[-0.0, 0.0]) === zero(T)
    @test IndBallL0(0)(T[-0.0, nextfloat(zero(T))]) === T(Inf)
    y = T[1, 1]
    @test_throws ArgumentError prox!(y, f, T[-Inf, Inf])
    @test y == T[1, 1]
  end
end

end
