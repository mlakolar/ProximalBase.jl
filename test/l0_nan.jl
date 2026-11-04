module L0NaNTest

using Test, ProximalBase, LinearAlgebra
using JLArrays: JLArray
using GPUArraysCore: allowscalar

# NaN in NormL0 and IndBallL0 (E2 with E4 first, #560): a NaN entry gives NaN wherever the result
# depends on it, and never a finite zero, a kept entry or a feasible point by the failed comparison.
# NaN is compared with isequal, never ==.

same(a, b) = isequal(a, b)
dense(y) = Array(y)

####################################
#  NormL0
####################################

# the proximal map of γ·λ·#{xᵢ ≠ 0} entrywise, E4 first: a zero level is the identity at every step,
# an infinite level and an infinite step are the zero map (constant at finite arguments, so NaN too),
# a zero step is the identity; otherwise hard thresholding, NaN kept
function ref_l0_prox(v, λ, γ)
  iszero(λ) && return v
  (isinf(λ) || isinf(γ)) && return zero(v)
  iszero(γ) && return v
  isnan(v) && return v
  v^2 > 2γ * λ ? v : zero(v)
end
# the term λ·[v ≠ 0]; a NaN entry is NaN unless the level is zero
ref_l0_value(v, λ) = iszero(v) ? zero(v) : isnan(v) && !iszero(λ) ? oftype(v, NaN) : oftype(v, λ)

@testset "NormL0 at NaN, $T" for T in (Float32, Float64)
  x = T[NaN, 1.5, 0, -3, 0.25, Inf, -Inf]
  levels = (T(0), T(1), T(Inf)), (T(0), T(1), T(Inf))
  @testset "λ = $λ, γ = $γ" for λ in levels[1], γ in levels[2]
    f = NormL0(λ)
    want = ref_l0_prox.(x, λ, γ)
    @test same(prox(f, x, γ), want)
    @test same([prox_entry(f, v, i, γ) for (i, v) in enumerate(x)], want)
    y = fill(T(7), length(x))                       # a sentinel: every entry is overwritten
    @test prox!(y, f, x, γ) === y && same(y, want)
    z = copy(x); prox!(z, f, z, γ)                  # aliasing
    @test same(z, want)
    @test same(prox(f, x, fill(γ, length(x))), want)    # a vector step
    @test same(f(x), sum(v -> ref_l0_value(v, λ), x))
    y = fill(T(7), length(x))
    v = value_and_prox!(y, f, x, γ)                 # the value at the stored output
    @test same(y, want) && same(v, sum(w -> ref_l0_value(w, λ), want))
  end
  # the named cases
  @test isnan(prox_entry(NormL0(T(1)), T(NaN), 1, T(1)))
  @test same(prox(NormL0(T(1)), T[NaN, 1.5], T(1)), T[NaN, 1.5])
  @test same(prox(NormL0(T(0)), T[NaN, 1.5], T(1)), T[NaN, 1.5])             # zero level: identity
  @test same(prox(NormL0(T(1)), T[NaN, 1.5], T(0)), T[NaN, 1.5])             # γ = 0: identity
  @test same(prox(NormL0(T(1)), T[NaN, 1.5], T(Inf)), T[0, 0])               # constant map: zeros stay
  @test same(prox(NormL0(T(0)), T[NaN, 1.5], T(Inf)), T[NaN, 1.5])           # zero level wins over γ = Inf
  @test same(prox(NormL0(T(Inf)), T[NaN, 1.5], T(1)), T[0, 0])
  @test same(prox(NormL0(T(Inf)), T[NaN, 1.5], T(0)), T[0, 0])
  @test isnan(NormL0(T(1))(T[NaN, 1.5])) && isnan(NormL0(T(Inf))(T[NaN, 0]))
  @test NormL0(T(0))(T[NaN, 1.5]) == 0                                        # E4: the zero function
  @test isnan(value_and_prox!(zeros(T, 2), NormL0(T(1)), T[NaN, 1.5], T(1)))
  @test value_and_prox!(zeros(T, 2), NormL0(T(1)), T[NaN, 1.5], T(Inf)) == 0
  @test value_and_prox!(zeros(T, 2), NormL0(T(0)), T[NaN, 1.5], T(1)) == 0
  # 2γλ outside the normal range: the exceptional comparison keeps NaN where γ and λ are finite
  big = floatmax(T)
  @test same(prox(NormL0(big), T[NaN, 1], big), T[NaN, 0])
  @test same(prox(NormL0(big), T[NaN, 1], T(Inf)), T[0, 0])
  @test same(prox(NormL0(T(0)), T[NaN, 1], big), T[NaN, 1])
  # finite arguments are unchanged: the strict threshold and the first kept entry
  @test prox(NormL0(T(2)), T[2, 3, -2, 0], T(1)) == T[0, 3, 0, 0]
  @test prox(NormL0(T(0.5)), T[0.5, -2, 0.05, 1.2], T(1)) == T[0, -2, 0, 1.2]
  @test NormL0(T(0.5))(T[0.5, -2, 0, 1.2]) == T(1.5)
end

@testset "NormL0 on iterates at NaN" begin
  f = NormL0(1.0)
  x = SparseIterate(4); x[2] = NaN; x[3] = 1.5
  y = SparseIterate(4)
  prox!(y, f, x, 1.0)
  @test same(dense(y), [0.0, NaN, 1.5, 0.0]) && y.nnz == 2
  @test isnan(f(x))
  @test isnan(value_and_prox!(SparseIterate(4), f, x, 1.0))
  y = SparseIterate(4); @test value_and_prox!(y, NormL0(0.0), x, 1.0) == 0 && same(dense(y), [0.0, NaN, 1.5, 0.0])
  z = copy(x); prox!(z, f, z, 0.0)                        # γ = 0 keeps every stored entry
  @test same(dense(z), [0.0, NaN, 1.5, 0.0])
  S = SymmetricSparseIterate(3); S[2, 1] = NaN; S[3, 3] = 2.0
  Y = SymmetricSparseIterate(3)
  prox!(Y, f, S, 1.0)
  @test same(Matrix(Y), [0.0 NaN 0.0; NaN 0.0 0.0; 0.0 0.0 2.0])
  @test isnan(f(S)) && isnan(value_and_prox!(Y, f, S, 1.0))
  @test same(Matrix(prox!(SymmetricSparseIterate(3), NormL0(0.0), S, Inf)), [0.0 NaN 0.0; NaN 0.0 0.0; 0.0 0.0 2.0])
end

@testset "NormL0 on a device array at NaN" begin
  allowscalar(false)
  for T in (Float32, Float64)
    x = T[NaN, 1.5, 0, -3]
    for (λ, γ) in ((T(1), T(1)), (T(0), T(1)), (T(1), T(0)), (T(1), T(Inf)), (T(0), T(Inf)))
      f = NormL0(λ)
      d = JLArray(x)
      @test same(Array(prox(f, d, γ)), ref_l0_prox.(x, λ, γ))
    end
  end
end

####################################
#  IndBallL0
####################################

# A reference by coordinates. A coordinate is the entries it writes (a mirrored pair is one), its weight
# in the selection, and whether it counts. Free coordinates (a free diagonal) are copied. Returns
# (output, standalone value, fused value).
struct Coord
  pos::Vector{CartesianIndex{2}}
  w::Float64
  counted::Bool
end

function ref_l0(X::AbstractMatrix{T}, coords, s) where {T}
  counted = filter(c -> c.counted, coords)
  m = length(counted)
  val(c) = X[first(c.pos)]
  nan = any(c -> isnan(val(c)), counted)
  nz = count(c -> !iszero(val(c)), counted)
  standalone = nan && s < m ? T(NaN) : nz <= s ? T(0) : T(Inf)
  Y = copy(X)
  if nan && 0 < s < m
    for c in counted, p in c.pos
      Y[p] = T(NaN)
    end
    return Y, standalone, T(NaN)
  elseif nz <= s
    return Y, standalone, T(0)
  end
  order = sortperm([-T(c.w) * abs(val(c)) for c in counted]; alg = MergeSort)   # stable: the first entries win ties
  for c in counted[order[(s + 1):end]], p in c.pos
    Y[p] = zero(T)
  end
  Y, standalone, T(0)
end

# the coordinates of a p × q matrix whose entries count one by one, and of a symmetric matrix (lower triangle)
function plain_coords(p, q, diagonal)
  [Coord([CartesianIndex(r, c)], 1.0, diagonal || r != c) for c in 1:q for r in 1:p]
end
function symmetric_coords(p, diagonal)
  [Coord(r == c ? [CartesianIndex(r, c)] : [CartesianIndex(r, c), CartesianIndex(c, r)],
         diagonal && r != c ? sqrt(2.0) : 1.0, diagonal || r != c) for c in 1:p for r in c:p]
end

# no NaN, then each coordinate (a free diagonal included) set to NaN in turn, and the first with the last
function nan_variants(X, coords)
  out = [copy(X)]
  for c in coords
    Z = copy(X)
    Z[c.pos] .= NaN
    push!(out, Z)
  end
  Z = copy(X)
  Z[first(coords).pos] .= NaN
  Z[last(coords).pos] .= NaN
  push!(out, Z)
  out
end

function check_l0(f, X, coords, make_x, make_y, T; alias = true)
  Y, standalone, fused = ref_l0(X, coords, f.s)
  x = make_x(X)
  @test same(T(f(x)), standalone)
  y = make_y(fill(T(7), size(X)))                     # a sentinel: every counted output is overwritten
  prox!(y, f, x, 1)
  ok = same(dense(y), Y)
  @test ok
  y = make_y(fill(T(7), size(X)))
  v = value_and_prox!(y, f, x, 1)
  @test same(dense(y), Y) && same(T(v), fused)
  if alias && !(x isa Symmetric)
    z = make_x(X)
    prox!(z, f, z, 1)
    @test same(dense(z), Y)
  end
  ok
end

@testset "IndBallL0 at NaN, $T" for T in (Float32, Float64)
  # vectors, dense and SparseIterate
  n = 5
  for s in 0:(n + 1), X0 in (T[1.5, 0, -3, 0.25, 3], T[2, 2, 0, 1, 2])
    X = reshape(X0, n, 1)
    coords = plain_coords(n, 1, true)
    for Z in nan_variants(X, coords)
      f = IndBallL0(s)
      Y, standalone, fused = ref_l0(Z, coords, s)
      x = vec(copy(Z))
      @test same(T(f(x)), standalone)
      y = fill(T(7), n); prox!(y, f, x, 1); @test same(y, vec(Y))
      y = fill(T(7), n); v = value_and_prox!(y, f, x, 1); @test same(y, vec(Y)) && same(T(v), fused)
      z = copy(x); prox!(z, f, z, 1); @test same(z, vec(Y))
      xs = SparseIterate(copy(x))
      @test same(T(f(xs)), standalone)
      ys = SparseIterate(T, n); prox!(ys, f, xs, 1); @test same(dense(ys), vec(Y))
      ys = SparseIterate(T, n); v = value_and_prox!(ys, f, xs, 1); @test same(dense(ys), vec(Y)) && same(T(v), fused)
      zs = copy(xs); prox!(zs, f, zs, 1); @test same(dense(zs), vec(Y))
      yd = fill(T(7), n); prox!(yd, f, xs, 1); @test same(yd, vec(Y))        # a sparse argument, a dense output
    end
  end
  # plain matrices, the diagonal counting or free, dense and SparseIterate
  p = 3
  for diagonal in (true, false), s in 0:(p * p + 1)
    X = T[1.5 0 -3; 0.5 2 1.5; 0 -1 3]
    coords = plain_coords(p, p, diagonal)
    for Z in nan_variants(X, coords)
      f = IndBallL0(s; diagonal)
      check_l0(f, Z, coords, copy, identity, T)
      check_l0(f, Z, coords, A -> SparseIterate(copy(A)), A -> SparseIterate(T, p, p), T)
    end
  end
  # symmetric matrices: a mirrored pair is one coordinate, as input Symmetric or SymmetricSparseIterate and as output
  p = 4
  for diagonal in (true, false), s in 0:(p * (p + 1) ÷ 2 + 1)
    D = T[(i + 2j) % 5 + 0.5 * (i * j % 3) for i in 1:p, j in 1:p]
    D = (D + D') / 2 .+ T(0.01) .* T[(i * j) % 7 for i in 1:p, j in 1:p]
    coords = symmetric_coords(p, diagonal)
    for Z in nan_variants(D, coords)
      f = IndBallL0(s; diagonal)
      check_l0(f, Z, coords, A -> Symmetric(copy(A), :L), identity, T)
      check_l0(f, Z, coords, A -> Symmetric(copy(A), :L), A -> SymmetricSparseIterate(T, p), T)
      check_l0(f, Z, coords, A -> SymmetricSparseIterate(copy(A)), identity, T)
      check_l0(f, Z, coords, A -> SymmetricSparseIterate(copy(A)), A -> SymmetricSparseIterate(T, p), T)
    end
  end
end

@testset "IndBallL0: the named cases" begin
  @test isnan(IndBallL0(1)([NaN, 0.0]))                                        # no false feasibility
  @test same(prox(IndBallL0(1), [NaN, 0.0], 1.0), [NaN, NaN])
  @test same(prox(IndBallL0(1), [NaN, 1.5], 1.0), [NaN, NaN])
  @test same(prox(IndBallL0(1), [1.5, NaN], 1.0), [NaN, NaN])                  # the order of the entries does not matter
  @test same(prox(IndBallL0(2), [NaN, 1.5, 3.0], 1.0), [NaN, NaN, NaN])
  @test same(prox(IndBallL0(2), [NaN, 1.5], 1.0), [NaN, 1.5])                  # s ≥ m: the identity
  @test IndBallL0(2)([NaN, 1.5]) == 0 && value_and_prox!(zeros(2), IndBallL0(2), [NaN, 1.5], 1.0) == 0
  @test same(prox(IndBallL0(0), [NaN, 1.5], 1.0), [0.0, 0.0])                  # s = 0: the zero map
  @test value_and_prox!(zeros(2), IndBallL0(0), [NaN, 1.5], 1.0) == 0
  @test isnan(IndBallL0(0)([NaN, 1.5])) && isnan(IndBallL0(0)([0.0, NaN]))
  @test IndBallL0(0)([0.0, 1.5]) == Inf && IndBallL0(0)([0.0, 0.0]) == 0
  @test isnan(value_and_prox!(zeros(2), IndBallL0(1), [NaN, 1.5], 1.0))
  for γ in (0.0, 1.0, Inf)                                                    # the step does not matter
    @test same(prox(IndBallL0(1), [NaN, 1.5, 0.0], γ), [NaN, NaN, NaN])
  end
  # a free diagonal copies and never enters the value
  @test same(prox(IndBallL0(1; diagonal = false), [NaN 1.0; 1.0 2.0], 1.0), [NaN 0.0; 1.0 2.0])
  @test same(prox(IndBallL0(1; diagonal = false), [5.0 NaN; NaN 2.0], 1.0), [5.0 NaN; NaN 2.0])
  @test isnan(IndBallL0(1; diagonal = false)([5.0 NaN; NaN 2.0]))
  @test IndBallL0(1; diagonal = false)([NaN 1.0; 0.0 2.0]) == 0
  @test value_and_prox!(zeros(2, 2), IndBallL0(1; diagonal = false), [NaN 1.0; 1.0 2.0], 1.0) == 0
  # a mirrored pair is one coordinate: with one counted pair, s = 1 is the identity
  S = Symmetric([5.0 NaN; NaN 2.0], :L)
  @test same(prox(IndBallL0(1; diagonal = false), S), [5.0 NaN; NaN 2.0]) && IndBallL0(1; diagonal = false)(S) == 0
  # the step is validated before anything is written, NaN in the argument or not
  y = [5.0, 6.0]
  @test_throws ArgumentError prox!(y, IndBallL0(1), [NaN, 1.0], -1.0)
  @test_throws ArgumentError value_and_prox!(y, IndBallL0(1), [NaN, 1.0], NaN)
  @test y == [5.0, 6.0]
  # invalid inputs together: value_and_prox! checks the shapes before the step (as the generic body does) and
  # prox! the step first, whatever the argument; nothing is written either way
  for x in ([NaN, 2.0], [1.0, 2.0]), γ in (-1.0, NaN)
    y3 = fill(7.0, 3)                                     # the wrong size
    @test_throws DimensionMismatch value_and_prox!(y3, IndBallL0(1), x, γ)
    @test y3 == fill(7.0, 3)
    @test_throws ArgumentError prox!(y3, IndBallL0(1), x, γ)
    @test y3 == fill(7.0, 3)
    y2 = fill(7.0, 2)                                     # the right size, a bad step
    @test_throws ArgumentError value_and_prox!(y2, IndBallL0(1), x, γ)
    @test y2 == fill(7.0, 2)
  end
  # a NaN block cannot be stored in an integer output: the write is refused before it starts
  yi = [5, 6]
  @test_throws InexactError prox!(yi, IndBallL0(1), [NaN, 2.0], 1.0)
  @test yi == [5, 6]
  # the wrappers that reach the projection through value_and_prox! report the NaN value
  @test isnan(value_and_prox!(zeros(2), Scaled(IndBallL0(1), 2.0), [NaN, 1.0], 1.0))
  @test isnan(value_and_prox!(zeros(2), Translate(IndBallL0(1), [0.1, 0.1]), [NaN, 0.5], 1.0))
end

@testset "IndBallL0: finite arguments are unchanged" begin
  @test prox(IndBallL0(1), [1.0, 1.0, 1.0], 1.0) == [1.0, 0.0, 0.0]            # the first entry wins a tie
  @test prox(IndBallL0(2), [3.0, 1.0, 1.0, 1.0], 1.0) == [3.0, 1.0, 0.0, 0.0]
  @test prox(IndBallL0(1), [Inf, 1.5, 3.0], 1.0) == [Inf, 0.0, 0.0]
  @test prox(IndBallL0(2), [0.5, -2.0, 0.05, 1.2], 1.0) == [0.0, -2.0, 0.0, 1.2]
  @test IndBallL0(2)([1.0, 0.0, 2.0]) == 0 && IndBallL0(1)([1.0, 0.0, 2.0]) == Inf
  @test value_and_prox!(zeros(3), IndBallL0(1), [1.0, 0.0, 2.0], 1.0) == 0
  @test prox(IndBallL0(1; diagonal = false), [3.0 1.0; 2.0 4.0], 1.0) == [3.0 0.0; 2.0 4.0]
end

end
