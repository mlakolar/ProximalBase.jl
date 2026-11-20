module IndBoxTest

using Test, ProximalBase, OffsetArrays
const PB = ProximalBase

# ---- Scaled returns the indicator (the same checks for each built-in indicator) -----------------------------
const CANON_FACTORS = Any[2, 0, 1, 0.5, 2.0, 2f0, 3f0, 5e-324, 1e-300, floatmax(Float64), floatmax(Float32), nextfloat(0f0),
                          Inf, Inf32, 0.0, -0.0, big(2.0), big(0), big(Inf)]
const CANON_INVALID = Any[-1, -1.0, NaN, -Inf, -1f0, NaN32, -5e-324, -Inf32, big(-1.0), big(NaN)]
scaled_allocs(f, α) = (Scaled(f, α); @allocated Scaled(f, α))
times_allocs(f, α) = (α * f; @allocated α * f)

# The listed valid factors, nested chains whose product overflows, underflows or passes through Inf, and the
# invalid factors; construction is inferred and does not allocate.
function check_canonical(f)
  for α in CANON_FACTORS
    @test Scaled(f, α) === f
    @test α * f === f
    @test f * α === f
    @test Scaled{AbstractFunction, typeof(α)}(f, α) === f
    @test Scaled{typeof(f), typeof(α)}(f, α) === f
  end
  @test Scaled(Scaled(f, 1e300), 1e100) === f
  @test Scaled(Scaled(f, 1e-300), 1e-100) === f
  @test Scaled(Scaled(f, 1f30), 1f30) === f
  @test Scaled(Scaled(f, 5e-324), 5e-324) === f
  @test 0 * (Inf * f) === f
  @test Scaled(Scaled(f, Inf), 0) === f
  @test Inf * (0 * f) === f
  @test (2 * f) * 3 === f
  for α in CANON_INVALID
    @test_throws ArgumentError Scaled(f, α)
    @test_throws ArgumentError α * f
    @test_throws ArgumentError f * α
    @test_throws ArgumentError Scaled{AbstractFunction, typeof(α)}(f, α)
  end
  @test_throws ArgumentError Scaled(Scaled(f, 2), -1)
  @test_throws ArgumentError Scaled(Scaled(f, Inf), NaN)
  for α in (2, 2.0, 2f0, 0, Inf, -0.0, 1e-300)
    @test @inferred(Scaled(f, α)) === f
    @test @inferred(α * f) === f
    @test scaled_allocs(f, α) == 0
    @test times_allocs(f, α) == 0
  end
  @test @inferred(Scaled(Scaled(f, 1e300), 1e100)) === f
  @test !(Scaled(f, 2) isa Scaled) && Scaled(f, 2) isa typeof(f)
  @test Scaled(NormL1(1.0), 2.0) isa Scaled
end

# A negative, NaN or tiny negative scalar step through a scaled indicator throws before anything is written, and
# the other steps give the result of the indicator at the points x (the vectors of one element type).
function check_scaled_steps(f, points)
  for α in (2, Inf, 1e-300), x in points
    T = eltype(x)
    s = α * f
    for γ in (-one(T), T(NaN), -nextfloat(zero(T)), -1)
      y = fill(T(7), length(x))
      @test_throws ArgumentError prox!(y, s, x, γ)
      @test all(==(T(7)), y)
      @test_throws ArgumentError value_and_prox!(y, s, x, γ)
      @test all(==(T(7)), y)
      @test_throws ArgumentError prox(s, x, γ)
    end
    for γ in (zero(T), one(T), T(0.3), T(Inf), 2, 0)
      y1, y2 = similar(x), similar(x)
      @test isequal(prox!(y1, s, x, γ), prox!(y2, f, x, γ)) && isequal(y1, y2)
      @test isequal(s(x), f(x))
      v1 = value_and_prox!(y1, s, x, γ)
      v2 = value_and_prox!(y2, f, x, γ)
      @test isequal(v1, v2) && isequal(y1, y2) && isequal(v1, s(y1))
    end
  end
end

at(b::Real, i) = b
at(b::AbstractArray, i) = b[i]

function refproject(l, u, x)
  map(CartesianIndices(x)) do I
    i = LinearIndices(x)[I]
    v = x[I]
    lo, hi = at(l, i), at(u, i)
    v < lo ? lo : v > hi ? hi : v
  end
end

function refvalue(l, u, x)
  T = eltype(x)
  any(isnan, x) && return T(NaN)
  for (i, v) in enumerate(x)
    (v < at(l, i) || v > at(u, i)) && return T(Inf)
  end
  zero(T)
end

@testset "Bounds, projection and numerical endpoints" begin
  for T in (Float32, Float64)
    values = T[-Inf, -floatmax(T), -2, -1, -nextfloat(zero(T)), -0.0, 0, 1, 2, floatmax(T), Inf, NaN]
    for (l, u) in ((T(-Inf), T(Inf)), (zero(T), T(Inf)), (T(-Inf), T(-1)),
                  (T(-1), one(T)), (one(T), one(T)), (-floatmax(T), floatmax(T)))
      f = IndBox(l, u)
      @test f isa SeparableFunction && isconvex(f) && isproximable(f) && isseparable(f)
      @test PB._isindicator(f) && Scaled(f, zero(T)) === f
      @test iszeropreserving(f) == (l <= 0 <= u)
      for v in values
        @test isequal(f(T[v]), refvalue(l, u, T[v]))
        @test isequal(value_entry(f, v, 1), refvalue(l, u, T[v]))
        @test isequal(prox_block(f, v, 1, zero(T)), only(refproject(l, u, T[v])))
      end
      for γ in (zero(T), T(0.7), T(Inf), fill(zero(T), length(values)))
        projected = refproject(l, u, values)
        y = copy(values)
        @test isequal(prox(f, values, γ), projected)
        @test isequal(value_and_prox!(y, f, y, γ), f(y))
        @test isequal(y, projected)
      end
    end
    @test IndBox(zero(T), one(T))(values[1:11]) === T(Inf)
    @test isnan(IndBox(zero(T), one(T))(values))
    f = IndBox(zero(T), one(T))
    long = zeros(T, 600)
    @test f(long) === zero(T)
    long[257] = -one(T)
    @test f(long) === T(Inf)
    long[end] = T(NaN)
    @test isnan(f(long))

    X = reshape(T[-2, 0, 3, -4], 2, 2)
    L, U = T[-1, 1, 2, -Inf], T[1 2; 2 Inf]
    for (l, u) in ((L, U), (T(-1), U), (L, T(3)))
      f = IndBox(l, u)
      Xv = view(X, :, :)
      y = similar(X)
      @test f(Xv) === refvalue(l, u, X)
      @test prox(f, Xv) == refproject(l, u, X)
      @test value_and_prox!(y, f, Xv, zero(T)) === f(y) === zero(T)
      @test y == refproject(l, u, X)
      g = Scaled(f, T(Inf))
      @test g === f
      @test value_and_prox!(y, g, Xv, zero(T)) === g(y) === zero(T)
      @test y == refproject(l, u, X)
      Xn = copy(X)
      Xn[end] = T(NaN)
      @test isnan(f(Xn))
      @test isnan(value_and_prox!(y, g, Xn, zeros(T, size(Xn))))
      @test isequal(y, refproject(l, u, Xn))
    end
    @test IndBox(T[], T[])(T[]) === zero(T)
  end
  # A Float32 output cannot represent the Float64 fixed endpoint.
  f = IndBox(0.1, 0.1)
  y = zeros(Float32, 1)
  @test value_and_prox!(y, f, [0.0], 0.0) === f(y) === Inf32
end

@testset "Scaled returns the box" begin
  boxes = (IndBox(0.0, 1.0), IndBox(-Inf, Inf), IndBox(0, 1), IndBox(1.5, 1.5), IndBallLinf(2.0), IndBallLinf(Inf),
           IndBox(Float32(-1), 2f0), IndBox([0.0, 1.0], 2.0), IndBox(Float32[-1, 0], Float32[1, 2]),
           IndBox([-Inf, 0.0], [Inf, 3.0]))
  for f in boxes
    check_canonical(f)
  end
  f = IndBox(0.0, 1.0)
  @test Scaled(f, 2.0) === f && Scaled(f, Inf) === f && Scaled(f, 0) === f
  @test Scaled{AbstractFunction, Float64}(f, 2.0) === f
  g = IndBallLinf(2.0)
  @test Scaled(g, 3) === g && 3 * g === g
  points = (Float64[-1, 0.5], Float64[0.5, 1.5], Float64[NaN, 0.5], Float64[0.25, 0.75], Float64[])
  for f in (IndBox(0.0, 1.0), IndBox(-Inf, Inf), IndBox(0, 1), IndBallLinf(0.5))
    check_scaled_steps(f, points)
  end
  check_scaled_steps(IndBox([0.0, 1.0], 2.0), (Float64[-1, 1.5], Float64[0.5, 3.0], Float64[NaN, 1.5], Float64[0.0, 2.0]))
  check_scaled_steps(IndBox(Float32[-1, 0], Float32[1, 2]), (Float32[-3, 1], Float32[0.5, 3], Float32[NaN, 1], Float32[0, 2]))
  # the hooks of the box are visible through a scale: the bound arrays are checked against the argument
  b = IndBox([0.0, 0.0], [1.0, 1.0])
  for α in (2, Inf, 1e-300)
    s = α * b
    @test s isa SeparableFunction && s === b
    @test_throws DimensionMismatch PB.check_parameters(s, [0.0, 0.0, 0.0])
    @test_throws DimensionMismatch prox(s, [0.0, 0.0, 0.0], 1.0)
    @test PB.entry_parameters(s) == PB.entry_parameters(b)
    @test iszeropreserving(s) == iszeropreserving(b)
  end
  @test iszeropreserving(Inf * IndBox(0.0, 1.0)) && iszeropreserving(Inf * IndBallLinf(1.0))
  @test !iszeropreserving(Inf * IndBox(1.0, 2.0))
  # an unsupported form of the score names the box
  err = try; kkt_violation(2 * IndBox(0.0, 1.0), [1.0], 1); catch e; e; end
  @test err isa MethodError && !occursin("Scaled", sprint(showerror, err))
end

@testset "Validation, ownership and restriction" begin
  for (l, u) in ((1, 0), (Inf, Inf), (-Inf, -Inf), (NaN, 1), (0, NaN),
                 (Inf, Float64[]), (Float64[], NaN), ([-1, NaN], 1), (0, [-Inf]),
                 ([2, 0], [1, 1]), (-1, [1 + im]), (im, 1), (OffsetArray([0], 0:0), 1))
    @test_throws ArgumentError IndBox(l, u)
  end
  @test_throws DimensionMismatch IndBox([0, 0], [1])
  l, u = [-2.0, 1.0, -1.0], [2.0, 3.0, 1.0]
  f = IndBox(view(l, :), u)
  fill!(l, 9.0)
  fill!(u, -9.0)
  @test prox(f, [-3.0, 0.0, 5.0]) == [-2.0, 1.0, 1.0]
  @test !iszeropreserving(f)
  g = restrict(f, [3, 2])
  @test prox(g, [5.0, 0.0]) == [1.0, 1.0]
  @test g([0.5, 2.0]) === f([0.0, 2.0, 0.5]) === 0.0
  @test_throws ArgumentError restrict(f, [1, 3])
  @test_throws ArgumentError restrict(f, [3, 3])
  @test_throws BoundsError restrict(f, [2, 4])
  @test_throws ArgumentError restrict(IndBox(1, 2), [1])
  @test restrict(IndBox(-1, 2), Int[])(Float64[]) === 0.0
  @test iszeropreserving(restrict(IndBox([0.0], Inf), Int[]))
  lo = [0.0, 0.0]
  zero_box = IndBox(lo, 1.0)
  lo[1] = 2.0
  @test iszeropreserving(zero_box) && zero_box(zeros(2)) === 0.0

  for op in (prox!, value_and_prox!)
    x = [4.0, -5.0]
    for γ in (-1.0, -Inf, NaN)
      @test_throws ArgumentError op(x, IndBox(-1, 1), x, γ)
      @test x == [4.0, -5.0]
    end
    @test_throws DimensionMismatch op(x, f, x, 1.0)
    @test_throws DimensionMismatch op(x, IndBox(-1, 1), x, [1.0])
    @test_throws ArgumentError op(x, IndBox(-1, 1), OffsetArray(copy(x), 0:1), 1.0)
    @test_throws ArgumentError op(x, IndBox(-1, 1), complex.(x), 1.0)
    @test x == [4.0, -5.0]
  end
end

@testset "Sparse, block and symmetric iterates" begin
  for T in (Float32, Float64)
    s = SparseIterate(T, 4)
    s[2] = T(-3)
    b = BlockIterate(Vector(s), [[1, 2], [3, 4]])
    for x in (s, b), (l, u) in ((T(-1), T[2, 1, 3, 1]), (T[0, -2, 1, 0], T(2)))
      f = IndBox(l, u)
      reference = refproject(l, u, Vector(x))
      @test f(x) === refvalue(l, u, Vector(x))
      @test Vector(prox(f, x, zero(T))) == reference
      y = copy(x)
      @test value_and_prox!(y, f, y, T(Inf)) === f(y) === zero(T)
      @test Vector(y) == reference
    end
    @test nactive(prox(IndBox(T(-1), one(T)), s, zero(T))) == 1

    x = SymmetricSparseIterate(T, 3)
    x[2, 1] = T(-3)
    x[3, 2] = T(4)
    U = T[1 2 3; 2 1 2; 3 2 1]
    for l in (T(-1), T[0 -2 1; -2 0 -2; 1 -2 0])
      f = IndBox(l, U)
      M = Matrix(x)
      reference = refproject(l, U, M)
      @test PB.check_symmetric_parameters(f, x)
      @test f(x) === refvalue(l, U, M)
      @test Matrix(prox(f, x, zero(T))) == reference
      @test prox!(similar(M), f, x, zero(T)) == reference
      y = copy(x)
      @test value_and_prox!(y, f, y, zero(T)) === f(y) === zero(T)
      @test Matrix(y) == reference
      @test nactive(y) == (iszeropreserving(f) ? 2 : 3)
    end
    f = IndBox(T(-1), U)
    asym = copy(U)
    asym[1, 3] = T(2)
    @test_throws ArgumentError PB.check_symmetric_parameters(IndBox(T(-1), asym), x)
    @test_throws ArgumentError prox!(x, f, x, fill(zero(T), 3, 3))
    @test_throws DimensionMismatch value_and_prox!(x, IndBox(T(-1), T[2]), x, zero(T))
    @test x[2, 1] == T(-3) && x[3, 2] == T(4) && nactive(x) == 2
    x[3, 3] = T(NaN)
    @test isnan(f(x))
    @test isnan(value_and_prox!(x, f, x, zero(T))) && isnan(x[3, 3])
  end
end

# Counting zero construction detects getindex on an unstored sparse entry.
const missing_reads = Ref(0)
struct Counted <: Real
  v::Float64
end
Base.zero(::Type{Counted}) = (missing_reads[] += 1; Counted(0.0))
Base.iszero(x::Counted) = iszero(x.v)
Base.isnan(x::Counted) = isnan(x.v)
Base.float(::Type{Counted}) = Float64
Base.Float64(x::Counted) = x.v
Base.convert(::Type{Counted}, x::Float64) = Counted(x)
Base.:<=(a::Float64, b::Counted) = a <= b.v
Base.:<=(a::Counted, b::Float64) = a.v <= b

@testset "Zero-preserving sparse paths skip implicit entries" begin
  p = 40
  s = SparseIterate(Counted, p)
  s[2] = Counted(-4.0)
  x = SymmetricSparseIterate(Counted, p)
  x[2, 1] = Counted(-4.0)
  for a in (s, x)
    f = IndBox(-1.0, ones(prod(size(a))))
    missing_reads[] = 0
    @test f(a) === Inf
    @test value_and_prox!(a, f, a, 0.0) === 0.0
    @test missing_reads[] == 0
    a isa SparseIterate ? a[1] : a[1, 1]
    @test missing_reads[] == 1
  end
end

end
