module SparseValueCostTest
using Test, ProximalBase, LinearAlgebra
const PB = ProximalBase

# The value of a LinearFunction at a matrix SparseIterate is computed from the stored nonzero
# entries: it forms no zero for an unstored entry and reads the coefficients at the nonzero
# stored entries (#810), on the ordinary path and on the recovery of an overflowing dot.
# Two detectors count it. A real type C whose zero is counted shows that no pass over the
# unstored entries is made (it is Float64-backed, so a Float32 coefficient array runs in
# Float64 here). A coefficient array that records the linear indices it is read at shows
# which entries are read, with plain Float32 and Float64 iterates.

# zero(C) is counted: a pass over unstored entries creates one for each entry.
const ZEROS = Ref(0)
struct C <: AbstractFloat
  v::Float64
end
Base.zero(::Type{C}) = (ZEROS[] += 1; C(0.0))
Base.zero(::C) = zero(C)
Base.float(::Type{C}) = Float64
Base.float(x::C) = x.v
Base.convert(::Type{C}, x::Real) = C(Float64(x))
Base.convert(::Type{Float64}, x::C) = x.v
Base.Float64(x::C) = x.v
Base.promote_rule(::Type{C}, ::Type{T}) where {T<:Real} = promote_type(Float64, T)
for op in (:abs, :abs2, :-)
  @eval Base.$op(x::C) = C($op(x.v))
end
Base.isnan(x::C) = isnan(x.v)
Base.isfinite(x::C) = isfinite(x.v)
Base.iszero(x::C) = iszero(x.v)
Base.:(==)(x::C, y::C) = x.v == y.v
Base.:<(x::C, y::C) = x.v < y.v
Base.:<=(x::C, y::C) = x.v <= y.v
Base.:+(x::C, y::C) = C(x.v + y.v)
Base.:*(x::C, y::C) = C(x.v * y.v)

zeros_made(f) = (ZEROS[] = 0; v = f(); (ZEROS[], v))

# a coefficient matrix that records the linear index of each read through getindex(c, i::Int)
struct Reading{T} <: AbstractMatrix{T}
  a::Matrix{T}
  seen::Vector{Int}
end
Reading(a) = Reading(a, Int[])
Base.size(c::Reading) = size(c.a)
Base.IndexStyle(::Type{<:Reading}) = IndexLinear()
Base.getindex(c::Reading, i::Int) = (push!(c.seen, i); c.a[i])
# the coefficient vector of a Diagonal, recording the position of each read
struct ReadingVector{T} <: AbstractVector{T}
  a::Vector{T}
  seen::Vector{Int}
end
ReadingVector(a) = ReadingVector(a, Int[])
Base.size(c::ReadingVector) = size(c.a)
Base.getindex(c::ReadingVector, i::Int) = (push!(c.seen, i); c.a[i])
reads!(c::Reading) = (r = copy(c.seen); empty!(c.seen); r)
reads!(c::ReadingVector) = (r = copy(c.seen); empty!(c.seen); r)

# Writing a nonzero first keeps the coordinate stored when its value is zero.
function make_iterate(T, p, entries)
  x = SparseIterate(T, p, p)
  for (r, c, v) in entries
    x[r, c] = T(1)
    x[r, c] = T(v)
  end
  x
end
linear(p, r, c) = (c - 1) * p + r
# the linear indices of the entries with a nonzero value (a NaN is nonzero)
nonzero_indices(p, entries) = Set(linear(p, r, c) for (r, c, v) in entries if !iszero(v))

@testset "Sparse value cost" begin
  @testset "No zero is formed for an unstored entry" begin
    for p in (100, 1000), cT in (Float64, Float32)
      entries = [(1, 1, 1.0), (2, 1, 0.5), (3, 3, 0.0), (4, 4, -0.0)]
      x = make_iterate(C, p, entries)
      @test nactive(x) == 4                         # the two zeros are stored
      c = ones(cT, p, p)
      c[5, 5] = NaN; c[3, 3] = Inf; c[4, 4] = -Inf   # an unstored zero and the stored zeros
      n, value = zeros_made(() -> LinearFunction(c, cT(0.25))(x))
      @test n == 0
      @test Float64(value) == 1.75
      # the recovery of an overflowing dot
      big = make_iterate(C, p, [(1, 1, 1e308), (2, 1, 1e308), (3, 3, 0.0), (4, 4, -0.0)])
      g = LinearFunction(fill(cT(1), p, p), -1.5e308)
      c2 = ones(cT, p, p); c2[5, 5] = Inf; c2[3, 3] = NaN
      n, value = zeros_made(() -> g(big))
      @test n == 0 && Float64(value) == 5.0e307
      n, value = zeros_made(() -> LinearFunction(c2, -1.5e308)(big))
      @test n == 0 && Float64(value) == 5.0e307
      # an empty iterate
      empty_x, g = make_iterate(C, p, []), LinearFunction(fill(cT(NaN), p, p), 0.5)
      n, value = zeros_made(() -> g(empty_x))
      @test n == 0 && Float64(value) == 0.5
    end
  end

  @testset "The coefficients are read at the nonzero stored entries" begin
    for T in (Float64, Float32)
      counts = Dict{Symbol, Int}()
      for p in (100, 1000)
        entries = [(1, 1, 1.0), (2, 1, -0.5), (3, 3, 0.0), (4, 4, -0.0), (7, 9, 3.0)]
        x = make_iterate(T, p, entries)
        allowed = nonzero_indices(p, entries)
        a = ones(T, p, p)
        a[3, 3] = NaN; a[4, 4] = Inf; a[5, 5] = -Inf; a[6, 6] = NaN     # stored and unstored zeros
        c = Reading(a)
        value = LinearFunction(c, T(0.25))(x)
        seen = reads!(c)
        @test value == T(3.75)
        @test !isempty(seen) && issubset(Set(seen), allowed)             # nonzero stored entries only
        @test length(seen) <= nactive(x)                                 # at most nactive(x) reads in all (a zero entry reads i0 again)
        counts[:ordinary] = length(seen)

        # the recovery of an overflowing dot: a pair of large entries and a negative d
        big = T == Float64 ? T(1e308) : T(2e38)
        d = T == Float64 ? T(-1.5e308) : T(-3e38)
        entries = [(1, 1, big), (2, 1, big), (3, 3, 0.0), (4, 4, -0.0)]
        x = make_iterate(T, p, entries)
        allowed = nonzero_indices(p, entries)
        c = Reading(a)
        value = LinearFunction(c, d)(x)
        seen = reads!(c)
        @test isfinite(value)
        @test isapprox(value, T == Float64 ? T(5.0e307) : T(1.0e38); rtol = 12 * eps(T))
        @test !isempty(seen) && issubset(Set(seen), allowed)
        @test length(seen) <= 2 * nactive(x) + 2                         # the ordinary sum, then the recomputation
        # the count of reads does not grow with the size
        if haskey(counts, :recovery)
          @test length(seen) == counts[:recovery]
        end
        counts[:recovery] = length(seen)
      end
    end
  end

  @testset "A Diagonal coefficient array is read at the nonzero stored diagonal entries" begin
    for T in (Float64, Float32)
      for p in (100, 1000)
        entries = [(1, 1, 2.0), (2, 2, 0.0), (5, 5, -0.0), (1, 2, NaN), (9, 3, 4.0)]    # (1,2) and (9,3) are off the diagonal
        x = make_iterate(T, p, entries)
        v = ones(T, p); v[2] = NaN; v[5] = Inf; v[3] = NaN
        c = ReadingVector(v)
        value = LinearFunction(Diagonal(c), T(0.25))(x)
        seen = reads!(c)
        @test value == T(2.25)
        @test !isempty(seen) && issubset(Set(seen), Set([1]))
        @test length(seen) <= p                                          # one read for each diagonal position at most
        # the recovery: the diagonal entries (1,1) and (2,2) are large, a NaN off the diagonal is ignored
        big = T == Float64 ? T(1e308) : T(2e38)
        d = T == Float64 ? T(-1.5e308) : T(-3e38)
        x = make_iterate(T, p, [(1, 1, big), (2, 2, big), (5, 5, 0.0), (1, 2, NaN), (3, 3, -0.0)])
        v = ones(T, p); v[5] = Inf; v[3] = NaN
        c = ReadingVector(v)
        value = LinearFunction(Diagonal(c), d)(x)
        seen = reads!(c)
        @test isapprox(value, T == Float64 ? T(5.0e307) : T(1.0e38); rtol = 12 * eps(T))
        @test !isempty(seen) && issubset(Set(seen), Set([1, 2]))
        @test length(seen) <= p + 4
      end
    end
  end

  # A fully stored iterate with stored zeros among the nonzero entries: no zero is formed, the coefficients read are at
  # nonzero stored entries, and the number of reads is bounded.
  @testset "A dense iterate: no zero formed, coefficients read at nonzero stored entries" begin
    p = 100
    entries = [(r, c, 1.0 + r / p) for c in 1:p for r in 1:p]
    for (r, c) in ((3, 3), (4, 4), (10, 20), (50, 50), (99, 1)), (k, e) in enumerate(entries)
      (e[1], e[2]) == (r, c) && (entries[k] = (r, c, k % 2 == 0 ? 0.0 : -0.0))
    end
    zeros_at = [(r, c) for (r, c, v) in entries if iszero(v)]
    @test length(zeros_at) == 5
    for cT in (Float64, Float32)
      x = make_iterate(C, p, entries)
      @test nactive(x) == p^2
      a = ones(cT, p, p)
      for (i, (r, c)) in enumerate(zeros_at)
        a[r, c] = (NaN, Inf, -Inf)[mod1(i, 3)]
      end
      g = LinearFunction(a, cT(0.25))
      n, value = zeros_made(() -> g(x))
      @test n == 0
      expected = sum(Float64(a[r, c]) * v for (r, c, v) in entries if !iszero(v)) + 0.25
      @test isapprox(Float64(value), expected; rtol = 1e-12)
    end
    for T in (Float64, Float32)
      x = make_iterate(T, p, entries)
      allowed = nonzero_indices(p, entries)
      a = ones(T, p, p)
      for (i, (r, c)) in enumerate(zeros_at)
        a[r, c] = (T(NaN), T(Inf), -T(Inf))[mod1(i, 3)]
      end
      c = Reading(a)
      value = LinearFunction(c, T(0.25))(x)
      seen = reads!(c)
      @test isfinite(value)
      @test !isempty(seen) && issubset(Set(seen), allowed)
      @test length(seen) <= p^2
    end
  end

  @testset "Detector control: a dense pass is caught" begin
    for p in (100, 200)
      entries = [(1, 1, 1.0), (2, 1, 0.5)]
      # the counted zero
      x = make_iterate(C, p, entries)
      c = ones(p, p)
      n, _ = zeros_made(() -> invoke(dot, Tuple{AbstractArray, AbstractArray}, c, x))
      @test n >= length(x) - nactive(x)
      n, _ = zeros_made(() -> sum(c .* x))
      @test n >= length(x) - nactive(x)
      # the reads
      y = make_iterate(Float64, p, entries)
      r = Reading(ones(p, p))
      invoke(dot, Tuple{AbstractArray, AbstractArray}, r, y)
      @test Set(reads!(r)) == Set(1:p^2)
      sum(r .* y)
      @test Set(reads!(r)) == Set(1:p^2)
    end
  end
end

end
