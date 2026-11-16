module OneBasedAxesTest
# One-based array arguments (#920). The built-in functions, their constructors and the products of the iterates read
# their array arguments by position, under @inbounds or in loops over 1:n, so an array whose axes do not start at 1
# is an ArgumentError, raised before anything is read or written. This file checks
#   - construction with an offset stored array;
#   - the oracles, the screens and the products with an offset input, destination or step array, and that the
#     destination, the scratch and the cached factorizations are left as they were;
#   - the routes that keep their earlier DimensionMismatch, and the extension routes that are not checked;
#   - that ordinary one-based arrays (views of longer arrays included) give the results of plain arrays;
#   - the same rejections in a child process with the default bounds checking, where the runner's
#     --check-bounds=yes would turn a misread into a late BoundsError.
using Test, ProximalBase, LinearAlgebra, SparseArrays, OffsetArrays
const PB = ProximalBase

####################################
#  Offset arrays
####################################

# An independent vector with the axes (o+1):(o+n), linear indexing in those axes and a bounds-checked getindex.
struct Shifted{T} <: AbstractVector{T}
  data::Vector{T}
  o::Int
end
Base.size(s::Shifted) = size(s.data)
Base.axes(s::Shifted) = (Base.IdentityUnitRange(s.o+1:s.o+length(s.data)),)
Base.IndexStyle(::Type{<:Shifted}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(s::Shifted, i::Int)
  @boundscheck checkbounds(s, i)
  s.data[i - s.o]
end
Base.@propagate_inbounds function Base.setindex!(s::Shifted, v, i::Int)
  @boundscheck checkbounds(s, i)
  s.data[i - s.o] = v
end
Base.similar(s::Shifted, ::Type{T}, d::Tuple{Base.IdentityUnitRange}) where {T} =
  Shifted(Vector{T}(undef, length(d[1])), first(d[1]) - 1)

# The same for a matrix, with the first axis starting at o1 + 1 and the second at o2 + 1.
struct ShiftedMatrix{T} <: AbstractMatrix{T}
  data::Matrix{T}
  o1::Int
  o2::Int
end
Base.size(s::ShiftedMatrix) = size(s.data)
Base.axes(s::ShiftedMatrix) = (Base.IdentityUnitRange(s.o1+1:s.o1+size(s.data, 1)),
                               Base.IdentityUnitRange(s.o2+1:s.o2+size(s.data, 2)))
Base.IndexStyle(::Type{<:ShiftedMatrix}) = IndexCartesian()
Base.@propagate_inbounds function Base.getindex(s::ShiftedMatrix, i::Int, j::Int)
  @boundscheck checkbounds(s, i, j)
  s.data[i - s.o1, j - s.o2]
end
Base.@propagate_inbounds function Base.setindex!(s::ShiftedMatrix, v, i::Int, j::Int)
  @boundscheck checkbounds(s, i, j)
  s.data[i - s.o1, j - s.o2] = v
end
Base.similar(s::ShiftedMatrix, ::Type{T}, d::Tuple{Base.IdentityUnitRange, Base.IdentityUnitRange}) where {T} =
  ShiftedMatrix(Matrix{T}(undef, length(d[1]), length(d[2])), first(d[1]) - 1, first(d[2]) - 1)

# The kinds of offset vector: a strided view with the axes 2:n+1 over a buffer whose entry 1 is a decoy, the type above,
# and an OffsetArray with the axes 0:n-1. A vector of ranges (a list of groups) is also built this way.
const KINDS = (:view, :shifted, :offsetarray)
const MATKINDS = (:view, :viewcols, :shifted, :offsetarray)
function mkvec(kind::Symbol, v::AbstractVector)
  S, n = eltype(v), length(v)
  pad = S <: Real ? S(-999) : first(v)
  kind === :view && return view(vcat(S[pad], v), Base.IdentityUnitRange(2:n+1))
  kind === :shifted && return Shifted(Vector{S}(v), 1)
  kind === :offsetarray && return OffsetVector(Vector{S}(v), 0:n-1)
  error("unknown kind $kind")
end
function mkmat(kind::Symbol, M::AbstractMatrix)
  T, (n, m) = eltype(M), size(M)
  kind === :view && return view(vcat(fill(T(-999), 1, m), M), Base.IdentityUnitRange(2:n+1), :)
  kind === :viewcols && return view(hcat(fill(T(-999), n, 1), M), :, Base.IdentityUnitRange(2:m+1))
  kind === :shifted && return ShiftedMatrix(Matrix{T}(M), 1, 0)
  kind === :offsetarray && return OffsetArray(Matrix{T}(M), 0:n-1, 1:m)
  error("unknown kind $kind")
end

# the storage behind an array, which a write through the array would change
raw(a::Array) = a
raw(a::SubArray) = parent(a)
raw(a::Shifted) = a.data
raw(a::ShiftedMatrix) = a.data
raw(a::OffsetArray) = parent(a)
raw(a::SparseIterate) = Array(a)
snap(a::AbstractArray) = copy(raw(a))
snap(S::SymmetricSparseIterate) = (Matrix(S), nactive(S))
snap(t::Tuple) = map(snap, t)
unchanged(a::AbstractArray, s) = size(raw(a)) == size(s) && all(map(===, raw(a), s))
unchanged(S::SymmetricSparseIterate, s) = all(map(===, Matrix(S), s[1])) && nactive(S) == s[2]
unchanged(t::Tuple, s::Tuple) = all(map(unchanged, t, s))

outcome(call) = try
  call()
  :returned
catch e
  typeof(e)
end
# The call throws the expected error, and each destination in outs holds the entries that it held before.
function check(label, call, outs...; expected = ArgumentError)
  snaps = map(snap, outs)
  @test (label, outcome(call)) == (label, expected)
  for k in eachindex(outs)
    @test (label, k, unchanged(outs[k], snaps[k])) == (label, k, true)
  end
  nothing
end
same(a::AbstractArray, b::AbstractArray) = size(a) == size(b) && all(map(===, a, b))
same(a::Tuple, b::Tuple) = length(a) == length(b) && all(map(same, a, b))
same(a::Number, b::Number) = a === b
same(a::SymmetricSparseIterate, b::SymmetricSparseIterate) = same(Matrix(a), Matrix(b))
same(a::SparseIterate, b::SparseIterate) = same(Array(a), Array(b))
same(a::SymmetricSparseIterate, b::AbstractArray) = same(Matrix(a), b)
same(a::AbstractArray, b::SymmetricSparseIterate) = same(a, Matrix(b))

####################################
#  Functions defined in this file
####################################

# A separable function of an extension, which runs the kernels of this package: Σᵢ wᵢ xᵢ² / 2.
struct WeightedSquare{W<:AbstractVector} <: SeparableFunction
  w::W
end
PB.value_entry(f::WeightedSquare, v, i) = f.w[i] * v^2 / 2
PB.prox_block(f::WeightedSquare, v, i, γ) = v / (1 + γ * f.w[i])
PB.iszeropreserving(::WeightedSquare) = true
PB.isconvex(::Type{WeightedSquare}) = true
PB.entry_parameters(f::WeightedSquare) = (:w => f.w,)

# A function of an extension with its own oracles, which record the calls that reach them.
struct Recorded <: PB.AbstractFunction
  log::Vector{Symbol}
end
(f::Recorded)(x::AbstractArray) = (push!(f.log, :value); zero(float(eltype(x))))
PB.isproximable(::Type{Recorded}) = true
PB.prox!(y, f::Recorded, x, γ::Real) = (push!(f.log, :prox); y)
PB.value_and_gradient!(g, f::Recorded, x) = (push!(f.log, :gradient); 0.0)

# A function of blocks, for the routes with tuples.
struct BlockF <: PB.AbstractFunction end
PB.isproximable(::Type{BlockF}) = true
(f::BlockF)(x::Tuple) = sum(b -> sum(abs2, b), x) / 2
function PB.prox!(y::Tuple, f::BlockF, x::Tuple, γ::Real)
  for k in eachindex(y)
    y[k] .= x[k] ./ (1 + γ)
  end
  y
end

####################################
#  Child process with the default bounds checking
####################################

# One-based indexing is checked before any read, so the default bounds checking (elided under @inbounds) gives the
# same ArgumentError as --check-bounds=yes, which the runner forces. The child reports every case that differs.
const CHILD = raw"""
using ProximalBase, LinearAlgebra
Base.JLOptions().check_bounds == 0 || (println(stderr, "the child runs with check_bounds = ", Base.JLOptions().check_bounds); exit(3))
struct Shifted{T} <: AbstractVector{T}
  data::Vector{T}
  o::Int
end
Base.size(s::Shifted) = size(s.data)
Base.axes(s::Shifted) = (Base.IdentityUnitRange(s.o+1:s.o+length(s.data)),)
Base.IndexStyle(::Type{<:Shifted}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(s::Shifted, i::Int)
  @boundscheck checkbounds(s, i)
  s.data[i - s.o]
end
Base.@propagate_inbounds function Base.setindex!(s::Shifted, v, i::Int)
  @boundscheck checkbounds(s, i)
  s.data[i - s.o] = v
end
mkvec(kind, v::Vector{T}) where {T} =
  kind === :view ? view(vcat(T[-999], v), Base.IdentityUnitRange(2:length(v)+1)) : Shifted(copy(v), 1)
raw(a::SubArray) = parent(a)
raw(a::Shifted) = a.data
raw(a::Array) = a
failures = String[]
function expect(label, call, outs...)
  saved = map(o -> copy(raw(o)), outs)
  r = try
    call()
    "returned"
  catch e
    e isa ArgumentError ? "ArgumentError" : string(typeof(e))
  end
  r == "ArgumentError" || push!(failures, "$label: $r")
  for (k, o) in enumerate(outs)
    all(map(===, raw(o), saved[k])) || push!(failures, "$label: destination $k changed")
  end
end
for T in (Float32, Float64), kind in (:view, :shifted)
  x, w, one_ = T[1, -2, 3, -4], T[1, 2, 3, 4], T[1, 1, 1, 1]
  off(v) = mkvec(kind, v)
  tag = "$T, $kind"
  expect("weighted NormL1 construction, $tag", () -> NormL1(one(T), off(w)))
  expect("weighted NormL1 value at an offset input, $tag", () -> NormL1(one(T), w)(off(x)))
  y = fill(T(7), 4)
  expect("prox! at an offset input, $tag", () -> prox!(y, NormL1(one(T), w), off(x), one(T)), y)
  expect("value_and_prox! at an offset input, $tag", () -> value_and_prox!(y, NormL1(one(T), w), off(x), one(T)), y)
  expect("prox! with an offset step array, $tag", () -> prox!(y, NormL1(one(T), w), x, off(one_)), y)
  expect("Scaled prox! with an offset step array, $tag", () -> prox!(y, Scaled(NormL1(one(T)), T(2)), x, off(one_)), y)
  expect("Scaled chain prox! with an offset step array, $tag",
         () -> prox!(y, Scaled(Scaled(NormL1(one(T), w), T(2)), T(3)), x, off(one_)), y)
  yo = off(fill(T(7), 4))
  expect("prox! into an offset output, $tag", () -> prox!(yo, NormL1(one(T), w), x, one(T)), yo)
  expect("least-squares response, $tag", () -> LeastSquaresLoss(off(w), Matrix{T}(I, 4, 4)))
  expect("least-squares design, $tag", () -> LeastSquaresLoss(w, view(vcat(zeros(T, 1, 4), Matrix{T}(I, 4, 4)), Base.IdentityUnitRange(2:5), :)))
end
if isempty(failures)
  println("all cases passed")
else
  foreach(println, failures)
  exit(1)
end
"""

@testset "default bounds checking in a child process" begin
  mktempdir() do dir
    script, out, err = joinpath(dir, "child.jl"), joinpath(dir, "out.txt"), joinpath(dir, "err.txt")
    write(script, CHILD)
    cmd = `$(Base.julia_cmd()) --check-bounds=auto --startup-file=no --project=$(dirname(Base.active_project())) $script`
    seconds = @elapsed ok = success(run(pipeline(ignorestatus(cmd), stdout = out, stderr = err)))
    ok || (print(stderr, read(out, String)); print(stderr, read(err, String)))
    @test ok
    @test occursin("all cases passed", read(out, String))
    @info "child process with the default bounds checking: $(round(seconds, digits = 1)) s"
  end
end

####################################
#  The oracles of every family
####################################

for T in (Float32, Float64)
@testset "$T" begin
  two, three = T(2), T(3)
  x, w, c = T[1, -2, 3, -4], T[1, 2, 3, 4], T[0.5, -1, 2, 1]
  γ1 = T[1, 1, 1, 1]
  A = T[2 1 0 0; 1 3 1 0; 0 1 4 1; 0 0 1 5]
  M4 = T[1 2 0 1; 0 3 1 0; 2 0 1 1; 1 1 0 2]

  @testset "construction with an offset array" begin
    for kind in KINDS
      off(v) = mkvec(kind, v)
      check("NormL1 weights, $kind", () -> NormL1(one(T), off(w)))
      check("SqrNormL2 center, $kind", () -> SqrNormL2(one(T), off(c)))
      check("LinearFunction coefficients, $kind", () -> LinearFunction(off(c)))
      check("LinearFunction coefficients and constant, $kind", () -> LinearFunction(off(c), one(T)))
      check("QuadraticFunction b, $kind", () -> QuadraticFunction(A, off(w)))
      check("LeastSquaresLoss response, $kind", () -> LeastSquaresLoss(off(w), M4))
      # the groups: the list, one group, and the weights
      groups = UnitRange{Int}[1:2, 3:4]
      check("GroupNormL2 list of groups, $kind", () -> GroupNormL2(one(T), mkvec(kind, groups)))
      check("GroupNormL2 list of groups and weights, $kind", () -> GroupNormL2(one(T), mkvec(kind, groups), T[1, 1]))
      check("GroupNormL2 one group, $kind", () -> GroupNormL2(one(T), [1:2, off([3, 4])]))
      check("GroupNormL2 weights, $kind", () -> GroupNormL2(one(T), groups, off(T[1, 1])))
      check("BlockIterate list of groups, $kind", () -> BlockIterate(4, mkvec(kind, groups)))
      check("BlockIterate type, list of groups, $kind", () -> BlockIterate(T, 4, mkvec(kind, groups)))
      check("BlockIterate one group, $kind", () -> BlockIterate(4, [1:2, off([3, 4])]))
      check("BlockIterate storage and list of groups, $kind", () -> BlockIterate(zeros(T, 4), mkvec(kind, groups)))
      b = BlockIterate(T, 4, groups)
      check("BlockIterate low-level constructor, $kind", () -> BlockIterate(b.storage, mkvec(kind, groups), b.group_of, nothing, zeros(Int, 2), zeros(Int, 2), 0))
      # restricting to an offset index set gives an offset parameter
      S = off([1, 2])
      check("restrict NormL1, $kind", () -> restrict(NormL1(one(T), w), S))
      check("restrict SqrNormL2, $kind", () -> restrict(SqrNormL2(one(T), c), S))
      check("restrict LinearFunction, $kind", () -> restrict(LinearFunction(c), S))
      check("restrict QuadraticFunction, $kind", () -> restrict(QuadraticFunction(A, w), S))
      check("restrict LeastSquaresLoss, $kind", () -> restrict(LeastSquaresLoss(w, M4), S))
    end
    # a group that is a range with offset axes (an array literal would convert it to a UnitRange)
    ranges = AbstractUnitRange{Int}[1:2, Base.IdentityUnitRange(3:4)]
    ranges2 = AbstractUnitRange{Int}[1:2, OffsetArrays.IdOffsetRange(1:2, 2)]
    for rs in (ranges, ranges2)
      check("GroupNormL2 group that is a range with offset axes, $(typeof(rs[2]))", () -> GroupNormL2(one(T), rs))
      check("BlockIterate group that is a range with offset axes, $(typeof(rs[2]))", () -> BlockIterate(4, rs))
    end
    for kind in MATKINDS
      offm(M) = mkmat(kind, M)
      check("LinearFunction matrix coefficients, $kind", () -> LinearFunction(offm(M4)))
      check("QuadraticFunction A, $kind", () -> QuadraticFunction(offm(A), w))
      check("QuadraticFunction A without b, $kind", () -> QuadraticFunction(offm(A)))
      check("LeastSquaresLoss design, $kind", () -> LeastSquaresLoss(w, offm(M4)))
      check("LeastSquaresLoss matrix response, $kind", () -> LeastSquaresLoss(offm(M4), M4))
    end
    # designs with no entries and offset axes
    for Xe in (view(ones(T, 5, 0), Base.IdentityUnitRange(2:5), :), view(ones(T, 0, 4), :, Base.IdentityUnitRange(2:3)),
               view(ones(T, 2, 2), Base.IdentityUnitRange(2:1), Base.IdentityUnitRange(2:1)), OffsetArray(zeros(T, 4, 0), 0:3, 1:0))
      @test isempty(Xe)
      check("LeastSquaresLoss with an empty offset design $(size(Xe))", () -> LeastSquaresLoss(zeros(T, size(Xe, 1)), Xe))
    end
  end

  # the functions with a scalar step, a step array, a gradient
  nl1w, nl1, sq, sqc = NormL1(one(T), w), NormL1(one(T)), SqrNormL2(two), SqrNormL2(two, c)
  lin, zf, ws = LinearFunction(c, one(T)), ZeroFunction(), WeightedSquare(w)
  chain(f) = Scaled(Scaled(f, two), three)
  prox_fs = (("NormL1 with weights", nl1w), ("NormL1", nl1), ("SqrNormL2", sq), ("SqrNormL2 with a center", sqc),
             ("LinearFunction", lin), ("ZeroFunction", zf), ("WeightedSquare", ws),
             ("Scaled NormL1", Scaled(nl1w, two)), ("Scaled chain", chain(nl1w)), ("Scaled SqrNormL2", Scaled(sqc, two)),
             ("Scaled ZeroFunction", Scaled(zf, two)), ("Scaled WeightedSquare", Scaled(ws, two)))
  array_step_fs = (("NormL1 with weights", nl1w), ("NormL1", nl1), ("ZeroFunction", zf), ("WeightedSquare", ws),
                   ("Scaled NormL1", Scaled(nl1w, two)), ("Scaled chain", chain(nl1w)), ("Scaled ZeroFunction", Scaled(zf, two)),
                   ("Scaled WeightedSquare", Scaled(ws, two)), ("Scaled chain of WeightedSquare", chain(ws)))
  gradient_fs = (("SqrNormL2", sq), ("SqrNormL2 with a center", sqc), ("LinearFunction", lin), ("ZeroFunction", zf),
                 ("Scaled SqrNormL2", Scaled(sqc, two)), ("Scaled chain", chain(sqc)), ("Scaled LinearFunction", Scaled(lin, two)))

  @testset "offset input, destination and step array: $kind" for kind in KINDS
    off(v) = mkvec(kind, v)
    for (name, f) in prox_fs
      tag = "$name, $kind"
      check("value at an offset input, $tag", () -> f(off(x)))
      y = fill(T(7), 4)
      check("prox! at an offset input, $tag", () -> prox!(y, f, off(x), one(T)), y)
      check("value_and_prox! at an offset input, $tag", () -> value_and_prox!(y, f, off(x), one(T)), y)
      yo = off(fill(T(7), 4))
      check("prox! into an offset output, $tag", () -> prox!(yo, f, x, one(T)), yo)
      check("value_and_prox! into an offset output, $tag", () -> value_and_prox!(yo, f, x, one(T)), yo)
      xo = off(x)
      check("prox! in place at an offset input, $tag", () -> prox!(xo, f, xo, one(T)), xo)
      check("prox at an offset input, $tag", () -> prox(f, off(x), one(T)))
    end
    for (name, f) in array_step_fs
      tag = "$name, $kind"
      y = fill(T(7), 4)
      check("prox! with an offset step array, $tag", () -> prox!(y, f, x, off(γ1)), y)
      check("value_and_prox! with an offset step array, $tag", () -> value_and_prox!(y, f, x, off(γ1)), y)
      check("prox with an offset step array, $tag", () -> prox(f, x, off(γ1)))
      check("prox! with an offset step array and an offset input, $tag", () -> prox!(y, f, off(x), off(γ1)), y)
      yo = off(fill(T(7), 4))
      check("prox! with a step array into an offset output, $tag", () -> prox!(yo, f, x, γ1), yo)
      check("value_and_prox! with a step array into an offset output, $tag", () -> value_and_prox!(yo, f, x, γ1), yo)
    end
    for (name, f) in gradient_fs
      tag = "$name, $kind"
      g, go = fill(T(7), 4), off(fill(T(7), 4))
      check("gradient! at an offset input, $tag", () -> gradient!(g, f, off(x)), g)
      check("gradient! into an offset output, $tag", () -> gradient!(go, f, x), go)
      check("value_and_gradient! at an offset input, $tag", () -> value_and_gradient!(g, f, off(x)), g)
      check("value_and_gradient! into an offset output, $tag", () -> value_and_gradient!(go, f, x), go)
      check("gradient at an offset input, $tag", () -> gradient(f, off(x)))
      check("value_and_gradient at an offset input, $tag", () -> value_and_gradient(f, off(x)))
    end
    # the screens: the gradient g and the point x
    for (name, f) in (("NormL1 with weights", nl1w), ("SqrNormL2", sq), ("SqrNormL2 with a center", sqc), ("LinearFunction", lin),
                      ("ZeroFunction", zf), ("Scaled NormL1", Scaled(nl1w, two)), ("Scaled chain", chain(nl1w)),
                      ("Scaled SqrNormL2", Scaled(sqc, two)), ("Scaled chain of SqrNormL2", chain(sqc)))
      tag = "$name, $kind"
      check("kkt_violation(f, g, i) at an offset gradient, $tag", () -> kkt_violation(f, off(x), 2))
      check("kkt_violation(f, g, x, i) at an offset gradient, $tag", () -> kkt_violation(f, off(x), x, 2))
      check("kkt_violation(f, g, x, i) at an offset point, $tag", () -> kkt_violation(f, x, off(x), 2))
    end
    # a custom function that is checked at the boundary of the public four-argument screen only
    check("kkt_violation of a custom separable function at an offset gradient, $kind", () -> kkt_violation(ws, off(x), x, 2))
    # sparse and block points, and a group norm
    sx = SparseIterate(T, 4); sx[2] = two; sx[4] = -three
    check("kkt_violation at a SparseIterate with an offset gradient, $kind", () -> kkt_violation(nl1w, off(x), sx, 2))
    gb = BlockIterate(T, 4, [1:2, 3:4]); gf = GroupNormL2(one(T), [1:2, 3:4])
    check("kkt_violation of GroupNormL2 at a BlockIterate with an offset gradient, $kind", () -> kkt_violation(gf, off(x), gb, 1))
    check("kkt_violation of GroupNormL2 at an array with an offset gradient, $kind", () -> kkt_violation(gf, off(x), x, 1))
    check("kkt_violation of GroupNormL2 at an offset array, $kind", () -> kkt_violation(gf, x, off(x), 1))
    sS = SymmetricSparseIterate(T, 2); sS[1, 1] = two
    check("kkt_violation at a SymmetricSparseIterate with an offset gradient, $kind",
          () -> kkt_violation(NormL1(one(T), T[1, 2, 2, 3]), off(x), sS, 1))
  end

  @testset "extensions: $kind" for kind in KINDS
    off(v) = mkvec(kind, v)
    # a function of an extension keeps its own oracles: the generic fallbacks forward to them whatever the array
    log = Symbol[]; r = Recorded(log)
    y, g = fill(T(7), 4), fill(T(7), 4)
    @test value_and_prox!(y, r, off(x), 1.0) === zero(T)
    @test log == [:prox, :value]
    @test gradient!(g, r, off(x)) === g
    @test last(log) === :gradient
    # a Scaled function whatever it wraps, and a separable function that runs the kernels of this package, are checked
    empty!(log)
    check("Scaled custom function at an offset input, $kind", () -> value_and_prox!(y, Scaled(r, two), off(x), 1.0), y)
    check("Scaled custom function with an offset step array, $kind", () -> prox!(y, Scaled(r, two), x, off(γ1)), y)
    check("Scaled custom function, gradient, $kind", () -> gradient!(g, Scaled(r, two), off(x)), g)
    check("Scaled custom function, value, $kind", () -> Scaled(r, two)(off(x)))
    @test isempty(log)
    check("custom separable function, value, $kind", () -> ws(off(x)))
    check("custom separable function, prox!, $kind", () -> prox!(y, ws, off(x), 1.0), y)
    # the allocating wrappers check x before they allocate for it, for every function
    check("prox of a custom function at an offset input, $kind", () -> prox(r, off(x), 1.0))
    check("gradient of a custom function at an offset input, $kind", () -> gradient(r, off(x)))
    check("value_and_gradient of a custom function at an offset input, $kind", () -> value_and_gradient(r, off(x)))
    @test isempty(log)
    # the parameters that a custom function declares for a symmetric iterate
    S2 = SymmetricSparseIterate(T, 2)
    check("check_symmetric_parameters with an offset declared array, $kind", () -> PB.check_symmetric_parameters(WeightedSquare(off(T[1, 2, 2, 3])), S2))
    check("check_symmetric_parameters with an offset declared array under Scaled, $kind",
          () -> PB.check_symmetric_parameters(Scaled(WeightedSquare(off(T[1, 2, 2, 3])), two), S2))
    @test PB.check_symmetric_parameters(WeightedSquare(T[1, 2, 2, 3]), S2)
    # blocks of a tuple, nested tuples included
    b1, b2 = T[1, 2], T[3, 4, 5]
    yb = (fill(T(7), 2), fill(T(7), 3))
    s = Scaled(BlockF(), two)
    check("Scaled function of blocks at an offset block, $kind", () -> s((b1, off(b2))))
    check("Scaled function of blocks at an offset nested block, $kind", () -> s((b1, (b2, off(b1)))))
    check("Scaled function of blocks, prox! into an offset block, $kind",
          () -> prox!((yb[1], off(fill(T(7), 3))), s, (b1, b2), one(T)), yb...)
    check("allocating prox of blocks at an offset block, $kind", () -> prox(BlockF(), (b1, off(b2)), one(T)))
  end

  @testset "QuadraticFunction: $kind" for kind in KINDS
    off(v) = mkvec(kind, v)
    for (name, q) in (("dense A", QuadraticFunction(A, w)), ("dense A without b", QuadraticFunction(A)),
                      ("Diagonal A", QuadraticFunction(Diagonal(w), w)), ("sparse A", QuadraticFunction(sparse(A), w)))
      tag = "$name, $kind"
      fill!(q.tmp, T(7))
      g, go, y, yo = fill(T(7), 4), off(fill(T(7), 4)), fill(T(7), 4), off(fill(T(7), 4))
      check("value, $tag", () -> q(off(x)), q.tmp)
      check("gradient! at an offset input, $tag", () -> gradient!(g, q, off(x)), g, q.tmp)
      check("gradient! into an offset output, $tag", () -> gradient!(go, q, x), go, q.tmp)
      check("value_and_gradient! at an offset input, $tag", () -> value_and_gradient!(g, q, off(x)), g, q.tmp)
      check("value_and_gradient! into an offset output, $tag", () -> value_and_gradient!(go, q, x), go, q.tmp)
      check("prox! at an offset input, $tag", () -> prox!(y, q, off(x), one(T)), y, q.tmp)
      check("prox! into an offset output, $tag", () -> prox!(yo, q, x, one(T)), yo, q.tmp)
      check("value_and_prox! at an offset input, $tag", () -> value_and_prox!(y, q, off(x), one(T)), y, q.tmp)
      check("value_and_prox! into an offset output, $tag", () -> value_and_prox!(yo, q, x, one(T)), yo, q.tmp)
      check("gradient at an offset input, $tag", () -> gradient(q, off(x)), q.tmp)
      check("prox at an offset input, $tag", () -> prox(q, off(x), one(T)), q.tmp)
      @test isempty(q.Λ) && isempty(q.Q)
    end
  end

  @testset "LeastSquaresLoss: $kind" for kind in KINDS
    off(v) = mkvec(kind, v)
    Xsq, Ysq = M4 + 4I, T[1, 2, 3, 4]
    Xrect, Yrect = T[1 2 0 1 3; 0 1 1 2 0; 2 0 1 1 1], T[1, 2, 3]
    for (name, X, Y) in (("square", Xsq, Ysq), ("p > n", Xrect, Yrect))
      tag = "$name, $kind"
      p = size(X, 2)
      f = LeastSquaresLoss(Y, X)
      β, v = T.(1:p) ./ 4, T.(p:-1:1) ./ 3
      fill!(f.tmp, T(7))
      g, go, out, outo = fill(T(7), p), off(fill(T(7), p)), fill(T(7), p), off(fill(T(7), p))
      check("value, $tag", () -> f(off(β)), f.tmp)
      check("gradient! at an offset input, $tag", () -> gradient!(g, f, off(β)), g, f.tmp)
      check("gradient! into an offset output, $tag", () -> gradient!(go, f, β), go, f.tmp)
      check("value_and_gradient! at an offset input, $tag", () -> value_and_gradient!(g, f, off(β)), g, f.tmp)
      check("value_and_gradient! into an offset output, $tag", () -> value_and_gradient!(go, f, β), go, f.tmp)
      check("prox! at an offset input, $tag", () -> prox!(out, f, off(v), one(T)), out, f.tmp)
      check("prox! into an offset output, $tag", () -> prox!(outo, f, v, one(T)), outo, f.tmp)
      check("value_and_prox! at an offset input, $tag", () -> value_and_prox!(out, f, off(v), one(T)), out, f.tmp)
      check("value_and_prox! into an offset output, $tag", () -> value_and_prox!(outo, f, v, one(T)), outo, f.tmp)
      check("prox! in place at an offset input, $tag", () -> (vo = off(v); prox!(vo, f, vo, one(T))), f.tmp)
      s = Scaled(f, two)
      check("Scaled value, $tag", () -> s(off(β)), f.tmp)
      check("Scaled gradient!, $tag", () -> gradient!(g, s, off(β)), g, f.tmp)
      check("Scaled prox!, $tag", () -> prox!(outo, s, v, one(T)), outo, f.tmp)
      @test !f.svd_ready
      @test isempty(f.V) && isempty(f.s2)
    end
  end

  @testset "LeastSquaresLoss with a matrix response: $kind" for kind in MATKINDS
    offm(M) = mkmat(kind, M)
    Y, X = T[1 2; 3 4; 5 6; 7 8], M4 + 4I
    B = T[1 2; 3 4; 5 6; 7 8] ./ 4
    f = LeastSquaresLoss(Y, X)
    fill!(f.tmp, T(7))
    G, Go, O, Oo = fill(T(7), 4, 2), offm(fill(T(7), 4, 2)), fill(T(7), 4, 2), offm(fill(T(7), 4, 2))
    check("value, $kind", () -> f(offm(B)), f.tmp)
    check("gradient! at an offset input, $kind", () -> gradient!(G, f, offm(B)), G, f.tmp)
    check("gradient! into an offset output, $kind", () -> gradient!(Go, f, B), Go, f.tmp)
    check("prox! at an offset input, $kind", () -> prox!(O, f, offm(B), one(T)), O, f.tmp)
    check("prox! into an offset output, $kind", () -> prox!(Oo, f, B, one(T)), Oo, f.tmp)
    @test !f.svd_ready
  end

  @testset "the routes that keep their DimensionMismatch" begin
    for kind in KINDS
      off(v) = mkvec(kind, v)
      # an iterate or a matrix built from offset storage
      @test_throws DimensionMismatch BlockIterate(off(x), [1:2, 3:4])
      @test_throws DimensionMismatch BlockIterate(off(x), 2)
      # the predictor and the residual oracles at a one-based response: an offset z or r is a DimensionMismatch, an offset
      # output g or d an ArgumentError, and the outputs are as they were
    end
    for kind in MATKINDS
      @test_throws DimensionMismatch SymmetricSparseIterate(mkmat(kind, A))
      S = SymmetricSparseIterate(T, 4); S[1, 1] = two
      before = snap(S)
      @test_throws DimensionMismatch copyto!(S, mkmat(kind, A))
      @test unchanged(S, before)
    end
    f = LeastSquaresLoss(T[1, 2, 3, 4], Matrix{T}(I, 4, 4))
    buf = T[100, 2, 4, 6, 8, 20]
    zoff = view(buf, Base.IdentityUnitRange(2:5))
    for g in (f, Scaled(f, two))
      gd, dd = fill(T(7), 4), fill(T(7), 4)
      @test_throws DimensionMismatch PB.value_from_predictor(g, zoff)
      @test_throws DimensionMismatch PB.value_from_residual(g, zoff)
      @test_throws DimensionMismatch PB.value_and_gradient_from_predictor!(gd, dd, g, zoff)
      @test all(==(T(7)), gd) && all(==(T(7)), dd)
      # an offset g is an ArgumentError; an offset d has other axes than Y, a DimensionMismatch
      goff = view(fill(T(7), 6), Base.IdentityUnitRange(2:5))
      @test_throws ArgumentError PB.value_and_gradient_from_predictor!(goff, dd, g, T[2, 4, 6, 8])
      @test all(==(T(7)), goff) && all(==(T(7)), dd)
      @test_throws DimensionMismatch PB.value_and_gradient_from_predictor!(gd, goff, g, T[2, 4, 6, 8])
      @test all(==(T(7)), gd) && all(==(T(7)), goff)
    end
  end

  @testset "norm_diff: $kind" for kind in KINDS
    off(v) = mkvec(kind, v)
    for p in (2, Inf)
      check("norm_diff of two offset vectors, p = $p, $kind", () -> norm_diff(off(x), off(c), p))
      check("norm_diff, offset first argument, p = $p, $kind", () -> norm_diff(off(x), c, p))
      check("norm_diff, offset second argument, p = $p, $kind", () -> norm_diff(x, off(c), p))
    end
  end
  @testset "norm_diff of matrices: $kind" for kind in MATKINDS
    @test_throws ArgumentError norm_diff(mkmat(kind, M4), M4)
    @test_throws ArgumentError norm_diff(M4, mkmat(kind, M4), Inf)
    @test_throws ArgumentError norm_diff(mkmat(kind, M4), mkmat(kind, M4))
  end

  @testset "SymmetricSparseIterate: $kind" for kind in MATKINDS
    offm(M) = mkmat(kind, M)
    Msym = T[2 1 0; 1 3 1; 0 1 4]
    for (name, f) in (("NormL1", NormL1(one(T))), ("NormL1 with weights", NormL1(one(T), ones(T, 9))), ("SqrNormL2", SqrNormL2(two)),
                      ("LinearFunction", LinearFunction(Msym)), ("ZeroFunction", ZeroFunction()), ("Scaled NormL1", Scaled(NormL1(one(T)), two)),
                      ("WeightedSquare", WeightedSquare(ones(T, 9))))
      S = SymmetricSparseIterate(T, 3); S[1, 1] = T(5); S[3, 2] = T(-1)
      check("prox! into a SymmetricSparseIterate at an offset matrix, $name, $kind", () -> prox!(S, f, offm(Msym), one(T)), S)
      check("value_and_prox! into a SymmetricSparseIterate at an offset matrix, $name, $kind", () -> value_and_prox!(S, f, offm(Msym), one(T)), S)
    end
    for (name, f) in (("SqrNormL2", SqrNormL2(two)), ("LinearFunction", LinearFunction(Msym)), ("Scaled SqrNormL2", Scaled(SqrNormL2(two), two)))
      S = SymmetricSparseIterate(T, 3); S[1, 1] = T(5)
      check("gradient! into a SymmetricSparseIterate at an offset matrix, $name, $kind", () -> gradient!(S, f, offm(Msym)), S)
      check("value_and_gradient! into a SymmetricSparseIterate at an offset matrix, $name, $kind", () -> value_and_gradient!(S, f, offm(Msym)), S)
    end
    # a symmetric output with an offset step array is a refusal of the step; a dense output at an offset matrix is checked
    G = fill(T(7), 3, 3)
    check("gradient! at an offset matrix, $kind", () -> gradient!(G, SqrNormL2(two), offm(Msym)), G)
    Go = offm(fill(T(7), 3, 3))
    check("gradient! into an offset matrix, $kind", () -> gradient!(Go, SqrNormL2(two), Msym), Go)
  end

  @testset "products of the iterates with an offset array" begin
    # the methods for strided arguments take the strided offset views; the methods for any vector take every kind
    sx = SparseIterate(T, 4); sx[2] = two; sx[4] = -three
    sX = SparseIterate(T, 4, 3); sX[1, 2] = two; sX[3, 1] = -three; sX[4, 3] = one(T)
    bx = BlockIterate(T[0, 2, 0, -3], [1:2, 3:4])
    Sx = SymmetricSparseIterate(T[2 1 0 0; 1 0 0 0; 0 0 0 0; 0 0 0 3])
    A43 = T[1 2 0 1; 0 3 1 0; 2 0 1 1]
    rowsoff(M) = mkmat(:view, M)
    colsoff(M) = mkmat(:viewcols, M)
    for (name, offA) in (("rows", rowsoff), ("columns", colsoff))
      out = fill(T(7), 3)
      check("mul! at an offset matrix, SparseIterate vector, $name", () -> mul!(out, offA(A43), sx), out)
      check("mul! at an offset matrix, SparseIterate vector, five arguments, $name", () -> mul!(out, offA(A43), sx, one(T), one(T)), out)
      Out = fill(T(7), 3, 3)
      check("mul! at an offset matrix, SparseIterate matrix, $name", () -> mul!(Out, offA(A43), sX), Out)
      out4 = fill(T(7), 4)
      check("mul! at an offset transposed matrix, SparseIterate vector, $name", () -> mul!(out4, transpose(offA(A43)), SparseIterate(T[1, 0, 2])), out4)
      check("mul! at an offset matrix, BlockIterate, $name", () -> mul!(out, offA(A43), bx), out)
      check("dot at an offset matrix, SymmetricSparseIterate, $name", () -> dot(offA(M4), Sx))
      check("dot at an offset matrix, SymmetricSparseIterate on the left, $name", () -> dot(Sx, offA(M4)))
      check("A * x at an offset matrix, SparseIterate, $name", () -> offA(A43) * sx)
      check("A * x at an offset matrix, BlockIterate, $name", () -> offA(A43) * bx)
      check("dot(a, X, b) at an offset vector, $name", () -> dot(mkvec(:view, x), sX, T[1, 2, 3]))
    end
    # offset output and offset vector, with a product that is dispatched on any vector
    for kind in KINDS
      off(v) = mkvec(kind, v)
      outo = off(fill(T(7), 3))
      check("mul! into an offset output, BlockIterate, $kind", () -> mul!(outo, A43, bx), outo)
      check("mul! into an offset output, BlockIterate, five arguments, $kind", () -> mul!(outo, A43, bx, one(T), zero(T)), outo)
      Cv, Bv = fill(T(7), 4), off(T[1, 2, 3, 4])
      Cvo = off(fill(T(7), 4))
      check("mul!, SymmetricSparseIterate times an offset vector, $kind", () -> mul!(Cv, Sx, Bv), Cv)
      check("mul!, SymmetricSparseIterate into an offset vector, $kind", () -> mul!(Cvo, Sx, T[1, 2, 3, 4]), Cvo)
      check("mul!, SymmetricSparseIterate times an offset vector, five arguments, $kind", () -> mul!(Cv, Sx, Bv, two, one(T)), Cv)
      check("mul!, SymmetricSparseIterate into an offset vector, five arguments, $kind", () -> mul!(Cvo, Sx, T[1, 2, 3, 4], two, one(T)), Cvo)
      check("Sx * offset vector, $kind", () -> Sx * Bv)
      check("dot(a, X, b) with an offset vector, SymmetricSparseIterate, $kind", () -> dot(Bv, Sx, T[1, 2, 3, 4]))
    end
    for kind in MATKINDS
      offm(M) = mkmat(kind, M)
      C, Co, Bm = fill(T(7), 4, 2), offm(fill(T(7), 4, 2)), T[1 2; 3 4; 5 6; 7 8]
      check("mul!, SymmetricSparseIterate times an offset matrix, $kind", () -> mul!(C, Sx, offm(Bm)), C)
      check("mul!, SymmetricSparseIterate into an offset matrix, $kind", () -> mul!(Co, Sx, Bm), Co)
      check("mul!, SymmetricSparseIterate times an offset matrix, five arguments, $kind", () -> mul!(C, Sx, offm(Bm), two, one(T)), C)
      check("mul!, SymmetricSparseIterate into an offset matrix, five arguments, $kind", () -> mul!(Co, Sx, Bm, two, one(T)), Co)
      check("Sx * offset matrix, $kind", () -> Sx * offm(Bm))
    end
    # the strided offset views: matrix products of a symmetric iterate with an offset strided matrix
    Cr = fill(T(7), 4, 2)
    check("mul!, SymmetricSparseIterate times an offset strided matrix", () -> mul!(Cr, Sx, view(vcat(zeros(T, 1, 2), T[1 2; 3 4; 5 6; 7 8]), Base.IdentityUnitRange(2:5), :)), Cr)
  end

  @testset "ordinary one-based arrays give the results of plain arrays" begin
    # a view of a longer array that starts at an index other than 1 is one-based, as are strided views of every kind
    bigvec(v) = view(vcat(T[-999, 77], v, T[88]), 3:length(v)+2)
    bigmat(M) = view(vcat(fill(T(-999), 2, size(M, 2) + 1), hcat(fill(T(-999), size(M, 1)), M)), 3:size(M, 1)+2, 2:size(M, 2)+1)
    for v in (x, w, c)
      @test bigvec(v) == v && axes(bigvec(v)) == axes(v)
    end
    @test bigmat(A) == A
    xv, wv, cv, γv = bigvec(x), bigvec(w), bigvec(c), bigvec(γ1)
    sample = (nl1 = NormL1(one(T)), nl1w = NormL1(one(T), w), sq = SqrNormL2(two), sqc = SqrNormL2(two, c), lin = LinearFunction(c, one(T)),
              zf = ZeroFunction(), ws = WeightedSquare(w))
    viewed = (nl1 = NormL1(one(T)), nl1w = NormL1(one(T), wv), sq = SqrNormL2(two), sqc = SqrNormL2(two, cv), lin = LinearFunction(cv, one(T)),
              zf = ZeroFunction(), ws = WeightedSquare(wv))
    for k in keys(sample)
      f, fv = sample[k], viewed[k]
      @test same(fv(xv), f(x))
      @test same(fv(x), f(x)) && same(fv(xv), f(xv))
      y, yv = fill(T(7), 4), view(fill(T(7), 6), 2:5)
      @test same(prox!(yv, fv, xv, two), prox!(y, f, x, two))
      @test same(value_and_prox!(yv, fv, xv, two), value_and_prox!(y, f, x, two)) && same(yv, y)
      @test same(prox(fv, xv, two), prox(f, x, two))
      Sf = Scaled(f, three)
      Sfv = Scaled(fv, three)
      @test same(Sfv(xv), Sf(x))
      @test same(prox!(yv, Sfv, xv, two), prox!(y, Sf, x, two))
      @test same(value_and_prox!(yv, Sfv, xv, two), value_and_prox!(y, Sf, x, two)) && same(yv, y)
      @test same(prox!(yv, Scaled(Sfv, two), xv, two), prox!(y, Scaled(Sf, two), x, two))
    end
    for k in (:nl1, :nl1w, :zf, :ws)
      f, fv = sample[k], viewed[k]
      y, yv = fill(T(7), 4), view(fill(T(7), 6), 2:5)
      γ = T[0.5, 1, 2, 3]; γvw = view(vcat(T[0], γ, T[0]), 2:5)
      @test same(prox!(yv, fv, xv, γvw), prox!(y, f, x, γ))
      @test same(value_and_prox!(yv, fv, xv, γvw), value_and_prox!(y, f, x, γ)) && same(yv, y)
      @test same(prox!(yv, Scaled(fv, two), xv, γvw), prox!(y, Scaled(f, two), x, γ))
      @test same(value_and_prox!(yv, Scaled(Scaled(fv, two), three), xv, γvw), value_and_prox!(y, Scaled(Scaled(f, two), three), x, γ))
      @test same(prox(fv, xv, γvw), prox(f, x, γ))
    end
    for k in (:sq, :sqc, :lin, :zf)
      f, fv = sample[k], viewed[k]
      g, gv = fill(T(7), 4), view(fill(T(7), 6), 2:5)
      @test same(gradient!(gv, fv, xv), gradient!(g, f, x)) && same(gradient(fv, xv), gradient(f, x))
      @test same(value_and_gradient!(gv, fv, xv), value_and_gradient!(g, f, x)) && same(gv, g)
      @test same(value_and_gradient!(gv, Scaled(fv, two), xv), value_and_gradient!(g, Scaled(f, two), x))
      @test same(kkt_violation(fv, xv, 2), kkt_violation(f, x, 2)) && same(kkt_violation(fv, xv, xv, 2), kkt_violation(f, x, x, 2))
    end
    @test same(kkt_violation(viewed.nl1w, xv, 3), kkt_violation(sample.nl1w, x, 3))
    @test same(kkt_violation(Scaled(viewed.nl1w, two), xv, 3), kkt_violation(Scaled(sample.nl1w, two), x, 3))
    @test same(norm_diff(xv, cv), norm_diff(x, c)) && same(norm_diff(xv, cv, Inf), norm_diff(x, c, Inf))
    @test same(norm_diff(bigmat(A), bigmat(M4)), norm_diff(A, M4))
    # sparse, diagonal and symmetric arguments
    for (name, Am) in (("dense", A), ("sparse", sparse(A)), ("Diagonal", Diagonal(w)), ("Symmetric", Symmetric(A)))
      q = QuadraticFunction(Am, w)
      qv = QuadraticFunction(Am, wv)
      @test same(qv(xv), q(x))
      g, gv = fill(T(7), 4), view(fill(T(7), 6), 2:5)
      @test same(gradient!(gv, qv, xv), gradient!(g, q, x))
      @test same(value_and_gradient!(gv, qv, xv), value_and_gradient!(g, q, x)) && same(gv, g)
      @test same(prox!(gv, qv, xv, one(T)), prox!(g, q, x, one(T)))
    end
    Dq = QuadraticFunction(Diagonal(wv), wv)
    Dp = QuadraticFunction(Diagonal(w), w)
    @test same(Dq(xv), Dp(x))
    let Xm = bigmat(A + 4I), Ym = bigvec(w), βv = bigvec(c)
      f, fv = LeastSquaresLoss(w, A + 4I), LeastSquaresLoss(Ym, Xm)
      g, gv = fill(T(7), 4), view(fill(T(7), 6), 2:5)
      @test same(fv(βv), f(c))
      @test same(gradient!(gv, fv, βv), gradient!(g, f, c))
      @test same(value_and_gradient!(gv, fv, βv), value_and_gradient!(g, f, c)) && same(gv, g)
      @test same(prox!(gv, fv, βv, two), prox!(g, f, c, two))
      Xr = T[1 2 0 1 3; 0 1 1 2 0; 2 0 1 1 1]
      fr, frv = LeastSquaresLoss(T[1, 2, 3], Xr), LeastSquaresLoss(bigvec(T[1, 2, 3]), bigmat(Xr))
      β5, β5v = T[1, 2, 3, 4, 5] ./ 3, bigvec(T[1, 2, 3, 4, 5] ./ 3)
      o5, o5v = fill(T(7), 5), view(fill(T(7), 7), 2:6)
      @test same(frv(β5v), fr(β5))
      @test same(prox!(o5v, frv, β5v, two), prox!(o5, fr, β5, two))
      @test same(PB.value_from_predictor(fv, bigvec(T[2, 4, 6, 8])), PB.value_from_predictor(f, T[2, 4, 6, 8]))
    end
    # the iterates
    sx = SparseIterate(T, 4); sx[2] = two; sx[4] = -three
    sX = SparseIterate(T, 4, 3); sX[1, 2] = two; sX[3, 1] = -three; sX[4, 3] = one(T)
    bx = BlockIterate(T[0, 2, 0, -3], [1:2, 3:4])
    Sx = SymmetricSparseIterate(T[2 1 0 0; 1 0 0 0; 0 0 0 0; 0 0 0 3])
    A43 = T[1 2 0 1; 0 3 1 0; 2 0 1 1]
    A43v = bigmat(A43)
    for it in (sx, bx)
      out, outv = fill(T(7), 3), view(fill(T(7), 5), 2:4)
      @test same(mul!(outv, A43v, it), mul!(out, A43, it)) && same(mul!(outv, A43v, it, two, one(T)), mul!(out, A43, it, two, one(T)))
      @test same(A43v * it, A43 * it) && same(dot(xv, it), dot(x, it))
      @test same(mul!(view(fill(T(7), 6), 2:5), Symmetric(bigmat(A)), it), mul!(fill(T(7), 4), Symmetric(A), it))
      @test same(mul!(view(fill(T(7), 6), 2:5), Symmetric(A), it), mul!(fill(T(7), 4), Symmetric(A), it))
    end
    At = T[1 2 0; 0 3 1; 2 0 1; 1 0 1]
    sv = SparseIterate(T[1, 0, 2, 1])
    @test same(mul!(view(fill(T(7), 5), 2:4), transpose(bigmat(At)), sv), mul!(fill(T(7), 3), transpose(At), sv))
    @test same(mul!(view(fill(T(7), 5, 4), 2:4, 2:4), A43v, sX), mul!(fill(T(7), 3, 3), A43, sX))
    @test same(dot(bigvec(T[1, 2, 3, 4]), sX, bigvec(T[1, 2, 3])), dot(T[1, 2, 3, 4], sX, T[1, 2, 3]))
    B4, B42 = T[1, 2, 3, 4], T[1 2; 3 4; 5 6; 7 8]
    @test same(mul!(view(fill(T(7), 7), 2:5), Sx, bigvec(B4)), mul!(fill(T(7), 4), Sx, B4))
    @test same(mul!(view(fill(T(7), 7), 2:5), Sx, bigvec(B4), two, one(T)), mul!(fill(T(7), 4), Sx, B4, two, one(T)))
    @test same(mul!(view(fill(T(7), 6, 3), 2:5, 1:2), Sx, bigmat(B42)), mul!(fill(T(7), 4, 2), Sx, B42))
    @test same(dot(bigmat(M4), Sx), dot(M4, Sx)) && same(dot(Sx, bigmat(M4)), dot(Sx, M4))
    @test same(Sx * bigvec(B4), Sx * B4)
    # the proximal operator into a symmetric iterate, at a plain, a view and a Symmetric matrix
    Msym = T[2 1 0; 1 3 1; 0 1 4]
    for f in (NormL1(one(T)), SqrNormL2(two), ZeroFunction(), Scaled(NormL1(one(T)), two), LinearFunction(Msym))
      for xa in (Msym, bigmat(Msym), Symmetric(Msym))
        S1, S2 = SymmetricSparseIterate(T, 3), SymmetricSparseIterate(T, 3)
        @test same(prox!(S1, f, xa, one(T)), prox!(S2, f, Msym, one(T)))
      end
    end
    # blocks of a tuple, and the group norm
    s = Scaled(BlockF(), two)
    b1, b2 = T[1, 2], T[3, 4, 5]
    yb, yr = (fill(T(7), 2), fill(T(7), 3)), (fill(T(7), 2), fill(T(7), 3))
    @test same(prox!(yb, s, (bigvec(b1), bigvec(b2)), one(T)), prox!(yr, s, (b1, b2), one(T)))
    @test same(s((bigvec(b1), bigvec(b2))), s((b1, b2)))
    @test same(prox(BlockF(), (bigvec(b1), b2), one(T)), prox(BlockF(), (b1, b2), one(T)))
    gf = GroupNormL2(one(T), [1:2, 3:4], T[1, 2])
    gb = BlockIterate(T[0, 2, 0, -3], [1:2, 3:4])
    gy, gyv = fill(T(7), 4), view(fill(T(7), 6), 2:5)
    @test same(gf(xv), gf(x)) && same(prox!(gyv, gf, xv, one(T)), prox!(gy, gf, x, one(T)))
    @test same(kkt_violation(gf, xv, gb, 1), kkt_violation(gf, x, gb, 1)) && same(kkt_violation(gf, xv, xv, 1), kkt_violation(gf, x, x, 1))
    @test same(gf(gb), gf(Vector(gb)))
    # a custom function with its own oracles is called as before
    log = Symbol[]
    @test value_and_prox!(fill(T(7), 4), Recorded(log), x, 1.0) === zero(T) && log == [:prox, :value]
  end
end
end

end # module
