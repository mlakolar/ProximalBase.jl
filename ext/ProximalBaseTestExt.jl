module ProximalBaseTestExt

# The checks of ProximalBase.Testing, with Test's @test

using ProximalBase
using LinearAlgebra
using SparseArrays
using Test
import ProximalBase.Testing: test_prox, test_gradient, test_separable, test_conjugate, test_kkt, test_traits,
                             test_iterate, allocations, brute_force_prox

# random points from the global generator, or from rng
_randn(::Nothing, dims...) = randn(dims...)
_randn(rng, dims...) = randn(rng, dims...)

_dense(x::AbstractArray) = Array(x)
_prox_objective(f, z, x, γ) = f(z) + norm_diff(z, x)^2 / (2γ)

####################################
#  allocations and the scalar reference
####################################

# one method per number of arguments, typed so that they specialize: Julia 1.10
# does not specialize a method that only passes its varargs on, and the dynamic
# dispatch would show as allocations. Warm the measured entry too: a plain
# call can inline and leave its out-of-line foreign symbols unresolved.
allocations(f::F, a::A) where {F, A} = (@allocated f(a); @allocated f(a))
allocations(f::F, a::A, b::B) where {F, A, B} = (@allocated f(a, b); @allocated f(a, b))
allocations(f::F, a::A, b::B, c::C) where {F, A, B, C} = (@allocated f(a, b, c); @allocated f(a, b, c))
allocations(f::F, a::A, b::B, c::C, d::D) where {F, A, B, C, D} = (@allocated f(a, b, c, d); @allocated f(a, b, c, d))
allocations(f::F, a::A, b::B, c::C, d::D, e::E) where {F, A, B, C, D, E} =
  (@allocated f(a, b, c, d, e); @allocated f(a, b, c, d, e))
allocations(f::F, a::A, b::B, c::C, d::D, e::E, g::G) where {F, A, B, C, D, E, G} =
  (@allocated f(a, b, c, d, e, g); @allocated f(a, b, c, d, e, g))

function brute_force_prox(fi, v::Real, γ::Real; lo = v - 5abs(v) - 5, hi = v + 5abs(v) + 5, n = 200_001)
  h(t) = fi(t) + (t - v)^2 / (2γ)
  ts = range(lo, hi; length = n)
  vals = h.(ts)
  i = argmin(vals)
  best, hb = ts[i], vals[i]
  for c in (0.0, v)                # the kinks and the identity
    hc = h(c)
    hc < hb && ((best, hb) = (c, hc))
  end
  # golden-section refinement in the neighboring cells
  a, b = ts[max(i - 1, 1)], ts[min(i + 1, n)]
  φ = (sqrt(5) - 1) / 2
  for _ in 1:100
    c, d = b - φ * (b - a), a + φ * (b - a)
    h(c) < h(d) ? (b = d) : (a = c)
  end
  t = (a + b) / 2
  h(t) < hb ? (t, h(t)) : (best, hb)
end

####################################
#  traits
####################################

function test_traits(f; smooth = false, proximable = false, separable = false, convex = false)
  @test issmooth(f) == smooth
  @test isproximable(f) == proximable
  @test isseparable(f) == separable
  @test isconvex(f) == convex
  @test issmooth(typeof(f)) == smooth
  @test isproximable(typeof(f)) == proximable
  @test isseparable(typeof(f)) == separable
  @test isconvex(typeof(f)) == convex
end

####################################
#  iterate bookkeeping
####################################

function test_iterate(x::BlockIterate)
  ok = length(x.group_of) == length(x.storage)
  @test length(x.group_of) == length(x.storage)
  ok || return x
  ok = length(x.position) == length(x.groups) == length(x.active)
  @test length(x.position) == length(x.groups) == length(x.active)
  ok || return x
  ok = 0 <= x.nactive <= length(x.groups)
  @test 0 <= x.nactive <= length(x.groups)
  ok || return x
  bad = [(k, i) for (k, G) in enumerate(x.groups) for i in G
         if !(1 <= i <= length(x.storage))]
  @test isempty(bad)
  isempty(bad) || return x
  memberships = zeros(Int, length(x.storage))
  for G in x.groups, i in G
    memberships[i] += 1
  end
  bad = (missing = findall(iszero, memberships), repeated = findall(>(1), memberships))
  @test bad == (missing = Int[], repeated = Int[])
  isempty(bad.missing) && isempty(bad.repeated) || return x
  bad = [(k, i) for (k, G) in enumerate(x.groups) for i in G if x.group_of[i] != k]
  @test isempty(bad)
  active = falses(length(x.groups))
  duplicates, out_of_range = Int[], Int[]
  for a in 1:x.nactive
    k = x.active[a]
    if 1 <= k <= length(x.groups)
      active[k] && push!(duplicates, a)
      active[k] = true
    else
      push!(out_of_range, a)
    end
  end
  @test isempty(duplicates)
  @test isempty(out_of_range)
  isempty(out_of_range) || return x
  bad = [a for a in 1:x.nactive if x.position[x.active[a]] != a]
  @test isempty(bad)
  bad = [k for k in eachindex(x.groups) if !active[k] && x.position[k] != 0]
  @test isempty(bad)
  bad = [i for k in eachindex(x.groups) if !active[k] for i in x.groups[k]
         if !iszero(x.storage[i])]
  @test isempty(bad)
  x
end

function test_iterate(x::Union{SparseIterate, SymmetricSparseIterate})
  ok = length(x.nzval) == length(x.nzval2ind) == length(x.ind2nzval)
  @test length(x.nzval) == length(x.nzval2ind) == length(x.ind2nzval)
  ok || return x
  if x isa SymmetricSparseIterate
    ok = length(x.ind2nzval) == x.p * (x.p + 1) ÷ 2
    @test length(x.ind2nzval) == x.p * (x.p + 1) ÷ 2
    ok || return x
  end
  ok = 0 <= x.nnz <= length(x.ind2nzval)
  @test 0 <= x.nnz <= length(x.ind2nzval)
  ok || return x
  stored = falses(length(x.ind2nzval))
  duplicates, out_of_range = Int[], Int[]
  for j in 1:x.nnz
    i = x.nzval2ind[j]
    if 1 <= i <= length(x.ind2nzval)
      stored[i] && push!(duplicates, j)
      stored[i] = true
    else
      push!(out_of_range, j)
    end
  end
  @test isempty(duplicates)
  @test isempty(out_of_range)
  isempty(out_of_range) || return x
  bad = [j for j in 1:x.nnz if x.ind2nzval[x.nzval2ind[j]] != j]
  @test isempty(bad)
  bad = [i for i in eachindex(x.ind2nzval) if !(0 <= x.ind2nzval[i] <= x.nnz)]
  @test isempty(bad)
  bad = [i for i in eachindex(x.ind2nzval) if !stored[i] && x.ind2nzval[i] != 0]
  @test isempty(bad)
  x
end

####################################
#  proximal operators
####################################

function test_prox(f, x::AbstractArray, γ::Real; rtol = 1e-8, atol = 1e-10, local_points = true, symmetric = false,
                   rng = nothing)
  sym(w) = symmetric ? (w .+ w') ./ 2 : w
  p = prox(f, x, γ)
  @test size(p) == size(x)
  p2 = similar(p)
  @test prox!(p2, f, x, γ) === p2
  @test _dense(p2) ≈ _dense(p) rtol = rtol atol = atol
  p3 = similar(p)
  v = value_and_prox!(p3, f, x, γ)
  @test _dense(p3) ≈ _dense(p) rtol = rtol atol = atol
  fp = f(p)
  @test isfinite(fp)
  @test v ≈ fp rtol = sqrt(rtol) atol = sqrt(atol)
  z = copy(x)
  prox!(z, f, z, γ)
  @test _dense(z) ≈ _dense(p) rtol = rtol atol = atol
  # in place, the combined call stores the point and returns the value of the separate call (O7)
  z = copy(x)
  vz = value_and_prox!(z, f, z, γ)
  @test _dense(z) ≈ _dense(p3) rtol = rtol atol = atol
  @test vz ≈ v rtol = sqrt(rtol) atol = sqrt(atol)

  # optimality
  xd, pd = _dense(x), _dense(p)
  obj = _prox_objective(f, p, x, γ)
  candidates = Any[]
  if local_points
    for δ in (1e-6, 1e-3, 1e-1)
      push!(candidates, pd .+ δ .* sym(_randn(rng, size(pd)...)))
    end
  end
  for _ in 1:3
    push!(candidates, _dense(prox(f, xd .+ sym(_randn(rng, size(xd)...)), γ)))
  end
  slack = 1e-7 * max(1, abs(obj))
  for w in candidates
    @test _prox_objective(f, w, xd, γ) >= obj - slack
    if isconvex(f)
      fw = f(w)
      isfinite(fw) && @test fw >= fp + dot(xd .- pd, w .- pd) / γ - slack
    end
  end
  p
end

####################################
#  gradients
####################################

function test_gradient(f, x::AbstractArray; h = nothing, rtol = nothing, atol = nothing, directions = 3,
                       rng = nothing)
  R = float(eltype(x))
  c = cbrt(eps(R) / eps(Float64))
  h === nothing && (h = 1e-6c)
  rtol === nothing && (rtol = 1e-5c)
  atol === nothing && (atol = 1e-6c)
  g = gradient(f, x)
  g2 = similar(g)
  @test gradient!(g2, f, x) === g2
  @test g2 ≈ g
  g3 = similar(g)
  v = value_and_gradient!(g3, f, x)
  @test g3 ≈ g
  @test v ≈ f(x)
  v4, g4 = value_and_gradient(f, x)
  @test v4 ≈ v && g4 ≈ g
  xd = Array(x)
  xp = similar(xd, float(eltype(x)))
  xm = similar(xp)
  for _ in 1:directions
    d = _randn(rng, eltype(xp), size(xd)...)
    xp .= xd .+ h .* d
    xm .= xd .- h .* d
    fp, fm = f(xp), f(xm)
    fd = (fp - fm) / (2h)
    round_err = 8 * eps(R) * max(abs(fp), abs(fm)) / (2h)
    @test fd ≈ dot(g, d) rtol = rtol atol = max(atol * max(1, abs(fd)), round_err)
  end
  g
end

# A tuple argument holds arrays or nested tuples of arrays (the product spaces of SeparableSum), visited block by
# block and never flattened. Its leaves share one floating type, which sets the same scale as for an array.
# The type is found, and an unsupported argument rejected, before any oracle is called.
_leaf_type(x::AbstractArray) = float(eltype(x))
function _leaf_type(x::Tuple)
  isempty(x) && throw(ArgumentError("test_gradient: a tuple argument needs at least one block"))
  types = map(_leaf_type, x)
  for T in types
    T === types[1] || throw(ArgumentError(
      "test_gradient requires tuple blocks with a common floating-point type; got $(types[1]) and $T"))
  end
  types[1]
end
_leaf_type(x) = throw(ArgumentError("test_gradient: a block must be an array or a tuple of blocks, not a $(typeof(x))"))

# f(x, ys...) on the arrays of x and the matching arrays of ys, collected in the structure of x
_mapblocks(f, x::AbstractArray, ys::AbstractArray...) = f(x, ys...)
@generated function _mapblocks(f, x::Tuple, ys::Tuple...)
  ProximalBase._blocktuple(k -> :(_mapblocks(f, x[$k], $((:(ys[$j][$k]) for j in eachindex(ys))...))), fieldcount(x))
end

# the same blocks, of the same sizes, and blockwise approximately equal
_same_blocks(a::AbstractArray, b::AbstractArray) = size(a) == size(b)
_same_blocks(a::Tuple, b::Tuple) = length(a) == length(b) && all(map(_same_blocks, a, b))
_same_blocks(a, b) = false
_approx(a::AbstractArray, b::AbstractArray) = a ≈ b
_approx(a::Tuple, b::Tuple) = all(map(_approx, a, b))
_agree(a, b) = _same_blocks(a, b) && _approx(a, b)

function test_gradient(f, x::Tuple; h = nothing, rtol = nothing, atol = nothing, directions = 3, rng = nothing)
  R = _leaf_type(x)
  c = cbrt(eps(R) / eps(Float64))
  h === nothing && (h = 1e-6c)
  rtol === nothing && (rtol = 1e-5c)
  atol === nothing && (atol = 1e-6c)
  g = gradient(f, x)
  ok = _same_blocks(g, x)
  @test ok
  ok || return g
  g2 = _mapblocks(similar, g)
  @test gradient!(g2, f, x) === g2
  @test _agree(g2, g)
  g3 = _mapblocks(similar, g)
  v = value_and_gradient!(g3, f, x)
  @test _agree(g3, g)
  @test v ≈ f(x)
  v4, g4 = value_and_gradient(f, x)
  @test v4 ≈ v && _agree(g4, g)
  xd = _mapblocks(Array, x)
  xp = _mapblocks(a -> similar(a, float(eltype(a))), xd)
  xm = _mapblocks(similar, xp)
  for _ in 1:directions
    d = _mapblocks(a -> _randn(rng, eltype(a), size(a)...), xp)
    _mapblocks((p, a, e) -> (p .= a .+ h .* e), xp, xd, d)
    _mapblocks((m, a, e) -> (m .= a .- h .* e), xm, xd, d)
    fp, fm = f(xp), f(xm)
    fd = (fp - fm) / (2h)
    round_err = 8 * eps(R) * max(abs(fp), abs(fm)) / (2h)
    @test fd ≈ ProximalBase._tuple_dot(g, d) rtol = rtol atol = max(atol * max(1, abs(fd)), round_err)
  end
  g
end

####################################
#  separable functions
####################################

# x as floats (the iterates store prox results, which an Int array cannot), with about a third of its entries set to zero
function _with_zeros(x::AbstractArray, rng)
  z = float.(x)
  for i in eachindex(z)
    (rng === nothing ? rand() : rand(rng)) < 1 / 3 && (z[i] = 0)
  end
  z
end

# Whether f has a standalone value at the array z: ProximalBase's own list of functions without one
# (a conjugate whose value is not implemented), and a method that accepts z. A method that exists but
# throws is not "missing": the call is made without a try, and the error surfaces.
_has_standalone_value(f, z) = ProximalBase._has_value(f) && applicable(f, z)

# Whether the value checks at the array a are skipped for lack of a method although f has a value
# method for some arrays: a skip that would otherwise pass silently. A function without a value
# (_has_value false) or without a one-argument method for any array is not reported.
_value_check_skipped(f, a) =
  ProximalBase._has_value(f) && !isempty(methods(f, (AbstractArray,))) && !applicable(f, a)

# the function in a message: a wrapper's full type can be long
_function_name(f) = string(nameof(typeof(f)))

function test_separable(f, x::AbstractVecOrMat, γ::Real; groups = nothing, symmetric = false, rng = nothing)
  @test isseparable(f)
  p = prox(f, x, γ)
  @test [prox_block(f, x[i], i, γ) for i in eachindex(x)] ≈ vec(_dense(p))
  if f isa SeparableFunction
    vx = f(x)
    vs = sum(i -> value_entry(f, x[i], i), eachindex(x))
    isfinite(vx) ? (@test vs ≈ vx) : (@test !isfinite(vs))
  end

  # coordinate steps on a sparse iterate
  xz = _with_zeros(x, rng)
  pz = prox(f, xz, γ)
  xs = SparseIterate(xz)
  y = SparseIterate(float(eltype(x)), size(x)...)
  for k in 1:ncoordinates(xs)
    xs[k] = prox_block(f, xs[k], k, γ)
    y[k] = prox_block(f, xz[k], k, γ)
  end
  @test _dense(xs) ≈ _dense(pz)
  @test _dense(y) ≈ _dense(pz)
  @test nactive(y) == count(!iszero, activevalues(y))
  test_iterate(y)

  # the stored entries only, when zero is preserved
  if iszeropreserving(f)
    T = float(eltype(x))
    @test all(i -> iszero(prox_block(f, zero(T), i, γ)), eachindex(x))
    xs = SparseIterate(xz)
    ys = SparseIterate(xz)
    @test _dense(prox!(ys, f, xs, γ)) ≈ _dense(pz)
    # the per-entry contract f_i(0) = 0, where the entries have values
    f isa SeparableFunction && @test all(i -> iszero(value_entry(f, zero(T), i)), eachindex(x))
    # the aggregate consequence f(0) = 0 and the sparse paths, where f has a value
    z = zeros(T, size(x))
    _has_standalone_value(f, z) && @test iszero(f(z))
    # a value method that rejects the array skips the checks below: say so (at most one message)
    if _value_check_skipped(f, z)
      @info "test_separable: skipped the value checks at zero and on a SparseIterate: $(_function_name(f)) has a value method that does not accept a $(typeof(z))"
    else
      for a in (xs, pz)
        _value_check_skipped(f, a) || continue
        @info "test_separable: skipped the value checks on a SparseIterate: $(_function_name(f)) has a value method that does not accept a $(typeof(a))"
        break
      end
    end
    # a value method may accept only dense arrays: the sparse paths need one for the iterate
    if _has_standalone_value(f, z) && _has_standalone_value(f, xs) && _has_standalone_value(f, pz)
      @test f(xs) ≈ f(xz)
      # a fresh destination, which prox! has not already filled: the value and the written point
      yv = SparseIterate(xz)
      @test value_and_prox!(yv, f, xs, γ) ≈ f(pz)
      @test _dense(yv) ≈ _dense(pz)
    end
  end

  if groups !== nothing
    x isa AbstractVector || throw(ArgumentError("groups need a vector x"))
    a = BlockIterate(xz, groups)
    y = BlockIterate(float(eltype(x)), length(x), groups)
    for k in 1:ncoordinates(a)
      for i in blockindices(a, k)
        a[i] = prox_block(f, a[i], i, γ)
        y[i] = prox_block(f, xz[i], i, γ)
      end
    end
    @test _dense(a) ≈ _dense(pz)
    @test _dense(y) ≈ _dense(pz)
    @test nactive(y) == count(k -> any(!iszero, blockview(y, k)), activecoordinates(y))
    test_iterate(y)
  end

  if symmetric
    size(x, 1) == size(x, 2) || throw(ArgumentError("symmetric = true needs a square x"))
    X = (_dense(x) .+ _dense(x)') ./ 2
    S = SymmetricSparseIterate(X)
    valid = try
      ProximalBase.check_symmetric_parameters(f, S)
    catch err
      err isa ArgumentError || rethrow()
      false
    end
    @test valid
    valid || return p
    y = SymmetricSparseIterate(float(eltype(x)), size(x, 1))
    for k in 1:ncoordinates(S)
      r, c = tril_cartesian(size(x, 1), k)
      i = (c - 1) * size(x, 1) + r
      coordinates(S)[k] = prox_block(f, coordinates(S)[k], i, γ)
      coordinates(y)[k] = prox_block(f, X[r, c], i, γ)
    end
    @test Matrix(S) ≈ _dense(prox(f, X, γ))
    @test Matrix(y) ≈ _dense(prox(f, X, γ))
    @test nactive(y) == count(!iszero, activevalues(y))
    test_iterate(y)
  end
  p
end

####################################
#  conjugates
####################################

function test_conjugate(f, x::AbstractArray, γ::Real; conjugate = Conjugate(f), rtol = 1e-8)
  p = prox(f, x, γ)
  q = prox(conjugate, x ./ γ, inv(γ))
  @test _dense(p) .+ γ .* _dense(q) ≈ _dense(x) rtol = rtol atol = rtol
  y = (_dense(x) .- _dense(p)) ./ γ               # a subgradient of f at p
  fy = try
    conjugate(y)
  catch err
    err isa ArgumentError || rethrow()
    nothing                                         # the value of the conjugate is not known
  end
  if fy !== nothing
    fp = f(p)
    @test fp + fy ≈ dot(_dense(p), y) rtol = sqrt(rtol) atol = sqrt(rtol) * max(1, abs(fp))
    for _ in 1:3
      z = _dense(p) .+ _randn(nothing, size(y)...)
      fz = f(z)
      isfinite(fz) && @test fz + fy >= dot(z, y) - 1e-7 * max(1, abs(fz))
    end
  end
  p
end

####################################
#  optimality of zero
####################################

function test_kkt(f, x::AbstractArray, γ::Real; trials = 5, rng = nothing, tol = 1e-8)
  for _ in 1:trials
    # gradients of various sizes, so that some coordinates stay at zero
    g = _randn(rng, size(x)...) .* exp.(_randn(rng, size(x)...))
    z = _dense(prox(f, -γ .* g, γ))                 # a proximal gradient step from zero
    if x isa BlockIterate
      x0 = similar(x)
      fill!(x0, 0)
      for k in 1:ncoordinates(x)
        viol = kkt_violation(f, g, x0, k)
        abs(viol) > tol && @test (viol <= 0) == all(iszero, view(z, x.groups[k]))
      end
    else
      for i in eachindex(g)
        viol = kkt_violation(f, g, i)
        abs(viol) > tol && @test (viol <= 0) == iszero(z[i])
      end
    end
  end
end

end
