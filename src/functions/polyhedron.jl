####################################
#
#  Polyhedra {x : l ≤ Ax ≤ u, xmin ≤ x ≤ xmax}
#
####################################

"""
    IndPolyhedron(A, l, u; maxiter = 100 * (sum(size(A)) + 1))
    IndPolyhedron(A, l, u, xmin, xmax; maxiter = 100 * (sum(size(A)) + 1))

The indicator of `{x : l ≤ Ax ≤ u, xmin ≤ x ≤ xmax}` for a finite real
`m × n` matrix `A`, dense or sparse, and vectors `x` of length `n`. Each bound
is a scalar or a vector of the corresponding length. Coordinate bounds
default to `-Inf` and `Inf`. Infinite sides are omitted; equal finite bounds
are native equalities with free multipliers. A zero row is allowed when its
bounds contain zero. Bounds must be ordered and cannot contain `NaN` or an
equality at infinity. The data are copied at construction; construct a new
function to change the constraints.

The proximal operator is the Euclidean projection, independent of the scalar
step. A dual active-set method adds violated constraints and drops inequalities
whose multipliers reach zero. It keeps a thin orthogonal factorization of at
most `n` active normals, with reorthogonalized additions and Givens deletions.
Equalities are never dropped. The previous working set is reused after solving
its equality-constrained problem and dropping negative inequality multipliers.
[`reset_cache!`](@ref) clears this warm start. Owned workspaces make concurrent
calls on the same function unsafe.

The membership value uses `√eps` relative to the row norm, the argument norm
and the bounds, with an absolute floor in the original constraint units.
The solver instead uses `8eps(T)` times each normalized row's bound and
absolute dot-product scale. Working-set checks also allow cancellation at
the input's scale. Residual corrections refine the working projection, and
every row is reconsidered before returning, including skipped equalities.
Its dependence threshold is `8n * eps(T)` times the
normal's norm. A dependent violated row takes a dual-only step when possible;
otherwise a Farkas certificate is checked to `64n * eps(T)` and verified in
exact arithmetic on the stored data before reporting an empty set with
`ArgumentError`. This exceptional verification allocates and can cost more
than a factor update. A genuine small direction is retained when the rank
test was inconclusive. Numerical failure and exhaustion of `maxiter` steps
throw distinct `ErrorException`s. As for other indicators,
`value_and_prox!` returns zero without a feasibility certificate.

Rows and offsets are scaled by powers of two. Nonzero coefficients and finite
offsets must remain representable after scaling; iterates, triangular solves
and multipliers must also be representable. Accuracy is limited by numerical
rank and by the less precise of the data and output types. The workspace type
is the promoted floating type of `A` and the finite bounds; default infinite
bounds do not promote `Float32`. Values use `float(eltype(x))`.

Workspace storage is `O(n² + m + n)` in addition to the copied matrix. Each
search costs `O(mn)` for dense data or `O(nnz(A) + m + n)` for sparse data;
factor updates cost `O(n|S|)` for working set `S`. Many active constraints can
therefore be expensive even for a sparse matrix. Warmed `Float32` and
`Float64` calls allocate no memory outside exceptional certificate verification.
"""
mutable struct IndPolyhedron{T<:AbstractFloat, M<:AbstractMatrix{T}} <: AbstractFunction
  At::M                       # columns are scaled rows, including on the sparse path
  l::Vector{T}
  u::Vector{T}
  xmin::Vector{T}
  xmax::Vector{T}
  scale::Vector{Int}           # scaled row = 2^scale[i] times the original row
  norms::Vector{T}
  rows::Vector{Int}            # signed row of A, or signed coordinate m + j
  bounds::Vector{T}
  equalities::Int              # equalities precede inequality sides
  n::Int
  m::Int
  maxiter::Int
  active::Vector{Int}          # a negative index reverses an equality's normal
  used::BitVector
  count::Int
  Q::Matrix{T}
  R::Matrix{T}
  multipliers::Vector{T}       # in scaled constraint units
  origin::Vector{T}
  point::Vector{T}
  normal::Vector{T}
  direction::Vector{T}
  coeff::Vector{T}
  dual::Vector{T}
  residual::Vector{T}
  valid::Bool
  iterations::Int
end

function _poly_bound_type(::Type{T}, b::Real) where {T}
  isnan(b) && throw(ArgumentError("polyhedron bounds cannot contain NaN"))
  isfinite(b) ? float(promote_type(T, typeof(b))) : T
end
function _poly_bound_type(::Type{T}, b::AbstractVector{<:Real}) where {T}
  S = T
  for v in b
    S = _poly_bound_type(S, v)
  end
  S
end

function _poly_bounds(b, ::Type{T}, n) where {T}
  b isa Real || b isa AbstractVector{<:Real} ||
    throw(ArgumentError("polyhedron bounds must be real scalars or vectors"))
  b isa Real || length(b) == n || throw(DimensionMismatch("a polyhedron bound must have length $n"))
  v = b isa Real ? fill(T(b), n) : Vector{T}(b)
  for a in v
    isnan(a) && throw(ArgumentError("polyhedron bounds cannot contain NaN"))
  end
  v
end

function _poly_order(l, u)
  for i in eachindex(l, u)
    l[i] <= u[i] || throw(ArgumentError("polyhedron lower bounds must not exceed upper bounds"))
    (l[i] == Inf || u[i] == -Inf) &&
      throw(ArgumentError("polyhedron bounds describe an empty set (an infinite equality)"))
  end
end

function IndPolyhedron(A::AbstractMatrix{<:Real}, l, u, xmin = -Inf, xmax = Inf;
                       maxiter::Integer = 100 * (sum(size(A)) + 1))
  m, n = size(A)
  maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
  T = float(eltype(A))
  for b in (l, u, xmin, xmax)
    b isa Real || b isa AbstractVector{<:Real} ||
      throw(ArgumentError("polyhedron bounds must be real scalars or vectors"))
    T = _poly_bound_type(T, b)
  end
  lo, hi = _poly_bounds(l, T, m), _poly_bounds(u, T, m)
  lower, upper = _poly_bounds(xmin, T, n), _poly_bounds(xmax, T, n)
  _poly_order(lo, hi)
  _poly_order(lower, upper)
  At = A isa SparseMatrixCSC ? copy(sparse(transpose(T.(A)))) : Matrix{T}(transpose(A))
  scale, norms = zeros(Int, m), zeros(T, m)
  for i in 1:m
    a = view(At, :, i)
    ma = _maximum_abs(a)
    isfinite(ma) || throw(ArgumentError("A must have finite real entries"))
    if iszero(ma)
      lo[i] <= 0 <= hi[i] || throw(ArgumentError("polyhedron is empty (a zero row excludes zero)"))
      continue
    end
    scale[i] = -exponent(ma)
    _poly_scale_column!(At, i, scale[i])
    norms[i] = norm(a)
    for v in (lo, hi)
      if isfinite(v[i])
        b = ldexp(v[i], scale[i])
        isfinite(b) && (iszero(v[i]) || !iszero(b)) ||
          throw(ArgumentError("a scaled polyhedron bound is outside the supported range"))
        v[i] = b
      end
    end
  end
  rows, bounds = Int[], T[]
  equalities = 0
  # Native equalities need only one multiplier, whose sign is unrestricted.
  for eq in (true, false)
    for i in 1:(m + n)
      i <= m && iszero(norms[i]) && continue
      a, b = i <= m ? (lo[i], hi[i]) : (lower[i - m], upper[i - m])
      if eq
        if a == b
          push!(rows, i)
          push!(bounds, a)
        end
      elseif a != b
        if isfinite(b)
          push!(rows, i)
          push!(bounds, b)
        end
        if isfinite(a)
          push!(rows, -i)
          push!(bounds, -a)
        end
      end
    end
    eq && (equalities = length(rows))
  end
  IndPolyhedron(At, lo, hi, lower, upper, scale, norms, rows, bounds, equalities,
    n, m, Int(maxiter), zeros(Int, n), falses(length(rows)), 0, zeros(T, n, n),
    zeros(T, n, n), zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n),
    zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n), false, 0)
end

function _poly_scale_entry(a, e)
  b = ldexp(a, e)
  iszero(a) || !iszero(b) || throw(ArgumentError("a scaled coefficient is outside the supported range"))
  b
end
function _poly_scale_column!(A::AbstractMatrix, i, e)
  @inbounds for j in axes(A, 1)
    A[j, i] = _poly_scale_entry(A[j, i], e)
  end
end
function _poly_scale_column!(A::SparseMatrixCSC, i, e)
  vals = nonzeros(A)
  @inbounds for h in nzrange(A, i)
    vals[h] = _poly_scale_entry(vals[h], e)
  end
end

isproximable(::Type{<:IndPolyhedron}) = true
isconvex(::Type{<:IndPolyhedron}) = true
_isindicator(::Type{<:IndPolyhedron}) = true

function _checkparams(f::IndPolyhedron, x)
  x isa AbstractVector || throw(DimensionMismatch("IndPolyhedron requires a vector"))
  length(x) == f.n || throw(DimensionMismatch("x has length $(length(x)); the polyhedron requires $(f.n)"))
  size(f.At) == (f.n, f.m) && length(f.l) == length(f.u) == f.m &&
    length(f.xmin) == length(f.xmax) == f.n ||
    throw(DimensionMismatch("polyhedron data sizes changed; construct a new IndPolyhedron"))
  true
end

function _checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::IndPolyhedron, y, x)
  _checkparams(f, x)
  y isa AbstractVector || throw(DimensionMismatch("IndPolyhedron requires a vector output"))
  _checksize(y, x)
  true
end

# Reading columns of the transpose avoids sparse row indexing and allocation.
function _poly_dot(A::AbstractMatrix, i, x, s)
  v = zero(promote_type(eltype(A), eltype(x), typeof(s[1])))
  @inbounds for j in axes(A, 1)
    v += A[j, i] * _scale_value(x[j], s)
  end
  v
end
function _poly_dot(A::SparseMatrixCSC, i, x, s)
  v = zero(promote_type(eltype(A), eltype(x), typeof(s[1])))
  rows, vals = rowvals(A), nonzeros(A)
  @inbounds for h in nzrange(A, i)
    v += vals[h] * _scale_value(x[rows[h]], s)
  end
  v
end

function (f::IndPolyhedron{T})(x::AbstractArray) where {T}
  _checkparams(f, x)
  V = float(eltype(x))
  S = promote_type(T, V)
  mx = _maximum_abs(x)
  isfinite(mx) || return V(Inf)
  tol = max(_tol(T), _tol(V))
  # Common powers of two keep dot products and absolute floors in range.
  for i in 1:f.m
    lo, hi = f.l[i], f.u[i]
    lo == -Inf && hi == Inf && continue
    k = _comparison_exponent(_magnitude_exponent(mx), f.scale[i],
      isfinite(lo) ? _magnitude_exponent(abs(lo)) : 0,
      isfinite(hi) ? _magnitude_exponent(abs(hi)) : 0)
    s = _power_factors(S, -k)
    a = _poly_dot(f.At, i, x, s)
    nx = sqrt(_scaled_sumabs2(x, s))
    floor = ldexp(one(S), f.scale[i] - k)
    for upper in (false, true)
      b = upper ? hi : lo
      isfinite(b) || continue
      d = ldexp(S(b), -k)
      slack = max(tol * floor, tol * abs(d), _positive_product(S(tol), S(f.norms[i]), S(nx)))
      (upper ? a <= d + slack : a >= d - slack) || return V(Inf)
    end
  end
  @inbounds for j in 1:f.n
    a = float(x[j])
    lo, hi = f.xmin[j], f.xmax[j]
    isfinite(lo) && a < lo && lo - a > tol * max(one(S), abs(a), abs(lo)) && return V(Inf)
    isfinite(hi) && a > hi && a - hi > tol * max(one(S), abs(a), abs(hi)) && return V(Inf)
  end
  zero(V)
end

@inline _poly_bound(f, id) = id > 0 ? f.bounds[id] : -f.bounds[-id]
@inline function _poly_row(f, id)
  r = f.rows[abs(id)]
  id > 0 ? r : -r
end
@inline function _poly_norm(f, id)
  r = abs(_poly_row(f, id))
  r <= f.m ? f.norms[r] : one(eltype(f.point))
end
@inline function _poly_constraint(f::IndPolyhedron{T}, id, x) where {T}
  r = _poly_row(f, id)
  i = abs(r)
  a = i <= f.m ? _poly_dot(f.At, i, x, (one(T), one(T))) : x[i - f.m]
  r > 0 ? a : -a
end

function _poly_column!(a, A::AbstractMatrix, i, sign)
  @inbounds for j in eachindex(a)
    a[j] = sign * A[j, i]
  end
end
function _poly_column!(a, A::SparseMatrixCSC, i, sign)
  fill!(a, zero(eltype(a)))
  rows, vals = rowvals(A), nonzeros(A)
  @inbounds for h in nzrange(A, i)
    a[rows[h]] = sign * vals[h]
  end
end
function _poly_normal!(a, f::IndPolyhedron{T}, id) where {T}
  r = _poly_row(f, id)
  i, sign = abs(r), r > 0 ? one(T) : -one(T)
  if i <= f.m
    _poly_column!(a, f.At, i, sign)
  else
    fill!(a, zero(T))
    a[i - f.m] = sign
  end
  a
end

@noinline _poly_failure() = error("IndPolyhedron numerical failure: the working factorization could not be recovered")
function _poly_tick!(f)
  f.iterations += 1
  f.iterations <= f.maxiter || error("IndPolyhedron iteration limit exhausted (maxiter = $(f.maxiter))")
end

# For inequality sides Bz ≤ d, the dual minimizes
# ½‖B'λ‖² + (d - Bx)'λ over λ ≥ 0; equality multipliers are free.
# Invariant: C = Q*R, z = x - C*λ, C' z = d. A row scaled by 2^e
# has original multiplier 2^e*λ (with its side's sign). Twice
# orthogonalizing also refines Q'a without forming the normal equations.
function _poly_direction!(f::IndPolyhedron{T}, id) where {T}
  _poly_normal!(f.normal, f, id)
  copyto!(f.direction, f.normal)
  k = f.count
  fill!(f.coeff, zero(T))
  for pass in 1:2
    @inbounds for j in 1:k
      a = zero(T)
      for i in 1:f.n
        a += f.Q[i, j] * f.direction[i]
      end
      f.coeff[j] += a
      for i in 1:f.n
        f.direction[i] -= a * f.Q[i, j]
      end
    end
  end
  # r solves C*r = projection_C(a). Updating λ by -t*r and the incoming
  # multiplier by t moves z by -t*(a - C*r), preserving stationarity.
  @inbounds for j in k:-1:1
    a = f.coeff[j]
    for h in (j + 1):k
      a -= f.R[j, h] * f.dual[h]
    end
    f.dual[j] = a / f.R[j, j]
    isfinite(f.dual[j]) || _poly_failure()
  end
  norm(f.direction)
end

function _poly_append!(f, id, multiplier, nv)
  k = f.count + 1
  k <= f.n && nv > 0 && isfinite(nv) || _poly_failure()
  @inbounds for i in 1:f.n
    f.Q[i, k] = f.direction[i] / nv
    f.R[i, k] = i < k ? f.coeff[i] : zero(nv)
  end
  f.R[k, k] = nv
  f.active[k] = id
  f.used[abs(id)] = true
  f.multipliers[k] = multiplier
  f.count = k
end

# Removing a column leaves an upper Hessenberg R. Stable Givens rotations
# restore triangular form and rotate the corresponding columns of Q.
function _poly_drop!(f::IndPolyhedron{T}, j) where {T}
  k = f.count
  f.used[abs(f.active[j])] = false
  @inbounds for h in j:(k - 1)
    f.active[h] = f.active[h + 1]
    f.multipliers[h] = f.multipliers[h + 1]
    for i in 1:k
      f.R[i, h] = f.R[i, h + 1]
    end
  end
  @inbounds for h in j:(k - 1)
    a, b = f.R[h, h], f.R[h + 1, h]
    r = hypot(a, b)
    iszero(r) && _poly_failure()
    c, s = a / r, b / r
    for i in h:(k - 1)
      a, b = f.R[h, i], f.R[h + 1, i]
      f.R[h, i] = c * a + s * b
      f.R[h + 1, i] = -s * a + c * b
    end
    f.R[h + 1, h] = zero(T)
    for i in 1:f.n
      a, b = f.Q[i, h], f.Q[i, h + 1]
      f.Q[i, h] = c * a + s * b
      f.Q[i, h + 1] = -s * a + c * b
    end
  end
  f.count = k - 1
end

# Rebuild without allocating if accumulated updates lose orthogonality.
function _poly_refactor!(f::IndPolyhedron{T}) where {T}
  k = f.count
  f.count = 0
  for j in 1:k
    id, multiplier = f.active[j], f.multipliers[j]
    nv = _poly_direction!(f, id)
    # A previously verified independent direction can be below the rank threshold.
    nv > 0 && isfinite(nv) || _poly_failure()
    _poly_append!(f, id, multiplier, nv)
  end
end

# Apply eps before accumulating magnitudes: neither a row's absolute dot
# product nor its product of norms needs to fit in the working type.
function _poly_absdot(A::AbstractMatrix{T}, i, x, tol) where {T}
  scale = zero(T)
  @inbounds for j in axes(A, 1)
    scale += _positive_product(tol, abs(A[j, i]), abs(x[j]))
  end
  scale
end
function _poly_absdot(A::SparseMatrixCSC{T}, i, x, tol) where {T}
  scale = zero(T)
  rows, vals = rowvals(A), nonzeros(A)
  @inbounds for h in nzrange(A, i)
    scale += _positive_product(tol, abs(vals[h]), abs(x[rows[h]]))
  end
  scale
end
@inline function _poly_absdot(f, id, x, tol)
  row = abs(_poly_row(f, id))
  row <= f.m ? _poly_absdot(f.At, row, x, tol) : tol * abs(x[row - f.m])
end
@inline function _poly_slack(f::IndPolyhedron{T}, id, x, tol = 8 * eps(T)) where {T}
  slack = max(tol, tol * abs(_poly_bound(f, id)), _poly_absdot(f, id, x, tol))
  isfinite(slack) || _poly_failure()
  slack
end

function _poly_project_working!(f::IndPolyhedron{T}) where {T}
  k = f.count
  # R' h = C'x - d, R λ = h avoids the squared condition number of C'C.
  @inbounds for j in 1:k
    a = _poly_constraint(f, f.active[j], f.origin) - _poly_bound(f, f.active[j])
    for h in 1:(j - 1)
      a -= f.R[h, j] * f.coeff[h]
    end
    f.coeff[j] = a / f.R[j, j]
  end
  @inbounds for j in k:-1:1
    a = f.coeff[j]
    for h in (j + 1):k
      a -= f.R[j, h] * f.multipliers[h]
    end
    f.multipliers[j] = a / f.R[j, j]
    isfinite(f.multipliers[j]) || _poly_failure()
  end
  copyto!(f.point, f.origin)
  @inbounds for j in 1:k, i in 1:f.n
    f.point[i] -= f.Q[i, j] * f.coeff[j]
  end
  all(isfinite, f.point) || _poly_failure()
end

function _poly_working_valid(f::IndPolyhedron{T}) where {T}
  tol = 64 * eps(T)
  nx = _norm2_scaled(_ScaledArray(f.point, tol))
  isfinite(nx) || return false
  for j in 1:f.count
    id = f.active[j]
    slack = max(_poly_slack(f, id, f.point, tol),
      _poly_absdot(f, id, f.origin, tol), _poly_norm(f, id) * nx)
    isfinite(slack) || return false
    abs(_poly_constraint(f, id, f.point) - _poly_bound(f, id)) <=
      slack || return false
  end
  true
end

# Subtracting the projection from a much larger input can lose boundary
# digits. Correct those residuals through the same QR, without normal equations.
function _poly_refine!(f::IndPolyhedron{T}) where {T}
  k = f.count
  for pass in 1:2
    accurate = true
    @inbounds for j in 1:k
      id = f.active[j]
      a = _poly_constraint(f, id, f.point) - _poly_bound(f, id)
      accurate &= abs(a) <= _poly_slack(f, id, f.point)
      for h in 1:(j - 1)
        a -= f.R[h, j] * f.coeff[h]
      end
      f.coeff[j] = a / f.R[j, j]
    end
    accurate && return
    @inbounds for j in k:-1:1
      a = f.coeff[j]
      for h in (j + 1):k
        a -= f.R[j, h] * f.dual[h]
      end
      f.dual[j] = a / f.R[j, j]
      f.multipliers[j] += f.dual[j]
      isfinite(f.multipliers[j]) || _poly_failure()
    end
    @inbounds for j in 1:k, i in 1:f.n
      f.point[i] -= f.Q[i, j] * f.coeff[j]
    end
    all(isfinite, f.point) || _poly_failure()
  end
end

function _poly_restore!(f)
  while true
    _poly_project_working!(f)
    _poly_refine!(f)
    if !_poly_working_valid(f)
      _poly_refactor!(f)
      _poly_project_working!(f)
      _poly_refine!(f)
      _poly_working_valid(f) || _poly_failure()
    end
    drop = 0
    for j in 1:f.count
      if abs(f.active[j]) > f.equalities && f.multipliers[j] < 0
        drop = j
        break
      end
    end
    iszero(drop) && return
    _poly_tick!(f)
    _poly_drop!(f, drop)
  end
end

# With w = (-r, 1), a - C*r = 0 and d_new - d_W'r < 0 certify
# infeasibility. Inequality components of w must be nonnegative; equality
# components are unrestricted. Residuals are checked in scaled row units.
function _poly_certificate(f::IndPolyhedron{T}, id) where {T}
  copyto!(f.residual, f.normal)
  gap = _poly_bound(f, id)
  na, nd = _poly_norm(f, id), abs(gap)
  for j in 1:f.count
    r, old = f.dual[j], f.active[j]
    abs(old) > f.equalities && r > 0 && return false
    _poly_normal!(f.direction, f, old)
    @inbounds for i in 1:f.n
      f.residual[i] -= r * f.direction[i]
    end
    d = _poly_bound(f, old)
    gap -= r * d
    na += abs(r) * _poly_norm(f, old)
    nd += abs(r * d)
  end
  tol = 64 * max(1, f.n) * eps(T)
  isfinite(gap) && isfinite(na) && isfinite(nd) &&
    norm(f.residual) <= tol * na && gap < -tol * max(one(T), nd)
end

# A small nonzero direction is not a proof of dependence. Verify the proposed
# certificate on the exact binary data, only when no floating-point step exists.
# This slow path also distinguishes a genuine tiny direction from a redundant
# normal whose orthogonalization residual consists entirely of rounding.
function _poly_verify_certificate(f, id)
  k, n = f.count, f.n
  C = Matrix{Rational{BigInt}}(undef, n, k + 1)
  for j in 1:k
    _poly_normal!(f.residual, f, f.active[j])
    for i in 1:n
      C[i, j] = Rational{BigInt}(f.residual[i])
    end
  end
  for i in 1:n
    C[i, k + 1] = Rational{BigInt}(f.normal[i])
  end
  # Elimination solves C_W*r = a without deciding rank from a tolerance.
  for j in 1:k
    pivot = j
    while pivot <= n && iszero(C[pivot, j])
      pivot += 1
    end
    pivot <= n || return :uncertain
    if pivot != j
      for h in j:(k + 1)
        C[j, h], C[pivot, h] = C[pivot, h], C[j, h]
      end
    end
    for i in (j + 1):n
      iszero(C[i, j]) && continue
      a = C[i, j] / C[j, j]
      for h in (j + 1):(k + 1)
        C[i, h] -= a * C[j, h]
      end
      C[i, j] = 0
    end
  end
  for i in (k + 1):n
    iszero(C[i, k + 1]) || return :independent
  end
  r = Vector{Rational{BigInt}}(undef, k)
  gap = Rational{BigInt}(_poly_bound(f, id))
  for j in k:-1:1
    a = C[j, k + 1]
    for h in (j + 1):k
      a -= C[j, h] * r[h]
    end
    r[j] = a / C[j, j]
    abs(f.active[j]) > f.equalities && r[j] > 0 && return :uncertain
    gap -= r[j] * Rational{BigInt}(_poly_bound(f, f.active[j]))
  end
  gap < 0 ? :infeasible : :uncertain
end

function _poly_add!(f::IndPolyhedron{T}, id) where {T}
  incoming = zero(T)
  refreshed = false
  while true
    _poly_tick!(f)
    nv = _poly_direction!(f, id)
    violation = _poly_constraint(f, id, f.point) - _poly_bound(f, id)
    dependent = f.count == f.n || nv <= 8 * max(1, f.n) * eps(T) * _poly_norm(f, id)
    if dependent && !refreshed
      _poly_refactor!(f)
      refreshed = true
      continue
    end
    if dependent && abs(id) <= f.equalities &&
        abs(violation) <= _poly_slack(f, id, f.point)
      return
    end
    # Only positive r_i can make an inequality multiplier hit zero.
    dualstep, drop = T(Inf), 0
    for j in 1:f.count
      if abs(f.active[j]) > f.equalities && f.dual[j] > 0
        t = max(zero(T), f.multipliers[j]) / f.dual[j]
        if t < dualstep
          dualstep, drop = t, j
        end
      end
    end
    primalstep = dependent ? T(Inf) : (max(zero(T), violation) / nv) / nv
    if !isfinite(dualstep) && !isfinite(primalstep)
      _poly_certificate(f, id) || _poly_failure()
      certificate = _poly_verify_certificate(f, id)
      certificate === :infeasible && throw(ArgumentError("polyhedron is empty (infeasible linear constraints)"))
      certificate === :independent && f.count < f.n && nv > 0 || _poly_failure()
      dependent = false
      nv = _poly_direction!(f, id)
      primalstep = (max(zero(T), violation) / nv) / nv
    end
    t = min(dualstep, primalstep)
    isfinite(t) || _poly_failure()
    @inbounds for j in 1:f.count
      f.multipliers[j] -= t * f.dual[j]
    end
    incoming += t
    isfinite(incoming) || _poly_failure()
    if !dependent
      @inbounds for i in 1:f.n
        f.point[i] -= t * f.direction[i]
      end
    end
    if primalstep <= dualstep
      _poly_append!(f, id, incoming, nv)
      return
    end
    _poly_drop!(f, drop)
    refreshed = false
  end
end

function _poly_solve!(f::IndPolyhedron{T}, x) where {T}
  warm = f.valid
  f.valid = false
  f.iterations = 0
  copyto!(f.origin, x)
  if isempty(f.rows)
    copyto!(f.point, f.origin)
    f.count = 0
    f.valid = true
    return f.point
  end
  if warm
    _poly_restore!(f)
  else
    f.count = 0
    fill!(f.used, false)
    copyto!(f.point, f.origin)
    for id in 1:f.equalities
      residual = _poly_constraint(f, id, f.point) - _poly_bound(f, id)
      _poly_add!(f, residual < 0 ? -id : id)
      _poly_restore!(f)
    end
  end
  while true
    _poly_restore!(f)
    best, violation = 0, zero(T)
    # A previously skipped row can become violated after other rows enter,
    # or when a warm input changes. Check every row in its own constraint units.
    for id in eachindex(f.rows)
      a = _poly_constraint(f, id, f.point) - _poly_bound(f, id)
      isfinite(a) || _poly_failure()
      eq = id <= f.equalities
      residual = eq ? abs(a) : a
      if residual > _poly_slack(f, id, f.point)
        f.used[id] && _poly_failure()
        if residual / _poly_norm(f, id) > violation
          best, violation = eq && a < 0 ? -id : id, residual / _poly_norm(f, id)
        end
      end
    end
    iszero(best) && break
    _poly_add!(f, best)
  end
  f.valid = true
  f.point
end

function prox!(y::AbstractArray, f::IndPolyhedron{T}, x::AbstractArray, ::Real) where {T}
  try
    _checkwrite(prox!, f, y, x)
    # The solver needs finite entries in its working precision.
    for a in x
      if !isfinite(a) || !isfinite(T(a))
        f.valid = false
        return fill!(y, convert(eltype(y), NaN))
      end
    end
    _poly_solve!(f, x)
    copyto!(y, f.point)
  catch
    f.valid = false
    rethrow()
  end
  y
end

function value_and_prox!(y::AbstractArray, f::IndPolyhedron, x::AbstractArray, γ::Real = 1)
  prox!(y, f, x, γ)
  zero(_valtype(y))
end
