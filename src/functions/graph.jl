####################################
#
#  Graphs of linear maps {(x, y) : y = Ax}
#
####################################

"""
    IndGraph(A; operator = !(A isa AbstractMatrix), rtol = √eps(T), maxiter = 2min(size(A)...) + 10)

The indicator of the graph `{(x, y) : y = Ax}` of a real `m × n` matrix or
linear operator `A`, on pairs of vectors of lengths `n` and `m`. Its proximal
operator projects `(c, d)` onto the graph, independently of the scalar step:

- with `n ≤ m`, solve `(I + AᵀA)x = c + Aᵀd` and return `(x, Ax)`;
- with `n > m`, solve `(I + AAᵀ)q = d - Ac` and return `(c + Aᵀq, d - q)`.

The input pair is scaled by a power of two before forming the right-hand side,
and the result is rescaled into the output blocks' types. This avoids overflow
in the right-hand side when the projection itself is representable.
At extreme scales, direct solves use one residual correction before rescaling
to avoid overflowing a representable boundary result through solve rounding.

For a dense matrix, a Cholesky factorization of the smaller system is cached
at construction, and projections allocate nothing. A sparse matrix is factored
sparsely through `cholesky`; its solves allocate and may compute in `Float64`
before conversion to `T`. Here `T = float(eltype(A))` is the workspace type;
integer matrices are supported. The Gram matrix must be representable in `T`
and retain positive definiteness after adding the identity in that precision.
Nonfinite matrices, overflowing Gram matrices and failed factorizations throw
`ArgumentError`; no singular directions are truncated.

With `operator = true`, or for an `A` that is not an `AbstractMatrix`, `A`
must provide `size(A)`, a real `eltype(A)`, `mul!(y, A, x)` and
`mul!(x, A', y)`. Conjugate gradients solve the smaller system from the previous
solution, stopping when the true residual satisfies `‖r‖ ≤ rtol * ‖rhs‖`.
A warm start larger in norm than the new right-hand side is discarded.
A zero right-hand side returns zero immediately. At `maxiter`, the last iterate
is returned, so the projection can be inexact. `f.converged` reports whether
this residual test was met, and `f.iterations` counts the iterations of the
last call (zero for a direct solve). Nonfinite products or iterates throw
`ArgumentError` before writing output. After the first call, operator calls
allocate nothing if the operator's products allocate nothing.

The value is `0` when
`‖Ax - y‖ ≤ √eps(S) * max(1, ‖y‖, ‖A‖_F * ‖x‖)`, and `Inf` otherwise,
with `S` the promoted floating type of `A` and both input blocks. For operators
the bound instead uses `max(1, ‖y‖, ‖Ax‖)`. Nonfinite operator images give `Inf`.
Matrix entries and the input vector are converted to `S` before the value's
product. Operators must support products in this promoted type. The value
buffers are reused while this promoted type stays unchanged.
[`value_and_prox!`](@ref) returns a typed zero in the promoted floating type of
the data and all input and output blocks, without checking feasibility.
Allocating [`prox`](@ref) preserves each block's floating type.

Outputs may alias inputs. Both pairs' shapes are checked before any write.
`A` is stored by reference: construct a new `IndGraph` after changing it;
[`reset_cache!`](@ref) throws. The function owns its buffers and is not
thread-safe.
"""
mutable struct IndGraph{T<:AbstractFloat, M, F} <: AbstractFunction
  A::M
  fact::F
  tall::Bool
  rhs::Vector{T}
  solution::Vector{T}
  residual::Vector{T}
  direction::Vector{T}
  product::Vector{T}
  x::Vector{T}
  y::Vector{T}
  c::Vector{T}
  d::Vector{T}
  value::Scratch
  value_input::Scratch
  value_matrix::Scratch
  scale::T
  rtol::T
  maxiter::Int
  iterations::Int
  converged::Bool
end

function IndGraph(A; operator::Bool = !(A isa AbstractMatrix),
                  rtol::Real = _tol(float(eltype(A))),
                  maxiter::Integer = 2 * min(size(A)...) + 10)
  length(size(A)) == 2 || throw(ArgumentError("A must have two dimensions"))
  eltype(A) <: Real || throw(ArgumentError("IndGraph requires real data"))
  T = float(eltype(A))
  isconcretetype(T) && T <: AbstractFloat ||
    throw(ArgumentError("A must have a concrete real floating-point workspace type"))
  tol = convert(T, rtol)
  isfinite(tol) && tol > 0 || throw(ArgumentError("rtol must be positive and finite"))
  maxiter >= 1 || throw(ArgumentError("maxiter must be positive"))
  A isa AbstractMatrix && !all(isfinite, _stored_values(A)) &&
    throw(ArgumentError("A must have finite entries"))
  operator || A isa AbstractMatrix || throw(ArgumentError("a linear operator requires operator = true"))
  m, n = size(A)
  k = min(m, n)
  tall = n <= m
  fact = operator ? nothing : _graph_factor(A, T, tall)
  scale = operator ? zero(T) : convert(T, norm(A))
  IndGraph(A, fact, tall, zeros(T, k), zeros(T, k), zeros(T, k),
    zeros(T, k), zeros(T, k), zeros(T, n), zeros(T, m), zeros(T, n), zeros(T, m),
    Scratch(zeros(T, m)), Scratch(zeros(T, n)), Scratch(),
    scale, tol, Int(maxiter), 0, false)
end

_graph_matrix(A::AbstractMatrix, ::Type{T}) where {T} = Matrix{T}(A)
_graph_matrix(A::SparseMatrixCSC, ::Type{T}) where {T} = SparseMatrixCSC{T, Int}(A)

function _graph_factor(A, ::Type{T}, tall) where {T}
  k = min(size(A)...)
  k == 0 && return cholesky(Symmetric(zeros(T, 0, 0)))
  B = _graph_matrix(A, T)
  G = tall ? B' * B : B * B'
  for i in axes(G, 1)
    G[i, i] += one(T)
  end
  all(isfinite, _stored_values(G)) ||
    throw(ArgumentError("the Gram matrix of A overflows the workspace precision"))
  try
    cholesky(Symmetric(G))
  catch err
    err isa PosDefException || err isa ZeroPivotException || rethrow()
    throw(ArgumentError("the Gram matrix plus identity lost positive definiteness at the workspace precision"))
  end
end

isproximable(::Type{<:IndGraph}) = true
_isindicator(::Type{<:IndGraph}) = true
isconvex(::Type{<:IndGraph}) = true

function _checkparams(f::IndGraph, z)
  z isa Tuple && length(z) == 2 ||
    throw(DimensionMismatch("IndGraph needs a tuple of two vector blocks"))
  z[1] isa AbstractVector && z[2] isa AbstractVector ||
    throw(DimensionMismatch("IndGraph blocks must be vectors"))
  length(z[1]) == length(f.x) && length(z[2]) == length(f.y) ||
    throw(DimensionMismatch("IndGraph needs blocks of lengths $(length(f.x)) and $(length(f.y))"))
  eltype(z[1]) <: Real && eltype(z[2]) <: Real ||
    throw(ArgumentError("IndGraph requires real blocks"))
  true
end

function _checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::IndGraph, out, z)
  _checkparams(f, z)
  _checkparams(f, out)
  _checksize(out, z)
  true
end

function (f::IndGraph{T})(z) where {T}
  _checkparams(f, z)
  x, y = z
  S = float(promote_type(T, eltype(x), eltype(y)))
  all(isfinite, x) && all(isfinite, y) || return S(Inf)
  r = _array_buffer!(f.value, S, (length(f.y),))
  xs = eltype(x) === S ? x : copyto!(_array_buffer!(f.value_input, S, (length(f.x),)), x)
  A = _graph_value_matrix!(f.value_matrix, f.A, S)
  mul!(r, A, xs)
  f.fact === nothing && return _graph_operator_value!(r, y, S)
  r .-= y
  nr = _norm2(r)
  slack = _tol(S) * max(one(S), _norm2(y), f.scale * _norm2(x))
  isfinite(nr) && isfinite(slack) && return _ind(nr <= slack, S)
  # The affine value uses the same residual, including its overflow-safe scaling.
  _affine_scaled_value((A = f.A, b = y, rb = r), x, S)
end

_graph_value_matrix!(::Scratch, A, ::Type) = A
function _graph_value_matrix!(scratch::Scratch, A::AbstractMatrix, ::Type{S}) where {S}
  eltype(A) === S && return A
  copyto!(_array_buffer!(scratch, S, size(A)), A)
end
function _graph_value_matrix!(scratch::Scratch, A::SparseMatrixCSC, ::Type{S}) where {S}
  eltype(A) === S && return A
  B = scratch.buf
  B isa SparseMatrixCSC{S, Int} && size(B) == size(A) && return B
  B = _graph_matrix(A, S)
  scratch.buf = B
  B
end

function _graph_operator_value!(r, y, ::Type{S}) where {S}
  all(isfinite, r) || return S(Inf)
  k = _comparison_exponent(_magnitude_exponent(_maximum_abs(r)),
    _magnitude_exponent(_maximum_abs(y)))
  s = _power_factors(S, -k)
  na = sqrt(_scaled_sumabs2(r, s))
  ny = sqrt(_scaled_sumabs2(y, s))
  @inbounds for i in eachindex(r, y)
    r[i] = _scale_value(r[i], s) - _scale_value(y[i], s)
  end
  _ind(_norm2(r) <= _tol(S) * max(ldexp(one(S), -k), na, ny), S)
end

_graph_finite(v) = all(isfinite, v) ||
  throw(ArgumentError("IndGraph encountered a nonfinite product or iterate"))

function _graph_system!(out, f::IndGraph, v)
  if f.tall
    mul!(f.y, f.A, v)
    _graph_finite(f.y)
    mul!(out, f.A', f.y)
  else
    mul!(f.x, f.A', v)
    _graph_finite(f.x)
    mul!(out, f.A, f.x)
  end
  out .+= v
  _graph_finite(out)
  out
end

function _graph_ldiv!(out, fact::Cholesky, rhs)
  copyto!(out, rhs)
  ldiv!(fact, out)
end
_graph_ldiv!(out, fact, rhs) = copyto!(out, fact \ rhs)

function _graph_solve!(f::IndGraph)
  _graph_ldiv!(f.solution, f.fact, f.rhs)
  f.converged = true
end

function _graph_refine!(f::IndGraph{T}, k) where {T}
  f.fact === nothing && return
  k >= exponent(floatmax(T)) - 1 || return
  # A rounding error near floatmax can turn a finite boundary solution into Inf.
  _graph_system!(f.product, f, f.solution)
  f.residual .= f.rhs .- f.product
  _graph_finite(f.residual)
  all(iszero, f.residual) && return
  _graph_ldiv!(f.residual, f.fact, f.residual)
  f.solution .+= f.residual
end

function _graph_solve!(f::IndGraph{T, M, Nothing}) where {T<:AbstractFloat, M}
  rhs, q, r, p, v = f.rhs, f.solution, f.residual, f.direction, f.product
  nb = _norm2(rhs)
  _graph_finite(nb)
  if iszero(nb)
    fill!(q, zero(T))
    f.converged = true
    return
  end
  tol = _positive_product(f.rtol, one(T), nb)
  # The solution of this SPD system has norm at most that of its rhs.
  _norm2(q) > nb && fill!(q, zero(T))
  _graph_system!(v, f, q)
  r .= rhs .- v
  _graph_finite(r)
  nr = _norm2(r)
  if nr <= tol
    f.converged = true
    return
  end
  if nr > nb
    fill!(q, zero(T))
    copyto!(r, rhs)
    nr = nb
  end
  # Powers of two keep squared directions in range without rounding a rescaling.
  unit = ldexp(one(T), _magnitude_exponent(nr))
  r ./= unit
  copyto!(p, r)
  rr = dot(r, r)
  for j in 1:f.maxiter
    _graph_system!(v, f, p)
    pv = dot(p, v)
    isfinite(rr) && isfinite(pv) && rr > 0 && pv > 0 ||
      throw(ArgumentError("IndGraph conjugate gradients needs finite positive curvature"))
    α = rr / pv
    _graph_finite(α)
    step = _positive_product(α, unit, one(T))
    _graph_finite(step)
    q .+= step .* p
    _graph_finite(q)
    f.iterations = j
    # Check the true residual, rather than the recursively updated residual.
    _graph_system!(v, f, q)
    r .= rhs .- v
    _graph_finite(r)
    if _norm2(r) <= tol
      f.converged = true
      return
    end
    r ./= unit
    next = dot(r, r)
    β = next / rr
    _graph_finite(β)
    p .= r .+ β .* p
    rr = next
  end
  nothing
end

function _graph_scale_inputs!(f::IndGraph, z)
  c, d = z
  _graph_finite(c)
  _graph_finite(d)
  k = _magnitude_exponent(max(_maximum_abs(c), _maximum_abs(d)))
  @inbounds for i in eachindex(f.c, c)
    f.c[i] = ldexp(float(c[i]), -k)
  end
  @inbounds for i in eachindex(f.d, d)
    f.d[i] = ldexp(float(d[i]), -k)
  end
  k
end

function _graph_scaled_output(v, ::Type{T}, k) where {T}
  S = float(promote_type(typeof(v), T))
  convert(T, ldexp(S(v), k))
end

function _graph_project!(f::IndGraph, k)
  c, d = f.c, f.d
  if f.tall
    mul!(f.rhs, f.A', d)
    f.rhs .+= c
    _graph_finite(f.rhs)
    _graph_solve!(f)
    _graph_refine!(f, k)
    copyto!(f.x, f.solution)
    mul!(f.y, f.A, f.x)
  else
    mul!(f.y, f.A, c)
    _graph_finite(f.y)
    f.rhs .= d .- f.y
    _graph_finite(f.rhs)
    _graph_solve!(f)
    _graph_refine!(f, k)
    mul!(f.x, f.A', f.solution)
    f.x .+= c
    # This identity avoids cancellation in A * (c + Aᵀq).
    f.y .= d .- f.solution
  end
  _graph_finite(f.x)
  _graph_finite(f.y)
end

function prox!(out, f::IndGraph, z, ::Real)
  _checkwrite(prox!, f, out, z)
  f.iterations, f.converged = 0, false
  k = 0
  try
    k = _graph_scale_inputs!(f, z)
    _graph_project!(f, k)
    # Check both rescaled blocks before an aliased output can overwrite the input.
    for v in f.x
      _graph_finite(_graph_scaled_output(v, eltype(out[1]), k))
    end
    for v in f.y
      _graph_finite(_graph_scaled_output(v, eltype(out[2]), k))
    end
  catch
    fill!(f.solution, zero(eltype(f.solution)))
    f.converged = false
    rethrow()
  end
  @inbounds for i in eachindex(out[1], f.x)
    out[1][i] = _graph_scaled_output(f.x[i], eltype(out[1]), k)
  end
  @inbounds for i in eachindex(out[2], f.y)
    out[2][i] = _graph_scaled_output(f.y[i], eltype(out[2]), k)
  end
  out
end

function value_and_prox!(out, f::IndGraph{T}, z, γ::Real = 1) where {T}
  _checkwrite(value_and_prox!, f, out, z)
  prox!(out, f, z, γ)
  zero(float(promote_type(T, eltype(z[1]), eltype(z[2]), eltype(out[1]), eltype(out[2]))))
end
