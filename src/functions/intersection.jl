####################################
#
#  The intersection of sets, by Dykstra's algorithm
#
####################################

"""
    IndIntersection(C₁, C₂, …; tol = 1e-10, maxiter = 10_000, warmstart = true)

The indicator of the intersection of the sets `C₁, C₂, …`, given by their
indicator functions, such as [`IndPSD`](@ref), [`IndBox`](@ref) or
[`IndHyperplane`](@ref). Its proximal operator, the projection onto the
intersection, need not have a closed form even when each set has one. It is
computed by Dykstra's algorithm (Boyle and Dykstra, 1986), which projects onto
the sets in turn and corrects each projection by the increment of the previous
round. For closed convex sets with a nonempty intersection it converges to the
projection onto that intersection. Plain alternating projections can converge
to a different point of the intersection.

The proximal operator is iterative and so inexact: it stops when the sum of
squared step lengths in a round is at most
`max(tol, 10eps(T))^2 * max(1, norm(y))^2`, with `T` the floating-point type of
the output, or after `maxiter` rounds. Only the first non-converged call in a
session warns; later calls are silent and must be checked through `f.converged`.
This step-length test does not bound the distance to the intersection. Nearly
parallel sets can satisfy it far from the intersection, and `maxiter` can also stop before
feasibility. The last set's projection determines membership in that set only
to its own accuracy. `f.iterations` counts the rounds of the last call, and
`f.converged` reports only whether the step-length test was met; neither is a
feasibility certificate.

With `warmstart`, each call starts from the corrections of the previous call,
which Dykstra's algorithm allows, since it is block coordinate ascent on the
dual of the projection (Tibshirani, 2017). Inside a solver the successive inputs
are close, so the rounds per call drop as the solver converges. The corrections
and the iterate live in buffers that `f` owns, sized by the first input, so
later calls allocate nothing; do not share `f` between threads.

Each set is visited through its own `prox!`, its projection. With a nonconvex
set, such as [`IndBallL0`](@ref), the iteration is only a heuristic: it may
stop at a point of the intersection that is not the projection, or cycle
without converging.

For example, the nearest correlation matrix (Higham, 2002) is the projection
onto the positive semidefinite matrices with unit diagonal:

```julia
L = fill(-Inf, p, p); L[diagind(L)] .= 1
U = fill(Inf, p, p);  U[diagind(U)] .= 1
C = prox(IndIntersection(IndPSD(), IndBox(L, U)), R)
```
"""
mutable struct IndIntersection{S<:Tuple} <: AbstractFunction
  sets::S
  tol::Float64
  maxiter::Int
  warmstart::Bool
  buffers::Any          # a _DykstraBuffers, sized by the input
  iterations::Int       # the rounds of the last call
  converged::Bool       # whether the last call converged

  function IndIntersection{S}(sets::S, tol::Float64, maxiter::Int, warmstart::Bool) where {S<:Tuple}
    isempty(sets) && throw(ArgumentError("IndIntersection needs at least one set"))
    for C in sets
      isproximable(C) && _isindicator(C) ||
        throw(ArgumentError("IndIntersection needs the indicator functions of sets, whose proximal operators are projections, but $(typeof(C)) is not one"))
    end
    tol > 0 || throw(ArgumentError("tol must be positive, got $tol"))
    maxiter >= 1 || throw(ArgumentError("maxiter must be positive, got $maxiter"))
    new{S}(sets, tol, maxiter, warmstart, nothing, 0, false)
  end
end
IndIntersection(sets::AbstractFunction...; tol::Real = 1e-10, maxiter::Integer = 10_000, warmstart::Bool = true) =
  IndIntersection{typeof(sets)}(sets, Float64(tol), Int(maxiter), warmstart)

isproximable(::Type{<:IndIntersection}) = true
_isindicator(::Type{<:IndIntersection}) = true
isconvex(::Type{IndIntersection{S}}) where {S} = all(isconvex, fieldtypes(S))

# 0 when x is in every set, within the tolerance of each
(f::IndIntersection)(x::AbstractArray) = _ind(_in_every_set(f.sets, x), float(eltype(x)))
_in_every_set(::Tuple{}, x) = true
_in_every_set(sets::Tuple, x) = iszero(first(sets)(x)) && _in_every_set(Base.tail(sets), x)

# the iterate y, a buffer w and the correction of each set
mutable struct _DykstraBuffers{T, N}
  y::Array{T, N}
  w::Array{T, N}
  corrections::Vector{Array{T, N}}
  started::Bool         # whether the corrections come from a previous call
end

function _dykstra_buffers!(f::IndIntersection, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
  b = f.buffers
  b isa _DykstraBuffers{T, N} && size(b.y) == dims && return b
  b = _DykstraBuffers{T, N}(zeros(T, dims), zeros(T, dims), [zeros(T, dims) for _ in f.sets], false)
  f.buffers = b
  b
end

function _checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::IndIntersection, out::AbstractArray, x::AbstractArray)
  _checksize(out, x)
  _checkintersection(f, float(eltype(out)), size(x))
end
function _checkintersection(f::IndIntersection, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
  b = f.buffers
  if b isa _DykstraBuffers{T, N} && size(b.y) == dims
    y, w = b.y, b.w
  else
    y, w = Array{T, N}(undef, dims), Array{T, N}(undef, dims)
  end
  map(C -> _checkwrite(prox!, C, y, w), f.sets)
  true
end

# One round over the sets, unrolled over the tuple of sets: for set k,
# y ← P(y + pₖ) and pₖ ← (y + pₖ) - P(y + pₖ). The change of pₖ is the step of y,
# so the sum of the squared changes is the squared length of the path of y.
_dykstra_round!(::Tuple{}, P, k, y, w, s) = s
function _dykstra_round!(sets::Tuple, P, k, y, w, s)
  p = P[k]
  w .= y .+ p
  prox!(y, first(sets), w, one(eltype(y)))
  @inbounds for i in eachindex(y, w, p)
    q = w[i] - y[i]
    s += abs2(q - p[i])
    p[i] = q
  end
  _dykstra_round!(Base.tail(sets), P, k + 1, y, w, s)
end

function prox!(out::AbstractArray, f::IndIntersection, x::AbstractArray, ::Real)
  _checkwrite(prox!, f, out, x)
  T = float(eltype(out))
  b = _dykstra_buffers!(f, T, size(x))
  y, w, P = b.y, b.w, b.corrections
  y .= x
  if f.warmstart && b.started
    for p in P                  # x = y + Σₖ pₖ holds throughout Dykstra's algorithm
      y .-= p
    end
  else
    for p in P
      fill!(p, zero(T))
    end
  end
  b.started = true
  tol = max(T(f.tol), 10 * eps(T))
  converged = false
  rounds = 0
  while rounds < f.maxiter
    rounds += 1
    s = _dykstra_round!(f.sets, P, 1, y, w, zero(T))
    if s <= abs2(tol * max(one(T), norm(y)))
      converged = true
      break
    end
  end
  f.iterations, f.converged = rounds, converged
  converged || _warn_dykstra(f.maxiter)
  copyto!(out, y)
  out
end

@noinline _warn_dykstra(maxiter) =
  @warn "IndIntersection: Dykstra's algorithm did not converge in $maxiter rounds; the projection is inexact" maxlog = 1
