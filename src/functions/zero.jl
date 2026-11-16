####################################
#  zero
####################################

"""
    ZeroFunction()

Zero function ``f(x) = 0`` for an unpenalized problem. Its gradient is zero and
its proximal operator is the identity. Real, one-based arrays are required;
shape and output checks still apply.

# Examples
```jldoctest
julia> f = ZeroFunction();

julia> f([2.0, -1.0])
0.0

julia> prox(f, [2.0, -1.0])
2-element Vector{Float64}:
  2.0
 -1.0
```

# Edge cases
- Distinct matching sparse/block iterate inputs and outputs use `copyto!`,
  replacing destination activity, including stored zeros.
- Symmetric-iterate proximal writes preserve destination activity; gradients
  at or into symmetric iterates are not implemented.

See [zero-function operations](@ref Zero-function) for copying and step behavior.
"""
struct ZeroFunction <: AbstractFunction end

issmooth(::Type{ZeroFunction}) = true
isproximable(::Type{ZeroFunction}) = true
isseparable(::Type{ZeroFunction}) = true
isconvex(::Type{ZeroFunction}) = true
iszeropreserving(::ZeroFunction) = true
_checkwrite(op::Union{typeof(prox!), typeof(value_and_prox!), typeof(gradient!), typeof(value_and_gradient!)},
            f::ZeroFunction, y::_ArrayOrSymmetricIterate, x::_ArrayOrSymmetricIterate) =
  (_check_one_based(y, x); _checkreal(f, x); _checkreal(f, y); (op === gradient! || op === value_and_gradient!) && _checkgradient_output(y, x); _checksize(y, x); true)

(f::ZeroFunction)(x::AbstractArray) = (_check_one_based(x); _checkreal(f, x); zero(_valtype(x)))
# the value at a symmetric iterate, which a wrapper of ZeroFunction needs (its other oracles at a symmetric iterate are the proximal operator)
(f::ZeroFunction)(x::SymmetricSparseIterate) = (_checkreal(f, x); zero(_valtype(x)))

function prox!(y::AbstractArray, f::ZeroFunction, x::AbstractArray, ::Real)
  x = _symmetric_argument(y, x)
  _checkwrite(prox!, f, y, x)
  y === x || copyto!(y, x)
  y
end
value_and_prox!(y::AbstractArray, f::ZeroFunction, x::AbstractArray, γ::Real) =
  (x = _symmetric_argument(y, x); _prox_unchecked!(y, f, x, γ); zero(_valtype(y)))
# a symmetric iterate output takes the symmetric part of the argument, one coordinate at a time: a coordinate that is
# active stays active (the gradient of ZeroFunction is defined for arrays only)
function prox!(y::SymmetricSparseIterate, f::ZeroFunction, x::_ArrayOrSymmetricIterate, ::Real)
  x = _symmetric_argument(y, x)
  _checkwrite(prox!, f, y, x)
  _assign!(y, x)
end
value_and_prox!(y::SymmetricSparseIterate, f::ZeroFunction, x::_ArrayOrSymmetricIterate, γ::Real) =
  (prox!(y, f, x, γ); zero(_valtype(y)))

function value_and_gradient!(g::AbstractArray, f::ZeroFunction, x::AbstractArray)
  _checkwrite(value_and_gradient!, f, g, x)
  fill!(g, zero(eltype(g)))
  zero(_valtype(x))
end
gradient!(g::AbstractArray, f::ZeroFunction, x::AbstractArray) = (_value_and_gradient_unchecked!(g, f, x); g)

prox_block(::ZeroFunction, v::Real, ::Integer, ::Real) = v
kkt_violation(::ZeroFunction, g::AbstractArray, i::Integer) =
  (_check_one_based(g); eltype(g) <: Real || _real_error(:kkt_violation); abs(g[i]))
restrict(f::ZeroFunction, ::AbstractVector{<:Integer}) = f


lipschitz(::ZeroFunction) = 0.0
