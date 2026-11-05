# The precision of the data of a function, from its type alone (numerics T2, T3). The model precision
# of a call is the widest of the destination, the argument and the data, so that a `Float32` call of a
# function with `Float64` data is a `Float64` model. `Conjugate` takes it into account when it forms the
# reciprocal of a step (`_moreau_reciprocal`), which the inner function then receives. A function
# without data, or one not listed here, gives `Union{}`, which promotes to the other types. Parameters
# (levels, weights, centers, bounds, steps) are not data.
_data_type(::Type) = Union{}

# the float type of the entries of an array or a number type; nothing for other types
_data_float(::Type{T}) where {T} = T <: Union{Real, AbstractArray} && eltype(T) <: Real ? float(eltype(T)) : Union{}

@generated function _data_types(::Type{F}) where {F<:Tuple}
  n = fieldcount(F)
  n == 0 && return :(Union{})
  Expr(:call, :promote_type, (:(_data_type($(fieldtype(F, k)))) for k in 1:n)...)
end

# functions with data matrices and the vectors of their equations (T2)
_data_type(::Type{<:LeastSquaresLoss{T, Ty, Tx}}) where {T, Ty, Tx} = promote_type(_data_float(Ty), _data_float(Tx))
_data_type(::Type{<:GLMLoss{L, T, Ty, TX}}) where {L, T, Ty, TX} = promote_type(_data_float(Ty), _data_float(TX))
_data_type(::Type{<:QuadraticFunction{T, M, V}}) where {T, M, V} = promote_type(_data_float(M), _data_float(V))
_data_type(::Type{<:KronQuadratic{T}}) where {T} = T
_data_type(::Type{<:LogDetLoss{T}}) where {T} = T
_data_type(::Type{<:IndAffine{T, M}}) where {T, M} = promote_type(T, _data_float(M))
_data_type(::Type{<:IndGraph{T, M}}) where {T, M} = promote_type(T, _data_float(M))
_data_type(::Type{<:IndPolyhedron{T, M}}) where {T, M} = promote_type(T, _data_float(M))
_data_type(::Type{<:IndHyperplane{A, T}}) where {A, T} = promote_type(_data_float(A), float(T))
_data_type(::Type{<:IndHalfspace{A, T}}) where {A, T} = promote_type(_data_float(A), float(T))
_data_type(::Type{<:IndHyperslab{A, T}}) where {A, T} = promote_type(_data_float(A), float(T))
_data_type(::Type{<:LinearImage{F, M}}) where {F, M} = promote_type(_data_type(F), _data_float(M))
_data_type(::Type{<:AffineConstrained{T, F, S}}) where {T, F, S} = promote_type(T, _data_type(F), _data_type(S))

# wrappers and sums: the data of what they hold
_data_type(::Type{<:Union{Scaled{F}, Tilt{F}, Translate{F}, Conjugate{F}, MoreauEnvelope{F}, Regularize{F},
                          _ArrayPrecompose{F}, SqrDistL2{F}, DistL2{F}, BoxConstrained{F}, BallConstrained{F},
                          SphereConstrained{F}, SingularValue{F}}}) where {F} = _data_type(F)
_data_type(::Type{<:Precompose{F, M}}) where {F, M} = promote_type(_data_type(F), _data_float(M))
_data_type(::Type{<:LinearConstrained{F, H}}) where {F, H} = promote_type(_data_type(F), _data_type(H))
_data_type(::Type{<:FiberSum{F, D}}) where {F, D} = promote_type(_data_type(F), _data_type(D))
_data_type(::Type{<:Union{SeparableSum{F}, Sum{F}, PointwiseMinimum{F}}}) where {F<:Tuple} = _data_types(F)
