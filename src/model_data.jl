# The precision of the data of a function, from its type alone (numerics T2, T3). The model precision
# of a call is the widest of the destination, the argument and the data, so that a `Float32` call of a
# function with `Float64` data is a `Float64` model. `Conjugate` takes it into account when it forms the
# reciprocal of a step (`_moreau_reciprocal`), which the inner function then receives. A function
# without data, or one not listed here, gives `Union{}`, which promotes to the other types. Parameters
# (levels, weights, centers, bounds, steps) are not data.
_data_type(::Type) = Union{}

# the float type of the entries of an array, a linear operator or a number type; Union{} for a type
# without a real element type (nothing, a function)
_data_float(::Type{T}) where {T} = eltype(T) <: Real ? float(eltype(T)) : Union{}

# functions with data matrices and the vectors of their equations (T2)
_data_type(::Type{<:LeastSquaresLoss{T, Ty, Tx}}) where {T, Ty, Tx} = promote_type(_data_float(Ty), _data_float(Tx))
# T is the promotion of A, b and the constant c
_data_type(::Type{<:QuadraticFunction{T, M, V}}) where {T, M, V} = promote_type(T, _data_float(M), _data_float(V))
_data_type(::Type{<:Scaled{F}}) where {F} = _data_type(F)
_data_of(f) = _data_type(typeof(f))
_data_of(::Nothing) = Union{}
_data_of(f::Scaled) = _data_of(f.f)
