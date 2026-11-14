####################################
#
#  Diagonal metrics: a step per entry, and per-coordinate curvatures
#
####################################

# In the metric of D = Diagonal(d), the proximal operator of a separable g is
#   argmin_z g(z) + Σᵢ dᵢ (zᵢ - xᵢ)² / 2,
# the scalar proximal operator with the step γᵢ = 1/dᵢ in every entry, and for a
# group norm a metric constant on each group keeps the closed form. A gradient
# method with the step 1/Lₖ in coordinate k, from the per-coordinate curvatures
# below, takes the scale of each column into account without standardizing X.

prox(f::AbstractFunction, x::AbstractArray, γ::AbstractArray) = prox!(similar(x, float(eltype(x))), f, x, γ)

prox!(y::AbstractArray, f::ZeroFunction, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, f, y, x, γ); _prox_unchecked!(y, f, x, γ))
function _prox_unchecked!(y::AbstractArray, f::ZeroFunction, x::AbstractArray, γ::AbstractArray)
  _checkwrite(prox!, f, y, x)
  _checkparam(γ, x)
  y === x ? y : copyto!(y, x)
end
value_and_prox!(y::AbstractArray, f::ZeroFunction, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, f, y, x, γ); _value_and_prox_unchecked!(y, f, x, γ))
_value_and_prox_unchecked!(y::AbstractArray, f::ZeroFunction, x::AbstractArray, γ::AbstractArray) =
  (_checkreal(f, γ); _prox_unchecked!(y, f, x, γ); zero(_valtype(y)))

# separable functions: the step of each entry; on a SymmetricSparseIterate the
# steps are a p × p array read at the lower triangle, so give a symmetric one.
# Scalar and array steps share the loops of separable.jl; _entry specializes
# to the scalar or to γ[i] without a branch inside the loop.
prox!(y::AbstractArray, f::SeparableFunction, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, f, y, x, γ); _prox_sep!(y, f, x, γ))
_prox_unchecked!(y::AbstractArray, f::SeparableFunction, x::AbstractArray, γ::AbstractArray) =
  _isbuiltin(f) ? _prox_sep!(y, f, x, γ) : prox!(y, f, x, γ)
value_and_prox!(y::AbstractArray, f::SeparableFunction, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, f, y, x, γ); _value_and_prox_sep!(y, f, x, γ))
_value_and_prox_unchecked!(y::AbstractArray, f::SeparableFunction, x::AbstractArray, γ::AbstractArray) =
  _isbuiltin(f) ? _value_and_prox_sep!(y, f, x, γ) : value_and_prox!(y, f, x, γ)

