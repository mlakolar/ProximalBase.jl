
function prox_block(f::AbstractFunction, u::AbstractVector, k::Integer, γ::Real)
  out = Vector{float(eltype(u))}(undef, length(u))
  prox_block!(out, f, u, k, γ)
end
