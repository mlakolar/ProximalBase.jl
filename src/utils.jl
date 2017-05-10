
"""
  shrink(v, c)

Soft-threshold operator. Returns sign(v)⋅max(0, |v|-c)
"""
shrink{T<:AbstractFloat}(v::T, c::T) = v > c ? v - c : (v < -c ? v + c : zero(T))
