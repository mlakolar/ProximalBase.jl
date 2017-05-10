
"""
  shrink(v, c)

Soft-threshold operator. Returns sign(v)⋅max(0, |v|-c)
"""
shrink{T<:AbstractFloat}(v::T, c::T) = v > c ? v - c : (v < -c ? v + c : zero(T))


"""
Computes a proximal operation for the penalty
\[ \lambda_1\cdot(|u|+|v|) + \lambda_2\cdot|u-v| . \]

Returns (u, v) = arg min (1/2γ) * ((u - x)^2 + (v-y)^2) + λ1 * (|u|+|v|) + λ2 * |u-v|
"""
function proxL1Fused{T <: AbstractFloat}(x::T, y::T, λ1::T, λ2::T, γ::T=one(T))

  if λ2 > zero(T)
    t = λ2 * γ
    if x > y + 2. * t
      u = x - t
      v = y + t
    elseif y > x + 2. * t
      u = x + t
      v = y - t
    else
      u = v = (x + y) / 2.
    end
  else
    u = x
    v = y
  end

  t = λ1 * γ
  (shrink(u, t), shrink(v, t))
end
