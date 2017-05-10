shrink{T<:AbstractFloat}(v::T, c::T) = v > c ? v - c : (v < -c ? v + c : zero(T))

function f1{T <: AbstractFloat}(
  out_x::AbstractArray{T}, x::AbstractArray{T}, S::AbstractArray{T}, tmp::AbstractArray{T}, λ::T)

  ρ = one(T) / λ
  @. tmp = ρ * x  - S

  ef = eigfact!(tmp)::Base.LinAlg.Eigen{T,T}
  @inbounds @simd for i in eachindex(ef[:values])
    t = ef[:values][i]
    ef[:values][i] = sqrt( (t + sqrt(t^2. + 4.*ρ)) / (2.*ρ) )
  end
  scale!(ef[:vectors], ef[:values])
  A_mul_Bt!(out_x, ef[:vectors],ef[:vectors])
end

function f2{T <: AbstractFloat}(
  out_x::AbstractArray{T}, x::AbstractArray{T}, S::AbstractArray{T}, tmp::AbstractArray{T}, λ::T)


  ρ = one(T) / λ
  @. tmp = x * ρ - S
  ef = eigfact!(Symmetric(tmp))
  d = getindex(ef, :values)::Vector{T}
  U = getindex(ef, :vectors)::Matrix{T}
  @inbounds @simd for i in eachindex(d)
    t = d[i]
    d[i] = sqrt( (t + sqrt(t^2. + 4.*ρ)) / (2.*ρ) )
  end
  scale!(U, d)
  A_mul_Bt!(out_x, U, U)
end


function test_f()
  numtest = 1000
  λ = 1.
  n = 1000
  p = 50
  x = randn(n, p)
  S = x'x / n

  v = eye(p)
  tmp = similar(S)

  y1 = similar(S)
  y2 = similar(S)
  # y3 = similar(S)
  @time for i=1:numtest
    y1 = f1(y1, v, S, tmp, λ)
  end
  @time for i=1:numtest
    y2 = f2(y2, v, S, tmp, λ)
  end
  # @time for i=1:numtest
  #   y3 = f3(y3, x, λ)
  # end

  @show maximum(abs.(y1 - y2))
  # @show maximum(abs.(y1 - y3))
  nothing
end
