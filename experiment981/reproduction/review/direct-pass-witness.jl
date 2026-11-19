using ProximalBase, Test
struct Reads{T} <: AbstractVector{T}
  a::Vector{T}
  reads::Base.RefValue{Int}
end
Base.size(v::Reads) = size(v.a)
Base.getindex(v::Reads,i::Int) = (v.reads[] += 1; v.a[i])
function passes(u, c)
  T = eltype(u)
  v = Reads(-u, Ref(0))
  τ = ProximalBase._linf_face_level(v, ones(T,length(u)), one(T), ProximalBase._ValRead{T}(), c)
  τ, div(v.reads[],length(v))-1
end
for (T,n) in [(Float32,8),(Float64,13),(BigFloat,35)]
  g = T[one(T)]
  for k in 2:n
    push!(g, 2*(k+2)*last(g)-sum(g))
  end
  g .*= T(0.4)/last(g)
  @show T n passes([T(0.5),T(0.5),-g...],one(T))
end
function sample()
  v = [1f0; fill(.009486833f0,100000)]
  f = NormLinf(.1f0)
  @show passes(v,.1f0)
  for x in (zero(v), -ones(Float32,length(v)))
    @show kkt_violation(f,v,x)
    @show @allocated kkt_violation(f,v,x)
  end
end
sample()
