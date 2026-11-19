using ProximalBase
struct OnesVector <: AbstractVector{Float32}
    n::Int
end
Base.size(v::OnesVector) = (v.n,)
Base.IndexStyle(::Type{OnesVector}) = IndexLinear()
Base.getindex(v::OnesVector, i::Int) = (@boundscheck checkbounds(v, i); 1f0)
n = 1 << 26
v = isempty(ARGS) ? OnesVector(n) : ones(Float32, n)
println((; VERSION, n, storage=typeof(v), observed=kkt_violation(NormLinf(0f0), v, v), expected=sqrt(Float32(n))))
