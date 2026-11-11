module CustomFunctionExample
using ProximalBase
struct WeightedSquare{A} <: SeparableFunction
    w::A
end
ProximalBase.value_entry(f::WeightedSquare,v,i)=f.w[i]*v^2/2
ProximalBase.prox_block(f::WeightedSquare,v::Real,i::Integer,γ::Real)=v/(1+γ*f.w[i])
ProximalBase.iszeropreserving(::WeightedSquare)=true
ProximalBase.isconvex(::Type{<:WeightedSquare})=true
ProximalBase.entry_parameters(f::WeightedSquare)=(:w=>f.w,)
ProximalBase.check_parameters(f::WeightedSquare,x)=length(f.w)==prod(size(x)) ? nothing : throw(DimensionMismatch("weights"))
f=WeightedSquare([1.,2,3]);x=[1.,-2,0]
@assert prox(f,x,.5)≈x ./ (1 .+ 0.5 .* f.w)
@assert f(SparseIterate(x))≈sum(f.w.*x.^2)/2
end
