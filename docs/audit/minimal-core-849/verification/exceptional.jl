using LinearAlgebra,Test
module Frozen
include(joinpath(get(ENV,"PB_BASELINE","/tmp/pb849-baseline"),"src","ProximalBase.jl"))
end
module Current
include(joinpath(get(ENV,"PB_CANDIDATE",normpath(joinpath(@__DIR__,"..","..","..",".."))),"src","ProximalBase.jl"))
end
const B=Frozen.ProximalBase
const C=Current.ProximalBase
canon(x::AbstractArray)=Array(x)
canon(x)=x isa Union{B.SymmetricSparseIterate,C.SymmetricSparseIterate} ? Matrix(x) : x
function capture(thunk)
 try
  canon(thunk())
 catch e
  (nameof(typeof(e)),)
 end
end
function check(P,T,family,lambda,scale,xvalues,mode,step)
 w=T[0,1,Inf,2]
 x=mode===:owner ? P.SymmetricSparseIterate(reshape(xvalues,2,2)) : mode===:sparse ? P.SparseIterate(xvalues) : mode===:block ? P.BlockIterate(xvalues,[[4,1],[3,2]]) : copy(xvalues)
 f=family===:l1 ? P.NormL1(lambda,mode===:owner ? reshape(T[1,2,2,0],2,2) : w) : family===:ridge ? P.SqrNormL2(lambda) : P.SqrNormL2(lambda,mode===:owner ? reshape(T[1,2,2,0],2,2) : T[1,2,-1,0])
 f=scale===:none ? f : scale===:nested ? P.Scaled(P.Scaled(f,ldexp(one(T),-100)),ldexp(one(T),-100)) : P.Scaled(f,scale)
 result=Any[capture(()->f(x)),capture(()->P.prox(f,x,step))]
 push!(result,capture() do
  out=copy(x); v=P.value_and_prox!(out,f,x,step); (v,mode===:owner ? Matrix(out) : Array(out))
 end)
 push!(result,capture() do
  out=copy(x); v=P.value_and_prox!(out,f,out,step); (v,mode===:owner ? Matrix(out) : Array(out))
 end)
 if P.issmooth(f) && mode!==:owner
  push!(result,capture(()->P.gradient(f,x)))
  push!(result,capture() do
   out=similar(xvalues); v=P.value_and_gradient!(out,f,x); (v,Array(out))
  end)
 end
 result
end
mismatches=0;total=0
for T in (Float32,Float64),family in (:l1,:ridge,:centered),lambda in (zero(T),one(T),T(Inf),floatmin(T),floatmax(T)),scale in (:none,:nested,zero(T),one(T),T(Inf),floatmin(T),floatmax(T)),mode in (:dense,:sparse,:block,:owner),xs in (T[0,0,0,0],T[1,2,2,-1],T[NaN,0,0,Inf],T[floatmax(T),0,0,-floatmax(T)]),step in (zero(T),one(T),T(Inf))
 global total+=1
 b=capture(()->check(B,T,family,lambda,scale,xs,mode,step));c=capture(()->check(C,T,family,lambda,scale,xs,mode,step))
 if !isequal(b,c)
  global mismatches+=1
  mismatches<=8 && println("MISMATCH ",(T,family,lambda,scale,xs,mode,step),"\n",b,"\n",c)
 end
end
println("CASES=",total," MISMATCHES=",mismatches)
@test mismatches==0
