using ProximalBase, Test, LinearAlgebra
const PB=ProximalBase
struct CustomSquare{A} <: SeparableFunction; w::A; end
PB.value_entry(f::CustomSquare,v,i)=f.w[i]*v^2/2
PB.prox_block(f::CustomSquare,v,i,gamma)=v/(1+gamma*f.w[i])
PB.iszeropreserving(::CustomSquare)=true
PB.entry_parameters(f::CustomSquare)=(:w=>f.w,)
PB.check_parameters(f::CustomSquare,x)=length(f.w)==prod(size(x)) ? nothing : throw(DimensionMismatch())
@testset "Scaled custom extension hooks" begin
 for T in (Float32,Float64), kind in (:dense,:sparse,:block,:owner),scale in (T(.2),T(2))
  a=kind===:owner ? T[1 2;2 0] : T[1,2,0,-1]
  w=kind===:owner ? T[1 3;3 2] : T[1,3,0,2]
  x=kind===:owner ? SymmetricSparseIterate(a) : kind===:sparse ? SparseIterate(a) : kind===:block ? BlockIterate(a,[[4,1],[3,2]]) : copy(a)
  f=Scaled(CustomSquare(w),scale);gamma=T(.3)
  @test f(x)≈scale*sum(w.*a.^2)/2
  z=prox(f,x,gamma)
  @test (kind===:owner ? Matrix(z) : Array(z))≈a./(1 .+gamma*scale.*w)
  out=copy(x);v=value_and_prox!(out,f,out,gamma)
  @test v≈f(out)
  before=kind===:owner ? Matrix(x) : Array(x)
  bad=Scaled(CustomSquare(w[1:2]),scale)
  @test_throws DimensionMismatch prox!(x,bad,x,gamma)
  @test (kind===:owner ? Matrix(x) : Array(x))==before
 end
end
