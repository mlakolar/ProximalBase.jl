module CoreFunctionsTest
using Test, ProximalBase, LinearAlgebra, SparseArrays, Random
const PB=ProximalBase

# A user extension with only mathematical hooks and explicit parameter metadata.
struct WeightedSquare{A} <: SeparableFunction
    w::A
end
PB.value_entry(f::WeightedSquare,v,i)=f.w[i]*v^2/2
PB.prox_block(f::WeightedSquare,v::Real,i::Integer,γ::Real)=v/(1+γ*f.w[i])
PB.iszeropreserving(::WeightedSquare)=true
PB.isconvex(::Type{<:WeightedSquare})=true
PB.entry_parameters(f::WeightedSquare)=(:w=>f.w,)
PB.check_parameters(f::WeightedSquare,x)=length(f.w)==prod(size(x)) ? nothing : throw(DimensionMismatch("weights"))
struct CompactSquare <: AbstractFunction end
function PB.prox_block!(out::AbstractVector,::CompactSquare,u::AbstractVector,k::Integer,γ::Real)
    length(out)==length(u) || throw(DimensionMismatch("block"))
    out .= u ./ (1+γ*k)
    out
end

@testset "public bindings and dispatch" begin
    @test all(n->isdefined(PB,n),names(PB))
    @test isempty(Test.detect_ambiguities(PB;recursive=true))
end
for T in (Float32,Float64)
    @testset "$T values, gradients and proximal points" begin
        x=T[1,-2,0,3];γ=T(.4);y=similar(x)
        for f in (ZeroFunction(),LinearFunction(T[.2,-.3,.1,.5],T(.7)),
                  SqrNormL2(T(.3)),SqrNormL2(T(.3),T[.5,.2,-1,2]),
                  QuadraticFunction(Diagonal(T[1,2,3,4]),T[.1,.2,.3,.4],T(.5)),
                  NormL1(T(.3)),NormL1(T(.3),T[1,0,2,3]))
            @test value_and_prox!(y,f,x,γ)≈f(y)
            @test prox(f,x,γ)≈y
            alias=copy(x);prox!(alias,f,alias,γ);@test alias≈y
            if issmooth(f)
                @test value_and_gradient!(y,f,x)≈f(x)
                @test gradient(f,x)≈y
                h=T===Float32 ? T(.003) : T(1e-6)
                finite_difference=[(f(x+h*Matrix{T}(I,4,4)[:,i])-f(x-h*Matrix{T}(I,4,4)[:,i]))/(2h) for i in 1:4]
                @test y≈finite_difference rtol=(T===Float32 ? .003 : 1e-8) atol=(T===Float32 ? .003 : 1e-8)
            end
        end
        f=NormL1(T(.3),T[1,0,2,3]);steps=T[.1,.2,.3,.4]
        @test prox(f,x,steps)≈sign.(x).*max.(abs.(x).-T(.3).*f.w.*steps,zero(T))
        @test prox_block(f,x[2],2,γ)==x[2]
        @test restrict(f,[4,1]).w==f.w[[4,1]]
        @test kkt_violation(NormL1(T(.3)),T[.2,-.7],2)≈T(.4)
        layout=BlockIterate(T,4,[Int[4,1],Int[3,2]])
        grad=T[.1,.2,4,8]
        @test kkt_violation(NormL1(one(T)),grad,layout,1)≈T(7)
        @test kkt_violation(Scaled(NormL1(one(T)),T(2)),grad,layout,1)≈T(6)
        @test kkt_violation(NormL1(one(T)),grad,layout,2)≈T(3)
        @test PB.Squared()(T(3),T(1))==T(2)
        @test PB.value_and_deriv(PB.Squared(),T(3),T(1))==(T(2),T(2))
    end
    @testset "$T least squares and lasso building blocks" begin
        X=T[1 2;3 1;0 2;2 -1];Y=T[1 -1;2 .5;3 1;-.5 2];β=T[.2 -.4;.3 .1]
        f=LeastSquaresLoss(Y,X);residual=X*β-Y
        @test f(β)≈sum(abs2,residual)/2
        @test gradient(f,β)≈X'*residual
        γ=T(.3);@test prox(f,β,γ)≈(I+γ*X'*X)\(β+γ*X'*Y)
        loss=(one(T)/size(X,1))*LeastSquaresLoss(Y[:,1],X)
        penalty=NormL1(T(.2));v=T[.3,-.4]
        @test loss(v)+penalty(v)≈sum(abs2,X*v-Y[:,1])/(2size(X,1))+T(.2)*sum(abs,v)
        @test gradient(loss,v)≈X'*(X*v-Y[:,1])/size(X,1)
    end
    @testset "$T extension and symmetric preflight" begin
        f=WeightedSquare(T[1 2;2 3]);s=SymmetricSparseIterate(T[1 2;2 0]);snapshot=copy(s)
        @test f(s)≈sum(f.w.*Matrix(s).^2)/2
        @test Matrix(prox(f,s,T(.5)))≈Matrix(s)./(1 .+T(.5).*f.w)
        # asymmetric weights are an invalid model, rejected by the explicit validation
        bad=WeightedSquare(T[1 2;3 1]);out=copy(s)
        @test PB.check_symmetric_parameters(f,s)
        @test_throws ArgumentError PB.check_symmetric_parameters(bad,s)
        @test_throws ArgumentError prox!(out,NormL1(),s,fill(one(T),2,2))
        @test Matrix(out)==Matrix(snapshot) && collect(activecoordinates(out))==collect(activecoordinates(snapshot))
        x=BlockIterate(T[1,0,2,-1],[Int[3,1],Int[4,2]])
        before=Vector(x);u=collect(blockview(x,1));z=prox_block(CompactSquare(),u,1,T(.5))
        @test z≈u/T(1.5) && z!==u
        @test Vector(x)==before
        z[1]=T(10);@test Vector(x)==before
    end
end
end
