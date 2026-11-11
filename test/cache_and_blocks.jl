module CacheAndBlocksTest
using Test, ProximalBase, LinearAlgebra
for T in (Float32,Float64)
    @testset "$T retained cache lifecycle" begin
        X=T[1 2;3 0;0 1];Y=T[1,2,3];v=T[.2,-.4];γ=T(.3)
        f=LeastSquaresLoss(Y,X);prox(f,v,γ)
        X[1,1]=T(2);Y[2]=T(-1)
        @test reset_cache!(f)===f
        @test prox(f,v,γ)≈prox(LeastSquaresLoss(copy(Y),copy(X)),v,γ)
        wrapper=Scaled(f,T(.5));prox(wrapper,v,γ);X[2,2]=T(1)
        @test reset_cache!(wrapper)===wrapper
        @test prox(wrapper,v,γ)≈prox(Scaled(LeastSquaresLoss(copy(Y),copy(X)),T(.5)),v,γ)
        A=T[2 .2;.2 3];q=QuadraticFunction(A);prox(q,v,γ);A[1,1]=T(4)
        @test reset_cache!(q)===q
        @test prox(q,v,γ)≈prox(QuadraticFunction(copy(A)),v,γ)
        z=ZeroFunction();@test reset_cache!(z)===z
    end
    @testset "$T block coordinate KKT" begin
        g=T[.1,.2,4,8];w=T[1,2,.5,3];λ=T(.5)
        for groups in ([Int[4,1],Int[3,2]],[Int[1,4],Int[2,3]])
            x=BlockIterate(T,4,groups)
            for k in eachindex(groups), f in (NormL1(λ,w),Scaled(NormL1(λ,w),T(2)))
                scale=f isa Scaled ? T(2) : one(T)
                @test kkt_violation(f,g,x,k)≈maximum(abs(g[i])-scale*λ*w[i] for i in groups[k])
            end
        end
    end
end
end
