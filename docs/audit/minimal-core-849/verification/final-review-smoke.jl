using ProximalBase, LinearAlgebra, Random, Test
BLAS.set_num_threads(1)
Random.seed!(850)
const PB = ProximalBase
struct ReviewerSquare{A} <: SeparableFunction
    w::A
end
PB.value_entry(f::ReviewerSquare,v,i)=f.w[i]*v^2/2
PB.prox_block(f::ReviewerSquare,v::Real,i::Integer,g::Real)=v/(1+g*f.w[i])
PB.iszeropreserving(::ReviewerSquare)=true
PB.entry_parameters(f::ReviewerSquare)=(:w=>f.w,)
PB.check_parameters(f::ReviewerSquare,x)=length(f.w)==prod(size(x)) ? nothing : throw(DimensionMismatch())
@testset "Independent final review #850" begin
    for T in (Float32,Float64)
        tol=T===Float32 ? 5e-5 : 5e-13
        for (n,p) in ((5,3),(3,5),(0,3),(3,0)), responses in (1,3)
            A=randn(T,n,p);Y=responses==1 ? randn(T,n) : randn(T,n,responses)
            v=responses==1 ? randn(T,p) : randn(T,p,responses)
            f=LeastSquaresLoss(Y,A);g=T(.4)
            expected=(I+g*A'*A)\(v+g*A'*Y)
            out=prox(f,v,g)
            @test out≈expected rtol=tol atol=tol
            alias=copy(v);prox!(alias,f,alias,g)
            @test alias≈expected rtol=tol atol=tol
            @test value_and_prox!(alias,f,v,g)≈f(alias) rtol=tol atol=tol
            if n*p>0
                A[1,1]+=T(.5);Y[1]+=T(.25);reset_cache!(f)
                @test prox(f,v,g)≈(I+g*A'*A)\(v+g*A'*Y) rtol=tol atol=tol
            end
        end
        groups=[[4,1],[3,2]];x=BlockIterate(T,4,groups);g=T[.2,5,-2,4]
        f=Scaled(NormL1(T(.3),T[1,2,3,0]),T(1.5))
        for k in 1:2
            @test kkt_violation(f,g,x,k)≈maximum(abs(g[i])-T(1.5)*T(.3)*T[1,2,3,0][i] for i in groups[k]) rtol=tol
        end
        a=T[1 2;2 0];s=SymmetricSparseIterate(a);mv=matrixview(s)
        rejected=try mv[2,1]=T(5);false catch;true end
        @test rejected
        @test Matrix(s)==a
        s[2,1]=T(3);@test mv[1,2]==T(3)
        w=T[1 3;3 2];cf=Scaled(ReviewerSquare(w),T(.7));before=Matrix(s);out=copy(s)
        @test value_and_prox!(out,cf,out,T(.4))≈cf(out) rtol=tol
        @test Matrix(out)≈before./(1 .+T(.4)*T(.7).*w) rtol=tol
        @test Matrix(s)==before
        bad=Scaled(ReviewerSquare(T[1 2;3 4]),T(.7));snapshot=copy(out)
        @test_throws ArgumentError prox!(out,bad,out,T(.4))
        @test Matrix(out)==Matrix(snapshot)
        @test collect(activecoordinates(out))==collect(activecoordinates(snapshot))
    end
end
