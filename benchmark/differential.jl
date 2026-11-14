using LinearAlgebra, Random, Test
BLAS.set_num_threads(1)
module Baseline end
module Candidate end
Base.include(Baseline,joinpath(get(ENV,"PB_BASELINE","/tmp/pb849-baseline"),"src","ProximalBase.jl"))
Base.include(Candidate,joinpath(get(ENV,"PB_CANDIDATE","/tmp/pb849-minimal"),"src","ProximalBase.jl"))
function differential(PB,::Type{T}) where T
    rng=Xoshiro(849);tol=T===Float32 ? 3e-5 : 3e-13
    @testset "$PB $T" begin
        for family in (:sparse,:symmetric,:block)
            p=31; groups=[collect(i:7:p) for i in 1:7]
            x=family===:symmetric ? PB.SymmetricSparseIterate(T,8) : family===:sparse ? PB.SparseIterate(T,p) : PB.BlockIterate(T,p,groups)
            c=family===:block ? x : PB.coordinates(x);q=length(c);ref=zeros(T,q)
            logical()=family===:symmetric ? Matrix(x) : Vector(x)
            dense_ref()=if family===:symmetric
                matrixref=zeros(T,8,8);k=0
                for col in 1:8,row in col:8
                    k+=1;matrixref[row,col]=matrixref[col,row]=ref[k]
                end
                matrixref
            else
                copy(ref)
            end
            # Includes absent zero, repeated writes, signed zero, first activation and retained zero.
            activated=Set{Int}()
            for j in 1:500
                i=rand(rng,1:q);v=j%5==0 ? zero(T) : T(randn(rng))
                c[i]=v;ref[i]=v
                if !iszero(v)
                    push!(activated,family===:block ? findfirst(g->i in g,groups) : i)
                end
                @test collect(c)==ref
                @test logical()==dense_ref()
            end
            @test sort(collect(PB.activecoordinates(x)))==sort(collect(activated))
            PB.dropzeros!(x)
            @test logical()==dense_ref()
            @test PB.nactive(x)==(family===:block ? count(g->any(!iszero,ref[g]),groups) : count(!iszero,ref))
            if family===:symmetric
                for row in 1:8,col in 1:8
                    @test x[row,col]==dense_ref()[row,col]
                end
                for row in 1:8
                    x[row,row]=T(row);ref[PB.tril_index(8,row,row)]=T(row)
                end
                @test Matrix(x)==dense_ref()
            end
            y=copy(x);@test (family===:symmetric ? Matrix(y) : Vector(y))==dense_ref()
            fill!(family===:block ? y : PB.coordinates(y),zero(T));copyto!(y,x)
            @test (family===:symmetric ? Matrix(y) : Vector(y))==dense_ref()
            copyto!(y,dense_ref());@test (family===:symmetric ? Matrix(y) : Vector(y))==dense_ref()
            y32=similar(x,Float32);copyto!(y32,x)
            @test (family===:symmetric ? Matrix(y32) : Vector(y32))==Float32.(dense_ref())
            if family===:symmetric
                b=randn(rng,T,8);out=zeros(T,8)
                mul!(out,x,b);@test out≈dense_ref()*b rtol=tol atol=tol
                @test x*b≈dense_ref()*b rtol=tol atol=tol
                B=randn(rng,T,8,5);C=zeros(T,8,5)
                mul!(C,x,B);@test C≈dense_ref()*B rtol=tol atol=tol
                @test dot(b,x,b)≈dot(b,dense_ref(),b) rtol=tol atol=tol
            else
                A=randn(rng,T,19,q);out=zeros(T,19)
                mul!(out,A,x);@test out≈A*ref rtol=tol atol=tol
                @test A*x≈A*ref rtol=tol atol=tol
                Y=randn(rng,T,19);f=PB.LeastSquaresLoss(Y,A);g=zeros(T,q)
                @test f(x)≈sum(abs2,A*ref-Y)/2 rtol=tol
                PB.gradient!(g,f,x);@test g≈A'*(A*ref-Y) rtol=tol atol=tol
                @test PB.value_and_gradient!(g,f,x)≈sum(abs2,A*ref-Y)/2 rtol=tol
                @test g≈A'*(A*ref-Y) rtol=tol atol=tol
            end
            penalty=PB.NormL1(T(.3));out=similar(x)
            PB.prox!(out,penalty,x,T(.7))
            expected=sign.(dense_ref()).*max.(abs.(dense_ref()).-T(.3)*T(.7),zero(T))
            @test (family===:symmetric ? Matrix(out) : Vector(out))≈expected rtol=tol atol=tol
            @test penalty(x)≈T(.3)*sum(abs,dense_ref()) rtol=tol
        end
        for groups in ([collect(1:2:20),collect(2:2:20)],[collect(10:-1:1),collect(20:-1:11)],[1:10,11:20])
            x=PB.BlockIterate(T,20,groups);ref=zeros(T,20)
            grad=randn(rng,T,20);weights=T[1+i/20 for i in 1:20]
            penalty=PB.NormL1(T(.3),weights)
            for k in eachindex(groups)
                expected=maximum(abs(grad[i])-T(.3)*weights[i] for i in groups[k])
                @test PB.kkt_violation(penalty,grad,x,k)≈expected rtol=tol atol=tol
                scaled=PB.Scaled(penalty,T(1.2))
                @test PB.kkt_violation(scaled,grad,x,k)≈maximum(abs(grad[i])-T(1.2)*T(.3)*weights[i] for i in groups[k]) rtol=tol atol=tol

            end
            for k in (2,1,2)
                u=randn(rng,T,10);u[1]=zero(T)
                copyto!(PB.blockview(x,k),u);ref[groups[k]].=u
                @test Vector(x)==ref
                @test collect(PB.blockview(x,k))==u
                @test collect(PB.blockindices(x,k))==groups[k]
            end
        end
        f=PB.NormL1(T(.3))
        for u in T[0,-0.,.1,-.1,2,-3,1e-20,1e20],γ in T[0,.7,2]
            expected=sign(u)*max(abs(u)-γ*T(.3),zero(T))
            @test PB.prox_block(f,u,1,γ)≈expected rtol=tol atol=tol
        end
        # Multi-response least squares, including sparse matrix coefficients.
        A=randn(rng,T,13,9);Y=randn(rng,T,13,3);β=randn(rng,T,9,3);β[abs.(β).<T(.7)].=zero(T)
        for x in (β,PB.SparseIterate(β))
            f=PB.LeastSquaresLoss(Y,A);g=similar(β);r=A*β-Y
            @test f(x)≈sum(abs2,r)/2 rtol=tol
            PB.gradient!(g,f,x);@test g≈A'*r rtol=tol atol=tol
            @test PB.value_and_gradient!(g,f,x)≈sum(abs2,r)/2 rtol=tol
        end
        # Independent arbitrary-precision centered ridge prox/gradient/value,
        # including γλ overflow and tiny but finite levels and large centers.
        E=T===Float32 ? 100 : 800
        for λ in (T(.3),ldexp(one(T),-E),ldexp(one(T),E)),γ in (T(.7),ldexp(one(T),E))
            v=T[2,-3];c=T[.5,-.25];out=similar(v);g=similar(v)
            f=PB.SqrNormL2(λ,c)
            PB.prox!(out,f,v,γ)
            expected=setprecision(BigFloat,256) do
                den=1+BigFloat(γ)*BigFloat(λ)
                T.((BigFloat.(v).+BigFloat(γ)*BigFloat(λ).*BigFloat.(c))./den)
            end
            @test out≈expected rtol=tol atol=zero(T)
            PB.gradient!(g,f,v)
            expectedg=T.(BigFloat(λ).*(BigFloat.(v).-BigFloat.(c)))
            @test g≈expectedg rtol=tol atol=zero(T)
            expectedvalue=T(BigFloat(λ)*sum(abs2,BigFloat.(v).-BigFloat.(c))/2)
            @test f(v)≈expectedvalue rtol=tol
        end
        λ=ldexp(one(T),-E);v=T[ldexp(one(T),E)];c=-v;f=PB.SqrNormL2(λ,c);g=similar(v)
        PB.gradient!(g,f,v)
        @test g≈T.(BigFloat(λ).*(BigFloat.(v).-BigFloat.(c))) rtol=tol
        @test f(v)≈T(BigFloat(λ)*sum(abs2,BigFloat.(v).-BigFloat.(c))/2) rtol=tol
        for scale in (T(.5),ldexp(one(T),-E)),λ in (T(.3),ldexp(one(T),E))
            v=T[2,-3];γ=T(.7);f=PB.Scaled(PB.NormL1(λ),scale);out=similar(v)
            PB.prox!(out,f,v,γ)
            expected=setprecision(BigFloat,256) do
                T.(sign.(BigFloat.(v)).*max.(abs.(BigFloat.(v)).-BigFloat(γ)*BigFloat(λ)*BigFloat(scale),BigFloat(0)))
            end
            @test out≈expected rtol=tol atol=zero(T)
            expectedvalue=T(BigFloat(scale)*BigFloat(λ)*sum(abs,BigFloat.(v)))
            @test f(v)≈expectedvalue rtol=tol
        end
        # Independently compute representable least-squares values with BigFloat,
        # covering ordinary, underflowed squares and overflowed intermediate squares.
        for scale in (one(T),sqrt(floatmin(T))/T(4),sqrt(floatmax(T))/T(4))
            A=T[1 0;0 1];Y=zeros(T,2);β=T[scale,-scale]
            f=PB.LeastSquaresLoss(Y,A)
            expected=T(setprecision(BigFloat,256) do
                sum(abs2,BigFloat.(β))/2
            end)
            @test f(β)≈expected rtol=tol atol=zero(T)
        end
    end
end
for PB in (Baseline.ProximalBase,Candidate.ProximalBase),T in (Float32,Float64)
    differential(PB,T)
end
