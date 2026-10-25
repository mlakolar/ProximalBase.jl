# Benchmark-only alternatives. These do not change package dispatch or defaults.
include(joinpath(@__DIR__, "..", "campaign", "run.jl"))

struct CholPrototype{T,F}
    X::Matrix{T}
    y::Vector{T}
    gram::Matrix{T}
    factor::F
    linear::Vector{T}
    residual::Vector{T}
end

function chol_prototype(X::Matrix{T}, y, γ) where T
    tall = size(X,1) >= size(X,2)
    gram = tall ? X'X : X*X'
    A = γ .* gram
    for i in axes(A,1); A[i,i] += 1; end
    F = cholesky!(Symmetric(A))
    CholPrototype(X,y,gram,F,X'y,similar(y))
end

function chol_prox!(out, c::CholPrototype, x, γ)
    if size(c.X,1) >= size(c.X,2)
        @. out = x + γ*c.linear
        ldiv!(c.factor,out)
    else
        mul!(c.residual,c.X,x)
        c.residual .-= c.y
        ldiv!(c.factor,c.residual)
        mul!(out,c.X',c.residual)
        @. out = x - γ*out
    end
    out
end

function chol_refactor!(c::CholPrototype, γ)
    A = c.factor.factors
    @. A = γ*c.gram
    for i in axes(A,1); A[i,i] += 1; end
    cholesky!(Symmetric(A))
    nothing
end

function ls_case!(cases,X,y,x,γ,steps,method,state)
    T=eltype(X);m,n=size(X)
    out=similar(x)
    f=LeastSquaresLoss(y,X)
    c=chol_prototype(X,y,γ)
    warm = method=="ThinSVD" ? ()->prox!(out,f,x,γ) : ()->chol_prox!(out,c,x,γ)
    cold = method=="ThinSVD" ? ()->prox!(out,LeastSquaresLoss(y,X),x,γ) :
                             ()->chol_prox!(out,chol_prototype(X,y,γ),x,γ)
    function trace!()
        for step in steps
            if method=="ThinSVD"
                prox!(out,f,x,step)
            else
                chol_refactor!(c,step);chol_prox!(out,c,x,step)
            end
        end
    end
    op=state=="fixed_step" ? warm : state=="first_prox" ? cold : trace!
    step=state=="four_changing_steps" ? last(steps) : γ
    check!()=(op();close_enough(out,prox(LeastSquaresLoss(y,X),x,step)))
    addcase!(cases,"ls_backend","$method/$state",op,check!;T,
            shape="$(m)x$n",zero_alloc=false,work=(state=="four_changing_steps" ? 4 : 1)*m*n,
            profile=T==Float64 && n==256)
end

function ls_study!(cases,rep)
    for T in (Float32,Float64), (m,n) in ((128,32),(1024,256),(32,128),(128,1024))
        rng=scenario_rng(rep,"ls_backend/$T/$m/$n")
        X,y,x=randn(rng,T,m,n),randn(rng,T,m),randn(rng,T,n)
        γ=T(0.7); steps=T[0.1,0.3,0.7,1.2]
        for method in ("ThinSVD","Cholesky"), state in ("fixed_step","first_prox","four_changing_steps")
            ls_case!(cases,X,y,x,γ,steps,method,state)
        end
    end
end

function sort_kernel!(f,x::AbstractVector{T},alg) where T
    n=length(x)
    a=ProximalBase._buffer!(f.sums,T,n)
    p=ProximalBase._buffer!(f.perm,Int,n)
    @inbounds for i in 1:n; a[i]=abs(x[i]);p[i]=i;end
    sortperm!(p,a;rev=true,initialized=true,alg,scratch=ProximalBase._buffer!(f.sizes,Int,n))
    nothing
end

baseline_sort!(f,x::AbstractVector{T}) where T = (ProximalBase._sortperm_abs!(f,x,T);nothing)

function slope_study!(cases,rep)
    for T in (Float32,Float64), n in (256,4096,100_000)
        rng=scenario_rng(rep,"slope_sort/$T/$n")
        x=randn(rng,T,n)
        for (pattern,input) in (("normal",x),("ties",round.(x)),("sorted",sort(abs.(x);rev=true)))
            expected=sort(abs.(input);rev=true)
            for (name,alg) in (("BaseDefault",nothing),("QuickSort",QuickSort),("MergeSort",MergeSort))
                f=SLOPE(collect(range(T(0.5),T(0.1);length=n)))
                op = alg===nothing ? ()->baseline_sort!(f,input) :
                                     ()->sort_kernel!(f,input,alg)
                function check!()
                    op()
                    p=ProximalBase._buffer!(f.perm,Int,n)
                    a=ProximalBase._buffer!(f.sums,T,n)
                    close_enough(a[p],expected)
                    sort(p)==collect(1:n) || error("not a permutation")
                end
                addcase!(cases,"slope_sort","$name/$pattern",op,check!;T,shape=string(n),
                    zero_alloc=false,profile=T==Float64 && n==100_000,evals=1,work=n)
            end
        end
    end
end

function remaining_cases(rep)
    cases=PerfCase[]
    ls_study!(cases,rep);slope_study!(cases,rep)
    cases
end
main(remaining_cases)
