# Case definitions are independent of the scheduler and of measurement settings.
# Random data use stable scenario keys, so adding a scenario does not change others.
struct PerfCase{F,S,V,M}
    meta::M
    operation::F
    prepare::S
    validate::V
    evals::Int
    profile::Bool
end

function scenario_rng(rep, key)
    seed = UInt64(0xcbf29ce484222325) ⊻ UInt64(rep)
    for b in codeunits(key)
        seed = (seed ⊻ UInt64(b)) * UInt64(0x100000001b3)
    end
    Xoshiro(seed)
end

noop() = nothing
tolerance(::Type{Float32}) = 5e-4
tolerance(::Type{Float64}) = 2e-8
dense(x) = Array(x)

function close_enough(x, y)
    t = tolerance(float(eltype(x)))
    isapprox(dense(x), dense(y); atol=t, rtol=t) || error("output differs from dense reference")
end

function addcase!(cases, family, name, op, validate; T=Float64, shape="", layout="dense",
                  phase="warm", prepare=noop, evals=1, profile=false, work=0,
                  zero_alloc=phase == "warm")
    id = join((family, name, string(T), shape, layout, phase), "/")
    meta = (;id, family, name, dtype=string(T), shape, layout, phase, work, zero_alloc)
    push!(cases, PerfCase(meta, op, prepare, validate, evals, profile))
end

function addprox!(cases, family, name, f, x; γ=eltype(x)(0.7), layout="dense",
                  prepare=noop, profile=false, alias=false, value=true, reference_function=f)
    T = eltype(x)
    evals = length(x)<=512 && prepare===noop ? 25 : 1
    y = similar(x)
    reference = Ref{Any}(nothing)
    # Reference construction and checks are deliberately outside all measurements.
    function check!()
        r = prox(reference_function, dense(x), γ)
        reference[] = r
        prepare(); prox!(y, f, x, γ)
        close_enough(y, r)
        all(isfinite, y) || error("nonfinite proximal output")
        nothing
    end
    addcase!(cases, family, name * "/prox!", () -> prox!(y, f, x, γ), check!;
             T, shape=join(size(x), "x"), layout, prepare, profile, evals, work=length(x))
    if value
        function checkvalue!()
            check!()
            prepare(); v = value_and_prox!(y, f, x, γ)
            close_enough(y, reference[])
            expected = f(y)
            t = tolerance(T)
            (v == expected || isapprox(v, expected; atol=t, rtol=t)) ||
                error("returned value differs from f(prox): $v versus $expected")
        end
        addcase!(cases, family, name * "/value_and_prox!", () -> value_and_prox!(y, f, x, γ), checkvalue!;
                 T, shape=join(size(x), "x"), layout, prepare, evals, work=length(x))
    end
    if alias
        z = copy(x)
        prep_alias!() = copyto!(z, x)
        check_alias!() = (prep_alias!(); prox!(z, f, z, γ); close_enough(z, prox(f, dense(x), γ)))
        addcase!(cases, family, name * "/prox!_alias", () -> prox!(z, f, z, γ), check_alias!;
                 T, shape=join(size(x), "x"), layout, prepare=prep_alias!, work=length(x))
    end
end

function addcold!(cases, family, name, makef, x; γ=eltype(x)(0.7))
    y = similar(x)
    op() = prox!(y, makef(), x, γ)
    check!() = (op(); close_enough(y, prox(makef(), dense(x), γ)))
    addcase!(cases, family, name, op, check!; T=eltype(x), shape=join(size(x), "x"),
             phase="construct_and_first_prox", profile=eltype(x)==Float64, work=length(x))
end

function addgrad!(cases, family, name, f, x; profile=false, second=true)
    T = eltype(x)
    g = zeros(T, size(x))
    v = fill(T(inv(sqrt(length(x)))), size(x))
    hv = similar(g)
    function checkgrad!()
        val = value_and_gradient!(g, f, x)
        t = tolerance(T)
        isapprox(val, f(x); atol=t, rtol=t) || error("value/gradient objective mismatch")
        all(isfinite, g) || error("nonfinite gradient")
        if T == Float64
            δ = 1e-5
            fd = (f(dense(x) .+ δ .* v) - f(dense(x) .- δ .* v)) / (2δ)
            isapprox(dot(g, v), fd; atol=1e-5, rtol=1e-4) || error("directional gradient check failed")
        end
    end
    shape = join(size(x), "x")
    addcase!(cases, family, name * "/value", () -> f(x), checkgrad!; T, shape, work=length(x))
    addcase!(cases, family, name * "/gradient!", () -> gradient!(g, f, x), checkgrad!; T, shape, profile, work=length(x))
    addcase!(cases, family, name * "/value_and_gradient!", () -> value_and_gradient!(g, f, x), checkgrad!; T, shape, work=length(x))
    if second && hashessian(f)
        function checkhess!()
            hessian_vec!(hv, f, x, v)
            δ = T == Float32 ? T(0.01) : T(1e-5)
            gp = gradient(f, dense(x) .+ δ .* v)
            gm = gradient(f, dense(x) .- δ .* v)
            isapprox(hv, (gp .- gm) ./ (2δ); atol=T==Float32 ? 0.02 : 1e-5,
                     rtol=T==Float32 ? 0.02 : 1e-4) || error("Hessian direction check failed")
        end
        addcase!(cases, family, name * "/hessian_vec!", () -> hessian_vec!(hv, f, x, v), checkhess!;
                 T, shape, work=length(x))
        q = quadratic_model(f, x)
        checkmodel!() = (quadratic_model!(q, f, x); close_enough(gradient(q, x), gradient(f, x)))
        addcase!(cases, family, name * "/quadratic_model!", () -> quadratic_model!(q, f, x), checkmodel!;
                 T, shape, work=length(x))
    end
end

function separable_cases!(cases, rep)
    for T in (Float32, Float64), n in (256, 100_000)
        rng = scenario_rng(rep, "separable/$T/$n")
        x, w = randn(rng, T, n), rand(rng, T, n) .+ T(0.2)
        fs = (NormL1(T(0.3)), NormL1(T(0.3), w), NormL1Nonneg(T(0.3)),
              ElasticNet(T(0.3), T(0.4)), IndBox(T(-1), T(1)), IndNonnegative(),
              NormL0(T(0.1)), MCP(T(0.3)), SCAD(T(0.3)), NegEntropy(T(0.2)),
              KLDivergence(w; λ=T(0.2)), NegLog(T(0.2)), Berhu(T(0.3), T(1)),
              CheckLoss(T(0.3)), HingeLoss(), EpsInsensitiveLoss(T(0.2)),
              CappedL1(T(0.3), T(1)), LogPenalty(T(0.3), T(0.5)), NormLq(T(0.3), T(0.5)))
        for (k, f) in enumerate(fs)
            name = string(nameof(typeof(f)), k == 2 ? "_weighted" : "")
            addprox!(cases, "separable", name, f, x; profile=T==Float64 && n==100_000 && k in (1,9,10,19), alias=k in (1,4))
        end
        sx = SparseIterate(sprandn(rng, T, n, 0.01))
        for f in (NormL1(T(0.3)), ElasticNet(T(0.3), T(0.4)), NormL0(T(0.1)), MCP(T(0.3)))
            addprox!(cases, "sparse_penalties", string(nameof(typeof(f))), f, sx; layout="sparse_1pct", profile=T==Float64 && n==100_000)
        end
        steps = T(0.2) .+ rand(rng, T, n)
        for f in (NormL1(T(0.3)), ElasticNet(T(0.3), T(0.4)))
            addprox!(cases, "metric", string(nameof(typeof(f))), f, x; γ=steps, layout="diagonal_step")
        end
        xv = view(randn(rng, T, 2n), 1:2:2n)
        addprox!(cases, "layouts", "NormL1", NormL1(T(0.3)), xv; layout="stride2_input", alias=true)
        for f in (NormL2(T(0.3)), NormLinf(T(0.3)), IndBallL0(max(1,n÷10)),
                  IndBallL1(T(n/10)), IndBallL2(T(1)), IndSphereL2(T(1)), IndSimplex(T(1)),
                  IndHyperplane(one(T), zero(T)), IndHalfspace(one(T), zero(T)))
            addprox!(cases, "norms_sets", string(nameof(typeof(f))), f, x;
                     profile=T==Float64 && n==100_000 && f isa Union{NormLinf,IndSimplex})
        end
        for f in (TotalVariation1D(T(0.4)), FusedLasso(T(0.2),T(0.4)), IndIsotonic(),
                  SLOPE(collect(range(T(0.5),T(0.1);length=n))))
            addprox!(cases, "ordered", string(nameof(typeof(f))), f, x; profile=T==Float64 && n==100_000)
        end
        groups = [j:min(j+7,n) for j in 1:8:n]
        weights = ones(T, length(groups))
        for f in (GroupNormL2(T(0.3),groups,weights), GroupNormLinf(T(0.3),groups,weights),
                  SparseGroupNormL2(T(0.2),T(0.3),groups,weights),
                  GroupMCP(T(0.3),groups,weights), GroupSCAD(T(0.3),groups,weights))
            addprox!(cases, "groups", string(nameof(typeof(f))), f, x; profile=T==Float64 && n==100_000 && f isa SparseGroupNormL2)
            if f isa Union{GroupNormL2,GroupNormLinf,SparseGroupNormL2}
                xa = copy(x); xa[1:n÷2] .= 0
                atomx = AtomIterate(xa,f)
                addprox!(cases, "groups", string(nameof(typeof(f))), f, atomx; layout="atom_half_active")
                z = copy(atomx)
                prep!() = copyto!(z,atomx)
                check!() = (prep!(); prox_coordinate!(z,f,length(groups),T(0.7)); all(isfinite,z) || error("nonfinite group update"))
                addcase!(cases, "coordinates", string(nameof(typeof(f))), () -> prox_coordinate!(z,f,length(groups),T(0.7)), check!;
                         T, shape=string(n), layout="atom", prepare=prep!, work=length(last(groups)))
            end
        end
    end
end

function smooth_cases!(cases, rep)
    for T in (Float32,Float64), (n,p) in ((128,32),(1024,256))
        rng = scenario_rng(rep,"smooth/$T/$n/$p")
        X,y,x = randn(rng,T,n,p),randn(rng,T,n),T(0.03).*randn(rng,T,p)
        for (name,f) in (("least_squares",LeastSquaresLoss(y,X)), ("logistic",LogisticLoss(rand(rng,0:1,n),X)),
                  ("poisson",PoissonLoss(rand(rng,0:3,n),X)), ("huber",HuberLoss(y,X,T(0.5))),
                  ("logistic_weighted_offset",LogisticLoss(rand(rng,0:1,n),X;weights=rand(rng,T,n),offset=randn(rng,T,n))))
            addgrad!(cases,"smooth",name,f,x; profile=T==Float64 && n==1024)
        end
        ls = LeastSquaresLoss(y,X)
        addprox!(cases,"least_squares","tall",ls,x;profile=T==Float64 && n==1024)
        addcold!(cases,"least_squares","tall",() -> LeastSquaresLoss(y,X),x)
        B = randn(rng,T,p,3); Y = randn(rng,T,n,3)
        addgrad!(cases,"least_squares_multi","three_responses",LeastSquaresLoss(Y,X),B)
        L = zeros(T,p)
        checkL!() = (coordinate_lipschitz!(L,ls); close_enough(L,vec(sum(abs2,X;dims=1))))
        addcase!(cases,"coordinates","least_squares_curvatures",() -> coordinate_lipschitz!(L,ls),checkL!;T,shape="$(n)x$p",work=n*p)
        G = zeros(T,p); penalty=NormL1(T(0.3)); step=T(0.1)/sum(abs2,X)
        checkG!() = (gradient_mapping!(G,ls,penalty,x,step);all(isfinite,G) || error("nonfinite gradient mapping"))
        addcase!(cases,"calculus","gradient_mapping!",() -> gradient_mapping!(G,ls,penalty,x,step),checkG!;T,shape="$(n)x$p",work=n*p)
        H=Matrix(Symmetric(X'X+I)); q=QuadraticFunction(H,randn(rng,T,p),zero(T))
        addgrad!(cases,"quadratic","dense",q,x)
        addprox!(cases,"quadratic","dense",q,x)
        addcold!(cases,"quadratic","dense",() -> QuadraticFunction(H,zeros(T,p),zero(T)),x)
        for f in (Regularize(ls,T(0.3)), Precompose(SqrNormL2(T(0.3)),T(-2)),
                  Translate(SqrNormL2(T(0.3)),ones(T,p)), Sum(ls,SqrNormL2(T(0.3))))
            addgrad!(cases,"compositions",string(nameof(typeof(f))),f,x)
        end
        # An actual fixed-work solver trace, with reset outside the measured block.
        β=zeros(T,p); grad=similar(β); r=copy(y); a=vec(sum(abs2,X;dims=1))
        prep_cd!() = (fill!(β,0);copyto!(r,y))
        function cdtrace!()
            for sweep in 1:5, j in 1:p
                old=β[j]; col=view(X,:,j)
                new=prox_entry(penalty,old+dot(col,r)/a[j],j,inv(a[j]))
                β[j]=new; axpy!(old-new,col,r)
            end
            β
        end
        checkcd!() = (prep_cd!();cdtrace!(); close_enough(r,y-X*β); ls(β)+penalty(β)<=sum(abs2,y)/2+1e-3 || error("CD objective increased"))
        addcase!(cases,"solver_traces","lasso_5_sweeps",cdtrace!,checkcd!;T,shape="$(n)x$p",prepare=prep_cd!,profile=T==Float64 && n==1024,work=5n*p)
    end
    for T in (Float32,Float64), (n,p) in ((32,128),(128,1024))
        rng=scenario_rng(rep,"wide/$T/$n/$p")
        X,y,x=randn(rng,T,n,p),randn(rng,T,n),randn(rng,T,p)
        f=LeastSquaresLoss(y,X)
        addprox!(cases,"least_squares","wide",f,x;profile=T==Float64 && p==1024)
        addcold!(cases,"least_squares","wide",() -> LeastSquaresLoss(y,X),x)
        sparsex=SparseIterate(sprandn(rng,T,p,0.01))
        addgrad!(cases,"sparse_smooth","least_squares",f,sparsex;second=false)
        A=randn(rng,T,4,p); b=zeros(T,4); C=IndAffine(A,b)
        constrained=Constrained(f,C)
        addprox!(cases,"affine_constrained","least_squares",constrained,x)
        addcold!(cases,"affine_constrained","least_squares",() -> Constrained(LeastSquaresLoss(y,X),C),x)
    end
end

function matrix_cases!(cases, rep)
    for T in (Float32,Float64), p in (32,256)
        rng=scenario_rng(rep,"matrices/$T/$p")
        M=randn(rng,T,p,p÷2); A=Matrix(Symmetric(randn(rng,T,p,p)))
        for f in (NuclearNorm(T(0.5)), IndBallRank(4), SingularValue(NormL1(T(0.5))))
            addprox!(cases,"svd",string(nameof(typeof(f))),f,M;profile=T==Float64 && p==256)
        end
        for f in (IndPSD(),IndPSD(rank=4),TracePSD(T(0.2)),Spectral(NormL1(T(0.2))))
            addprox!(cases,"eigen",string(nameof(typeof(f)),f isa IndPSD && f.rank !== nothing ? "_rank4" : ""),f,A;
                     profile=T==Float64 && p==256)
        end
        for f in (IndPattern(rand(rng,p,p).>0.8), IndBanded(3,p), IndToeplitz(symmetric=true))
            addprox!(cases,"matrix_sets",string(nameof(typeof(f))),f,A;profile=T==Float64 && p==256)
        end
        B=randn(rng,T,2p,p); S=Matrix(Symmetric(B'B/(2p)+I)); P=Matrix(T(1)I,p,p)
        addgrad!(cases,"logdet","positive_definite",LogDetLoss(S),P;profile=T==Float64 && p==256)
        addprox!(cases,"logdet","positive_definite",LogDetLoss(S),P)
        kron=KronQuadratic(S,Matrix(T(1)I,4,4),randn(rng,T,p,4))
        addgrad!(cases,"quadratic","kron",kron,randn(rng,T,p,4);profile=T==Float64 && p==256)
        sp=SymmetricSparseIterate(A .* (abs.(A).>T(1.5)))
        addprox!(cases,"sparse_penalties","symmetric_NormL1",NormL1(T(0.3)),sp;layout="symmetric_sparse")
    end
    for T in (Float32,Float64), p in (128,512)
        rng=scenario_rng(rep,"partial/$T/$p")
        U=Matrix(qr(randn(rng,T,p,4)).Q)[:,1:4]
        V=Matrix(qr(randn(rng,T,p÷2,4)).Q)[:,1:4]
        M=U*Diagonal(T[20,16,13,10])*V' + T(0.001).*randn(rng,T,p,p÷2)
        A=U*Diagonal(T[20,16,13,10])*U' + T(0.001).*Matrix(Symmetric(randn(rng,T,p,p)))
        for (name,f,x) in (("nuclear",NuclearNorm(T(0.5);partial=true),M),
                           ("rank4",IndBallRank(4;partial=true),M),
                           ("psd_rank4",IndPSD(rank=4,partial=true),A),
                           ("trace",TracePSD(T(0.5);partial=true),A))
            full = name=="nuclear" ? NuclearNorm(T(0.5)) : name=="rank4" ? IndBallRank(4) :
                   name=="psd_rank4" ? IndPSD(rank=4) : TracePSD(T(0.5))
            previous=x .+ T(1e-5).*randn(rng,T,size(x)); scratch=similar(x)
            # The warm call always starts from a nearby point, not its own answer.
            prep!() = prox!(scratch,f,previous,T(0.7))
            addprox!(cases,"partial",name,f,x;prepare=prep!,profile=T==Float64 && p==512,reference_function=full)
            addprox!(cases,"partial",name*"_full",full,x;value=false)
        end
    end
end

function constrained_cases!(cases,rep)
    for T in (Float32,Float64), n in (256,100_000)
        rng=scenario_rng(rep,"constrained/$T/$n"); x=randn(rng,T,n)
        fs=(Constrained(NormL1(T(0.3)),IndBox(T(-1),T(1))),
            Constrained(SCAD(T(0.3)),IndBox(T(-1),T(1))),
            Constrained(NormL1(T(0.3)),IndHyperplane(one(T),zero(T))),
            Constrained(IndBox(zero(T),one(T)),IndHyperplane(one(T),T(n÷2))),
            Constrained(ElasticNet(T(0.3),T(0.4)),IndSimplex(T(n/10))),
            Constrained(NormL1(T(0.3)),IndBallL2(one(T))),
            Constrained(NormL1(T(0.3)),IndSphereL2(one(T))))
        for (name,f) in zip(("l1_box","scad_box","zero_sum_lasso","capped_simplex","simplex_elastic_net","l1_ball","l1_sphere"),fs)
            previous=x .+ T(0.001).*randn(rng,T,n); scratch=similar(x)
            prep!() = prox!(scratch,f,previous,T(0.7))
            addprox!(cases,"constrained",name,f,x;prepare=prep!,profile=T==Float64 && n==100_000)
        end
        for (name,f) in (("scaled",T(2)*NormL1(T(0.3))),
                         ("translated",Translate(NormL1(T(0.3)),ones(T,n))),
                         ("tilted",Tilt(NormL1(T(0.3)),fill(T(0.1),n))),
                         ("conjugate",Conjugate(NormL2(T(0.3)))),
                         ("regularize",Regularize(NormL1(T(0.3)),T(0.4))),
                         ("precompose",Precompose(NormL1(T(0.3)),T(-2))),
                         ("envelope",MoreauEnvelope(NormL1(T(0.3)),T(0.5))))
            addprox!(cases,"compositions",name,f,x;profile=T==Float64 && n==100_000 && name=="envelope")
        end
        # Dykstra has an independent specialized reference for this intersection.
        if n == 256
            C=IndIntersection(IndNonnegative(),IndHyperplane(one(T),T(10));warmstart=false)
            out=similar(x)
            check!() = (prox!(out,C,x,T(0.7));C.converged || error("Dykstra failed to converge");close_enough(out,prox(IndSimplex(T(10)),x)))
            addcase!(cases,"intersection","simplex_cold_corrections",()->prox!(out,C,x,T(0.7)),check!;
                     T,shape=string(n),profile=T==Float64,work=n)
        end
    end
end

function tensor_cases!(cases,rep)
    for T in (Float32,Float64), p in (8,48)
        rng=scenario_rng(rep,"tensors/$T/$p"); X=randn(rng,T,p,p,4)
        X=(X+permutedims(X,(2,1,3)))./2
        for (name,f) in (("fiber_l2",FiberSum(NormL2(T(0.3)))),
                         ("fiber_tv",FiberSum(TotalVariation1D(T(0.3)))),
                         ("fiber_symmetric_l2",FiberSum(NormL2(T(0.3));symmetric=true,diagonal=nothing)),
                         ("slice_l1",SliceSum([NormL1(T(0.3)) for _ in 1:4])))
            addprox!(cases,"tensors",name,f,X;profile=T==Float64 && p==48)
        end
    end
end

function product_cases!(cases,rep)
    for T in (Float32,Float64), p in (256,1024), density in (0.01,0.2), triangle in (:U,:L)
        rng=scenario_rng(rep,"products/$T/$p/$density/$triangle")
        A=Symmetric(randn(rng,T,p,p),triangle)
        xs=SparseIterate(sprandn(rng,T,p,density))
        groups=[j:min(j+7,p) for j in 1:8:p]
        xa=AtomIterate(dense(xs),GroupNormL2(T(0.3),groups))
        for (layout,x) in (("sparse",xs),("atom",xa))
            y=zeros(T,p); ref=A*dense(x)
            check!()=(mul!(y,A,x);close_enough(y,ref))
            addcase!(cases,"products","symmetric_$triangle",()->mul!(y,A,x),check!;
                     T,shape=string(p),layout="$(layout)_$density",evals=10,
                     profile=T==Float64 && p==1024 && density==0.01 && triangle==:U,work=nnz(xs))
        end
    end
end

function build_cases(rep)
    cases=PerfCase[]
    for builder in (separable_cases!,smooth_cases!,matrix_cases!,constrained_cases!,tensor_cases!,product_cases!)
        builder(cases,rep)
    end
    ids=[c.meta.id for c in cases]
    length(unique(ids))==length(ids) || error("duplicate case IDs")
    cases
end
