# Research probes for #843, not package tests or a proposed implementation.
# Run in an environment that develops the two pinned source checkouts; see README.md.
using ProximalBase, ProximalSolvers, LinearAlgebra, SparseArrays, Random, TOML, Test, SHA
const PB = ProximalBase
const PS = ProximalSolvers
BLAS.set_num_threads(1)
const observations = Dict{String,Any}[]
const checks = Dict{String,Any}[]

materialize(x::SymmetricSparseIterate) = Matrix(x)
materialize(x::AbstractArray) = Array(x)
materialize(x) = x
brief(x) = sprint(show, materialize(x); context=:limit => true)
function observe(thunk, family, representation, operation)
  try
    v = thunk()
    push!(observations, Dict("family" => family, "representation" => representation,
      "operation" => operation, "status" => "returned", "result" => brief(v)))
    return v
  catch e
    push!(observations, Dict("family" => family, "representation" => representation,
      "operation" => operation, "status" => string(nameof(typeof(e))),
      "result" => sprint(showerror, e)))
    return nothing
  end
end
function check(thunk, name)
  details = thunk()
  push!(checks, Dict("name" => name, "details" => brief(details)))
  println("CHECK ", name, ": ", brief(details))
  flush(stdout)
end

# A documented function-author extension, with no PB private methods.
struct UserSquare{T} <: SeparableFunction
  w::Vector{T}
end
PB.value_entry(f::UserSquare, v, i) = f.w[i] * v^2 / 2
PB.prox_entry(f::UserSquare, v, i, γ) = v / (1 + γ * f.w[i])
PB.isconvex(::Type{<:UserSquare}) = true
PB.iszeropreserving(::UserSquare) = true
PB.entry_parameters(f::UserSquare) = (:w => f.w,)
PB.check_parameters(f::UserSquare, x) =
  (length(f.w) == prod(size(x)) || throw(DimensionMismatch("weight length")); nothing)

function vector_families(::Type{T}) where T
  G = [[1,3], [2,4]] # Noncontiguous groups and unequal weights.
  w = T[1,2]
  a, b = T(0.3), T(0.2)
  X = T[1 0 2 0; 0 1 0 2; 1 1 0 0; 0 0 1 1; 1 0 0 1]
  y = T[1,0,1,0,1]
  g = GroupNormL2(a, G, w)
  Pair{String,Any}[
    "ZeroFunction"=>ZeroFunction(), "SqrNormL2"=>SqrNormL2(a),
    "QuadraticFunction"=>QuadraticFunction(Diagonal(T[1,2,3,4])),
    "LeastSquaresLoss"=>LeastSquaresLoss(y, X), "LogisticLoss"=>LogisticLoss(y, X),
    "PoissonLoss"=>PoissonLoss(y, X), "NormL1"=>NormL1(a, T[1,2,0,1]),
    "NormL1Nonneg"=>NormL1Nonneg(a), "ElasticNet"=>ElasticNet(a,b),
    "NormL2"=>NormL2(a), "NormLinf"=>NormLinf(a),
    "GroupNormL2"=>g, "GroupNormLinf"=>GroupNormLinf(a,G,w),
    "SparseGroupNormL2"=>SparseGroupNormL2(a,b,G,w),
    "GroupMCP"=>GroupMCP(a,G,w), "GroupSCAD"=>GroupSCAD(a,G,w),
    "NormL0"=>NormL0(a), "MCP"=>MCP(a), "SCAD"=>SCAD(a),
    "NegEntropy"=>NegEntropy(a), "NegLog"=>NegLog(a),
    "IndBox"=>IndBox(T(-1),T(1)), "IndSimplex"=>IndSimplex(T(1)),
    "ScaledL1"=>Scaled(NormL1(a),T(2)),
    "TranslateL1"=>Translate(NormL1(a),fill(b,4)),
    "TiltL1"=>Tilt(NormL1(a),fill(b,4)),
    "RegularizeL1"=>Regularize(NormL1(a),b),
    "BoxL1"=>Constrained(NormL1(a),IndBox(T(-1),T(1))),
    "ConjugateL1"=>Conjugate(NormL1(a)),
    "SumSmooth"=>Sum(SqrNormL2(a),SqrNormL2(b)),
    "ScaledGroup"=>Scaled(g,T(2)), "TranslateGroup"=>Translate(g,fill(b,4)),
    "TiltGroup"=>Tilt(g,fill(b,4)), "RegularizeGroup"=>Regularize(g,b),
    "PrecomposeGroup"=>Precompose(g,T[2,3,2,3]),
    "UserSquare"=>UserSquare(T[1,2,3,4]),
  ]
end

function capabilities(::Type{T}) where T
  x = T[2,-1,0,3]
  for (name,f) in vector_families(T)
    family = "$name/$T"
    observe(family,"traits","traits") do
      (; smooth=issmooth(f), proximable=isproximable(f), separable=isseparable(f),
         convex=isconvex(f), hessian=hashessian(f), zero=iszeropreserving(f))
    end
    for (rep,make) in ("Array"=>()->copy(x), "SparseIterate"=>()->SparseIterate(x),
                      "AtomIterate"=>()->AtomIterate(x,[[1,3],[2,4]]))
      # Each operation starts from a fresh point and output.
      observe(()->f(make()),family,rep,"value")
      observe(()->prox(f,make(),T(0.5)),family,rep,"prox allocating")
      observe(()->(z=make(); prox!(z,f,z,T(0.5))),family,rep,"prox exact alias")
      observe(family,rep,"combined value agreement") do
        z=make(); v=value_and_prox!(z,f,make(),T(0.5))
        (; v, evaluated=f(z), agrees=isapprox(v,f(z);rtol=30eps(T),atol=30eps(T)))
      end
      observe(()->prox(f,make(),fill(T(0.5),4)),family,rep,"four entry steps")
      observe(()->prox(f,make(),T[0.5,0.25]),family,rep,"two group steps")
      if issmooth(f)
        observe(()->gradient(f,make()),family,rep,"gradient")
        observe(family,rep,"combined gradient agreement") do
          z=make(); v=value_and_gradient!(z,f,make())
          (; v, agrees=isapprox(v,f(make());rtol=30eps(T),atol=30eps(T)))
        end
      end
      if hashessian(f)
        observe(()->hessian_vec(f,make(),make()),family,rep,"hessian_vec")
        observe(()->typeof(quadratic_model(f,make())),family,rep,"quadratic_model")
      end
      if rep != "Array"
        observe(()->prox_coordinate!(make(),f,1,T(0.5)),family,rep,"coordinate existing")
        observe(()->kkt_violation(f,copy(x),make(),1),family,rep,"zero-coordinate KKT")
      end
    end
    observe(()->prox_entry(f,T(2),1,T(0.5)),family,"scalar","prox_entry")
    observe(()->value_entry(f,T(2),1),family,"scalar","value_entry")
    observe(()->typeof(restrict(f,[1,3])),family,"subset","restrict entries 1,3")
    observe(()->coordinate_lipschitz(f,1),family,"scalar","coordinate_lipschitz")
    observe(()->coordinate_lipschitz!(zeros(T,4),f),family,"Array","coordinate_lipschitz!")
    observe(()->lla_weights(f,copy(x)),family,"Array","lla_weights")
  end
  # Matrix-only functions and the non-array symmetric owner have a separate domain.
  M = T[2 0.2; 0.2 1]
  for (name,f) in ("NormL1"=>NormL1(T(0.3)), "SqrNormL2"=>SqrNormL2(T(0.3)),
    "LogDetLoss"=>LogDetLoss(Matrix{T}(I,2,2)), "NuclearNorm"=>NuclearNorm(T(0.3)),
    "IndPSD"=>IndPSD(), "SpectralL1"=>Spectral(NormL1(T(0.3))),
    "SingularL1"=>SingularValue(NormL1(T(0.3))),
    "UserSquare"=>UserSquare(T[1,2,2,1]))
    for (rep,make) in ("Matrix"=>()->copy(M), "SparseMatrixIterate"=>()->SparseIterate(M),
      "SymmetricSparseIterate"=>()->SymmetricSparseIterate(M))
      family="$name/$T"
      observe(()->f(make()),family,rep,"value")
      observe(()->prox(f,make(),T(0.5)),family,rep,"prox allocating")
      observe(()->(z=make();prox!(z,f,z,T(0.5))),family,rep,"prox exact alias")
      observe(family,rep,"combined value agreement") do
        z=make(); v=value_and_prox!(z,f,make(),T(0.5))
        (; v, evaluated=f(z), agrees=isapprox(v,f(z);rtol=50eps(T),atol=50eps(T)))
      end
      issmooth(f) && observe(()->gradient(f,make()),family,rep,"gradient")
      hashessian(f) && observe(()->hessian_vec(f,make(),make()),family,rep,"hessian_vec")
    end
  end
end

function numerical_checks(::Type{T}) where T
  for (label,v) in ("overflow"=>(T===Float32 ? T(1e20) : T(1e155)),
                     "underflow"=>(T===Float32 ? T(1e-30) : T(1e-200)))
    check("group norm $label/$T") do
      u=T[v,0]; λ=v; L=T(2)
      f=GroupNormL2(λ,[[1,2]],T[1]); ref=prox(f,u,inv(L))
      factor=PS._groupshrink(sqrt(sum(abs2,u)),λ,L); localout=factor .* u
      @test ref[1] ≈ v/2
      @test !isapprox(localout[1]/v,T(0.5);rtol=T(0.1))
      # Execute the actual PS model step, including lazy curvature and cache writes.
      q=QuadraticFunction(Diagonal(T[2,2]),T[-2v,0])
      x=AtomIterate(zeros(T,2),f); m=PS.cdmodel(q,x); PS.initialize!(m,x)
      PS._groupstep!(m,f,x,1,false)
      @test !isapprox(x[1]/v,T(0.5);rtol=T(0.1))
      models=Dict("quadratic"=>Vector(x))
      for (name,loss) in ("LS"=>LeastSquaresLoss(T[v,0],Matrix{T}(I,2,2)),
        "MLS"=>LeastSquaresLoss(reshape(T[v,0],1,2),ones(T,1,1)))
        penalty=GroupNormL2(v/2,[[1,2]],T[1]); point=AtomIterate(zeros(T,2),penalty)
        model=PS.cdmodel(loss,point); PS.check_penalty(model,penalty)
        PS.initialize!(model,point); PS._groupstep!(model,penalty,point,1,false)
        @test !isapprox(point[1]/v,T(0.5);rtol=T(0.1))
        models[name]=Vector(point)
      end
      (; trial=u, PB=ref, PS_formula=localout, PS_models=models, expected=v/2)
    end
  end
  check("metric L2 finite output failure/$T") do
    v=T===Float32 ? T(1e20) : T(1e200)
    z=T[v,0]; g=GroupNormL2(2v,[[1,2]],T[1]); d=ones(T,2)
    m=PS._DiagonalMetric(d,sqrt.(d),similar(d)); out=similar(z)
    PS._metric_prox!(out,g,z,one(T),m)
    @test all(iszero,prox(g,z,one(T)))
    @test any(isnan,out)
    (; input=z, PS=out, expected=zeros(T,2))
  end
  check("metric Linf overflow/$T") do
    v=floatmax(T)/4; z=T[v,v]; d=fill(T(8),2)
    # True clipping level is v - v/16; d_i*abs(z_i) overflows individually.
    g=GroupNormLinf(v,[[1,2]],T[1]); m=PS._DiagonalMetric(d,sqrt.(d),similar(d))
    out=similar(z); PS._metric_prox!(out,g,z,one(T),m)
    expected=fill(v*(T(15)/16),2)
    @test !isapprox(out[1]/v,T(15)/16;rtol=10eps(T))
    (; PS=out, expected)
  end
  check("metric ordinary KKT/$T") do
    z=T[3,-2]; d=T[1,4]; g=GroupNormL2(T(0.7),[[1,2]],T[1])
    out=copy(z); m=PS._DiagonalMetric(d,sqrt.(d),similar(d))
    PS._metric_prox!(out,g,z,one(T),m)
    residual=norm(d.*(out-z) .+ T(0.7).*out./norm(out))
    radial=prox(g,z,one(T))
    @test residual < 100eps(T)
    @test norm(out-radial)>T(0.1)
    (; output=out, residual, scalar_step_output=radial)
  end
  check("diagonal Linf independent level reference/$T") do
    rng=Xoshiro(843); worst=zero(T)
    for n in (1,2,5,32), _ in 1:5
      z=randn(rng,T,n); d=exp.(randn(rng,T,n)); c=T(0.4)
      g=GroupNormLinf(c,[collect(1:n)],T[1]); out=copy(z)
      PS._metric_prox!(out,g,z,one(T),PS._DiagonalMetric(d,sqrt.(d),similar(d)))
      ref=setprecision(BigFloat,256) do
        zb=BigFloat.(z); db=BigFloat.(d); cb=BigFloat(c)
        lo=BigFloat(0); hi=maximum(abs,zb)
        for _ in 1:300
          mid=(lo+hi)/2
          sum(db.*max.(abs.(zb).-mid,0))>cb ? (lo=mid) : (hi=mid)
        end
        T.(sign.(zb).*min.(abs.(zb),hi))
      end
      worst=max(worst,maximum(abs,out-ref))
      @test out≈ref rtol=50eps(T) atol=50eps(T)
    end
    (; cases=20,worst_absolute_error=worst)
  end
  check("group differential and alias/$T") do
    rng=Xoshiro(843); cases=0
    for n in (1,2,5,32,256), _ in 1:5
      u=randn(rng,T,n); c=T(0.7); step=T(0.5)
      f=GroupNormL2(c,[collect(1:n)],T[1])
      ref=prox(NormL2(c),u,step); out=prox(f,u,step)
      x=AtomIterate(zeros(T,n),f); r=prox_coordinate!(x,f,1,u,step)
      alias=copy(u); prox!(alias,f,alias,step)
      @test out≈ref rtol=30eps(T) atol=30eps(T)
      @test Vector(x)≈ref rtol=30eps(T) atol=30eps(T)
      @test alias≈ref rtol=30eps(T) atol=30eps(T)
      @test collect(r)≈ref rtol=30eps(T) atol=30eps(T)
      cases+=1
    end
    (; cases)
  end
  check("group NaN and endpoint semantics/$T") do
    g=GroupNormL2(one(T),[[1,2]],T[1]); u=T[NaN,2]
    @test all(isnan,prox(g,u,one(T)))
    @test isequal(prox(g,u,zero(T)),u)
    @test all(iszero,prox(g,u,T(Inf)))
    @test all(iszero,prox(GroupNormL2(one(T),[[1,2]],T[Inf]),u,one(T)))
    x=AtomIterate(zeros(T,2),g); prox_coordinate!(x,g,1,T[0.1,0],one(T))
    @test nactive(x)==0
    prox_coordinate!(x,g,1,T[3,0],one(T)); @test nactive(x)==1
    prox_coordinate!(x,g,1,one(T)); prox_coordinate!(x,g,1,one(T))
    @test all(iszero,x)
    (; nan=prox(g,u,one(T)), zero_step=prox(g,u,zero(T)),
       infinite_step=prox(g,u,T(Inf)), stored_zero_groups=nactive(x))
  end
end

function interface_checks()
  check("public validator is weaker for builtins") do
    f=NormL1(0.3,ones(3)); x=zeros(4); out=fill(17.0,4)
    @test PB.check_parameters(f,x)===nothing
    @test_throws DimensionMismatch PB._checkparams(f,x)
    @test_throws DimensionMismatch prox!(out,f,x,1.0)
    @test out==fill(17.0,4)
    @test_throws DimensionMismatch PB.check_parameters(UserSquare(ones(3)),x)
    (; public_result=PB.check_parameters(f,x), private_rejects=true, output_unchanged=true)
  end
  check("custom extension and symmetric validation") do
    f=UserSquare([1.0,2,2,1]); M=[2.0 0.2;0.2 1]
    x=SymmetricSparseIterate(M); out=similar(x)
    prox!(out,f,x,0.5)
    @test Matrix(out)≈M./(1 .+0.5.*reshape(f.w,2,2))
    bad=UserSquare([1.0,2,3,1]); before=Matrix(out)
    @test_throws ArgumentError prox!(out,bad,x,0.5)
    @test Matrix(out)==before
    (; valid_custom_owner=true, asymmetric_rejected_before_write=true)
  end
  check("affine minimizers and bounded flat group limitation") do
    @test PS._coordinate_minimizer(ElasticNet(1.0,2.0),0.0,3.0,0.0,1)==-1
    @test PS._coordinate_minimizer(NegLog(2.0),1.0,4.0,0.0,1)==0.5
    @test PS._coordinate_minimizer(IndBox(2.0,3.0),0.0,1.0,0.0,1)==2
    @test_throws ArgumentError PS._coordinate_minimizer(MCP(1.0),0.0,0.1,0.0,1)
    @test_throws ArgumentError PS._coordinate_minimizer(SCAD(1.0),0.0,0.1,0.0,1)
    f=QuadraticFunction(zeros(2,2),[0.1,0.0])
    g=SparseGroupNormL2(1.0,1.0,[[1,2]],[1.0]); x=AtomIterate(zeros(2),g)
    m=PS.cdmodel(f,x); PS.initialize!(m,x)
    @test_throws ArgumentError PS._groupstep!(m,g,x,1,false)
    # |b'z| <= 0.1||z||2 < ||z||1+||z||2 proves 0 is the global minimizer.
    (; ridge=-1.0, barrier=0.5, bounded_sparse_group_rejected=true)
  end
  for T in (Float32,Float64)
    check("KKT signs at zero/$T") do
      rng=Xoshiro(844); count=0; mismatches=Dict{String,Int}()
      G=[[1,3],[2,4]]
      families=(NormL1(T(0.3)),NormL1Nonneg(T(0.3)),ElasticNet(T(0.3),T(0.2)),
        MCP(T(0.3)),SCAD(T(0.3)),GroupNormL2(T(0.3),G,T[1,2]),
        Scaled(SparseGroupNormL2(T(0.4),T(0.6),G,T[1,2]),T(0.3)),
        Constrained(NormL1(T(0.3)),IndBox(T(0),T(1))))
      for g in families, _ in 1:20
        grouped=g isa PB.AbstractGroupPenalty || g isa Scaled{<:PB.AbstractGroupPenalty}
        x=grouped ? AtomIterate(zeros(T,4),G) : SparseIterate(zeros(T,4))
        grad=randn(rng,T,4)
        for k in 1:ncoordinates(x)
          pb=kkt_violation(g,grad,x,k)>0
          ps=PS._kktviolated(g,grad,x,k,PS._level(g))
          if g isa NormL1Nonneg
            @test pb==(-grad[k]>g.λ0)
            @test ps==(abs(grad[k])>g.λ0)
          else
            @test pb==ps
          end
          pb!=ps && (mismatches[string(nameof(typeof(g)))]=get(mismatches,string(nameof(typeof(g))),0)+1)
          count+=1
        end
      end
      (; cases=count, mismatches)
    end
    check("sparse-group KKT underflow/$T") do
      v=T===Float32 ? T(1e-30) : T(1e-200)
      g=Scaled(SparseGroupNormL2(zero(T),v/2,[[1,2]],T[1]),one(T))
      x=AtomIterate(zeros(T,2),[[1,2]]); grad=T[v,0]
      pb=kkt_violation(g,grad,x,1); ps=PS._kktviolated(g,grad,x,1,one(T))
      @test pb>0
      @test !ps
      (; gradient=grad,PB_violation=pb,PS_violated=ps)
    end
    check("KKT exact ties and zero infinite weights/$T") do
      g=NormL1(one(T),T[0,1,Inf]); x=SparseIterate(zeros(T,3))
      for grad in (T[0,1,2],T[1,nextfloat(one(T)),3],T[0,prevfloat(one(T)),0])
        for k in 1:3
          @test (kkt_violation(g,grad,x,k)>0)==PS._kktviolated(g,grad,x,k,one(T))
        end
      end
      (; cases=9)
    end
  end
  check("curvature index and mutation convention") do
    X=[1.0 2;3 4;5 6]; f=LeastSquaresLoss(zeros(3,2),X); L=zeros(2,2)
    @test coordinate_lipschitz!(L,f)===L
    @test L==[35.0 35;56 56]
    @test coordinate_lipschitz(f,3)==35
    @test !applicable(coordinate_lipschitz,f)
    (; L, scalar_entry_3=coordinate_lipschitz(f,3), allocating_all_entries=false)
  end
  check("one-sided KKT and lambda_max discrepancy") do
    f=QuadraticFunction(reshape([1.0],1,1),[2.0]); g=NormL1Nonneg(1.0); x=SparseIterate(1)
    @test kkt_violation(g,[2.0],x,1)==-3.0
    @test PS._kktviolated(g,[2.0],x,1,1.0)
    @test lambda_max(f,g,x)==2.0
    # For z>=0, f(z)+lambda*z = z^2/2+(2+lambda)z is minimized at 0 for every lambda>=0.
    (; PB_violation=-3.0, PS_violated=true, PS_lambda_max=2.0, mathematical_lambda_max=0.0)
  end
  check("second order and gradient mapping") do
    X=[1.0 2;3 4;5 6]; f=LogisticLoss([0.0,1,0],X); x=[0.1,-0.2]; v=[0.3,0.4]
    hv=hessian_vec(f,x,v); h=1e-5
    @test hv≈(gradient(f,x+h*v)-gradient(f,x-h*v))/(2h) rtol=1e-7
    q=quadratic_model(f,x); quadratic_model!(q,f,x)
    @test gradient(q,x)≈gradient(f,x)
    g=NormL1(0.3); step=0.1
    @test gradient_mapping(f,g,x,step)≈(x-prox(g,x-step*gradient(f,x),step))/step
    @test_throws ArgumentError gradient!(x,f,x)
    @test_throws ArgumentError hessian_vec!(v,f,x,v)
    (; hessian=hv, model_gradient_agrees=true, gradient_alias_rejected=true)
  end
  check("scalar loss Fenchel equality") do
    worst=0.0; count=0
    for loss in (PB.Squared(),PB.Logistic(),PB.Poisson(),PB.Huber(1.0)), η in (-2.0,0,2.0)
      y=0.3; u=PB.deriv(loss,η,y)
      gap=loss(η,y)+PS._conjugate(loss,u,y)-u*η
      @test abs(gap)<1e-12
      worst=max(worst,abs(gap)); count+=1
    end
    @test !(PB.Squared() isa AbstractFunction)
    (; cases=count,worst_error=worst,scalar_loss_is_AbstractFunction=false)
  end
  check("LMO and block product operations") do
    g=[2.0,-3,1]; C=IndBallL1(2.0); s=lmo(C,g)
    @test haslmo(C); @test dot(s,g)==-6
    f=SeparableSum(NormL1(0.3),SqrNormL2(0.2)); x=([2.0,-1],[0.5,3.0])
    z=(zeros(2),zeros(2)); val=value_and_prox!(z,f,x,0.5)
    @test val≈f(z)
    (; lmo=s, compact_atom=typeof(lmo_atom(C,g)),block_value=val)
  end
end

function metric_dispatch()
  for (name,g) in vector_families(Float64)
    g isa PB.AbstractGroupPenalty || g isa Scaled{<:PB.AbstractGroupPenalty} || continue
    z=[3.0,-2,1,0]; d=[1.0,2,3,4]; out=fill(17.0,4)
    observe(name,"PS heterogeneous metric","metric_prox and output on error") do
      try
        v=PS._metric_prox!(out,g,z,1.0,PS._DiagonalMetric(d,sqrt.(d),similar(d)))
        (; value=v, output=out)
      catch e
        (; exception=string(nameof(typeof(e))), output=out, changed=out!=fill(17.0,4))
      end
    end
  end
end

function family_catalog(outdir)
  open(joinpath(outdir,"family-traits.tsv"),"w") do io
    println(io,"public_type\tsupertype\tsmooth\tproximable\tseparable\tconvex\thessian")
    for name in sort(names(PB))
      F=getfield(PB,name)
      F isa Type && F<:AbstractFunction || continue
      trait(t)=try string(t(F)) catch; "requires concrete type or instance" end
      println(io,join((name,supertype(F),trait(issmooth),trait(isproximable),
        trait(isseparable),trait(isconvex),trait(hashessian)),'\t'))
    end
  end
end

function measure(thunk::F; repetitions=1000,samples=5) where F
  thunk(); thunk()
  bytes=@allocated for _ in 1:repetitions; thunk(); end
  times=[@elapsed(for _ in 1:repetitions; thunk(); end) for _ in 1:samples]
  (; bytes_per_call=bytes/repetitions,minimum_ns=minimum(times)*1e9/repetitions,
    median_ns=sort(times)[cld(samples,2)]*1e9/repetitions,repetitions,samples)
end
function kernel_measurements(::Type{T},n) where T
    u=fill(T(0.5),n); out=similar(u); g=GroupNormL2(T(0.1),[collect(1:n)],T[1])
    x=AtomIterate(zeros(T,n),g); f=NormL2(T(0.1))
    for (name,thunk) in ("full one-group prox"=>()->prox!(out,g,u,T(0.5)),
      "NormL2 block alternative"=>()->prox!(out,f,u,T(0.5)),
      "storage coordinate step"=>()->prox_coordinate!(x,g,1,u,T(0.5)))
      observe(()->measure(thunk),"$name/$T/n=$n","warmed CPU","measurement")
    end
end
function solver_measurements(::Type{T}) where T
  # Baselines of actual solver calls. No API candidate is installed or benchmarked.
    q=QuadraticFunction(Diagonal(T[2,3,4,5]),T[-3,2,-1,4])
    g=GroupNormL2(T(0.3),[[1,2],[3,4]],T[1,1])
    options=ProximalOptions(;ftol=eps(T)^2,xtol=eps(T)^2,gradtol=T(1e-6),maxiter=1000)
    for (name,method,make) in (("CD",CDSolver(;tol=T(1e-6)),()->AtomIterate(zeros(T,4),g)),
      ("PGD diagonal",ProximalGradientDescent(;preconditioner=:diagonal),()->zeros(T,4)))
      result=observe("solver $name/$T","whole solve","baseline and stationarity") do
        solvepoint=method isa CDSolver ? x->solve!(x,q,g,method) : x->solve!(x,q,g,method,options)
        run=()->solvepoint(make())
        x=make(); solvepoint(x)
        residual=maximum(abs,gradient_mapping(q,g,Vector(x),T(0.1)))
        @test residual < (T===Float32 ? T(1e-4) : T(1e-5))
        (; point=Vector(x),residual,measurement=measure(run;repetitions=10,samples=3))
      end
      @test result!==nothing
    end
end

outdir=isempty(ARGS) ? joinpath(@__DIR__,"results") : abspath(ARGS[1])
mkpath(outdir)
metadata=Dict("julia"=>string(VERSION),"kernel"=>string(Sys.KERNEL),"arch"=>string(Sys.ARCH),
  "cpu"=>Sys.CPU_NAME,"threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),
  "blas"=>string(BLAS.get_config()),"pb_commit"=>readchomp(`git -C $(pkgdir(PB)) rev-parse HEAD`),
  "ps_commit"=>readchomp(`git -C $(pkgdir(PS)) rev-parse HEAD`),
  "pb_source"=>pathof(PB),"ps_source"=>pathof(PS),
  "harness_sha256"=>bytes2hex(sha256(read(@__FILE__))),"bounds_checking"=>"yes")
println("METADATA ", metadata)
flush(stdout)
@testset "issue 843 research reproductions" begin
  for T in (Float32,Float64)
    capabilities(T)
    numerical_checks(T)
  end
  interface_checks()
  metric_dispatch()
  for T in (Float32,Float64)
    for n in (2,32,256); kernel_measurements(T,n); end
    solver_measurements(T)
  end
end
family_catalog(outdir)
open(joinpath(outdir,"runtime.toml"),"w") do io
  TOML.print(io,Dict("metadata"=>metadata,"checks"=>checks,
    "observation_count"=>length(observations),"measurements"=>filter(r->
      r["operation"] in ("measurement","baseline and stationarity"),observations));sorted=true)
end
open(joinpath(outdir,"capabilities.tsv"),"w") do io
  println(io,"family\trepresentation\toperation\tstatus\tresult")
  for row in observations
    println(io,join((replace(row[k],'\t'=>" ",'\n'=>" ") for k in
      ("family","representation","operation","status","result")), '\t'))
  end
end
println("WROTE ",length(observations)," observations and ",length(checks)," checked reproductions to ",outdir)
