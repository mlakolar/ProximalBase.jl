module SymmetricParametersPathsTest

using Test, ProximalBase, LinearAlgebra
using ..TestUtils: allocs

const PB = ProximalBase
state(x) = map(n -> deepcopy(getfield(x,n)), fieldnames(typeof(x)))
function unchanged(op, x; message = "must be symmetric")
  before = state(x)
  err = try op(x); nothing catch e; e end
  @test err isa ArgumentError
  @test occursin(message, sprint(showerror,err))
  @test isequal(before,state(x))
end

struct CountingMatrix{T} <: AbstractMatrix{T}
  data::Matrix{T}
  reads::Base.RefValue{Int}
end
CountingMatrix(A::Matrix) = CountingMatrix(A,Ref(0))
Base.size(A::CountingMatrix) = size(A.data)
Base.IndexStyle(::Type{<:CountingMatrix}) = IndexLinear()
Base.getindex(A::CountingMatrix,i::Int) = (A.reads[] += 1; A.data[i])

@testset "Issue 142 and exact equality" begin
  W=[0. 1.;3. 0.]; f=NormL1(.1,W)
  x=SymmetricSparseIterate([0. .5;.5 0.]); k=tril_index(2,2,1)
  message="the array w of NormL1 must be symmetric on a SymmetricSparseIterate, but entry (2, 1) is 3.0 and entry (1, 2) is 1.0; use Symmetric(A, :L) to keep the lower triangle of A, or (A + A')/2 for its symmetric part"
  for op in (y -> f(y), y -> prox!(y,f,x,1.), y -> value_and_prox!(y,f,x,1.),
             y -> prox_coordinate!(y,f,k,1.), y -> kkt_violation(f,fill(.25,2,2),y,k))
    before=copy(x)
    err=try op(before); nothing catch e; e end
    @test err isa ArgumentError
    @test err.msg == message
    @test isequal(state(before),state(x))
  end
  fv=NormL1(.1,vec(W))
  @test_throws ArgumentError fv(x)
  unchanged(y -> prox!(y,fv,x,1.),copy(x))
  unchanged(y -> value_and_prox!(y,fv,x,1.),copy(x))
  unchanged(y -> prox_coordinate!(y,fv,k,1.),copy(x))
  unchanged(y -> kkt_violation(fv,ones(2,2),y,k),copy(x))
  unchanged(y -> prox!(y,NormL1(.1),x,vec(W)),copy(x);message="the steps γ must be symmetric")
  @test PB._checkparams(f,x) # the coordinate shape check never scans arrays
  for B in (Matrix(Symmetric(W,:L)),Symmetric(W,:L),vec(Matrix(Symmetric(W,:L))))
    h=NormL1(.1,B)
    Y=[sign(x[r,c])*max(abs(x[r,c])-.1B[(c-1)*2+r],0) for r in 1:2,c in 1:2]
    @test h(x) ≈ sum(.1B[(c-1)*2+r]*abs(x[r,c]) for r in 1:2,c in 1:2)
    @test Matrix(prox!(similar(x),h,x,1.)) ≈ Y
    @test value_and_prox!(similar(x),h,x,1.) ≈ sum(.1B[(c-1)*2+r]*abs(Y[r,c]) for r in 1:2,c in 1:2)
    z=copy(x); @test prox_coordinate!(z,h,k,1.) ≈ Y[2,1]
    @test kkt_violation(h,fill(.25,2,2),x,k) ≈ .25-.1B[2]
  end
  for (a,b) in ((0.,-0.),(NaN,NaN),(missing,missing),(1,1.))
    @test PB._symequal(a,b)
    A=Any[0 b;a 0]
    @test PB._check_entry_array(f,:w,A,x,PB._AllPairs())
  end
  D=[.4 .6;.6 .8]; B=ones(2,2); B[1]=2.
  t=Translate(NormL1(.1),B)
  for z in (D,SparseIterate(D),SymmetricSparseIterate(D))
    @test t(z) === .1sum(abs,D.-B)
  end
  A=[1. 1.;nextfloat(1.) 1.]
  err=try NormL1(1.,A)(fill!(similar(x),1)); nothing catch e; e end
  @test endswith(err.msg," (they differ only by rounding)")
end

@testset "Existing value shape messages" begin
  A=ones(3,3); x=SymmetricSparseIterate(2)
  for f in (LinearFunction(A),Tilt(NormL1(),A),Precompose(NormL1(),A),
            Sum(LinearFunction(A),SqrNormL2()),Scaled(LinearFunction(A),2.))
    err=try f(x); nothing catch e; e end
    @test err isa DimensionMismatch
    @test err.msg == "the output has size (3, 3) and the input (2, 2)"
  end
  f=KronQuadratic(Matrix{Float64}(I,3,3),Matrix{Float64}(I,3,3),A)
  err=try f(x); nothing catch e; e end
  @test err isa DimensionMismatch
  @test err.msg == "Θ has size (2, 2) and f takes 3 × 3 matrices"
  err=try Translate(NormL1(),A)(x); nothing catch e; e end
  @test err isa DimensionMismatch
  @test err.msg == "x has size (2, 2) and b (3, 3)"
  x=fill!(SymmetricSparseIterate(3),.5)
  r=Regularize(NormL1(.1),1.,ones(9))
  for h in (r,Scaled(r,2.),Sum(r,NormL1()),Translate(r,zeros(3,3)),
            Precompose(r,2.),PointwiseMinimum(r,NormL1()))
    err=try h(x); nothing catch e; e end
    @test err isa DimensionMismatch
    @test err.msg == "the arrays have sizes (3, 3) and (9,)"
  end
end

@testset "Coordinate, stored and nested read domains" begin
  for p in (4,40)
    W=CountingMatrix(ones(p,p)); W.data[3,2]=2.
    f=NormL1(.1,W); x=SymmetricSparseIterate(p); x[2,1]=.5
    k=tril_index(p,2,1); j=tril_index(p,3,2)
    W.reads[]=0
    @test prox_coordinate!(x,f,k,1.) ≈ .4
    @test W.reads[] == 3
    W.reads[]=0
    @test kkt_violation(f,zeros(p,p),x,k) == -.1
    @test W.reads[] == 3
    unchanged(y -> prox_coordinate!(y,f,j,1.),copy(x))
    W.reads[]=0
    @test PB._checkparams(f,x)
    @test W.reads[] == 0
    W.reads[]=0
    @test f(x) ≈ .08
    @test W.reads[] == 4
    nested=Scaled(Conjugate(Conjugate(f)),2.)
    z=copy(x); W.reads[]=0
    prox!(z,nested,x,1.)
    @test W.reads[] == 3
    @test Matrix(z)[2,1] ≈ .2
    y=SymmetricSparseIterate(p); y[3,1]=1.
    W.reads[]=0
    prox!(y,f,x,1.)
    @test W.reads[] == 5 # two pairs in the union, then the scalar prox
    W.data[3,1]=2.
    unchanged(z -> value_and_prox!(z,f,x,1.),y)
    unchanged(z -> prox!(z,f,x,1.),y)
    W.data[3,1]=1.
    # Tilt reads unstored zeros too.
    unchanged(z -> prox!(z,Tilt(f,zeros(p,p)),x,1.),y)
  end
  p=3; W=CountingMatrix(ones(p,p)); x=fill!(SymmetricSparseIterate(p),.5)
  f=Scaled(Tilt(NormL1(.1,W),zeros(p,p)),2.)
  y=similar(x); prox!(y,f,x,1.); W.reads[]=0
  prox!(y,f,x,1.)
  @test W.reads[] == 12 # six entries in the check, six lower entries in the prox
  C=CountingMatrix(ones(p,p)); h=Scaled(Tilt(LinearFunction(C),zeros(p,p)),2.)
  gradient!(y,h,x); C.reads[]=0
  gradient!(y,h,x)
  @test C.reads[] == 12 # the copy does not rescan C
  C.reads[]=0
  prox!(y,LinearFunction(C),x,1.)
  @test C.reads[] == 12 # the broadcast does not rescan C
  C.reads[]=0
  value_and_gradient!(y,h,x)
  @test 21 <= C.reads[] <= 22 # dot may read its first entry to choose the accumulator
  steps=CountingMatrix(ones(p,p)); f=Scaled(NormL1(.1),0.)
  prox!(y,f,x,steps); steps.reads[]=0
  prox!(y,f,x,steps)
  @test steps.reads[] == 6 # check the original steps even when scaling avoids reading them
  S=CountingMatrix(ones(p,p)); S.reads[]=0
  @test PB._check_entry_array(f,:w,Symmetric(S,:L),x,PB._AllPairs())
  @test S.reads[] == 0
end

@testset "Tiled and stored pair traversal" begin
  for p in (33, 65)
    W = CountingMatrix(ones(p, p)); f = NormL1(.1, W)
    x = fill!(SymmetricSparseIterate(p), .5)
    for domain in (PB._AllPairs(), PB._StoredPairs(x, x))
      W.reads[] = 0
      @test PB._check_entry_arrays(f, x, domain)
      @test W.reads[] == p * (p - 1)
    end
    z = SymmetricSparseIterate(p)
    for (r, c) in ((p, 1), (p, 3), (4, 3), (3, 1), (p, 2), (2, 2))
      z[r, c] = .5
    end
    W.reads[] = 0
    @test PB._check_entry_arrays(f, z, PB._StoredPairs(z, z))
    @test W.reads[] == 10
    W.data[3, 1] = nextfloat(1.)
    @test_throws ArgumentError PB._check_entry_arrays(f, z, PB._StoredPairs(z, z))
  end
end

@testset "Wrapper preflights and block atomicity" begin
  A=ones(3,3); A[2,1]=nextfloat(1.); inner=NormL1(.1,A)
  x=fill!(SymmetricSparseIterate(3),2.)
  wrappers=(Scaled(inner,2.),Precompose(inner,2.),Precompose(inner,ones(3,3)),
            Regularize(inner,1.),Translate(inner,zeros(3,3)),Tilt(inner,zeros(3,3)),
            Constrained(inner,IndBox(-1.,1.)),Constrained(inner,IndHyperplane(ones(3,3),1.)),
            Constrained(inner,IndBallL2(1.)),Constrained(inner,IndSphereL2(1.)),
            MoreauEnvelope(inner,1.),Conjugate(inner),Sum(inner,NormL1()),
            PointwiseMinimum(inner,NormL1()))
  for f in wrappers
    # Conjugate(NormL1) has a value oracle; Sum has no proximal operator.
    unchanged(_ -> f(x),copy(x))
    if isproximable(f)
      unchanged(y -> prox!(y,f,x,1.),copy(x))
      unchanged(y -> value_and_prox!(y,f,x,1.),copy(x))
    end
  end
  badset=IndBox(-A,2.)
  for f in (IndIntersection(IndBox(-2.,2.),badset),SqrDistL2(badset),DistL2(badset))
    unchanged(_ -> f(x),copy(x))
    unchanged(y -> prox!(y,f,x,1.),copy(x))
    unchanged(y -> value_and_prox!(y,f,x,1.),copy(x))
  end
  f=SeparableSum(NormL1(),inner)
  for wrapped in (Scaled(f,2.),Conjugate(Conjugate(f)))
    y=(copy(x),copy(x)); before=map(state,y)
    @test_throws ArgumentError prox!(y,wrapped,(x,x),1.)
    @test isequal(map(state,y),before)
  end
  y=(copy(x),copy(x)); before=map(state,y)
  @test_throws ArgumentError prox!(y,f,(x,x),1.)
  @test isequal(map(state,y),before)
  @test_throws ArgumentError value_and_prox!(y,f,(x,x),1.)
  @test isequal(map(state,y),before)
  @test_throws ArgumentError f((x,x))
  empty=SymmetricSparseIterate(3); y=(copy(empty),copy(empty)); before=map(state,y)
  @test_throws ArgumentError prox!(y,f,(empty,empty),1.)
  @test isequal(map(state,y),before)
end

@testset "Symmetric gradient view and all KKT oracles" begin
  x=SymmetricSparseIterate(3); k=tril_index(3,2,1)
  # All built-in gradient-only oracles that accept a symmetric iterate.
  functions=(NormL1(.2),NormL1Nonneg(.2),ElasticNet(.2,.3),SqrNormL2(),
             MCP(.2,3.),SCAD(.2,3.7),IndBox(-1.,1.),IndPattern(trues(3,3)),
             Constrained(NormL1(.2),IndBox(-1.,1.)),ZeroFunction(),
             Scaled(NormL1(.2),2.),Regularize(NormL1(.2),1.,ones(3,3)),
             LinearFunction(ones(3,3)),Tilt(NormL1(.2),ones(3,3)),
             Precompose(NormL1(.2),2.),Precompose(NormL1(.2),ones(3,3)),
             CappedL1(.2,1.),LogPenalty(.2,1.),CheckLoss(.3),HingeLoss(),
             SqrHingeLoss(),Berhu(.2,1.),EpsInsensitiveLoss(.2))
  G=zeros(3,3); G[2,1]=1.; G[1,2]=3.
  H=zeros(3,3); H[2,1]=H[1,2]=2.
  for f in functions
    @test kkt_violation(f,G,x,k) == kkt_violation(f,H,2)
  end
  for f in (NormL1(0.),Scaled(NormL1(0.),2.),Regularize(NormL1(0.),1.))
    G[2,1]=floatmax(); G[1,2]=-floatmax()
    @test kkt_violation(f,G,x,k) == 0.
    G[2,1]=Inf; G[1,2]=1.
    @test kkt_violation(f,G,x,k) == Inf
    G[1,2]=-Inf
    err=try kkt_violation(f,G,x,k); nothing catch e; e end
    @test err isa ArgumentError
    @test err.msg == "the gradient g has no symmetric part: entry (2, 1) is Inf and entry (1, 2) is -Inf"
  end
  G[2,1]=G[1,2]=nextfloat(0.)
  @test PB._SymmetricGradient(G,3)[2] === nextfloat(0.)
  @test kkt_violation(NormL1(0.),G,x,k) === nextfloat(0.)
end

@testset "Original array steps and second-order preflights" begin
  A=ones(3,3); A[2,1]=nextfloat(1.)
  x=fill!(SymmetricSparseIterate(3),.5)
  functions=(ZeroFunction(),NormL1(),
             Scaled(NormL1(),0.),Tilt(NormL1(),zeros(3,3)),Translate(NormL1(),zeros(3,3)),
             Regularize(NormL1(),1.),Precompose(NormL1(),2.),Precompose(NormL1(),ones(3,3)),
             Conjugate(NormL1()),Constrained(NormL1(),IndBox(-1.,1.)),
             IndHyperplane(ones(3,3),1.),IndHalfspace(ones(3,3),1.),
             IndHyperslab(ones(3,3),-1.,1.),IndSimplex(1.),IndBallL1(1.))
  for f in functions, op in (prox!,value_and_prox!)
    unchanged(y -> op(y,f,x,A),copy(x);message="the steps γ must be symmetric")
    # Symmetric -> dense remains the ordinary array-step operation.
    @test op(zeros(3,3),f,x,A) == op(zeros(3,3),f,Matrix(x),A)
  end
  groups=[[1],[2,4],[3,7],[5],[6,8],[9]]
  steps=fill(.2,length(groups)); f=Scaled(GroupNormL2(.1,groups),2.)
  Y=fill(.5,3,3)
  for G in groups
    # Keep the existing group path's norm over stored symmetric coordinates.
    lower=filter(i -> mod1(i,3) >= (i-1)÷3+1,G)
    Y[G] .*= max(1-.04/sqrt(sum(abs2,Y[lower])),0)
  end
  @test Matrix(prox!(similar(x),f,x,steps)) ≈ Y
  for f in (LinearFunction(A),SqrNormL2(1.,A),Tilt(SqrNormL2(),A),
            Translate(SqrNormL2(),A),Regularize(SqrNormL2(),1.,A),
            Precompose(SqrNormL2(),A),Scaled(SqrNormL2(1.,A),2.))
    @test_throws ArgumentError quadratic_model(f,x)
    unchanged(y -> hessian_vec!(y,f,x,ones(3,3)),copy(x))
    unchanged(y -> coordinate_lipschitz!(y,f),copy(x))
  end
end

@testset "Checked paths allocate nothing after warm-up" begin
  x=fill!(SymmetricSparseIterate(3),.5); y=copy(x); A=ones(3,3)
  for f in (NormL1(.1,A),LinearFunction(A),SqrNormL2(1.,A),Scaled(NormL1(.1,A),2.))
    f(x)
    @test allocs(f,x) == 0
    prox!(y,f,x,1.)
    @test allocs(prox!,y,f,x,1.) == 0
    value_and_prox!(y,f,x,1.)
    @test allocs(value_and_prox!,y,f,x,1.) == 0
    prox_coordinate!(y,f,2,1.)
    @test allocs(prox_coordinate!,y,f,2,1.) == 0
    kkt_violation(f,A,x,2)
    @test allocs(kkt_violation,f,A,x,2) == 0
    if issmooth(f)
      gradient!(y,f,x)
      @test allocs(gradient!,y,f,x) == 0
      value_and_gradient!(y,f,x)
      @test allocs(value_and_gradient!,y,f,x) == 0
    end
    if applicable(prox!,y,f,x,A)
      prox!(y,f,x,A)
      @test allocs(prox!,y,f,x,A) == 0
      value_and_prox!(y,f,x,A)
      @test allocs(value_and_prox!,y,f,x,A) == 0
    end
    if hashessian(f)
      hessian_vec!(y,f,x,A)
      @test allocs(hessian_vec!,y,f,x,A) == 0
      quadratic_model(f,x)
      @test allocs(quadratic_model,f,x) == 0
      coordinate_lipschitz!(y,f)
      @test allocs(coordinate_lipschitz!,y,f) == 0
    end
  end
end

end
