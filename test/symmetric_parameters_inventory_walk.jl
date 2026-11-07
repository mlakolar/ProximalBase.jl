module SymmetricParametersInventoryWalkTest

using Test, ProximalBase, LinearAlgebra

const PB = ProximalBase

function constructor_cases()
  A=Matrix{Float64}(I,3,3); v=ones(3); groups=[[1],[2,3]]
  specified=Dict{Symbol,Any}(
    :LinearFunction=>LinearFunction(A), :SqrNormL2=>SqrNormL2(1.,A),
    :QuadraticFunction=>QuadraticFunction(A,v), :KronQuadratic=>KronQuadratic(A,A,A),
    :LeastSquaresLoss=>LeastSquaresLoss(v,A), :GLMLoss=>GLMLoss(PB.Squared(),v,A),
    :LogDetLoss=>LogDetLoss(A), :HuberLoss=>HuberLoss(v,A,1.),
    :LogisticLoss=>LogisticLoss(v,A), :PoissonLoss=>PoissonLoss(v,A),
    :IndBallL0=>IndBallL0(1), :BernoulliLoss=>BernoulliLoss(fill(.5,3,3),A),
    :ElasticNet=>ElasticNet(.1,.2,A), :CappedL1=>CappedL1(.1,1.,A),
    :LogPenalty=>LogPenalty(.1,1.,A), :NormLq=>NormLq(.1,.5,A),
    :Berhu=>Berhu(.1,1.,A), :CheckLoss=>CheckLoss(.3,A),
    :HingeLoss=>HingeLoss(ones(3,3),A), :SqrHingeLoss=>SqrHingeLoss(ones(3,3),A),
    :EpsInsensitiveLoss=>EpsInsensitiveLoss(.2,A), :KLDivergence=>KLDivergence(ones(3,3)),
    :GroupNormL2=>GroupNormL2(.1,groups), :GroupNormLinf=>GroupNormLinf(.1,groups),
    :SparseGroupNormL2=>SparseGroupNormL2(.1,.2,groups),
    :GroupMCP=>GroupMCP(.1,groups), :GroupSCAD=>GroupSCAD(.1,groups),
    :MCP=>MCP(.1,3.), :SCAD=>SCAD(.1,3.7), :NormL1Fused=>NormL1Fused(.1,.2),
    :FusedLasso=>FusedLasso(.1,.2), :SLOPE=>SLOPE([.3,.2,.1]),
    :SumLargest=>SumLargest(1), :IndBallRank=>IndBallRank(1),
    :IndBallGroupL0=>IndBallGroupL0(1,groups), :IndPattern=>IndPattern(trues(3,3)),
    :IndBox=>IndBox(-A,A), :IndBinary=>IndBinary(-ones(3,3),ones(3,3)),
    :IndBanded=>IndBanded(1,3), :IndHyperplane=>IndHyperplane(A,1.),
    :IndHalfspace=>IndHalfspace(A,1.), :IndHyperslab=>IndHyperslab(A,-1.,1.),
    :IndAffine=>IndAffine(A,v), :IndPolyhedron=>IndPolyhedron(A,-v,v),
    :IndGraph=>IndGraph(A), :Scaled=>Scaled(NormL1(),2.),
    :Tilt=>Tilt(NormL1(),A), :Translate=>Translate(NormL1(),A),
    :Conjugate=>Conjugate(NormL1()), :SeparableSum=>SeparableSum(NormL1(),NormL2()),
    :Sum=>Sum(SqrNormL2(),LinearFunction(A)), :Spectral=>Spectral(NormL1()),
    :SingularValue=>SingularValue(NormL1()), :MoreauEnvelope=>MoreauEnvelope(NormL1(),1.),
    :SqrDistL2=>SqrDistL2(IndBox(-1.,1.)), :DistL2=>DistL2(IndBox(-1.,1.)),
    :Precompose=>Precompose(NormL1(),2.), :LinearImage=>LinearImage(NormL1(),A,1.),
    :Regularize=>Regularize(NormL1(),1.,A),
    :PointwiseMinimum=>PointwiseMinimum(NormL1(),NormL2()),
    :IndIntersection=>IndIntersection(IndBox(-1.,1.),IndBallL2()),
    :SliceSum=>SliceSum(NormL1(),NormL2()), :FiberSum=>FiberSum(NormL1()))
  cases=Any[]
  for name in names(PB)
    F=getfield(PB,name)
    F isa Type && F <: AbstractFunction || continue
    isabstracttype(Base.unwrap_unionall(F)) && continue
    f=get(specified,name,nothing)
    if f === nothing
      f=try F() catch e; e end
    end
    @test f isa F
    f isa AbstractFunction && push!(cases,f)
  end
  append!(cases,(NormL1(1.,A),NormL1Nonneg(1.,A),Precompose(NormL1(),ones(3,3)),
                 Precompose(NormL1(),A,1.),Constrained(NormL1(),IndBox(-1.,1.)),
                 Constrained(NormL1(),IndHyperplane(A,1.)),
                 Constrained(NormL1(),IndBallL2()),Constrained(NormL1(),IndSphereL2())))
  cases
end

# These families use image/design data, group structure, rank-indexed data,
# or work buffers rather than caller-supplied numbers in the entry domain.
const excluded=Union{QuadraticFunction,LeastSquaresLoss,GLMLoss,LogDetLoss,
                     PB.AbstractGroupPenalty,IndBallGroupL0,IndSOC,IndRotatedSOC,
                     IndExpPrimal,IndExpDual,SliceSum,FiberSum,IndAffine,
                     IndPolyhedron,IndGraph,LinearImage,SLOPE,Spectral,SingularValue,
                     NuclearNorm,TracePSD,IndPSD,IndBallRank,IndStiefel,
                     TotalVariation1D,FusedLasso,IndIsotonic,SumLargest,NormL1Fused}

@testset "Constructor coverage and array-field inventory" begin
  for f in constructor_cases()
    own=first.(PB._entry_arrays(f))
    for name in fieldnames(typeof(f))
      a=getfield(f,name)
      a isa AbstractArray || continue
      if name in own
        @test any(pair -> first(pair) == name && last(pair) === a,PB._entry_arrays(f))
      else
        computed=f isa KronQuadratic && name in (:A,:B,:D,:AD,:W)
        image=f isa Precompose && !(f isa PB._ScalarPrecompose) && name == :A
        @test computed || image || f isa excluded
      end
    end
    for inner in PB._inner_functions(f)
      @test inner isa AbstractFunction
    end
  end
end

end
