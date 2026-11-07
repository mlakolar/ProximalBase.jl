module SymmetricParametersWritesTest

using Test, ProximalBase, LinearAlgebra

struct CountingMatrix{T} <: AbstractMatrix{T}
  data::Matrix{T}
  reads::Base.RefValue{Int}
end
CountingMatrix(A::Matrix) = CountingMatrix(A,Ref(0))
Base.size(A::CountingMatrix) = size(A.data)
Base.IndexStyle(::Type{<:CountingMatrix}) = IndexLinear()
Base.getindex(A::CountingMatrix,i::Int) = (A.reads[] += 1; A.data[i])

@testset "Just-checked parameters are not rescanned by writes" begin
  x=fill!(SymmetricSparseIterate(3),.5)
  constructors=(a -> NormL1(.1,a),a -> LinearFunction(a),a -> SqrNormL2(1.,a),
                a -> Tilt(NormL1(.1),a),a -> Translate(NormL1(.1),a),
                a -> Regularize(NormL1(.1),1.,a),
                a -> IndHyperplane(a,1.),a -> IndHalfspace(a,1.),a -> IndHyperslab(a,-1.,1.))
  for make in constructors
    A=CountingMatrix(ones(3,3)); f=make(A); h=make(Symmetric(A,:L))
    for step in (1.,fill(.2,3,3))
      applicable(prox!,x,f,x,step) || continue
      y=copy(x); z=copy(x)
      prox!(y,f,x,step); prox!(z,h,x,step)
      A.reads[]=0; prox!(y,f,x,step); raw=A.reads[]
      A.reads[]=0; prox!(z,h,x,step); typed=A.reads[]
      @test raw == typed + 6
      @test Matrix(y) == Matrix(z)
    end
  end
  A=CountingMatrix(Real[1 1.0 1;1.0 1 1;1 1.0 1])
  f=LinearFunction(A,0.); y=copy(x)
  gradient!(y,f,x); A.reads[]=0
  gradient!(y,f,x)
  @test A.reads[] == 12
  @test Matrix(y) == ones(3,3)
  A.reads[]=0; prox!(y,f,x,1.)
  @test A.reads[] == 12
  @test Matrix(y) == fill(-.5,3,3)
end

@testset "Just-checked steps are not rescanned by writes" begin
  x=fill!(SymmetricSparseIterate(3),.5)
  functions=(NormL1(.1),Scaled(NormL1(.1),2.),Tilt(NormL1(.1),ones(3,3)),
             Translate(NormL1(.1),ones(3,3)),Regularize(NormL1(.1),1.,ones(3,3)),
             Conjugate(ElasticNet(.1,.2)),IndHyperplane(ones(3,3),1.),
             IndHalfspace(ones(3,3),1.),IndHyperslab(ones(3,3),-1.,1.),
             IndSimplex(1.),IndBallL1(1.))
  for f in functions
    A=CountingMatrix(fill(.2,3,3)); B=Symmetric(A,:L)
    for op in (prox!,value_and_prox!)
      y=copy(x); z=copy(x)
      op(y,f,x,A); op(z,f,x,B)
      A.reads[]=0; op(y,f,x,A); raw=A.reads[]
      A.reads[]=0; op(z,f,x,B); typed=A.reads[]
      @test raw == typed + 6
      @test Matrix(y) == Matrix(z)
    end
  end
end

end
