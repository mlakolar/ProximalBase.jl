module BlockStepFailuresTest
using Test, ProximalBase

struct Flaky <: AbstractVector{Float64} end
Base.size(::Flaky) = (2,)
Base.getindex(::Flaky, i::Int) = i == 1 ? 3. : error("boom")

@testset "block preflight before writes" begin
  f = GroupNormL2(1., [[1,3], [2,4]])
  for (u, k, E) in ((ones(3), 1, DimensionMismatch), (ones(2), 0, BoundsError),
                    (ones(2), 3, BoundsError), (ones(ComplexF64,2), 1, ArgumentError))
    out = fill(-7., 2)
    @test_throws E prox_block!(out, f, u, k, 1.)
    @test out == [-7.,-7.]
  end
  out = ones(ComplexF64, 2)
  @test_throws ArgumentError prox_block!(out, f, ones(2), 1, 1.)
  @test out == ones(ComplexF64, 2)
  buf = [1.,2.,3.]
  @test_throws ArgumentError prox_block!(view(buf,1:2), f, view(buf,2:3), 1, 1.)
  @test buf == [1.,2.,3.]
  @test_throws MethodError prox_block(SLOPE([1.,.5]), 2., 1, 1.)
  @test_throws ArgumentError prox_block!(zeros(2), NormL1(), ones(2), 1, 1.)
end

@testset "nested public preflight remains before writes" begin
  for T in (Float32,Float64)
    groups = [[1,3],[2,4]]
    for f in (GroupNormL2(T(.3),groups),GroupNormLinf(T(.3),groups),
              SparseGroupNormL2(T(.1),T(.3),groups),GroupMCP(T(.3),groups),GroupSCAD(T(.3),groups))
      for h in (Scaled(f,T(2)),Regularize(f,T(.2),ones(T,4)),
                Regularize(Scaled(Regularize(f,T(.1)),T(2)),T(.2)))
        for (input,k,E) in ((ones(T,3),1,DimensionMismatch),(ones(T,2),0,BoundsError),
                            (ones(T,2),3,BoundsError),(ones(Complex{T},2),1,ArgumentError))
          output=fill(T(-7),2)
          @test_throws E prox_block!(output,h,input,k,one(T))
          @test output == fill(T(-7),2)
        end
        buf=T[1,2,3]
        @test_throws ArgumentError prox_block!(view(buf,1:2),h,view(buf,2:3),1,one(T))
        @test buf == T[1,2,3]
      end
    end
  end
  # A valid stored SCAD shape can round to invalid in the call's narrower model.
  f=GroupSCAD(.3,[[1,3],[2,4]];a=2+2.0^-30)
  for h in (f,Scaled(f,2.),Regularize(Scaled(f,2.),.2))
    output=fill(-7f0,2)
    @test_throws ArgumentError prox_block!(output,h,ones(Float32,2),1,1f0)
    @test output == fill(-7f0,2)
  end
end

@testset "independent outputs with public storage inputs" begin
  for T in (Float32, Float64)
    s = SparseIterate(T, 2); s[1] = 3; s[2] = 4
    X = SymmetricSparseIterate(T, 2); coordinates(X)[1] = 3; coordinates(X)[2] = 4
    b = BlockIterate(T, 4, [[1,3], [2,4]]); b[1] = 3; b[3] = 4
    f = GroupNormL2(one(T), [1:2])
    for (owner, u) in ((s, activevalues(s)), (s, coordinates(s)),
                       (X, activevalues(X)), (X, view(coordinates(X), 1:2)),
                       (b, blockview(b, 1)))
      before = owner isa SymmetricSparseIterate ? Matrix(owner) : Vector(owner)
      ids = collect(activecoordinates(owner))
      # Storage-backed inputs can be read. The caller supplies independent output;
      # public activevalues views are not tagged with their iterate's identity.
      out = fill(T(-7), 2)
      @test prox_block!(out, f, u, 1, one(T)) === out
      @test out ≈ (1 - sqrt(T(2))/5) .* T[3,4]
      allocated = prox_block(f, u, 1, one(T))
      @test allocated isa Vector{T} && !Base.mightalias(allocated, u)
      @test allocated == out
      @test (owner isa SymmetricSparseIterate ? Matrix(owner) : Vector(owner)) == before
      @test collect(activecoordinates(owner)) == ids
    end
  end
end

@testset "element failures affect only the numerical buffer" begin
  f = Regularize(GroupNormL2(1., [1:2]), 1., Flaky())
  out = zeros(2)
  owner = BlockIterate(2, [1:2])
  @test_throws ErrorException("boom") prox_block!(out, f, zeros(2), 1, 1.)
  @test out == [1.5,0.]
  @test nactive(owner) == 0 && all(iszero, owner)
end
end
