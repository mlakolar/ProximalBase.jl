module SymmetricParametersExtensionsTest

using Test, ProximalBase

const PB=ProximalBase
const calls=zeros(Int,11)
struct PublicSquare <: AbstractFunction end
struct PublicSeparableSquare <: SeparableFunction end
for F in (PublicSquare,PublicSeparableSquare)
  @eval begin
    PB.issmooth(::Type{$F})=true
    PB.isproximable(::Type{$F})=true
    PB.hashessian(::Type{$F})=true
    (f::$F)(x::AbstractArray)=(calls[1]+=1; sum(abs2,x)/2)
    PB.prox!(y::AbstractArray,f::$F,x::AbstractArray,γ::Real)=(calls[2]+=1; y .= x ./ (1+γ))
    PB.value_and_prox!(y::AbstractArray,f::$F,x::AbstractArray,γ::Real)=(calls[3]+=1; prox!(y,f,x,γ); f(y))
    PB.gradient!(y::AbstractArray,f::$F,x::AbstractArray)=(calls[4]+=1; copyto!(y,x))
    PB.value_and_gradient!(y::AbstractArray,f::$F,x::AbstractArray)=(calls[5]+=1; gradient!(y,f,x); f(x))
    PB.hessian_vec!(y::AbstractArray,f::$F,x::AbstractArray,v::AbstractArray)=(calls[6]+=1; copyto!(y,v))
    PB.quadratic_model(f::$F,x)=(calls[7]+=1; f)
    PB.quadratic_model!(q::$F,f::$F,x)=(calls[8]+=1; q)
    PB.coordinate_lipschitz!(L::AbstractArray,f::$F)=(calls[9]+=1; fill!(L,1))
    PB.kkt_violation(f::$F,g::AbstractArray,i::Integer)=(calls[10]+=1; abs(g[i]))
    PB.kkt_violation(f::$F,g::AbstractArray,x::Array,k::Integer)=(calls[10]+=1; calls[11]+=1; abs(g[k]))
    PB.kkt_violation(f::$F,g::AbstractArray,x::SymmetricSparseIterate,k::Integer)=(calls[10]+=1; calls[11]+=1; abs(g[k]))
  end
end
PB.value_entry(::PublicSeparableSquare,v,i)=v^2/2
PB.prox_entry(::PublicSeparableSquare,v,i,γ)=v/(1+γ)

@testset "Unchecked forwarding preserves public extensions" begin
  X=[.5 .2;.2 .8]
  for f in (PublicSquare(),PublicSeparableSquare()), x in (copy(X),SymmetricSparseIterate(X))
    h=Scaled(f,2.); y=copy(x)
    fill!(calls,0)
    @test h(x) ≈ sum(abs2,X)
    @test calls[1] == 1
    fill!(calls,0)
    @test Matrix(prox!(y,h,x,1.)) ≈ X/3
    @test calls[2] == 1
    fill!(calls,0)
    @test value_and_prox!(y,h,x,1.) ≈ sum(abs2,X/3)
    @test calls[3] == 1
    fill!(calls,0)
    @test Matrix(gradient!(y,h,x)) ≈ 2X
    @test calls[4] == 1
    fill!(calls,0)
    @test value_and_gradient!(y,h,x) ≈ sum(abs2,X)
    @test Matrix(y) ≈ 2X
    @test calls[5] == 1
    fill!(calls,0)
    @test Matrix(hessian_vec!(y,h,x,X)) ≈ 2X
    @test calls[6] == 1
    fill!(calls,0)
    q=quadratic_model(h,x)
    @test q(x) ≈ sum(abs2,X)
    @test calls[7] == 1
    fill!(calls,0)
    @test quadratic_model!(q,h,x) === q
    @test calls[8] == 1
    fill!(calls,0)
    @test Matrix(coordinate_lipschitz!(y,h)) == fill(2.,2,2)
    @test calls[9] == 1
    fill!(calls,0)
    @test kkt_violation(h,fill(.4,2,2),x,2) == .4
    @test calls[10] == 1
  end
end


struct ProxOnlySquare <: AbstractFunction end
PB.isproximable(::Type{ProxOnlySquare}) = true
(f::ProxOnlySquare)(x::AbstractArray) = sum(abs2, x) / 2
PB.prox!(y::AbstractArray, ::ProxOnlySquare, x::AbstractArray, γ::Real) = (y .= x ./ (1 + γ))

struct EntrySquare <: SeparableFunction end
PB.value_entry(::EntrySquare, v, i) = v^2 / 2
PB.prox_entry(::EntrySquare, v, i, γ) = v / (1 + γ)
PB.kkt_violation(::EntrySquare, g::AbstractArray, i::Integer) = abs(g[i])
PB.iszeropreserving(::EntrySquare) = true
PB.isconvex(::Type{EntrySquare}) = true
PB.kkt_violation(f::PublicSeparableSquare, g::AbstractArray, x::SparseIterate, k::Integer) =
  (calls[10] += 1; calls[11] += 1; abs(g[k]))

function kkt_loop(f, g, x, n)
  s = 0.0
  for i in 1:n
    s += kkt_violation(f, g, x, mod1(i, ncoordinates(x)))
  end
  s
end
function coordinate_loop(f, x, n)
  for i in 1:n
    prox_coordinate!(x, f, mod1(i, ncoordinates(x)), .1)
  end
end
function loop_allocations(f, g, x)
  kkt_loop(f, g, x, 1000)
  coordinate_loop(f, x, 1000)
  (@allocated(kkt_loop(f, g, x, 1000)), @allocated(coordinate_loop(f, x, 1000)))
end

@testset "User functions using generic public oracles" begin
  X = [.5 .2; .2 .8]
  for x in (copy(X), SparseIterate(X), SymmetricSparseIterate(X))
    f = EntrySquare()
    for (h, scale) in ((f, 1.), (Scaled(f, 2.), 2.),
                       (Tilt(f, zeros(2, 2)), 1.), (Precompose(f, 2.), 4.))
      y = similar(x)
      @test h(x) ≈ scale * sum(abs2, X) / 2
      @test Matrix(prox!(y, h, x, .25)) ≈ X / (1 + .25scale)
      @test value_and_prox!(y, h, x, .25) ≈ scale * sum(abs2, X / (1 + .25scale)) / 2
    end
    @test Sum(f, f)(x) ≈ sum(abs2, X)
  end
  for f in (ProxOnlySquare(), Scaled(ProxOnlySquare(), 2.))
    y = zeros(2, 2)
    scale = f isa Scaled ? 2. : 1.
    @test value_and_prox!(y, f, X, .25) ≈ scale * sum(abs2, X / (1 + .25scale)) / 2
  end
  @test_throws ArgumentError hessian_vec!(zeros(2, 2), EntrySquare(), X, X)
  @test_throws ArgumentError quadratic_model(EntrySquare(), X)
  @test_throws ArgumentError quadratic_model!(EntrySquare(), EntrySquare(), X)
end

@testset "User overrides through entrywise wrappers" begin
  X = [.5 .2; .2 .8]
  f = PublicSeparableSquare()
  for x in (copy(X), SparseIterate(X), SymmetricSparseIterate(X))
    for (h, scale) in ((f, 1.), (Scaled(f, 2.), 2.),
                       (Tilt(f, zeros(2, 2)), 1.), (Precompose(f, 2.), 4.))
      y = similar(x)
      fill!(calls, 0)
      @test Matrix(prox!(y, h, x, .25)) ≈ X / (1 + .25scale)
      @test calls[2] == 1
      fill!(calls, 0)
      @test value_and_prox!(y, h, x, .25) ≈ scale * sum(abs2, X / (1 + .25scale)) / 2
      @test calls[3] == 1
      fill!(calls, 0)
      @test Matrix(gradient!(y, h, x)) ≈ scale * X
      @test calls[4] == 1
      fill!(calls, 0)
      @test value_and_gradient!(y, h, x) ≈ scale * sum(abs2, X) / 2
      @test calls[5] == 1
    end
    h = Sum(f, f); y = similar(x)
    fill!(calls, 0)
    @test h(x) ≈ sum(abs2, X)
    @test calls[1] == 2
    fill!(calls, 0)
    @test value_and_gradient!(y, h, x) ≈ sum(abs2, X)
    @test Matrix(y) ≈ 2X
    @test calls[5] == 2
    fill!(calls, 0)
    @test kkt_violation(f, fill(.4, 2, 2), x, 2) == .4
    @test calls[10] == 1
    @test calls[11] == 1
    for h in (Scaled(f, 2.), Tilt(f, zeros(2, 2)), Precompose(f, 2.))
      fill!(calls, 0)
      @test kkt_violation(h, fill(.4, 2, 2), x, 2) == .4
      @test calls[10] == 1
      @test calls[11] == (h isa Scaled && !(x isa SymmetricSparseIterate) ? 1 : 0)
    end
  end
end

@testset "Warmed coordinate loops do not allocate" begin
  X = [.5 .2; .2 .8]; g = fill(.4, 2, 2)
  for f in (EntrySquare(), PublicSeparableSquare()),
      x in (SparseIterate(X), SymmetricSparseIterate(X))
    for h in (f, Scaled(f, 2.), Tilt(f, zeros(2, 2)), Precompose(f, 2.))
      @test loop_allocations(h, g, copy(x)) == (0, 0)
    end
  end
end


@testset "Built-in group KKT dispatch keeps its domain" begin
  groups = [[1, 2], [3, 4]]
  f = GroupNormL2(.1, groups); g = [.3, .4, .6, .8]
  for x in (zeros(4), SparseIterate(4), AtomIterate(4, groups))
    @test kkt_violation(f, g, x, 1) ≈ .5 - .1sqrt(2)
    @test kkt_violation(Scaled(f, 2.), g, x, 1) ≈ .5 - .2sqrt(2)
    @test kkt_violation(Regularize(f, .2), g, x, 1) ≈ .5 - .1sqrt(2)
  end
  for h in (f, Scaled(f, 2.), Regularize(f, .2))
    @test_throws ArgumentError kkt_violation(h, g, SymmetricSparseIterate(2), 1)
  end
end

struct ArrayStepSquare <: AbstractFunction end
PB.value_and_prox!(y::AbstractArray, ::ArrayStepSquare, x::AbstractArray, γ::AbstractArray) =
  (calls[3] += 1; y .= x ./ (1 .+ γ); sum(abs2, y) / 2)

@testset "Unchecked array-step fallback reaches public extensions" begin
  X = [.5 .2; .2 .8]; γ = ones(2, 2)
  for x in (copy(X), SymmetricSparseIterate(X)),
      h in (Scaled(ArrayStepSquare(), 2.), Translate(ArrayStepSquare(), zeros(2, 2)),
            Tilt(ArrayStepSquare(), zeros(2, 2)))
    fill!(calls, 0); y = similar(x)
    divisor = h isa Scaled ? 3 : 2
    scale = h isa Scaled ? 2 : 1
    @test value_and_prox!(y, h, x, γ) ≈ scale * sum(abs2, X / divisor) / 2
    @test Matrix(y) ≈ X / divisor
    @test calls[3] == 1
  end
end

end
