module SymmetricParametersInventoryTest

using Test, ProximalBase, LinearAlgebra
using ..TestUtils: allocs

const PB = ProximalBase
state(x) = map(n -> deepcopy(getfield(x, n)), fieldnames(typeof(x)))
function rejects(op, y, name, F)
  before = state(y)
  err = try op(y); nothing catch e; e end
  @test err isa ArgumentError
  @test occursin("the array $name of $F must be symmetric", sprint(showerror, err))
  @test occursin("entry (2, 1)", sprint(showerror, err))
  @test isequal(state(y), before)
end

# A scalar objective independent of the package's entry oracles. A grid
# selects an interval; golden section refines that interval.
function scalar_prox(φ, v, a, γ; lo = -3.0, hi = 3.0)
  objective(z) = φ(z, a) + (z - v)^2 / (2γ)
  grid = range(lo, hi; length = 1201)
  j = argmin(objective.(grid))
  z = grid[j]
  l, u = grid[max(1, j - 1)], grid[min(length(grid), j + 1)]
  τ = (sqrt(5.0) - 1) / 2
  for _ in 1:80
    c, d = u - τ * (u - l), l + τ * (u - l)
    objective(c) <= objective(d) ? (u = d) : (l = c)
  end
  m = (l + u) / 2
  objective(m) < objective(z) ? m : z
end

# Each array field gets its own constructor case. Scalar formulas also give
# full-matrix references, so both off-diagonal entries contribute to values.
const cases = (
  (:w, a -> NormL1(.1, a), (v,a) -> .1a * abs(v)),
  (:w, a -> NormL1Nonneg(.1, a), (v,a) -> v < 0 ? Inf : .1a * v),
  (:w, a -> ElasticNet(.1, .2, a), (v,a) -> .1a * abs(v) + .1v^2),
  (:w, a -> CappedL1(.1, 1., a), (v,a) -> .1a * min(abs(v), 1.)),
  (:w, a -> LogPenalty(.1, 1., a), (v,a) -> .1a * log1p(abs(v))),
  (:w, a -> NormLq(.1, .5, a), (v,a) -> .1a * sqrt(abs(v))),
  (:w, a -> Berhu(.1, 1., a), (v,a) -> .1a * (abs(v) <= 1 ? abs(v) : (v^2 + 1)/2)),
  (:w, a -> CheckLoss(.3, a), (v,a) -> a * (.3max(v,0) + .7max(-v,0))),
  (:w, a -> HingeLoss(nothing, a), (v,a) -> a * max(1-v,0)),
  (:y, a -> HingeLoss(a), (v,a) -> max(1-a*v,0)),
  (:w, a -> SqrHingeLoss(nothing, a), (v,a) -> a * max(1-v,0)^2),
  (:y, a -> SqrHingeLoss(a), (v,a) -> max(1-a*v,0)^2),
  (:w, a -> EpsInsensitiveLoss(.2, a), (v,a) -> a * max(abs(v)-.2,0)),
)

@testset "Entry weights and labels with independent references" begin
  X = fill(.6, 3, 3)
  x = SymmetricSparseIterate(X)
  k = tril_index(3, 2, 1)
  for (name, make, φ) in cases
    A = ones(3, 3); A[2,1] = nextfloat(1.)
    f = make(A)
    F = nameof(typeof(f))
    @testset "$F.$name" begin
      @test name in first.(PB._entry_arrays(f))
      rejects(_ -> f(x), copy(x), name, F)
      for op in (prox!, value_and_prox!), step in (.2, fill(.2,3,3))
        applicable(op,x,f,x,step) || continue
        rejects(y -> op(y, f, x, step), copy(x), name, F)
        rejects(y -> op(y, f, X, step), copy(x), name, F)
      end
      if isseparable(f)
        rejects(y -> prox_coordinate!(y, f, k, .2), copy(x), name, F)
      end
      if applicable(kkt_violation, f, zeros(3,3), 1)
        rejects(_ -> kkt_violation(f, zeros(3,3), x, k), copy(x), name, F)
      end
      if issmooth(f)
        for op in (gradient!, value_and_gradient!), point in (x,X)
          rejects(y -> op(y, f, point), copy(x), name, F)
        end
        rejects(y -> coordinate_lipschitz!(y,f),copy(x),name,F)
      end
      if hashessian(f)
        rejects(_ -> quadratic_model(f, x), copy(x), name, F)
        rejects(y -> hessian_vec!(y, f, x, X), copy(x), name, F)
      end
      for B in (Matrix(Symmetric(A, :L)), Symmetric(A, :L))
        h = make(B)
        @test allocs(PB._check_entry_arrays,h,x,PB._AllPairs()) == 0
        @test h(x) ≈ sum(φ(X[i], B[i]) for i in eachindex(X))
        Y = if h isa IndBinary
          name == :l ? fill(-1.,3,3) : Matrix(B)
        else
          [scalar_prox(φ, X[r,c], B[r,c], .2;
                       lo = h isa BernoulliLoss ? 1e-9 : -3.,
                       hi = h isa BernoulliLoss ? 1-1e-9 : 3.) for r in 1:3, c in 1:3]
        end
        # Both binary fixtures are closer to the upper endpoint.
        h isa IndBinary && (Y = name == :l ? fill(2.,3,3) : Matrix(B))
        y = copy(x)
        @test Matrix(prox!(y, h, x, .2)) ≈ Y atol=2e-7
        @test value_and_prox!(y, h, x, .2) ≈ sum(φ(Y[i], B[i]) for i in eachindex(Y)) atol=2e-6
        if applicable(prox!,y,h,x,fill(.2,3,3))
          @test Matrix(prox!(y,h,x,fill(.2,3,3))) ≈ Y atol=2e-7
          @test value_and_prox!(y,h,x,fill(.2,3,3)) ≈ sum(φ(Y[i],B[i]) for i in eachindex(Y)) atol=2e-6
        end
        if isseparable(h)
          z=copy(x)
          @test prox_coordinate!(z,h,k,.2) ≈ Y[2,1] atol=2e-7
        end
        if applicable(kkt_violation,h,X,1)
          δ=1e-7
          dplus=(φ(δ,B[2,1])-φ(0.,B[2,1]))/δ
          dminus=(φ(0.,B[2,1])-φ(-δ,B[2,1]))/δ
          reference=h isa NormLq ? -Inf : max(-.4-dplus,.4+dminus)
          @test kkt_violation(h,fill(.4,3,3),x,k) ≈ reference atol=2e-6
        end
        if issmooth(h)
          D = [(φ(X[r,c]+1e-5,B[r,c])-φ(X[r,c]-1e-5,B[r,c]))/2e-5 for r in 1:3,c in 1:3]
          @test Matrix(gradient!(similar(x), h, x)) ≈ D atol=2e-6
          g = similar(x)
          @test value_and_gradient!(g,h,x) ≈ sum(φ(X[i],B[i]) for i in eachindex(X))
          @test Matrix(g) ≈ D atol=2e-6
          δ=1e-4
          L=h isa BernoulliLoss ? fill(Inf,3,3) :
            [(φ(X[r,c]+δ,B[r,c])-2φ(X[r,c],B[r,c])+φ(X[r,c]-δ,B[r,c]))/δ^2 for r in 1:3,c in 1:3]
          @test Matrix(coordinate_lipschitz!(similar(x),h)) ≈ L atol=2e-6
        end
        if h isa Union{LinearFunction,SqrNormL2}
          H=h isa LinearFunction ? zeros(3,3) : ones(3,3)
          @test Matrix(hessian_vec!(similar(x),h,x,ones(3,3))) == H
          @test quadratic_model(h,x)(x) ≈ sum(φ(X[i],B[i]) for i in eachindex(X))
        end
      end
      # The dense output keeps ordinary per-entry semantics.
      @test prox!(zeros(3,3), f, x, .2) == prox!(zeros(3,3), f, X, .2)
    end
  end
end

end
