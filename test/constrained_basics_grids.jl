module ConstrainedBasicsGridsTest

# The cases of constrained_basics.jl that repeat cases which that file keeps. They are the Float32 passes of five of the
# six representations that take the fused value and of the parameter types on a vector (the file keeps the Float32 vector
# and the Float64 passes), the destinations of Rational, Real, AbstractFloat, BigFloat and Bool element types, and the
# refusal of fifteen of the eighteen forms of views, reshapes and other representations (the file keeps three of them).
# The testsets have the names that they have in that file; the helpers are copies of its helpers.

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays

using ..TestUtils

const PB = ProximalBase


# The value of the conjugate of ElasticNet(λ1, λ2, w) at y, from the explicit formula in BigFloat: for
# λ2 > 0, Σᵢ max(|yᵢ| - λ1 wᵢ, 0)² / (2λ2); for λ2 = 0, the indicator of |yᵢ| ≤ λ1 wᵢ, within the
# relative tolerance sqrt(eps) that the package gives to the values of indicators.
function elastic_conjugate_reference(λ1, λ2, w, y)
  S = BigFloat
  level(i) = S(λ1) * (w === nothing ? one(S) : S(w[i]))
  if λ2 > 0
    sum((max(abs(S(y[i])) - level(i), 0)^2 for i in eachindex(y)); init = zero(S)) / (2 * S(λ2))
  else
    tol = sqrt(S(eps(float(eltype(y)))))
    all(i -> abs(S(y[i])) - level(i) <= tol * max(1, level(i)), eachindex(y)) ? zero(S) : S(Inf)
  end
end

function check_elastic_value(v, T, ref)
  r = T(ref)
  @test v isa T
  if isinf(r)
    @test v == r
  elseif iszero(r)
    @test v == 0
  else
    @test isapprox(v, r; rtol = 1000 * eps(T))
  end
end

# true when the call throws the refusal of a value that is not available
refused_value(f, args...) = try
  f(args...)
  false
catch e
  e isa ArgumentError && occursin("is not implemented", e.msg)
end
snap(a) = collect(a)
snap(a::AtomIterate) = copy(a.storage)
snap(a::SymmetricSparseIterate) = Matrix(a)
snap(t::Tuple) = map(snap, t)

# The seven role cases of a representation `arr(val)` that gives a 2×2 array holding `val`: it is the output, the input, the
# lower or the upper bound, the array step and the weights (for both ridge levels) of the fused value. A case is
# `nothing` when Base refuses to build the function with that representation.
function role_cases(arr)
  box = IndBox(-2.0, 2.0)
  x2 = fill(5.0, 2, 2)
  dest() = fill(-77.0, 2, 2)
  base = Constrained(Conjugate(ElasticNet(1.0, 0.5)), box)
  attempt(f) = try
    f()
  catch
    nothing
  end
  specs = (("output", () -> (base, arr(-77.0), x2, 0.5)),
           ("input", () -> (base, dest(), arr(5.0), 0.5)),
           ("lower", () -> (Constrained(Conjugate(ElasticNet(1.0, 0.5)), IndBox(arr(-2.0), 2.0)), dest(), x2, 0.5)),
           ("upper", () -> (Constrained(Conjugate(ElasticNet(1.0, 0.5)), IndBox(-2.0, arr(2.0))), dest(), x2, 0.5)),
           ("step", () -> (base, dest(), x2, arr(0.5))),
           ("weights", () -> (Constrained(Conjugate(ElasticNet(1.0, 0.5, arr(1.0))), box), dest(), x2, 0.5)),
           ("weights, λ2 = 0", () -> (Constrained(Conjugate(ElasticNet(1.0, 0.0, arr(1.0))), box), dest(), x2, 0.5)))
  [(role, attempt(spec)) for (role, spec) in specs]
end
dest_snapshot(y) = y isa SubArray ? copy(parent(y)) : copy(y)
# the predicate says no without throwing, and the call refuses before writing: directly, in place and as a later block
function check_refused_case(role, case)
  c, y, x, γ = case
  before = dest_snapshot(y)
  @test PB._elastic_value_route(c, y, x, γ) === false
  @test refused_value(value_and_prox!, y, c, x, γ)
  @test isequal(dest_snapshot(y), before)
  if γ isa Real
    s = SeparableSum(SqrNormL2(2.0), c)
    dest = ([7.0], deepcopy(y))
    before2 = dest_snapshot(dest[2])
    @test refused_value(value_and_prox!, dest, s, ([3.0], x), γ)
    @test dest[1] == [7.0] && isequal(dest_snapshot(dest[2]), before2)
  end
  if role == "output"
    z = deepcopy(y)
    filled = try
      copyto!(z, x)
      true
    catch
      false          # Base cannot store into this representation: there is no in-place fixture
    end
    if filled
      zb = dest_snapshot(z)
      @test refused_value(value_and_prox!, z, c, z, γ)
      @test isequal(dest_snapshot(z), zb)
    end
  end
end

function check_fused(c, mkdest, x, γ; alias = nothing)
  f = c.f.f
  T = eltype(mkdest())
  expected = prox!(mkdest(), c, x, γ)
  y = mkdest()
  v = value_and_prox!(y, c, x, γ)
  @test snap(y) == snap(expected)
  w = f.w === nothing ? nothing : collect(f.w)
  check_elastic_value(v, T, elastic_conjugate_reference(f.λ1, f.λ2, w, snap(y)))
  if alias !== nothing
    z = alias()
    v = value_and_prox!(z, c, z, γ)
    @test snap(z) == snap(expected)
    check_elastic_value(v, T, elastic_conjugate_reference(f.λ1, f.λ2, w, snap(z)))
  end
end

@testset "representations that take the fused value" begin
  for T in (Float32,)
    vx = T[5, -5, 0.5, 1.2]
    mx = T[5 -5; 0.5 1.2]
    S3 = T[5 1 0; 1 -5 0; 0 0 0.5]
    N3 = T[5 2 0; 0 -5 0; 0 1 0.5]
    dense(mk, data) = () -> (y = mk(); copyto!(y, data); y)
    reps = (
      ("vector", () -> zeros(T, 4), vx, dense(() -> zeros(T, 4), vx), false),
      ("matrix", () -> zeros(T, 2, 2), mx, dense(() -> zeros(T, 2, 2), mx), false),
      ("reshape of a Vector (an Array)", () -> reshape(zeros(T, 4), 2, 2), mx, dense(() -> reshape(zeros(T, 4), 2, 2), mx), false),
      ("SparseIterate, Vector x", () -> SparseIterate(T, 4), vx, () -> SparseIterate(sparse(vx)), false),
      ("SparseIterate, SparseIterate x", () -> SparseIterate(T, 4), SparseIterate(sparse(vx)), () -> SparseIterate(sparse(vx)), false),
      ("vector, Int x", () -> zeros(T, 4), [5, -5, 1, 2], nothing, false))
    # weights of the shape of the output, in each admitted element type and wrapper; symmetric for a symmetric output
    weight(kind, n, sym) = begin
      base = sym ? [1 2 0; 2 1 1; 0 1 3] : [1, 0, 2, 1, 3, 1, 0, 1, 2][1:n]
      kind === :f64 ? Float64.(base) : kind === :f32 ? Float32.(base) : Int.(base)
    end
    for (name, mkdest, x, mkalias, sym) in reps[2:end]
      n = length(mkdest())
      sz = size(mkdest())
      @testset "$name, $T" begin
        for (λ1, λ2) in ((T(0.25), T(0.7)), (T(0.25), T(0))), kind in (:none, :f64, :f32, :int),
            box in (IndBox(T(-2), T(2)), IndBox(fill(T(-2), sz), fill(T(2), sz)), IndBox(fill(-2.0, sz), fill(2.0, sz)), IndNonnegative()),
            γ in (T(0.5), fill(T(0.5), sz))
          w = kind === :none ? nothing : weight(kind, n, sym)
          c = Constrained(Conjugate(ElasticNet(λ1, λ2, w)), box)
          check_fused(c, mkdest, x, γ; alias = mkalias)
        end
      end
    end

    @testset "parameter types on a vector, $T" begin
      xs = T[5, -5, 0.5, 1.2]
      mk = () -> zeros(T, 4)
      al = () -> copy(xs)
      wdense = [1.0, 0, 2, 0.5]
      levels = ((T(0.25), T(0.7)), (0.25, 0.7), (0.25f0, 0.7f0), (1, 1), (0.25, 0.0), (1, 0), (0.25f0, 0f0))
      weights = (nothing, wdense, Float32.(wdense), [1, 0, 2, 1], Matrix(reshape(wdense, 2, 2)), Matrix(reshape(wdense, 4, 1)))
      for (λ1, λ2) in levels, w in weights
        check_fused(Constrained(Conjugate(ElasticNet(λ1, λ2, w)), IndBox(T(-2), T(2))), mk, xs, T(0.5); alias = al)
      end
      steps = (0.5, 0.5f0, 1, T(0.5), [0.5, 0.25, 1, 2], Float32[0.5, 0.25, 1, 2], [1, 1, 2, 2])
      bounds = (IndBox(-2.0, 2.0), IndBox(-2, 2), IndBox(T(-2), T(2)), IndBox([-2.0, -1, -3, -2], [2.0, 1, 3, 2]),
                IndBox([-2, -1, -3, -2], [2, 1, 3, 2]), IndBox(Float32[-2, -1, -3, -2], Float32[2, 1, 3, 2]), IndNonnegative())
      for γ in steps, box in bounds, (λ1, λ2) in ((T(0.25), T(0.7)), (T(0.25), T(0)))
        check_fused(Constrained(Conjugate(ElasticNet(λ1, λ2)), box), mk, xs, γ; alias = al)
      end
    end
  end
end

@testset "representations that keep the refusal" begin
  c0 = Constrained(Conjugate(ElasticNet(1.0, 0.5)), IndBox(-2.0, 2.0))
  x = [5.0, -5.0, 0.5]
  # destinations
  function refusal(c, mkdest, xdata, γ; mkalias = nothing)
    y = mkdest()
    before, xb = snap(y), snap(xdata)
    @test refused_value(value_and_prox!, y, c, xdata, γ)
    @test isequal(snap(y), before) && isequal(snap(xdata), xb)
    if γ isa Real
      s = SeparableSum(SqrNormL2(2.0), c)
      dest, point = ([7.0], mkdest()), ([3.0], xdata)
      db = snap(dest)
      @test refused_value(value_and_prox!, dest, s, point, γ)
      @test isequal(snap(dest), db) && isequal(snap(point), ([3.0], xb))
    end
    if mkalias !== nothing
      z = mkalias()
      zb = snap(z)
      @test refused_value(value_and_prox!, z, c, z, γ)
      @test isequal(snap(z), zb)
    end
  end
  for (name, mk, xdata, mkalias) in (("Rational", () -> fill(-7 // 1, 3), x, () -> [5 // 1, -5 // 1, 1 // 2]),
                                     ("Real", () -> Real[-7, -7, -7], x, () -> Real[5, -5, 1]),
                                     ("AbstractFloat", () -> AbstractFloat[-7f0, -7.0, -7f0], x, () -> AbstractFloat[5f0, -5.0, 1f0]),
                                     ("BigFloat", () -> fill(big(-7.0), 3), x, () -> big.([5.0, -5.0, 0.5])),
                                     ("Bool", () -> fill(true, 3), x, () -> fill(true, 3)))
    @testset "destination $name" begin
      refusal(c0, mk, xdata, 0.5; mkalias)
    end
  end
  @testset "views, reshapes and other representations are refused in every role" begin
    U = Union{Int8, Int16}
    forms = (("strided view", v -> view(fill(v, 4, 4), 1:2:3, 1:2:3)),
             ("colon view", v -> view(fill(v, 2, 3), :, 1:2)),
             ("scalar-index view", v -> view(fill(v, 2, 2, 2), 1, :, :)),
             ("view of a reshape", v -> view(reshape(fill(v, 8), 2, 4), 1:2, 2:3)),
             ("UInt8-range view", v -> view(fill(v, 3, 3), UInt8(1):UInt8(2), 1:2)),
             ("Int8 step-range view", v -> view(fill(v, 4, 4), Int8(1):Int8(2):Int8(3), 1:2)),
             ("OneTo view", v -> view(fill(v, 3, 3), Base.OneTo(2), 1:2)),
             ("reshape of a view", v -> reshape(view(fill(v, 6), 2:5), 2, 2)),
             ("index-vector view", v -> view(fill(v, 3, 3), [1, 2], 1:2)),
             ("repeated-index view", v -> view(fill(v, 3, 3), [1, 1], 1:2)),
             ("view of a view", v -> SubArray(view(fill(v, 3, 3), 1:2, 1:3), (1:2, 1:2))),
             ("zero-step range", v -> view(fill(v, 3, 3), StepRangeLen(1, 0, 2), 1:2)),
             ("StepRange{Union} view", v -> view(fill(v, 3, 3), StepRange{U, Int8}(Int8(1), Int8(1), Int16(2)), 1:2)),
             ("adjoint", v -> fill(v, 2, 2)'),
             ("view of a Union-element Array", v -> view(Array{Union{Float64, Int}}(fill(v, 3, 3)), 1:2, 1:2)))
    unbuilt = String[]
    for (name, arr) in forms
      built = try
        arr(0.0) isa AbstractArray
      catch
        false
      end
      built || (push!(unbuilt, name); continue)
      for (role, case) in role_cases(arr)
        if case === nothing
          push!(unbuilt, name * ": " * role)
        else
          @testset "$name, $role" check_refused_case(role, case)
        end
      end
    end
  end
end

end
