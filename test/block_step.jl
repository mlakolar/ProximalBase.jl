module BlockStepTest

using Test
using ProximalBase
using SparseArrays

struct Flaky <: AbstractVector{Float64} end
Base.size(::Flaky) = (2,)
Base.getindex(::Flaky, i::Int) = i == 1 ? 3.0 : error("boom")

struct CustomGroupPenalty <: ProximalBase.AbstractGroupPenalty
  groups::Vector{UnitRange{Int}}
end

function ProximalBase.prox_coordinate!(x::AtomIterate, f::CustomGroupPenalty, k::Integer, γ::Real)
  ProximalBase._check_group_coordinate(f, x, k)
  atom(x, k) ./= 1 + γ
  atom(x, k)
end

function consistent(x::AtomIterate)
  act = collect(activecoordinates(x))
  length(unique(act)) == nactive(x) &&
    count(!iszero, x.position) == nactive(x) &&
    all(a -> x.position[act[a]] == a, eachindex(act)) &&
    all(k -> isactive(x, k) || all(iszero, x.storage[x.groups[k]]), eachindex(x.groups))
end
consistent(x) = all(j -> x.ind2nzval[x.nzval2ind[j]] == j, 1:nnz(x))
state(x::AtomIterate) = (copy(x.storage), nactive(x), collect(activecoordinates(x)), copy(x.position))
state(x) = (Array(x), nnz(x), collect(activecoordinates(x)), copy(x.ind2nzval))
nested_scaled(f, α) = (s = Scaled(f, α); Scaled{typeof(s), typeof(α)}(s, α))

function check_block(x::AtomIterate, f, k, u, γ)
  x[x.groups[2][1]] = 2
  before, trial = Vector(x), copy(u)
  G = x.groups[k]
  dense = copy(before)
  dense[G] .= eltype(x).(u)
  expected = prox(f, dense, γ)[G]
  written = copy(x)
  atom(written, k) .= u
  prox_coordinate!(written, f, k, γ)
  result = prox_coordinate!(x, f, k, u, γ)
  @test result ≈ expected
  @test result == atom(written, k)
  @test parent(result) === x
  @test Vector(x)[setdiff(1:length(x), G)] == before[setdiff(1:length(x), G)]
  @test u == trial
  @test consistent(x) && consistent(written)
  result[1] = one(eltype(x))
  @test isactive(x, k) && consistent(x)
end

function check_block(x, f, k, u, γ)
  x[4] = 2
  before = Array(x)
  written = copy(x)
  if x isa SymmetricSparseIterate
    r, c = tril_cartesian(x.p, k)
    i = (c - 1) * x.p + r
    coordinates(written)[k] = u
  else
    i = k
    written[k] = u
  end
  expected = prox_entry(f, convert(eltype(x), u), i, γ)
  reference = prox_coordinate!(written, f, k, γ)
  result = prox_coordinate!(x, f, k, u, γ)
  @test result == expected == reference
  dense = copy(before)
  dense[i] = convert(eltype(x), expected)
  if x isa SymmetricSparseIterate
    dense[c, r] = convert(eltype(x), expected)
  end
  @test Array(x) == dense
  @test consistent(x) && consistent(written)
end

@testset "block steps" begin
  @testset "references" begin
    for T in (Float32, Float64)
      fs = (NormL1(T(0.3)), ElasticNet(T(0.3), T(0.2)), IndBox(T(1), T(2)),
            Translate(NormL1(T(0.3)), T[1, 2, 2, 4]))
      scalarfs = (fs..., Scaled(fs[1], T(2)), nested_scaled(fs[1], T(2)),
                  Regularize(fs[1], T(0.2)), Regularize(fs[1], T(0.2), T[1, 2, 2, 4]))
      for f in scalarfs
        for x in (SparseIterate(zeros(T, 4)), SparseIterate(zeros(T, 2, 2)))
          check_block(x, f, 2, T(3), T(0.7))
        end
        for k in (1, 2)
          check_block(SymmetricSparseIterate(zeros(T, 2, 2)), f, k, T(3), T(0.7))
        end
      end
      for gs in ([1:2, 3:4], [[1, 3], [2, 4]], [[4, 2], [3, 1]])
        groupfs = (GroupNormL2(T(0.3), gs), GroupNormLinf(T(0.3), gs),
                   SparseGroupNormL2(T(0.1), T(0.3), gs), GroupMCP(T(0.3), gs), GroupSCAD(T(0.3), gs))
        for f in (fs..., groupfs...)
          check_block(AtomIterate(zeros(T, 4), gs), f, 1, T[3, -4], T(0.7))
        end
        for f in (fs[1], groupfs[1])
          for h in (Scaled(f, T(2)), nested_scaled(f, T(2)),
                    Regularize(f, T(0.2)), Regularize(f, T(0.2), T[1, 2, 3, 4]))
            check_block(AtomIterate(zeros(T, 4), gs), h, 1, T[3, -4], T(0.7))
          end
        end
      end
    end
  end

  @testset "activation and order" begin
    gs = [1:2, 3:4, 5:6]
    fs = (NormL1(1.), GroupNormL2(1., gs), GroupNormLinf(1., gs),
          SparseGroupNormL2(0.2, 1., gs), GroupMCP(1., gs), GroupSCAD(1., gs),
          Regularize(GroupNormL2(1., gs), 0.3, ones(6)))
    for f in fs
      x = AtomIterate(6, gs)
      activate!(x, 3)
      activate!(x, 2)
      order = collect(activecoordinates(x))
      prox_coordinate!(x, f, 1, [0.01, 0.01], 1.)
      @test !isactive(x, 1) && nactive(x) == 2
      @test activecoordinates(x) == order && consistent(x)
      prox_coordinate!(x, f, 1, [10., 10.], 1.)
      @test isactive(x, 1) && activecoordinates(x) == [3, 2, 1] && consistent(x)
      prox_coordinate!(x, f, 1, zeros(2), 1.)
      @test isactive(x, 1) && all(iszero, atom(x, 1)) && consistent(x)
    end
    for x in (SparseIterate(zeros(4)), SparseIterate(zeros(2, 2)), SymmetricSparseIterate(2))
      for k in (1, 2)
        order = collect(activecoordinates(x))
        n = nnz(x)
        prox_coordinate!(x, NormL1(1.), k, 0.1, 1.)
        @test nnz(x) == n && activecoordinates(x) == order && consistent(x)
        prox_coordinate!(x, NormL1(1.), k, 3., 1.)
        @test nnz(x) == n + 1 && consistent(x)
        prox_coordinate!(x, NormL1(1.), k, 0., 1.)
        @test nnz(x) == n + 1 && consistent(x)
      end
    end
  end

  @testset "eager group staging on floating-point iterates" begin
    gs = [1:2, 3:4, 5:6]
    fs = (GroupNormL2(1., gs), GroupNormLinf(1., gs), SparseGroupNormL2(0.2, 1., gs),
          GroupMCP(1., gs), GroupSCAD(1., gs))
    for f in fs
      for h in (f, Regularize(f, 0.3), Regularize(f, 0.3, ones(6)))
        for u in ([1//100, 1//100], [10//1, -10//1]), active in (false, true)
          x = AtomIterate(6, gs)
          activate!(x, 3)
          activate!(x, 2)
          active && activate!(x, 1)
          y = copy(x)
          prox_coordinate!(x, h, 1, u, 1.)
          prox_coordinate!(y, h, 1, Float64.(u), 1.)
          @test x == y
          @test nactive(x) == nactive(y) && x.position == y.position
          @test activecoordinates(x) == activecoordinates(y)
          @test consistent(x) && consistent(y)
        end
      end
      for (c, nonzero) in ((0.01, false), (10., true))
        x = AtomIterate(6, gs)
        activate!(x, 3)
        activate!(x, 2)
        r = Regularize(f, 1., fill(c, 6))
        expected = prox(r, Vector(x), 1.)[gs[1]]
        prox_coordinate!(x, r, 1, 1.)
        @test atom(x, 1) ≈ expected
        @test isactive(x, 1) == nonzero
        @test activecoordinates(x) == (nonzero ? [3, 2, 1] : [3, 2])
        @test consistent(x)
      end
    end
  end

  @testset "group staging reads and stored zeros" begin
    x = AtomIterate(2, [1:2])
    @test_throws ErrorException("boom") prox_coordinate!(x, GroupNormL2(1., x.groups), 1, Flaky(), 1.)
    @test consistent(x)
    @test x.storage == [3., 0.] && isactive(x, 1)

    q = nextfloat(0f0)
    for active in (false, true)
      x = AtomIterate(Float32, 4, [1:4])
      active && activate!(x, 1)
      prox_coordinate!(x, GroupNormL2(3q, x.groups, Float32[1]), 1, fill(2q, 4), 1f0)
      @test all(iszero, x.storage)
      @test isactive(x, 1) == active && consistent(x)

      x = AtomIterate(Float32, 4, [1:4])
      active && activate!(x, 1)
      f = GroupNormL2(6q, x.groups, Float32[1])
      r = Regularize(f, 1f0, fill(4q, 4))
      prox_coordinate!(x, r, 1, 1f0)
      @test all(iszero, x.storage)
      @test isactive(x, 1) == active && consistent(x)
    end

    gs = [1:2, 3:4, 5:6]
    buf = [3., -4., 0.]
    u = view(buf, 1:2)
    @test u isa StridedVector
    for f in (GroupNormL2(1., gs), GroupNormLinf(1., gs), SparseGroupNormL2(0.2, 1., gs),
              GroupMCP(1., gs), GroupSCAD(1., gs))
      x = AtomIterate(6, gs)
      activate!(x, 3)
      activate!(x, 2)
      y = copy(x)
      prox_coordinate!(x, f, 1, u, 1.)
      prox_coordinate!(y, f, 1, copy(u), 1.)
      @test state(x) == state(y) && consistent(x)
      @test buf == [3., -4., 0.]
    end
  end

  @testset "zero separable steps preserve active order" begin
    for trial in (false, true)
      x = AtomIterate(6, [1:2, 3:4, 5:6])
      activate!(x, 3)
      activate!(x, 2)
      order = collect(activecoordinates(x))
      if trial
        prox_coordinate!(x, NormL1(1.), 1, [0.1, -0.1], 1.)
      else
        prox_coordinate!(x, NormL1(1.), 1, 1.)
      end
      @test activecoordinates(x) == order
      @test !isactive(x, 1) && consistent(x)
    end
  end

  @testset "four-argument wrappers support custom group penalties" begin
    gs = [1:2, 3:4, 5:6]
    f = CustomGroupPenalty(gs)
    initial = [3., -6., 0., 0., 2., 0.]
    for h in (Scaled(f, 2.), Regularize(f, 1.), Regularize(f, 1., ones(6)))
      x = AtomIterate(initial, gs)
      y = copy(x)
      if h isa Scaled
        prox_coordinate!(y, f, 1, 2.)
      else
        atom(y, 1) .= h.c === nothing ? initial[gs[1]] ./ 2 : (initial[gs[1]] .+ 1) ./ 2
        prox_coordinate!(y, f, 1, 0.5)
      end
      result = prox_coordinate!(x, h, 1, 1.)
      @test state(x) == state(y) && consistent(x)
      @test parent(result) === x && result == atom(x, 1)
      before = state(x)
      msg = "$(typeof(f)) is not separable: prox_coordinate! needs a separable function or a group penalty on matching groups"
      @test_throws ArgumentError(msg) prox_coordinate!(x, h, 1, [4., -8.], 1.)
      @test state(x) == before && consistent(x)
    end
  end

  @testset "rational regularized group steps" begin
    gs = [1:2, 3:4, 5:6]
    f = GroupNormL2(1., gs, ones(3))
    for c in (nothing, [0.1, 0.1, 0., 0., 0., 0.], [6., 8., 0., 0., 0., 0.])
      x = AtomIterate(Rational{Int}, 6, gs)
      activate!(x, 3)
      activate!(x, 2)
      y = AtomIterate(Float64, 6, gs)
      activate!(y, 3)
      activate!(y, 2)
      r = Regularize(f, 1., c)
      prox_coordinate!(x, r, 1, 1.)
      prox_coordinate!(y, r, 1, 1.)
      @test Vector(x) == Rational{Int}.(Vector(y))
      @test activecoordinates(x) == activecoordinates(y)
      @test activecoordinates(x) == (c !== nothing && c[1] == 6 ? [3, 2, 1] : [3, 2])
      @test consistent(x)
    end
    x = AtomIterate(Rational{Int}, 6, gs)
    activate!(x, 3)
    activate!(x, 2)
    prox_coordinate!(x, Regularize(f, 1.), 1, [1//10, 1//10], 1.)
    @test all(iszero, atom(x, 1)) && !isactive(x, 1)
    @test activecoordinates(x) == [3, 2] && consistent(x)
  end

  @testset "steps at zero" begin
    for f in (IndBox(1., 2.), Translate(NormL1(0.3), ones(4)))
      for x in (SparseIterate(zeros(4)), SymmetricSparseIterate(2), AtomIterate(4, [1:2, 3:4]))
        u = x isa AtomIterate ? zeros(2) : 0.
        prox_coordinate!(x, f, 1, u, 1.)
        @test !isempty(activecoordinates(x)) && consistent(x)
        @test any(!iszero, Array(x))
      end
    end
  end

  @testset "stored precision" begin
    f = LinearFunction(fill(1e-100, 4))
    for x in (AtomIterate(zeros(Float32, 4), [1:2, 3:4]), SparseIterate(zeros(Float32, 4)))
      u = x isa AtomIterate ? zeros(2) : 0.
      prox_coordinate!(x, f, 1, 1.)
      @test isempty(activecoordinates(x)) && consistent(x)
      prox_coordinate!(x, f, 1, u, 1.)
      @test isempty(activecoordinates(x)) && consistent(x)
    end
    x = AtomIterate(zeros(Float32, 4), [1:2, 3:4])
    y = copy(x)
    u = [1.123456789, -2.987654321]
    prox_coordinate!(x, NormL1(0.3), 1, u, 0.7)
    prox_coordinate!(y, NormL1(0.3), 1, Float32.(u), 0.7)
    @test x == y && consistent(x)
  end

  @testset "pre-write errors" begin
    x = AtomIterate([0., 0., 2., 0.], [1:2, 3:4])
    bad = GroupNormL2(1., [[1, 3], [2, 4]])
    arrayf = Precompose(GroupNormL2(1., x.groups), ones(4))
    for f in (bad, NormL2(1.), IndBallGroupL0(1, x.groups), arrayf)
      before = state(x)
      err4 = try prox_coordinate!(x, f, 1, 1.); nothing catch e; e end
      err5 = try prox_coordinate!(x, f, 1, ones(2), 1.); nothing catch e; e end
      @test err4 isa Exception && typeof(err4) == typeof(err5)
      @test sprint(showerror, err4) == sprint(showerror, err5)
      @test state(x) == before && consistent(x)
    end
    for (f, k, u, E) in ((NormL1(1.), 1, ones(3), DimensionMismatch),
                         (NormL1(1.), 3, ones(2), BoundsError),
                         (bad, 3, ones(3), DimensionMismatch))
      before = state(x)
      @test_throws E prox_coordinate!(x, f, k, u, 1.)
      @test state(x) == before && consistent(x)
    end
    for z in (x, SparseIterate(zeros(4)), SymmetricSparseIterate(2))
      u = z isa AtomIterate ? ones(2) : 1.
      before = state(z)
      @test_throws DimensionMismatch prox_coordinate!(z, NormL1(1., ones(3)), 20, u, 1.)
      @test_throws ArgumentError prox_coordinate!(z, NormL2(1.), 20, u, 1.)
      @test state(z) == before && consistent(z)
    end
    for x in (x, SparseIterate(zeros(4)), SymmetricSparseIterate(2)), imag in (0., 1.)
      u = x isa AtomIterate ? fill(complex(1., imag), 2) : complex(1., imag)
      before = state(x)
      @test_throws ArgumentError("NormL1 requires real entries") prox_coordinate!(x, NormL1(1.), 1, u, 1.)
      @test state(x) == before && consistent(x)
      @test_throws BoundsError prox_coordinate!(x, NormL1(1.), 20, u, 1.)
      @test state(x) == before && consistent(x)
    end
  end

  @testset "computation failures" begin
    for u in ([1, 0.5], [0.5, 1])
      x = AtomIterate(zeros(Int, 4), [1:2, 3:4])
      @test_throws InexactError prox_coordinate!(x, NormL1(0.), 1, u, 1.)
      @test consistent(x)
      x = AtomIterate(zeros(Int, 4), [1:2, 3:4])
      @test_throws InexactError prox_coordinate!(x, GroupNormL2(0., x.groups), 1, u, 1.)
      @test consistent(x)
    end
    x = AtomIterate(zeros(Int, 4), [1:2, 3:4])
    r = Regularize(GroupNormL2(0.3, x.groups), 1., [2., 1., 0., 0.])
    @test_throws InexactError prox_coordinate!(x, r, 1, 1.)
    @test isactive(x, 1) && consistent(x)
    x = AtomIterate(zeros(Int, 4), [1:2, 3:4])
    @test_throws InexactError prox_coordinate!(x, GroupNormL2(0.3, x.groups), 1, [3, 4], 1.)
    @test consistent(x)
    for x in (AtomIterate(4, [1:2, 3:4]), SparseIterate(zeros(4)), SymmetricSparseIterate(2))
      u = x isa AtomIterate ? [3., 4.] : 3.
      @test_throws ArgumentError prox_coordinate!(x, MoreauEnvelope(NormL1(1.), 1.), 1, u, -1.)
      @test consistent(x)
    end
  end

  @testset "aliased trial points" begin
    for f in (NormL1(0.3), GroupNormL2(0.3, [1:2, 3:4]))
      for source in (x -> atom(x, 1), x -> atom(x, 2), x -> view(x, 2:3),
                     x -> view(x.storage, 1:2))
        x = AtomIterate([1., 2., 3., 4.], [1:2, 3:4])
        y = copy(x)
        u = source(x)
        prox_coordinate!(y, f, 1, copy(u), 0.7)
        prox_coordinate!(x, f, 1, u, 0.7)
        @test x == y && consistent(x)
      end
    end
  end
end

end
