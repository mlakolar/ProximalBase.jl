module StepChecksTest

# Translate, Tilt and Regularize pass a transformed input to the inner function
# in a buffer of their own: when the inner function rejects a step, the output
# and an input that it aliases are unchanged.

using Test
using ProximalBase
using LinearAlgebra
# GPUArrays' reference backend runs on the CPU and forbids scalar indexing.
using JLArrays

using ..TestUtils

const proximal = (prox!, value_and_prox!)

# the type of the exception that a call on fresh arrays throws, or nothing when it returns
function thrown(op, f, z, γ)
  try
    op(similar(z), f, copy(z), γ)
  catch e
    return typeof(e)
  end
  nothing
end

# the call throws E, with a separate output and with the output aliasing the
# input, and leaves both as they were
function unchanged(op, h, x, γ, E)
  y = fill!(similar(x), 7)
  x0, y0 = copy(x), copy(y)
  @test_throws E op(y, h, x, γ)
  @test isequal(x, x0)
  @test isequal(y, y0)
  z = copy(x)
  @test_throws E op(z, h, z, γ)
  @test isequal(z, x0)
end

const I2 = [1.0 0; 0 1.0]

# inner functions that reject a step
const rejecting = (
  (MoreauEnvelope(NormL1(1.0), 1.0), -2.0),
  (LinearImage(NormL1(1.0), I2, 1.0), -1.0),
  (SumLargest(1, 1.0), -1.0),
  (NegLog(), -1.0),            # at positive points
)

@testset "an inner function that rejects a step leaves the output unchanged" begin
  b, c, ρ = [0.5, 0.25], [0.25, -0.5], 0.25
  shrink(γ) = 1 + γ * ρ
  for (f, γ) in rejecting, T in (Float64, Float32)
    x = T[2, 3]
    # (h, the function it calls, the point and the step of that call, the point
    # up to rounding); Conjugate and LinearImage check the step themselves
    cases = Any[
      (Translate(f, b), f, x .- b, γ),
      (Tilt(f, c), f, x .- γ .* c, γ),
      (Regularize(f, ρ), f, x ./ shrink(γ), γ / shrink(γ)),
      (Regularize(f, ρ, c), f, (x .+ γ * ρ .* c) ./ shrink(γ), γ / shrink(γ)),
      (Translate(Tilt(f, c), b), f, x .- b .- γ .* c, γ),
      (Tilt(Regularize(f, ρ), c), f, (x .- γ .* c) ./ shrink(γ), γ / shrink(γ)),
      (Translate(Scaled(f, 2.0), b), f, x .- b, 2γ),
      (Translate(Precompose(f, 2.0), b), f, 2 .* (x .- b), 4γ),
      (Translate(LinearImage(f, I2, 1.0), b), LinearImage(f, I2, 1.0), x .- b, γ),
    ]
    # The conjugates of MoreauEnvelope and SumLargest accept a negative step.
    f isa Union{MoreauEnvelope, SumLargest} ||
      push!(cases, (Translate(Conjugate(f), b), Conjugate(f), x .- b, γ))
    for (h, g, z, s) in cases, op in proximal
      E = thrown(op, g, T.(z), T(s))
      @test E !== nothing
      unchanged(op, h, x, γ, E)
    end
  end
end

@testset "a rejected array step leaves the output unchanged" begin
  b, c = [0.5, 0.25], [0.25, -0.5]
  x = [2.0, 3.0]
  for (f, γ) in ((NormL2(1.0), [1.0, 1.0]),                       # no array steps
                 (NormL1(1.0), [1.0, 1.0, 1.0]),                  # a step per entry
                 (PointwiseMinimum(NormL1(1.0), NormL2(1.0)), [1.0, 1.0]),
                 (MoreauEnvelope(NormL1(1.0), 1.0), [-2.0, -2.0]))
    tilted = length(γ) == length(x) ? x .- γ .* c : x
    for (h, z) in ((Translate(f, b), x .- b), (Tilt(f, c), tilted)), op in proximal
      E = thrown(op, f, z, γ)
      @test E !== nothing
      unchanged(op, h, x, γ, E)
    end
  end
end

@testset "the reproduction of the issue" begin
  h = Translate(MoreauEnvelope(NormL1(1.0), 1.0), [1.0])
  x = [2.0]
  @test_throws ArgumentError prox!(x, h, x, -2.0)
  @test x == [2.0]
end

@testset "values" begin
  x = [1.5, -0.4, 0.05, -2.0, 0.8]
  b = [0.3, -0.2, 0.1, 0.4, -0.5]
  c = [-0.5, 0.25, -1.0, 0.75, -0.2]     # of the sign opposite to x
  f, γ, ρ = NormL1(0.3), 0.7, 0.5
  steps = [0.7, 0.2, 1.1, 0.5, 0.9]
  # the result of op on a separate output and on the input itself
  function results(op, h, x, γ)
    y = similar(x)
    v = op(y, h, x, γ)
    z = copy(x)
    w = op(z, h, z, γ)
    y, v, z, w
  end
  function check(op, h, x, γ, p, v)
    y, vy, z, vz = results(op, h, x, γ)
    @test y == p && z == p
    op === value_and_prox! && @test vy == v && vz == v
  end

  for s in (γ, steps)
    # Translate: b + prox_{γf}(x - b)
    p = similar(x)
    v = value_and_prox!(p, f, x .- b, s)
    p .+= b
    for op in proximal
      check(op, Translate(f, b), x, s, p, v)
    end
    # Tilt: prox_{γf}(x - γc)
    t = Tilt(f, c, 0.25)
    p = similar(x)
    v = value_and_prox!(p, f, x .- s .* c, s) + dot(c, p) + 0.25
    for op in proximal
      check(op, t, x, s, p, v)
    end
  end

  # Regularize: prox_{γ/(1+γρ) f}((x + γρc) / (1 + γρ)), with γρ ≤ 1
  d = 1 + γ * ρ
  a, w = inv(d), γ * ρ / d
  for (r, z) in ((Regularize(f, ρ), a .* x), (Regularize(f, ρ, c), a .* x .+ w .* c))
    p = similar(x)
    v = value_and_prox!(p, f, z, γ / d)
    v += Regularize(ZeroFunction(), ρ, r.c)(p)
    for op in proximal
      check(op, r, x, γ, p, v)
    end
  end
end

a_prox!(y, f, x, γ) = prox!(y, f, x, γ)
a_vprox!(y, f, x, γ) = value_and_prox!(y, f, x, γ)

@testset "the buffers allocate once" begin
  n = 50
  x = randn(n)
  c = randn(n)
  steps = fill(0.7, n)
  for (h, arraysteps) in ((Translate(NormL1(0.2), c), true), (Tilt(NormL1(0.2), c), true),
                          (Regularize(NormL1(0.2), 0.5), false), (Regularize(NormL1(0.2), 0.5, c), false))
    for γ in (arraysteps ? (0.7, steps) : (0.7,))
      y = similar(x)
      @test allocs(a_prox!, y, h, x, γ) == 0
      @test allocs(a_vprox!, y, h, x, γ) == 0
      z = copy(x)
      @test allocs(a_prox!, z, h, z, γ) == 0
      @test allocs(a_vprox!, z, h, z, γ) == 0
    end
  end
end

@testset "a GPU output reuses its buffer" begin
  x, c = [1.5, -0.4, 0.05, -2.0], [0.3, -0.2, 0.1, 0.4]
  dx, dc = JLArray(x), JLArray(c)
  scratch(h::Translate) = h.proxscratch
  scratch(h) = h.scratch
  for (wrap, dwrap) in ((f -> Translate(f, c), f -> Translate(f, dc)), (f -> Tilt(f, c), f -> Tilt(f, dc)),
                        (f -> Regularize(f, 0.5), f -> Regularize(f, 0.5)),
                        (f -> Regularize(f, 0.5, c), f -> Regularize(f, 0.5, dc)))
    h = dwrap(NormL1(0.3))
    y = JLArray(zeros(4))
    prox!(y, h, dx, 0.7)
    buf = scratch(h).buf
    prox!(y, h, dx, 0.7)
    @test buf isa JLArray && scratch(h).buf === buf
    @test Array(y) == prox(wrap(NormL1(0.3)), x, 0.7)
    bad = dwrap(MoreauEnvelope(NormL1(1.0), 1.0))
    y = JLArray(fill(7.0, 4))
    @test_throws ArgumentError prox!(y, bad, dx, -2.0)
    @test Array(y) == fill(7.0, 4)
    z = copy(dx)
    @test_throws ArgumentError prox!(z, bad, z, -2.0)
    @test Array(z) == x
  end
end

end
