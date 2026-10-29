module StepChecksTest

# Translate, Tilt and Regularize pass a transformed input to the inner function
# in a buffer of their own: when the inner function rejects a step, the output
# and an input that it aliases are unchanged.

using Test
using ProximalBase
using LinearAlgebra
using Random
# GPUArrays' reference backend runs on the CPU and forbids scalar indexing.
using JLArrays
import Adapt

using ..TestUtils

const proximal = (prox!, value_and_prox!)

# the exception that a call on fresh arrays throws, or nothing when it returns
function thrown(op, f, z, γ)
  @nospecialize
  try
    op(similar(z), f, copy(z), γ)
  catch e
    return e
  end
  nothing
end

function thrown_into(op, h, y, x, γ)
  @nospecialize
  try
    op(y, h, x, γ)
  catch e
    return e
  end
  nothing
end

message(e) = first(split(sprint(showerror, e), '\n'))
same_exception(e, ref) = e isa typeof(ref) && message(e) == message(ref)

# the call throws an exception of the type and with the message of ref, with a
# separate output and with the output aliasing the input, and leaves both as they were
function unchanged(op, h, x, γ, ref)
  @nospecialize
  y = fill!(similar(x), 7)
  x0, y0 = copy(x), copy(y)
  @test same_exception(thrown_into(op, h, y, x, γ), ref)
  @test isequal(x, x0)
  @test isequal(y, y0)
  z = copy(x)
  @test same_exception(thrown_into(op, h, z, z, γ), ref)
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
  for (f, γ) in rejecting, T in (Float64, Float32)
    x = T[2, 3]
    # Regularize computes its step in the precision of the output
    shrink(γ) = 1 + γ * ρ
    rstep(γ) = (g = T(γ); g / (1 + g * T(ρ)))
    # (h, the function it calls, the point and the step of that call, the point
    # up to rounding); Conjugate and LinearImage check the step themselves
    cases = Any[
      (Translate(f, b), f, x .- b, γ),
      (Tilt(f, c), f, x .- γ .* c, γ),
      (Regularize(f, ρ), f, x ./ shrink(γ), rstep(γ)),
      (Regularize(f, ρ, c), f, (x .+ γ * ρ .* c) ./ shrink(γ), rstep(γ)),
      (Translate(Tilt(f, c), b), f, x .- b .- γ .* c, γ),
      (Tilt(Regularize(f, ρ), c), f, (x .- γ .* c) ./ shrink(γ), rstep(γ)),
      (Translate(Scaled(f, 2.0), b), f, x .- b, 2γ),
      (Translate(Precompose(f, 2.0), b), f, 2 .* (x .- b), 4γ),
      (Translate(LinearImage(f, I2, 1.0), b), LinearImage(f, I2, 1.0), x .- b, γ),
    ]
    # The conjugates of MoreauEnvelope and SumLargest accept a negative step.
    f isa Union{MoreauEnvelope, SumLargest} ||
      push!(cases, (Translate(Conjugate(f), b), Conjugate(f), x .- b, γ))
    for (h, g, z, s) in cases, op in proximal
      e = thrown(op, g, T.(z), s)
      @test e !== nothing
      unchanged(op, h, x, γ, e)
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
      e = thrown(op, f, z, γ)
      @test e !== nothing
      unchanged(op, h, x, γ, e)
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

# The computation before the buffers: the transformed input in y, and f in place.
_step(y, γ::Real) = convert(float(eltype(y)), γ)
_step(y, γ) = γ
function old!(y, h::Translate, x, γ)
  @nospecialize
  y .= x .- h.b
  v = value_and_prox!(y, h.f, y, γ)
  y .+= h.b
  v
end
old!(@nospecialize(y), @nospecialize(h::Tilt), @nospecialize(x), @nospecialize(γ)) = (y .= x .- _step(y, γ) .* h.c; value_and_prox!(y, h.f, y, γ) + dot(h.c, y) + h.d)
# with γρ ≤ 1, and a center of the sign opposite to x
function old!(y, h::Regularize, x, γ)
  @nospecialize
  T = float(eltype(y))
  g, ρ = T(γ), T(h.ρ)
  t = g * ρ
  a, w = inv(one(T) + t), t / (one(T) + t)
  h.c === nothing ? (y .= a .* x) : (y .= a .* T.(x) .+ w .* T.(h.c))
  v = value_and_prox!(y, h.f, y, g / (one(T) + t))
  v + Regularize(ZeroFunction(), h.ρ, h.c)(y)
end

@testset "values on views and in another precision, bit for bit" begin
  Random.seed!(7)
  n = 16
  # Float32 outputs on the strided vector view, and NormL1 on the contiguous
  # view, only, to keep compilation short
  l1, l2 = NormL1(0.3), NormL2(0.7)
  outputs = (
    (T -> view(zeros(T, n + 3), 2:n+1), (n,), (Float64,), (l1,)),
    (T -> view(zeros(T, 3n), 1:3:3n), (n,), (Float64, Float32), (l1, l2)),
    (T -> view(zeros(T, 9, 9), 1:2:9, 1:2:8), (5, 4), (Float64,), (l1, l2)),
  )
  for (mk, dims, types, fs) in outputs, trial in 1:2
    x = randn(dims...)
    b = randn(dims...)
    c = -sign.(x) .* (0.1 .+ rand(dims...))
    steps = 0.2 .+ rand(dims...)
    for f in fs
      hs = (Translate(f, b), Tilt(f, c, 0.25), Regularize(f, 0.5), Regularize(f, 0.5, c))
      for h in hs, γ in (h isa Regularize || f isa NormL2 ? (0.7,) : (0.7, steps)), T in types
        y1, y2 = mk(T), mk(T)
        v1, v2 = value_and_prox!(y1, h, x, γ), old!(y2, h, x, γ)
        # the value has the output's type: the reference rounded once
        @test isequal(v1, convert(T, v2)) && isequal(parent(y1), parent(y2))
        y1 = mk(T)
        prox!(y1, h, x, γ)
        @test isequal(parent(y1), parent(y2))
        T === Float64 || continue
        # the output aliasing the input
        y1, y2 = copyto!(mk(T), x), copyto!(mk(T), x)
        v1, v2 = value_and_prox!(y1, h, y1, γ), old!(y2, h, y2, γ)
        @test isequal(v1, v2) && isequal(parent(y1), parent(y2))
      end
    end
  end
end

@testset "a view gets a buffer of its own type" begin
  A, M = randn(12), randn(6, 7)
  h = Translate(NormL1(0.3), zeros(3))
  for y in (view(A, 2:5), view(A, 1:3:12), view(A, 12:-2:1), view(M, :, 2), view(M, Base.OneTo(3), 2:4),
            view(M, 6:-2:1, 7:-3:1), view(M, 2, 3:5), view(A, 5:4), view(A, 3:-1:4), view(A, :))
    h = Translate(NormL1(0.3), zeros(size(y)))
    prox!(y, h, randn(size(y)), 0.7)
    z = ProximalBase._output_buffer!(h.proxscratch, y)
    @test typeof(z) === typeof(y) && size(z) == size(y)
    @test parent(z) === h.proxscratch.buf
  end
  # other indices get an Array
  y = view(A, [1, 3, 4])
  h = Translate(NormL1(0.3), zeros(3))
  prox!(y, h, randn(3), 0.7)
  @test ProximalBase._output_buffer!(h.proxscratch, y) isa Vector{Float64}
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
  # strided views of vectors and matrices
  for (dims, mk) in (((n,), () -> view(zeros(3n), 1:3:3n)), ((5, 4), () -> view(zeros(9, 9), 1:2:9, 1:2:8)))
    x, c, steps = randn(dims...), randn(dims...), fill(0.7, dims)
    for (h, arraysteps) in ((Translate(NormL1(0.2), c), true), (Tilt(NormL1(0.2), c), true),
                            (Regularize(NormL1(0.2), 0.5), false), (Regularize(NormL1(0.2), 0.5, c), false))
      for γ in (arraysteps ? (0.7, steps) : (0.7,))
        y = mk()
        @test allocs(a_prox!, y, h, x, γ) == 0
        @test allocs(a_vprox!, y, h, x, γ) == 0
        z = copyto!(mk(), x)
        @test allocs(a_prox!, z, h, z, γ) == 0
        @test allocs(a_vprox!, z, h, z, γ) == 0
      end
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

@testset "a regularized bits type stays a bits type on the device" begin
  @test isbitstype(typeof(Adapt.adapt(JLArray, Regularize(NormL1(1.0), 0.5))))
  bc = Constrained(Regularize(NormL1(1.0), 0.5), IndBox(0.0, Inf))
  dbc = Adapt.adapt(JLArray, bc)
  @test isbitstype(typeof(dbc))
  x = [1.5, -0.4, 0.05, -2.0, 0.8]
  y = JLArray(zeros(5))
  prox!(y, dbc, JLArray(x), 0.7)
  @test Array(y) == prox(bc, x, 0.7)
  # back on the host, with a buffer
  c = [0.3, -0.2, 0.1, 0.4, -0.5]
  r = Regularize(NormL1(1.0), 0.5, c)
  hr = Adapt.adapt(Array, Adapt.adapt(JLArray, r))
  @test hr.scratch isa ProximalBase.Scratch && hr.c isa Vector{Float64}
  @test prox(hr, x, 0.7) == prox(r, x, 0.7)
end

end
