module WrapperBuffersTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using ..TestUtils

wrappers(f, c) = (Tilt(f, c), Translate(f, c), Regularize(f, 0.5, c))

# Compare storage order directly; findnz sorts the stored indices.
state(x::Union{SparseIterate, SymmetricSparseIterate}) =
  (copy(x.nzval[1:x.nnz]), copy(x.nzval2ind[1:x.nnz]), copy(x.ind2nzval), x.nnz)
state(x::AtomIterate) =
  (copy(x.storage), copy(x.active), copy(x.position), x.nactive)

function failure(op, y, f, x, γ)
  try
    op(y, f, x, γ)
  catch e
    return e
  end
  nothing
end

function stored(T, symmetric)
  y = symmetric ? SymmetricSparseIterate(T, 8) : SparseIterate(T, 64)
  y[9] = 3
  y[1] = 0.25
  y[5] = 1
  y[5] = 0
  y
end

# The original path transformed y, called f in place, then added the translation.
function original!(op, y, f, x, γ)
  if f isa Tilt
    y .= x .- (γ isa Real ? convert(float(eltype(y)), γ) : γ) .* f.c
    v = op(y, f.f, y, γ)
    return op === prox! ? y : ProximalBase._value_at(v + dot(f.c, y) + f.d, y)
  elseif f isa Translate
    y .= x .- f.b
    v = op(y, f.f, y, γ)
    y .+= f.b
    return op === prox! ? y : ProximalBase._value_at(v, y)
  end
  step = ProximalBase._shrunk!(y, f, x, γ)
  v = op(y, f.f, y, step)
  op === prox! ? y : ProximalBase._value_at(v + ProximalBase._ridge(f, y), y)
end

@testset "Stored wrapper buffers" begin
  for T in (Float32, Float64), symmetric in (false, true)
    template = stored(T, symmetric)
    c = fill(T(0.125), size(template))
    for op in (prox!, value_and_prox!), f in wrappers(MoreauEnvelope(NormL1(), 1.0), c), aliased in (false, true)
      y = deepcopy(template)
      x = aliased ? y : deepcopy(template)
      before, input = state(y), state(x)
      error = failure(op, y, f, x, -2.0)
      reference = failure(op, similar(y), f.f, similar(y), -2.0)
      @test error isa ArgumentError
      @test typeof(error) === typeof(reference)
      @test sprint(showerror, error) == sprint(showerror, reference)
      @test isequal(state(y), before)
      @test isequal(state(x), input)
    end
    for op in (prox!, value_and_prox!), f in wrappers(NormL1(0.5), c), aliased in (false, true), step in (f isa Regularize ? (T(0.75),) : (T(0.75), fill(T(0.75), size(template))))
      y, expected = deepcopy(template), deepcopy(template)
      x = aliased ? y : deepcopy(template)
      refx = aliased ? expected : deepcopy(template)
      value = original!(op, expected, f, refx, step)
      result = op(y, f, x, step)
      @test isequal(state(y), state(expected))
      @test op === prox! ? result === y : isequal(result, value)
      @test allocs(op, y, f, x, step) == 0
    end
  end
end

@testset "Atom buffers on inner failure" begin
  for T in (Float32, Float64), op in (prox!, value_and_prox!), aliased in (false, true)
    for f in wrappers(MoreauEnvelope(NormL1(), 1.0), fill(0.125, 6))
      y = AtomIterate(T, 6, 3)
      y[6] = 3
      activate!(y, 1)
      x = aliased ? y : AtomIterate(ones(T, 6), 3)
      before, input = state(y), state(x)
      error = failure(op, y, f, x, -2.0)
      reference = failure(op, similar(y), f.f, similar(y), -2.0)
      @test error isa ArgumentError
      @test typeof(error) === typeof(reference)
      @test sprint(showerror, error) == sprint(showerror, reference)
      @test isequal(state(y), before)
      @test isequal(state(x), input)
    end
  end
end

@testset "Atom wrapper cache" begin
  for op in (prox!, value_and_prox!), make in (Tilt, Translate, (f, c) -> Regularize(f, 0.5, c))
    f = make(NormL1(0.25), fill(0.125, 6))
    for T in (Float32, Float64, Float32)
      y = AtomIterate(T, 6, 3)
      x = AtomIterate(T[1, 2, 0, 0, 3, 4], 3)
      for k in (3, 1, 2)
        activate!(y, k)
      end
      for aliased in (false, true)
        arg = aliased ? y : x
        op(y, f, arg, T(0.75))
        @test allocs(op, y, f, arg, T(0.75)) == 0
        fill!(y, 0)
        activate!(y, 2) # An active zero group must survive the copy into scratch.
        b = ProximalBase._output_buffer!(f isa Translate ? f.proxscratch : f.scratch, y)
        @test b.groups === y.groups
        @test isequal(state(b), state(y))
      end
    end
  end
end

@testset "Atom layout cache" begin
  y = AtomIterate(ones(6), 3)
  f = GroupNormL2(0.25, y.groups)
  other = GroupNormL2(0.25, y.groups)
  for op in (prox!, value_and_prox!), h in wrappers(f, fill(0.125, 6))
    op(y, h, y, 0.75)
    scratch = h isa Translate ? h.proxscratch : h.scratch
    b = scratch.buf
    ProximalBase._samelayout(f, b)
    ProximalBase._samelayout(other, y)
    @test ProximalBase._output_buffer!(scratch, y) === b
    @test b.layout_cache === f.groups
    @test y.layout_cache === other.groups
    @test allocs(op, y, h, y, 0.75) == 0
    # Equal layouts owned by different iterates must replace the cached buffer.
    z = AtomIterate(ones(6), 3)
    @test ProximalBase._output_buffer!(scratch, z) !== b
    @test scratch.buf.groups === z.groups
  end
end

@testset "Dense allocations and unsupported matrix outputs" begin
  for op in (prox!, value_and_prox!), f in wrappers(NormL1(0.25), fill(0.125, 3, 3))
    x = fill(2.0, 3, 3)
    @test allocs(op, similar(x), f, x, 0.75) == 0
    for wrap in (Symmetric, Hermitian), aliased in (false, true)
      y = wrap(copy(x))
      before = copy(parent(y))
      @test_throws ArgumentError op(y, f, aliased ? y : x, 0.75)
      @test parent(y) == before
    end
  end
end

end
