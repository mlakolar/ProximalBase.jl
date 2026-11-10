module TestUtils

# The checks of the test suite are those of ProximalBase.Testing, which the
# package extension on Test defines; these names keep the test files short.

using Test
using ProximalBase

export check_prox, check_gradient, check_traits, allocs, prox_objective, brute_force_prox

const PT = ProximalBase.Testing

check_prox(f, x, γ; kwargs...) = PT.test_prox(f, x, γ; kwargs...)
check_gradient(f, x; kwargs...) = PT.test_gradient(f, x; kwargs...)
check_traits(f; kwargs...) = PT.test_traits(f; kwargs...)
const allocs = PT.allocations
const brute_force_prox = PT.brute_force_prox

prox_objective(f, z, x, γ) = f(z) + norm_diff(z, x)^2 / (2γ)

# Solver-style integration helper for tests: math first, public storage writes second.
# Kept outside ProximalBase so the mathematical API never owns activation policy.
function take_block_step!(x, f, k, γ::Real)
  u = x isa AtomIterate ? collect(atom(x, k)) :
      x isa SymmetricSparseIterate ? coordinates(x)[k] : x[k]
  take_block_step!(x, f, k, u, γ)
end
function take_block_step!(x::Union{SparseIterate, SymmetricSparseIterate}, f, k, u::Number, γ::Real)
  isseparable(f) || throw(ArgumentError("a scalar step needs a separable penalty"))
  ProximalBase._checkparams(f, x)
  cx = x isa SymmetricSparseIterate ? coordinates(x) : x
  checkbounds(cx, k)
  ProximalBase._checkreal(f, u)
  i = k
  if x isa SymmetricSparseIterate
    r, c = tril_cartesian(size(x, 1), k)
    i = (c - 1) * size(x, 1) + r
    ProximalBase._check_entry_arrays(f, x, ProximalBase._CoordinatePair(Int(k)))
  end
  v = prox_block(f, convert(eltype(x), u), i, γ)
  cx[k] = v
  v
end
_test_group_penalty(f::Union{Scaled, Regularize}) = _test_group_penalty(f.f)
_test_group_penalty(f) = f
function take_block_step!(x::AtomIterate, f, k, u::AbstractVector, γ::Real)
  ProximalBase._checkparams(f, x)
  if !isseparable(f)
    penalty = _test_group_penalty(f)
    penalty isa ProximalBase.AbstractGroupPenalty || throw(ArgumentError("a group step needs a separable or grouped penalty"))
    ProximalBase.samegroups(x, penalty) || throw(DimensionMismatch("f and x must have the same groups."))
  end
  checkbounds(Base.OneTo(ncoordinates(x)), k)
  ProximalBase._checkreal(f, u)
  G = atomindices(x, k)
  length(G) == length(u) || throw(DimensionMismatch("trial block length"))
  buf = eltype(x).(u)
  if isseparable(f)
    for (j, i) in enumerate(G)
      buf[j] = prox_block(f, buf[j], i, γ)
    end
  else
    prox_block!(buf, f, buf, k, γ)
  end
  copyto!(atom(x, k), buf)
  atom(x, k)
end

end
