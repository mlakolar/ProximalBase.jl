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

end
