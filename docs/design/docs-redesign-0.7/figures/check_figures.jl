# Check the closed forms plotted by generate_figures.py against ProximalBase.
# Run from the repository root: julia --project=docs docs/design/docs-redesign-0.7/figures/check_figures.jl
using ProximalBase, LinearAlgebra

λ, a_mcp, a_scad, γ = 1.0, 3.0, 3.7, 1.0
grid = collect(range(-5, 5; length=2001))

l1(t) = λ * abs(t)
mcp(t) = (m = abs(t); m <= a_mcp * λ ? λ * m - m^2 / (2a_mcp) : a_mcp * λ^2 / 2)
scad(t) = (m = abs(t); m <= λ ? λ * m : m <= a_scad * λ ? λ * m - (m - λ)^2 / (2(a_scad - 1)) : (a_scad + 1) * λ^2 / 2)
soft(t) = sign(t) * max(abs(t) - γ * λ, 0)
hard(t) = t^2 > 2γ * λ ? t : 0.0
firm(t) = (m = abs(t); m <= γ * λ ? 0.0 : m <= a_mcp * λ ? sign(t) * (m - γ * λ) / (1 - γ / a_mcp) : t)

for t in grid
    @assert NormL1(λ)([t]) ≈ l1(t)
    @assert MCP(λ, a_mcp)([t]) ≈ mcp(t) atol = 1e-14
    @assert SCAD(λ, a_scad)([t]) ≈ scad(t) atol = 1e-14
    @assert prox(NormL1(λ), [t], γ)[1] ≈ soft(t) atol = 1e-14
    @assert prox(NormL0(λ), [t], γ)[1] == hard(t)
    @assert prox(MCP(λ, a_mcp), [t], γ)[1] ≈ firm(t) atol = 1e-13
end

for u in ([2.0, 1.2], [-1.6, 1.8], [-2.2, -0.9], [0.9, -2.0], [0.45, 0.5])
    expected = max(1 - γ * λ / norm(u), 0) .* u
    @assert prox(GroupNormL2(λ, [1:2], [1.0]), u, γ) ≈ expected
end

u = [1.5, 1.0]
@assert prox(IndBallL1(1.0), u, γ) ≈ [0.75, 0.25]
@assert prox(IndBallL2(1.0), u, γ) ≈ u ./ norm(u)
@assert prox(IndBox(-1.0, 1.0), u, γ) == [1.0, 1.0]
println("All figure closed forms agree with ProximalBase.")
