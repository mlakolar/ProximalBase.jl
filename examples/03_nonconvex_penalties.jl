# Nonconvex penalties: MCP, SCAD and ℓ0 against the lasso.
#
# The proximal operator of a separable penalty is a thresholding function of one
# variable, which `prox_entry` evaluates: soft thresholding for the ℓ1 norm,
# firm thresholding for MCP, SCAD's three pieces, and hard thresholding for ℓ0.
# MCP and SCAD leave large values untouched, so in regression they do not shrink
# large coefficients the way the lasso does. Their proximal operators are exact
# also in the nonconvex range, so coordinate descent applies to them unchanged.

using ProximalBase, LinearAlgebra, Random

# --- the thresholding functions, at γ = 1 -----------------------------------
λ = 1.0
penalties = ("ℓ1" => NormL1(λ), "MCP" => MCP(λ, 3.0), "SCAD" => SCAD(λ, 3.7), "ℓ0" => NormL0(λ^2 / 2))
println(rpad("v", 6), join((rpad(name, 8) for (name, _) in penalties)))
for v in 0.0:0.5:5.0
    println(rpad(v, 6), join((rpad(round(prox_entry(pen, v, 1, 1.0), digits = 3), 8) for (_, pen) in penalties)))
end

@assert prox_entry(NormL1(λ), 2.5, 1, 1.0) == 1.5                 # soft: shrinks by λ
@assert prox_entry(MCP(λ, 3.0), 0.8, 1, 1.0) == 0                  # MCP: zero below λ,
@assert prox_entry(MCP(λ, 3.0), 2.0, 1, 1.0) ≈ 1.5                 # firm between λ and aλ,
@assert prox_entry(MCP(λ, 3.0), 5.0, 1, 1.0) == 5.0                # and untouched beyond aλ
@assert prox_entry(SCAD(λ, 3.7), 1.5, 1, 1.0) ≈ 0.5                # SCAD: soft up to 2λ,
@assert prox_entry(SCAD(λ, 3.7), 5.0, 1, 1.0) == 5.0               # untouched beyond aλ
@assert prox_entry(NormL0(λ^2 / 2), 0.9, 1, 1.0) == 0 && prox_entry(NormL0(λ^2 / 2), 1.2, 1, 1.0) == 1.2

# --- regression by coordinate descent ---------------------------------------
#     minimize  ‖y - Xβ‖² / (2n) + Σⱼ pen(βⱼ)
# With columns scaled to ‖Xⱼ‖² = n, the coordinate problem has curvature 1, and the
# step is βⱼ ← prox_entry(pen, βⱼ + Xⱼ'r / n, j, 1), r = y - Xβ. For MCP with a > 1
# and SCAD with a > 2 that coordinate problem is convex, so each step is exact.
rng = Xoshiro(3)
n, p = 200, 100
X = randn(rng, n, p)
X .*= sqrt(n) ./ sqrt.(sum(abs2, X, dims = 1))           # ‖Xⱼ‖² = n
β_true = [3.0; -2.5; 2.0; zeros(p - 3)]
y = X * β_true + randn(rng, n)

function coordinate_descent(X, y, pen; tol = 1e-10, maxsweeps = 10_000)
    n, p = size(X)
    β, r = zeros(p), copy(y)
    for _ in 1:maxsweeps
        δ = 0.0
        for j in 1:p
            old = β[j]
            new = prox_entry(pen, old + dot(view(X, :, j), r) / n, j, 1.0)
            if new != old
                β[j] = new
                axpy!(old - new, view(X, :, j), r)
                δ = max(δ, abs(new - old))
            end
        end
        δ < tol && return β
    end
    error("no convergence")
end

λ = 0.25
estimates = ("lasso" => NormL1(λ), "MCP" => MCP(λ, 3.0), "SCAD" => SCAD(λ, 3.7), "ℓ0" => NormL0(λ^2 / 2))
errors = Dict{String, Float64}()
for (name, pen) in estimates
    β = coordinate_descent(X, y, pen)
    errors[name] = norm(β[1:3] - β_true[1:3])
    println(rpad(name, 6), "β[1:3] = ", round.(β[1:3], digits = 3), ", nonzeros ", count(!iszero, β),
            ", error on the support ", round(errors[name], digits = 3))
    @assert all(!iszero, β[1:3])                         # every method finds the signals
    name == "lasso" || @assert count(!iszero, β) == 3     # the nonconvex ones exactly
end
@assert errors["MCP"] < errors["lasso"] / 2 && errors["SCAD"] < errors["lasso"] / 2
