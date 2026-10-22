module TestUtils

using Test
using LinearAlgebra
using ProximalBase

export check_prox, check_gradient, check_traits, allocs, prox_objective, brute_force_prox

"""
    allocs(f, args...)

The bytes allocated by `f(args...)`, measured inside a function after a warm-up
call, as solvers call it.
"""
# one method per number of arguments, typed so that they specialize: Julia 1.10
# does not specialize a method that only passes its varargs on, and the dynamic
# dispatch would show as allocations
allocs(f::F, a::A) where {F, A} = (f(a); @allocated f(a))
allocs(f::F, a::A, b::B) where {F, A, B} = (f(a, b); @allocated f(a, b))
allocs(f::F, a::A, b::B, c::C) where {F, A, B, C} = (f(a, b, c); @allocated f(a, b, c))
allocs(f::F, a::A, b::B, c::C, d::D) where {F, A, B, C, D} = (f(a, b, c, d); @allocated f(a, b, c, d))
allocs(f::F, a::A, b::B, c::C, d::D, e::E) where {F, A, B, C, D, E} = (f(a, b, c, d, e); @allocated f(a, b, c, d, e))

prox_objective(f, z, x, γ) = f(z) + norm_diff(z, x)^2 / (2γ)

# a dense copy of an iterate, and a perturbation of it of the same kind
_dense(x::AbstractArray) = Array(x)

"""
    check_prox(f, x, γ; rtol, nonconvex_local = true)

Checks of the proximal operator of `f` at `x` with step `γ`: `prox`, `prox!`
and `value_and_prox!` agree, `prox!` may overwrite its input, the value is that
at the result, and the result satisfies the optimality conditions: for a convex
`f`, the subgradient inequality `f(z) ≥ f(p) + ⟨(x - p)/γ, z - p⟩` at points `z`
near `p` and at proximal points of other inputs; for any `f`, no such point has
a smaller proximal objective. With `symmetric = true` the points are symmetric
matrices, for the functions of symmetric matrices.
"""
function check_prox(f, x::AbstractArray, γ::Real; rtol = 1e-8, atol = 1e-10, local_points = true, symmetric = false)
  sym(w) = symmetric ? (w .+ w') ./ 2 : w
  p = prox(f, x, γ)
  @test size(p) == size(x)
  p2 = similar(p)
  @test prox!(p2, f, x, γ) === p2
  @test _dense(p2) ≈ _dense(p) rtol = rtol atol = atol
  p3 = similar(p)
  v = value_and_prox!(p3, f, x, γ)
  @test _dense(p3) ≈ _dense(p) rtol = rtol atol = atol
  fp = f(p)
  @test isfinite(fp)
  @test v ≈ fp rtol = sqrt(rtol) atol = sqrt(atol)
  z = copy(x)
  prox!(z, f, z, γ)
  @test _dense(z) ≈ _dense(p) rtol = rtol atol = atol

  # optimality
  xd, pd = _dense(x), _dense(p)
  obj = prox_objective(f, p, x, γ)
  candidates = Any[]
  if local_points
    for δ in (1e-6, 1e-3, 1e-1)
      push!(candidates, pd .+ δ .* sym(randn(size(pd)...)))
    end
  end
  for _ in 1:3
    push!(candidates, _dense(prox(f, xd .+ sym(randn(size(xd)...)), γ)))
  end
  slack = 1e-7 * max(1, abs(obj))
  for w in candidates
    @test prox_objective(f, w, xd, γ) >= obj - slack
    if isconvex(f)
      fw = f(w)
      isfinite(fw) && @test fw >= fp + dot(xd .- pd, w .- pd) / γ - slack
    end
  end
  p
end

"""
    check_gradient(f, x; h, rtol)

Checks of the gradient of `f` at `x` against central finite differences along
random directions, and of `gradient!`, `gradient`, `value_and_gradient!` and
`value_and_gradient` against each other.
"""
function check_gradient(f, x::AbstractArray; h = 1e-6, rtol = 1e-5, directions = 3)
  g = gradient(f, x)
  g2 = similar(g)
  @test gradient!(g2, f, x) === g2
  @test g2 ≈ g
  g3 = similar(g)
  v = value_and_gradient!(g3, f, x)
  @test g3 ≈ g
  @test v ≈ f(x)
  v4, g4 = value_and_gradient(f, x)
  @test v4 ≈ v && g4 ≈ g
  xd = Array(x)
  for _ in 1:directions
    d = randn(size(xd)...)
    fd = (f(xd .+ h .* d) - f(xd .- h .* d)) / (2h)
    @test fd ≈ dot(g, d) rtol = rtol atol = 1e-6 * max(1, abs(fd))
  end
  g
end

"""
    check_traits(f; smooth, proximable, separable, convex)
"""
function check_traits(f; smooth = false, proximable = false, separable = false, convex = false)
  @test issmooth(f) == smooth
  @test isproximable(f) == proximable
  @test isseparable(f) == separable
  @test isconvex(f) == convex
  @test issmooth(typeof(f)) == smooth
end

"""
    brute_force_prox(fᵢ, v, γ; lo, hi, n)

The minimizer over a fine grid of `fᵢ(t) + (t - v)²/(2γ)` for a scalar function
`fᵢ`, refined around the best grid point: a reference for separable proximal
operators, including nonconvex ones.
"""
function brute_force_prox(fi, v::Real, γ::Real; lo = v - 5abs(v) - 5, hi = v + 5abs(v) + 5, n = 200_001)
  h(t) = fi(t) + (t - v)^2 / (2γ)
  ts = range(lo, hi; length = n)
  vals = h.(ts)
  i = argmin(vals)
  best, hb = ts[i], vals[i]
  for c in (0.0, v)                # the kinks and the identity
    hc = h(c)
    hc < hb && ((best, hb) = (c, hc))
  end
  # golden-section refinement in the neighboring cells
  a, b = ts[max(i - 1, 1)], ts[min(i + 1, n)]
  φ = (sqrt(5) - 1) / 2
  for _ in 1:100
    c, d = b - φ * (b - a), a + φ * (b - a)
    h(c) < h(d) ? (b = d) : (a = c)
  end
  t = (a + b) / 2
  h(t) < hb ? (t, h(t)) : (best, hb)
end

end
