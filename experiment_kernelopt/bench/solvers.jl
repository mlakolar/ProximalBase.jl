# Level 3: fixed-iteration solver loops written against ProximalBase's public interface, one implementation per
# source tree. They are isolated experiments, not ProximalSolvers: the same loop runs every tree's oracles, so a
# ratio is the effect of the tree's kernels on a whole iteration (gradient, proximal step, value).
using SparseArrays

# proximal gradient with a fixed step: value_and_gradient!, a step, value_and_prox!
function pgd!(vg!::VG, vp!::VP, x, z, g, f, pen, γ, iters) where {VG, VP}
  fill!(x, 0)
  v = zero(eltype(x))
  for _ in 1:iters
    v = vg!(g, f, x)
    @inbounds @simd for i in eachindex(z, x, g)
      z[i] = x[i] - γ * g[i]
    end
    v += vp!(x, pen, z, γ)
  end
  v
end
# accelerated proximal gradient (FISTA momentum), fixed step
function apg!(vg!::VG, vp!::VP, x, xold, y, z, g, f, pen, γ, iters) where {VG, VP}
  fill!(x, 0); fill!(y, 0); fill!(xold, 0)
  T = eltype(x)
  t = one(T); v = zero(T)
  for _ in 1:iters
    v = vg!(g, f, y)
    @inbounds @simd for i in eachindex(z, y, g)
      z[i] = y[i] - γ * g[i]
    end
    copyto!(xold, x)
    v = vp!(x, pen, z, γ)
    tn = (one(T) + sqrt(one(T) + 4 * t * t)) / 2
    β = (t - one(T)) / tn
    @inbounds @simd for i in eachindex(y, x, xold)
      y[i] = x[i] + β * (x[i] - xold[i])
    end
    t = tn
  end
  v
end
# proximal gradient in a diagonal metric: a step per entry
function pgd_steps!(g!::G, p!::P, x, z, g, f, pen, γ::AbstractVector, iters) where {G, P}
  fill!(x, 0)
  for _ in 1:iters
    g!(g, f, x)
    @inbounds @simd for i in eachindex(z, x, g)
      z[i] = x[i] - γ[i] * g[i]
    end
    p!(x, pen, z, γ)
  end
  x
end
# cyclic coordinate descent for the lasso with the residual r = Y - Xβ kept, the scalar prox_block per coordinate
function cd!(prox_block::PBk, x::Vector{T}, r::Vector{T}, X::Matrix{T}, Y::Vector{T}, pen, L::Vector{T}, sweeps) where {PBk, T}
  fill!(x, 0); copyto!(r, Y)
  n, p = size(X)
  @inbounds for _ in 1:sweeps, j in 1:p
    s = zero(T)
    @simd for i in 1:n
      s += X[i, j] * r[i]
    end
    xj = x[j]
    xn = convert(T, prox_block(pen, xj + s / L[j], j, inv(L[j])))
    d = xn - xj
    if d != 0
      @simd for i in 1:n
        r[i] -= d * X[i, j]
      end
      x[j] = xn
    end
  end
  x
end
# block coordinate descent for the group lasso: the compact prox_block! per group, step 1/L_k
function bcd!(prox_block!::PBk, x::Vector{T}, r::Vector{T}, X::Matrix{T}, Y::Vector{T}, pen, groups, L::Vector{T}, us, outs, sweeps) where {PBk, T}
  fill!(x, 0); copyto!(r, Y)
  n = size(X, 1)
  @inbounds for _ in 1:sweeps, k in eachindex(groups)
    G = groups[k]; u = us[k]; out = outs[k]
    for (a, j) in enumerate(G)
      s = zero(T)
      @simd for i in 1:n
        s += X[i, j] * r[i]
      end
      u[a] = x[j] + s / L[k]
    end
    prox_block!(out, pen, u, k, inv(L[k]))
    for (a, j) in enumerate(G)
      d = out[a] - x[j]
      if d != 0
        @simd for i in 1:n
          r[i] -= d * X[i, j]
        end
        x[j] = out[a]
      end
    end
  end
  x
end

function cases_solver(pbs, ::Type{T}) where {T}
  cases = Case[]
  r = Ref(zero(T))
  tol = T === Float32 ? 1f-3 : 1e-9
  dense = ((200, 200, 40), (2000, 500, 20), (500, 2000, 20), (100, 10000, 10))
  for (n, p, iters) in dense
    rng = MersenneTwister(4000 + n + p)
    X = randn(rng, T, n, p) ./ T(sqrt(n)); β0 = zeros(T, p); β0[1:max(5, p ÷ 50)] .= randn(rng, T, max(5, p ÷ 50))
    Y = X * β0 .+ T(0.1) .* randn(rng, T, n)
    L = T(opnorm(Float64.(X))^2 * 1.01); γ = inv(L)
    λ = T(0.1) * maximum(abs, X' * Y)
    x, xold, y, z, g = (zeros(T, p) for _ in 1:5)
    w = T(0.5) .+ rand(rng, T, p)
    γv = fill(γ, p) .* (T(0.5) .+ T(0.5) .* rand(rng, T, p))
    c = Case("solver/pgd_lasso/$T/$(n)x$p", "solver", REFNAME; rtol = tol)
    for (name, pb) in pbs
      add!(c, name, 3, thunkref(r, pgd!, pb.value_and_gradient!, pb.value_and_prox!, x, z, g, pb.LeastSquaresLoss(Y, X), pb.NormL1(λ), γ, iters), outboth(r, x))
    end
    push!(cases, c)
    c = Case("solver/apg_lasso/$T/$(n)x$p", "solver", REFNAME; rtol = tol)
    for (name, pb) in pbs
      add!(c, name, 3, thunkref(r, apg!, pb.value_and_gradient!, pb.value_and_prox!, x, xold, y, z, g, pb.LeastSquaresLoss(Y, X), pb.NormL1(λ), γ, iters), outboth(r, x))
    end
    push!(cases, c)
    c = Case("solver/apg_wlasso/$T/$(n)x$p", "solver", REFNAME; rtol = tol)
    for (name, pb) in pbs
      add!(c, name, 3, thunkref(r, apg!, pb.value_and_gradient!, pb.value_and_prox!, x, xold, y, z, g, pb.LeastSquaresLoss(Y, X), pb.NormL1(λ, w), γ, iters), outboth(r, x))
    end
    push!(cases, c)
    c = Case("solver/pgd_lasso_steps/$T/$(n)x$p", "solver", REFNAME; rtol = tol)
    for (name, pb) in pbs
      add!(c, name, 3, thunk(pgd_steps!, pb.gradient!, pb.prox!, x, z, g, pb.LeastSquaresLoss(Y, X), pb.NormL1(λ, w), γv, iters), outarr(x))
    end
    push!(cases, c)
    for m in (5, 20)
      p % m == 0 || continue
      groups = [(k-1)*m+1:k*m for k in 1:p÷m]
      λg = T(0.1) * maximum(norm(X[:, G]' * Y) for G in groups) / T(sqrt(m))
      c = Case("solver/apg_group$m/$T/$(n)x$p", "solver", REFNAME; rtol = tol)
      for (name, pb) in pbs
        add!(c, name, 3, thunkref(r, apg!, pb.value_and_gradient!, pb.value_and_prox!, x, xold, y, z, g, pb.LeastSquaresLoss(Y, X), pb.GroupNormL2(λg, groups), γ, iters), outboth(r, x))
      end
      push!(cases, c)
      if n * p <= 1_000_000
        Lk = T[T(opnorm(Float64.(X[:, G]))^2 * 1.01) for G in groups]
        us = [zeros(T, m) for _ in groups]; outs = [zeros(T, m) for _ in groups]
        rr = zeros(T, n)
        c = Case("solver/bcd_group$m/$T/$(n)x$p", "solver", REFNAME; rtol = tol)
        for (name, pb) in pbs
          add!(c, name, 3, thunk(bcd!, pb.prox_block!, x, rr, X, Y, pb.GroupNormL2(λg, groups), groups, Lk, us, outs, 3), outarr(x))
        end
        push!(cases, c)
      end
    end
    if n * p <= 1_000_000
      Lj = T[sum(abs2, view(X, :, j)) for j in 1:p]
      rr = zeros(T, n)
      c = Case("solver/cd_lasso/$T/$(n)x$p", "solver", REFNAME; rtol = tol)
      for (name, pb) in pbs
        add!(c, name, 3, thunk(cd!, pb.prox_block, x, rr, X, Y, pb.NormL1(λ), Lj, 5), outarr(x))
      end
      push!(cases, c)
    end
  end
  # a sparse design: the gradient costs a few operations per coordinate, so the proximal step is a large share
  for (n, p, k, iters) in ((20000, 20000, 5, 20), (2000, 100000, 3, 20))
    rng = MersenneTwister(4100 + p)
    X = sprandn(rng, T, n, p, k / n)
    β0 = zeros(T, p); β0[1:p÷100] .= randn(rng, T, p ÷ 100)
    Y = X * β0 .+ T(0.1) .* randn(rng, T, n)
    L = T(opnorm(Float64.(X), 1) * opnorm(Float64.(X), Inf)); γ = inv(L)
    λ = T(0.1) * maximum(abs, X' * Y)
    x, xold, y, z, g = (zeros(T, p) for _ in 1:5)
    for (nm, m) in (("lasso", 0), ("group5", 5))
      c = Case("solver/apg_$(nm)_sparseX/$T/$(n)x$p", "solver", REFNAME; rtol = tol)
      for (name, pb) in pbs
        pen = m == 0 ? pb.NormL1(λ) : pb.GroupNormL2(λ / T(sqrt(m)), [(q-1)*m+1:q*m for q in 1:p÷m])
        add!(c, name, 3, thunkref(r, apg!, pb.value_and_gradient!, pb.value_and_prox!, x, xold, y, z, g, pb.LeastSquaresLoss(Y, X), pen, γ, iters), outboth(r, x))
      end
      push!(cases, c)
    end
  end
  cases
end
