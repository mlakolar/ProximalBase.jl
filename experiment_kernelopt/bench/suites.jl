# The cases of the study. Every function takes the implementations as named tuples of functions (so that
# the timed closures dispatch statically): `pbs` is a vector of (name, api) for the ProximalBase trees, `K`
# and `KC` the standalone kernels and their independently compiled copy, `PO` ProximalOperators.jl.
using Random, LinearAlgebra
using .Harness: Case, add!

thunk(f::F, args...) where {F} = () -> (f(args...); nothing)
thunkref(r::Base.RefValue, f::F, args...) where {F} = () -> (r[] = f(args...); nothing)
# r[] = a * f(args...)
thunkscaled(r::Base.RefValue, a, f::F, args...) where {F} = () -> (r[] = a * f(args...); nothing)
outarr(y) = () -> y
outref(r) = () -> r[]
outboth(r, y) = () -> (r[], y)

const ST_VARIANTS = (:st_maxmin, :st_selmaxmin, :st_clamp, :st_clampsel, :st_clampz, :st_copysign, :st_copysignsel, :st_ifelse, :st_po)

big_shrink(v, c) = sign(big(v)) * max(abs(big(v)) - c, 0)

####################################
#  A. NormL1
####################################

function l1_data(::Type{T}, n, pattern) where {T}
  rng = MersenneTwister(1000 + n + (pattern === :zeroheavy))
  x = randn(rng, T, n)
  pattern === :zeroheavy && (x[rand(rng, n) .< 0.9] .= 0)
  (; x, y = similar(x), w = T(0.5) .+ rand(rng, T, n), γv = T(0.5) .+ rand(rng, T, n), λ = T(0.6), γ = T(0.8))
end

function cases_l1(pbs, K, KC, PO, ::Type{T}; sizes = (16, 128, 1024, 16384, 1048576)) where {T}
  cases = Case[]
  tol = 16 * eps(T)
  for n in sizes, pattern in (n == 1024 ? (:normal, :zeroheavy) : (:normal,))
    d = l1_data(T, n, pattern)
    x, y, w, γv, λ, γ = d.x, d.y, d.w, d.γv, d.λ, d.γ
    id(op) = "l1/$op/$T/n$n/$pattern"
    c0 = γ * λ
    r = Ref(zero(T))
    bigref = n <= 16384
    # one BigFloat pass for every expected output of this input
    cb = big(γ) * big(λ)

    # --- prox, scalar step, no weights
    c = Case(id("prox"), "l1", REFNAME; rtol = tol, expected = bigref ? T.(big_shrink.(x, cb)) : nothing)
    for (name, pb) in pbs
      add!(c, name, 2, thunk(pb.prox!, y, pb.NormL1(λ), x, γ), outarr(y))
    end
    for st in ST_VARIANTS
      add!(c, "k_$st", 1, thunk(K.l1prox!, getfield(K, st), y, x, c0), outarr(y))
    end
    add!(c, "kC_st_maxmin", 1, thunk(KC.l1prox!, KC.st_maxmin, y, x, c0), outarr(y))
    add!(c, "k_plain_st_maxmin", 1, thunk(K.l1prox_plain!, K.st_maxmin, y, x, c0), outarr(y))
    add!(c, "k_plain_st_clamp", 1, thunk(K.l1prox_plain!, K.st_clamp, y, x, c0), outarr(y))
    add!(c, "po", 2, thunk(PO.prox!, y, PO.NormL1(λ), x, γ), outarr(y); note = "also returns the value")
    push!(cases, c)

    # --- prox, scalar step, weights
    c = Case(id("prox_w"), "l1", REFNAME; rtol = tol, expected = bigref ? T.(big_shrink.(x, cb .* big.(w))) : nothing)
    for (name, pb) in pbs
      add!(c, name, 2, thunk(pb.prox!, y, pb.NormL1(λ, w), x, γ), outarr(y))
    end
    for st in (:st_maxmin, :st_selmaxmin, :st_clamp, :st_clampsel, :st_clampz, :st_ifelse)
      add!(c, "k_$st", 1, thunk(K.l1prox_w!, getfield(K, st), y, x, c0, w), outarr(y))
    end
    add!(c, "po", 2, thunk(PO.prox!, y, PO.NormL1(λ .* w), x, γ), outarr(y); note = "also returns the value")
    push!(cases, c)

    # --- prox, a step per entry, without and with weights
    c = Case(id("prox_g"), "l1", REFNAME; rtol = tol, expected = bigref ? T.(big_shrink.(x, big.(γv) .* big(λ))) : nothing)
    for (name, pb) in pbs
      add!(c, name, 2, thunk(pb.prox!, y, pb.NormL1(λ), x, γv), outarr(y))
    end
    for st in (:st_maxmin, :st_selmaxmin, :st_clamp, :st_clampz)
      add!(c, "k_$st", 1, thunk(K.l1prox_g!, getfield(K, st), y, x, γv, λ), outarr(y))
    end
    push!(cases, c)
    c = Case(id("prox_gw"), "l1", REFNAME; rtol = tol, expected = bigref ? T.(big_shrink.(x, big.(γv) .* big(λ) .* big.(w))) : nothing)
    for (name, pb) in pbs
      add!(c, name, 2, thunk(pb.prox!, y, pb.NormL1(λ, w), x, γv), outarr(y))
    end
    for st in (:st_maxmin, :st_selmaxmin, :st_clamp, :st_clampz)
      add!(c, "k_$st", 1, thunk(K.l1prox_gw!, getfield(K, st), y, x, γv, λ, w), outarr(y))
    end
    push!(cases, c)

    # --- value, without and with weights
    vtol = 4 * n * eps(T)
    c = Case(id("value"), "l1", REFNAME; rtol = vtol, expected = bigref ? T(big(λ) * sum(abs ∘ big, x)) : nothing)
    for (name, pb) in pbs
      add!(c, name, 2, thunkref(r, pb.NormL1(λ), x), outref(r))
    end
    add!(c, "k_simd", 1, thunkscaled(r, λ, K.sumabs_simd, x), outref(r))
    add!(c, "kC_simd", 1, thunkscaled(r, λ, KC.sumabs_simd, x), outref(r))
    add!(c, "k_base", 1, thunkscaled(r, λ, K.sumabs_base, x), outref(r))
    add!(c, "k_blas", 1, thunkscaled(r, λ, K.sumabs_blas, x), outref(r))
    add!(c, "po", 2, thunkref(r, PO.NormL1(λ), x), outref(r))
    push!(cases, c)
    c = Case(id("value_w"), "l1", REFNAME; rtol = vtol, expected = bigref ? T(big(λ) * sum(big.(w) .* abs.(big.(x)))) : nothing)
    for (name, pb) in pbs
      add!(c, name, 2, thunkref(r, pb.NormL1(λ, w), x), outref(r))
    end
    add!(c, "k_simd", 1, thunkref(r, K.wsumabs_simd, x, λ, w), outref(r))
    add!(c, "po", 2, thunkref(r, PO.NormL1(λ .* w), x), outref(r))
    push!(cases, c)

    # --- value and prox
    ex = bigref ? (yb = big_shrink.(x, cb); (T(big(λ) * sum(abs, yb)), T.(yb))) : nothing
    c = Case(id("valprox"), "l1", REFNAME; rtol = vtol, expected = ex)
    for (name, pb) in pbs
      add!(c, name, 2, thunkref(r, pb.value_and_prox!, y, pb.NormL1(λ), x, γ), outboth(r, y))
    end
    for st in (:st_maxmin, :st_selmaxmin, :st_clamp)
      add!(c, "k_fused_$st", 1, thunkref(r, K.l1valprox_fused!, getfield(K, st), y, x, c0, λ), outboth(r, y))
      add!(c, "k_two_$st", 1, thunkref(r, K.l1valprox_two!, getfield(K, st), y, x, c0, λ), outboth(r, y))
    end
    add!(c, "po", 2, thunkref(r, PO.prox!, y, PO.NormL1(λ), x, γ), outboth(r, y))
    push!(cases, c)
    ex = bigref ? (yb = big_shrink.(x, cb .* big.(w)); (T(big(λ) * sum(big.(w) .* abs.(yb))), T.(yb))) : nothing
    c = Case(id("valprox_w"), "l1", REFNAME; rtol = vtol, expected = ex)
    for (name, pb) in pbs
      add!(c, name, 2, thunkref(r, pb.value_and_prox!, y, pb.NormL1(λ, w), x, γ), outboth(r, y))
    end
    for st in (:st_maxmin, :st_selmaxmin, :st_clamp)
      add!(c, "k_fused_$st", 1, thunkref(r, K.l1valprox_w_fused!, getfield(K, st), y, x, γ, λ, w), outboth(r, y))
      add!(c, "k_two_$st", 1, thunkref(r, K.l1valprox_w_two!, getfield(K, st), y, x, γ, λ, w), outboth(r, y))
    end
    add!(c, "po", 2, thunkref(r, PO.prox!, y, PO.NormL1(λ .* w), x, γ), outboth(r, y))
    push!(cases, c)
  end
  cases
end

####################################
#  B. NormL2
####################################

big_norm(x) = sqrt(sum(abs2 ∘ big, x))
function big_l2prox(x, c)
  n = big_norm(x)
  n > c ? (1 - c / n) .* big.(x) : zero(big.(x))
end

function cases_l2(pbs, K, KC, PO, ::Type{T}; sizes = (2, 5, 10, 32, 128, 1024, 65536)) where {T}
  cases = Case[]
  for n in sizes, pattern in (:shrink, :tozero, :zero)
    rng = MersenneTwister(2000 + n)
    x = pattern === :zero ? zeros(T, n) : randn(rng, T, n)
    y = similar(x)
    γ = T(0.8)
    # the threshold γλ is a fraction of the norm (shrink) or above it (tozero)
    λ = pattern === :tozero ? T(2 * sqrt(n)) : T(0.3)
    c0 = γ * λ
    cb = big(γ) * big(λ)
    r = Ref(zero(T))
    id(op) = "l2/$op/$T/n$n/$pattern"
    tol = (n + 16) * eps(T)

    c = Case(id("prox"), "l2", REFNAME; rtol = tol, expected = T.(big_l2prox(x, cb)))
    for (name, pb) in pbs
      add!(c, name, 2, thunk(pb.prox!, y, pb.NormL2(λ), x, γ), outarr(y))
    end
    add!(c, "k_direct", 1, thunk(K.l2prox!, y, x, c0), outarr(y))
    add!(c, "kC_direct", 1, thunk(KC.l2prox!, y, x, c0), outarr(y))
    if pattern !== :zero                                   # 0/0 in the norm-then-broadcast forms
      add!(c, "k_norm", 1, thunk(K.l2prox_norm!, y, x, c0), outarr(y))
      add!(c, "k_nrm2", 1, thunk(K.l2prox_nrm2!, y, x, c0), outarr(y))
    end
    add!(c, "po", 2, thunk(PO.prox!, y, PO.NormL2(λ), x, γ), outarr(y); note = "also returns the value")
    push!(cases, c)

    yb = big_l2prox(x, cb)
    c = Case(id("valprox"), "l2", REFNAME; rtol = tol, expected = (T(big(λ) * big_norm(yb)), T.(yb)))
    for (name, pb) in pbs
      add!(c, name, 2, thunkref(r, pb.value_and_prox!, y, pb.NormL2(λ), x, γ), outboth(r, y))
    end
    add!(c, "k_direct", 1, thunkscaled(r, λ, K.l2prox!, y, x, c0), outboth(r, y))
    add!(c, "po", 2, thunkref(r, PO.prox!, y, PO.NormL2(λ), x, γ), outboth(r, y))
    push!(cases, c)

    pattern === :tozero && continue
    c = Case(id("value"), "l2", REFNAME; rtol = tol, expected = T(big(λ) * big_norm(x)))
    for (name, pb) in pbs
      add!(c, name, 2, thunkref(r, pb.NormL2(λ), x), outref(r))
    end
    for nm in (:norm_simd, :norm_plain, :norm_base, :norm_la, :norm_blas, :norm_dot, :norm_scaled, :norm_blue)
      add!(c, "k_$nm", 1, thunkscaled(r, λ, getfield(K, nm), x), outref(r))
    end
    add!(c, "kC_norm_simd", 1, thunkscaled(r, λ, KC.norm_simd, x), outref(r))
    add!(c, "po", 2, thunkref(r, PO.NormL2(λ), x), outref(r))
    push!(cases, c)
  end
  cases
end

####################################
#  B. GroupNormL2
####################################

function group_data(::Type{T}, m, layout, pattern; total = 4096) where {T}
  rng = MersenneTwister(900 + m + (layout === :perm) + 2 * (pattern === :sparse) + 4 * (pattern === :mixed))
  Kg = total ÷ m
  p = Kg * m
  order = layout === :perm ? randperm(rng, p) : collect(1:p)
  groups = layout === :perm ? [order[(k-1)*m+1:k*m] for k in 1:Kg] : [(k-1)*m+1:k*m for k in 1:Kg]
  x = randn(rng, T, p)
  for (k, G) in enumerate(groups)
    if pattern === :sparse && k % 4 != 1
      x[G] .= 0                                  # three groups in four are exactly zero
    elseif pattern !== :allshrink && k % 3 == 0
      x[G] .*= T(0.05)                           # below the threshold: set to zero by the prox
    end
  end
  (; groups, x, y = similar(x), λ = T(0.6), γ = T(0.8), w = fill(sqrt(T(m)), Kg), m, Kg, p)
end

function big_group_prox(x, groups, w, c0)
  yb = big.(x)
  for (k, G) in enumerate(groups)
    yb[G] .= big_l2prox(x[G], c0 * big(w[k]))
  end
  yb
end
big_group_value(yb, groups, w, λ) = big(λ) * sum(big(w[k]) * big_norm(yb[G]) for (k, G) in enumerate(groups))

function cases_group(pbs, K, KC, PO, ::Type{T}; sizes = (2, 5, 10, 32, 128, 1024), total = 4096) where {T}
  cases = Case[]
  for m in sizes, layout in (:range, :perm), pattern in (:allshrink, :mixed, :sparse)
    d = group_data(T, m, layout, pattern; total = total)
    groups, x, y, λ, γ, w = d.groups, d.x, d.y, d.λ, d.γ, d.w
    c0 = γ * λ
    cb = big(γ) * big(λ)
    r = Ref(zero(T))
    id(op) = "group/$op/$T/m$m/$layout/$pattern" * (total == 4096 ? "" : "/p$total")
    tol = (m + 16) * eps(T)
    yb = big_group_prox(x, groups, w, cb)
    X2, Y2 = reshape(x, m, d.Kg), reshape(y, m, d.Kg)       # the groups as columns (range layout)
    poλ = λ * w[1]

    c = Case(id("prox"), "group", REFNAME; rtol = tol, expected = T.(yb))
    for (name, pb) in pbs
      add!(c, name, 2, thunk(pb.prox!, y, pb.GroupNormL2(λ, groups, w), x, γ), outarr(y))
    end
    add!(c, "k_direct", 1, thunk(K.group_prox!, y, x, groups, w, c0), outarr(y))
    add!(c, "kC_direct", 1, thunk(KC.group_prox!, y, x, groups, w, c0), outarr(y))
    add!(c, "k_plain", 1, thunk(K.group_prox_plain!, y, x, groups, w, c0), outarr(y))
    add!(c, "k_early", 1, thunk(K.group_prox_early!, y, x, groups, w, c0), outarr(y))
    if layout === :range
      pattern === :sparse || add!(c, "k_cols_norm", 1, thunk(K.group_prox_cols_norm!, Y2, X2, w, c0), outarr(y))
      add!(c, "po", 2, thunk(PO.prox!, Y2, PO.NormL21(poλ, 1), X2, γ), outarr(y); note = "matrix columns; also returns the value")
    end
    push!(cases, c)

    vtol = (m + d.Kg + 16) * eps(T)
    c = Case(id("valprox"), "group", REFNAME; rtol = vtol, expected = (T(big_group_value(yb, groups, w, λ)), T.(yb)))
    for (name, pb) in pbs
      add!(c, name, 2, thunkref(r, pb.value_and_prox!, y, pb.GroupNormL2(λ, groups, w), x, γ), outboth(r, y))
    end
    add!(c, "k_direct", 1, thunkref(r, K.group_valprox!, y, x, groups, w, γ, λ), outboth(r, y))
    layout === :range && add!(c, "po", 2, thunkref(r, PO.prox!, Y2, PO.NormL21(poλ, 1), X2, γ), outboth(r, y))
    push!(cases, c)

    c = Case(id("value"), "group", REFNAME; rtol = vtol, expected = T(big_group_value(big.(x), groups, w, λ)))
    for (name, pb) in pbs
      add!(c, name, 2, thunkref(r, pb.GroupNormL2(λ, groups, w), x), outref(r))
    end
    add!(c, "k_direct", 1, thunkref(r, K.group_value, x, groups, w, λ), outref(r))
    layout === :range && add!(c, "po", 2, thunkref(r, PO.NormL21(poλ, 1), X2), outref(r))
    push!(cases, c)

    # one sweep of compact block proximal operators, as block coordinate descent calls them
    if layout === :range
      U = [x[G] for G in groups]
      outs = [similar(u) for u in U]
      c = Case(id("block_sweep"), "group", REFNAME; rtol = tol, expected = [T.(yb[G]) for G in groups])
      for (name, pb) in pbs
        f = pb.GroupNormL2(λ, groups, w)
        add!(c, name, 2, block_sweep_thunk(pb.prox_block!, outs, f, U, γ), () -> outs)
      end
      add!(c, "k_direct", 1, block_sweep_kernel_thunk(K.block_prox!, outs, U, w, c0), () -> outs)
      add!(c, "kC_direct", 1, block_sweep_kernel_thunk(KC.block_prox!, outs, U, w, c0), () -> outs)
      push!(cases, c)
    end
  end
  cases
end
function block_sweep_thunk(prox_block!::F, outs, f, U, γ) where {F}
  () -> begin
    for k in eachindex(U)
      @inbounds prox_block!(outs[k], f, U[k], k, γ)
    end
    nothing
  end
end
function block_sweep_kernel_thunk(block_prox!::F, outs, U, w, c0) where {F}
  () -> begin
    for k in eachindex(U)
      @inbounds block_prox!(outs[k], U[k], c0 * w[k])
    end
    nothing
  end
end
# relative to the largest entry of all blocks, as the array prox case is: a block shrunk almost to zero has the
# cancellation of 1 - c/‖u‖ in every implementation, which a per-block scale would report as a difference
Harness.reldiff(a::Vector{<:Vector}, b::Vector{<:Vector}) = Harness.reldiff(reduce(vcat, a), reduce(vcat, b))

####################################
#  D. ElasticNet (on trees that have it), and the coordinate loops of the maintainers' gate cells
####################################

# y[i] = prox_block(f, x[i], i, γ) for every entry, as a coordinate method calls the scalar operator
function block_loop_thunk(prox_block::F, y::Vector{T}, f, x::Vector{T}, γ) where {F, T}
  () -> begin
    @inbounds for i in eachindex(y, x)
      y[i] = prox_block(f, x[i], i, γ)
    end
    nothing
  end
end

function cases_enet(pbs, K, KC, PO, ::Type{T}; sizes = (16, 128, 1024, 16384, 1048576)) where {T}
  cases = Case[]
  tol = 32 * eps(T)
  for n in sizes
    d = l1_data(T, n, :normal)
    x, y, w, γv, γ = d.x, d.y, d.w, d.γv, d.γ
    λ1, λ2 = T(0.6), T(0.5)
    id(op) = "enet/$op/$T/n$n/normal"
    r = Ref(zero(T))
    bigref = n <= 16384
    c0 = γ * λ1; a = inv(one(T) + γ * λ2)
    bγ, b1, b2 = big(γ), big(λ1), big(λ2)
    has(pb) = pb.ElasticNet !== nothing

    c = Case(id("prox"), "enet", REFNAME; rtol = tol, expected = bigref ? T.(big_shrink.(x, bγ * b1) ./ (1 + bγ * b2)) : nothing)
    for (name, pb) in pbs
      has(pb) && add!(c, name, 2, thunk(pb.prox!, y, pb.ElasticNet(λ1, λ2), x, γ), outarr(y))
    end
    for st in (:st_maxmin, :st_clampz)
      add!(c, "k_$st", 1, thunk(K.enprox!, getfield(K, st), y, x, c0, a), outarr(y))
    end
    add!(c, "po", 2, thunk(PO.prox!, y, PO.ElasticNet(λ1, λ2), x, γ), outarr(y); note = "also returns the value")
    push!(cases, c)

    c = Case(id("prox_w"), "enet", REFNAME; rtol = tol, expected = bigref ? T.(big_shrink.(x, bγ * b1 .* big.(w)) ./ (1 + bγ * b2)) : nothing)
    for (name, pb) in pbs
      has(pb) && add!(c, name, 2, thunk(pb.prox!, y, pb.ElasticNet(λ1, λ2, w), x, γ), outarr(y))
    end
    for st in (:st_maxmin, :st_clampz)
      add!(c, "k_$st", 1, thunk(K.enprox_w!, getfield(K, st), y, x, c0, a, w), outarr(y))
    end
    push!(cases, c)

    c = Case(id("prox_g"), "enet", REFNAME; rtol = tol, expected = bigref ? T.(big_shrink.(x, big.(γv) .* b1) ./ (1 .+ big.(γv) .* b2)) : nothing)
    for (name, pb) in pbs
      has(pb) && add!(c, name, 2, thunk(pb.prox!, y, pb.ElasticNet(λ1, λ2), x, γv), outarr(y))
    end
    for st in (:st_maxmin, :st_clampz)
      add!(c, "k_$st", 1, thunk(K.enprox_g!, getfield(K, st), y, x, γv, λ1, λ2), outarr(y))
    end
    push!(cases, c)

    vtol = 8 * n * eps(T)
    c = Case(id("value"), "enet", REFNAME; rtol = vtol, expected = bigref ? T(b1 * sum(abs ∘ big, x) + b2 * sum(abs2 ∘ big, x) / 2) : nothing)
    for (name, pb) in pbs
      has(pb) && add!(c, name, 2, thunkref(r, pb.ElasticNet(λ1, λ2), x), outref(r))
    end
    add!(c, "k_onepass", 1, thunkref(r, K.envalue, x, λ1, λ2), outref(r))
    add!(c, "po", 2, thunkref(r, PO.ElasticNet(λ1, λ2), x), outref(r))
    push!(cases, c)

    ex = bigref ? (yb = big_shrink.(x, bγ * b1) ./ (1 + bγ * b2); (T(b1 * sum(abs, yb) + b2 * sum(abs2, yb) / 2), T.(yb))) : nothing
    c = Case(id("valprox"), "enet", REFNAME; rtol = vtol, expected = ex)
    for (name, pb) in pbs
      has(pb) && add!(c, name, 2, thunkref(r, pb.value_and_prox!, y, pb.ElasticNet(λ1, λ2), x, γ), outboth(r, y))
    end
    add!(c, "po", 2, thunkref(r, PO.prox!, y, PO.ElasticNet(λ1, λ2), x, γ), outboth(r, y))
    push!(cases, c)

    # the scalar operator in a loop over the coordinates: ElasticNet and NormL1
    if n <= 16384
      c = Case(id("block_loop"), "enet", REFNAME; rtol = tol, expected = T.(big_shrink.(x, bγ * b1 .* big.(w)) ./ (1 + bγ * b2)))
      for (name, pb) in pbs
        has(pb) && add!(c, name, 2, block_loop_thunk(pb.prox_block, y, pb.ElasticNet(λ1, λ2, w), x, γ), outarr(y))
      end
      push!(cases, c)
      c = Case("l1/block_loop/$T/n$n/normal", "l1", REFNAME; rtol = tol, expected = T.(big_shrink.(x, bγ * b1 .* big.(w))))
      for (name, pb) in pbs
        add!(c, name, 2, block_loop_thunk(pb.prox_block, y, pb.NormL1(λ1, w), x, γ), outarr(y))
      end
      push!(cases, c)
    end
  end
  filter!(c -> !isempty(c.impls) && any(i -> i.name == c.ref, c.impls), cases)
end

####################################
#  Tiny inputs and failing guards: the cases behind the small-call findings of the patch review
####################################

# zero groups at a threshold below the guard of the zero-group shortcut (level floatmin) and at an ordinary one
function cases_zero_groups(pbs, ::Type{T}) where {T}
  cases = Case[]
  for (m, Kg) in ((1, 1000), (5, 200), (32, 32), (1024, 4)), (lname, λ) in (("floatmin", floatmin(T)), ("ordinary", T(0.3)))
    x = zeros(T, m * Kg); y = similar(x); groups = [(k-1)*m+1:k*m for k in 1:Kg]; w = ones(T, Kg)
    c = Case("tinygroup/prox/$T/m$m/zero/$lname", "tiny", REFNAME; rtol = 0.0, expected = zeros(T, m * Kg))
    for (name, pb) in pbs
      add!(c, name, 2, thunk(pb.prox!, y, pb.GroupNormL2(λ, groups, w), x, one(T)), outarr(y))
    end
    push!(cases, c)
  end
  cases
end
const TINY_SIZES = (1, 2, 3, 4, 7, 8, 9, 12, 15, 17, 24, 31, 33, 48, 100)

####################################
#  C. Least squares
####################################

function cases_ls(pbs, K, KC, PO, ::Type{T}; lens = (100, 1000, 10000, 100000, 1000000),
                  shapes = ((100, 20), (1000, 100), (2000, 500), (500, 2000)),
                  proxshapes = ((200, 50), (1000, 200), (2000, 500), (200, 1000))) where {T}
  cases = Case[]
  r = Ref(zero(T))
  # --- values at a residual or predictor
  for n in lens
    rng = MersenneTwister(3000 + n)
    res, z, Y = randn(rng, T, n), randn(rng, T, n), randn(rng, T, n)
    X0 = zeros(T, n, 0)
    tol = 4 * n * eps(T)
    c = Case("ls/value_res/$T/n$n", "ls", REFNAME; rtol = tol, expected = n <= 100000 ? T(sum(abs2 ∘ big, res) / 2) : nothing)
    for (name, pb) in pbs
      pb.value_from_residual === nothing && continue
      add!(c, name, 2, thunkref(r, pb.value_from_residual, pb.LeastSquaresLoss(Y, X0), res), outref(r))
    end
    for nm in (:halfsumsq_simd, :halfsumsq_base, :halfsumsq_dot, :halfsumsq_nrm2)
      add!(c, "k_$nm", 1, thunkref(r, getfield(K, nm), res), outref(r))
    end
    add!(c, "kC_halfsumsq_simd", 1, thunkref(r, KC.halfsumsq_simd, res), outref(r))
    push!(cases, c)
    c = Case("ls/value_pred/$T/n$n", "ls", REFNAME; rtol = tol, expected = n <= 100000 ? T(sum(abs2, big.(z) .- big.(Y)) / 2) : nothing)
    for (name, pb) in pbs
      pb.value_from_predictor === nothing && continue
      add!(c, name, 2, thunkref(r, pb.value_from_predictor, pb.LeastSquaresLoss(Y, X0), z), outref(r))
    end
    add!(c, "k_simd", 1, thunkref(r, K.halfsumsq_diff_simd, z, Y), outref(r))
    add!(c, "kC_simd", 1, thunkref(r, KC.halfsumsq_diff_simd, z, Y), outref(r))
    push!(cases, c)
  end
  # --- value, value and gradient
  for (n, p) in shapes
    rng = MersenneTwister(3100 + n + p)
    X, Y, β = randn(rng, T, n, p), randn(rng, T, n), randn(rng, T, p) ./ T(sqrt(p))
    z = X * β
    g, d = zeros(T, p), zeros(T, n)
    tol = sqrt(eps(T))
    rb = big.(X) * big.(β) .- big.(Y)
    exv, exg = T(sum(abs2, rb) / 2), T.(big.(X)' * rb)
    c = Case("ls/value/$T/$(n)x$p", "ls", REFNAME; rtol = tol, expected = exv)
    for (name, pb) in pbs
      add!(c, name, 2, thunkref(r, pb.LeastSquaresLoss(Y, X), β), outref(r))
    end
    add!(c, "k_direct", 1, thunkref(r, ls_value_direct, K.halfsumsq_diff_simd, d, X, Y, β), outref(r))
    add!(c, "po", 2, thunkref(r, PO.LeastSquares(X, Y), β), outref(r))
    push!(cases, c)
    c = Case("ls/valgrad/$T/$(n)x$p", "ls", REFNAME; rtol = tol, expected = (exv, exg))
    for (name, pb) in pbs
      add!(c, name, 2, thunkref(r, pb.value_and_gradient!, g, pb.LeastSquaresLoss(Y, X), β), outboth(r, g))
    end
    add!(c, "k_three", 1, thunkref(r, K.ls_valgrad!, g, d, X, Y, β), outboth(r, g))
    add!(c, "kC_three", 1, thunkref(r, KC.ls_valgrad!, g, d, X, Y, β), outboth(r, g))
    add!(c, "k_fused", 1, thunkref(r, K.ls_valgrad_fused!, g, d, X, Y, β), outboth(r, g))
    add!(c, "k_gemv", 1, thunkref(r, K.ls_valgrad_gemv!, g, d, X, Y, β), outboth(r, g))
    add!(c, "po", 2, thunkref(r, po_valgrad!, PO.gradient!, g, PO.LeastSquares(X, Y), β), outboth(r, g))
    push!(cases, c)
    db = big.(z) .- big.(Y)                      # the predictor as supplied (the rounded Xβ), not the exact product
    c = Case("ls/valgrad_pred/$T/$(n)x$p", "ls", REFNAME; rtol = tol, expected = (T(sum(abs2, db) / 2), T.(big.(X)' * db)))
    for (name, pb) in pbs
      pb.value_and_gradient_from_predictor! === nothing && continue
      add!(c, name, 2, thunkref(r, pb.value_and_gradient_from_predictor!, g, d, pb.LeastSquaresLoss(Y, X), z), outboth(r, g))
    end
    add!(c, "k_three", 1, thunkref(r, K.ls_valgrad_pred!, g, d, X, Y, z), outboth(r, g))
    add!(c, "k_fused", 1, thunkref(r, K.ls_valgrad_pred_fused!, g, d, X, Y, z), outboth(r, g))
    push!(cases, c)
  end
  # --- the proximal operator: cached SVD (ProximalBase) and cached Cholesky
  for (n, p) in proxshapes
    rng = MersenneTwister(3200 + n + p)
    X, Y, v = randn(rng, T, n, p), randn(rng, T, n), randn(rng, T, p)
    β = zeros(T, p)
    γ1, γ2 = T(0.7), T(1.3)
    tol = T === Float32 ? 2f-2 : 1e-8
    ex = n * p <= 250000 ? T.((big.(X)' * big.(X) + I / big(γ1)) \ (big.(X)' * big.(Y) .+ big.(v) ./ big(γ1))) : nothing
    c = Case("ls/prox_fixed/$T/$(n)x$p", "ls", REFNAME; rtol = tol, expected = ex)
    for (name, pb) in pbs
      f = pb.LeastSquaresLoss(Y, X)
      pb.prox!(β, f, v, γ1)                    # the SVD is computed here, outside the timing
      add!(c, name, 2, thunk(pb.prox!, β, f, v, γ1), outarr(β); note = "cached thin SVD")
    end
    ch = K.CholLS(X, Y)
    K.chol_prox!(β, ch, v, γ1)
    add!(c, "k_chol", 1, thunk(K.chol_prox!, β, ch, v, γ1), outarr(β); note = "cached Cholesky; fixed step")
    pof = PO.LeastSquares(X, Y)
    PO.prox!(β, pof, v, γ1)
    add!(c, "po", 2, thunk(PO.prox!, β, pof, v, γ1), outarr(β); note = "cached Cholesky; also returns the value")
    push!(cases, c)

    # two calls with different steps: a Cholesky factor is recomputed at each call
    c = Case("ls/prox_vary/$T/$(n)x$p", "ls", REFNAME; rtol = tol)
    β2 = zeros(T, p)
    for (name, pb) in pbs
      f = pb.LeastSquaresLoss(Y, X)
      pb.prox!(β, f, v, γ1)
      add!(c, name, 2, two_steps_thunk(pb.prox!, β, β2, f, v, γ1, γ2), outboth2(β, β2))
    end
    ch2 = K.CholLS(X, Y)
    add!(c, "k_chol", 1, two_steps_thunk(K.chol_prox!, β, β2, ch2, v, γ1, γ2), outboth2(β, β2); note = "refactorized at each call")
    pof2 = PO.LeastSquares(X, Y)
    add!(c, "po", 2, two_steps_thunk(PO.prox!, β, β2, pof2, v, γ1, γ2), outboth2(β, β2); note = "refactorized at each call")
    push!(cases, c)

    # construction and the first prox (the factorization)
    c = Case("ls/prox_first/$T/$(n)x$p", "ls", REFNAME; rtol = tol, expected = ex)
    for (name, pb) in pbs
      add!(c, name, 2, first_prox_thunk(pb.LeastSquaresLoss, pb.prox!, β, Y, X, v, γ1), outarr(β); note = "thin SVD")
    end
    add!(c, "k_chol", 1, first_prox_thunk_k(K.CholLS, K.chol_prox!, β, Y, X, v, γ1), outarr(β); note = "Gram matrix and Cholesky")
    add!(c, "po", 2, first_prox_thunk_po(PO.LeastSquares, PO.prox!, β, Y, X, v, γ1), outarr(β))
    push!(cases, c)
  end
  cases
end
ls_value_direct(hs::F, d, X, Y, β) where {F} = (mul!(d, X, β); hs(d, Y))
# ProximalOperators' gradient! returns the value
po_valgrad!(gradient!::F, g, f, β) where {F} = gradient!(g, f, β)
two_steps_thunk(prox!::F, β, β2, f, v, γ1, γ2) where {F} = () -> (prox!(β, f, v, γ1); prox!(β2, f, v, γ2); nothing)
outboth2(a, b) = () -> (a, b)
first_prox_thunk(L::C, prox!::F, β, Y, X, v, γ) where {C, F} = () -> (prox!(β, L(Y, X), v, γ); nothing)
first_prox_thunk_k(L::C, prox!::F, β, Y, X, v, γ) where {C, F} = () -> (prox!(β, L(X, Y), v, γ); nothing)
first_prox_thunk_po(L::C, prox!::F, β, Y, X, v, γ) where {C, F} = () -> (prox!(β, L(X, Y), v, γ); nothing)
