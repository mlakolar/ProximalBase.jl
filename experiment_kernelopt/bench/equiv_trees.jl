# Bitwise comparison of source trees through the public entry points of NormL1 and ElasticNet, on inputs that hold
# every class of value: for each element type, length, level, ridge level, weights and step (a scalar or an array),
# prox! out of place and in place, prox, value_and_prox!, prox_block entry by entry, and prox! on a SparseIterate are
# and prox! from a strided view into a view, are run in the reference tree and in every candidate tree, and their
# results are compared bit for bit (the sign of a zero and the sign and payload of a NaN included). Results that
# differ only in the bits of a NaN are counted separately. A call that throws must throw an exception of the same
# type in both.
#   julia --check-bounds=yes bench/equiv_trees.jl <reference root> <name>=<root> [<name>=<root> ...]
# Exit status 1 if any result differs, the bits of a NaN included.
using Random, Printf
function loadtree(name, root)
  m = Core.eval(Main, :(module $(Symbol("Tree_", name)) end))
  Base.include(m, joinpath(abspath(root), "src", "ProximalBase.jl"))
  Base.invokelatest(getfield, m, :ProximalBase)
end
const REF = loadtree("ref", ARGS[1])
const CANDS = [(String(split(a, '=')[1]), loadtree(String(split(a, '=')[1]), String(split(a, '=')[2]))) for a in ARGS[2:end]]

outcome(g) = try
  (true, g())
catch e
  (false, nameof(typeof(e)))
end
bitsof(a::AbstractArray{T}) where {T<:AbstractFloat} = reinterpret(Base.uinttype(T), collect(a))
bitsof(a::AbstractFloat) = [reinterpret(Base.uinttype(typeof(a)), a)]
# 0: the same bits; 1: NaNs with different bits; 2: different
function cmp(a, b)
  (a isa Tuple) && return maximum(cmp(a[i], b[i]) for i in eachindex(a))
  size(a) == size(b) || return 2
  ba, bb = bitsof(a), bitsof(b)
  ba == bb && return 0
  all(i -> ba[i] == bb[i] || (isnan(a[i]) && isnan(b[i])), eachindex(ba)) ? 1 : 2
end
function compare(ro, co)
  ro[1] == co[1] || return 2
  ro[1] ? cmp(ro[2], co[2]) : (ro[2] === co[2] ? 0 : 2)
end

# a quiet NaN with a payload, and a signaling NaN
payloadnan(::Type{T}, payload) where {T} = reinterpret(T, reinterpret(Base.uinttype(T), T(NaN)) | Base.uinttype(T)(payload))
signaling(::Type{T}) where {T} = reinterpret(T, (reinterpret(Base.uinttype(T), T(Inf)) | Base.uinttype(T)(1)))
# the entries: ordinary values, every special value, and values at and next to the thresholds ±c
function data(::Type{T}, n, cs, rng) where {T}
  x = T(2) .* randn(rng, T, n)
  sp = T[0, -0.0, nextfloat(zero(T)), -nextfloat(zero(T)), floatmin(T), -floatmin(T), prevfloat(floatmin(T)), 1, -1,
         floatmax(T), -floatmax(T), Inf, -Inf, NaN, -T(NaN), payloadnan(T, 0x5a5), -payloadnan(T, 0x123), signaling(T)]
  for c in cs
    (isfinite(c) && c > 0) || continue
    append!(sp, T[c, -c, nextfloat(c), -nextfloat(c), prevfloat(c), -prevfloat(c)])
  end
  n >= 3 * length(sp) ? (x[randperm(rng, n)[1:length(sp)]] .= sp) : (for i in 1:n; isodd(i) && (x[i] = sp[mod1(i + n, length(sp))]); end)
  x
end

# the entry points of one tree on one input; every result is an (ok, value or exception name) pair
function results(M, kind, T, λ, λ2, w, x, γ)
  f = outcome(() -> kind === :l1 ? M.NormL1(λ, w) : M.ElasticNet(λ, λ2, w))
  f[1] || return Any[f]
  f = f[2]
  out = Any[]
  push!(out, outcome(() -> (y = fill(T(7), size(x)); M.prox!(y, f, x, γ); y)))
  push!(out, outcome(() -> (z = copy(x); M.prox!(z, f, z, γ); z)))
  push!(out, outcome(() -> M.prox(f, x, γ)))
  push!(out, outcome(() -> (y = fill(T(7), size(x)); v = M.value_and_prox!(y, f, x, γ); (T(v), y))))
  m = min(length(x), 48)
  push!(out, outcome(() -> T[M.prox_block(f, x[i], i, γ isa AbstractArray ? γ[i] : γ) for i in 1:m]))
  push!(out, outcome(() -> (s = M.SparseIterate(x); o = copy(s); M.prox!(o, f, s, γ); collect(o))))
  # a strided view as the argument and a view as the output, with untouched neighbours
  push!(out, outcome(() -> (xb = repeat(x, inner = 2); yb = fill(T(7), length(x) + 2); M.prox!(view(yb, 2:length(x)+1), f, view(xb, 1:2:length(xb)), γ); yb)))
  out
end

function main()
  total = 0; nanbits = zeros(Int, length(CANDS)); diffs = zeros(Int, length(CANDS)); shown = 0
  for T in (Float32, Float64)
    rng = MersenneTwister(969)
    levels = T[0.6, 0, nextfloat(zero(T)), floatmin(T), floatmax(T), Inf]
    steps = T[0.8, 0, nextfloat(zero(T)), floatmin(T), floatmax(T), Inf, -1, NaN]
    for n in (1, 2, 3, 7, 8, 15, 16, 17, 31, 33, 100, 1024, 2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192, 8193, 20000), λ in levels, γ0 in steps
      n > 1024 && !(λ == T(0.6) || λ == floatmin(T) || isinf(λ)) && continue
      n > 1024 && !(γ0 == T(0.8) || γ0 == 0 || isinf(γ0) || isnan(γ0)) && continue
      c = γ0 * λ
      x = data(T, n, (c, c / 2, 2c), rng)
      wpos = T(0.5) .+ rand(rng, T, n)
      wmix = copy(wpos); wmix[1:3:end] .= zero(T); n > 1 && (wmix[2:5:end] .= T(Inf))
      γv = γ0 .* (T(0.5) .+ rand(rng, T, n))
      γx = copy(γv); γx[end] = zero(T); n > 2 && (γx[2] = T(Inf))
      for w in (nothing, wpos, wmix), γ in (γ0, γv, γx), (kind, λ2) in ((:l1, zero(T)), (:enet, zero(T)), (:enet, T(0.5)), (:enet, T(Inf)))
        ref = results(REF, kind, T, λ, λ2, w, x, γ)
        for (k, (name, M)) in enumerate(CANDS)
          got = results(M, kind, T, λ, λ2, w, x, γ)
          for j in eachindex(ref)
            total += k == 1
            r = length(got) == length(ref) ? compare(ref[j], got[j]) : 2
            nanbits[k] += r == 1; diffs[k] += r == 2
            if r == 2 && shown < 20
              shown += 1
              println("DIFF ", name, " ", T, " n=", n, " ", kind, " λ=", λ, " λ2=", λ2, " γ=", γ isa AbstractArray ? "array" : γ,
                      " w=", w === nothing ? "none" : (w === wpos ? "positive" : "mixed"), " entry point ", j)
            end
          end
        end
      end
    end
  end
  for (k, (name, _)) in enumerate(CANDS)
    @printf("%s: %d results compared with the reference, %d different, %d equal except for the bits of a NaN\n", name, total, diffs[k], nanbits[k])
  end
  exit(any(>(0), diffs) || any(>(0), nanbits) ? 1 : 0)
end
Base.invokelatest(main)
