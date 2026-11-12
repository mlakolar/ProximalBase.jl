# Bitwise characterization of the numerical recovery paths touched by issue #865.
#
#   julia --project=. docs/audit/numerical-kernels-865/characterize.jl BASELINE_TREE CANDIDATE_TREE
#
# Loads both source trees as separate modules in one process, runs the same grid of ordinary and
# exceptional inputs through the public oracles (and the value-parts mechanism through `Scaled`),
# and compares every result bit for bit (NaN payloads ignored). Exceptions compare by type.
# Exits nonzero on any difference.
using LinearAlgebra, SparseArrays, Random

const TREES = length(ARGS) >= 2 ? abspath.(ARGS[1:2]) : error("usage: characterize.jl BASELINE CANDIDATE")
module Baseline end
module Candidate end
Base.include(Baseline, joinpath(TREES[1], "src", "ProximalBase.jl"))
Base.include(Candidate, joinpath(TREES[2], "src", "ProximalBase.jl"))
const MODULES = (Baseline.ProximalBase, Candidate.ProximalBase)

canon(x::AbstractFloat) = isnan(x) ? (typeof(x), "NaN") : (typeof(x), bitstring(x))
canon(x::Integer) = x
canon(x::Nothing) = nothing
canon(x::Tuple) = map(canon, x)
canon(x::AbstractArray) = map(canon, collect(x))
canon(x) = applicable(Matrix, x) ? (nameof(typeof(x)), map(canon, Matrix(x))) : (nameof(typeof(x)), string(x))

function attempt(f)
  try
    canon(f())
  catch e
    (:error, nameof(typeof(e)))
  end
end

# Exponent grids that reach both ends of the range, the subnormals included.
exps(::Type{Float64}) = (-1074, -1060, -1022, -1000, -540, -30, 0, 30, 540, 1000, 1022, 1023, 1024)
exps(::Type{Float32}) = (-149, -140, -126, -120, -60, -10, 0, 10, 60, 120, 126, 127, 128)
specials(::Type{T}) where {T} = (T(NaN), T(Inf), -T(Inf), zero(T), -zero(T))

scaled(rng, ::Type{T}, e) where {T} = (s = rand(rng, Bool) ? -1 : 1; T(s) * ldexp(T(0.5) + T(rand(rng)) / 2, e))

function vectors(rng, ::Type{T}, n) where {T}
  E = exps(T)
  out = Vector{Vector{T}}()
  for e in E
    push!(out, [scaled(rng, T, e) for _ in 1:n])                         # one scale
    push!(out, [scaled(rng, T, rand(rng, E)) for _ in 1:n])              # mixed scales
  end
  for s in specials(T)
    v = [scaled(rng, T, rand(rng, E)) for _ in 1:n]
    v[rand(rng, 1:n)] = s
    push!(out, v)
  end
  push!(out, zeros(T, n))
  out
end

levels(::Type{T}) where {T} = (one(T), zero(T), T(Inf), ldexp(one(T), exps(T)[2]), ldexp(one(T), exps(T)[4]),
                               ldexp(T(0.75), exps(T)[end-2]), ldexp(one(T), exps(T)[end-1] - 1), T(3))

# --- SqrNormL2 ---------------------------------------------------------------------------------
function ridge_cases!(res, PB, ::Type{T}, rng) where {T}
  n = 5
  vs = vectors(rng, T, n)
  centers = Any[nothing]
  for _ in 1:6
    push!(centers, rand(rng, vs))
  end
  T === Float32 && for _ in 1:4
    push!(centers, Float64[scaled(rng, Float64, rand(rng, (-1060, -300, -140, -20, 0, 20, 130, 300, 1000))) for _ in 1:n])
  end
  push!(centers, big.(rand(rng, vs)))
  # centers that land on the recovery path: entries below T's normal range on rounding beside entries
  # that round, at the scale of the small arguments
  tiny = (exps(T)[1] - 10, exps(T)[1], exps(T)[3] - 1, exps(T)[3] + 2)
  W = T === Float32 ? Float64 : BigFloat
  for _ in 1:4
    push!(centers, W[rand(rng, Bool) ? ldexp(W(rand(rng)) + 1, rand(rng, tiny)) : W(scaled(rng, T, rand(rng, tiny))) * (1 + W(rand(rng)) * eps(T)) for _ in 1:n])
  end
  # opposite-sign centers make x - c overflow (the halved residual); nextfloat makes it cancel
  for x in vs, λ in levels(T), c in (centers..., -x, nextfloat.(x), T(-0.75) .* x)
    c !== nothing && length(c) != length(x) && continue
    f = PB.SqrNormL2(λ, c)
    push!(res, attempt(() -> f(x)))
    push!(res, attempt(() -> PB.gradient!(similar(x), f, x)))
    push!(res, attempt(() -> (g = similar(x); v = PB.value_and_gradient!(g, f, x); (v, g))))
    for γ in (one(T), ldexp(one(T), exps(T)[3]), ldexp(one(T), exps(T)[end-1]), T(Inf))
      push!(res, attempt(() -> PB.prox!(similar(x), f, x, γ)))
      push!(res, attempt(() -> PB.prox_block(f, x[2], 2, γ)))
    end
    for α in (ldexp(one(T), exps(T)[end-1]), ldexp(one(T), exps(T)[3]), Float64(2)^700, 2.0^-700)
      push!(res, attempt(() -> PB.Scaled(f, α)(x)))
    end
  end
  # Float64 gradient destination for a Float32 argument (model precision of the gradient)
  if T === Float32
    for x in vs[1:8], λ in levels(T), c in centers[1:4]
      f = PB.SqrNormL2(λ, c)
      push!(res, attempt(() -> PB.gradient!(zeros(Float64, n), f, x)))
    end
  end
  # stored iterates: sparse, block and symmetric owners
  for x in vs[1:12], λ in levels(T)
    p = length(x)
    xs = PB.SparseIterate(T, 2p)
    for i in 1:p
      xs[2i] = x[i]
    end
    xb = PB.BlockIterate(T, 2p, [collect(i:p:2p) for i in 1:p])
    for i in 1:p
      xb[i] = x[i]
    end
    xm = PB.SymmetricSparseIterate(T, 3)
    for (k, (r, cc)) in enumerate(((1, 1), (2, 1), (3, 2), (3, 3), (3, 1)))
      xm[r, cc] = x[k]
    end
    f = PB.SqrNormL2(λ)
    for z in (xs, xb, xm)
      push!(res, attempt(() -> f(z)))
      push!(res, attempt(() -> PB.Scaled(f, ldexp(one(T), exps(T)[end-1]))(z)))
      push!(res, attempt(() -> (g = copy(z); PB.gradient!(g, f, z); g)))
      push!(res, attempt(() -> (g = copy(z); PB.prox!(g, f, z, ldexp(one(T), exps(T)[3])); g)))
    end
    fc = PB.SqrNormL2(λ, reshape(T[scaled(rng, T, rand(rng, exps(T))) for _ in 1:9], 3, 3))
    push!(res, attempt(() -> fc(xm)))
    push!(res, attempt(() -> (g = zeros(T, 3, 3); PB.gradient!(g, fc, xm); g)))
  end
end

# --- LeastSquaresLoss ---------------------------------------------------------------------------
function leastsquares_cases!(res, PB, ::Type{T}, rng) where {T}
  E = exps(T)
  for trial in 1:120
    n, p = rand(rng, 1:4), rand(rng, 1:4)
    eX, eβ, eY = rand(rng, E), rand(rng, E), rand(rng, E)
    X = T[scaled(rng, T, rand(rng, Bool) ? eX : rand(rng, E)) for _ in 1:n, _ in 1:p]
    β = T[scaled(rng, T, eβ) for _ in 1:p]
    Y = T[scaled(rng, T, eY) for _ in 1:n]
    if trial % 5 == 0                                     # Y - Xβ overflows, Xβ itself finite
      h = exponent(floatmax(T)) ÷ 2
      X = T[scaled(rng, T, h) for _ in 1:n, _ in 1:p]
      β = T[abs(scaled(rng, T, h - 1 - ndigits(p, base = 2))) for _ in 1:p]
      Y = -X * β
    end
    if trial % 3 == 0                                     # partial products of Xβ overflow, Xβ cancels
      a = ldexp(T(0.75), exponent(floatmax(T)) ÷ 2 + 4)
      X = hcat(fill(a, n), fill(a, n), T[scaled(rng, T, 0) for _ in 1:n])
      β = T[a, -a, scaled(rng, T, rand(rng, E))]
      p = 3
    end
    trial % 7 == 0 && (Y = X * β)                         # an exact or nearly exact fit
    trial % 11 == 0 && (Y[1] = rand(rng, specials(T)))
    trial % 13 == 0 && (β[1] = rand(rng, specials(T)))
    f = PB.LeastSquaresLoss(Y, X)
    push!(res, attempt(() -> f(β)))
    push!(res, attempt(() -> (g = similar(β); v = PB.value_and_gradient!(g, f, β); (v, g))))
    push!(res, attempt(() -> PB.gradient!(similar(β), f, β)))
    bs = PB.SparseIterate(T, p)
    for i in 1:p
      iszero(β[i]) || (bs[i] = β[i])
    end
    push!(res, attempt(() -> f(bs)))
    Ym = hcat(Y, reverse(Y))
    fm = PB.LeastSquaresLoss(Ym, X)
    push!(res, attempt(() -> fm(hcat(β, -β))))
  end
end

# --- LinearFunction, NormL1, norm_diff ----------------------------------------------------------
function other_cases!(res, PB, ::Type{T}, rng) where {T}
  vs = vectors(rng, T, 4)
  for x in vs, c in vs[1:2:end]
    for d in (zero(T), scaled(rng, T, rand(rng, exps(T))))
      f = PB.LinearFunction(c, d)
      push!(res, attempt(() -> f(x)))
    end
    push!(res, attempt(() -> PB.norm_diff(x, c)))
  end
  for x in vs, λ in levels(T), w in (nothing, T[1, 0, Inf, 3], T[ldexp(one(T), exps(T)[end-1]), 1, ldexp(one(T), exps(T)[2]), 2])
    f = PB.NormL1(λ, w)
    push!(res, attempt(() -> f(x)))
    for γ in (one(T), ldexp(one(T), exps(T)[3]), ldexp(one(T), exps(T)[end-1]))
      push!(res, attempt(() -> PB.prox!(similar(x), f, x, γ)))
      push!(res, attempt(() -> PB.prox_block(f, x[3], 3, γ)))
    end
  end
end

function run_module(PB)
  res = Any[]
  for T in (Float64, Float32)
    rng = Xoshiro(865)
    ridge_cases!(res, PB, T, rng)
    leastsquares_cases!(res, PB, T, rng)
    other_cases!(res, PB, T, rng)
  end
  res
end

a, b = map(run_module, MODULES)
length(a) == length(b) || error("case counts differ: $(length(a)) vs $(length(b))")
bad = findall(i -> a[i] != b[i], eachindex(a))
nerr = count(r -> r isa Tuple && length(r) == 2 && r[1] === :error, a)
println("characterized $(length(a)) results ($nerr raise in both); $(length(bad)) differ")
# every differing leaf, as (baseline, candidate) classes
leaves(x) = x isa Tuple && length(x) == 2 && x[1] isa DataType && x[1] <: AbstractFloat ? [x] :
            x isa Union{Tuple, AbstractArray} ? reduce(vcat, map(leaves, collect(x)); init = Any[]) : Any[x]
class(l) = l isa Tuple && length(l) == 2 && l[1] isa DataType && l[1] <: AbstractFloat ?
           (l[2] == "NaN" ? "NaN" : (v = reinterpret(l[1] === Float64 ? Float64 : Float32, parse(l[1] === Float64 ? UInt64 : UInt32, l[2]; base = 2));
                              isinf(v) ? (v > 0 ? "+Inf" : "-Inf") : iszero(v) ? "zero" : "finite")) : string(l)
classes = Dict{Tuple{String, String}, Int}()
for i in bad
  for (u, w) in zip(leaves(a[i]), leaves(b[i]))
    u == w || (classes[(class(u), class(w))] = get(classes, (class(u), class(w)), 0) + 1)
  end
end
isempty(classes) || println("differing leaves (baseline => candidate): ", classes)
for i in first(bad, 10)
  println("  #$i\n    baseline:  ", a[i], "\n    candidate: ", b[i])
end
isempty(bad) || exit(1)
