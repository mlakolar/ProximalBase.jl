# Supplementary bitwise characterization: the recovery paths that the #865 harness
# (docs/audit/numerical-kernels-865/characterize.jl) does not reach.
#
#   julia --project=. docs/audit/numerical-kernels-file/characterize_extra.jl BASELINE_TREE CANDIDATE_TREE
#
# Same method as the #865 harness: both source trees are loaded as modules in one process, run the
# same grid of ordinary and exceptional inputs, and every result is compared bit for bit (NaN
# payloads ignored, exceptions by type). Covers QuadraticFunction (value, fused/standalone gradient,
# eigen prox), the LeastSquaresLoss SVD prox, Scaled and nested Scaled (value, gradient, prox) over
# the built-in families, the sparse/symmetric triple product A * X * B with its non-finite entry
# recovery, dense-by-sparse products, and norm_diff/SqrNormL2 at symmetric owners. Exits nonzero on
# any difference.
using LinearAlgebra, SparseArrays, Random

const TREES = length(ARGS) >= 2 ? abspath.(ARGS[1:2]) : error("usage: characterize_extra.jl BASELINE CANDIDATE")
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

exps(::Type{Float64}) = (-1074, -1060, -1022, -1000, -540, -30, 0, 30, 540, 1000, 1022, 1023, 1024)
exps(::Type{Float32}) = (-149, -140, -126, -120, -60, -10, 0, 10, 60, 120, 126, 127, 128)
specials(::Type{T}) where {T} = (T(NaN), T(Inf), -T(Inf), zero(T), -zero(T))
scaled(rng, ::Type{T}, e) where {T} = (s = rand(rng, Bool) ? -1 : 1; T(s) * ldexp(T(0.5) + T(rand(rng)) / 2, e))
emax(T) = exponent(floatmax(T))
steps(::Type{T}) where {T} = (one(T), ldexp(one(T), -emax(T) + 2), ldexp(one(T), emax(T) - 1), ldexp(one(T), emax(T) ÷ 2 + 3), T(3))

function vectors(rng, ::Type{T}, n) where {T}
  E = exps(T)
  out = Vector{Vector{T}}()
  for e in E
    push!(out, [scaled(rng, T, e) for _ in 1:n])
    push!(out, [scaled(rng, T, rand(rng, E)) for _ in 1:n])
  end
  for s in specials(T)
    v = [scaled(rng, T, rand(rng, E)) for _ in 1:n]
    v[rand(rng, 1:n)] = s
    push!(out, v)
  end
  push!(out, zeros(T, n))
  out
end

# --- QuadraticFunction ------------------------------------------------------------------------------
function quadratic_matrices(rng, ::Type{T}, n) where {T}
  E = exps(T)
  Ms = Any[]
  for e in (0, E[3] + 2, E[2] + 8, emax(T) - 3, emax(T) ÷ 2)      # finite PSD matrices at the scale 2^e:
    B = T[scaled(rng, T, 0) for _ in 1:n, _ in 1:n]               # near floatmin, subnormal, near floatmax
    push!(Ms, Matrix(Symmetric(ldexp.(B' * B, e))))
  end
  push!(Ms, Diagonal(T[abs(scaled(rng, T, rand(rng, E))) for _ in 1:n]))
  push!(Ms, Diagonal(T[abs(scaled(rng, T, E[2])) for _ in 1:n]))
  S = Matrix(Symmetric(T[scaled(rng, T, rand(rng, E[4:end-3])) for _ in 1:n, _ in 1:n]))
  push!(Ms, S)                                                      # indefinite, mixed scales
  push!(Ms, sparse(Ms[1]))
  if n >= 2                                                         # x'Ax cancels exactly at x = (1, 1, …)
    a = ldexp(T(0.75), emax(T) - 2)
    C = zeros(T, n, n); C[1, 1] = C[2, 2] = a; C[1, 2] = C[2, 1] = -a
    push!(Ms, C)
  end
  Ms
end

function quadratic_cases!(res, PB, ::Type{T}, rng) where {T}
  for n in (1, 2, 4)
    vs = vectors(rng, T, n)
    n >= 2 && push!(vs, ones(T, n), ldexp.(ones(T, n), emax(T) ÷ 2), ldexp.(ones(T, n), exps(T)[3] ÷ 2))
    for A in quadratic_matrices(rng, T, n), b in (nothing, T[scaled(rng, T, rand(rng, exps(T))) for _ in 1:n]),
        c in (zero(T), scaled(rng, T, rand(rng, exps(T)[2:end-1])))
      f = PB.QuadraticFunction(A, b, c)
      for x in vs
        push!(res, attempt(() -> f(x)))
        push!(res, attempt(() -> (g = similar(x); v = PB.value_and_gradient!(g, f, x); (v, g))))
        push!(res, attempt(() -> PB.gradient!(similar(x), f, x)))
        for γ in steps(T)
          push!(res, attempt(() -> PB.prox!(similar(x), f, x, γ)))
        end
      end
    end
  end
end

# --- LeastSquaresLoss proximal operator -----------------------------------------------------------
function leastsquares_prox_cases!(res, PB, ::Type{T}, rng) where {T}
  E = exps(T)
  for trial in 1:60
    n, p = rand(rng, 1:4), rand(rng, 1:4)
    eX = rand(rng, (0, E[3] ÷ 2, E[end-2] ÷ 2, emax(T) ÷ 2 + 2, rand(rng, E[2:end-1])))
    X = T[scaled(rng, T, eX) for _ in 1:n, _ in 1:p]
    Y = T[scaled(rng, T, rand(rng, E)) for _ in 1:n]
    f = PB.LeastSquaresLoss(Y, X)
    fm = PB.LeastSquaresLoss(hcat(Y, -Y), X)
    for v in (T[scaled(rng, T, rand(rng, E)) for _ in 1:p], zeros(T, p)), γ in steps(T)
      push!(res, attempt(() -> PB.prox!(similar(v), f, v, γ)))
      push!(res, attempt(() -> (y = similar(v); w = PB.value_and_prox!(y, f, v, γ); (w, y))))
      push!(res, attempt(() -> PB.prox!(similar(v, p, 2), fm, hcat(v, -v), γ)))
    end
  end
end

# --- Scaled and nested Scaled ---------------------------------------------------------------------
function scaled_cases!(res, PB, ::Type{T}, rng) where {T}
  E = exps(T)
  n = 3
  vs = vectors(rng, T, n)
  αs = (ldexp(one(T), emax(T) - 1), ldexp(one(T), -emax(T) + 2), ldexp(one(T), emax(T) ÷ 2), T(3), T(0.5),
        Float64(2)^700, 2.0^-700)
  X = T[scaled(rng, T, rand(rng, (0, E[end-2] ÷ 2))) for _ in 1:4, _ in 1:n]
  fams = Any[PB.SqrNormL2(one(T)), PB.SqrNormL2(ldexp(one(T), E[end-2]), T[1, -2, 3]), PB.NormL1(one(T)),
             PB.NormL1(ldexp(one(T), E[4]), T[1, 0, Inf]), PB.LinearFunction(T[1, -1, 2], one(T)),
             PB.LeastSquaresLoss(T[scaled(rng, T, 0) for _ in 1:4], X), PB.QuadraticFunction(Diagonal(T[1, 2, 3]))]
  for f in fams, α in αs, x in vs
    for s in (PB.Scaled(f, α), PB.Scaled(PB.Scaled(f, α), α), PB.Scaled(PB.Scaled(f, α), inv(α)))
      push!(res, attempt(() -> s(x)))
      PB.issmooth(s) && push!(res, attempt(() -> (g = similar(x); v = PB.value_and_gradient!(g, s, x); (v, g))))
      PB.issmooth(s) && push!(res, attempt(() -> PB.gradient!(similar(x), s, x)))
      for γ in (one(T), ldexp(one(T), E[3]))
        push!(res, attempt(() -> PB.prox!(similar(x), s, x, γ)))
        push!(res, attempt(() -> (y = similar(x); w = PB.value_and_prox!(y, s, x, γ); (w, y))))
      end
    end
  end
end

# --- Sparse and symmetric products ---------------------------------------------------------------
function product_cases!(res, PB, ::Type{T}, rng) where {T}
  E = exps(T)
  h = emax(T) ÷ 2 + 2
  for trial in 1:80
    m = rand(rng, 2:4)
    eA, eX, eB = rand(rng, (0, h, -h, rand(rng, E))), rand(rng, (0, h, -h, rand(rng, E))), rand(rng, (0, h, -h, rand(rng, E)))
    A = Symmetric(T[scaled(rng, T, eA) for _ in 1:m, _ in 1:m])
    B = Symmetric(T[scaled(rng, T, eB) for _ in 1:m, _ in 1:m])
    X = PB.SparseIterate(T, m, m)
    S = PB.SymmetricSparseIterate(T, m)
    for _ in 1:rand(rng, 1:4)
      r, c = rand(rng, 1:m), rand(rng, 1:m)
      v = scaled(rng, T, eX)
      trial % 9 == 0 && (v = rand(rng, specials(T)))
      X[r, c] = v
      S[max(r, c), min(r, c)] = v
    end
    if trial % 4 == 0                                   # +big +big -big at one entry: the rescue path
      a = ldexp(T(0.75), h)
      A = Symmetric(fill(a, m, m)); B = Symmetric(fill(a, m, m))
      X = PB.SparseIterate(T, m, m); X[1, 1] = a; X[2, 2] = a; X[1, 2] = -a
      S = PB.SymmetricSparseIterate(T, m); S[1, 1] = a; S[2, 2] = a; S[2, 1] = -a / 2
    end
    push!(res, attempt(() -> A * X * B))
    push!(res, attempt(() -> A * S * B))
    x = PB.SparseIterate(T, m)
    x[1] = scaled(rng, T, eX); x[m] = scaled(rng, T, rand(rng, E))
    push!(res, attempt(() -> Matrix(A) * x))
    push!(res, attempt(() -> A * x))
    push!(res, attempt(() -> transpose(Matrix(B)) * x))
    push!(res, attempt(() -> PB.norm_diff(S, Matrix(B))))
    push!(res, attempt(() -> PB.norm_diff(S, S)))
    push!(res, attempt(() -> PB.norm_diff(S, Matrix(B), Inf)))
    push!(res, attempt(() -> PB.SqrNormL2(one(T))(S)))
  end
end

function run_module(PB)
  res = Any[]
  for T in (Float64, Float32)
    rng = Xoshiro(20261111)
    quadratic_cases!(res, PB, T, rng)
    leastsquares_prox_cases!(res, PB, T, rng)
    scaled_cases!(res, PB, T, rng)
    product_cases!(res, PB, T, rng)
  end
  res
end

a, b = map(run_module, MODULES)
length(a) == length(b) || error("case counts differ: $(length(a)) vs $(length(b))")
bad = findall(i -> a[i] != b[i], eachindex(a))
nerr = count(r -> r isa Tuple && length(r) == 2 && r[1] === :error, a)
println("characterized $(length(a)) results ($nerr raise in both); $(length(bad)) differ")
for i in first(bad, 10)
  println("  #$i\n    baseline:  ", a[i], "\n    candidate: ", b[i])
end
isempty(bad) || exit(1)
