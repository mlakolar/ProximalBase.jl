####################################
#
#  Partial decompositions
#
####################################

# With `partial = true`, the low-rank functions (NuclearNorm, IndBallRank,
# IndPSD, TracePSD) compute only the leading singular triplets or eigenpairs of
# their input instead of the full decomposition, which costs O(p³) for a p × p
# matrix. They run randomized subspace iteration (Halko, Martinsson and Tropp,
# 2011, §4.5) with a Rayleigh–Ritz step on a block of ℓ vectors, at O(p²ℓ) per
# iteration.
#
# A call starts from the block of the previous one after one more power step,
# with fresh random vectors in its last _PARTIAL_FRESH columns. A solver calls
# the proximal operator on nearby matrices, so few iterations suffice, and a
# direction that enters the leading subspace is amplified from call to call
# until the block holds it.
#
# A result is accepted when the residuals of the pairs kept are below
# _partial_rtol times the largest value, and the next _PARTIAL_GUARD pairs lie
# below the boundary (the threshold, or the smallest pair kept) by more than
# their residuals. That is no proof that the block missed no larger pair:
# from a random block the Ritz values are small and pass it, so a call that
# starts from one takes power steps first, and a call that starts from the
# previous block has had the previous calls' steps. Otherwise, after
# _PARTIAL_MAXITER iterations, the function falls back to the full
# decomposition, which starts the next call. A block of more than a quarter of
# the dimension takes the full decomposition directly: benchmark/partial.jl
# finds the two about equally fast there.

const _PARTIAL_MAXITER = 8      # subspace iterations of the SVD before falling back; the
                                # eigensolver, at one product per iteration, gets twice as many
const _PARTIAL_GUARD = 3        # pairs after those kept that must lie below the boundary
const _PARTIAL_FRESH = 2        # random vectors in each call's starting block

# the relative tolerance on the residuals of the pairs kept: eps^(3/4), or what
# the rounding of a product with a p × p matrix allows
_partial_rtol(::Type{T}, p::Integer) where {T} = max(eps(T)^T(0.75), 10 * eps(T) * sqrt(T(p)))

# the block size for `need` pairs: the oversampling makes the pairs kept
# converge at the rate of the gap to the (ℓ + 1)-st value
_partial_block(need::Integer) = need + max(10, need ÷ 4)

# fills the columns cols of A with numbers uniform on [-1, 1) from the
# xorshift generator whose state ws.seed advances: no allocation, and the same
# numbers in every session
function _randomize!(A::AbstractMatrix{T}, cols::UnitRange{Int}, ws) where {T}
  s = ws.seed
  @inbounds for j in cols, i in axes(A, 1)
    s ⊻= s << 13
    s ⊻= s >> 7
    s ⊻= s << 17
    A[i, j] = T((s >> 11) * 0x1.0p-52 - 1)
  end
  ws.seed = s
  A
end

# copies the columns b of B to the columns a of A, by loops (a view passed to a
# call that is not inlined would be allocated)
function _copycols!(A::Matrix{T}, a::UnitRange{Int}, B::AbstractMatrix{T}, b::UnitRange{Int}) where {T}
  @inbounds for (ja, jb) in zip(a, b)
    for i in axes(A, 1)
      A[i, ja] = B[i, jb]
    end
  end
  A
end

# X as a dense matrix of eltype T that BLAS can read: X itself, or a copy in
# the workspace's buffer
_dense_input(ws, X::StridedMatrix{T}, ::Type{T}) where {T} = X
_dense_input(ws, X::AbstractMatrix, ::Type{T}) where {T} = copyto!(_array_buffer!(ws.dense, T, size(X)), X)

####################################
#  Thin QR decomposition (geqrf, orgqr)
####################################

mutable struct QRWorkspace{T<:_BlasReal}
  m::Int
  n::Int
  tau::Vector{T}
  work::Vector{T}
  info::Base.RefValue{BlasInt}
end

for (geqrf, orgqr, T) in ((:dgeqrf_, :dorgqr_, :Float64), (:sgeqrf_, :sorgqr_, :Float32))
  @eval function _geqrf!(A::Matrix{$T}, tau::Vector{$T}, work::Vector{$T}, lwork::Integer,
                         info::Base.RefValue{BlasInt})
    m, n = size(A)
    ccall((@blasfunc($geqrf), libblastrampoline), Cvoid,
          (Ref{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ptr{$T}, Ref{BlasInt}, Ptr{BlasInt}),
          m, n, A, max(1, m), tau, work, lwork, info)
    info[]
  end
  @eval function _orgqr!(A::Matrix{$T}, tau::Vector{$T}, work::Vector{$T}, lwork::Integer,
                         info::Base.RefValue{BlasInt})
    m, n = size(A)
    ccall((@blasfunc($orgqr), libblastrampoline), Cvoid,
          (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ptr{$T},
           Ref{BlasInt}, Ptr{BlasInt}),
          m, n, length(tau), A, max(1, m), tau, work, lwork, info)
    info[]
  end
end

function QRWorkspace(::Type{T}, m::Integer, n::Integer) where {T<:_BlasReal}
  m >= n || throw(ArgumentError("need m ≥ n, got m = $m and n = $n"))
  A, tau, work, info = zeros(T, m, n), zeros(T, n), zeros(T, 1), Ref{BlasInt}(0)
  _geqrf!(A, tau, work, -1, info)                         # workspace queries
  lwork = max(1, round(Int, work[1]))
  _orgqr!(A, tau, work, -1, info)
  QRWorkspace{T}(m, n, tau, resize!(work, max(lwork, round(Int, work[1]))), info)
end

# Overwrites A (m × n, m ≥ n) with the Q factor of its thin QR decomposition,
# which has orthonormal columns also when A is rank deficient, and writes the
# R factor to the n × n matrix R unless R is nothing.
function _thin_qr!(A::Matrix{T}, ws::QRWorkspace{T}, R::Union{Nothing, Matrix{T}} = nothing) where {T}
  info = _geqrf!(A, ws.tau, ws.work, length(ws.work), ws.info)
  info == 0 || throw(LAPACKException(info))
  if R !== nothing
    n = size(A, 2)
    @inbounds for j in 1:n, i in 1:n
      R[i, j] = i <= j ? A[i, j] : zero(T)
    end
  end
  info = _orgqr!(A, ws.tau, ws.work, length(ws.work), ws.info)
  info == 0 || throw(LAPACKException(info))
  A
end

# ‖Y[:, c] - t U[:, c]‖, the residual of a pair
function _pair_residual(Y::Matrix{T}, U::Matrix{T}, t::T, c::Int) where {T}
  e = zero(T)
  @inbounds for i in axes(Y, 1)
    e += abs2(Y[i, c] - t * U[i, c])
  end
  sqrt(e)
end

####################################
#  Leading singular triplets
####################################

mutable struct PartialSVDWorkspace{T<:_BlasReal}
  m::Int
  n::Int
  ℓ::Int                       # the block size
  Q::Matrix{T}                 # m × ℓ, an orthonormal basis of the range estimate
  U::Matrix{T}                 # m × ℓ, the left singular vectors
  Y::Matrix{T}                 # m × ℓ, X V, and the starting block of the next call
  W::Matrix{T}                 # n × ℓ, XᵀQ, then the Q factor of its QR decomposition
  V::Matrix{T}                 # n × ℓ, the right singular vectors
  s::Vector{T}                 # the singular values, in descending order
  small::SVDWorkspace{T}       # the SVD of the ℓ × ℓ R factor
  qrm::QRWorkspace{T}
  qrn::QRWorkspace{T}
  dense::Scratch               # a dense m × n copy of an input or output that is not strided
  seed::UInt64
  warm::Int                    # the leading columns of Y that start the next call
  rank::Int                    # the number of triplets kept at the last call
  accepted::Bool               # the last partial attempt met the residual/guard checks
end

function PartialSVDWorkspace(::Type{T}, m::Integer, n::Integer, ℓ::Integer) where {T<:_BlasReal}
  PartialSVDWorkspace{T}(m, n, ℓ, zeros(T, m, ℓ), zeros(T, m, ℓ), zeros(T, m, ℓ), zeros(T, n, ℓ), zeros(T, n, ℓ),
                         zeros(T, ℓ), SVDWorkspace(T, ℓ, ℓ), QRWorkspace(T, m, ℓ), QRWorkspace(T, n, ℓ),
                         Scratch(), 0x9e3779b97f4a7c15, 0, 0, false)
end

# The workspace for `need` triplets of an m × n matrix: the cached one while
# its block is between ℓ and 2ℓ, so that the block stays while the rank
# changes little; nothing when the block would exceed a quarter of the
# dimension, where the full SVD is about as fast. A new workspace inherits the starting block.
function _partial_svd_workspace(cache, ::Type{T}, m::Integer, n::Integer, need::Integer) where {T<:_BlasReal}
  ℓ = _partial_block(need)
  4ℓ > min(m, n) && return nothing
  old = cache isa PartialSVDWorkspace{T} && cache.m == m && cache.n == n ? cache : nothing
  old !== nothing && ℓ <= old.ℓ <= 2ℓ && return old
  ws = PartialSVDWorkspace(T, m, n, ℓ)
  if old !== nothing
    w = min(old.warm, ℓ)
    _copycols!(ws.Y, 1:w, old.Y, 1:w)
    ws.warm, ws.rank, ws.seed = w, old.rank, old.seed
  end
  ws
end

# The leading singular triplets of X: the r largest when r > 0, otherwise all
# the singular values above τ (by more than the tolerance δ). Returns the number k of triplets kept, with the
# values in ws.s[1:k] and the vectors in the first k columns of ws.U and ws.V;
# -2 when fewer than _PARTIAL_GUARD values of the block are below τ, and -1
# when the iteration did not converge.
#
# Each iteration computes the SVD of B = QᵀX from the QR decomposition of
# W = XᵀQ = Q_W R and the SVD R = U_R Σ V_Rᵀ, so that B = V_R Σ (Q_W U_R)ᵀ:
# U = Q V_R and V = Q_W U_R. Then XᵀU = VΣ exactly, and Y = XV gives both the
# residuals XV - UΣ and the next block.
function _partial_svd!(ws::PartialSVDWorkspace{T}, X::StridedMatrix{T}, τ::T, r::Int) where {T}
  ws.accepted = false
  ℓ = ws.ℓ
  Q, U, Y, W, V, s = ws.Q, ws.U, ws.Y, ws.W, ws.V, ws.s
  w = min(ws.warm, ℓ - _PARTIAL_FRESH)
  _copycols!(Q, 1:w, Y, 1:w)
  _randomize!(Q, w+1:ℓ, ws)
  _thin_qr!(Q, ws.qrm)
  tol = _partial_rtol(T, max(ws.m, ws.n))
  minit = w == 0 ? 3 : 1             # a random block takes two power steps before it can be accepted
  for it in 1:_PARTIAL_MAXITER
    mul!(W, X', Q)
    _thin_qr!(W, ws.qrn, ws.small.A)
    sR, UR, VRt = _svd!(ws.small)
    mul!(U, Q, transpose(VRt))
    mul!(V, W, UR)
    copyto!(s, sR)
    mul!(Y, X, V)
    δ = tol * s[1]                     # values within δ of τ weigh at most δ: not kept
    k = r > 0 ? r : count(>(τ + δ), s)
    g = min(_PARTIAL_GUARD, ℓ - k)
    if r == 0 && g < _PARTIAL_GUARD
      ws.warm = ℓ
      return -2
    end
    if it >= minit && _svd_accepted(Y, U, s, k, g, τ + δ, r, δ)
      ws.warm, ws.rank = ℓ, k
      ws.accepted = true
      return k
    end
    copyto!(Q, Y)                                          # one more power step
    _thin_qr!(Q, ws.qrm)
  end
  ws.warm = ℓ
  return -1
end

# whether the k triplets kept have residuals at most bound, and the next g are
# certified below the boundary: σ + ‖Xv - σu‖ ≤ τ, or below the smallest pair
# kept when r > 0 (Xᵀu = σv holds by construction)
function _svd_accepted(Y::Matrix{T}, U::Matrix{T}, s::Vector{T}, k::Int, g::Int, τ::T, r::Int, bound::T) where {T}
  ρ = zero(T)
  for c in 1:k
    ρ = _pair_residual(Y, U, s[c], c)
    ρ <= bound || return false
  end
  edge = r > 0 && k > 0 ? s[k] - ρ : τ
  for c in k+1:k+g
    s[c] + _pair_residual(Y, U, s[c], c) <= edge || return false
  end
  return true
end

# Runs _partial_svd!, growing the block while too few of its values are below
# τ, and caches the workspace in f.pws. Returns the number of triplets kept, or
# -1 when the caller falls back to the full SVD.
function _leading_svd!(f, X::AbstractMatrix, ::Type{T}, τ::T, r::Int) where {T}
  m, n = size(X)
  cache = f.pws
  old = cache isa PartialSVDWorkspace{T} ? cache : nothing   # f.pws is untyped: no dynamic call
  prev = old === nothing ? 0 : old.rank
  ws = _partial_svd_workspace(old, T, m, n, r > 0 ? r : prev + _PARTIAL_GUARD)
  # A larger oversampling target may cross the cutoff even though the last
  # partial result passed with a sufficient warm block. Reuse that block;
  # a failed attempt or a full result must not trigger repeated partial retries.
  if ws === nothing && r == 0 && old !== nothing && old.accepted && old.warm > 0 &&
      old.m == m && old.n == n && 4old.ℓ <= min(m, n) && prev + _PARTIAL_GUARD <= old.ℓ
    ws = old
  end
  ws === nothing && return -1
  f.pws = ws
  Xd = _dense_input(ws, X, T)
  while true
    k = _partial_svd!(ws, Xd, τ, r)
    k == -2 || return k
    bigger = _partial_svd_workspace(ws, T, m, n, 2ws.ℓ)
    bigger === nothing && return -1
    f.pws = ws = bigger
  end
end

# The leading k triplets times the weights t: Y = U[:, 1:k] Diagonal(t.(σ)) V[:, 1:k]ᵀ
function _partial_svd_out!(Y::AbstractMatrix, ws::PartialSVDWorkspace{T}, k::Int, t::F) where {T, F}
  U, V, s = ws.U, ws.V, ws.s
  @inbounds for j in 1:k
    c = t(s[j])
    for i in axes(U, 1)
      U[i, j] *= c
    end
  end
  if k == 0
    fill!(Y, zero(eltype(Y)))
  elseif Y isa StridedMatrix
    mul!(Y, view(U, :, 1:k), transpose(view(V, :, 1:k)))
  else
    B = _array_buffer!(ws.dense, T, size(Y))
    mul!(B, view(U, :, 1:k), transpose(view(V, :, 1:k)))
    copyto!(Y, B)
  end
  Y
end

# After a full SVD, its leading left vectors start the next partial call of f
function _warm_start_svd!(f, U::Matrix{T}, k::Int) where {T}
  cache = f.pws
  cache isa PartialSVDWorkspace{T} || return nothing
  cache.m == size(U, 1) || return nothing
  cache.accepted = false
  w = min(cache.ℓ, size(U, 2))
  _copycols!(cache.Y, 1:w, U, 1:w)
  cache.warm, cache.rank = w, k
  nothing
end

####################################
#  Leading eigenpairs of a symmetric matrix
####################################

mutable struct PartialEigenWorkspace{T<:_BlasReal}
  n::Int
  ℓ::Int                       # the block size
  V::Matrix{T}                 # n × ℓ, an orthonormal block
  Z::Matrix{T}                 # n × ℓ, A V, and the starting block of the next call
  U::Matrix{T}                 # n × ℓ, the Ritz vectors V W
  AU::Matrix{T}                # n × ℓ, A U = Z W
  θ::Vector{T}                 # the Ritz values, in ascending order
  small::SymEigenWorkspace{T}  # the eigendecomposition of the ℓ × ℓ matrix VᵀAV
  qr::QRWorkspace{T}
  dense::Scratch               # a dense copy of an input or output that is not strided
  seed::UInt64
  warm::Int                    # the last columns of Z that start the next call
  rank::Int                    # the number of pairs kept at the last call
  shift::T                     # the shift c of the last call: the block iterates on A + cI
end

function PartialEigenWorkspace(::Type{T}, n::Integer, ℓ::Integer) where {T<:_BlasReal}
  PartialEigenWorkspace{T}(n, ℓ, zeros(T, n, ℓ), zeros(T, n, ℓ), zeros(T, n, ℓ), zeros(T, n, ℓ), zeros(T, ℓ),
                           SymEigenWorkspace(T, ℓ), QRWorkspace(T, n, ℓ), Scratch(), 0x9e3779b97f4a7c15, 0, 0, zero(T))
end

function _partial_eigen_workspace(cache, ::Type{T}, n::Integer, need::Integer) where {T<:_BlasReal}
  ℓ = _partial_block(need)
  4ℓ > n && return nothing
  old = cache isa PartialEigenWorkspace{T} && cache.n == n ? cache : nothing
  old !== nothing && ℓ <= old.ℓ <= 2ℓ && return old
  ws = PartialEigenWorkspace(T, n, ℓ)
  if old !== nothing
    w = min(old.warm, ℓ)
    _copycols!(ws.Z, ℓ-w+1:ℓ, old.Z, old.ℓ-w+1:old.ℓ)
    ws.warm, ws.rank, ws.seed, ws.shift = w, old.rank, old.seed, old.shift
  end
  ws
end

# The leading eigenpairs of the symmetric matrix A (its upper triangle is
# read): the r largest when r > 0, otherwise all the eigenvalues above τ (by
# more than the tolerance δ). Returns the number k of pairs kept, in the last k columns of ws.U with the
# values ascending in ws.θ; -2 when fewer than _PARTIAL_GUARD values of the
# block are below τ, and -1 when the iteration did not converge.
#
# The block iterates on A + cI, with the shift c raised past the most negative
# Ritz value seen when that is larger in magnitude than the boundary, so that
# eigenvalues that are negative but large in magnitude do not crowd out the
# largest ones.
function _partial_eigen!(ws::PartialEigenWorkspace{T}, A::StridedMatrix{T}, τ::T, r::Int) where {T}
  ℓ = ws.ℓ
  V, Z, U, AU, θ = ws.V, ws.Z, ws.U, ws.AU, ws.θ
  w = min(ws.warm, ℓ - _PARTIAL_FRESH)
  _copycols!(V, ℓ-w+1:ℓ, Z, ℓ-w+1:ℓ)
  _randomize!(V, 1:ℓ-w, ws)
  _thin_qr!(V, ws.qr)
  tol = _partial_rtol(T, ws.n)
  minit = w == 0 ? 5 : 1             # a random block takes four power steps before it can be accepted
  c = ws.shift
  for it in 1:2_PARTIAL_MAXITER
    BLAS.symm!('L', 'U', one(T), A, V, zero(T), Z)
    mul!(ws.small.A, transpose(V), Z)
    θR, WR = _eigen!(ws.small)
    copyto!(θ, θR)
    mul!(U, V, WR)
    mul!(AU, Z, WR)
    δ = tol * max(abs(θ[1]), abs(θ[ℓ]))  # values within δ of τ weigh at most δ: not kept
    k = r > 0 ? r : count(>(τ + δ), θ)
    g = min(_PARTIAL_GUARD, ℓ - k)
    if r == 0 && g < _PARTIAL_GUARD
      @. Z = AU + c * U
      ws.warm = ℓ
      return -2
    end
    # negative values smaller in magnitude than the boundary cannot crowd out
    # the pairs kept; otherwise the block must have been made with a shift that
    # covers them (once shifted, the block turns to the largest values, and its
    # smallest Ritz value moves up)
    edge = r > 0 ? (k > 0 ? θ[ℓ-k+1] : zero(T)) : τ + δ
    shifted = -θ[1] < edge || -θ[1] <= c + δ
    accepted = it >= minit && shifted && _eig_accepted(AU, U, θ, k, g, τ + δ, r, δ)
    shifted || (c = -θ[1])
    @. Z = AU + c * U                                      # (A + cI)U, one more power step
    if accepted
      ws.warm, ws.rank, ws.shift = ℓ, k, c                 # the next call keeps the shift
      return k
    end
    copyto!(V, Z)
    _thin_qr!(V, ws.qr)
  end
  ws.warm = ℓ
  return -1
end

# whether the k pairs kept (the last k) have residuals at most bound, and the g
# before them are certified below the boundary: θ + ‖Au - θu‖ ≤ τ, or below the
# smallest pair kept when r > 0
function _eig_accepted(AU::Matrix{T}, U::Matrix{T}, θ::Vector{T}, k::Int, g::Int, τ::T, r::Int, bound::T) where {T}
  ℓ = length(θ)
  ρ = zero(T)
  for c in ℓ:-1:ℓ-k+1
    ρ = _pair_residual(AU, U, θ[c], c)
    ρ <= bound || return false
  end
  edge = r > 0 && k > 0 ? θ[ℓ-k+1] - ρ : τ
  for c in ℓ-k:-1:ℓ-k-g+1
    θ[c] + _pair_residual(AU, U, θ[c], c) <= edge || return false
  end
  return true
end

# Runs _partial_eigen!, growing the block while too few of its values are below
# τ, and caches the workspace in f.pws. Returns the number of pairs kept, or -1
# when the caller falls back to the full eigendecomposition.
function _leading_eigen!(f, X::AbstractMatrix, ::Type{T}, τ::T, r::Int) where {T}
  n = checksquare(X)
  cache = f.pws
  old = cache isa PartialEigenWorkspace{T} ? cache : nothing  # f.pws is untyped: no dynamic call
  prev = old === nothing ? 0 : old.rank
  ws = _partial_eigen_workspace(old, T, n, r > 0 ? r : prev + _PARTIAL_GUARD)
  ws === nothing && return -1
  f.pws = ws
  Xd = _dense_input(ws, X, T)
  while true
    k = _partial_eigen!(ws, Xd, τ, r)
    k == -2 || return k
    bigger = _partial_eigen_workspace(ws, T, n, 2ws.ℓ)
    bigger === nothing && return -1
    f.pws = ws = bigger
  end
end

# Y = Σⱼ f(θⱼ) uⱼuⱼᵀ over the last k pairs of the workspace; returns
# Σⱼ max(f(θⱼ), 0). The Ritz vectors are copied first, since _rebuild_sym!
# destroys its vectors, and the other values become -Inf, so that f gives them
# no weight (f is nondecreasing and f(-Inf) ≤ 0).
function _partial_eigen_out!(Y::AbstractMatrix, ws::PartialEigenWorkspace{T}, k::Int, f::F) where {T, F}
  ℓ, θ = ws.ℓ, ws.θ
  copyto!(ws.AU, ws.U)
  s = zero(T)
  @inbounds for j in 1:ℓ
    if j <= ℓ - k
      θ[j] = T(-Inf)
    else
      s += max(f(θ[j]), zero(T))
    end
  end
  if Y isa StridedMatrix
    _rebuild_sym!(Y, ws.AU, θ, f)
  else
    B = _array_buffer!(ws.dense, T, (ws.n, ws.n))
    _rebuild_sym!(B, ws.AU, θ, f)
    copyto!(Y, B)
  end
  s
end

# After a full eigendecomposition and _rebuild_sym!, which moves the k vectors
# it keeps to the first columns of V in ascending order of their values, the
# largest of them start the next partial call of f
function _warm_start_eigen!(f, V::AbstractMatrix{T}, k::Int) where {T}
  cache = f.pws
  cache isa PartialEigenWorkspace{T} || return nothing
  cache.n == size(V, 1) || return nothing
  ℓ = cache.ℓ
  w = min(ℓ, k)
  _copycols!(cache.Z, ℓ-w+1:ℓ, V, k-w+1:k)
  cache.warm, cache.rank = w, k
  nothing
end

####################################
#  The low-rank functions with partial = true
####################################

# Each returns the value of the proximal operator's result as the full
# computation does, or nothing when it falls back; the caller then runs the
# full decomposition, which starts the next partial call.

# NuclearNorm: Y = U max(Σ - c, 0) Vᵀ; returns Σᵢ max(σᵢ - c, 0)
function _partial_nuclear!(Y::AbstractMatrix, f, X::AbstractMatrix, c::T) where {T}
  k = _leading_svd!(f, X, T, c, 0)
  k < 0 && return nothing
  ws = f.pws::PartialSVDWorkspace{T}
  v = zero(T)
  @inbounds for j in 1:k
    v += ws.s[j] - c
  end
  _partial_svd_out!(Y, ws, k, σ -> σ - c)
  v
end

# IndBallRank: the r largest triplets; returns Y, or nothing to fall back
function _partial_rank!(Y::AbstractMatrix, f, X::AbstractMatrix, ::Type{T}) where {T}
  k = _leading_svd!(f, X, T, zero(T), f.r)
  k < 0 && return nothing
  _partial_svd_out!(Y, f.pws::PartialSVDWorkspace{T}, k, identity)
end

# IndPSD and TracePSD: Y = Σⱼ max(θⱼ - c, 0) uⱼuⱼᵀ over the eigenvalues above
# c, or over the r largest when r > 0; returns Σⱼ max(θⱼ - c, 0)
function _partial_psd!(Y::AbstractMatrix, f, X::AbstractMatrix, c::T, r::Int) where {T}
  k = _leading_eigen!(f, X, T, c, r)
  k < 0 && return nothing
  _partial_eigen_out!(Y, f.pws::PartialEigenWorkspace{T}, k, t -> max(t - c, zero(t)))
end
