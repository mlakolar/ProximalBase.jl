####################################
#
#  LAPACK workspaces
#
####################################

# Julia's LAPACK wrappers allocate their work arrays on every call. The
# workspaces below query the sizes once and keep the arrays, so that the
# spectral functions (eigenvalue and singular value decompositions of a matrix
# of a fixed size) allocate nothing in their hot calls.

const _BlasReal = Union{Float32, Float64}

# the functions built on LAPACK take Float32 and Float64 entries only
_checkblas(f, ::Type{<:_BlasReal}) = true
@noinline _checkblas(f, ::Type{T}) where {T} =
  throw(ArgumentError("$(nameof(typeof(f))) needs Float32 or Float64 entries, got $T"))

####################################
#  Symmetric eigendecomposition, all eigenpairs (divide and conquer, syevd)
####################################

mutable struct SymEigenWorkspace{T<:_BlasReal}
  n::Int
  A::Matrix{T}                 # the input, overwritten with the eigenvectors
  w::Vector{T}                 # the eigenvalues, in ascending order
  work::Vector{T}
  iwork::Vector{_BlasInt}
  info::Base.RefValue{_BlasInt}
end

for (syevd, T) in ((:dsyevd_, :Float64), (:ssyevd_, :Float32))
  @eval function _syevd!(jobz::Char, A::Matrix{$T}, w::Vector{$T}, work::Vector{$T}, lwork::Integer,
                         iwork::Vector{_BlasInt}, liwork::Integer, info::Base.RefValue{_BlasInt})
    n = size(A, 1)
    ccall(($(QuoteNode(_blasfunc(syevd))), _libblastrampoline), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{_BlasInt}, Ptr{$T}, Ref{_BlasInt}, Ptr{$T},
           Ptr{$T}, Ref{_BlasInt}, Ptr{_BlasInt}, Ref{_BlasInt}, Ptr{_BlasInt}, Clong, Clong),
          jobz, 'U', n, A, max(1, n), w, work, lwork, iwork, liwork, info, 1, 1)
    info[]
  end
end

function SymEigenWorkspace(::Type{T}, n::Integer) where {T<:_BlasReal}
  A, w = zeros(T, n, n), zeros(T, n)
  work, iwork, info = zeros(T, 1), zeros(_BlasInt, 1), Ref{_BlasInt}(0)
  _syevd!('V', A, w, work, -1, iwork, -1, info)       # workspace query
  lwork, liwork = max(1, round(Int, work[1])), max(1, Int(iwork[1]))
  SymEigenWorkspace{T}(n, A, w, resize!(work, lwork), resize!(iwork, liwork), info)
end

# Eigendecomposition of the symmetric matrix in ws.A (its upper triangle is
# read), in place: returns the eigenvalues in ascending order and the matrix of
# eigenvectors, both owned by ws.
function _eigen!(ws::SymEigenWorkspace)
  info = _syevd!('V', ws.A, ws.w, ws.work, length(ws.work), ws.iwork, length(ws.iwork), ws.info)
  info == 0 || throw(LAPACKException(info))
  ws.w, ws.A
end

# the eigenvalues only, in ascending order, destroying ws.A
function _eigvals!(ws::SymEigenWorkspace)
  info = _syevd!('N', ws.A, ws.w, ws.work, length(ws.work), ws.iwork, length(ws.iwork), ws.info)
  info == 0 || throw(LAPACKException(info))
  ws.w
end

# Workspaces that a function object caches in an untyped field, for the size and
# element type of its input: the hot calls find them in place, and the type
# checks keep the code that uses them type stable.
function _svd_workspace(cache, ::Type{T}, m::Integer, n::Integer) where {T<:_BlasReal}
  if cache isa SVDWorkspace{T}
    cache.m == m && cache.n == n && return cache
  end
  SVDWorkspace(T, m, n)
end
function _eig_workspace(cache, ::Type{T}, n::Integer) where {T<:_BlasReal}
  if cache isa SymEigenWorkspace{T}
    cache.n == n && return cache
  end
  SymEigenWorkspace(T, n)
end
function _eigtop_workspace(cache, ::Type{T}, n::Integer, r::Integer) where {T<:_BlasReal}
  if cache isa SymEigenTopWorkspace{T}
    cache.n == n && cache.r == r && return cache
  end
  SymEigenTopWorkspace(T, n, r)
end

####################################
#  Symmetric eigendecomposition, the largest r eigenpairs (MRRR, syevr)
####################################

mutable struct SymEigenTopWorkspace{T<:_BlasReal}
  n::Int
  r::Int
  A::Matrix{T}                 # the input, destroyed
  w::Vector{T}                 # w[1:r] are the r largest eigenvalues, ascending
  Z::Matrix{T}                 # their eigenvectors
  isuppz::Vector{_BlasInt}
  work::Vector{T}
  iwork::Vector{_BlasInt}
  m::Base.RefValue{_BlasInt}
  info::Base.RefValue{_BlasInt}
end

for (syevr, T) in ((:dsyevr_, :Float64), (:ssyevr_, :Float32))
  @eval function _syevr!(A::Matrix{$T}, il::Integer, iu::Integer, m::Base.RefValue{_BlasInt}, w::Vector{$T},
                         Z::Matrix{$T}, isuppz::Vector{_BlasInt}, work::Vector{$T}, lwork::Integer,
                         iwork::Vector{_BlasInt}, liwork::Integer, info::Base.RefValue{_BlasInt})
    n = size(A, 1)
    ccall(($(QuoteNode(_blasfunc(syevr))), _libblastrampoline), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{_BlasInt}, Ptr{$T}, Ref{_BlasInt},
           Ref{$T}, Ref{$T}, Ref{_BlasInt}, Ref{_BlasInt}, Ref{$T}, Ptr{_BlasInt}, Ptr{$T},
           Ptr{$T}, Ref{_BlasInt}, Ptr{_BlasInt}, Ptr{$T}, Ref{_BlasInt}, Ptr{_BlasInt}, Ref{_BlasInt},
           Ptr{_BlasInt}, Clong, Clong, Clong),
          'V', 'I', 'U', n, A, max(1, n), zero($T), zero($T), il, iu, -one($T), m, w,
          Z, max(1, n), isuppz, work, lwork, iwork, liwork, info, 1, 1, 1)
    info[]
  end
end

function SymEigenTopWorkspace(::Type{T}, n::Integer, r::Integer) where {T<:_BlasReal}
  1 <= r <= n || throw(ArgumentError("need 1 ≤ r ≤ n, got r = $r and n = $n"))
  A, w, Z = zeros(T, n, n), zeros(T, n), zeros(T, n, r)
  isuppz, m, info = zeros(_BlasInt, 2r), Ref{_BlasInt}(0), Ref{_BlasInt}(0)
  work, iwork = zeros(T, 1), zeros(_BlasInt, 1)
  _syevr!(A, n - r + 1, n, m, w, Z, isuppz, work, -1, iwork, -1, info)    # workspace query
  lwork, liwork = max(1, round(Int, work[1])), max(1, Int(iwork[1]))
  SymEigenTopWorkspace{T}(n, r, A, w, Z, isuppz, resize!(work, lwork), resize!(iwork, liwork), m, info)
end

# the r largest eigenpairs of the symmetric matrix in ws.A (upper triangle),
# destroying ws.A: returns views of the eigenvalues, ascending, and eigenvectors
function _eigen_top!(ws::SymEigenTopWorkspace)
  info = _syevr!(ws.A, ws.n - ws.r + 1, ws.n, ws.m, ws.w, ws.Z, ws.isuppz,
                 ws.work, length(ws.work), ws.iwork, length(ws.iwork), ws.info)
  info == 0 || throw(LAPACKException(info))
  view(ws.w, 1:ws.r), ws.Z
end

####################################
#  Thin singular value decomposition (divide and conquer, gesdd)
####################################

mutable struct SVDWorkspace{T<:_BlasReal}
  m::Int
  n::Int
  A::Matrix{T}                 # the input, destroyed
  s::Vector{T}                 # singular values, in descending order
  U::Matrix{T}                 # m × min(m, n); 0 × 0 until the first call with vectors
  Vt::Matrix{T}                # min(m, n) × n; 0 × 0 until the first call with vectors
  work::Vector{T}
  iwork::Vector{_BlasInt}
  info::Base.RefValue{_BlasInt}
end

for (gesdd, T) in ((:dgesdd_, :Float64), (:sgesdd_, :Float32))
  @eval function _gesdd!(jobz::Char, A::Matrix{$T}, s::Vector{$T}, U::Matrix{$T}, Vt::Matrix{$T},
                         work::Vector{$T}, lwork::Integer, iwork::Vector{_BlasInt}, info::Base.RefValue{_BlasInt})
    m, n = size(A)
    ccall(($(QuoteNode(_blasfunc(gesdd))), _libblastrampoline), Cvoid,
          (Ref{UInt8}, Ref{_BlasInt}, Ref{_BlasInt}, Ptr{$T}, Ref{_BlasInt}, Ptr{$T},
           Ptr{$T}, Ref{_BlasInt}, Ptr{$T}, Ref{_BlasInt}, Ptr{$T}, Ref{_BlasInt},
           Ptr{_BlasInt}, Ptr{_BlasInt}, Clong),
          jobz, m, n, A, max(1, m), s, U, max(1, m), Vt, max(1, min(m, n)), work, lwork, iwork, info, 1)
    info[]
  end
end

# The vectors and the larger workspace they need are allocated on the first
# call with vectors, so a workspace used only for values stays small.
function SVDWorkspace(::Type{T}, m::Integer, n::Integer) where {T<:_BlasReal}
  k = min(m, n)
  A, s, U, Vt = zeros(T, m, n), zeros(T, k), zeros(T, 0, 0), zeros(T, 0, 0)
  work, iwork, info = zeros(T, 1), zeros(_BlasInt, max(1, 8k)), Ref{_BlasInt}(0)
  _gesdd!('N', A, s, U, Vt, work, -1, iwork, info)        # workspace query
  SVDWorkspace{T}(m, n, A, s, U, Vt, resize!(work, max(1, round(Int, work[1]))), iwork, info)
end

# allocates U and Vt and grows the workspace for the vectors, once
function _svd_vectors!(ws::SVDWorkspace{T}) where {T}
  k = min(ws.m, ws.n)
  (size(ws.U) == (ws.m, k) && size(ws.Vt) == (k, ws.n)) && return ws
  ws.U, ws.Vt = zeros(T, ws.m, k), zeros(T, k, ws.n)
  work = zeros(T, 1)
  _gesdd!('S', ws.A, ws.s, ws.U, ws.Vt, work, -1, ws.iwork, ws.info)    # workspace query
  length(ws.work) < round(Int, work[1]) && (ws.work = zeros(T, round(Int, work[1])))
  ws
end

# The thin SVD of the matrix in ws.A, destroying it: returns s, U and Vt, owned
# by ws. With vectors = false only s is computed, and U and Vt are not touched.
function _svd!(ws::SVDWorkspace; vectors::Bool = true)
  vectors && _svd_vectors!(ws)
  info = _gesdd!(vectors ? 'S' : 'N', ws.A, ws.s, ws.U, ws.Vt, ws.work, length(ws.work), ws.iwork, ws.info)
  info == 0 || throw(LAPACKException(info))
  ws.s, ws.U, ws.Vt
end

####################################
#  Rebuilding a symmetric matrix from scaled eigenvectors
####################################

# Only device inputs need a host copy before entrywise access.
_is_device_array(::AbstractArray) = false

# whether BLAS can read or write Y in place: a StridedMatrix such as
# view(A, 1:2:n, :) has columns that are not contiguous, which Julia's syrk and
# symm reject and its syr, which does not check, writes past
_blas_ready(Y::AbstractMatrix) = Y isa StridedMatrix && stride(Y, 1) == 1

# Y = Σⱼ μⱼ vⱼvⱼ' for the eigenvectors vⱼ in the columns of V and the
# nonnegative weights μⱼ = f(λⱼ), λⱼ = λ[j]: moves the columns with μⱼ > 0 to
# the front of V, scaled by √μⱼ, and forms the product with syrk, in O(n²k) for
# k such columns. Destroys V. Returns Σⱼ log(μⱼ) when logs = true.
function _rebuild_sym!(Y::StridedMatrix{T}, V::StridedMatrix{T}, λ::AbstractVector{T}, f::F;
                       logs::Bool = false) where {T<:_BlasReal, F}
  n = size(V, 1)
  size(Y) == (n, n) || throw(DimensionMismatch("the output must be $n × $n"))
  k = 0
  s = zero(T)
  @inbounds for j in eachindex(λ)
    μ = f(λ[j])
    μ > 0 || continue
    logs && (s += log(μ))
    k += 1
    c = sqrt(μ)
    @simd for i in 1:n
      V[i, k] = c * V[i, j]
    end
  end
  if k == 0
    fill!(Y, zero(T))
  else
    BLAS.syrk!('U', 'N', one(T), view(V, :, 1:k), zero(T), Y)
    _copy_upper_to_lower!(Y)
  end
  s
end

function _copy_upper_to_lower!(Y::AbstractMatrix)
  n = size(Y, 1)
  @inbounds for j in 1:n, i in j+1:n
    Y[i, j] = Y[j, i]
  end
  Y
end
