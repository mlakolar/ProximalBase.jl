import re, sys, pathlib
root = pathlib.Path(sys.argv[1])
K_IMG = '''struct _LSImage{TY<:AbstractVecOrMat, TX<:AbstractMatrix, A<:Real, T<:AbstractFloat}
  Y::TY
  X::TX
  α::A                  # the scale of a Scaled loss, one otherwise
end

_LSImage(f::LeastSquaresLoss{T}, α::Real) where {T} = _LSImage{typeof(f.Y), typeof(f.X), typeof(α), T}(f.Y, f.X, α)
_image_model(f::LeastSquaresLoss) = _LSImage(f, true)
_image_model(f::Scaled{<:LeastSquaresLoss}) = _LSImage(f.f, f.α)
_image_buffer(m::_LSImage{TY, TX, A, T}) where {TY, TX, A, T} = zeros(T, size(m.Y))
_image!(z, m::_LSImage, x) = mul!(z, m.X, x)
_value!(x, m, z) = (_image!(z, m, x); convert(_eltype(x), _image_value(m, z)))
_value_and_gradient!(G, x, m, z, d) = convert(_eltype(x), _image_value_and_gradient!(G, m, z, d))
function _image_value(m::_LSImage{TY, TX, A, T}, z) where {TY, TX, A, T}
  Y = m.Y
  v = zero(T)
  @inbounds @simd for i in eachindex(z, Y)
    v += abs2(Y[i] - z[i])
  end
  m.α * (v / 2)
end

function _image_value_and_gradient!(G, m::_LSImage, z, d)
  d .= z .- m.Y
  mul!(G, transpose(m.X), d)
  v = sum(abs2, d) / 2
  G .*= m.α
  m.α * v
end
'''
B_IMG = '''# PB #911 prototype: the value and gradient come from the PB-side oracle (pb_oracle.jl); PS keeps
# the image caches, X for the products and α for the certified bound.
struct _LSImage{F, TY<:AbstractVecOrMat, TX<:AbstractMatrix, A<:Real, T<:AbstractFloat}
  f::F                  # the loss as passed to solve!
  Y::TY
  X::TX
  α::A                  # the scale of a Scaled loss, one otherwise (the certified bound only)
end

_LSImage(f, base::LeastSquaresLoss{T}, α::Real) where {T} = _LSImage{typeof(f), typeof(base.Y), typeof(base.X), typeof(α), T}(f, base.Y, base.X, α)
_image_model(f::LeastSquaresLoss) = _LSImage(f, f, true)
_image_model(f::Scaled{<:LeastSquaresLoss}) = _LSImage(f, f.f, f.α)
_image_buffer(m::_LSImage{F, TY, TX, A, T}) where {F, TY, TX, A, T} = zeros(T, size(m.Y))
_image!(z, m::_LSImage, x) = mul!(z, m.X, x)
_value!(x, m, z) = (_image!(z, m, x); convert(_eltype(x), _image_value(m, z)))
_value_and_gradient!(G, x, m, z, d) = convert(_eltype(x), _image_value_and_gradient!(G, m, z, d))
_image_value(m::_LSImage, z) = PBOracle.value_at_image(m.f, z)
_image_value_and_gradient!(G, m::_LSImage, z, d) = PBOracle.value_and_gradient_at_image!(G, d, m.f, z)
'''
C_IMG = K_IMG.replace('''  Y = m.Y
  v = zero(T)
  @inbounds @simd for i in eachindex(z, Y)
    v += abs2(Y[i] - z[i])
  end
  m.α * (v / 2)''', '''  m.α * PBOracle.half_sumsq_diff(z, m.Y)''').replace('''  v = sum(abs2, d) / 2
  G .*= m.α
  m.α * v''', '''  v = PBOracle.half_sumsq(d)
  m.α === true || (G .*= m.α)
  m.α * v''')
K_CD_HEAD = '''mutable struct LSModel{T<:AbstractFloat, MX<:AbstractMatrix{T}, VY<:AbstractVector{T}} <: CDModel{T}
  X::MX
  y::VY
  s::T
  r::Vector{T}
  a::Vector{T}
end
LSModel(y::AbstractVector{T}, X::AbstractMatrix{T}, s::Real) where T =
  LSModel(X, y, T(s), zeros(T, length(y)), fill(T(NaN), size(X, 2)))
cdmodel(f::LeastSquaresLoss{T, <:AbstractVector}, x) where T = LSModel(f.Y, f.X, one(T))
cdmodel(f::Scaled{<:LeastSquaresLoss{T, <:AbstractVector}}, x) where T = LSModel(f.f.Y, f.f.X, f.α)
'''
B_CD_HEAD = '''mutable struct LSModel{T<:AbstractFloat, MX<:AbstractMatrix{T}, VY<:AbstractVector{T}, F} <: CDModel{T}
  X::MX
  y::VY
  s::T
  r::Vector{T}
  a::Vector{T}
  f::F                 # PB #911 prototype: the loss, for the reported value (PB-side oracle)
end
LSModel(y::AbstractVector{T}, X::AbstractMatrix{T}, s::Real, f) where T =
  LSModel(X, y, T(s), zeros(T, length(y)), fill(T(NaN), size(X, 2)), f)
cdmodel(f::LeastSquaresLoss{T, <:AbstractVector}, x) where T = LSModel(f.Y, f.X, one(T), f)
cdmodel(f::Scaled{<:LeastSquaresLoss{T, <:AbstractVector}}, x) where T = LSModel(f.f.Y, f.f.X, f.α, f)
'''
K_LV = '''function loss_value(m::LSModel{T}, x) where T
  v = zero(T)
  @inbounds @simd for i in eachindex(m.r)
    v += m.r[i]*m.r[i]
  end
  m.s*v/2
end'''
B_LV = '''loss_value(m::LSModel, x) = PBOracle.value_at_residual(m.f, m.r)'''
C_LV = '''loss_value(m::LSModel{T}, x) where T = m.s*PBOracle.half_sumsq(m.r)'''
MAIN_ANCHOR = '''  value_and_prox!
'''
def sub(path, old, new):
    s = path.read_text()
    assert s.count(old) == 1, (path, old[:60])
    path.write_text(s.replace(old, new))
for v in ['B1', 'B2', 'B3', 'C']:
    src = root / v / 'src'
    sub(src / 'ProximalSolvers.jl', MAIN_ANCHOR, MAIN_ANCHOR + 'include("pb_oracle.jl")\n')
    if v == 'C':
        sub(src / 'solvers/image_models.jl', K_IMG, C_IMG)
        sub(src / 'cd/leastsquares.jl', K_LV, C_LV)
    else:
        sub(src / 'solvers/image_models.jl', K_IMG, B_IMG)
        sub(src / 'cd/leastsquares.jl', K_CD_HEAD, B_CD_HEAD)
        sub(src / 'cd/leastsquares.jl', K_LV, B_LV)
print("ok")
