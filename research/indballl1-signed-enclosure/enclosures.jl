# Research only. Included into a module containing a pinned ProximalBase.
const BinaryFloat = Union{Float32,Float64}
const Margin = Tuple{Float64,Float64}

@inline function supported_radius(r::T) where {T<:BinaryFloat}
    isfinite(r) && r >= floatmin(T) && !Base.get_zero_subnormals()
end

@inline function two_sum(a, b)
    hi = a + b
    bpart = hi - a
    lo = (a - (hi - bpart)) + (b - bpart)
    hi, lo
end

# Exact addition of the hi and lo words belongs to one or two adjacent floats.
@inline function exact_pair_bounds(a, b)
    hi, lo = two_sum(a, b)
    isfinite(hi) && isfinite(lo) || return nothing
    lower = lo < 0 ? prevfloat(hi) : hi
    upper = lo > 0 ? nextfloat(hi) : hi
    finite_bounds(lower, upper)
end

@inline finite_bounds(lo, hi) = isfinite(lo) && isfinite(hi) ? (lo, hi) : nothing
@inline max_depth(::Type{Float32}) = 1 << 22
@inline max_depth(::Type{Float64}) = 1 << 51

# For depth k and c = k*u <= 1/4, gamma_k/(1-gamma_k) <= 2c.
# magnitude is the computed nonnegative sum serving as an absolute-error scale.
@inline function error_radius(magnitude, depth, ::Type{W}) where {W<:BinaryFloat}
    depth == 0 || magnitude == 0 ? 0.0 : nextfloat((Float64(depth) * eps(W)) * magnitude)
end

@inline function cheap_from_sum(s::W, r, n) where {W<:BinaryFloat}
    k = max(0, n - 1)
    isfinite(s) && k <= max_depth(W) || return nothing
    h = Float64(s)
    e = error_radius(h, k, W)
    e == 0 && return exact_pair_bounds(h, -Float64(r))
    m = h - Float64(r)
    finite_bounds(prevfloat(prevfloat(m) - e), nextfloat(nextfloat(m) + e))
end

# A. One nonnegative reduction, widened exactly for Float32.
function wide_screen(x::Vector{T}, r::T) where {T<:BinaryFloat}
    supported_radius(r) && length(x) - 1 <= max_depth(Float64) || return nothing
    s = ProximalBase._linf_sum(x, ProximalBase._AbsRead{Float64}(), Float64)
    cheap_from_sum(s, r, length(x))
end

@inline straddles(b::Margin, cutoff) = b[1] <= cutoff < b[2]

# Tallying the signed errors by plain summation can lose at most gamma_k
# times their exact absolute mass; q is its computed nonnegative sum.
@inline function residual_bounds(hi, lo, q, depth)
    isfinite(hi) && isfinite(lo) && isfinite(q) || return nothing
    e = error_radius(q, depth, Float64)
    e == 0 && return exact_pair_bounds(hi, lo)
    finite_bounds(prevfloat(hi + prevfloat(lo - e)), nextfloat(hi + nextfloat(lo + e)))
end

# B refinement. Start with the signed radius and retain the addition errors.
function signed_residual(x::Vector{T}, r::T) where {T<:BinaryFloat}
    k = max(0, length(x) - 1)
    supported_radius(r) && k <= max_depth(Float64) || return nothing
    hi, lo, mass = -Float64(r), 0.0, 0.0
    @inbounds for v in x
        hi, e = two_sum(hi, abs(Float64(v)))
        lo += e
        mass += abs(e)
    end
    residual_bounds(hi, lo, mass, k)
end

function screen_then_signed(x::Vector{T}, r::T, cutoff::Float64=0.0) where {T<:BinaryFloat}
    b = wide_screen(x, r)
    b === nothing && return b
    straddles(b, cutoff) ? signed_residual(x, r) : b
end

# C. Neumaier's fixed-order compensated reduction with a certified residual
# mass. Since hi and a are nonnegative, their order is also their magnitude order.
function compensated(x::Vector{T}, r::T) where {T<:BinaryFloat}
    k = max(0, length(x) - 1)
    supported_radius(r) && k <= max_depth(Float64) || return nothing
    hi, lo, mass = 0.0, 0.0, 0.0
    @inbounds for v in x
        a = abs(Float64(v))
        swap = hi < a
        big = ifelse(swap, a, hi)
        small = ifelse(swap, hi, a)
        next = hi + a
        e = small - (next - big)
        hi = next
        lo += e
        mass += abs(e)
    end
    d, e = two_sum(hi, -Float64(r))
    # The first high-word residual was zero. At most n terms reach the low sum.
    residual_bounds(d, lo + e, mass + abs(e), k)
end
