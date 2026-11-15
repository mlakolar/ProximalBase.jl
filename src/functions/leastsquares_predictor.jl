####################################
#
#  Least-squares oracles at a predictor or a residual supplied by the caller
#
####################################

# The domain: a vector-response LeastSquaresLoss with strided Float32 or Float64 data, or one Scaled of it.
# Any other loss or array is a MethodError (a nested Scaled, a matrix Y, a non-strided array, mixed types).
const _LSStrided{T} = LeastSquaresLoss{T, <:StridedVector{T}, <:StridedMatrix{T}}
const _LSScaledStrided{T} = Scaled{<:_LSStrided{T}}
const _LSOracleLoss{T} = Union{_LSStrided{T}, _LSScaledStrided{T}}

# The loss and the scale a of the computation: one(T) for a LeastSquaresLoss, and for a Scaled the factor
# as supplied when it is a positive finite Float32 or Float64 (not rounded to T), or the Float64 of an
# integer α with Float64(α) == α. The type of a is determined by the type of the factor. Any other
# factor is refused here, before the callers write.
@inline _ls_base_scale(f::LeastSquaresLoss{T}) where {T} = (f, one(T))
@inline _ls_base_scale(s::Scaled) = (s.f, _ls_scale(s.α))
@inline _ls_scale(α::Union{Float32, Float64}) = isfinite(α) & (α > 0) ? α : _ls_scale_error(α)
@inline function _ls_scale(α::Integer)
  a = Float64(α)
  isfinite(a) && a > 0 && a == α ? a : _ls_scale_error(α)
end
@inline _ls_scale(α) = _ls_scale_error(α)
# The same for the value oracles, which write nothing, without the check of a Float32 or Float64 factor on the
# ordinary path: the Scaled constructor refuses a negative or NaN factor and gives no Scaled for a zero one, so
# the factor is positive, finite or Inf. An infinite factor makes α s infinite or NaN, which _ls_square_value
# sends to _ls_value_exceptional, which checks the factor (_ls_base_scale) and raises the ArgumentError.
@inline _ls_base_factor(f::LeastSquaresLoss{T}) where {T} = (f, one(T))
@inline _ls_base_factor(s::Scaled{<:AbstractFunction, <:Union{Float32, Float64}}) = (s.f, s.α)
@inline _ls_base_factor(s::Scaled) = _ls_base_scale(s)
# Whether the product p = α s of the ordinary path is below floatmax(T) for data of type T (_ls_square_value), p in the
# promoted type of α and T: for a loss α = 1 and p = s, which _ls_ordinary_sum keeps at most floatmax(T).
@inline _ls_product_fits(::LeastSquaresLoss, p, ::Type{T}) where {T} = true
@inline _ls_product_fits(::Scaled, p, ::Type{T}) where {T} = p < floatmax(T)
@noinline _ls_scale_error(α) = throw(ArgumentError(
  "the scale $α of type $(typeof(α)) is not supported by the least-squares predictor and residual oracles: " *
  "a positive finite Float32 or Float64, or an integer that Float64 represents exactly, is required"))

@noinline _ls_length_error(name, got, expected) =
  throw(DimensionMismatch("$name has length $got, expected $expected"))
@inline _ls_checklength(name, v, n) = length(v) == n || _ls_length_error(name, length(v), n)
@noinline _ls_offset_error() = throw(ArgumentError("the outputs g and d must have axes starting at 1"))

# Whether two arrays might share memory, by Base.mightalias; an empty array shares none.
@inline _ls_overlap(a, b) = !isempty(a) && !isempty(b) && Base.mightalias(a, b)
@noinline _ls_alias_error(out, other) =
  throw(ArgumentError("the output $out might share memory with $other (Base.mightalias)"))
# The outputs d and g against each other and against the arrays that the call reads or owns, before any write
function _ls_checkalias(g, d, f::LeastSquaresLoss, z)
  X, Y, t = f.X, f.Y, f.tmp
  _ls_overlap(d, g) && _ls_alias_error("d", "the output g")
  _ls_overlap(d, X) && _ls_alias_error("d", "the design X")
  _ls_overlap(d, Y) && _ls_alias_error("d", "the response Y")
  _ls_overlap(d, z) && _ls_alias_error("d", "the predictor z")
  _ls_overlap(d, t) && _ls_alias_error("d", "the scratch of the loss")
  _ls_overlap(g, X) && _ls_alias_error("g", "the design X")
  _ls_overlap(g, Y) && _ls_alias_error("g", "the response Y")
  _ls_overlap(g, z) && _ls_alias_error("g", "the predictor z")
  _ls_overlap(g, t) && _ls_alias_error("g", "the scratch of the loss")
  nothing
end

# α times the gradient, applied last; none for a LeastSquaresLoss
@inline _ls_scale_gradient!(g, ::LeastSquaresLoss, a) = g
@inline function _ls_scale_gradient!(g, ::Scaled, a)
  g .*= a
  g
end

"""
    ProximalBase.value_from_predictor(f, z)

The value `α ‖z - Y‖² / 2` of the least-squares loss `f` at the linear predictor `z = Xβ` computed by the
caller, where `α = 1` for a `LeastSquaresLoss(Y, X)` and `α` is the factor of `Scaled(LeastSquaresLoss(Y, X), α)`.
No entry of `X` is read and the scratch `f.tmp` is neither read nor written, so `z` may be a view of it.

The domain is a vector-response `LeastSquaresLoss{T}` with strided `Float32` or `Float64` `Y` and `X`, or one
`Scaled` of it, and `z::StridedVector{T}` with the axes of `Y`. Another argument type is a `MethodError`; other
axes (another length, or an offset view of the same length) are a `DimensionMismatch`.

A `Scaled` factor is used as supplied, not rounded to `T`: a positive finite `Float32` or `Float64`, or an integer
`α` with `Float64(α) == α`, which is converted to `Float64` once. Any other factor (`Inf`, an integer that
`Float64` does not hold, a `Float16`, `BigFloat` or `Rational`) is an `ArgumentError`. The factor as stored is
what [`ProximalBase.leastsquares_data`](@ref) returns.

The result has type `T`. The sum of squares `s` is formed in `T` and `α s / 2` in the promoted type of `α` and
`T`, then rounded to `T`. Where the squares overflow or underflow, where `z - Y` overflows, and where the ordinary
value `α s / 2` is above `floatmax(T)/2`, the value is recomputed from the squares in units of a power of two with `α` folded
in, summed compensated. For finite `z` and `Y` and a value `v` representable with a margin, the result is within
`(n + 8) eps(T) |v|` of `v`, plus `2 nextfloat(zero(T))` where `|v| < floatmin(T)`. A non-finite entry of `z` or
`Y` gives a non-finite value: unlike `f(β)`, the call does not form `Xβ` again. The behavior near the overflow
threshold, and the difference from `Scaled(f, α)(β)` on the exceptional range (issue #609), are stated in the
section "Least squares at a predictor or residual that the caller holds" of the guide to the smooth functions.

For a raw loss, or a `Scaled` one with a `Float32`, `Float64` or `Int` factor, the call allocates nothing on its
ordinary path. It writes no field of `f`, so several tasks may call it on one `f` while `X` and `Y` are not
mutated.
"""
function value_from_predictor end

"""
    ProximalBase.value_and_gradient_from_predictor!(g, d, f, z)

Writes the residual `d = z - Y` and the gradient `g = α fl(Xᵀd)` of the least-squares loss `f` at the linear
predictor `z = Xβ` computed by the caller, and returns the value `α ‖d‖² / 2`. The call forms one product with
`Xᵀ` and none with `X`, and neither reads nor writes the scratch `f.tmp` (`z` may be a view of it).

The domain, the factor `α`, the result type and the value's rounding and recomputation are those of
[`ProximalBase.value_from_predictor`](@ref), with `g`, `d` and `z` of type `StridedVector{T}`: `z` and `d` have the
axes of `Y` and length `size(X, 1)`, and `g` has length `size(X, 2)`.

The gradient is `α · fl(Xᵀd)`: an `Xᵀd` that overflows or underflows before the scaling is not recovered, even
where `α` would bring the exact gradient back into range. A difference `z - Y` that overflows with finite
entries gives an infinite entry of `d`, a non-finite value and, when `size(X, 2) > 0`, non-finite gradient
entries; `value_from_predictor` recovers the value in that case. The value sums the squares of `d`, and
`value_from_predictor` those of the differences `z - Y`; the two are not promised to agree in the last bits.

Checks before any write: lengths and axes (`DimensionMismatch`), outputs `g` and `d` whose first index is 1, the
factor, and the overlap of the outputs (`ArgumentError`). `d` must not overlap `g`, `X`, `Y`, `z` or `f.tmp`, and `g`
must not overlap `X`, `Y`, `z` or `f.tmp`, as `Base.mightalias` decides for nonempty arrays; memory shared in ways
that `mightalias` does not detect is a precondition on the caller. On any error `g`, `d` and `f.tmp` are unchanged.

The call writes no field of `f`, so tasks that use distinct `g` and `d` may call it on one `f` while `X` and `Y`
are not mutated. For a raw loss, or a `Scaled` one with a `Float32`, `Float64` or `Int` factor, it allocates
nothing on its ordinary path.
"""
function value_and_gradient_from_predictor! end

"""
    ProximalBase.value_from_residual(f, r)

The value `α ‖r‖² / 2` of the least-squares loss `f` at the residual `r = Xβ - Y`, or `r = Y - Xβ`, computed by
the caller; `α = 1` for a `LeastSquaresLoss(Y, X)` and `α` is the factor of a `Scaled` one. No entry of `X` or `Y`
is read and `f.tmp` is neither read nor written, so `r` may be a view of it.

The domain, the factor, the result type, the rounding, the recomputation and the accuracy are those of
[`ProximalBase.value_from_predictor`](@ref), with `r::StridedVector{T}` with the axes of `Y` (other axes are a
`DimensionMismatch`) and a non-finite entry of `r` in place of one of `z`. The call allocates nothing on its
ordinary path in the same cases, and it is reentrant as `value_from_predictor` is.
"""
function value_from_residual end

"""
    ProximalBase.leastsquares_data(f)

The data of a least-squares loss as `(; Y, X, α)`: `f.Y` and `f.X` themselves (`===`, neither copied
nor converted), and the scale `α`. `α` is `one(T)` for a `LeastSquaresLoss{T}`, and the stored factor
`α` of one `Scaled` of a `LeastSquaresLoss`, as it was supplied, with its type and without a
check or conversion: the scale that [`ProximalBase.value_from_predictor`](@ref) and its companions
compute with can differ from it, and they refuse a factor outside their domain. Any
`LeastSquaresLoss` is accepted here, also with array types outside the strided domain; any other function is a
`MethodError`.

The arrays are borrowed, not copied and not protected: do not mutate `Y` or `X` while the loss or
one of its oracles is in use, also on another task.
"""
function leastsquares_data end

# The axes of z are checked against those of Y (whose length is size(X, 1) by the constructor of the loss) by
# _ls_sumabs2_diff, which throws a DimensionMismatch before it reads an entry; the call reads no entry or size of X.
# value_from_residual checks the axes of r against those of Y in the same way.
function value_from_predictor(f::_LSOracleLoss{T}, z::StridedVector{T}) where {T<:Union{Float32, Float64}}
  _ls_value(f, z, _ls_base_factor(f)[1].Y)
end

function value_and_gradient_from_predictor!(g::StridedVector{T}, d::StridedVector{T}, f::_LSOracleLoss{T},
    z::StridedVector{T}) where {T<:Union{Float32, Float64}}
  base, a = _ls_base_scale(f)
  n, p = size(base.X)
  _ls_checklength(:z, z, n)
  _ls_checklength(:d, d, n)
  _ls_checklength(:g, g, p)
  Y = base.Y
  axes(z) == axes(Y) || _ls_axes_error("z", axes(z, 1), axes(Y, 1))
  axes(d) == axes(Y) || _ls_axes_error("d", axes(d, 1), axes(Y, 1))
  firstindex(g) == firstindex(d) == 1 || _ls_offset_error()   # the product with Xᵀ (BLAS) refuses offset axes
  _ls_checkalias(g, d, base, z)
  v = _leastsquares_residual_gradient!(g, d, base.X, z, base.Y, f)
  _ls_scale_gradient!(g, f, a)
  v
end

function value_from_residual(f::_LSOracleLoss{T}, r::StridedVector{T}) where {T<:Union{Float32, Float64}}
  Y = _ls_base_factor(f)[1].Y
  axes(r) == axes(Y) || _ls_axes_error("r", axes(r, 1), axes(Y, 1))
  _ls_value(f, r, nothing)
end

leastsquares_data(f::LeastSquaresLoss{T}) where {T} = (; Y = f.Y, X = f.X, α = one(T))
leastsquares_data(s::Scaled{<:LeastSquaresLoss}) = (; Y = s.f.Y, X = s.f.X, α = s.α)
