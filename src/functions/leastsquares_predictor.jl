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
@noinline _ls_scale_error(α) = throw(ArgumentError(
  "the scale $α of type $(typeof(α)) is not supported by the least-squares predictor and residual oracles: " *
  "a positive finite Float32 or Float64, or an integer that Float64 represents exactly, is required"))

@noinline _ls_length_error(name, got, expected) =
  throw(DimensionMismatch("$name has length $got, expected $expected"))
@inline _ls_checklength(name, v, n) = length(v) == n || _ls_length_error(name, length(v), n)

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

The value `α ‖z - Y‖² / 2` of the least-squares loss `f` at the linear predictor `z = Xβ`
computed by the caller, where `α = 1` for a `LeastSquaresLoss(Y, X)` and `α` is the factor of
`f = Scaled(LeastSquaresLoss(Y, X), α)`. The entries of `X` are not read, no product is formed and the
scratch `f.tmp` is neither read nor written, so `z` may be a view of it.

The domain is a vector-response `LeastSquaresLoss{T}` with strided `Float32` or `Float64` `Y` and
`X` (`T` their type), or one `Scaled` of it, and `z::StridedVector{T}` of length `size(X, 1)`; any
other argument is a `MethodError`, and a wrong length is a `DimensionMismatch`.

**The scale.** A `Scaled` factor `α` is used as supplied, not rounded to `T` before the product: a
positive finite `Float32` or `Float64` (including a `Float64` factor with `Float32` data), or an
integer `α` with `Float64(α) == α` (for example `16777217` with `Float32` data), which is
converted to `Float64` once. The factor as stored, `f.α`, is what
[`ProximalBase.leastsquares_data`](@ref) returns; the factor of the computation is a `Float32` or
`Float64` and is the integer's `Float64` value for an integer. Any other factor (an integer that `Float64` does not hold,
a `Float16`, `BigFloat` or `Rational`, `Inf`) is an `ArgumentError`, raised before the
computation. The `Scaled` constructor itself is not changed.

**The result** has type `T`. The product `α s / 2` of the sum of squares `s` is formed in the
promoted type of `α` and `T` and rounded to `T` at the result (a subnormal result can round twice).
Where the squares overflow or underflow, or the difference `z - Y` overflows, the value is
recomputed from the squares in units of a power of two with `α` folded in, as for `SqrNormL2`, which
gives a finite value where the value is representable with margin (see the range statement below). For finite `z` and `Y` and a value `v` that is representable
with a margin, the tests check the result against 256-bit references within `(n + 8) eps(T) |v|`,
plus `2 nextfloat(zero(T))` where `|v| < floatmin(T)`.

**Range.** Let `u = floatmax(T) - prevfloat(floatmax(T))`, so that `floatmax(T) + u/2` is the
round-to-nearest overflow threshold, and let `w = (n + 8) eps(T)` be the relative error bound above. For
finite `z` and `Y` and an exact value `v` with `|v| (1 + w) ≤ floatmax(T)`, the result is finite and within
that bound. For `v` above `(floatmax(T) + u/2) (1 + w)` the result is `Inf`. In between, within the relative
width `w` of the threshold, the recomputation that these oracles share with `SqrNormL2` does not decide on which
side the exact value lies: the result can be `floatmax(T)` or `Inf`, also where `v` rounds to the other, and a
finite result there is within `w |v|` of `v`. Correct overflow classification in that region is not
guaranteed (the same kind of boundary uncertainty as for the `NormL2` recomputation, see the numerics
page of the documentation), and no materially wrong finite value arises away from it. An exact positive value
below the smallest subnormal can round to `nextfloat(zero(T))` rather than to zero (`Float64`, `α = 1.5`,
`z = [2.0^-537]`, `Y = [0.0]`: exact `0.75 nextfloat(0.0)`, result `nextfloat(0.0)`). A
non-finite entry of `z` or of `Y` gives a non-finite value, `Inf` or `NaN`: the call has `z`, not `β`,
and does not form `Xβ` again to recover from it. By contrast `f(β)` and `value_and_gradient!` recompute
`Xβ` from `β` where it overflows although `β`, `X` and `Y` are finite.

For a `Scaled` loss the squares are recomputed with `α` folded in also where the inner value
`‖z - Y‖²/2` leaves the range of `T` and `α` brings it back. `Scaled(f, α)(β)` forms the inner value
first and can give an infinite or an inaccurate value there (issue #609).

For a raw loss, or a `Scaled` one with a `Float32`, `Float64` or `Int` scale, the call allocates nothing on its ordinary
path. It is reentrant: it writes no field of `f`, so several tasks may call it on one `f` provided
that `X` and `Y` are not mutated meanwhile.
"""
function value_from_predictor end

"""
    ProximalBase.value_and_gradient_from_predictor!(g, d, f, z)

Writes the residual `d = z - Y` and the gradient `g = α fl(Xᵀd)` of the least-squares loss `f` at the
linear predictor `z = Xβ` computed by the caller, and returns the value `α ‖d‖² / 2`. Here `α = 1`
for a `LeastSquaresLoss(Y, X)` and `α` is the factor of `Scaled(LeastSquaresLoss(Y, X), α)`. The call
forms one product with `Xᵀ` and none with `X`, and does not read or write the scratch `f.tmp`.

The domain, the scale `α` as stored and as computed, the type of the result and its rounding are those
of [`ProximalBase.value_from_predictor`](@ref); `g`, `d` and `z` are `StridedVector{T}`, `z` and `d` of
length `size(X, 1)` and `g` of length `size(X, 2)`. The sum of squares is formed over `d` in one
vectorizing loop, as for the value of `value_and_gradient!`, while `value_from_predictor` forms it over the
differences `z - Y`; the two loops are not promised to give the same last bits.

**The gradient order** is `α · fl(Xᵀd)`: the product `Xᵀd` is rounded to `T` and then multiplied by `α`
(not for a `LeastSquaresLoss`, where `α = 1`). An `Xᵀd` that overflows to `Inf` or underflows to
zero before the scaling is not recovered, even where `α` would bring the exact gradient back into
range. A difference `z - Y` that overflows with finite entries gives an infinite entry of `d` and a
non-finite value; when `size(X, 2) > 0` the entries of `g` are non-finite (with the BLAS of the tests, also
the entries of a zero column of `X`, where `0 · Inf` is `NaN`), and `g` is empty when `size(X, 2) == 0`.
`value_from_predictor` recovers the value in that case.

**Checks before any write.** Lengths (`DimensionMismatch`), the scale (`ArgumentError`) and the
overlap of the outputs (`ArgumentError`): `d` must not overlap `g`, `X`, `Y`, `z` or `f.tmp`, and `g`
must not overlap `X`, `Y`, `z` or `f.tmp`. On any error `g`, `d` and `f.tmp` are unchanged. The overlap
test is `Base.mightalias`, applied to nonempty arrays (an empty array is never counted as overlapping, so an
empty view of its parent is accepted). For nonempty arrays it refuses two views of one array whose indices
overlap, and in the tested cases accepts two views of one array whose index ranges do not (separate ranges,
separate columns). It refuses a view against its parent, and against a `vec` or `reshape` of the same memory,
also where the two are disjoint, and not every disjoint layout is accepted. Memory shared in ways that `mightalias` does not detect
(`unsafe_wrap`, an array type without `Base.dataids`) is a precondition on the caller. `z` may be a view
of `f.tmp`: it is read, and `f.tmp` is not written.

The call writes no field of `f`, so tasks that use distinct `g` and `d` may call it on one `f`
while `X` and `Y` are not mutated. For a raw loss, or a `Scaled` one with a `Float32`, `Float64` or `Int` scale, it
allocates nothing on its ordinary path.
"""
function value_and_gradient_from_predictor! end

"""
    ProximalBase.value_from_residual(f, r)

The value `α ‖r‖² / 2` of the least-squares loss `f` at the residual `r = Xβ - Y`, or `r = Y - Xβ`
(the sign does not matter), computed by the caller; `α = 1` for a `LeastSquaresLoss(Y, X)` and `α` is
the factor of a `Scaled` one. Neither `X` nor `Y` is read, no product is formed and `f.tmp` is
neither read nor written, so `r` may be a view of it.

The domain, the scale `α`, the type of the result and its rounding, the recomputation where the
squares overflow or underflow, the range statement (the region around the overflow threshold and the
subnormal rounding included) and a non-finite entry (here of `r`) are those of
[`ProximalBase.value_from_predictor`](@ref), with `r::StridedVector{T}` of length `size(X, 1)`. The
sum of squares is formed over `r` in one vectorizing loop, as for the value of `value_and_gradient!`.

For a raw loss, or a `Scaled` one with a `Float32`, `Float64` or `Int` scale, the call allocates nothing on its ordinary
path, and it is reentrant as `value_from_predictor` is.
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

function value_from_predictor(f::_LSOracleLoss{T}, z::StridedVector{T}) where {T<:Union{Float32, Float64}}
  base, a = _ls_base_scale(f)
  _ls_checklength(:z, z, size(base.X, 1))
  _square_value_wide(a, z, base.Y)
end

function value_and_gradient_from_predictor!(g::StridedVector{T}, d::StridedVector{T}, f::_LSOracleLoss{T},
    z::StridedVector{T}) where {T<:Union{Float32, Float64}}
  base, a = _ls_base_scale(f)
  n, p = size(base.X)
  _ls_checklength(:z, z, n)
  _ls_checklength(:d, d, n)
  _ls_checklength(:g, g, p)
  _ls_checkalias(g, d, base, z)
  v = _leastsquares_residual_gradient!(g, d, base.X, z, base.Y, a)
  _ls_scale_gradient!(g, f, a)
  v
end

function value_from_residual(f::_LSOracleLoss{T}, r::StridedVector{T}) where {T<:Union{Float32, Float64}}
  base, a = _ls_base_scale(f)
  _ls_checklength(:r, r, size(base.X, 1))
  _square_value_wide(a, r, nothing)
end

leastsquares_data(f::LeastSquaresLoss{T}) where {T} = (; Y = f.Y, X = f.X, α = one(T))
leastsquares_data(s::Scaled{<:LeastSquaresLoss}) = (; Y = s.f.Y, X = s.f.X, α = s.α)
