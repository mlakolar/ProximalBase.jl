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
# The same for SqrtLeastSquaresLoss, which shares the oracles value_from_predictor and value_from_residual and the accessor
# leastsquares_data with the least-squares loss, and has no gradient oracle.
const _SqrtLSStrided{T} = SqrtLeastSquaresLoss{T, <:StridedVector{T}, <:StridedMatrix{T}}
const _SqrtLSOracleLoss{T} = Union{_SqrtLSStrided{T}, Scaled{<:_SqrtLSStrided{T}}}

# The loss and the scale a of the computation: one(T) for a LeastSquaresLoss, and for a Scaled the factor
# as supplied when it is a positive finite Float32 or Float64 (not rounded to T), or the Float64 of an
# integer α with Float64(α) == α. The type of a is determined by the type of the factor. Any other
# factor is refused here, before the callers write.
@inline _ls_base_scale(f::LeastSquaresLoss{T}) where {T} = (f, one(T))
@inline _ls_base_scale(f::SqrtLeastSquaresLoss{T}) where {T} = (f, one(T))
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
# sends through _ls_value_cold to _ls_value_exceptional, which checks the factor (_ls_base_scale) and raises the
# ArgumentError.
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
@noinline _ls_offset_error() =
  throw(ArgumentError("the outputs g and d, and a nonempty design X, must have axes starting at 1"))

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

Evaluate the least-squares value `α*sum(abs2, z-Y)/2` from a predictor `z = X*β`
already computed by the caller. Here `α = 1` for a [`LeastSquaresLoss`](@ref), or
is the factor of one [`Scaled`](@ref) wrapper. No design entries are read and no
loss fields are written. The call neither reads nor writes `f.tmp`; the supplied `z` may be a view of `f.tmp`.

Use a vector-response loss with strided `Float32` or `Float64` data of one type
`T`, and a `StridedVector{T}` predictor with the response's axes. Other types
throw `MethodError`; mismatched axes throw `DimensionMismatch`. The result is `T`.

# Examples
```jldoctest
julia> f = LeastSquaresLoss([1.0], reshape([2.0], 1, 1));

julia> ProximalBase.value_from_predictor(f, [3.0])
2.0
```

For [`SqrtLeastSquaresLoss`](@ref), or one `Scaled` wrapper, this returns
`α*norm(z-Y)` rather than a squared loss. The strided type, axes and factor
restrictions are the same, including rejection of an infinite scale **before**
evaluating a zero residual. Extreme sums and products use power-of-two range
recovery, not `sqrt(2*v)` of the squared-loss value; see the
[`SqrtLeastSquaresLoss`](@ref) accuracy contract. Neither `X`, `f.tmp` nor `z`
is read or written beyond the caller-supplied predictor entries.

# Edge cases
- A scale must be positive finite `Float32`/`Float64`, or an integer exactly
  representable as `Float64`; other factors throw `ArgumentError`.
- A non-finite predictor gives a non-finite value; the call does not recompute `X*β`.
- Finite-input value recovery has explicit rounding and overflow-boundary limits.
- Calls write no loss fields and can share one loss across tasks while its data
  remain unchanged. The ordinary path allocates nothing for raw losses and
  `Float32`, `Float64` or `Int` scales.

See [predictor domains, accuracy and ownership](@ref Least-squares-at-a-predictor-or-residual-that-the-caller-holds)
and [`ProximalBase.value_from_residual`](@ref).
"""
function value_from_predictor end

"""
    ProximalBase.value_and_gradient_from_predictor!(g, d, f, z)

Write residual `d = z-Y` and gradient `g = α*(X'*d)` from the caller's predictor
`z = X*β`, and return `α*sum(abs2, d)/2`. This forms one product with `X'` and
none with `X`; no loss fields or private scratch are written.

Use the domain and scale of [`ProximalBase.value_from_predictor`](@ref), with
strided vectors `g`, `d` and `z` of the data type. `z` and `d` have the response's
axes; `g` has length `size(X, 2)` and must be one-based.

# Examples
```jldoctest
julia> f = LeastSquaresLoss([1.0], reshape([2.0], 1, 1));

julia> g = zeros(1); d = zeros(1);

julia> ProximalBase.value_and_gradient_from_predictor!(g, d, f, [3.0])
2.0

julia> (g[1], d[1])
(4.0, 2.0)
```

# Edge cases
- Outputs must not overlap each other, the predictor, design, response or loss
  scratch. Nonempty overlaps reported by `Base.mightalias` are rejected;
  undetected sharing remains the caller's responsibility. In-place `d === z`
  is not supported.
- Length/axis, factor and overlap checks precede writes. On any error, `g`,
  `d` and `f.tmp` remain unchanged.
- Overflow/underflow in `X'*d` is not recovered before applying `α`.
  Overflow in `z-Y` makes the value non-finite, even where the standalone
  predictor value can recover it. Last-bit agreement is not promised.

See [predictor accuracy, failure and concurrency rules](@ref Least-squares-at-a-predictor-or-residual-that-the-caller-holds).
"""
function value_and_gradient_from_predictor! end

"""
    ProximalBase.value_from_residual(f, r)

Evaluate `α*sum(abs2, r)/2` from a least-squares residual already held by the
caller. Either sign, `r = X*β-Y` or `r = Y-X*β`, is accepted. No design or response
entries are read and no loss fields are written. The call neither reads nor writes `f.tmp`;
the supplied `r` may be a view of `f.tmp`.

Use the loss, scale and strided floating-point domain of
[`ProximalBase.value_from_predictor`](@ref). The residual must have the response's
axes (`DimensionMismatch` otherwise). The result type, accuracy, allocation and
concurrency guarantees are the same, with non-finite residuals producing
non-finite values.

For [`SqrtLeastSquaresLoss`](@ref), or one `Scaled` wrapper, the value is
`α*norm(r)` for either residual sign. It has the same scale, domain, accuracy
and range-recovery rules as that loss's `value_from_predictor`; no `X`, `Y` or
`f.tmp` entries are accessed and `r` is not modified.

# Examples
```jldoctest
julia> f = LeastSquaresLoss([1.0], reshape([2.0], 1, 1));

julia> ProximalBase.value_from_residual(f, [-2.0])
2.0
```

See [predictor and residual operations](@ref Least-squares-at-a-predictor-or-residual-that-the-caller-holds).
"""
function value_from_residual end

"""
    ProximalBase.leastsquares_data(f)

Return borrowed least-squares data as `(; Y, X, α)`. `Y` and `X` are the stored
arrays themselves, with no copy or conversion. The scale is `one(T)` for a
[`LeastSquaresLoss`](@ref) of type `T`, or the stored factor of one [`Scaled`](@ref)
wrapper, with its original type.

Any `LeastSquaresLoss` or `SqrtLeastSquaresLoss` is accepted, including data outside the strided domain of
the predictor oracles. Other functions throw `MethodError`. This accessor does
not validate or convert the scale; the predictor oracles can reject it or use
its converted value.

The accessor also accepts [`SqrtLeastSquaresLoss`](@ref) and one `Scaled`
wrapper of it; `α` is the raw loss's `one(T)` or the wrapper's stored factor,
respectively. The returned data do not distinguish the two losses; their
types do. Calling the accessor does not validate that a scale is accepted
by the numerical oracles.

# Examples
```jldoctest
julia> y = [1.0]; X = reshape([2.0], 1, 1);

julia> data = ProximalBase.leastsquares_data(LeastSquaresLoss(y, X));

julia> (data.Y === y, data.X === X, data.α)
(true, true, 1.0)
```

Do not mutate the borrowed arrays while the loss or any of its oracles is in use,
including on another task. See [ownership rules](@ref numerical-usage).
"""
function leastsquares_data end

# The axes of z are checked against those of Y (whose length is size(X, 1) by the constructor of the loss) by
# _ls_value, which throws a DimensionMismatch before it reads an entry; the call reads no entry or size of X.
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
  X = base.X              # the product with a nonempty Xᵀ (BLAS) refuses offset axes; an empty one is not passed to BLAS
  (firstindex(g) == firstindex(d) == 1 && (isempty(X) || firstindex(X, 1) == firstindex(X, 2) == 1)) ||
    _ls_offset_error()
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

@inline _sqrt_ls_response(src::Scaled{<:SqrtLeastSquaresLoss}) = src.f.Y

# The square-root loss: the factor is checked first (_ls_base_scale: a positive finite Float32 or Float64, or an integer
# that Float64 holds), then the axes of z or r against those of Y, before any entry is read; the oracles read no entry
# of X and neither read nor write f.tmp.
function value_from_predictor(f::_SqrtLSOracleLoss{T}, z::StridedVector{T}) where {T<:Union{Float32, Float64}}
  base, a = _ls_base_scale(f)
  _sqrt_ls_value(a, z, base.Y, f)
end

function value_from_residual(f::_SqrtLSOracleLoss{T}, r::StridedVector{T}) where {T<:Union{Float32, Float64}}
  base, a = _ls_base_scale(f)
  axes(r) == axes(base.Y) || _ls_axes_error("r", axes(r, 1), axes(base.Y, 1))
  _sqrt_ls_value(a, r, nothing, nothing)
end

leastsquares_data(f::LeastSquaresLoss{T}) where {T} = (; Y = f.Y, X = f.X, α = one(T))
leastsquares_data(s::Scaled{<:LeastSquaresLoss}) = (; Y = s.f.Y, X = s.f.X, α = s.α)
leastsquares_data(f::SqrtLeastSquaresLoss{T}) where {T} = (; Y = f.Y, X = f.X, α = one(T))
leastsquares_data(s::Scaled{<:SqrtLeastSquaresLoss}) = (; Y = s.f.Y, X = s.f.X, α = s.α)
