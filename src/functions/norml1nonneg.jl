####################################
#  ℓ1 norm on the nonnegative orthant
####################################

"""
    NormL1Nonneg(λ0 = 1)
    NormL1Nonneg(λ0, w)

Weighted ℓ₁ penalty on the nonnegative orthant: `f(x) = λ₀ Σᵢ wᵢ xᵢ` where every
entry is `≥ 0`, and `Inf` where one is negative; all weights equal one when `w` is
omitted. The level and weights must be nonnegative and not `NaN`; these are checked
at construction. Weights are retained by reference and paired with entries by
linear index; later array mutations are not revalidated.

[`prox`](@ref) applies `max(vᵢ - t̂ᵢ, 0)` to each entry, with the threshold
`t̂ᵢ = γλ₀wᵢ` formed as [`NormL1`](@ref) forms it: the proximal operator of
`λ₀wᵢxᵢ` on `xᵢ ≥ 0`. The function is separable, convex and zero-preserving; value and
same-kind sparse-iterate prox calls visit only stored entries.

# Examples
```jldoctest
julia> f = NormL1Nonneg(2.0);

julia> f([1.0, 0.5])
3.0

julia> f([1.0, -1e-300])
Inf

julia> prox(NormL1Nonneg(0.5, [1.0, 2.0]), [2.0, -1.0], 1.0)
2-element Vector{Float64}:
 1.5
 0.0
```

# Edge cases
- Use real, one-based arrays and one weight per entry.
- The domain is `xᵢ ≥ 0` exactly: `-0.0` and `+Inf` are feasible, and any negative number,
  a subnormal one included, is not. A zero level or weight keeps the domain:
  `NormL1Nonneg(0.0)` is the indicator of the orthant, with value `0` at a feasible point
  (`+Inf` entries included) and `Inf` at a negative entry.
- The value is `NaN` when an entry that is read is `NaN`, and otherwise `Inf` when one is
  negative; `NaN` takes precedence over `Inf`. At a feasible point the terms are those of
  `NormL1`: `-0.0` gives `+0`.
- A zero level, a zero weight or a zero step gives the projection onto the orthant,
  `max(vᵢ, 0)`, not the identity: `-Inf` and `-0.0` give `+0`, `+Inf` stays and `NaN` stays.
- An infinite level fixes positive-weight entries at `+0` for every step `γ ≥ 0`, `NaN` and
  `±Inf` inputs included; zero-weight entries are projected. The value of a positive-weight
  entry is `0` at `0`, `NaN` at `NaN` and `Inf` at any other entry.
- An infinite weight at a finite level maps finite inputs and `-Inf` to `+0`, also at zero
  level or step, and `+Inf` and `NaN` inputs to `NaN`, as for `NormL1`.
- A negative or `NaN` scalar step throws an `ArgumentError` before any write in `prox`,
  `prox!` and `value_and_prox!`. The entries of an array of steps and the scalar step of
  `prox_block` are not checked, and neither is a step that reaches the function through
  [`Scaled`](@ref); there a negative step violates the precondition `γ ≥ 0`, and the
  result depends on the representation (dense, sparse or symmetric).
- On a [`SymmetricSparseIterate`](@ref), weights have `p²` entries in full-matrix
  linear order, interpreted as an exactly symmetric `p × p` array. Oracles assume this; validate with
  [`ProximalBase.check_symmetric_parameters`](@ref).

See [ℓ₁ endpoint rules](@ref l1-endpoints), the
[numerical contract](@ref numerics) and [`prox!`](@ref) for step and alias rules.
"""
struct NormL1Nonneg{T<:Real, W<:Union{Nothing, AbstractArray}} <: SeparableFunction
  λ0::T
  w::W
  # NormL1(λ0, w), built once: its constructor scans the weights, and the value reaches its kernels through it
  l1::NormL1{T, W}

  function NormL1Nonneg{T, W}(λ0::T, w::W) where {T<:Real, W<:Union{Nothing, AbstractArray}}
    _check_one_based(w)
    w === nothing || eltype(w) <: Real || _real_error(:NormL1Nonneg)
    _checknonneg("λ0", λ0)
    _checknonneg("the weights", w)
    new{T, W}(λ0, w, NormL1(λ0, w))
  end
end
NormL1Nonneg(λ0::Real = 1, w::Union{Nothing, AbstractArray} = nothing) = NormL1Nonneg{typeof(λ0), typeof(w)}(λ0, w)

isconvex(::Type{<:NormL1Nonneg}) = true
iszeropreserving(::NormL1Nonneg) = true
_checkparams(f::NormL1Nonneg, x) = _checkparams(f.l1, x)

# The NormL1 entry value where v >= 0 (an absent weight is one in the entry's type, #684), NaN at a NaN and Inf at a
# negative entry, all in the type of that value.
Base.@propagate_inbounds function value_entry(f::NormL1Nonneg, v::Real, i::Integer)
  p = value_entry(f.l1, v, i)
  ifelse(v >= 0, p, ifelse(isnan(v), oftype(p, NaN), oftype(p, Inf)))
end

# The proximal operator of λ0 wᵢ xᵢ on xᵢ >= 0: max(v - t, 0) with the threshold t = γλ0wᵢ that NormL1 forms
# (_threshold_product, and _weight_product outside the ordinary range of γλ0). The exceptional branch also keeps the
# infinite-level rule of NormL1 (_l1_fixed: an infinite level fixes the entries with a positive weight at +0 for every
# γ >= 0, NaN and ±Inf inputs included; numerics.md, "NaN, infinities and zeros").
Base.@propagate_inbounds function prox_block(f::NormL1Nonneg, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  w = _entry(f.w, i)
  t, normal = _threshold_product(S, γ, f.λ0, w)
  normal && return max(S(v) - t, zero(S))
  t = _weight_product(S, γ, f.λ0, w)
  r = max(S(v) - t, zero(S))
  ifelse(_l1_fixed(γ, f.λ0, w), zero(r), r)
end

# Step check (numerics.md, "NaN, infinities and zeros"), as for ElasticNet: the public array oracles refuse a negative
# or NaN scalar step before anything is written, and then take the checked separable route. prox_block and the routes
# inside ProximalBase do not test the step: Scaled hands its own step on. Array steps are not read here. With a negative
# step the proximal point of 0 is -t̂ > 0, so the loops over the stored entries of a sparse iterate and over every entry
# of a dense array would disagree.
@noinline _nonneg_step_error(γ) = throw(ArgumentError("NormL1Nonneg requires a nonnegative step γ, got $γ"))
@inline _check_nonneg_step(γ::Real) = (γ >= 0 || _nonneg_step_error(γ); nothing)
prox!(y::_ArrayOrSymmetricIterate, f::NormL1Nonneg, x::_ArrayOrSymmetricIterate, γ::Real) =
  (_check_nonneg_step(γ); invoke(prox!, _SeparableScalarStep, y, f, x, γ))
value_and_prox!(y::_ArrayOrSymmetricIterate, f::NormL1Nonneg, x::_ArrayOrSymmetricIterate, γ::Real) =
  (_check_nonneg_step(γ); invoke(value_and_prox!, _SeparableScalarStep, y, f, x, γ))

# The value. The term of an entry is that of NormL1, with Inf in place of it at a negative entry, so that a sum of the
# terms is the NormL1 sum at a feasible point (the NormL1 loops take the term as the hook _l1_value), Inf when an entry
# is negative, and NaN when one is NaN (NaN + Inf is NaN). A term is `Inf` in the type of the accumulator.
@inline _l1_value(::NormL1Nonneg, v, a) = ifelse(v < 0, oftype(a, Inf), a * abs(v))

# Dense Float32 or Float64 arrays with weights of the type of the data: the NormL1 loop and the checks of its sum
# (norml1.jl): a coefficient out of range or zero, 0⋅Inf or NaN, and a sum below n floatmin (#373).
function _value_dense(f::NormL1Nonneg{T, <:Array{T}}, x::Array{T}) where {T<:Union{Float32, Float64}}
  s, ordinary = _l1_value_sum(f, x, false)
  ordinary && _sum_kept(s, length(x)) && return s
  _l1_exceptional_value(f, x, s, ordinary)
end
# The recovery of a sum below n floatmin (#373) is that of NormL1 over the same entries; a sum that is Inf or NaN (a
# negative or NaN entry) is kept by _sum_kept.
@inline _l1_underflow_checked(f::NormL1Nonneg, x, s, n::Integer) = _l1_underflow_checked(f.l1, x, s, n)

# The other representations take the entrywise loops of separable.jl (value_entry) and the check of NormL1 for their
# sum: dense arrays other than the one above (views, mixed precision, unweighted arrays that are not dense), n entries;
# sparse iterates, nnz terms; block iterates, at most the length of the storage; symmetric iterates, at most two terms
# per stored coordinate.
_nonneg_generic_value(f, x) = _l1_underflow_checked(f, x, invoke(_value_dense, Tuple{Any, AbstractArray}, f, x), length(x))
_value_dense(f::NormL1Nonneg, x::AbstractArray) = _nonneg_generic_value(f, x)
_value_sep(f::NormL1Nonneg, x::SparseIterate) =
  _l1_underflow_checked(f, x, invoke(_value_sep, Tuple{Any, SparseIterate}, f, x), x.nnz)
_value_sep(f::NormL1Nonneg, x::BlockIterate) =
  _l1_underflow_checked(f, x, invoke(_value_sep, Tuple{Any, BlockIterate}, f, x), length(x.storage))
_value_sep(f::NormL1Nonneg, x::SymmetricSparseIterate) =
  _l1_underflow_checked(f, x, invoke(_value_sep, Tuple{Any, SymmetricSparseIterate}, f, x), 2 * x.nnz)

# Without weights, a dense array of floats takes the route of the call operator of NormL1 (norml1.jl) with the
# membership of the entries in the sum. A term of the sum of magnitudes is Inf at a negative entry and |v| otherwise,
# and the sums are those of _value_sumabs (a loop for _VALUE_LOOP_MIN to _VALUE_LOOP_MAX entries of a strided
# Float32 or Float64 array and Julia's reduction otherwise). A sum cannot tell a feasible +Inf from a negative
# entry where the level is zero or does not fit, so those two cases take a sum of membership terms: Inf at a
# negative entry, 0 at any other number, NaN at NaN. The level converted to T that does not keep a normal number
# (_level_fits) multiplies the sum of magnitudes in the wider type after membership (_converted_l1_value). The
# overflow of the sum or of its product with the level takes the entrywise value, which also gives a negative entry its
# Inf: the call of the generic method goes through invoke, since _value_dense on f would be this method.
@inline _nonneg_member(v::T) where {T} = ifelse(v < 0, T(Inf), ifelse(v == v, zero(T), v))
@inline _nonneg_abs(v::T) where {T} = ifelse(v < 0, T(Inf), abs(v))
@inline _nonneg_reduce(g, x) = isempty(x) ? zero(float(eltype(x))) : sum(g, x)
@inline _nonneg_sum(g, x) = _nonneg_reduce(g, x)
@inline _nonneg_sum(g, x::StridedArray{T}) where {T<:Union{Float32, Float64}} =
  _VALUE_LOOP_MIN <= length(x) <= _VALUE_LOOP_MAX ? _value_loop(g, x) : _nonneg_reduce(g, x)
function _value_dense(f::NormL1Nonneg{<:Real, Nothing}, x::DenseArray{<:AbstractFloat})
  T = float(eltype(x))
  λ = f.λ0
  iszero(λ) && return _nonneg_sum(_nonneg_member, x)
  if !_level_fits(T, λ)
    m = _nonneg_sum(_nonneg_member, x)
    return iszero(m) ? _converted_l1_value(λ, x) : m
  end
  s = _nonneg_sum(_nonneg_abs, x)
  # an infinite level fixes every entry at zero: the value at the zero vector is 0, not Inf⋅0
  isinf(λ) && iszero(s) && return zero(T)
  v = convert(T, λ) * s
  isfinite(v) || !isfinite(λ) || !all(isfinite, x) ? v : _nonneg_generic_value(f, x)
end

kkt_violation(f::NormL1Nonneg, g::AbstractArray, i::Integer) =
  (_check_one_based(g); eltype(g) <: Real || _real_error(:kkt_violation); -g[i] - _kkt_coefficient(eltype(g), f.λ0, _entry(f.w, i)))

# The point-aware score (see kkt_violation): the coefficient cᵢ = λ0 wᵢ of the NormL1 score (_kkt_state and
# _kkt_entry_coefficient are NormL1's), then -vᵢ - cᵢ at a zero entry (the screen above in the working type), |vᵢ + cᵢ|
# at a positive one, and +Inf at a negative one, outside the domain. The three are selected without a branch, so that
# the loop over a dense array vectorizes. The flag of the loop is NaN where the natural score is infinite or NaN
# (at vᵢ = +Inf the score at a zero entry is -Inf, which the rules for an infinite entry turn into +Inf), so that the
# pass by entry decides those entries (interface.jl, _kkt_reduce); the default flag tests xᵢ alone.
@inline _kkt_state(f::NormL1Nonneg, ::Type{W}) where {W} = _kkt_state(f.l1, W)
Base.@propagate_inbounds _kkt_entry_coefficient(f::NormL1Nonneg, s, ::Type{W}, i) where {W} = _kkt_entry_coefficient(f.l1, s, W, i)
Base.@propagate_inbounds function _kkt_natural(f::NormL1Nonneg, s, ::Type{W}, v::W, x::W, i) where {W}
  c = _kkt_entry_coefficient(f, s, W, i)
  ifelse(iszero(x), -v - c, ifelse(x < 0, W(Inf), abs(v + c)))
end
@inline _kkt_flag(::NormL1Nonneg, x, r) = (x - x) + (r - r)
@_kkt_entrywise NormL1Nonneg

restrict(f::NormL1Nonneg, S::AbstractVector{<:Integer}) = NormL1Nonneg(f.λ0, _restrict_entries(f.w, S))
