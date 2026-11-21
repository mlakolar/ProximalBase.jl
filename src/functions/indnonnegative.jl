####################################
#  indicator of the nonnegative orthant
####################################

"""
    IndNonnegative()

The indicator of the nonnegative orthant `{x : x ≥ 0}`: the value is `0` at a point
whose entries are all nonnegative and `Inf` at a point with a negative entry and no `NaN` entry
(a `NaN` entry gives `NaN`, see below). The proximal operator is the projection
`max(xᵢ, 0)` of each entry at a step `γ ≥ 0`, including `γ = 0` and `γ = Inf`.
The function is convex, separable and zero-preserving.

# Examples
```jldoctest
julia> f = IndNonnegative();

julia> f([1.0, 0.0]), f([1.0, -0.5])
(0.0, Inf)

julia> prox(f, [2.0, -1.0], 0.5)
2-element Vector{Float64}:
 2.0
 0.0
```

# Edge cases
- Use real, one-based arrays. The result type is `float(eltype(x))`.
- A `NaN` entry gives value `NaN` (also when another entry is negative) and stays `NaN`
  in the projection; it is not reported as feasible. `-0.0` and `+Inf` are feasible with
  value `0`; the projection of `-0.0` and of `-Inf` is `+0.0`.
- A scalar step that is negative or `NaN` throws an `ArgumentError` before anything is written in
  [`prox`](@ref), [`prox!`](@ref) and [`value_and_prox!`](@ref); another check (argument axes, shape or
  type) can raise its error first. Array-step entries and scalar `prox_block` steps do not check the step,
  and project; the step does not enter the result.
- `Scaled(f, α)`, `α * f` and `f * α` return `f` itself for a nonnegative `α`,
  including `0` and `Inf`. A negative or `NaN` factor throws `ArgumentError`.
- [`value_and_prox!`](@ref) returns the value at the stored output: `NaN` when an
  output entry is `NaN`, and otherwise `0`.
- The value of a [`SparseIterate`](@ref) or [`SymmetricSparseIterate`](@ref), and the proximal operator
  from a `SparseIterate` into a `SparseIterate` or from a `SymmetricSparseIterate` into a
  `SymmetricSparseIterate`, read stored entries only. A dense output from a sparse or symmetric input
  goes through the dense loop and reads every entry, the unstored ones as zeros.
- [`kkt_violation`](@ref) has the screen `-v[i]` only (positive when `v[i] < 0`, where the first-order
  optimality condition fails at a zero coordinate: increasing it decreases the objective to first
  order); the point-aware forms are not defined.

See [the numerical contract](@ref numerics) for the endpoint rules.
"""
struct IndNonnegative <: SeparableFunction end

isconvex(::Type{IndNonnegative}) = true
iszeropreserving(::IndNonnegative) = true
_isindicator(::Type{IndNonnegative}) = true

# Three cases with one select: 0 at v >= 0 (-0.0 and +Inf included); at a negative v (-Inf and the subnormals
# included) the product S(v) * (-Inf) is +Inf; a NaN fails the comparison and NaN * (-Inf) is NaN. The sum of such
# values is NaN when an entry is NaN, otherwise Inf when an entry is negative, otherwise 0.
Base.@propagate_inbounds function value_entry(::IndNonnegative, v::T, ::Integer) where {T<:Real}
  S = float(T)
  ifelse(v >= 0, zero(S), S(v) * S(-Inf))
end
# max(float(v), +0): -0.0 and -Inf give +0.0, NaN stays NaN; the step does not enter
Base.@propagate_inbounds prox_block(::IndNonnegative, v::T, ::Integer, ::Real) where {T<:Real} =
  max(float(v), zero(float(T)))

# Step check (#908; numerics.md, "NaN, infinities and zeros"): the public array oracles refuse a negative or NaN
# scalar step before anything is written, and then take the checked separable route, as ElasticNet does
# (_SeparableScalarStep is defined in elasticnet.jl). prox_block and array-step entries do not test the step.
@noinline _indnonnegative_step_error(γ) =
  throw(ArgumentError("IndNonnegative requires a nonnegative step γ, got $γ"))
@inline _check_indnonnegative_step(γ::Real) = (γ >= 0 || _indnonnegative_step_error(γ); nothing)
prox!(y::_ArrayOrSymmetricIterate, f::IndNonnegative, x::_ArrayOrSymmetricIterate, γ::Real) =
  (_check_indnonnegative_step(γ); invoke(prox!, _SeparableScalarStep, y, f, x, γ))
value_and_prox!(y::_ArrayOrSymmetricIterate, f::IndNonnegative, x::_ArrayOrSymmetricIterate, γ::Real) =
  (_check_indnonnegative_step(γ); invoke(value_and_prox!, _SeparableScalarStep, y, f, x, γ))

# the screen of coordinate i assumed zero: positive when v[i] < 0, where the first-order optimality condition
# fails at zero (increasing the coordinate decreases the objective to first order)
kkt_violation(::IndNonnegative, g::AbstractArray, i::Integer) =
  (_check_one_based(g); eltype(g) <: Real || _real_error(:kkt_violation); -g[i])

restrict(f::IndNonnegative, ::AbstractVector{<:Integer}) = f
