# A compact-vector block proximal extension
#
# BlockSquare defines h_k(u)=a[k]*||u||²/2 on each ordered group.
# This example implements a custom group function, not a built-in group penalty.
using ProximalBase, LinearAlgebra
struct BlockSquare <: AbstractFunction
    groups::Vector{Vector{Int}}
    a::Vector{Float64}
end
function ProximalBase.prox_block!(out::AbstractVector, f::BlockSquare,
                                  u::AbstractVector, k::Integer, γ::Real)
    Base.require_one_based_indexing(out, u)
    eltype(out) <: Real && eltype(u) <: Real || throw(ArgumentError("real buffers required"))
    checkbounds(f.groups, k)
    length(out) == length(u) == length(f.groups[k]) || throw(DimensionMismatch("compact block length"))
    out === u || !Base.mightalias(out, u) || throw(ArgumentError("overlapping buffers"))
    length(f.a) == length(f.groups) || throw(DimensionMismatch("one level per block"))
    a = f.a[k]
    isfinite(a) && a >= 0 || throw(ArgumentError("finite nonnegative level required"))
    isfinite(γ) && γ > 0 || throw(ArgumentError("positive finite step required"))
    ## This illustrative extension accepts ordinary finite arithmetic. Unlike the
    ## built-in ridge kernel, it does not promise extreme-range recovery.
    denominator = 1 + γ * a
    isfinite(denominator) || throw(ArgumentError("step times level overflowed"))
    out .= u ./ denominator
    out
end
# A block-only hook does not establish a full-array proximal capability.
f = BlockSquare([[3, 1], [4, 2]], [2.0, 4])
x = BlockIterate(4, f.groups)
@assert all(collect(blockindices(x, k)) == f.groups[k] for k in 1:2)
u = [3.0, -2]
out = zeros(2)
@assert prox_block!(out, f, u, 1, 0.5) === out
@assert out ≈ u / 2 && nactive(x) == 0
allocated = prox_block(f, u, 1, 0.5)
@assert allocated !== u && allocated ≈ out
inplace = copy(u)
prox_block!(inplace, f, inplace, 1, 0.5)
@assert inplace ≈ out
# Only this separate public storage write activates the block.
copyto!(blockview(x, 1), out)
@assert collect(activecoordinates(x)) == [1]
copyto!(blockview(x, 2), prox_block(f, zeros(2), 2, 0.5))
@assert !isactive(x, 2)
# Other overlapping buffers fail before any output changes.
buffer = [1.0, 2, 3]
try
    prox_block!(view(buffer, 1:2), f, view(buffer, 2:3), 1, 0.5)
    error("expected overlap to fail")
catch err
    @assert err isa ArgumentError
end
@assert buffer == [1, 2, 3]
Vector(x)
