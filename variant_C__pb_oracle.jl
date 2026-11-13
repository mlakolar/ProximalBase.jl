# PB #911 prototype C — one shared PB numerical primitive; PS keeps the formula and α.
# The primitives are PB's existing `_square_value` kernel at level one (fast loop, fast-path test,
# exact zeros, finite residuals recomputed in power-of-two units; a non-finite entry keeps the
# ordinary result). In a real C they would be made public in PB. Disposable.
module PBOracle
import ProximalBase
const PB = ProximalBase
"""half_sumsq_diff(z, Y) -> T: Σ (zᵢ - Yᵢ)²/2 with PB's recovery."""
half_sumsq_diff(z::AbstractVector{T}, Y::AbstractVector{T}) where {T} = PB._square_value(one(T), z, Y)
"""half_sumsq(r) -> T: Σ rᵢ²/2 with PB's recovery."""
half_sumsq(r::AbstractVector{T}) where {T} = PB._square_value(one(T), r, nothing)
end
