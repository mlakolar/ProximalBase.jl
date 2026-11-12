# A custom separable function
#
# *Assumes: Julia structs and convex separable functions · You will learn: a custom function with mathematical value and proximal hooks.*
# See [zero-preserving functions](../glossary.md#Zero-preserving-function) for the sparse-computation trait.
#
# WeightedSquare represents sum(w[i]*x[i]^2)/2, with finite nonnegative weights.
using ProximalBase, LinearAlgebra
struct WeightedSquare{A<:AbstractArray} <: SeparableFunction
    w::A
    function WeightedSquare(w::A) where {A<:AbstractArray}
        eltype(w) <: Real || throw(ArgumentError("real weights required"))
        all(v -> isfinite(v) && v >= 0, w) || throw(ArgumentError("finite nonnegative weights required"))
        new{A}(copy(w))
    end
end
ProximalBase.value_entry(f::WeightedSquare, v::Real, i::Integer) = f.w[i] * v^2 / 2
ProximalBase.prox_block(f::WeightedSquare, v::Real, i::Integer, γ::Real) = v / (1 + γ * f.w[i])
ProximalBase.iszeropreserving(::WeightedSquare) = true
ProximalBase.isconvex(::Type{<:WeightedSquare}) = true
ProximalBase.entry_parameters(f::WeightedSquare) = (:w => f.w,)
function ProximalBase.check_parameters(f::WeightedSquare, x)
    length(f.w) == prod(size(x)) || throw(DimensionMismatch("weights must match the number of entries"))
    nothing
end
# No smoothness trait or gradient is synthesized by SeparableFunction.
f = WeightedSquare([1.0, 2, 3])
x = [1.0, -2, 0]
@assert isseparable(f) && isproximable(f) && !issmooth(f)
@assert f(x) ≈ sum(f.w .* x.^2) / 2
@assert prox(f, x, 0.5) ≈ x ./ (1 .+ 0.5 .* f.w)
s = SparseIterate(x)
activate!(s, 3)
z = prox(f, s, 0.5)
@assert Vector(z) ≈ prox(f, x, 0.5)
@assert nactive(z) == 2  # fresh output does not inherit an active zero
# On a symmetric iterate the weights must be exactly symmetric. The oracles assume
# this; the declaration above lets the explicit check validate the model once.
h = WeightedSquare([1.0 2; 2 3])
sym = SymmetricSparseIterate([1.0 2; 2 0])
@assert ProximalBase.check_symmetric_parameters(h, sym)
@assert h(sym) ≈ sum(h.w .* Matrix(sym).^2) / 2
@assert Matrix(prox(h, sym, 0.5)) ≈ Matrix(sym) ./ (1 .+ 0.5 .* h.w)
# Asymmetric weights are an invalid model, rejected by the explicit check before any
# oracle is called. The oracles give them no meaning.
bad = WeightedSquare([1.0 2; 3 1])
try
    ProximalBase.check_symmetric_parameters(bad, sym)
    error("expected asymmetric weights to fail")
catch err
    @assert err isa ArgumentError
end
Vector(z)
