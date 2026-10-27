# Type declarations

Implementation declarations, including private workspaces and views. This is
an architecture inventory; internal types are not promoted as extension APIs.
Multiline parameters include their parent. Compatibility bindings are separate.

| Source | Kind | Declaration |
| --- | --- | --- |
| `ext/ProximalBaseGPUArraysCoreExt.jl` | struct | `_InBox{T}` |
| `ext/ProximalBaseGPUArraysCoreExt.jl` | struct | `_Above{G, T}` |
| `ext/ProximalBaseGPUArraysCoreExt.jl` | struct | `_CountAbove{G, T}` |
| `ext/ProximalBaseGPUArraysCoreExt.jl` | struct | `_GroupBlock{M, R, W}` |
| `ext/ProximalBaseGPUArraysCoreExt.jl` | struct | `_GroupLayout{B, V}` |
| `ext/ProximalBaseGPUArraysCoreExt.jl` | struct | `DeviceGroups{F<:_GroupFunction, L<:_GroupLayout} <: AbstractFunction` |
| `ext/ProximalBaseGPUArraysCoreExt.jl` | struct | `_Masked{H, T}` |
| `ext/ProximalBaseGPUArraysCoreExt.jl` | struct | `_Level{T}` |
| `ext/ProximalBaseGPUArraysCoreExt.jl` | struct | `_Weighted{T}` |
| `ext/ProximalBaseGPUArraysCoreExt.jl` | struct | `_L2Factor{T}` |
| `ext/ProximalBaseGPUArraysCoreExt.jl` | struct | `_L2Value{T}` |
| `ext/ProximalBaseGPUArraysCoreExt.jl` | struct | `_AbsMapped{H}` |
| `ext/ProximalBaseGPUArraysCoreExt.jl` | struct | `_ScaledNormSquare{H}` |
| `ext/ProximalBaseGPUArraysCoreExt.jl` | struct | `_AbsIn{T}` |
| `ext/ProximalBaseGPUArraysCoreExt.jl` | struct | `_Shrink{T}` |
| `ext/ProximalBaseGPUArraysCoreExt.jl` | struct | `_SparseScale{T}` |
| `ext/ProximalBaseGPUArraysCoreExt.jl` | struct | `_ObsValue{T, L}` |
| `ext/ProximalBaseGPUArraysCoreExt.jl` | struct | `_ObsDeriv{T, L}` |
| `src/affineconstrained.jl` | mutable struct | `AffineConstrained{T<:AbstractFloat, F<:Union{LeastSquaresLoss, QuadraticFunction}, S<:IndAffine} <: AbstractFunction` |
| `src/atom_iterate.jl` | mutable struct | `AtomIterate{T, G<:AbstractVector{<:AbstractVector{<:Integer}}} <: AbstractVector{T}` |
| `src/ballconstrained.jl` | struct | `BallConstrained{F<:AbstractFunction, T<:Real} <: AbstractFunction` |
| `src/ballconstrained.jl` | struct | `SphereConstrained{F<:AbstractFunction, T<:Real} <: AbstractFunction` |
| `src/calculus.jl` | struct | `_ScaledArray{T, N, A<:AbstractArray} <: AbstractArray{T, N}` |
| `src/calculus.jl` | struct | `_ShiftedArray{T, N, A<:AbstractArray, C<:AbstractArray} <: AbstractArray{T, N}` |
| `src/calculus.jl` | struct | `Scaled{F<:AbstractFunction, T<:Real} <: AbstractFunction` |
| `src/calculus.jl` | struct | `Tilt{F<:AbstractFunction, C<:AbstractArray, D<:Real} <: AbstractFunction` |
| `src/calculus.jl` | struct | `Translate{F<:AbstractFunction, B<:AbstractArray} <: AbstractFunction` |
| `src/calculus.jl` | struct | `Conjugate{F<:AbstractFunction} <: AbstractFunction` |
| `src/calculus.jl` | struct | `SeparableSum{F<:Tuple} <: AbstractFunction` |
| `src/calculus.jl` | struct | `Sum{F<:Tuple} <: AbstractFunction` |
| `src/composition.jl` | struct | `Precompose{F<:AbstractFunction, M, T<:Real} <: AbstractFunction` |
| `src/composition.jl` | struct | `Regularize{F<:AbstractFunction, T<:Real, C<:Union{Nothing, AbstractArray}} <: AbstractFunction` |
| `src/constrained.jl` | struct | `BoxConstrained{F<:AbstractFunction, B<:IndBox} <: AbstractFunction` |
| `src/constrained.jl` | struct | `LinearConstrained{F<:AbstractFunction, H<:Union{IndHyperplane, IndHalfspace}} <: AbstractFunction` |
| `src/constrained.jl` | struct | `_Live{T}` |
| `src/constrained.jl` | struct | `_Reindexed{F, V}` |
| `src/constrained.jl` | struct | `_AtEnd{F, A}` |
| `src/envelope.jl` | struct | `MoreauEnvelope{F<:AbstractFunction, T<:Real} <: AbstractFunction` |
| `src/envelope.jl` | struct | `Smoothed{F<:SeparableFunction, T<:Real} <: ScalarLoss` |
| `src/functions/affine.jl` | struct | `IndAffine{T<:AbstractFloat, M, F} <: AbstractFunction` |
| `src/functions/entropy.jl` | struct | `NegEntropy{T<:Real} <: SeparableFunction` |
| `src/functions/entropy.jl` | struct | `KLDivergence{T<:Real, Q<:Union{Real, AbstractArray}} <: SeparableFunction` |
| `src/functions/entropy.jl` | struct | `NegLog{T<:Real} <: SeparableFunction` |
| `src/functions/entropy.jl` | struct | `Berhu{T<:Real, W<:Union{Nothing, AbstractArray}} <: SeparableFunction` |
| `src/functions/fibers.jl` | struct | `SliceSum{F<:Union{Tuple, AbstractVector}} <: AbstractFunction` |
| `src/functions/fibers.jl` | struct | `FiberSum{F<:AbstractFunction, D<:Union{Nothing, AbstractFunction}} <: AbstractFunction` |
| `src/functions/glm.jl` | abstract type | `ScalarLoss` |
| `src/functions/glm.jl` | struct | `Squared <: ScalarLoss` |
| `src/functions/glm.jl` | struct | `Logistic <: ScalarLoss` |
| `src/functions/glm.jl` | struct | `Poisson <: ScalarLoss` |
| `src/functions/glm.jl` | struct | `Huber{T<:Real} <: ScalarLoss` |
| `src/functions/glm.jl` | struct | `GLMLoss{L<:ScalarLoss, T<:AbstractFloat, Ty<:AbstractVector, TX<:AbstractMatrix, W<:Union{Nothing, AbstractVector}, O<:Union{Nothing, AbstractVector}, V<:AbstractVector{T}} <: AbstractFunction` |
| `src/functions/intersection.jl` | mutable struct | `IndIntersection{S<:Tuple} <: AbstractFunction` |
| `src/functions/intersection.jl` | mutable struct | `_DykstraBuffers{T, N}` |
| `src/functions/leastsquares.jl` | mutable struct | `LeastSquaresLoss{T<:AbstractFloat, Ty<:AbstractVecOrMat, Tx<:AbstractMatrix, N, A<:AbstractArray{T, N}} <: AbstractFunction` |
| `src/functions/logdet.jl` | struct | `LogDetLoss{T<:_BlasReal} <: AbstractFunction` |
| `src/functions/losses.jl` | struct | `CheckLoss{T<:Real, W<:Union{Nothing, AbstractArray}} <: SeparableFunction` |
| `src/functions/losses.jl` | struct | `HingeLoss{Y<:Union{Nothing, AbstractArray}, W<:Union{Nothing, AbstractArray}} <: SeparableFunction` |
| `src/functions/losses.jl` | struct | `EpsInsensitiveLoss{T<:Real, W<:Union{Nothing, AbstractArray}} <: SeparableFunction` |
| `src/functions/losses.jl` | struct | `SquaredHinge <: ScalarLoss` |
| `src/functions/losses.jl` | struct | `SmoothCheck{T<:Real} <: ScalarLoss` |
| `src/functions/losses.jl` | struct | `ConvolutionCheck{T<:Real, K} <: ScalarLoss` |
| `src/functions/matrixsets.jl` | struct | `IndPattern{N} <: SeparableFunction` |
| `src/functions/matrixsets.jl` | struct | `IndBanded <: SeparableFunction` |
| `src/functions/matrixsets.jl` | struct | `IndToeplitz <: AbstractFunction` |
| `src/functions/nonconvex.jl` | struct | `CappedL1{T<:Real, W<:Union{Nothing, AbstractArray}} <: SeparableFunction` |
| `src/functions/nonconvex.jl` | struct | `LogPenalty{T<:Real, W<:Union{Nothing, AbstractArray}} <: SeparableFunction` |
| `src/functions/nonconvex.jl` | struct | `NormLq{T<:Real, W<:Union{Nothing, AbstractArray}} <: SeparableFunction` |
| `src/functions/nonconvex.jl` | struct | `GroupMCP{T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}} <: AbstractGroupPenalty` |
| `src/functions/nonconvex.jl` | struct | `GroupSCAD{T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}} <: AbstractGroupPenalty` |
| `src/functions/nonconvex.jl` | struct | `IndBallGroupL0{G<:AbstractVector{<:AbstractVector{<:Integer}}} <: AbstractFunction` |
| `src/functions/norms.jl` | struct | `NormL2{T<:Real} <: AbstractFunction` |
| `src/functions/norms.jl` | struct | `NormLinf{T<:Real} <: AbstractFunction` |
| `src/functions/norms.jl` | abstract type | `AbstractGroupPenalty <: AbstractFunction` |
| `src/functions/norms.jl` | abstract type | `HomogeneousGroupNorm <: AbstractGroupPenalty` |
| `src/functions/norms.jl` | struct | `GroupNormL2{T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}} <: HomogeneousGroupNorm` |
| `src/functions/norms.jl` | struct | `GroupNormLinf{T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}} <: HomogeneousGroupNorm` |
| `src/functions/norms.jl` | struct | `NormL1Fused{T<:Real} <: AbstractFunction` |
| `src/functions/ordered.jl` | struct | `TotalVariation1D{T<:Real} <: AbstractFunction` |
| `src/functions/ordered.jl` | struct | `FusedLasso{T<:Real} <: AbstractFunction` |
| `src/functions/ordered.jl` | struct | `IndIsotonic <: AbstractFunction` |
| `src/functions/ordered.jl` | struct | `SLOPE{V<:AbstractVector{<:Real}} <: AbstractFunction` |
| `src/functions/separable.jl` | abstract type | `SeparableFunction <: AbstractFunction` |
| `src/functions/separable.jl` | struct | `NormL1{T<:Real, W<:Union{Nothing, AbstractArray}} <: SeparableFunction` |
| `src/functions/separable.jl` | struct | `NormL1Nonneg{T<:Real, W<:Union{Nothing, AbstractArray}} <: SeparableFunction` |
| `src/functions/separable.jl` | struct | `ElasticNet{T<:Real, W<:Union{Nothing, AbstractArray}} <: SeparableFunction` |
| `src/functions/separable.jl` | struct | `IndBox{L<:Union{Real, AbstractArray}, U<:Union{Real, AbstractArray}} <: SeparableFunction` |
| `src/functions/separable.jl` | struct | `IndNonnegative <: SeparableFunction` |
| `src/functions/separable.jl` | struct | `NormL0{T<:Real} <: SeparableFunction` |
| `src/functions/separable.jl` | struct | `MCP{T<:Real} <: SeparableFunction` |
| `src/functions/separable.jl` | struct | `SCAD{T<:Real} <: SeparableFunction` |
| `src/functions/sets.jl` | struct | `IndBallL0 <: AbstractFunction` |
| `src/functions/sets.jl` | struct | `IndBallL1{T<:Real} <: AbstractFunction` |
| `src/functions/sets.jl` | struct | `IndBallL2{T<:Real} <: AbstractFunction` |
| `src/functions/sets.jl` | struct | `IndSphereL2{T<:Real} <: AbstractFunction` |
| `src/functions/sets.jl` | struct | `IndSimplex{T<:Real} <: AbstractFunction` |
| `src/functions/sets.jl` | struct | `IndHyperplane{A<:Union{Real, AbstractArray}, T<:Real} <: AbstractFunction` |
| `src/functions/sets.jl` | struct | `IndHalfspace{A<:Union{Real, AbstractArray}, T<:Real} <: AbstractFunction` |
| `src/functions/singular.jl` | mutable struct | `SingularValue{F<:AbstractFunction} <: AbstractFunction` |
| `src/functions/smooth.jl` | struct | `ZeroFunction <: AbstractFunction` |
| `src/functions/smooth.jl` | struct | `LinearFunction{C<:AbstractArray, D<:Real} <: AbstractFunction` |
| `src/functions/smooth.jl` | struct | `SqrNormL2{T<:Real, Y<:Union{Nothing, AbstractArray}} <: AbstractFunction` |
| `src/functions/smooth.jl` | mutable struct | `QuadraticFunction{T<:AbstractFloat, M<:AbstractMatrix, V<:Union{Nothing, AbstractVector}} <: AbstractFunction` |
| `src/functions/smooth.jl` | struct | `KronQuadratic{T<:AbstractFloat, MC<:Union{Nothing, Matrix{T}}} <: AbstractFunction` |
| `src/functions/sparsegroup.jl` | struct | `SparseGroupNormL2{T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}} <: HomogeneousGroupNorm` |
| `src/functions/spectral.jl` | mutable struct | `NuclearNorm{T<:Real} <: AbstractFunction` |
| `src/functions/spectral.jl` | mutable struct | `IndBallRank <: AbstractFunction` |
| `src/functions/spectral.jl` | mutable struct | `IndPSD{R<:Union{Nothing, Int}} <: AbstractFunction` |
| `src/functions/spectral.jl` | mutable struct | `TracePSD{T<:Real} <: AbstractFunction` |
| `src/interface.jl` | abstract type | `AbstractFunction` |
| `src/lapack.jl` | mutable struct | `SymEigenWorkspace{T<:_BlasReal}` |
| `src/lapack.jl` | mutable struct | `SymEigenTopWorkspace{T<:_BlasReal}` |
| `src/lapack.jl` | mutable struct | `SVDWorkspace{T<:_BlasReal}` |
| `src/metric.jl` | struct | `_ReciprocalArray{T, N, A<:AbstractArray} <: AbstractArray{T, N}` |
| `src/partial.jl` | struct | `QRWorkspace{T<:_BlasReal}` |
| `src/partial.jl` | mutable struct | `PartialSVDWorkspace{T<:_BlasReal}` |
| `src/partial.jl` | mutable struct | `PartialEigenWorkspace{T<:_BlasReal}` |
| `src/sparse_iterate.jl` | mutable struct | `SparseIterate{T, N} <: AbstractArray{T, N}` |
| `src/sparse_iterate.jl` | mutable struct | `SymmetricSparseIterate{T} <: AbstractMatrix{T}` |
| `src/sparse_iterate.jl` | struct | `SymmetricCoordinates{T} <: AbstractVector{T}` |
| `src/spectralfunction.jl` | mutable struct | `Spectral{F<:AbstractFunction} <: AbstractFunction` |
| `src/utils.jl` | mutable struct | `Scratch` |
