module ElasticNetTest
# ElasticNet (#908): the elastic-net penalty λ1 Σᵢ wᵢ|xᵢ| + (λ2/2) Σᵢ xᵢ². Constructor and traits, the analytic
# cases, the scalar step check of the public array oracles (D1), the representations, the KKT screen and
# restriction, agreement with the archived implementation (eff21b6d) and allocations. The endpoint truth
# table, the extreme ranges and Scaled are in elasticnet_extremes.jl.
#
# References are independent of the kernels: the closed form shrink(v, t̂)/(1 + γλ2) in 4096-bit BigFloat,
# with t̂ the threshold that the NormL1 kernel forms (the ridge kernel multiplies by the rounded weight
# 1/(1 + γλ2), so a result is compared within 4 units in the last place), and the value
# λ1 Σ wᵢ|xᵢ| + λ2/2 Σ xᵢ² in BigFloat, within (n + 2) eps relative for n terms. Equality is asserted only
# where the result is exact (zeros, identities, the cases that reduce to NormL1 or SqrNormL2).
using Test, ProximalBase, LinearAlgebra, SparseArrays, Random
const PB = ProximalBase

big4096(f) = setprecision(f, BigFloat, 4096)
weight(f, i) = f.w === nothing ? 1 : f.w[i]
# the threshold γλ1wᵢ as the NormL1 kernel rounds it (it does not depend on the trial)
threshold(::Type{T}, f, i, γ) where {T} = PB._l1_prox_threshold(zero(T), γ, f.λ1, PB._entry(f.w, i), T)[2]
function proxref(::Type{T}, f, v, i, γ) where {T}
  t = threshold(T, f, i, γ)
  big4096() do
    b = big(T(v))
    T(sign(b) * max(abs(b) - big(t), 0) / (1 + big(γ) * big(f.λ2)))
  end
end
within(got, ref, k = 4) = isequal(got, ref) || abs(got - ref) <= k * eps(abs(ref))
# λ1 Σ wᵢ|xᵢ| + λ2/2 Σ xᵢ² over the entries of a dense array, in BigFloat
valref(f, x) = big4096() do
  big(f.λ1) * sum(iszero(x[i]) ? big(0) : big(weight(f, i)) * abs(big(x[i])) for i in eachindex(x); init = big(0)) +
    big(f.λ2) / 2 * sum(big(v)^2 for v in x; init = big(0))
end
valok(got::T, ref, n) where {T} = isfinite(got) && abs(big(got) - ref) <= (n + 2) * eps(T) * abs(ref)

# The archived ElasticNet (eff21b6d) at ordinary inputs, computed once from that source tree: prox_block of
# ElasticNet(λ1, λ2, w) (w = nothing, or the single weight given) at the trial v and step γ, and values at a
# fixed vector. The candidate agrees within 4 units in the last place (proximal points) and within
# 2(n + 2) eps relative (values, each within (n + 2) eps of the exact value).
const ARCHIVED_PROX = [
  # (T, λ1, λ2, w, γ, v, archived prox_block)
  (Float32, 0.3, 0.5, nothing, 0.25, -2.5, -2.1555555f0),
  (Float32, 0.3, 0.5, nothing, 0.25, 0.7, 0.5555556f0),
  (Float32, 0.3, 0.5, nothing, 0.25, 4.0, 3.4888887f0),
  (Float32, 0.3, 0.5, nothing, 0.25, 0.001, 0.0f0),
  (Float32, 0.3, 0.5, nothing, 0.25, -0.35, -0.24444443f0),
  (Float32, 0.3, 0.5, nothing, 1.0, -2.5, -1.4666667f0),
  (Float32, 0.3, 0.5, nothing, 1.0, 0.7, 0.26666665f0),
  (Float32, 0.3, 0.5, nothing, 1.0, 4.0, 2.4666667f0),
  (Float32, 0.3, 0.5, nothing, 1.0, 0.001, 0.0f0),
  (Float32, 0.3, 0.5, nothing, 1.0, -0.35, -0.03333332f0),
  (Float32, 0.3, 0.5, nothing, 3.0, -2.5, -0.64f0),
  (Float32, 0.3, 0.5, nothing, 3.0, 0.7, 0.0f0),
  (Float32, 0.3, 0.5, nothing, 3.0, 4.0, 1.24f0),
  (Float32, 0.3, 0.5, nothing, 3.0, 0.001, 0.0f0),
  (Float32, 0.3, 0.5, nothing, 3.0, -0.35, 0.0f0),
  (Float32, 0.3, 0.5, 1.5, 0.25, -2.5, -2.1222222f0),
  (Float32, 0.3, 0.5, 1.5, 0.25, 0.7, 0.5222222f0),
  (Float32, 0.3, 0.5, 1.5, 0.25, 4.0, 3.4555557f0),
  (Float32, 0.3, 0.5, 1.5, 0.25, 0.001, 0.0f0),
  (Float32, 0.3, 0.5, 1.5, 0.25, -0.35, -0.2111111f0),
  (Float32, 0.3, 0.5, 1.5, 1.0, -2.5, -1.3666667f0),
  (Float32, 0.3, 0.5, 1.5, 1.0, 0.7, 0.16666664f0),
  (Float32, 0.3, 0.5, 1.5, 1.0, 4.0, 2.3666666f0),
  (Float32, 0.3, 0.5, 1.5, 1.0, 0.001, 0.0f0),
  (Float32, 0.3, 0.5, 1.5, 1.0, -0.35, 0.0f0),
  (Float32, 0.3, 0.5, 1.5, 3.0, -2.5, -0.45999998f0),
  (Float32, 0.3, 0.5, 1.5, 3.0, 0.7, 0.0f0),
  (Float32, 0.3, 0.5, 1.5, 3.0, 4.0, 1.0600001f0),
  (Float32, 0.3, 0.5, 1.5, 3.0, 0.001, 0.0f0),
  (Float32, 0.3, 0.5, 1.5, 3.0, -0.35, 0.0f0),
  (Float32, 1.7, 2.0, nothing, 0.25, -2.5, -1.3833333f0),
  (Float32, 1.7, 2.0, nothing, 0.25, 0.7, 0.18333332f0),
  (Float32, 1.7, 2.0, nothing, 0.25, 4.0, 2.3833334f0),
  (Float32, 1.7, 2.0, nothing, 0.25, 0.001, 0.0f0),
  (Float32, 1.7, 2.0, nothing, 0.25, -0.35, 0.0f0),
  (Float32, 1.7, 2.0, nothing, 1.0, -2.5, -0.26666665f0),
  (Float32, 1.7, 2.0, nothing, 1.0, 0.7, 0.0f0),
  (Float32, 1.7, 2.0, nothing, 1.0, 4.0, 0.76666665f0),
  (Float32, 1.7, 2.0, nothing, 1.0, 0.001, 0.0f0),
  (Float32, 1.7, 2.0, nothing, 1.0, -0.35, 0.0f0),
  (Float32, 1.7, 2.0, nothing, 3.0, -2.5, 0.0f0),
  (Float32, 1.7, 2.0, nothing, 3.0, 0.7, 0.0f0),
  (Float32, 1.7, 2.0, nothing, 3.0, 4.0, 0.0f0),
  (Float32, 1.7, 2.0, nothing, 3.0, 0.001, 0.0f0),
  (Float32, 1.7, 2.0, nothing, 3.0, -0.35, 0.0f0),
  (Float32, 1.7, 2.0, 1.5, 0.25, -2.5, -1.2416667f0),
  (Float32, 1.7, 2.0, 1.5, 0.25, 0.7, 0.041666627f0),
  (Float32, 1.7, 2.0, 1.5, 0.25, 4.0, 2.2416666f0),
  (Float32, 1.7, 2.0, 1.5, 0.25, 0.001, 0.0f0),
  (Float32, 1.7, 2.0, 1.5, 0.25, -0.35, 0.0f0),
  (Float32, 1.7, 2.0, 1.5, 1.0, -2.5, 0.0f0),
  (Float32, 1.7, 2.0, 1.5, 1.0, 0.7, 0.0f0),
  (Float32, 1.7, 2.0, 1.5, 1.0, 4.0, 0.48333326f0),
  (Float32, 1.7, 2.0, 1.5, 1.0, 0.001, 0.0f0),
  (Float32, 1.7, 2.0, 1.5, 1.0, -0.35, 0.0f0),
  (Float32, 1.7, 2.0, 1.5, 3.0, -2.5, 0.0f0),
  (Float32, 1.7, 2.0, 1.5, 3.0, 0.7, 0.0f0),
  (Float32, 1.7, 2.0, 1.5, 3.0, 4.0, 0.0f0),
  (Float32, 1.7, 2.0, 1.5, 3.0, 0.001, 0.0f0),
  (Float32, 1.7, 2.0, 1.5, 3.0, -0.35, 0.0f0),
  (Float32, 0.0, 0.5, nothing, 0.25, -2.5, -2.2222223f0),
  (Float32, 0.0, 0.5, nothing, 0.25, 0.7, 0.6222222f0),
  (Float32, 0.0, 0.5, nothing, 0.25, 4.0, 3.5555556f0),
  (Float32, 0.0, 0.5, nothing, 0.25, 0.001, 0.00088888896f0),
  (Float32, 0.0, 0.5, nothing, 0.25, -0.35, -0.3111111f0),
  (Float32, 0.0, 0.5, nothing, 1.0, -2.5, -1.6666666f0),
  (Float32, 0.0, 0.5, nothing, 1.0, 0.7, 0.46666667f0),
  (Float32, 0.0, 0.5, nothing, 1.0, 4.0, 2.6666667f0),
  (Float32, 0.0, 0.5, nothing, 1.0, 0.001, 0.0006666667f0),
  (Float32, 0.0, 0.5, nothing, 1.0, -0.35, -0.23333333f0),
  (Float32, 0.0, 0.5, nothing, 3.0, -2.5, -1.0f0),
  (Float32, 0.0, 0.5, nothing, 3.0, 0.7, 0.28f0),
  (Float32, 0.0, 0.5, nothing, 3.0, 4.0, 1.6f0),
  (Float32, 0.0, 0.5, nothing, 3.0, 0.001, 0.00040000002f0),
  (Float32, 0.0, 0.5, nothing, 3.0, -0.35, -0.14f0),
  (Float32, 0.0, 0.5, 1.5, 0.25, -2.5, -2.2222223f0),
  (Float32, 0.0, 0.5, 1.5, 0.25, 0.7, 0.6222222f0),
  (Float32, 0.0, 0.5, 1.5, 0.25, 4.0, 3.5555556f0),
  (Float32, 0.0, 0.5, 1.5, 0.25, 0.001, 0.00088888896f0),
  (Float32, 0.0, 0.5, 1.5, 0.25, -0.35, -0.3111111f0),
  (Float32, 0.0, 0.5, 1.5, 1.0, -2.5, -1.6666666f0),
  (Float32, 0.0, 0.5, 1.5, 1.0, 0.7, 0.46666667f0),
  (Float32, 0.0, 0.5, 1.5, 1.0, 4.0, 2.6666667f0),
  (Float32, 0.0, 0.5, 1.5, 1.0, 0.001, 0.0006666667f0),
  (Float32, 0.0, 0.5, 1.5, 1.0, -0.35, -0.23333333f0),
  (Float32, 0.0, 0.5, 1.5, 3.0, -2.5, -1.0f0),
  (Float32, 0.0, 0.5, 1.5, 3.0, 0.7, 0.28f0),
  (Float32, 0.0, 0.5, 1.5, 3.0, 4.0, 1.6f0),
  (Float32, 0.0, 0.5, 1.5, 3.0, 0.001, 0.00040000002f0),
  (Float32, 0.0, 0.5, 1.5, 3.0, -0.35, -0.14f0),
  (Float32, 0.3, 0.0, nothing, 0.25, -2.5, -2.425f0),
  (Float32, 0.3, 0.0, nothing, 0.25, 0.7, 0.625f0),
  (Float32, 0.3, 0.0, nothing, 0.25, 4.0, 3.925f0),
  (Float32, 0.3, 0.0, nothing, 0.25, 0.001, 0.0f0),
  (Float32, 0.3, 0.0, nothing, 0.25, -0.35, -0.27499998f0),
  (Float32, 0.3, 0.0, nothing, 1.0, -2.5, -2.2f0),
  (Float32, 0.3, 0.0, nothing, 1.0, 0.7, 0.39999998f0),
  (Float32, 0.3, 0.0, nothing, 1.0, 4.0, 3.7f0),
  (Float32, 0.3, 0.0, nothing, 1.0, 0.001, 0.0f0),
  (Float32, 0.3, 0.0, nothing, 1.0, -0.35, -0.049999982f0),
  (Float32, 0.3, 0.0, nothing, 3.0, -2.5, -1.5999999f0),
  (Float32, 0.3, 0.0, nothing, 3.0, 0.7, 0.0f0),
  (Float32, 0.3, 0.0, nothing, 3.0, 4.0, 3.1f0),
  (Float32, 0.3, 0.0, nothing, 3.0, 0.001, 0.0f0),
  (Float32, 0.3, 0.0, nothing, 3.0, -0.35, 0.0f0),
  (Float32, 0.3, 0.0, 1.5, 0.25, -2.5, -2.3875f0),
  (Float32, 0.3, 0.0, 1.5, 0.25, 0.7, 0.5875f0),
  (Float32, 0.3, 0.0, 1.5, 0.25, 4.0, 3.8875f0),
  (Float32, 0.3, 0.0, 1.5, 0.25, 0.001, 0.0f0),
  (Float32, 0.3, 0.0, 1.5, 0.25, -0.35, -0.23749998f0),
  (Float32, 0.3, 0.0, 1.5, 1.0, -2.5, -2.05f0),
  (Float32, 0.3, 0.0, 1.5, 1.0, 0.7, 0.24999997f0),
  (Float32, 0.3, 0.0, 1.5, 1.0, 4.0, 3.55f0),
  (Float32, 0.3, 0.0, 1.5, 1.0, 0.001, 0.0f0),
  (Float32, 0.3, 0.0, 1.5, 1.0, -0.35, 0.0f0),
  (Float32, 0.3, 0.0, 1.5, 3.0, -2.5, -1.15f0),
  (Float32, 0.3, 0.0, 1.5, 3.0, 0.7, 0.0f0),
  (Float32, 0.3, 0.0, 1.5, 3.0, 4.0, 2.65f0),
  (Float32, 0.3, 0.0, 1.5, 3.0, 0.001, 0.0f0),
  (Float32, 0.3, 0.0, 1.5, 3.0, -0.35, 0.0f0),
  (Float32, 1.0, 1.0, nothing, 0.25, -2.5, -1.8f0),
  (Float32, 1.0, 1.0, nothing, 0.25, 0.7, 0.35999998f0),
  (Float32, 1.0, 1.0, nothing, 0.25, 4.0, 3.0f0),
  (Float32, 1.0, 1.0, nothing, 0.25, 0.001, 0.0f0),
  (Float32, 1.0, 1.0, nothing, 0.25, -0.35, -0.08f0),
  (Float32, 1.0, 1.0, nothing, 1.0, -2.5, -0.75f0),
  (Float32, 1.0, 1.0, nothing, 1.0, 0.7, 0.0f0),
  (Float32, 1.0, 1.0, nothing, 1.0, 4.0, 1.5f0),
  (Float32, 1.0, 1.0, nothing, 1.0, 0.001, 0.0f0),
  (Float32, 1.0, 1.0, nothing, 1.0, -0.35, 0.0f0),
  (Float32, 1.0, 1.0, nothing, 3.0, -2.5, 0.0f0),
  (Float32, 1.0, 1.0, nothing, 3.0, 0.7, 0.0f0),
  (Float32, 1.0, 1.0, nothing, 3.0, 4.0, 0.25f0),
  (Float32, 1.0, 1.0, nothing, 3.0, 0.001, 0.0f0),
  (Float32, 1.0, 1.0, nothing, 3.0, -0.35, 0.0f0),
  (Float32, 1.0, 1.0, 1.5, 0.25, -2.5, -1.7f0),
  (Float32, 1.0, 1.0, 1.5, 0.25, 0.7, 0.26f0),
  (Float32, 1.0, 1.0, 1.5, 0.25, 4.0, 2.9f0),
  (Float32, 1.0, 1.0, 1.5, 0.25, 0.001, 0.0f0),
  (Float32, 1.0, 1.0, 1.5, 0.25, -0.35, 0.0f0),
  (Float32, 1.0, 1.0, 1.5, 1.0, -2.5, -0.5f0),
  (Float32, 1.0, 1.0, 1.5, 1.0, 0.7, 0.0f0),
  (Float32, 1.0, 1.0, 1.5, 1.0, 4.0, 1.25f0),
  (Float32, 1.0, 1.0, 1.5, 1.0, 0.001, 0.0f0),
  (Float32, 1.0, 1.0, 1.5, 1.0, -0.35, 0.0f0),
  (Float32, 1.0, 1.0, 1.5, 3.0, -2.5, 0.0f0),
  (Float32, 1.0, 1.0, 1.5, 3.0, 0.7, 0.0f0),
  (Float32, 1.0, 1.0, 1.5, 3.0, 4.0, 0.0f0),
  (Float32, 1.0, 1.0, 1.5, 3.0, 0.001, 0.0f0),
  (Float32, 1.0, 1.0, 1.5, 3.0, -0.35, 0.0f0),
  (Float64, 0.3, 0.5, nothing, 0.25, -2.5, -2.1555555555555554),
  (Float64, 0.3, 0.5, nothing, 0.25, 0.7, 0.5555555555555556),
  (Float64, 0.3, 0.5, nothing, 0.25, 4.0, 3.488888888888889),
  (Float64, 0.3, 0.5, nothing, 0.25, 0.001, 0.0),
  (Float64, 0.3, 0.5, nothing, 0.25, -0.35, -0.2444444444444444),
  (Float64, 0.3, 0.5, nothing, 1.0, -2.5, -1.4666666666666668),
  (Float64, 0.3, 0.5, nothing, 1.0, 0.7, 0.26666666666666666),
  (Float64, 0.3, 0.5, nothing, 1.0, 4.0, 2.466666666666667),
  (Float64, 0.3, 0.5, nothing, 1.0, 0.001, 0.0),
  (Float64, 0.3, 0.5, nothing, 1.0, -0.35, -0.033333333333333326),
  (Float64, 0.3, 0.5, nothing, 3.0, -2.5, -0.64),
  (Float64, 0.3, 0.5, nothing, 3.0, 0.7, 0.0),
  (Float64, 0.3, 0.5, nothing, 3.0, 4.0, 1.24),
  (Float64, 0.3, 0.5, nothing, 3.0, 0.001, 0.0),
  (Float64, 0.3, 0.5, nothing, 3.0, -0.35, 0.0),
  (Float64, 0.3, 0.5, 1.5, 0.25, -2.5, -2.1222222222222222),
  (Float64, 0.3, 0.5, 1.5, 0.25, 0.7, 0.5222222222222221),
  (Float64, 0.3, 0.5, 1.5, 0.25, 4.0, 3.4555555555555557),
  (Float64, 0.3, 0.5, 1.5, 0.25, 0.001, 0.0),
  (Float64, 0.3, 0.5, 1.5, 0.25, -0.35, -0.2111111111111111),
  (Float64, 0.3, 0.5, 1.5, 1.0, -2.5, -1.3666666666666665),
  (Float64, 0.3, 0.5, 1.5, 1.0, 0.7, 0.16666666666666666),
  (Float64, 0.3, 0.5, 1.5, 1.0, 4.0, 2.3666666666666667),
  (Float64, 0.3, 0.5, 1.5, 1.0, 0.001, 0.0),
  (Float64, 0.3, 0.5, 1.5, 1.0, -0.35, 0.0),
  (Float64, 0.3, 0.5, 1.5, 3.0, -2.5, -0.4600000000000001),
  (Float64, 0.3, 0.5, 1.5, 3.0, 0.7, 0.0),
  (Float64, 0.3, 0.5, 1.5, 3.0, 4.0, 1.06),
  (Float64, 0.3, 0.5, 1.5, 3.0, 0.001, 0.0),
  (Float64, 0.3, 0.5, 1.5, 3.0, -0.35, 0.0),
  (Float64, 1.7, 2.0, nothing, 0.25, -2.5, -1.3833333333333335),
  (Float64, 1.7, 2.0, nothing, 0.25, 0.7, 0.18333333333333332),
  (Float64, 1.7, 2.0, nothing, 0.25, 4.0, 2.3833333333333333),
  (Float64, 1.7, 2.0, nothing, 0.25, 0.001, 0.0),
  (Float64, 1.7, 2.0, nothing, 0.25, -0.35, 0.0),
  (Float64, 1.7, 2.0, nothing, 1.0, -2.5, -0.26666666666666666),
  (Float64, 1.7, 2.0, nothing, 1.0, 0.7, 0.0),
  (Float64, 1.7, 2.0, nothing, 1.0, 4.0, 0.7666666666666666),
  (Float64, 1.7, 2.0, nothing, 1.0, 0.001, 0.0),
  (Float64, 1.7, 2.0, nothing, 1.0, -0.35, 0.0),
  (Float64, 1.7, 2.0, nothing, 3.0, -2.5, 0.0),
  (Float64, 1.7, 2.0, nothing, 3.0, 0.7, 0.0),
  (Float64, 1.7, 2.0, nothing, 3.0, 4.0, 0.0),
  (Float64, 1.7, 2.0, nothing, 3.0, 0.001, 0.0),
  (Float64, 1.7, 2.0, nothing, 3.0, -0.35, 0.0),
  (Float64, 1.7, 2.0, 1.5, 0.25, -2.5, -1.2416666666666667),
  (Float64, 1.7, 2.0, 1.5, 0.25, 0.7, 0.041666666666666664),
  (Float64, 1.7, 2.0, 1.5, 0.25, 4.0, 2.2416666666666667),
  (Float64, 1.7, 2.0, 1.5, 0.25, 0.001, 0.0),
  (Float64, 1.7, 2.0, 1.5, 0.25, -0.35, 0.0),
  (Float64, 1.7, 2.0, 1.5, 1.0, -2.5, 0.0),
  (Float64, 1.7, 2.0, 1.5, 1.0, 0.7, 0.0),
  (Float64, 1.7, 2.0, 1.5, 1.0, 4.0, 0.4833333333333334),
  (Float64, 1.7, 2.0, 1.5, 1.0, 0.001, 0.0),
  (Float64, 1.7, 2.0, 1.5, 1.0, -0.35, 0.0),
  (Float64, 1.7, 2.0, 1.5, 3.0, -2.5, 0.0),
  (Float64, 1.7, 2.0, 1.5, 3.0, 0.7, 0.0),
  (Float64, 1.7, 2.0, 1.5, 3.0, 4.0, 0.0),
  (Float64, 1.7, 2.0, 1.5, 3.0, 0.001, 0.0),
  (Float64, 1.7, 2.0, 1.5, 3.0, -0.35, 0.0),
  (Float64, 0.0, 0.5, nothing, 0.25, -2.5, -2.2222222222222223),
  (Float64, 0.0, 0.5, nothing, 0.25, 0.7, 0.6222222222222222),
  (Float64, 0.0, 0.5, nothing, 0.25, 4.0, 3.5555555555555554),
  (Float64, 0.0, 0.5, nothing, 0.25, 0.001, 0.0008888888888888889),
  (Float64, 0.0, 0.5, nothing, 0.25, -0.35, -0.3111111111111111),
  (Float64, 0.0, 0.5, nothing, 1.0, -2.5, -1.6666666666666667),
  (Float64, 0.0, 0.5, nothing, 1.0, 0.7, 0.4666666666666666),
  (Float64, 0.0, 0.5, nothing, 1.0, 4.0, 2.6666666666666665),
  (Float64, 0.0, 0.5, nothing, 1.0, 0.001, 0.0006666666666666666),
  (Float64, 0.0, 0.5, nothing, 1.0, -0.35, -0.2333333333333333),
  (Float64, 0.0, 0.5, nothing, 3.0, -2.5, -1.0),
  (Float64, 0.0, 0.5, nothing, 3.0, 0.7, 0.27999999999999997),
  (Float64, 0.0, 0.5, nothing, 3.0, 4.0, 1.6),
  (Float64, 0.0, 0.5, nothing, 3.0, 0.001, 0.0004),
  (Float64, 0.0, 0.5, nothing, 3.0, -0.35, -0.13999999999999999),
  (Float64, 0.0, 0.5, 1.5, 0.25, -2.5, -2.2222222222222223),
  (Float64, 0.0, 0.5, 1.5, 0.25, 0.7, 0.6222222222222222),
  (Float64, 0.0, 0.5, 1.5, 0.25, 4.0, 3.5555555555555554),
  (Float64, 0.0, 0.5, 1.5, 0.25, 0.001, 0.0008888888888888889),
  (Float64, 0.0, 0.5, 1.5, 0.25, -0.35, -0.3111111111111111),
  (Float64, 0.0, 0.5, 1.5, 1.0, -2.5, -1.6666666666666667),
  (Float64, 0.0, 0.5, 1.5, 1.0, 0.7, 0.4666666666666666),
  (Float64, 0.0, 0.5, 1.5, 1.0, 4.0, 2.6666666666666665),
  (Float64, 0.0, 0.5, 1.5, 1.0, 0.001, 0.0006666666666666666),
  (Float64, 0.0, 0.5, 1.5, 1.0, -0.35, -0.2333333333333333),
  (Float64, 0.0, 0.5, 1.5, 3.0, -2.5, -1.0),
  (Float64, 0.0, 0.5, 1.5, 3.0, 0.7, 0.27999999999999997),
  (Float64, 0.0, 0.5, 1.5, 3.0, 4.0, 1.6),
  (Float64, 0.0, 0.5, 1.5, 3.0, 0.001, 0.0004),
  (Float64, 0.0, 0.5, 1.5, 3.0, -0.35, -0.13999999999999999),
  (Float64, 0.3, 0.0, nothing, 0.25, -2.5, -2.425),
  (Float64, 0.3, 0.0, nothing, 0.25, 0.7, 0.625),
  (Float64, 0.3, 0.0, nothing, 0.25, 4.0, 3.925),
  (Float64, 0.3, 0.0, nothing, 0.25, 0.001, 0.0),
  (Float64, 0.3, 0.0, nothing, 0.25, -0.35, -0.27499999999999997),
  (Float64, 0.3, 0.0, nothing, 1.0, -2.5, -2.2),
  (Float64, 0.3, 0.0, nothing, 1.0, 0.7, 0.39999999999999997),
  (Float64, 0.3, 0.0, nothing, 1.0, 4.0, 3.7),
  (Float64, 0.3, 0.0, nothing, 1.0, 0.001, 0.0),
  (Float64, 0.3, 0.0, nothing, 1.0, -0.35, -0.04999999999999999),
  (Float64, 0.3, 0.0, nothing, 3.0, -2.5, -1.6),
  (Float64, 0.3, 0.0, nothing, 3.0, 0.7, 0.0),
  (Float64, 0.3, 0.0, nothing, 3.0, 4.0, 3.1),
  (Float64, 0.3, 0.0, nothing, 3.0, 0.001, 0.0),
  (Float64, 0.3, 0.0, nothing, 3.0, -0.35, 0.0),
  (Float64, 0.3, 0.0, 1.5, 0.25, -2.5, -2.3875),
  (Float64, 0.3, 0.0, 1.5, 0.25, 0.7, 0.5874999999999999),
  (Float64, 0.3, 0.0, 1.5, 0.25, 4.0, 3.8875),
  (Float64, 0.3, 0.0, 1.5, 0.25, 0.001, 0.0),
  (Float64, 0.3, 0.0, 1.5, 0.25, -0.35, -0.2375),
  (Float64, 0.3, 0.0, 1.5, 1.0, -2.5, -2.05),
  (Float64, 0.3, 0.0, 1.5, 1.0, 0.7, 0.25),
  (Float64, 0.3, 0.0, 1.5, 1.0, 4.0, 3.55),
  (Float64, 0.3, 0.0, 1.5, 1.0, 0.001, 0.0),
  (Float64, 0.3, 0.0, 1.5, 1.0, -0.35, 0.0),
  (Float64, 0.3, 0.0, 1.5, 3.0, -2.5, -1.1500000000000001),
  (Float64, 0.3, 0.0, 1.5, 3.0, 0.7, 0.0),
  (Float64, 0.3, 0.0, 1.5, 3.0, 4.0, 2.6500000000000004),
  (Float64, 0.3, 0.0, 1.5, 3.0, 0.001, 0.0),
  (Float64, 0.3, 0.0, 1.5, 3.0, -0.35, 0.0),
  (Float64, 1.0, 1.0, nothing, 0.25, -2.5, -1.8),
  (Float64, 1.0, 1.0, nothing, 0.25, 0.7, 0.36),
  (Float64, 1.0, 1.0, nothing, 0.25, 4.0, 3.0),
  (Float64, 1.0, 1.0, nothing, 0.25, 0.001, 0.0),
  (Float64, 1.0, 1.0, nothing, 0.25, -0.35, -0.07999999999999999),
  (Float64, 1.0, 1.0, nothing, 1.0, -2.5, -0.75),
  (Float64, 1.0, 1.0, nothing, 1.0, 0.7, 0.0),
  (Float64, 1.0, 1.0, nothing, 1.0, 4.0, 1.5),
  (Float64, 1.0, 1.0, nothing, 1.0, 0.001, 0.0),
  (Float64, 1.0, 1.0, nothing, 1.0, -0.35, 0.0),
  (Float64, 1.0, 1.0, nothing, 3.0, -2.5, 0.0),
  (Float64, 1.0, 1.0, nothing, 3.0, 0.7, 0.0),
  (Float64, 1.0, 1.0, nothing, 3.0, 4.0, 0.25),
  (Float64, 1.0, 1.0, nothing, 3.0, 0.001, 0.0),
  (Float64, 1.0, 1.0, nothing, 3.0, -0.35, 0.0),
  (Float64, 1.0, 1.0, 1.5, 0.25, -2.5, -1.7),
  (Float64, 1.0, 1.0, 1.5, 0.25, 0.7, 0.25999999999999995),
  (Float64, 1.0, 1.0, 1.5, 0.25, 4.0, 2.9),
  (Float64, 1.0, 1.0, 1.5, 0.25, 0.001, 0.0),
  (Float64, 1.0, 1.0, 1.5, 0.25, -0.35, 0.0),
  (Float64, 1.0, 1.0, 1.5, 1.0, -2.5, -0.5),
  (Float64, 1.0, 1.0, 1.5, 1.0, 0.7, 0.0),
  (Float64, 1.0, 1.0, 1.5, 1.0, 4.0, 1.25),
  (Float64, 1.0, 1.0, 1.5, 1.0, 0.001, 0.0),
  (Float64, 1.0, 1.0, 1.5, 1.0, -0.35, 0.0),
  (Float64, 1.0, 1.0, 1.5, 3.0, -2.5, 0.0),
  (Float64, 1.0, 1.0, 1.5, 3.0, 0.7, 0.0),
  (Float64, 1.0, 1.0, 1.5, 3.0, 4.0, 0.0),
  (Float64, 1.0, 1.0, 1.5, 3.0, 0.001, 0.0),
  (Float64, 1.0, 1.0, 1.5, 3.0, -0.35, 0.0),
]
const ARCHIVED_VALUES = [
  # (T, λ1, λ2, weighted, archived value at x = [-2.5, 0.7, 4.0, 0.001, -0.35, 0.0], w = [1.5, 0.5, 2.0, 1.0, 0.0, 3.0])
  (Float32, 0.3, 0.5, false, 7.980925f0),
  (Float32, 0.3, 0.5, true, 9.345925f0),
  (Float32, 1.7, 2.0, false, 35.6992f0),
  (Float32, 1.7, 2.0, true, 43.4342f0),
  (Float32, 0.0, 0.5, false, 5.7156253f0),
  (Float32, 0.0, 0.5, true, 5.7156253f0),
  (Float32, 0.3, 0.0, false, 2.2653f0),
  (Float32, 0.3, 0.0, true, 3.6303f0),
  (Float64, 0.3, 0.5, false, 7.98092525),
  (Float64, 0.3, 0.5, true, 9.34592525),
  (Float64, 1.7, 2.0, false, 35.699200999999995),
  (Float64, 1.7, 2.0, true, 43.434201),
  (Float64, 0.0, 0.5, false, 5.7156252499999995),
  (Float64, 0.0, 0.5, true, 5.7156252499999995),
  (Float64, 0.3, 0.0, false, 2.2653),
  (Float64, 0.3, 0.0, true, 3.6303),
]
# warmed allocations of the dense oracles (criterion A)
alloc_prox(y, f, x, γ) = @allocated prox!(y, f, x, γ)
alloc_vprox(y, f, x, γ) = @allocated value_and_prox!(y, f, x, γ)
alloc_value(f, x) = @allocated f(x)
function loop_prox_block(f, x, γ)
  s = zero(eltype(x))
  @inbounds for i in eachindex(x)
    s += prox_block(f, x[i], i, γ)
  end
  s
end
alloc_loop(f, x, γ) = @allocated loop_prox_block(f, x, γ)
@testset "ElasticNet" begin
  @testset "constructor, traits and parameters" begin
    f = ElasticNet(1, 0.5)
    @test f isa ElasticNet{Float64, Nothing} && f.λ1 === 1.0 && f.λ2 === 0.5 && f.w === nothing
    w = [1.0, 0.0, Inf]
    g = ElasticNet(0.25f0, 2.0f0, w)
    @test g.λ1 === 0.25f0 && g.λ2 === 2.0f0 && g.w === w
    @test isconvex(g) && isseparable(g) && isproximable(g) && !issmooth(g) && iszeropreserving(g)
    @test iszeropreserving(ElasticNet(Inf, Inf, w))
    @test PB.entry_parameters(g) === (:w => w,)
    for (λ1, λ2) in ((-1.0, 1.0), (1.0, -1.0), (NaN, 1.0), (1.0, NaN))
      @test_throws ArgumentError ElasticNet(λ1, λ2)
    end
    @test_throws ArgumentError ElasticNet(1.0, 1.0, [1.0, -1.0])
    @test_throws ArgumentError ElasticNet(1.0, 1.0, [1.0, NaN])
    @test_throws ArgumentError ElasticNet(1.0, 1.0, [1.0 + 0im])
    @test_throws ArgumentError ElasticNet(1.0, 1.0)(ComplexF64[1, 2])
    @test_throws DimensionMismatch ElasticNet(1.0, 1.0, [1.0, 2.0])([1.0, 2.0, 3.0])
    @test_throws DimensionMismatch prox(ElasticNet(1.0, 1.0, [1.0, 2.0]), [1.0, 2.0, 3.0])
    @test_throws ArgumentError lipschitz(g)
    # the vector form of prox_block has no ElasticNet hook
    @test_throws MethodError prox_block(ElasticNet(1.0, 1.0), [1.0, 2.0], 1, 1.0)
  end

  for T in (Float32, Float64)
    @testset "$T analytic cases" begin
      λ1, λ2, γ = T(0.5), T(2), T(0.25)
      x = T[3, -2, 0.1, -0.05, 0, 1.25, -0.0, 7]
      w = T[1, 2, 0, 0.5, 1, 3, 1, 0.25]
      n = length(x)
      for f in (ElasticNet(λ1, λ2), ElasticNet(λ1, λ2, w))
        y = prox(f, x, γ)
        @test eltype(y) === T
        @test all(within(y[i], proxref(T, f, x[i], i, γ)) for i in 1:n)
        @test all(y[i] === prox_block(f, x[i], i, γ) for i in 1:n)
        @test all(!signbit, y[iszero.(y)])                               # a zero result is +0, also from -0
        # optimality: (x - y)/γ - λ2 y is λ1 wᵢ sign(yᵢ) where yᵢ ≠ 0, and at most λ1 wᵢ in magnitude where yᵢ = 0
        for i in 1:n
          r = big4096(() -> (big(x[i]) - big(y[i])) / big(γ) - big(λ2) * big(y[i]))
          c = big(λ1) * weight(f, i)
          tol = 64 * eps(T) * (1 + abs(x[i])) / γ
          @test iszero(y[i]) ? abs(r) <= c + tol : abs(r - c * sign(y[i])) <= tol
        end
        @test valok(f(x), valref(f, x), n)
        @test valok(T(sum(value_entry(f, x[i], i) for i in 1:n)), valref(f, x), n)
        # in place, out of place over a nonzero output, the allocating form and the value at the stored point
        z = fill(T(99), n)
        @test prox!(z, f, x, γ) === z && z == y
        a = copy(x)
        @test prox!(a, f, a, γ) === a && a == y
        z .= T(99)
        v = value_and_prox!(z, f, x, γ)
        @test z == y && v === f(z)
        a .= x
        @test value_and_prox!(a, f, a, γ) === f(y) && a == y
        @test prox!(z, f, x) == prox(f, x, 1)                            # the default step is one
      end
      # a singleton, a matrix with matrix weights, and views
      @test prox(ElasticNet(λ1, λ2), T[3], γ) == [prox_block(ElasticNet(λ1, λ2), T(3), 1, γ)]
      X, W = reshape(x, 2, 4), reshape(w, 2, 4)
      @test prox(ElasticNet(λ1, λ2, W), X, γ) == reshape(prox(ElasticNet(λ1, λ2, w), x, γ), 2, 4)
      @test valok(ElasticNet(λ1, λ2, W)(X), valref(ElasticNet(λ1, λ2, w), x), n)
      u = vcat(T[5], x, T[5])
      @test prox(ElasticNet(λ1, λ2, w), view(u, 2:n+1), γ) == prox(ElasticNet(λ1, λ2, w), x, γ)
      @test valok(ElasticNet(λ1, λ2, w)(view(u, 2:n+1)), valref(ElasticNet(λ1, λ2, w), x), n)
      # exact threshold ties give +0: γλ1 = 1 at v = ±1
      tie = ElasticNet(T(0.5), λ2)
      @test prox_block(tie, one(T), 1, T(2)) === zero(T) && prox_block(tie, -one(T), 1, T(2)) === zero(T)
      @test prox_block(tie, -zero(T), 1, T(2)) === zero(T) && prox_block(ElasticNet(zero(T), λ2), -zero(T), 1, γ) === zero(T)
      # λ2 = 0 is NormL1 bit for bit, at the steps 0, 1/4, 1 and Inf and the trials below
      trials = T[3, -3, 0.2, -0.2, 0, -0.0, Inf, -Inf, NaN]
      for ww in (nothing, T[0, 1, 2, Inf, 0.5, 1, 3, 1, 1]), s in T[0, 0.25, 1, Inf]
        f, g = ElasticNet(λ1, zero(T), ww), NormL1(λ1, ww)
        @test all(isequal(prox_block(f, trials[i], i, s), prox_block(g, trials[i], i, s)) for i in eachindex(trials))
      end
      @test ElasticNet(λ1, zero(T), w)(x) === NormL1(λ1, w)(x)
      @test ElasticNet(λ1, zero(T), w)(T[NaN, 1, 1, 1, 1, 1, 1, 1]) |> isnan
      # λ1 = 0, or a zero weight, is the ridge SqrNormL2(λ2) bit for bit, except that -0 gives +0
      for s in T[0.25, 1, 4], v in T[3, -3, 0.2, 1e-3, 7]
        r = prox_block(SqrNormL2(λ2), v, 1, s)
        @test prox_block(ElasticNet(zero(T), λ2), v, 1, s) === r
        @test prox_block(ElasticNet(zero(T), λ2, T[2]), v, 1, s) === r
        @test prox_block(ElasticNet(λ1, λ2, T[0]), v, 1, s) === r
      end
      @test valok(ElasticNet(zero(T), λ2)(x), valref(ElasticNet(zero(T), λ2), x), n)
      # both levels zero: the identity, except that -0 gives +0, and the value 0, also at NaN and ±Inf entries
      f00 = ElasticNet(zero(T), zero(T), T[1, 2, 0])
      @test isequal(prox(f00, T[NaN, -Inf, -0.0], γ), T[NaN, -Inf, 0])
      @test prox(f00, T[3, -2, 0.5], γ) == T[3, -2, 0.5]
      @test f00(T[NaN, Inf, -Inf]) === zero(T)
      # integer arguments and levels
      @test prox(ElasticNet(1, 2), [3, -1, 0], 1) ≈ [2 / 3, 0, 0]
      @test ElasticNet(1, 2)([3, -1, 0]) === 14.0
    end
  end

  for T in (Float32, Float64)
    @testset "$T scalar steps are checked before writes (D1)" begin
      f = ElasticNet(T(1), T(1), T[1, 2])
      x = T[3, -2]
      for γ in (-one(T), T(NaN), -T(Inf), -T(0.5))
        y = T[7, 8]
        @test_throws ArgumentError prox!(y, f, x, γ)
        @test y == T[7, 8]
        @test_throws ArgumentError value_and_prox!(y, f, x, γ)
        @test y == T[7, 8]
        a = copy(x)
        @test_throws ArgumentError prox!(a, f, a, γ)
        @test a == x
        @test_throws ArgumentError value_and_prox!(a, f, a, γ)
        @test a == x
        @test_throws ArgumentError prox(f, x, γ)
        s = SparseIterate(x); s0 = copy(s)
        @test_throws ArgumentError prox!(s, f, s, γ)
        @test Vector(s) == Vector(s0) && collect(activecoordinates(s)) == collect(activecoordinates(s0))
        b = BlockIterate(copy(x), [[1], [2]])
        @test_throws ArgumentError prox!(b, f, b, γ)
        @test Vector(b) == x
        S = SymmetricSparseIterate(T[1 2; 2 0]); S0 = copy(S)
        fs = ElasticNet(T(1), T(1), ones(T, 2, 2))
        @test_throws ArgumentError prox!(S, fs, S, γ)
        @test_throws ArgumentError value_and_prox!(S, fs, S, γ)
        @test_throws ArgumentError prox(fs, S, γ)
        @test Matrix(S) == Matrix(S0) && collect(activecoordinates(S)) == collect(activecoordinates(S0))
      end
      # a zero step is accepted, -0 included: the identity, except that -0 gives +0
      @test prox(f, T[3, -0.0], zero(T)) == T[3, 0] && prox(f, x, -zero(T)) == x
      @test prox(f, x, T(Inf)) == T[0, 0]
      # the entries of an array of steps are not read for this check (a documented precondition)
      @test prox(f, x, T[-1, 1]) isa Vector{T}
      # prox_block does not check, and Scaled keeps its own step rule: Inf * f at γ = 0 forms Inf⋅0 = NaN (#311)
      @test prox_block(f, T(3), 1, -one(T)) isa T
      @test all(isnan, prox(T(Inf) * ElasticNet(T(1), T(1)), x, zero(T)))
    end
  end

  for T in (Float32, Float64)
    @testset "$T representations" begin
      λ1, λ2, γ = T(0.4), T(1.5), T(0.5)
      n = 10
      w = T[1, 2, 0, 0.5, Inf, 1, 3, 1, 0.25, 2]
      f = ElasticNet(λ1, λ2, w)
      # SparseIterate: stored entries only; in place, a stored ±0 stays stored as +0; absent entries are not activated
      x = SparseIterate(T, n)
      x[2] = 3; x[4] = -1; x[9] = T(0.1)
      activate!(x, 5); activate!(x, 7); x[7] = -zero(T)
      dense = Vector(x)
      y = prox(f, x, γ)                                                    # a new output stores the nonzero results
      @test y isa SparseIterate{T} && Vector(y) == prox(f, dense, γ)
      @test sort(collect(activecoordinates(y))) == [2, 4, 9]
      @test valok(f(x), valref(f, dense), n)
      out = SparseIterate(T, n); out[1] = 5                              # a stored entry of the output stays stored
      @test value_and_prox!(out, f, x, γ) === f(out) && Vector(out) == Vector(y) && isactive(out, 1) && out[1] === zero(T)
      @test prox!(x, f, x, γ) === x && Vector(x) == Vector(y)              # in place, every stored entry stays stored
      @test sort(collect(activecoordinates(x))) == [2, 4, 5, 7, 9] && x[7] === zero(T) && x[5] === zero(T)
      # array steps on dense and sparse arguments
      steps = T(0.25) .+ T.(1:n) ./ 8
      ys = prox(f, dense, steps)
      @test all(within(ys[i], proxref(T, f, dense[i], i, steps[i])) for i in 1:n)
      @test Vector(prox(f, SparseIterate(dense), steps)) == ys
      # BlockIterate: inactive groups stay inactive and zero
      groups = [[1, 3], [2, 4], [5, 6, 7], [8, 9, 10]]
      b = BlockIterate(T[1, -2, 3, 0.5, 0, 0, 0, 2, 0, -1], groups)
      @test sort(collect(activecoordinates(b))) == [1, 2, 4]
      yb = prox(f, b, γ)
      @test Vector(yb) == prox(f, Vector(b), γ) && sort(collect(activecoordinates(yb))) == [1, 2, 4]
      @test valok(f(b), valref(f, Vector(b)), n)
      vb = copy(b)
      @test value_and_prox!(vb, f, b, γ) === f(vb) && Vector(vb) == Vector(yb)
      @test Vector(prox(f, b, steps)) == prox(f, Vector(b), steps)
      # SymmetricSparseIterate: weights read at both positions, an off-diagonal coordinate counted twice in the value
      p = 4
      Wm = T[1 2 0 1; 2 1 3 0.5; 0 3 2 1; 1 0.5 1 4]
      S = SymmetricSparseIterate(T, p)
      S[1, 1] = 2; S[3, 1] = -1.5; S[4, 2] = T(0.2); S[4, 4] = -3
      fs = ElasticNet(λ1, λ2, Wm)
      @test PB.check_symmetric_parameters(fs, S)
      M = Matrix(S)
      @test valok(fs(S), valref(fs, M), p^2)
      @test fs(S) ≈ λ1 * sum(Wm .* abs.(M)) + λ2 / 2 * sum(abs2, M)
      YS = prox(fs, S, γ)
      @test YS isa SymmetricSparseIterate{T} && Matrix(YS) == prox(fs, M, γ)
      @test sort(collect(activecoordinates(YS))) == sort(collect(activecoordinates(S)))
      @test prox!(zeros(T, p, p), fs, S, γ) == prox(fs, M, γ)            # a dense output from a symmetric argument
      ZS = copy(S)
      @test value_and_prox!(ZS, fs, S, γ) === fs(ZS) && Matrix(ZS) == Matrix(YS)
      # the explicit check rejects asymmetric weights; the oracles do not compare them
      bad = copy(Wm); bad[2, 1] = 7
      @test_throws ArgumentError PB.check_symmetric_parameters(ElasticNet(λ1, λ2, bad), S)
      @test PB.check_symmetric_parameters(ElasticNet(λ1, λ2, Symmetric(bad, :L)), S)
      # a weight array of the wrong length is rejected before anything is written
      S0 = copy(S); short = ElasticNet(λ1, λ2, ones(T, 3, 3))
      @test_throws DimensionMismatch prox!(S, short, S, γ)
      @test_throws DimensionMismatch value_and_prox!(S, short, S, γ)
      @test_throws DimensionMismatch short(S)
      @test Matrix(S) == Matrix(S0) && collect(activecoordinates(S)) == collect(activecoordinates(S0))
      yd = fill(T(9), n)
      @test_throws DimensionMismatch prox!(yd, ElasticNet(λ1, λ2, ones(T, 3)), dense, γ)
      @test yd == fill(T(9), n)
      # unsupported: array steps with a symmetric output, or with a dense output and a symmetric argument
      @test_throws ArgumentError prox!(S, fs, S, ones(T, p, p))
      @test_throws ArgumentError value_and_prox!(S, fs, S, ones(T, p, p))
      @test Matrix(S) == Matrix(S0)
      @test_throws MethodError prox!(zeros(T, p, p), fs, S, ones(T, p, p))
      # an immutable argument is read and left as it was
      r = T(1):T(4)
      @test prox(ElasticNet(λ1, λ2), r, γ) == prox(ElasticNet(λ1, λ2), collect(r), γ) && r == T(1):T(4)
    end
  end

  for T in (Float32, Float64)
    @testset "$T KKT screen and restriction" begin
      w = T[1, 0, 2, Inf]
      f = ElasticNet(T(0.5), T(3), w)
      g = T[0.2, -0.7, 1.5, 4]
      @test kkt_violation(f, g, 1) ≈ T(0.2) - T(0.5)
      @test kkt_violation(f, g, 2) ≈ T(0.7) && kkt_violation(f, g, 3) ≈ T(0.5) && kkt_violation(f, g, 4) == -Inf
      @test kkt_violation(ElasticNet(T(Inf), T(1), w), g, 2) ≈ T(0.7)        # a zero weight is not penalized by the infinite level
      @test kkt_violation(ElasticNet(T(Inf), T(1), w), g, 1) == -Inf
      @test kkt_violation(ElasticNet(T(1), T(Inf)), g, 3) === -T(Inf)          # an infinite ridge level (D2)
      @test kkt_violation(f, g, 1) === kkt_violation(NormL1(T(0.5), w), g, 1)
      # the four-argument forms, with their bounds and extent checks
      x = SparseIterate(T, 4)
      @test kkt_violation(f, g, x, 3) === kkt_violation(f, g, 3)
      @test_throws BoundsError kkt_violation(f, g, x, 5)
      @test_throws DimensionMismatch kkt_violation(f, T[1, 2, 3], x, 1)
      fs = ElasticNet(T(0.5), T(3), T[1 2; 2 1])
      S = SymmetricSparseIterate(T, 2)
      G = T[0.1 3; 3 0.4]
      @test kkt_violation(fs, G, S, 2) ≈ 3 - T(0.5) * 2                          # coordinate (2, 1)
      @test_throws BoundsError kkt_violation(fs, G, S, 4)
      @test_throws DimensionMismatch kkt_violation(fs, T[1, 2, 3], S, 1)
      b = BlockIterate(T, 4, [[1, 3], [2, 4]])
      @test kkt_violation(f, g, b, 1) == max(kkt_violation(f, g, 1), kkt_violation(f, g, 3))
      @test kkt_violation(f, g, b, 2) == max(kkt_violation(f, g, 2), kkt_violation(f, g, 4))
      @test_throws ArgumentError kkt_violation(f, ComplexF64[1, 2, 3, 4], 1)
      # restriction keeps the levels and selects the weights
      r = restrict(f, [3, 1])
      @test r isa ElasticNet && r.λ1 === f.λ1 && r.λ2 === f.λ2 && r.w == T[2, 1]
      @test r(T[1, -2]) == f(T[-2, 0, 1, 0])
      @test restrict(ElasticNet(T(1), T(2)), [2]).w === nothing
    end
  end

  @testset "agreement with the archived implementation" begin
    for (T, λ1, λ2, wv, γ, v, a) in ARCHIVED_PROX
      f = ElasticNet(T(λ1), T(λ2), wv === nothing ? nothing : T[wv])
      @test within(prox_block(f, T(v), 1, T(γ)), a)
    end
    for (T, λ1, λ2, weighted, a) in ARCHIVED_VALUES
      x = T[-2.5, 0.7, 4.0, 0.001, -0.35, 0.0]
      f = ElasticNet(T(λ1), T(λ2), weighted ? T[1.5, 0.5, 2.0, 1.0, 0.0, 3.0] : nothing)
      @test abs(f(x) - a) <= 2 * (length(x) + 2) * eps(T) * abs(a)
    end
  end

  for T in (Float32, Float64)
    @testset "$T allocations" begin
      rng = Xoshiro(908)
      x = randn(rng, T, 1000); y = similar(x); steps = T(0.5) .+ rand(rng, T, 1000)
      for f in (ElasticNet(T(0.3), T(0.5)), ElasticNet(T(0.3), T(0.5), T(0.5) .+ rand(rng, T, 1000)),
                ElasticNet(zero(T), T(0.5)), ElasticNet(T(0.3), zero(T)))
        for _ in 1:2
          alloc_prox(y, f, x, T(0.7)); alloc_prox(y, f, x, steps); alloc_vprox(y, f, x, T(0.7)); alloc_value(f, x); alloc_loop(f, x, T(0.7))
        end
        @test alloc_prox(y, f, x, T(0.7)) == 0
        @test alloc_prox(y, f, x, steps) == 0
        @test alloc_vprox(y, f, x, T(0.7)) == 0
        @test alloc_value(f, x) == 0
        @test alloc_loop(f, x, T(0.7)) == 0
      end
    end
  end
end
end
