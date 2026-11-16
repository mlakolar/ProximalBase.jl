# Historical matrix-coefficient traversal measurements

This record preserves the measurements formerly embedded in
`src/sparse_iterate.jl` at commit
`18b2714955c224300f8924aeb3a205b84c62be57`. They explain an implementation
choice; they are not new measurements, a performance guarantee or a numerical
contract. The supported product domains and numerical behavior are described
in the rendered iterate guide and numerical contract.

For `mul!(out, A, coef, α, β)` with matrix `SparseIterate` coefficients,
stored-entry traversal wins with few coefficients. Column traversal reads
`A` sequentially and matches LinearAlgebra's generic loop for denser
coefficients. The implementation switches at a stored density of `1/32`.

The reported crossover density for `Float64` and random entry order was:

| Rows of `A` | Median crossover density |
|:--|:--|
| 64–512 | 7–16% |
| 1024 | 3–5% |
| 2048 | about 1.5–2% |
| 4096 | 1–5% |

Measurements covered Apple M3, AMD Zen 3 and Intel Skylake-SP. Crossovers
were larger for `Float32` and differed by a factor of two or more among CPUs.
No constant was best everywhere. Averaged over densities of 0.5–50%, the
reported cost of the `1/32` threshold was 2.5–5% relative to the faster
traversal, compared with 1–3% for the best constant (`1/16`). A cutoff in
bytes, `min(1/8, 256/(m*sizeof(T)))`, cost 0.6–1.5%; it was not adopted
because its gain over the best constant was under two percentage points.

The editorial change associated with issue #921 moves this history out of
the source comment without changing the threshold or either traversal.
