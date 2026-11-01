# ChatGPT's review of Draft 7

*Received from the maintainer on 2026-11-01 and recorded verbatim. First the
summary posted in conversation, then the full review file
(`ProximalBase-draft7-review.md`, with its headings demoted one level to nest
under this file's). Both review `13-claude-draft7.md` and `05-decisions.md` at
`93368c1c`.*

---

## Summary as posted

**I would approve Draft 7 as the design baseline and start implementation.** It addresses all six points from the previous review. In particular, it no longer commits the package to a public wrapper API that cannot recover an overflowing argument; it rejects an unrepresentable quadratic model rather than returning one a solver cannot use; and it distinguishes a restricted prox from a slice of the unrestricted prox. The Float16 and zero-scaling rules are also clear enough to implement. **I would not ask for another general rewrite.**   

:chatgpt-content-reference{index="17"}[Download the repository-ready review](sandbox:/mnt/data/ProximalBase-draft7-review.md)

Before publishing it as the 0.6 contract, I would make **one correction to a mathematical guarantee**, tighten two pieces of wording that affect implementation, and take an explicit vote on decision 46.

### 1. The stored matrix need not retain numerical rank `k`

A5 correctly adds destination-rounding error to the rank-projection **residual** bound. But it separately promises that the **stored matrix** has numerical rank at most `k` under the relative A4 cutoff. Ordinary `Float32` storage can violate that second promise even when the projection itself is exact. 

Let \(h=2^{-149}\), the smallest positive `Float32`, and take the `Float64` input \(X=hvv^\top\), with \(v=(1,5/8)\). Since \(X\) is already rank one, its exact rank-one projection is itself. Here is what happens when it is written to `Float32`, with all entries shown in units of \(h\):

\[
\underbrace{\begin{pmatrix}1&5/8\\5/8&25/64\end{pmatrix}}_{\text{exact rank-one result}}
\quad\longrightarrow\quad
\underbrace{\begin{pmatrix}1&1\\1&0\end{pmatrix}}_{\text{stored result}}.
\]

There are **no rounding ties**. The stored matrix’s smaller-to-larger singular-value ratio is approximately **0.382**, far above the A4 cutoff of approximately **0.000345**. It has numerical rank **two**. This is a storage effect, not a LAPACK failure: at subnormal magnitudes, rounding has an absolute error bound, not a uniform relative one. [Oracle Documentation](https://docs.oracle.com/cd/E37069_01/html/E39019/z4000ac019712.html?utm_source=chatgpt.com)

**The edit I would make:** promise that the first discarded singular value of the stored result is bounded by the pre-write rank defect plus the storage perturbation. Promise rank at most `k` under A4 **when that bound is below the cutoff**. Keep the near-optimal residual guarantee. This needs neither a new tolerance floor nor a second SVD on every call. It matters because a solver may ask whether the matrix it received actually passes `IndBallRank(k)` membership.  

### 2. Two wording changes that would prevent wrong implementations

**R3: a normal final product does not prove the ordinary path was safe.** For a weighted L1 threshold, take \(\gamma=3\cdot2^{-538}\), \(\lambda=2^{-538}\), \(w=2^{1022}\), and \(x=2^{-52}\), all `Float64`. The exact threshold is \(3\cdot2^{-54}\), so the prox should retain **\(2^{-54}\)**. But an unprotected `γ*λ` rounds up from three quarters of the smallest subnormal to one whole subnormal. Multiplying by `w` then gives \(4\cdot2^{-54}=x\), and the coordinate is wrongly set to **zero**. Both final thresholds look perfectly normal. **The good news is that your existing `_weight_product` already checks the partial product.** I would change R3 to describe that behavior and add the public regression case, not add another helper.   

**A3/T3: refining a calculation is not the same as requesting a wider model.** The draft suggests retrying a certification failure at higher precision. That is sensible, but V8 shows why the distinction matters: the `Float32` model thresholds at one and returns zero; a new `Float64`-destination call with the original weight thresholds just below one and returns \(2^{-30}\). The latter does not certify the former. I would add one sentence: **“Internal refinement uses wider arithmetic on the effective operands of the original call; it does not change the model or the destination being certified.”** The same principle applies to tests: apply T3’s prescribed conversions before evaluating a `BigFloat` reference, except where R3 explicitly requires range recovery.   

### 3. My vote on decision 46: treat the defining geometry as data

The draft now presents the tradeoff honestly. With the **same stored equation and point**, treating the normal as a parameter gives a projected third coordinate of **zero**; treating it as data gives approximately **one third**. That is a change of model, not a last-bit difference. 

I favor **option (b)**: treat the defining normal and offset of `IndHyperplane`, `IndHalfspace`, and `IndHyperslab` as geometric data. A user choosing between `IndHyperplane(a,b)` and a one-row `IndAffine(A,b)` is choosing how to represent the same constraint. I would preserve that equation in both constructors, without requiring their algorithms to produce identical bits. I would **not** classify every array as data merely to achieve this: option (c) would also change the meaning of array-valued ridge centers, including V9 and the scalar half of V11. The mixed-precision geometry path is the one to benchmark.  

I independently checked the two rounding examples above using exact rational arithmetic and binary floating-point conversions; I did **not** rerun the Julia package. **My stopping point:** settle decision 46, qualify the stored-rank claim, and make the R3/A3 edits in the documentation PR. The ridge, threshold, wrapper, and fused-oracle implementation work does not need to wait.

---

## Full review file

## ProximalBase.jl numerical contract — review of Draft 7

**Recommendation:** Approve the architecture for implementation planning. Do not commission another package-wide rewrite. Before treating the affected clauses as normative: qualify the rank-membership promise at destination underflow, align the ordinary-path predicate with the existing product helper, and distinguish certifying a model from promoting the model. Take a separate maintainer decision on the role of geometric normals and offsets.

**Scope and evidence.** I reviewed [the exact Draft 7 commit](https://github.com/mlakolar/ProximalBase.jl/blob/93368c1c8a2c17bc5aa10ea158d8bf31af468cdb/docs/design/numerical-contract/13-claude-draft7.md), the [decisions record](https://github.com/mlakolar/ProximalBase.jl/blob/93368c1c8a2c17bc5aa10ea158d8bf31af468cdb/docs/design/numerical-contract/05-decisions.md), and the relevant weighted-product source. I independently checked the rank and threshold examples below using exact rational arithmetic and binary floating-point conversions. I did not run the Julia package or reproduce its benchmarks. Draft 7 identifies `3f90a29` as the revision used for the authors’ reported behavioral checks. The document and the checks should not be confused with an approved or shipped 0.6 policy.

### What the revision resolves

| Draft 6 question | Judgment on Draft 7 |
|---|---|
| Can a wrapper recover both an overflowing inner value and an overflowing argument through one public return protocol? | **Resolved.** R1 distinguishes the two. The scaled-value capability stays private for 0.6; transformed arguments need an appropriate identity or scaled-input path. No speculative public API is necessary. |
| Do rank and Stiefel guarantees refer to what is actually written? | **Substantially resolved.** Both now account for storage; tests L3 and L4 capture a real difference. One claim about the numerical rank of the stored matrix still needs a qualification below. |
| May `quadratic_model!` emit a model with an accidentally infinite curvature? | **Resolved for 0.6.** S2/O6 choose an explicit failure before changing the caller’s model. Apply this to the effective coefficient, not the first overflowing intermediate. |
| Which Float16 calls are rejected? | **Resolved.** T6 distinguishes input, destination, direction and data from a level or step that widens exactly. F16a–c test the distinction. |
| Is the restricted prox a slice of the full prox? Can every function be zero-scaled? | **Both resolved.** Q1 distinguishes minimization on a support from slicing the unrestricted minimizer. E4 distinguishes zero scaling on an open domain from the limit of a positive-step prox. |
| Can the choice of representation for a plane change the model? | **Clearly disclosed; policy still awaits a vote.** T2 now shows the difference can be zero versus roughly one third in a coordinate. |

### 1. A5: the stored numerical-rank claim needs a storage-resolution qualification

The near-optimal residual now includes the error of writing to `D`. Keep it. Separately, A5 promises that the stored matrix has numerical rank at most `k` at the relative A4 cutoff. That need not hold after ordinary rounding into a destination's subnormal range, even on the reference CPU with gradual underflow.

Let `h = 2^-149` be the smallest positive Float32, take `v = (1, 5/8)`, and supply `X = h*v*v'` in Float64. The input has exact rank one. The exact rank-one projection is `X` itself. In units of `h`:

```text
X/h                    Float32 storage, then expressed in units of h
[1       5/8  ]        [1  1]
[5/8    25/64 ]        [1  0]
```

There are no rounding ties. The singular-value ratio of the stored matrix is `(3 - sqrt(5))/2 ≈ 0.381966`; the relative cutoff `sqrt(eps(Float32))` is about `0.000345267`. The stored matrix therefore has numerical rank two. An eigensolver error is not needed to make this happen; it is a consequence of writing the answer. Compute the ratio from the exactly rescaled stored entries, not by running a decomposition directly on tiny Float32 entries.

**Suggested rule.** For a stored rank-`k` projection, bound `σ[k+1](Ystored)` by the pre-write rank defect plus the spectral norm of the destination conversion error. Claim numerical rank at most `k` under A4 when this bound is below the documented cutoff. A bound inconsistent with certification is not proof that rank actually exceeds `k`. Do not add a `max(1, σ₁)` floor to rank or require a post-write SVD on every call.

In this example the caller needs to know whether the stored matrix is *near a rank-one answer* or is itself *classified rank one by the public membership test*. Those are distinct promises. Where an O2 fused indicator value describes the stored point, use the same membership semantics as standalone evaluation; a certificate about the pre-write point is not a substitute. Document any intended exception rather than silently returning the ideal indicator value.

[Draft A4–A5](https://github.com/mlakolar/ProximalBase.jl/blob/93368c1c8a2c17bc5aa10ea158d8bf31af468cdb/docs/design/numerical-contract/13-claude-draft7.md#accuracy-and-decisions) · [Oracle: error properties of gradual underflow](https://docs.oracle.com/cd/E37069_01/html/E39019/z4000ac019712.html) · [LAPACK Users’ Guide: SVD error bounds](https://www.netlib.org/lapack/lug/node97.html)

### 2. R3: a normal final product is not evidence that all its bits survived

For a weighted L1 step take the following exact Float64 inputs:

```julia
γ = ldexp(3.0, -538)
λ = ldexp(1.0, -538)
w = ldexp(1.0, 1022)
x = ldexp(1.0, -52)
## Proposed check: prox(NormL1(λ, [w]), [x], γ)[1] == ldexp(1.0, -54)
```

The exact threshold is `3*2^-54`. Evaluating `(γ*λ)*w` without recovery first rounds `3*2^-1076`, or three quarters of the least Float64 subnormal, up to `2^-1074`. The last multiplication therefore produces `4*2^-54 = x`. Both possible thresholds are **finite and normal**; one keeps the coordinate at `2^-54` and the other drops it.

**Suggested rule.** Ordinary left-to-right multiplication defines the effective parameter when no partial operation loses range or subnormal precision needed by a later factor. Otherwise apply R3 to the combined expression. A normal final scalar is not a sufficient fast-path predicate. This does not change V8, where the threshold’s conversion is ordinary normal-range rounding.

The [existing `_weight_product` and `_positive_product`](https://github.com/mlakolar/ProximalBase.jl/blob/93368c1c8a2c17bc5aa10ea158d8bf31af468cdb/src/utils.jl#L13-L64) already inspect the partial product. [Weighted L1’s entrywise method](https://github.com/mlakolar/ProximalBase.jl/blob/93368c1c8a2c17bc5aa10ea158d8bf31af468cdb/src/functions/separable.jl#L243-L257) uses `_weight_product`. This is a reason to match the contract to the sound code pattern and test the public route, not a claim of a reproduced package bug.

### 3. A3 and T3: distinguish refining an answer from changing the model

A wider accumulator is a numerical technique. A new public call with a wider destination can select a different mathematical model. A3 invites the solver to retry at higher precision; T3 specifically exempts wider temporaries for *range recovery*. Extend that sentence to *certification and refinement*, too.

Use V8 as an illustration, not as an example of a search failure. With `x = 1f0`, `λ = 1 - 2^-30` stored in Float64 and unit step, the Float32 model uses threshold one and returns zero. A new Float64-output call uses the original threshold and returns `2^-30`; narrowing that answer back to Float32 does not give zero. A search that is supposed to certify the first model cannot read the original wide parameters, solve the second model, and call the result a certificate for the first.

**Suggested sentence:** “Refinement and certification may widen the arithmetic on the effective operands of the original call, without changing the model, its prescribed range-recovery semantics, or the destination being certified. An explicit wider call is governed by its own model precision.” No new public precision switch or whole-array copy is implied.

### Maintainer vote: decision 46

The current recommendation **(a)** treats affine-set normals and offsets as parameters. It is coherent and documented. My preference is **(b)**: treat the defining normal and offset of `IndHyperplane`, `IndHalfspace` and `IndHyperslab` as geometric data. This keeps a stored linear equation the same when expressed using its dedicated constructor or a one-row affine system. It does not require the two algorithms to return identical bits. Keep centers, penalty weights and steps under their existing rules; benchmark the mixed geometry path instead of assuming the vector loop's cost.

I would not choose **(c), every array is data**, solely to solve this problem. It also changes centered ridge: V9's returned value **and V11's scalar** become `2^-57` instead of zero for the array-valued Float64 center; V11's wide gradient is unchanged. It would also require reconciling T6’s acceptance of an array of Float16 parameters/steps in a wider call with the rejection of Float16 data. Under (a) or (b), retain the one-row plane comparison in the tests and assert the chosen outcome.

[Decision 46 and alternatives](https://github.com/mlakolar/ProximalBase.jl/blob/93368c1c8a2c17bc5aa10ea158d8bf31af468cdb/docs/design/numerical-contract/05-decisions.md#46-the-normals-and-offsets-of-sets-t2)

### Handoff and stopping point

- Build each high-precision reference **after applying the specified model conversions** and stage boundaries, except where R3 requires recovery. A raw BigFloat evaluation on all stored parameters targets the wrong model in V8/V9. Use independent math inside that chosen model.
- Add a mutating counterpart to the allocating Poisson fixture: start with a valid `q`, arrange an overflowing *later* observation, and check the exception and that `q` is unchanged. Include a success control: at `η = 710` with observation weight `1/8` and response zero, the effective coefficient `exp(710)/8` is finite; an overflow in the unweighted exponential alone is not grounds to throw.
- Make W1’s support status a maintainer choice, not an alternative test expectation. If supported, require `M/2`; otherwise mark that combination outside the range guarantee with its tracking issue. Existing private helpers do not need to become third-party API.
- Put the policy choice, test evidence with commit, and quantitative bounds in separate support-table fields. There is no reason to hold up ridge, threshold, or fused-oracle work pending a proof for every spectral method.
