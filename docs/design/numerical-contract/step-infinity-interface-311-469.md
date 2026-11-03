# Design proposal: step endpoints and infinite inputs (#311, #469)

**Status: draft for critical review; documentation only. No implementation in this PR.**
The companion diff in `docs/src/numerics.md` is proposed contract language, not
evidence that current code complies. This proposal is intentionally narrow and
should not be merged until the questions under "Review decisions requested" are
resolved. Leave #311 and #469 open after merging a documentation decision.

References: [numerical contract](../../src/numerics.md) (E1–E4, P1–P2, O2,
O6–O7, C1–C3); [approved decisions](decisions.md) (21, 25, 26);
[#311](https://github.com/mlakolar/ProximalBase.jl/issues/311);
[#469](https://github.com/mlakolar/ProximalBase.jl/issues/469);
[#458](https://github.com/mlakolar/ProximalBase.jl/issues/458) /
[merged #460](https://github.com/mlakolar/ProximalBase.jl/pull/460).

## Why a joint decision is needed

- #311: negative and NaN steps must raise `ArgumentError` before any
  write, including when the argument itself is non-finite or a later block in
  `SeparableSum`, `SliceSum` or `FiberSum` is invalid. The private
  `_checkstep` pre-write protocol arrived in merged #304, but #311 identifies
  missing forwarding/validation for constrained operators and other families.
  P1 already permits selected zero/infinite step **endpoints**, not arbitrary
  nonnegative steps everywhere.
- #469: for bounded balls and spheres, one effective infinite coordinate
  often selects a unique limiting projection, while multiple independently
  infinite coordinates do not. At present the ℓ2 ball/sphere and the
  constrained examples typically return all NaN, and `IndBallL1` can return a
  partially NaN block. Merged #460 deliberately retained all NaN for the
  ℓ1-constrained sphere as a **holding policy**, not a mathematical assertion
  that a single infinity is ambiguous.
- Taken together: `γ = Inf` and `x = Inf` may each have a separately
  documented endpoint, yet the two limits need not commute. "Validate the
  step before using the argument" (E1) settles *invalid* steps but does not
  by itself settle simultaneous **valid** exceptional endpoints.

The existing 47 numerical-contract decisions remain the historical record.
This PR proposes a **limited refinement consistent with** decisions 21 and 26,
not a reinterpretation of unrelated coupled operators or an implementation
of #311.

### Review round 1: changes made in response to critique

These are design revisions, not an implementation or a merge approval.
[Review #508](https://github.com/mlakolar/ProximalBase.jl/pull/508#pullrequestreview-5746338902)
supplied four actionable findings:

1. **Conflicting limit order (E1, decision 21):**
   [review thread](https://github.com/mlakolar/ProximalBase.jl/pull/508#discussion_r4450186138).
   Withdraw the proposed step-endpoint-first rule. Require a
   path-independent joint result for simultaneous admitted exceptional steps
   and infinite inputs. If paths disagree, return NaN rather than pick the
   convenient iterated limit. The two iterated limits are useful witnesses
   of nonuniqueness, but their agreement **alone is insufficient**.
2. **Symmetric ℓ1 multiplicity (M2):**
   [review thread](https://github.com/mlakolar/ProximalBase.jl/pull/508#discussion_r4450186148).
   Correct the one-off-diagonal limit to `±r/2` for `IndBallL1`
   (versus `±r/sqrt(2)` for ℓ2 ball/sphere); the off-diagonal
   coordinate is stored once and counted twice.
3. **Separate rules from open questions:**
   [review thread](https://github.com/mlakolar/ProximalBase.jl/pull/508#discussion_r4450186157).
   Remove undecided NaN/hard-weight and parameter-degenerate questions
   from the proposed normative contract text. Retain them **here** for review.
4. **Convex ball `γ = Inf` endpoint (P1, decision 26):**
   [review thread](https://github.com/mlakolar/ProximalBase.jl/pull/508#discussion_r4450186167).
   Retain the existing projection onto minimizers for
   `Constrained(NormL1/GroupNormL2, IndBallL2)`; do not replace an
   existing correct endpoint with `ArgumentError`. The nonconvex
   penalized-sphere `γ = Inf` endpoint remains unsupported.

The [cross-package source comparison](https://github.com/mlakolar/ProximalBase.jl/pull/508#issuecomment-6593696105)
also argues that the full E3 guarantee is **optional**, not an established
interoperability requirement. The final review should still consider A
(all-NaN for coupled exceptional input) versus the scoped determined limits
proposed here, including the risk of hiding upstream divergence.

## Mathematical model and joint-limit criterion

For fixed parameters and finite `x`, the regular-step operator is

~~~text
P(γ, x) = argmin_z { (1/2)‖z-x‖₂² + γ f(z) },
~~~

with the package's documented selection/tie rule for nonunique projections.

The proposed meaning for exceptional inputs is:

1. Validate shape, types, step **admissibility**, relevant structural
   constraints, and applicable parameter restrictions before writing (E1/O6).
   An invalid `γ = -1`, `γ = NaN`, or an unsupported endpoint must raise,
   even if `x` also contains infinity. No output is changed.
2. For an **admitted** step endpoint (`γ = 0` or `γ = Inf`) with
   **finite** `x`, use P1's family-specific endpoint. Where E4 fixes a
   coordinate, keep that domain constraint.
3. At an admitted exceptional step **and** `x` with ±Inf entries, E3
   gives the value only if the joint limit as admissible *finite* steps
   and *finite* inputs approach their targets is path-independent. Otherwise
   NaN is returned in the dependent entries (the entire coupled projection
   block when no unique projected point exists). In particular, **do not
   select either of two disagreeing iterated limits**.

Two iterated limits are useful **necessary checks**, not a sufficient proof
of the joint limit. As a generic multivariable-limit counterexample,
`h(s,t) = st/(s²+t²)` has both iterated limits zero as
`s,t -> Inf`, but `h(t,t) -> 1/2`. This is not a proximal
counterexample; it demonstrates why the proposed contract must require
path independence rather than mere agreement of the two iterated limits.

**Validation precedence is not limit precedence.** The proposal respects
the path-dependent-NaN decision already adopted in decision 21. It does not
imply that every admitted exceptional pair has a computable joint limit.

### A counterexample to commuting limits

Let `f(z) = |z| + I_{[-1,1]}(z)` and take scalar `x = M > 0`.
Then `P(γ, M) = min(1, max(0, M-γ))` for finite values, so

~~~text
lim_{γ→∞} lim_{M→∞} P(γ,M) = 1
lim_{M→∞} lim_{γ→∞} P(γ,M) = 0.
~~~

Even ordinary `NormL1(1)` has this conflict at (`x = +Inf`, `γ = +Inf`):
argument-first yields `+Inf`, step-first yields `0`.
Indeed, `x = t + c`, `γ = t` gives
`prox_{γ|·|}(x) = c` for each `c ≥ 0`, so the joint limit is
not unique. The proposed result is therefore **NaN**, matching
decision 21 and the current implementation. The same conclusion holds
for a penalized ℓ2 ball when finite paths with `γ` and `x`
growing at different rates lead to zero versus boundary projections.

## Proposed contract changes

### E1: distinguish validation order from endpoint order

Preserve the existing priority of structural validation, step validity and
domain-defining parameters. For simultaneously exceptional but **admitted**
`γ` and `x`, require a path-independent **joint** limit instead of
choosing an order of limits. If two sequences give different outputs,
the result is NaN in the dependent block (decision 21); no
`ArgumentError` is introduced merely because an accepted step meets
an infinite argument. The E2/E4 question of NaN input in a hard-fixed
coordinate remains separate (see below).

### E3: a scoped, path-independent infinite-input guarantee

For **finite** radii and sign-symmetric, bounded ball/sphere projections
named below:

- For direct `IndBallL1(r)`, `IndBallL2(r)` and
  `IndSphereL2(r)` with `0 < r < Inf`, one independent
  ±Inf scalar coordinate and all other coordinates finite gives
  `±r e_k` at a fixed admitted step. A
  `SymmetricSparseIterate` **stores each off-diagonal coordinate
  once, but counts it twice** (M2), so one infinite off-diagonal
  coordinate yields `±r/2` for the ℓ1 ball and
  `±r/√2` for the ℓ2 ball/sphere in each of the
  two symmetric matrix entries.
- Two or more **independent, unconstrained** infinite coordinates generally
  do not have a path-independent direction; return NaN in the dependent
  coupled block. Example for the Euclidean sphere:
  `(M,M) -> (1/√2,1/√2)` but `(M,2M) -> (1/√5,2/√5)`.
  For `IndBallL1` the corresponding limits are `(r/2,r/2)` versus
  `(0,r)`. No arbitrary equal-rate convention.
- `IndBallL1(0)` and `IndBallL2(0)` have the constant zero projection
  for finite arguments; the result at ±Inf input is zero as well.
  Whether **NaN** input at radius zero is also independent (and should
  produce zero instead of following today's E2 block rule) needs an
  explicit E2 review; this PR does not decide that.
- Extend the same principle to **specifically supported** constrained
  sign-symmetric functions, starting with
  `Constrained(NormL1, IndBallL2/IndSphereL2)`,
  `Constrained(GroupNormL2, IndBallL2/IndSphereL2)`, and
  `Constrained(IndBallL0, IndBallL2/IndSphereL2)`, only after accounting
  for effective domains, active hard constraints, sparsity, and
  nonunique directions. One infinite *group* is **not** enough: a group
  containing two independent infinite coordinates can still have an
  undetermined direction.
- The generic `BallConstrained` accepts many more inner functions,
  including sign-restricted cones and spectral penalties. Do **not**
  automatically apply `±r e_k` to them. For example, the projection of
  a negative-infinite coordinate onto a nonnegative cone is not a negative
  boundary vector. Other coupled families (`IndStiefel`, `SumLargest`,
  `IndPolyhedron`, spectral projections) are **out of scope**.
- A hard-infinite-weight coordinate is fixed to zero under E4. An infinity
  there does not itself establish the direction; consider the remaining
  free coordinates. For a sphere, the effective domain may be empty and
  must fail before writing. A symmetric argument containing (`+Inf,-Inf`)
  in corresponding off-diagonal entries is also ambiguous if its symmetric
  part involves `Inf - Inf`.
- Inputs with NaN continue to follow **E2** except in future explicitly
  documented independence cases. No newly claimed NaN behavior is
  implemented or approved here.

**Bounded-radius scope:** `IndBallL1(Inf)` and `IndBallL2(Inf)` are currently
constructible and are not finite-radius projections; `IndSphereL2(Inf)`
is also constructible although the corresponding sphere has no finite
points. Decide parameter validation and any endpoint semantics separately;
do not extrapolate this proposal to infinite radii.

An infinite computed norm caused by **finite-input overflow** is not an
infinite input. Existing scaled/recovery arithmetic must continue to handle
that path. Never classify by `isinf(norm(x))` alone.

### P1: name zero/infinite step endpoints for constrained operators

- Preserve the already decided negative/NaN `γ` rejection before writes,
  including block compositions; preserve admitted scalar-step endpoints for
  indicators, separable functions, and ridge.
- Propose naming `γ = 0` for the specifically supported penalized
  `Constrained(NormL1, ball/sphere)` and
  `Constrained(GroupNormL2, ball/sphere)` families when their effective
  domain is nonempty and a projection with the documented tie rule is
  available. This is the projection onto the domain **after** applying
  domain-defining restrictions such as infinite coordinate/group weights;
  it is not necessarily the unrestricted ball/sphere projection.
- At `γ = Inf`, retain the existing convex-ball endpoint for
  `Constrained(NormL1, IndBallL2)` and
  `Constrained(GroupNormL2, IndBallL2)`: project a **finite**
  argument onto the nonempty closed convex minimizer set. This includes
  zero-penalty and zero-weight cases, where the minimizer set may
  contain many points but the projection is unique. This agrees with
  decision 26 and master (reviewer measured `γ = Inf` vs. large finite steps).
  With simultaneous `x = Inf`, require the joint-limit test above:
  `γ = Inf` on `Constrained(NormL1(1), IndBallL2(1))` at
  `[Inf,0]` gives an all-NaN coupled block because paths at
  different relative rates tend to the boundary or to zero.
- For the nonconvex **penalized sphere** variants
  `Constrained(NormL1/GroupNormL2, IndSphereL2)`, do not name
  `γ = Inf` by default; reject it before writing. A zero-penalty
  instance may mathematically reduce to an indicator; whether that
  should gain an exception to this method-family rule remains a
  separate question in the design note, not an unresolved clause in P1.
- `Constrained(IndBallL0, ball/sphere)` is an indicator, so the existing P1
  scalar-step rule applies (subject to feasibility). In particular,
  `γ = 0` and `γ = Inf` are **not** automatically rejected.
- Maintain the existing requirement that array steps on coupled operators
  be positive and finite. Do not generalize to array-step endpoints.
- Continue to reject an unsupported endpoint *before writing*, even at
  `x = Inf`, and account for forwarded/transformed steps through wrappers.

### O2 and cost: preserve the other promises

`value_and_prox!` returns the value at the **stored output** (O2), not the
value of `f(x)` at the original infinite input: an indicator has value
`+Inf` at `x = Inf` but can have value `0` at its finite projected point.
Follow E5 for value classes and O7 for exact aliasing. Prefer one
nonfinite classification fused with a scan that already occurs; no
allocation on finite hot paths and no unconditional BigFloat computation.
An overflowing norm from finite entries must not trigger #469's branch.
C2 demands measurement of an added pass; investigate >5% representative
slowdown under C3 rather than assuming one count is free.

## Proposed expected-result matrix (review targets, **not current results**)

Unless noted otherwise, finite radius `r = 1`, scalar step, Float64,
dense storage. "all NaN" means the whole dependent block.

| Operator | Input | γ | Proposed result / meaning |
|:--|:--|:--|:--|
| `IndBallL2(1)` | `[Inf,0]` | 1 | `[1,0]` |
| `IndSphereL2(1)` | `[-Inf,2]` | 0, 1, Inf | `[-1,0]`; indicator step-independent |
| `IndBallL1(1)` | `[Inf,0]` | 1 | `[1,0]`, not partial NaN |
| `IndBallL2(0)` or `IndBallL1(0)` | `[Inf,2]` | 1 | `[0,0]` |
| `IndSphereL2(1)` | `[Inf,Inf]` | 1 | all NaN: different relative growth rates |
| `Constrained(NormL1(1), IndSphereL2(1))` | `[Inf,2]` | 0 | `[1,0]` **if** proposed P1 extension approved |
| `Constrained(NormL1(1), IndBallL2(1))` | `[3,0]` | Inf | `[0,0]`: project onto the minimizer set |
| `Constrained(NormL1(1, [0,1]), IndBallL2(1))` | `[3,2]` | Inf | `[1,0]`: unpenalized coordinate remains |
| `Constrained(NormL1(0), IndBallL2(1))` | `[3,0]` | Inf | `[1,0]`: zero penalty, whole ball minimizes |
| `Constrained(GroupNormL2(1, [[1,2]]), IndBallL2(1))` | `[3,1]` | Inf | `[0,0]`: group penalty minimizer |
| `Constrained(NormL1(1), IndBallL2(1))` | `[Inf,0]` | Inf | all NaN: joint limit depends on `x/γ` |
| `Constrained(IndBallL0(1), IndSphereL2(1))` | `[Inf,0]` | Inf | `[1,0]` (indicator step) |
| `Constrained(GroupNormL2(1, [[1,2]]), IndSphereL2(1))` | `[Inf,0,1]` | 1 | `[1,0,0]` |
| Same group-sphere | `[Inf,Inf,1]` | 1 | all NaN: one group, two independent infinite entries |
| `IndSphereL2(1)` | `[Inf,0]` | -1 or NaN | `ArgumentError`; destination unchanged |
| `NormL1(1)` | `[Inf]` | Inf | `[NaN]`: path-dependent joint limit, unchanged from master |
| `Constrained(NormL1(1, [Inf,1]), IndSphereL2(1))` | `[Inf,3]` | 1 | `[0,1]` if E4 hard-fixed direction is honored |
| Direct `IndBallL1(1)` on symmetric storage | one `+Inf` off-diagonal coordinate | 1 | two symmetric matrix entries `+1/2`, others zero |
| Direct ℓ2 ball/sphere on symmetric storage | one `+Inf` off-diagonal coordinate | 1 | two symmetric matrix entries `+1/√2`, others zero |
| Direct sphere on symmetric output | asymmetric opposite infinities in one pair | 1 | NaN where symmetricization is ambiguous |

The `γ = 0` and determined-infinity rows are
**proposed contract amendments**, not assertions that present code passes.
For simultaneous `γ = Inf, x = Inf`, the proposed NaN result is a
path-independence rule; the reviewer verified `NormL1` already
returns NaN on master and reported partially NaN current output for
the penalized ball case.
When a proposed output is finite, the corresponding `value_and_prox!`
value must meet O2 at the stored result.

## Review decisions requested (please critique before implementation)

1. **Joint exceptional limits:** is the decision-21 rule correctly
   applied to simultaneous admitted `γ = Inf, x = Inf` (NaN for
   path dependence, finite answer only for a unique joint limit)?
   Can the reviewer exhibit a proximal counterexample in which the
   two iterated limits agree but the joint limit nevertheless differs
   along diagonal paths? The contract explicitly requires more
   than agreement of two iterated limits.
2. **Zero step on constrained penalties:** approve or reject naming
   `γ = 0` for the listed ball/sphere families. Can every admitted
   instance project onto its effective domain, including hard weights,
   sparse constraints, empty feasible sets and sphere tie rules?
3. **Scope of E3:** which constrained inner families can *prove* unique
   extended-real limits? Is the proposed narrow list still too broad?
   Confirm that a "single infinite group" is not a sufficient criterion.
4. **Degenerate radii:** decide how `r = 0` interacts with E2 NaNs,
   and whether infinite radii should be rejected at construction or
   treated under a separately documented rule. Do not silently expand
   the finite-radius guarantee.
5. **Hard constraints and NaNs:** does E1/E4 require a hard-infinite-weight
   coordinate with NaN argument to map to zero, rather than E2's NaN?
   Current `NormL1` behavior does not enforce that inference. Should
   it be a separate issue/change?
6. **Degenerate penalization:** the convex **ball** `γ = Inf`
   endpoint is now named, including `Constrained(NormL1(0), ball)`.
   Should a zero-penalty, effectively indicator **sphere** instance
   receive a special `γ = Inf` endpoint despite the unsupported
   generic penalized-sphere method? What is the simplest consistent
   rule for E4/P1 without surprising dispatch exceptions?
7. **Symmetric storage:** verify that output multiplicities and
   symmetricization occur in the same order as the proposed
   limit classification. For the ℓ1-sphere negative-margin reference
   oracle, the correct single-support score is
   `sqrt(m_i) * (|x_i|-threshold_i)`, **not** margin/sqrt(m_i)
   (see [#469 comment](https://github.com/mlakolar/ProximalBase.jl/issues/469#issuecomment-6578835884)).
8. **API and values:** confirm the P1 validity checks for `prox`,
   `prox!`, `value_and_prox!`, array steps, wrappers, and nested
   block operators, including O2/O6/O7 on exceptional paths.
9. **Cost:** can direct and constrained paths add a reliable nonfinite
   classification without changing ordinary finite results or
   meaningful performance?

A reviewer is invited to reject or narrow this proposal. Please
identify exact counterexamples (finite sequences where the claimed limit
differs), incompatible already-approved rules, and missing API surfaces.
Do not treat a passing current test as proof of a mathematical limit.

## Follow-up implementation plan (NOT part of this PR)

1. **#311:** enforce negative/NaN and unsupported-endpoint checks before
   any write in the affected operators and their wrappers/block sums.
   Use the existing private `_checkstep` protocol where appropriate;
   do not add a redundant public step API.
2. **#469, direct projections:** finite-radius `IndBallL1`,
   `IndBallL2` and `IndSphereL2`, including zero radius, symmetric
   multiplicity, the partial-NaN defect, and positive/negative
   infinities. Preserve existing finite overflow recovery.
3. **#469, constrained projections:** only mathematically reviewed
   inner families with the approved P1 endpoints, masking, feasible
   domains, nonconvex tie rules, and correctly reported stored values.
4. **Tests and docs:** finite BigFloat sequences as independent
   oracles (corrected symmetric negative-margin fallback), including
   independent sequences for simultaneous `γ,x -> Inf` with varying
   relative rates; both Float32/Float64 input/output combinations,
   sparse/atom/symmetric storage where supported, aliasing,
   ±Inf/NaN combinations, invalid steps with unchanged outputs, O2
   values and representative timing.
   Update each public function's docstring in its implementation PR.

This draft PR edits only documentation and must **not** close #311 or #469
or be interpreted as implementing a new exception guarantee.
