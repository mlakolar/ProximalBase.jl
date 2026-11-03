# Design review: step validation and exceptional inputs (#311, #469)

**Status: DRAFT / PROPOSED AMENDMENT; not approved, not implemented.**
This note accompanies the *proposed* E1/E3/P1 wording in
[`docs/src/numerics.md`](../../src/numerics.md). The contract page is the
normative target **only after this PR receives approval**. No claimed
exceptional-input behavior in this document should be mistaken for a passing
test on the current master branch. Neither #311 nor #469 should be closed by
this documentation PR.

**Recommendation for the next reviewer:** adopt **Option A, narrowly scoped
to bounded coupled ball/sphere projections**. When an exceptional input
participates in a nonconstant bounded coupled projection, deliberately return
NaN for the entire dependent block, even when a mathematical limit of that
projection is unique. Do **not** generalize that rule to every nonseparable
proximal operator: for instance, the `NormL2` proximal point of
`(M,0)` with fixed finite step tends to `(+Inf,0)`. Preserve the existing
E3 family-specific limits outside the stated bounded-projection scope.

This is a recommendation for review, not a maintainer decision. The
[latest reviewer argument](https://github.com/mlakolar/ProximalBase.jl/pull/508#issuecomment-6593831566)
prompted this revision. Please challenge the narrower scope, the
input-dependence exceptions, and the interaction with P1 before approving.

## 1. Issues and prior decisions

### #311: a valid step is not a merely nonnegative number

[#311](https://github.com/mlakolar/ProximalBase.jl/issues/311) asks for
consistent step validation across `prox`, `prox!`,
`value_and_prox!`, wrappers, and block operators.

- Negative and NaN steps are **errors before any caller-owned writes**
  (E1/P1/O6), independently of the argument's finite/NaN/Inf class.
- A zero or infinite step is accepted **only for a named family endpoint**.
  Some indicators ignore the scalar step because their proximal map is
  the projection. Separables have limits at endpoints named by P1; ridge
  has its own named cases. Coupled per-coordinate metric/array steps
  remain positive and finite.
- `_checkstep` from [merged #304](https://github.com/mlakolar/ProximalBase.jl/pull/304)
  is a private pre-write mechanism, not yet a universal guarantee.
  In particular, the default `_checkstep` in `src/interface.jl` returns
  `true`, and #311 identifies missing forwarding for several constraints
  and wrappers. A later invalid block must not leave earlier blocks modified.
- An explicit `isnan(γ)` test is **not required** if the predicate is
  written as `γ >= 0 || throw(...)`; NaN fails the comparison.
  `γ < 0 && throw(...)` is **not** equivalent: it lets NaN through.
  For families requiring strictly finite positive steps, use an additional
  finite/positive validation. The tests must cover NaN explicitly.

### #469: mathematical infinity limits are not the same as a desired API

[#469](https://github.com/mlakolar/ProximalBase.jl/issues/469) identifies
that coupled balls/spheres currently tend to produce all NaN, with
`IndBallL1` producing partly NaN, at inputs with one infinite entry.
For example, at master as checked by the reviewer:

| Operator, unit radius | Argument | Reported current output |
|:--|:--|:--|
| `IndBallL1` | `[Inf, 0]` | `[NaN, 0]` |
| `IndBallL2` | `[Inf, 0]` | all NaN |
| `IndSphereL2` | `[Inf, 0]` | all NaN |
| `Constrained(NormL1, IndSphereL2)` | `[Inf, 0]` | all NaN |
| `IndBallL2(0)` | `[Inf, 2]` | all NaN |
| `IndBallL1(0)` | `[Inf, 2]` | `[0, 0]` |

These are observations by the reviewer, not universal guarantees.
[Merged #460](https://github.com/mlakolar/ProximalBase.jl/pull/460)
made the ℓ1 sphere consistently all NaN as a **holding policy**. #469
asks whether to make that policy permanent or implement determined limits
across a much larger family.

### What the existing numerical contract says

The [contract](../../src/numerics.md) has the following relevant rules:

- **E1:** structure and step validity first; then zero/infinite
  parameters and domain constraints (E4); then exceptional argument entries.
  This is a precedence of *meaning*, not a required sequence of scans.
- **E2:** a NaN contaminates every value/proximal entry that **depends** on
  it; for a coupled block this can mean the whole block, not independent
  blocks or entries whose value is structurally fixed.
- **E3:** at an infinite argument, formulas/limits where determined;
  proximal outputs take a **documented** limit, otherwise NaN in the
  dependent entries.
- **E4:** zero levels preserve hard domains; an infinite coordinate/group
  weight can force an entry/group to zero, independently of the argument.
- **P1:** exceptional steps accepted only at documented endpoints.
- **O2:** `value_and_prox!` returns the value at the *stored output*,
  not at the input. **O6:** invalid arguments/steps throw without writes.
  **O7:** exact aliasing is supported. **C1–C3:** protect finite-input
  accuracy and ordinary-path performance.

[Decision 21](decisions.md#21-infinite-entries-e3) (approved
2026-11-01) favors a formula or limit where determined and NaN for
path-dependent limits. [Decision 26](decisions.md#26-zero-and-infinite-steps-p1)
accepts only named step endpoints. The proposed Option A is a **deliberate
choice to decline extending some bounded projections to infinity, even when a
unique mathematical limit exists**. Calling it "what E3 already says" would
conceal this substantive choice. Record it as an **explicit proposed
amendment to Decision 21** and obtain maintainer approval.

## 2. Mathematical comparison of the three choices

### A: refuse a nonconstant coupled projection at exceptional input

For an affected finite-radius bounded coupled projection, after E1/E4
constraints have been applied, any influential ±Inf or NaN coordinate
produces NaN in its **entire dependent block**. No partial result like
`[NaN,0]` is returned as a successful bounded projection.
If the proximal map is *structurally independent* of an exceptional input
coordinate, preserve that independent result instead.

This is an operational choice. It does **not** assert that the unique
mathematical limit is absent. It trades extension of the mathematical
domain for reliable signaling that the ordinary finite-domain call received
exceptional data.

### B: return the unique mathematical limit when one exists

For direct Euclidean ball/sphere at `r>0`,

~~~text
P(M, a) = r (M,a)/sqrt(M²+a²) -> (r,0)    as M -> +Inf
P(-M,a) -> (-r,0)
~~~

and the ℓ1-ball projection of `(M,a)` similarly tends to `(r,0)` for
fixed finite `a`. So a one-Inf output `[r,0]` is **mathematically
determined**; returning all NaN under A is a conscious *non-extension*.

For two infinities the situation is different:

~~~text
Euclidean sphere:
  P(M,M)   -> (r/√2, r/√2)
  P(M,2M)  -> (r/√5, 2r/√5)

L1 ball:
  P(M,M)   -> (r/2, r/2)
  P(M,2M)  -> (0, r)
~~~

Both sequences converge entrywise to `(+Inf,+Inf)` but have different
projected limits. This is genuine mathematical ambiguity, so even B must
return NaN rather than assuming equal relative growth rates.

On a `SymmetricSparseIterate`, the off-diagonal coordinate is **stored
once and counted twice** (M2). The determined *one-coordinate* limit is
`±r/2` for the ℓ1 ball (`2|a|<=r`), but
`±r/sqrt(2)` for the ℓ2 ball/sphere (`2a²<=r²`).
This correction was caught in the
[first review](https://github.com/mlakolar/ProximalBase.jl/pull/508#discussion_r4450186148).
Under A the values are still useful as evidence of what we are
deliberately *not* promising; they are **not** implementation targets.

Constrained examples add geometry and implementation obligations: group
directions, hard-fixed coordinates, empty effective domains, symmetry,
nonconvex ties, and zero/Inf steps. "One infinite group" is not sufficient:
a single group containing `[Inf,Inf]` contains two independently
infinite degrees of freedom, whose relative growth is unknown.

### C: return B for direct projections, but A for constrained ones

C is smaller than B initially, but representation-dependent behavior is a
serious drawback. By E4, `NormL1(0)` has no finite penalty, so

~~~julia
prox(IndSphereL2(1), [3.0, 4.0]) ==
prox(Constrained(NormL1(0), IndSphereL2(1)), [3.0, 4.0])
~~~

and both describe the **same** sphere projection. C would return
`[1,0]` for the direct representation at `[Inf,0]` but all NaN for
the constrained representation. This undermines equivalence of
mathematically identical function representations.

A uses the same failure semantics for both, and full B would require
implementing and proving their determined limits consistently.

### Trade-off table

| Criterion | A (recommended) | B | C |
|:--|:--|:--|:--|
| Unique single-Inf mathematical limit returned | No, by deliberate policy | Yes | Only some families |
| Exposes exceptional solver iterate | Yes, for affected blocks | Often no | Depends on representation |
| Direct/constrained zero-penalty equivalence | Yes | Yes, if fully completed | No |
| Exceptional-path proof/test complexity | Lower | High | Medium with inconsistent API |
| Existing #460 holding rule | Becomes final within this scope | Must change | Usually unchanged |
| Finite-input behavior and accuracy | Preserve | Preserve | Preserve |
| Ordinary-path overhead | Should be minimal if triggered lazily | Also can be minimal | Also can be minimal |

**Important qualification:** A is **not automatically faster** than B.
Both need enough classification to distinguish exceptional input from a
computed norm that overflowed on finite input. The advantage of A is the
smaller semantic and proof/test surface, not an established microbenchmark
win. Benchmark any additional scan under C2/C3.

## 3. Why A is preferred here

### 3.1 Preserve diagnostic information for optimization solvers

For `IndBallL2(1)`, the two policies make a sharp difference:

~~~julia
x = [Inf, 0.0]
value_and_prox!(y, IndBallL2(1.0), x, 1.0)
~~~

| Policy | Stored `y` | Returned value | What the caller sees |
|:--|:--|:--|:--|
| A | `[NaN,NaN]` | NaN (E2/O2) | A numerical failure remains observable |
| B or C | `[1,0]` | 0 (O2) | A feasible point and zero indicator value |

In the second row, neither the returned point nor the returned value tells
the solver that its input contained infinity. A caller *can* inspect the
original input itself, and sometimes infinity is intentionally supplied, so
this is not proof B is unsafe. But for finite-iterate solvers, preserving
an exceptional signal at the oracle boundary is a compelling conservative
default. It reduces the chance that a preceding overflow or divergence is
laundered into a plausible feasible iterate.

This is not an argument that every infinite *value* is a failure. E3
correctly retains `+Inf` function values for norms/indicators and allows
unbounded proximal outputs to have infinite entries where documented.
We are deliberately narrowing the policy to **bounded coupled
projections** that would otherwise erase evidence of an exceptional input.

### 3.2 A is consistent across equivalent representations

An ℓ2 sphere projector and a zero-penalty norm constrained to that sphere
must not disagree *only* because one was constructed as `IndSphereL2`
and the other as `Constrained(NormL1(0), IndSphereL2)`.
Similarly, a hard-domain indicator and its equivalent constrained
representation should agree in their exceptional-input classification.

The guarantee is **scope-of-dependence**, not globally poisoning every
unrelated block in `SeparableSum`. The same principle applies to
zero-radius and structurally hard-fixed coordinates below.

### 3.3 A is less demanding than B, without being an arbitrary IEEE accident

Under B, each special family needs a proof of path independence,
particularly around nonconvex selection and symmetrization. Group and
sparse iterates add distinctions between stored coordinates, actual
matrix entries, and independent directions. Both proximal output and
value, both input/destination types, every supported step endpoint, and
all aliases must be tested. A still requires robust NaN propagation,
but not deriving a new closed-form limiting vector for every coupled
family.

It also fixes the current **partially NaN** defect (`IndBallL1`) and
keeps the all-NaN holding rule of #460. It is not "the default from
IEEE": other packages frequently return `[NaN,0]` accidentally and
sometimes choose a random finite output for a NaN input.

### 3.4 What comparison packages do—and do not—establish

The [cross-package comparison](https://github.com/mlakolar/ProximalBase.jl/pull/508#issuecomment-6593696105)
and [independent verification](https://github.com/mlakolar/ProximalBase.jl/pull/508#issuecomment-6593831566)
cover ProximalOperators.jl, PyProximal, SigPy and Pyxu.
Their source formulas or executed checks typically produce
`[NaN,0]` for an ℓ2 ball on `[Inf,0]`. None of the inspected
implementations documents either **A** or **B** as its exceptional-input
contract. These findings therefore offer **no external precedent that
settles the choice**. ProximalBase's E1/E2/E3/O2 goals must decide it.

The ProximalCore interface docstrings show a step parameter and the
`1/(2γ)` formula but do **not** explicitly state `γ>0`; don't attribute
a documented positivity guarantee to that interface.

### 3.5 Future evolution is possible but not free

A future release could deliberately add a B-style finite limiting
extension for a particular family. That can be reviewed separately.
**Do not claim it is backward-compatible:** callers may rely on
`isnan`/`isfinite` to detect exceptional data. A change from NaN to
a finite point changes solver control flow and is observable API
behavior. A offers a smaller initial commitment, not a license to
change it without a versioned contract decision.

## 4. Precise proposed scope: avoid a blanket rule for all coupled maps

The recommendation is an **A-with-independence-exceptions** rule for
bounded **ball/sphere projection** families, not for all coupled proximal
operators and not for all bounded/separable sets.

### 4.1 Covered, finite-radius families

- Direct: `IndBallL1(r)` and `IndBallL2(r)` with finite
  `0<r<Inf`; `IndSphereL2(r)` with finite `r>0`.
- Supported `Constrained(f, IndBallL2(r))` and
  `Constrained(f, IndSphereL2(r))` **when the result is a bounded
  coupled projection or proximal point**. For example:
  `f = NormL1`, `GroupNormL2`, `IndBallL0`, or `ZeroFunction`,
  with valid fixed parameters/domain. The identical zero-penalty
  formulations of direct projections must have the same behavior.
- Other *bounded coupled projection families* (e.g. some spectral and
  matrix projections) may already return NaN for nonfinite input, but
  **are not added to the normative guarantee by this PR**. Their E3
  status remains family-specific, and extension requires a separate
  contract decision.

For a covered family, a **nonconstant effective coupled block** with a
structurally active input position equal to `±Inf` yields all-NaN
output in that dependent block; a NaN input follows E2.
The returned value from `value_and_prox!` is NaN when the stored output
is NaN, following E2/O2. No per-coordinate `[NaN,0]` mixed output
is promised as a bounded coupled projection.

### 4.2 Independent positions and degenerate domains

E1 applies domain-defining constraints **before** exceptional input
interpretation, as E4 already promises:

- `IndBallL1(0)` and `IndBallL2(0)` have the singleton
  feasible set `{0}`. Their projection is **identically zero for
  all finite inputs**. Under the proposed dependence interpretation,
  output zero even for `[Inf,2]` or `[NaN,2]`; the indicator value
  at the stored output is zero. This fixes the current ℓ2 zero-radius
  exceptional-input discrepancy.
- If an infinite coordinate/group weight fixes an entry/group to zero
  for **every finite step**, its input can be ignored for purposes of
  that coordinate's contribution to the projection. An otherwise
  feasible sphere can use the free entries: e.g.
  `Constrained(NormL1(1,[Inf,1]), IndSphereL2(1))` at
  `[Inf,3]` with finite step should give `[0,1]` (stored value 1),
  *if* E1/E4 independence is implemented correctly. The same logic
  suggests ignoring a NaN at a structurally fixed position; this is an
  E2/E4 conformance requirement for reviewer confirmation.
- If all entries of a positive-radius sphere are hard-fixed to zero,
  the **effective domain is empty**. An infeasibility
  `ArgumentError` is required before writing, regardless of x. It
  is **not** an ordinary NaN return.
- A zero *penalty level* removes only the finite penalty, **not**
  its domain restrictions. Thus `Constrained(NormL1(0), sphere)`
  is still a bounded sphere projector and follows A, rather than
  becoming the zero function whose prox copies infinity.

These exceptions must be based on **structural independence under the
finite-parameter operator**, not a coincidental zero at a
single step endpoint or a branch chosen because a NaN comparison
failed. A projection that happens to be constant in a local region
is not necessarily structurally independent by E4.

### 4.3 Other families remain on existing E2/E3 rules

- **Separable:** `NormL1` can map `+Inf` to `+Inf`
  at an ordinary finite step; `IndBox` or
  `IndBallLinf` may clamp it to a finite bound. Their output
  depends coordinatewise, so A's whole-block rule does not apply.
- **Coupled but unbounded proximal output:** `NormL2`
  at fixed finite positive `γ` sends `(M,0)` to
  `(M-γλ,0)` for sufficiently large M, tending to
  `(+Inf,0)`. A universal "every coupled prox on infinity is
  NaN" would suppress a genuine, existing E3-style limit.
  The mathematical observation does not assert that current code has
  implemented that limit correctly in every type/storage layout.
- **Unbounded or unusual constraints:** `IndBallL1(Inf)`,
  `IndBallL2(Inf)` and `IndSphereL2(Inf)` are
  currently constructible and require separate semantics. They
  are **not** included in the finite-radius A guarantee.
- **Other coupled families:** `IndStiefel`, `SumLargest`,
  `IndPolyhedron`, spectral and general linear projections
  are outside this change. The broader "A for all coupled blocks"
  is an alternative the reviewer should evaluate explicitly, not
  a consequence silently slipped into E3.
- A `SymmetricSparseIterate` stores one off-diagonal coordinate
  and counts it twice (M2); under A it receives NaN as part of the
  affected coupled block, regardless of the unique B-style
  `r/2` or `r/sqrt(2)` limit. Asymmetric
  off-diagonal `(+Inf,-Inf)` may symmetrize to NaN
  before the projection and is governed by E2; do not choose a
  direction from a NaN symmetric part.

### 4.4 What "depends" must mean

The contract should not define dependence by whether an implementation
read an array element, or by whether the *numerical formula*
evaluates `0*Inf`. An input entry is **structurally irrelevant**
if varying that entry over all finite values leaves the valid
proximal output (and value at the stored point) unchanged, after
applying E4's domain restrictions. This describes radius-zero balls
and genuinely fixed coordinates/groups.

The classification must be stable under equivalent representations
and not change merely because intermediate arithmetic is rewritten.
However, when a *simultaneous exceptional step* is involved,
independence cannot be inferred only from the map evaluated at
`γ=Inf`: the finite-parameter joint limit may be path-dependent,
as Section 5 explains.

**Review question:** is this criterion sufficiently precise for every
supported sign-restricted cone and group/spectral member of
`BallConstrained`? If not, narrow the normative list further and
make each exception explicit rather than creating a speculative
generic "depends" dispatcher.

## 5. #311: step validation, endpoints, and noncommuting limits

### 5.1 Invalid steps: error first, never an exceptional output

Examples (destination unchanged even when `y === x`):

~~~julia
prox(IndSphereL2(1), [1.0, 0.0], -1.0)    # ArgumentError (P1)
prox(IndSphereL2(1), [Inf, 0.0], NaN)    # ArgumentError before NaN propagation
~~~

The currently permissive indicator implementations are a gap #311
must close. The private `_checkstep` protocol should be
extended rather than introducing a parallel public API.
Where the family admits `γ=0` or `γ=Inf`,
rejecting all non-finite steps is equally incorrect.

### 5.2 Named step endpoints are preserved at finite arguments

- Direct ball/sphere **indicators** ignore their scalar step (P1).
- `Constrained(NormL1/GroupNormL2, IndBallL2)`
  is convex. At `γ=Inf` and **finite** argument,
  project onto its nonempty closed convex minimizer set, including
  zero/zero-weight penalties. This was checked by the prior reviewer
  against master and finite large steps. Do not replace this valid
  endpoint with an error merely because the function is "penalized".
- The named `γ=0` endpoint for supported constrained
  penalized balls/spheres is projection onto the **effective domain**,
  preserving hard-fixed coordinates and feasibility, rather than
  invariably the bare ball/sphere.
- Nonconvex penalized sphere variants
  `Constrained(NormL1/GroupNormL2, IndSphereL2)`
  have **no named `γ=Inf` endpoint** at present.
  Reject that step before writing, even when x is exceptional.
- `Constrained(IndBallL0(k), ball/sphere)` is an
  indicator; scalar-step invariance still applies, provided
  the intersection is feasible.
- Array steps on coupled sets remain positive and finite.
  Forwarded/transformed steps in wrappers must be checked before
  any writes, including inside block sums.

### 5.3 Valid `γ=Inf` plus `x=Inf`: do not pick a convenient limit

The step/argument orders need not commute. For
`f(z)=|z|+I_{[-1,1]}(z)`, finite
`M>0` and `γ>0`,

~~~text
prox_{γf}(M) = min(1, max(0, M-γ)).
lim_{γ→Inf} lim_{M→Inf} prox_{γf}(M) = 1,
lim_{M→Inf} lim_{γ→Inf} prox_{γf}(M) = 0.
~~~

Ordinary soft thresholding has the same issue:
`prox_{γ|·|}(t+c) = c` with `γ=t`,
for any `c>=0` and sufficiently large t.
Thus `prox(NormL1(1), [Inf], Inf)` should
remain NaN, rather than choosing a step-first zero.

For `Constrained(NormL1(1), IndBallL2(1))`:
at `γ=Inf` and **finite** x the minimizer projection is
`0`. At fixed finite positive `γ` and
`x=(M,0), M→Inf` the solution approaches `(1,0)`.
Along `x=(t+c,0), γ=t`, the constrained
solution changes with c. So at the **simultaneous** pair
`(γ=Inf,x=[Inf,0])` the appropriate output is
**all NaN**—even though the endpoint map at exactly
`γ=Inf` is constant for finite x.

This is a crucial distinction between:
1. **Structurally constant across all finite-step maps** (e.g.
   a radius-zero ball), where an exceptional argument can be ignored;
2. **Constant only after taking `γ→Inf` with fixed finite x**
   (e.g. a strictly penalized ball), where the joint limit can still
   be path-dependent.

A path-independent joint limit is a **necessary** condition for
choosing a finite result at simultaneous exceptional step/input, but
under the proposed bounded-projection Option A it is not **sufficient**:
the family intentionally declines extension at influential infinity
even when a mathematical limit exists.

Agreement of the two iterated limits is also not a proof of a joint
limit. The generic scalar function
`h(s,t)=st/(s²+t²)` has both iterated limits
zero as `s,t→Inf` but `h(t,t)→1/2`.
That generic example is not a proximal map, but shows why the
contract should never substitute two iterated limits for a
joint path-independence argument.

### 5.4 No E1 shortcut for a jointly exceptional pair

A **step validity** failure gives an `ArgumentError` before writing.
An **admitted** step with an invalid/nonfinite *argument* returns
an exceptional numerical output under E2/E3, not a step error.
A mathematical endpoint at finite x is not automatically the answer
at infinite x. Do not conflate these three separate meanings.

## 6. Proposed behavior matrix for review (NOT current test results)

Assume Float64, scalar step and r=1 unless stated. "all NaN" refers
to the entire *dependent* coupled block. `value_and_prox!` returns
the value at stored y (O2).

| Function | Input x | γ | Proposed output / return value | Rationale |
|:--|:--|:--|:--|:--|
| `IndBallL2(1)` | `[Inf,0]` | 1 | y all NaN; value NaN | A intentionally declines unique `[1,0]` limit |
| `IndBallL1(1)` | `[Inf,0]` | 1 | y all NaN; value NaN | Fix partial-NaN defect |
| `IndSphereL2(1)` | `[-Inf,2]` | 0,1,Inf | y all NaN; value NaN | Indicator accepts step endpoints; A applies to x |
| `IndBallL2(1)` | `[Inf,Inf]` | 1 | y all NaN; value NaN | Both A and B: ambiguous direction |
| `IndBallL1(0)` | `[Inf,2]` | 1 | y `[0,0]`; value 0 | Singleton set independent of input |
| `IndBallL2(0)` | `[Inf,2]` | 1 | y `[0,0]`; value 0 | Fix current discrepancy |
| `IndBallL2(0)` | `[NaN,2]` | 1 | y `[0,0]`; value 0 | Proposed E2 independence interpretation |
| `IndBallL2(1)` | `[NaN,2]` | 1 | y all NaN; value NaN | E2 coupled block |
| `IndBallL2(1)` | finite but enormous `x` | 1 | correct finite projection | Do not classify finite reduction overflow as input Inf |
| `IndSphereL2(1)` | `[Inf,0]` | NaN or -1 | `ArgumentError`; no writes | P1/E1 precede E3 |
| `Constrained(NormL1(0), IndSphereL2(1))` | `[Inf,0]` | 1 | y all NaN; value NaN | Equivalent sphere representation |
| `Constrained(NormL1(1), IndSphereL2(1))` | `[Inf,2]` | 0 or 1 | y all NaN; value NaN | A also at zero-step domain projection |
| `Constrained(GroupNormL2(1, [[1,2]]), IndSphereL2(1))` | `[Inf,0,1]` | 1 | y all NaN; value NaN | Even one infinite coordinate gets A |
| Same group sphere | `[Inf,Inf,1]` | 1 | y all NaN; value NaN | Also path-dependent under B |
| `Constrained(IndBallL0(1), IndSphereL2(1))` | `[Inf,0]` | Inf | y all NaN; value NaN | Indicator endpoint accepted; input still exceptional |
| `Constrained(NormL1(1,[Inf,1]), IndSphereL2(1))` | `[Inf,3]` | 1 | y `[0,1]`; value 1 | First coordinate structurally hard-fixed (E4), review implementation |
| Same hard-fixed sphere | `[NaN,3]` | 1 | y `[0,1]`; value 1 | E2/E4 independence; review separately |
| `Constrained(NormL1(1), IndBallL2(1))` | `[3,0]` | Inf | y `[0,0]`; value 0 | Named convex minimizer projection |
| `Constrained(NormL1(0), IndBallL2(1))` | `[3,0]` | Inf | y `[1,0]`; value 0 | Entire ball is minimizer set |
| `Constrained(NormL1(1), IndBallL2(1))` | `[Inf,0]` | Inf | y all NaN; value NaN | Simultaneous pair path-dependent; endpoint constant alone insufficient |
| `Constrained(NormL1(1), IndSphereL2(1))` | `[3,0]` | Inf | `ArgumentError`; unchanged y | Unsupported penalized-sphere endpoint |
| Direct sphere in `SymmetricSparseIterate` | one infinite off-diagonal | 1 | all NaN in coupled block | A despite B's ℓ2 `r/√2` limit |
| `IndBallL1(1)` in symmetric storage | one infinite off-diagonal | 1 | all NaN in coupled block | A despite B's ℓ1 `r/2` limit |
| `IndBox(-1,1)` or `IndBallLinf(1)` | `[Inf,0]` | valid scalar | `[1,0]` | Separable coordinatewise clipping, not covered |
| `NormL1(1)` | `[Inf]` | 1 | `[Inf]` | Separable E3, not covered |
| `NormL1(1)` | `[Inf]` | Inf | NaN | Joint step/input limit path-dependent |
| `NormL2(1)` | `[Inf,0]` | finite >0 | out of scope; preserve family E3 | Nonseparable does **not** automatically mean all NaN |
| `IndBallL2(Inf)` | `[Inf,0]` | 1 | out of scope | Infinite radius needs its own contract |

This table is a **design oracle**. It deliberately includes outputs
different from master, particularly `IndBallL1(1)`,
`IndBallL2(0)`, hard-fixed NaN coordinates, and invalid
indicator steps. Any proposed finite result from an independent
coordinate requires checking that value and projection agree at the
**stored** output across precision conversions (O2/T3).

## 7. Minimal implementation plan: correctness without overengineering

1. **Close #311 validation gaps first in affected methods.**
   Extend the existing pre-write path to validate the relevant scalar
   step for direct and constrained projections and for forwarded steps.
   Do not make `_checkstep` a public API, add a generic
   "exceptional parameter execution framework", or check after writing.
   Include `NaN`, negative, signed zero, `Inf`, finite
   huge/tiny steps and nested invalid later blocks.

2. **Direct bounded projection #469 patch.**
   On a valid step and after a known-constant shortcut
   (e.g. radius zero), use existing norm/threshold algorithms
   for finite inputs. If a reduction is exceptional, classify the
   *actual input entries*: NaN and ±Inf inputs follow E2/A,
   while finite inputs whose reduction overflowed follow R2
   scaled/recovery arithmetic. Write a whole NaN block and
   return an O2-consistent NaN value. Fix
   `IndBallL1` partial NaN and the zero-radius ℓ2 case.
   Do **not** scan the whole array unconditionally without a
   measured justification.

3. **Constrained ball/sphere patch.**
   Validate effective domains/hard weights before writing and
   classify exceptional input **after structurally irrelevant
   hard-fixed coordinates are excluded**, but before shrinking,
   selecting a support, or normalizing into the output.
   Avoid reading x after overwriting it for `y === x`.
   Do not change finite tie rules or confuse a group containing
   multiple infinities with an independent scalar coordinate.
   For genuinely penalized convex balls at `γ=Inf`,
   retain finite-argument minimizer projections; do not use
   their constant endpoint as a shortcut for joint infinity.

4. **Test semantics, representations and failure atomicity.**
   Cover Float32/Float64 input and destination combinations,
   dense, `SparseIterate`, `AtomIterate` and
   `SymmetricSparseIterate` where supported; ±Inf and
   NaN in first/middle/last positions, one or several;
   zero levels, zero/infinite weights, all-fixed infeasibility,
   symmetric/asymmetric off-diagonal inputs,
   scalar indicator endpoints, constrained endpoints, and
   array-step rejection. Test `prox`, `prox!` and
   `value_and_prox!`, aliased/nonaliased outputs, values at
   stored y, untouched destination on pre-write errors,
   cache history if relevant, and independent blocks.
   Reference **finite-input** BigFloat mathematics to identify
   the unique limits B would have chosen, but assert the
   deliberate A outputs at actual ±Inf input.
   Keep the corrected symmetric negative-margin oracle:
   fallback score `sqrt(m_i) * (|x_i|-threshold_i)`,
   not `(margin)/sqrt(m_i)`.

5. **Measure changes to the normal fast path.**
   Compare representative dense/sparse/mixed-precision
   finite workloads and warm allocations under C1–C3.
   No unconditional BigFloat, no broad array widening,
   no extra device synchronization. A scan on a rare
   exceptional branch is acceptable only after verifying
   that it distinguishes true Inf/NaN entries from finite
   overflow. Follow-up PRs update corresponding docstrings
   alongside implementation.

**Do not merge #311 and #469 into one implementation PR.** This
PR is only a contract/design review. Keep concrete patches small,
and retain test receipts and performance evidence on the
implementation PRs.

## 8. Proposed explicit amendment to Decision 21 (for approval)

**Not an approved change yet.** If accepted, append a new dated
decision/amendment in [`decisions.md`](decisions.md) rather than
silently rewriting the original 2026-11-01 decision history.

> For the proximal **outputs** of finite-radius bounded coupled
> ball/sphere projection families named in E3, a non-finite argument
> entry that participates in a nonconstant effective coupled block
> yields NaN for that entire dependent block. This is an explicit
> decision **not to extend** those projections to infinity, even if
> a unique finite-input limit exists. The value at the original
> argument still follows the usual E3 formula, and
> `value_and_prox!` follows O2 at its stored output.
> Structurally independent results (singleton feasible sets,
> hard-fixed coordinates/groups, independent blocks) are evaluated
> normally under E1/E2/E4. Separable proximal maps and unbounded
> coupled proximal maps retain their previously documented E3
> semantics. This amendment does not change which step endpoints
> P1 admits.

This is **narrower** than the reviewer's phrase "all coupled
blocks": it avoids rewriting established `NormL2`
extended-real semantics and separable clipping. If a reviewer
wants to include other bounded coupled families (polyhedra,
spectral projections, Stiefel, etc.), they should propose explicit
scope and compatibility analysis; it is not an implication
of this narrow amendment.

## 9. Questions for the next reviewer (please challenge aggressively)

1. **Policy A itself:** Is preserving an exceptional signal more valuable
   than returning a uniquely determined finite limit for bounded
   projections? Would it be better for every caller to check `isfinite(x)`
   themselves? Provide representative solver workflows, not only formulas.
2. **Decision 21 amendment:** Is the proposed deliberate non-extension
   clearly identified as an amendment rather than a consequence of
   "documented limit"? Is a formal decision-record addendum needed
   before accepting any E3 normative wording?
3. **Scope:** Should A cover only direct+constrained ℓ1/ℓ2
   ball/sphere projections, **all bounded coupled projections**, or
   **all coupled proximal operators**? Identify a precise counterexample
   to the broader scopes, especially `NormL2` and other
   unbounded functions.
4. **Meaning of dependence:** Is the "structurally fixed under all
   finite steps" exception mathematically robust under nested wrappers,
   zero penalty, hard coordinate/group weights, and
   `SymmetricSparseIterate` argument symmetrization?
   Should NaN in a hard-fixed input be ignored consistently
   with E2? Could this require a separate issue even if A is accepted?
5. **Zero-radius:** Do both `IndBallL1(0)` and
   `IndBallL2(0)` intentionally return zero at **NaN**
   input as well as ±Inf, or should E2 take a stricter view?
   If the latter, precisely articulate why a structurally
   constant map "depends" on its NaN input.
6. **Simultaneous accepted `γ=Inf, x=Inf`:**
   Is the joint-path criterion correct and does it coexist with
   A's deliberate refusal of a unique *fixed-γ* limit?
   Are there valid endpoint/penalty combinations for which
   a constant finite output should nonetheless be returned?
7. **P1:** Confirm existing convex ball `γ=Inf`
   minimizer projections and nonconvex penalized sphere rejection.
   Are there parameter-degenerate effectively indicator spheres
   that must receive a named `γ=Inf` endpoint?
8. **O2/O6/O7 and classes:** Is a wholly NaN stored projection
   with a NaN return value required even for indicator functions?
   Verify validation leaves all outputs unchanged and
   exact aliasing works for every supported container.
9. **Numerics/performance:** Is a nonfinite reduction a reliable
   exceptional-branch trigger in each operator? How do we prevent
   treating finite overflow as input Inf and avoid extra scans
   or allocations on the ordinary finite path?
10. **Compatibility:** Would any downstream caller legitimately
    supply infinite sentinels and expect limiting projections?
    Is moving from a documented all-NaN exceptional policy to
    finite limits later an observable breaking change?
11. **Existing sources:** The external library comparison is
    non-normative. Do not infer agreement on A from the
    other packages' accidental `[NaN,0]` outputs.

Please give concrete counterexamples, a proposed **minimal
replacement wording** if A is too broad, and a verdict
("approve A", "narrow A", or "choose B") before any
implementation begins. Leave #311/#469 open and PR #508
draft until the design decision is explicit.
