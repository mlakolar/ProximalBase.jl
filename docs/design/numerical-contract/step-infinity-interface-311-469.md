# Step endpoints and infinite arguments: staged determined-limit design (#311, #469)

**Status:** accepted by the maintainer on 2026-11-02 as
[decision 48](decisions.md); **Stage 2 (the direct `IndBallL1`, `IndBallL2`, `IndSphereL2` and `IndSimplex` on host arrays) is implemented** in [PR #529](https://github.com/mlakolar/ProximalBase.jl/pull/529). **Stage 3A (`Constrained(NormL1, IndBallL2/IndSphereL2)`) is implemented** in [PR #545](https://github.com/mlakolar/ProximalBase.jl/pull/545), and **Stage 3B (`Constrained(GroupNormL2, IndBallL2/IndSphereL2)`) is implemented** in [PR #557](https://github.com/mlakolar/ProximalBase.jl/pull/557), merged to master (squash, 36f21cbb). The remaining constrained families (`IndBallL0`, `ZeroFunction`), the device kernels of the ℓ1 ball and the simplex, and wrappers that return the generic indicator value are S4/#366 conformance gaps. The rules are E1, E2, E3
and P1 in [`numerics.md`](../../src/numerics.md), and their changes are listed
in [Compatibility and migration](../../src/migration.md). Implementation is
tracked in [#311](https://github.com/mlakolar/ProximalBase.jl/issues/311),
[#469](https://github.com/mlakolar/ProximalBase.jl/issues/469) and
[#366](https://github.com/mlakolar/ProximalBase.jl/issues/366). This note
records the derivations and the review on
[PR #508](https://github.com/mlakolar/ProximalBase.jl/pull/508) and the
[decision recommendation on #469](https://github.com/mlakolar/ProximalBase.jl/issues/469#issuecomment-6594432492).

**Decision:** **Option B, staged.** Return a documented
path-independent limiting proximal point at an infinite argument when it
is determined, and NaN in the dependent entries otherwise. Do not invent
a relative rate for several infinities; do not accidentally treat
finite-input arithmetic overflow as a genuine infinity. Preserve the
already-approved [Decision 21](decisions.md) rather than create a new
exception to it. This note is a derivation, conformance plan, and reviewer
record, not a claim that all affected implementations currently comply.

Relevant rules: [`numerics.md`](../../src/numerics.md) E1–E4,
P1–P2, O2, O6–O7, S4, M2, R1–R3, C1–C3;
[#366](https://github.com/mlakolar/ProximalBase.jl/issues/366)
(G2 and G3);
[#311](https://github.com/mlakolar/ProximalBase.jl/issues/311);
[#469](https://github.com/mlakolar/ProximalBase.jl/issues/469);
[merged #460](https://github.com/mlakolar/ProximalBase.jl/pull/460);
[merged #304](https://github.com/mlakolar/ProximalBase.jl/pull/304).

## 1. The decision and its limits

The two issues are related, but they concern distinct input dimensions.

- **#311 (P1/E1/O6):** validate the step. A negative, NaN, or
  *unsupported* zero/infinite step is an `ArgumentError`
  **before writing any output**, even if the input also contains an
  infinite or NaN entry. A named `γ = 0` or
  `γ = Inf` endpoint remains supported and must be
  interpreted according to its family.
- **#469 (E3):** interpret an argument with `±Inf`.
  An operator can have a uniquely determined extended-real proximal
  limit even when a naive formula computes `0*Inf` or
  `Inf-Inf`. If the limit depends on the relative growth
  of multiple infinite inputs, the dependent output is NaN.
- **E2:** a NaN argument propagates according to *which output entries
  depend on that input* after hard-domain restrictions, rather than
  according to whichever entries an implementation happened to write.
- **O2:** the value returned by `value_and_prox!` is
  `f(y)` at the **stored proximal output** (`y`),
  not `f(x)` at the original infinite/NaN input.

A mathematical limit can differ from the behavior of the current
implementation. **Option B is a target**, not permission to state a
test already passes. S4 says that function representations obey one
numerical contract; mismatches during staged implementation are
**explicit gaps tracked under #366**, not endorsed exceptions.

## 2. Why B rather than A or C?

### 2.1 Equivalent functions must agree

On one coordinate, all of these indicators describe `[-1,1]`:

~~~julia
IndBallL1(1)
IndBallL2(1)
IndBallLinf(1)
IndBox(-1, 1)
~~~

The one-dimensional projection at `x = +Inf` has
the unique limiting point `1`.
The reviewer confirmed on master that the first two implementations
return `[NaN]`, while `IndBox` returns `[1]`.
A narrowed all-NaN policy for balls would **preserve this difference
between mathematically identical functions**.

Likewise, on a positive finite-radius sphere and at a finite step,

~~~julia
IndSphereL2(1)
Constrained(NormL1(0), IndSphereL2(1))
~~~

describe the same projection. A direct-only Option C would make
their infinity results disagree. Staged B also has **temporary**
differences, but they are tracked conformance gaps to be resolved,
not two intentional long-term contracts.

### 2.2 An all-NaN rule is not a consistent safety signal

A proposed all-NaN rule for ball/sphere projections might help a
finite-iterate solver detect an upstream divergence, but other
existing operations already erase an infinity legitimately:

- `IndBox(-1,1)` and `IndBallLinf(1)`
  clamp `+Inf` to `1`.
- `IndNonnegative` projects `-Inf` to zero.
- The reviewer verified that
  `Constrained(IndNonnegative(), IndBallL2(1))` at
  `[-Inf,2]` already returns the finite unique
  limit `[0,1]`.

Therefore a solver that requires finite iterates **must inspect
`isfinite(x)` itself before trusting a projection**.
A special all-NaN rule for balls would provide only partial
diagnostic coverage, while creating representation-dependent semantics.
This does **not** claim all finite projections are desirable in every
solver; it separates the oracle's extended-real mathematics from the
solver's input-acceptance rule.

### 2.3 B matches the approved contract

[Decision 21](decisions.md) already selects a function's formula
or limit where determined and NaN where path-dependent. E3 says
proximal outputs take **documented** limits. A deliberately all-NaN
bounded-ball exception would require amending that approved decision,
even for a unique limit. PR #508 withdrew the proposed
"Amendment 21A" instead of turning that exception into a permanent
contract. Decision 21 is unchanged; decision 48 records the rules
that apply it here.

### 2.4 Proof/test cost is real, but bounded by staging

B requires derivations, independent oracles, and storage/aliasing tests
for each family. Those are real costs and should not be minimized.
The lower-cost alternative A, however, is not proven faster in the
ordinary hot path: **both** must distinguish a true infinite input
from a finite input whose norm or intermediate reduction overflowed.
The benefit of A would mainly be fewer case derivations and tests.

Staging controls B's cost without weakening the end-state contract.
The stage numbering below follows §8 throughout: Stage 0 is the
contract review, Stage 1 is the relevant #311 step-validation work,
Stage 2 adds direct projections with independent proofs, and
Stage 3 adds the corresponding constrained representations one family
at a time. Further sign-restricted, spectral and other cases remain
on their existing E3 rules, with S4 inconsistencies tracked.

A documented NaN is itself observable behavior: moving from A to B
later may alter a solver's `isnan` checks. Selecting B now
avoids promising an intentional failure result and subsequently
changing that API commitment.

### 2.5 External packages are context, not a norm

[Cross-package review](https://github.com/mlakolar/ProximalBase.jl/pull/508#issuecomment-6593696105)
and the [independent check](https://github.com/mlakolar/ProximalBase.jl/pull/508#issuecomment-6593831566)
covered ProximalOperators.jl, PyProximal, SigPy, and Pyxu.
Their ordinary formulas often return `[NaN,0]` for
a Euclidean ball on `[Inf,0]` by incidental IEEE
arithmetic; none of the examined packages promised a general
infinity-limit contract. This is **neither** a precedent for B
nor a justification for A. ProximalBase should make its own
explicit contract decision.

## 3. E2: define dependence per output entry, not per input block

The second review identified an ambiguity between "depends",
"participates", "influential", and "effective free block".
Use one definition.

> **Definition:** A result (the function value, a gradient entry,
> or a proximal entry `y_j`) depends on input `x_i` if,
> after imposing E4's domain restrictions, there exist finite
> values for the other input coordinates and two finite values
> for `x_i` for which that result differs, with the valid
> parameters and step fixed (where a step applies).
> An input-independent result remains determined when `x_i`
> is NaN. A valid endpoint step such as `γ=Inf` is
> also a fixed step for this definition.

This is a semantic criterion. **Do not implement a generic
dependency-matrix computation** in the proximal hot path; each
function already knows its separable coordinates, hard-fixed
entries, groups, and coupled reductions.

| Example | Expected NaN-input behavior | Reason |
|:--|:--|:--|
| `IndBallL2(0)` at `[NaN,2]` | `[0,0]` | Both outputs are constant |
| `IndBallL1(0)` at `[NaN,2]` | `[0,0]` | Same singleton domain |
| Unit sphere constrained by `NormL1(1,[Inf,1])`, `[NaN,3]` | `[0,1]` | First coordinate hard-fixed; second ignores first input |
| Same function at `[3,NaN]` | `[0,NaN]` | Fixed first output independent; free second output depends on NaN |
| `IndBallL1(1)` at `[NaN,0]` | all NaN | Both free outputs can depend on the first coordinate for some finite other input |
| `NormL1(1)` at `x=[NaN], γ=Inf` | `[0]` | The admitted endpoint prox is zero for all finite inputs |
| `Constrained(NormL1(1), IndBallL2(1))` at `x=[NaN,0], γ=Inf` | `[0,0]`; joint value 0 | The fixed infinite-step endpoint projects onto the singleton minimizer |

**The fixed-step NaN case is not the same as a joint infinite-input
limit.** In the two last rows, E2 considers the endpoint map at
`γ=Inf` with `x` an unknown *finite* value. It therefore
returns the input-independent zero. At `γ=Inf, x=Inf`,
E1/E3 instead examine **joint finite approach paths**; for
ordinary soft thresholding and convex penalized balls, different
rates produce different limits, so the result is NaN.
The standalone `NormL1(1)([NaN])` remains NaN because the
function **value** depends on its input, even though its infinite-step
proximal output does not. Under O2, `value_and_prox!`
at the stored zero point returns zero.

The test `[NaN,0]` in an ordinary ℓ1 ball must not
be mistaken for structural independence of its second output:
at other *finite* second inputs, the first input affects the
threshold and hence the second projected component.

**Do not apply these NaN outcomes mechanically to Inf under B.**
For the hard-fixed sphere at `[3,+Inf]`, the
free coordinate has a **unique infinite-input limit**:
`[0,1]`, not `[0,NaN]`. E2 is a
dependency/NaN rule; E3 is a mathematical limit rule.

**Original value versus returned joint value:**
`IndBallL2(0)([NaN,2])` is NaN under E2,
and `IndBallL2(0)([Inf,2])` is Inf under
E3 (the input is outside the singleton real set).
But `value_and_prox!` returns `0` when it
stores `[0,0]`, in accordance with O2. This
is not an exception to indicator membership semantics.

## 4. E3: direct bounded projections, with proofs and boundaries

### 4.1 Direct ℓ1/ℓ2 balls and ℓ2 sphere

For a finite positive radius and exactly one independent infinite
coordinate `k`, with all other coordinates finite:

~~~text
IndBallL1(r):  limit ±r e_k
IndBallL2(r):  limit ±r e_k
IndSphereL2(r): limit ±r e_k
~~~

For the Euclidean ball/sphere and finite input
`x(M)=(M,a)` with fixed finite `a`,

~~~text
P(x(M)) = r x(M)/‖x(M)‖₂ -> (r,0)  as M→+Inf.
~~~

For the ℓ1-ball, the soft-threshold level approaches
`M-r` so the one dominant coordinate retains radius
`r`. A large *finite* threshold or step does not
change this conclusion when the step remains fixed finite.

For two independent infinite coordinates the limit need not exist:

~~~text
ℓ2 sphere, r = 1:
  x=(M,M)  -> (1/√2,1/√2)
  x=(M,2M) -> (1/√5,2/√5)

ℓ1 ball, r = 1:
  x=(M,M)  -> (1/2,1/2)
  x=(M,2M) -> (0,1)
~~~

All approach the same extended-real argument
`(+Inf,+Inf)` but select different directions.
**Do not** choose the equal-rate direction as a tie rule.
NaN is required in the dependent coupled outputs when the
limit is path-dependent.

The radius-zero ℓ1/ℓ2 balls have singleton feasible set
`{0}` and therefore return zero for any input,
including NaN and ±Inf, consistent with E2's dependence
definition. The standalone indicator value at the original
exceptional point remains governed by E2/E3; see §3.

### 4.2 Symmetric storage and asymmetric input

A `SymmetricSparseIterate` stores an off-diagonal coordinate
once, but it counts twice in the full matrix norm (M2).
When one stored *off-diagonal degree of freedom* dominates:

~~~text
ℓ1:  2|a| ≤ r   =>   a -> ±r/2
ℓ2:  2a² ≤ r²   =>   a -> ±r/√2
~~~

Each corresponding symmetric matrix entry has the same value.
An infinite mirrored pair is **one** independent stored degree of
freedom, not two. This arithmetic is corrected from the first
PR review; keep a regression for the ℓ1 case.

When an asymmetric input has `(+Inf,-Inf)`
in a symmetric pair, symmetrization can involve
`Inf-Inf`, so E2's NaN propagation applies. Do
not infer a sign from an ambiguous symmetric part.

For the ℓ1-sphere negative-margin finite-input fallback, the
correct symmetric single-support score is
`sqrt(m_i) * (|x_i|-threshold_i)`, **not**
`(|x_i|-threshold_i)/sqrt(m_i)`; see
[#469's oracle correction](https://github.com/mlakolar/ProximalBase.jl/issues/469#issuecomment-6578835884).
Preserve this independently of the new infinity tests.

### 4.3 Simplex: do not reuse the symmetric-ball infinity rule

`IndSimplex(r; equality=true/false)` has
nonnegative coordinates and a total mass equal to or bounded
by `r`. The reviewer found that it shares
`_simplex_threshold!` with `IndBallL1`
and currently has the same partially NaN defect at
`[Inf,0]`. Include **both equality and inequality**
variants in the direct implementation.

At finite `r>0`:

- Exactly one **independent** `+Inf` coordinate
  (others finite or `-Inf`) selects the vertex `r e_k`
  for a vector; on symmetric storage, a single off-diagonal
  independent coordinate takes `r/2` in each mirrored entry
  under M2 (not `r` in each entry).
- A `-Inf` coordinate receives zero when the remaining
  finite/positive-infinite coordinates can carry the required mass.
  If no `+Inf` remains, project the remaining finite
  coordinates onto the corresponding reduced simplex.
- With two or more independent `+Inf` coordinates,
  the dependent simplex block is NaN (relative growth rates matter).
- If every independent coordinate is `-Inf`, the
  inequality simplex uniquely selects zero. For the equality
  simplex of positive radius, the output is NaN if there are
  two or more independent coordinates, but `[r]` in the
  length-one case (singleton feasible set).
- At `r=0` the feasible set is the singleton zero vector.
  Infinite/NaN input therefore has an input-independent zero
  projection.

Examples in two dimensions, radius one:

| Input | Equality simplex | Inequality simplex |
|:--|:--|:--|
| `[+Inf, 0]` | `[1,0]` | `[1,0]` |
| `[-Inf, 0]` | `[0,1]` | `[0,0]` |
| `[+Inf, -Inf]` | `[1,0]` | `[1,0]` |
| `[+Inf, +Inf]` | all NaN (dependent block) | all NaN |
| `[-Inf, -Inf]` | all NaN (for 2D, r=1) | `[0,0]` |

For the length-one **equality** simplex at finite radius
`r≥0`, the feasible set is `{r}`, so the projection
always returns `[r]`, including at `-Inf`, `+Inf`,
or NaN (E2 input independence). A length-one
**inequality** simplex is instead a clamp onto `[0,r]`.
For a symmetric matrix, one independent off-diagonal
`+Inf` coordinate gives `r/2` in both reflected entries,
not the vector's `r`.

The empty-dimensional or infinite-radius cases are not asserted
by these finite-radius formulas; separately check their
construction/feasibility rules before extending the contract.

### 4.4 Computed overflow is not an infinite input

The literal class of `x_i` is different from the class
of a reduction of finite `x`. A norm can overflow
even when every input entry is finite, and a threshold can be
unrepresentable in a narrow intermediate type. Such cases
belong to R1–R3 scaled/recovery arithmetic, **not** to
the true-Inf E3 branch. Keep normal finite-input behavior,
accuracy, and class unchanged.

## 5. Constrained families: a closed initial list

The generic `Constrained(f, IndBallL2(r))`
accepts many inner functions beyond norms: sign-restricted cones,
sparse/spectral penalties, and other representations. The
second review found 12 accepted trait classes in
`_scales_onto_ball`. It is **incorrect** to promise a
single `±r e_k` result for every such `f`.
For example, the cone case

~~~julia
prox(Constrained(IndNonnegative(), IndBallL2(1)), [-Inf, 2])
# currently [0, 1] on master, per reviewer
~~~

correctly absorbs a negative infinity and projects the remaining
finite coordinate; a blanket all-NaN or sign-symmetric formula
would be wrong.

The initial **named implementation targets** for the constrained
stage are:

| Constraint | Inner family | Reason for inclusion |
|:--|:--|:--|
| `IndBallL2(r)` | `NormL1` | Penalized ball; zero-level/domain consistency |
| `IndSphereL2(r)` | `NormL1` | Existing #460 holding behavior and finite oracle |
| `IndBallL2(r)` | `GroupNormL2` | Group coupling and hard-weight cases |
| `IndSphereL2(r)` | `GroupNormL2` | Group directions and ties |
| `IndBallL2(r)` | `IndBallL0` | Sparse indicator projection |
| `IndSphereL2(r)` | `IndBallL0` | Nonconvex sparse sphere |
| `IndBallL2(r)` | `ZeroFunction` | Equivalent to the direct ball |

These are *targets*, not verified implementations. Derive and test
each. In particular:

- **Hard-fixed coordinates/groups** are excluded before deciding
  which infinite coordinates remain active; E4 may make the remaining
  effective domain empty, which is a validation error, not NaN.
- One infinite *group* does **not** imply one independent infinite
  scalar coordinate. A group containing `[Inf,Inf]`
  lacks a unique internal direction; at `[Inf,0,1]`,
  a single infinite coordinate in the group may determine it.
- A zero-level norm does not remove its domain constraints (E4).
  Compare equivalent representations including
  `Constrained(NormL1(0), IndSphereL2(r))`.
- **Symmetric multiplicity** is part of the geometry; do not
  count a mirrored off-diagonal entry as two independent infinities.
- Cones and spectral/rank families are **not** covered by this
  seven-row closed list. Some already compute the correct limit;
  others require family-specific analysis and may still have S4/E3
  gaps in the implementation tracker.
- For parameter-constrained `γ = Inf` and infinite input, do not
  infer the answer by taking the endpoint at fixed finite x first.
  The joint limit may be path-dependent (next section).

Do **not** create a generic "limit if one infinity" method on
`BallConstrained`: the correct calculation depends on
the inner function's domain and geometry.

## 6. #311: step validity and simultaneous exceptional endpoints

### 6.1 Validation before writing

Negative and NaN steps raise `ArgumentError` before
a write, even for indicators whose projection is otherwise
step-independent. The *form* of the predicate matters:

~~~julia
γ >= 0 || throw(ArgumentError("invalid step"))  # rejects NaN
γ < 0 && throw(ArgumentError("invalid step"))   # BUG: NaN passes
~~~

Further check positivity/finite requirements for families
without zero or infinite endpoints; do not reject an admitted
scalar `γ = 0` or `γ = Inf`
unconditionally. A later block's invalid step must leave
**all** blocks untouched (O6). Reuse private
`_checkstep` and pre-write mechanisms from PR #304.

### 6.2 Named P1 endpoints

- Direct indicator projections accept scalar zero and infinite
  steps, since the projection does not change with the step.
- Named finite-radius penalized ball/sphere operators accept
  `γ=0` as the projection onto the **effective
  domain** (hard constraints survive). An empty effective domain
  raises before writing.
- Convex `Constrained(NormL1/GroupNormL2, IndBallL2)`
  at `γ=Inf` projects finite inputs onto the nonempty
  closed convex set of minimizers. The reviewer checked that
  master already does this, including zero-weight and zero-level
  cases. **Do not replace it with an error.**
- For `Constrained(NormL1/GroupNormL2, IndSphereL2)`
  at `γ=Inf`, a zero level or an effective domain containing
  an unpenalized **free coordinate or group** gives a named
  endpoint: project onto the sphere within the free
  zero-penalty coordinates/groups, respecting the sphere tie rule.
  In the all-zero-penalty case (including level zero, with any
  hard domain constraints retained) this is simply an indicator
  projection and is step-independent under E4/P1.
  A genuinely penalized case with at least one zero-weight free
  coordinate/group still has a minimizer sphere in the zero-penalty
  subspace. With every free coordinate/group positively penalized,
  `γ=Inf` is **not** a named endpoint and must raise
  before writing. This preserves existing master behavior for
  zero-level and mixed-weight cases; do not blanket-reject
  all penalized sphere instances.
- Array steps on coupled constraints remain positive and finite,
  and wrappers must validate their transformed/forwarded steps.

### 6.3 The **joint** limit, not a convenient iterated limit

For `f(z)=|z| + I_{[-1,1]}(z)` and finite
`M>0`,

~~~text
prox_{γf}(M) = min(1, max(0, M-γ)).

lim_{γ→Inf} lim_{M→Inf} prox_{γf}(M) = 1,
lim_{M→Inf} lim_{γ→Inf} prox_{γf}(M) = 0.
~~~

For ordinary soft thresholding with `λ=1`,
take `γ=t, x=t+c`: the prox equals
`c` for every fixed `c≥0`.
So `prox(NormL1(1), [Inf], Inf)`
has no path-independent joint limit and is NaN. In contrast,
at `γ=0`, `x=+Inf`, fixed
finite positive `λ`, the soft-thresholded output has
the unique `+Inf` joint limit.

Likewise for uncentered ridge with `λ>0`:

~~~text
prox_γ(x) = x/(1+γλ).
at (γ=0, x=±Inf): unique limit ±Inf;
at (γ=Inf, x=±Inf): x=cγ gives limit c/λ (path-dependent).
~~~

For `Constrained(NormL1(1), IndBallL2(1))`,
`γ=Inf` with fixed **finite** x projects onto
the minimizer `0`. At fixed finite γ and
`x=(M,0)`, `M→Inf`, the ball
projection approaches `(1,0)`. Thus at
`(γ=Inf,x=[Inf,0])` no joint limit is
determined and the dependent coupled output is NaN.
The existence of a finite step endpoint is **not** permission
to use a step-first answer at a doubly exceptional pair.

A unique joint limit is returned **where documented**; different
finite approach paths yield NaN in dependent entries. Agreement
of the two iterated limits is necessary but not sufficient to
prove a joint limit. Do not turn an accepted exceptional step
combined with an exceptional input into an `ArgumentError`
merely because the result is NaN.

## 7. Expected-result matrix (targets, NOT current behavior)

Finite radius one, Float64 and scalar step unless shown.
For values, `value_and_prox!` returns the value
at stored `y` under O2. "All NaN" denotes the
affected freely coupled block; structurally hard-fixed entries
may remain exact zero.

| Function | Argument / step | Target output | Reason |
|:--|:--|:--|:--|
| `IndBallL2(1)` | `[Inf,0]`, `γ=1` | `[1,0]`; value 0 | Unique direct limit |
| `IndBallL1(1)` | `[Inf,0]`, `γ=1` | `[1,0]`; value 0 | Fix current partial NaN |
| `IndSphereL2(1)` | `[-Inf,2]`, `γ=0/1/Inf` | `[-1,0]`; value 0 | Indicator step independence |
| `IndBallL2(1)` | `[Inf,Inf]`, `γ=1` | all NaN; value NaN | Path-dependent direction |
| `IndBallL1(0)` / `IndBallL2(0)` | `[NaN,2]` or `[Inf,2]`, `γ=1` | `[0,0]`; value 0 | Singleton-domain input independence |
| `IndSimplex(1)`, equality | `[Inf,0]`, `γ=1` | `[1,0]`; value 0 | Unique positive vertex |
| `IndSimplex(1)`, equality | `[-Inf,0]`, `γ=1` | `[0,1]`; value 0 | Exclude negative-infinite coordinate |
| `IndSimplex(1; equality=false)` | `[-Inf,0]`, `γ=1` | `[0,0]`; value 0 | Reduced inequality simplex |
| `IndSimplex(1)` | `[Inf,Inf]`, `γ=1` | all NaN; value NaN | Allocation depends on relative rates |
| `IndSimplex(1; equality=false)` | `[-Inf,-Inf]`, `γ=1` | `[0,0]`; value 0 | Unique zero projection |
| `IndSimplex(1)`, length 1 | `[-Inf]`, `γ=1` | `[1]`; value 0 | Equality simplex is the singleton `{r}` |
| `IndSimplex(1)`, symmetric output | one `+Inf` off-diagonal | both mirrored entries `+1/2` | M2 counts twice |
| `IndBallL2(1)` | finite but huge `x`, `γ=1` | accurate finite projection | R2 recovery, not Inf input |
| `IndSphereL2(1)` | `[Inf,0]`, `γ=-1/NaN` | `ArgumentError`; unchanged y | P1 before E3 |
| `Constrained(NormL1(0), IndSphereL2(1))` | `[Inf,0]`, `γ=1` | `[1,0]` | Equivalent sphere; Stage 3 conformance gap until implemented |
| `Constrained(GroupNormL2(1,[[1,2]]), IndSphereL2(1))` | `[Inf,0,1]`, `γ=1` | `[1,0,0]` | One independent infinite direction |
| Same group sphere | `[Inf,Inf,1]`, `γ=1` | all NaN | Internal group direction ambiguous |
| `Constrained(IndBallL0(1), IndSphereL2(1))` | `[Inf,0]`, `γ=Inf` | `[1,0]` | Indicator step |
| `Constrained(NormL1(1,[Inf,1]), IndSphereL2(1))` | `[Inf,3]`, `γ=1` | `[0,1]`; value 1 | Hard-fixed first coordinate |
| Same hard-fixed sphere | `[NaN,3]`, `γ=1` | `[0,1]`; value 1 | E2 independence |
| Same hard-fixed sphere | `[3,NaN]`, `γ=1` | `[0,NaN]`; value NaN | E2 per-output dependence |
| Same hard-fixed sphere | `[3,Inf]`, `γ=1` | `[0,1]`; value 1 | E3 unique limit, **not** E2 propagation |
| `Constrained(NormL1(1), IndBallL2(1))` | `[3,0]`, `γ=Inf` | `[0,0]`; value 0 | Valid convex minimizer endpoint |
| `Constrained(NormL1(0), IndBallL2(1))` | `[3,0]`, `γ=Inf` | `[1,0]`; value 0 | Whole ball minimizes |
| `Constrained(NormL1(1), IndBallL2(1))` | `[Inf,0]`, `γ=Inf` | all NaN; value NaN | No joint limit |
| `Constrained(NormL1(1), IndSphereL2(1))` | `[3,0]`, `γ=Inf` | `ArgumentError` | Every free coordinate penalized |
| `Constrained(NormL1(0), IndSphereL2(1))` | `[3,4]`, `γ=Inf` | `[0.6,0.8]`; value 0 | Zero-level effective indicator |
| `Constrained(NormL1(1,[0,Inf]), IndSphereL2(1))` | `[3,4]`, `γ=Inf` | `[1,0]`; value 0 | E4 zero-weight free direction, hard-fixed second entry |
| `Constrained(NormL1(1,[1,0]), IndSphereL2(1))` | `[3,4]`, `γ=Inf` | `[0,1]`; value 0 | Infinite-step minimizer among free zero-weight entries |
| `NormL1(1)` | `[NaN]`, `γ=Inf` | `[0]`; joint value 0 | E2 fixed endpoint map independent of x |
| `Constrained(NormL1(1), IndBallL2(1))` | `[NaN,0]`, `γ=Inf` | `[0,0]`; joint value 0 | E2 fixed infinite-step endpoint independent of finite inputs |
| `IndBallL1(1)`, symmetric storage | one +Inf off-diagonal | paired entries `+1/2` | M2 ℓ1 multiplicity |
| ℓ2 ball/sphere, symmetric storage | one +Inf off-diagonal | paired entries `+1/√2` | M2 ℓ2 multiplicity |
| `IndBox(-1,1)`, one dimension | `[Inf]`, scalar step | `[1]` | Same as length-1 balls |
| `NormL1(1)` | `[Inf]`, `γ=0` | `[Inf]` | Unique joint limit |
| `NormL1(1)` | `[Inf]`, `γ=Inf` | `[NaN]` | Nonunique joint limit |
| `NormL2(1)` | `[Inf,2]`, finite step | retain documented E3 behavior | Nonseparable does not imply NaN |

**These are design targets.** The direct sets are **Stage 2**;
the constrained cases are **Stage 3**. The intervening **Stage 1**
covers relevant #311 step validation, after **Stage 0**
contract approval. Current incompatible results are **S4/#366
conformance gaps**, not approved permanently different semantics.

## 8. Implementation sequence and acceptance gates

### Stage 0 — the documentation decision (done: PR #508, decision 48)

- Keep E1, E2, E3 and P1 changes in `numerics.md`.
- Keep proposed Amendment 21A **withdrawn** (already absent)
  and leave the approved Decision 21 intact; no new
  exception is required for B.
- Preserve mathematical proofs, cases and review history here.
- Decide the initial closed target list; do not silently promise
  uniform behavior for all twelve generic ball inner classes.
- The maintainer approved the normative wording on 2026-11-02
  (decision 48).

### Stage 1 — #311 validation for affected methods

Implement negative/NaN and unsupported-step rejection in the
direct sets and constrained families first. Preserve existing
named finite-argument `γ=0/Inf` endpoints. Reuse
the private `_checkstep` pre-write protocol from #304;
check forwarded/transformed steps in wrappers and all later
blocks before earlier block writes. Tests: scalar and array
steps (`0,-1,NaN,Inf`, huge/tiny), mixed types,
aliasing, unchanged caller-owned outputs, and method coverage.

The remainder of #311's broader sweep can proceed independently;
do not hold a mathematically independent direct E3 patch hostage
to every unrelated step validator.

### Stage 2 — direct #469 implementation

Start with `IndBallL1`, `IndBallL2`,
`IndSphereL2` and `IndSimplex`, including
equality and inequality variants and zero radius.

- Preserve the existing fast path on finite inputs; branch on
  exceptional reductions only after distinguishing true
  `NaN/Inf` inputs from finite arithmetic overflow.
- Prove and test unique one-infinity directions, relative-rate
  ambiguity, sign-sensitive simplex cases and M2 off-diagonal
  multiplicity. Use independent high-precision finite sequences
  (BigFloat) as mathematical oracles, not a copy of the algorithm
  under test. Correct the symmetric negative-margin fallback oracle
  before reuse.
- Check `prox`, `prox!`, `value_and_prox!`,
  all supported Float32/Float64 input/destination combinations,
  `SparseIterate` / `AtomIterate` /
  `SymmetricSparseIterate` where supported, aliases,
  zero/infinite weights when relevant, and all O2 value classes.
- Add a regression for the current partial-NaN simplex and ℓ1-ball
  outputs. An affected coupled block must not return a
  misleading partially NaN point solely because of
  `Inf-Inf` or `0*Inf` arithmetic.
- Benchmark ordinary finite workloads under C1–C3;
  no unconditional BigFloat, array widening or extra GPU
  synchronization. A repeatable slowdown above 5% requires
  investigation.

### Stage 3 — constrained #469 implementation

In separate reviewable PRs, implement *only* the named
closed-list families (`NormL1`, `GroupNormL2`,
`IndBallL0`, `ZeroFunction` ball) after
mathematical derivation and validation. Preserve the
finite-step/finite-input branch and the supported P1 endpoints.
Test direct-vs-constrained equalities at one/two infinities,
with zero levels/hard constraints and symmetric layout.
Track further cone/spectral/rank cases individually under
G2/#366 or their existing family trackers.

**No generic "count infinities and choose a direction" framework.**
A single group can contain multiple independent infinite
coordinates; a cone can absorb negative infinity; a spectral
decomposition has different geometry. Specialize only where the
limit is mathematically established.

**Stage 3B cost record (C2, C3).** The nonfinite test of the group ball is one extra vectorized read of the argument, over the span of `group_of`, and the group sphere checks its group norms and ungrouped entries in the existing loops, with one vectorized reduction over the argument when ungrouped entries exist. Measured on Apple M3 Pro against `origin/master` (min of 7, zero allocations on both): fully grouped dense ball about +6–8% at n = 3·10⁵, one-group sphere about +7%, all other workloads within ±5% and the sparse case at n = 10⁶ unchanged. This is below the 10% threshold of C3 and is accepted on PR #557 as the price of detecting NaN entries that a group kernel zeroes; the maintainer approved the cost by squash-merging that PR (36f21cbb).

### PR acceptance checklist

- A real, independently derived finite-input reference confirms
  every stated finite exceptional limit.
- Nonunique cases are witnessed by at least two finite sequences
  with different limits.
- E1 invalid/unsupported steps error *before* all writes;
  names, returned values, aliases, caches and blocks are checked.
- Exact structural zeros remain zero, and E2 NaN independence
  is not confused with E3 limit propagation.
- Ordinary finite inputs, especially enormous values whose norms
  overflow in narrow arithmetic, retain R1–R3 behavior.
- Performance regressions are measured rather than assumed;
  document impacted output classes and any all-Float64 changes.
- Public docstrings are updated **with** the implementation
  that makes each promise true. Every incomplete equivalent
  representation stays a named S4/#366 conformance gap.

## 9. Review record

The first review identified and corrected:

- step-endpoint-first as an unsound way to resolve
  `(γ=Inf,x=Inf)` (decision 21);
- ℓ1 off-diagonal multiplicity `r/2` rather than
  `r/√2` (M2);
- undecided commentary inside normative contract wording;
- valid `γ=Inf` minimizer projections for convex
  penalized balls.

The second review confirmed those fixes, then found:

- narrowed A creates a length-one ball-vs-box representation
  mismatch and offers only incomplete solver failure signaling;
- the `Constrained` wording was too broad (12 possible
  inner trait classes, including cones/spectral families);
- `IndSimplex` shares the ℓ1 threshold code and
  partly NaN defect;
- E2 needs one per-**output-entry** dependency definition;
- E1 should say **what to return** when a unique joint limit
  exists and name the relevant endpoints.

The next revision adopted B, incorporated those findings, and withdrew
proposed Decision 21A.

The third review of [PR #508](https://github.com/mlakolar/ProximalBase.jl/pull/508#pullrequestreview-5746851865)
verified the staged-B choice against master, including the signed
simplex patterns, the E1 joint-limit examples and the hard-fixed sphere
cases. It required four further refinements now reflected in the
normative proposal: a fixed-`γ=Inf` E2 endpoint example,
`r/2` for symmetric simplex off-diagonal storage, the
admitted `γ=Inf` zero-penalty/zero-weight sphere endpoints,
and removal of the leftover Option A restriction on indicator
steps. These are contract outcomes, not evidence that the
implementations already pass every conformance case.

### After approval

A fourth check, at `bd5459c7`, found no further issues, and the
maintainer approved the four rules on 2026-11-02 (decision 48).
The contract text is approved; the implementations are not:

- Keep #311 and #469 open until Stages 1–3 land.
- List each family that does not yet conform in #366, as S4 requires.
- In each implementation PR, check that unusual supported radii,
  element types, backends, sparse and atom layouts, and caches do not
  make a stated limit false or unrepresentable.
