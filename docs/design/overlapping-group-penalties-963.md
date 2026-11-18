# Nested and tree-structured group penalties (issue #963)

Status: research note, no implementation. Pinned code: master at `13891ccfd5c33a042c5df82d670abf1f2d786a0f`,
`legacy/full` at `eff21b6d7d33c622313122f46c6ee71693c101ee`. Numbers below are pasted from the probes in
`docs/audit/overlapping-group-penalties-963/`, whose `README.md` gives the commands and the recorded output.

## 1. Summary

**Policy.** ProximalBase supports a structured group penalty when an efficient proximal operator with a proof of
correctness exists for it. Groups with arbitrary overlap are not supported (Section 10).

**When one pass is exact.** "One pass" applies the proximal step of each group once, in an order where a group comes
after the groups it contains (Section 2). It returns the proximal point for all inputs under these conditions:

| Layout | Group functions | Step | One pass in inclusion order | Statement |
|---|---|---|---|---|
| disjoint | any closed convex function per group | scalar or diagonal | exact, in any order | Proposition 1 |
| laminar (tree, forest) | `ω‖·‖₂`, `ω ∈ [0, ∞]`, on each group that contains another; any closed sublinear function on the others | scalar | exact | Proposition 2 |
| laminar | `ω‖·‖∞` on all groups | scalar | exact | Proposition 3 |
| laminar | an `ℓ∞` group containing an `ℓ2` group | scalar | not exact on a recorded instance | Proposition 3 |
| laminar | `‖D z‖₂`, `D` diagonal, on containing groups, `D` constant on each contained group | scalar | exact | Proposition 4(a) |
| laminar | `D` not constant on a contained group that is a weighted `ℓ2` norm, positive where `D` differs | scalar | two groups: not exact for some input; the chains of Section 5: not exact for some input | Proposition 4(b), (c) |
| laminar | `ω‖·‖₂`, `ω ∈ [0, ∞]` | diagonal `τ`, constant on each group contained in a group of positive weight | exact | Proposition 5 |

Nesting alone does not make one pass exact: with weighted `ℓ2` groups the condition is on the weights inside a group.
The condition of Proposition 4(a) is sufficient; it is necessary when the contained group is a weighted `ℓ2` norm
(Proposition 4(b)), not for other contained functions (an `ℓ1` group, probe P5).

**The general weights of Bien, Bunea and Xiao.** For their weights `w_{ℓm} = √(2ℓ)/(ℓ−m+1)` and each `p ≥ 4` and
`λ > 0`, some input gives a one-pass point that is not the proximal point (Section 5(a), with a proof); for `p ≤ 3`
one pass is exact. The same holds for the weights of Yu and Bien, `1/(ℓ−m+1)²`, in rows with three or more penalized
coordinates. The paper's one-pass claim cites a result for one scalar weight per group; the accepted manuscript of the
published version keeps it, and `hierband` 1.0 computes the one-pass point. One pass can also keep bands that the
proximal point sets to zero. The proximal point can be computed to a certified accuracy (a duality gap): dual block
coordinate descent on the `p − 1` band norms needed from 1 to 7591 sweeps on our instances, and one instance had not
reached the gap after 100000 sweeps; a Newton-type method on the nonzero bands was certified in 92 of 100 instances,
88 at the first attempt with at most 9 Newton iterations and 4 at the second with up to 34 in all, and failed in the 8
instances whose dual block coordinate descent point has a nonzero group norm below `1e-9 λ w_ℓℓ` (a further attempt
with a tolerance rule for the zero prefix certified none of them). Neither has the fixed number of scalar root solves
of one pass. Section 5(e) lists the options for #965 and leaves the choice to the maintainer.

**Recommended penalties.** The sparse-group penalty (`ℓ1` plus disjoint group `ℓ2`) and a tree-structured group `ℓ2`
penalty with one scalar weight per group: both have an exact one-pass proximal operator built from the scalar group
shrink that master already has. Section 9 proposes two follow-up issues.

**Design.** (A), specialized penalties that share small kernels, is the recommendation (Section 8).

## 2. Setting and notation

- Coordinates `1, …, n`; groups `g_1, …, g_K ⊆ {1, …, n}`; `z_g` is the subvector of `z` on `g`. Groups may be equal
  (duplicates), and a coordinate may lie in no group.
- **Laminar:** any two groups are disjoint or one contains the other. A laminar family is a forest under inclusion
  (Jenatton et al., JMLR 2011, Definition 1, which also asks the groups to cover all coordinates; nothing below needs that).
- **Group function** `N_k` on `ℝ^{g_k}`: proper, closed, convex, positively homogeneous, `N_k(0) = 0` ("sublinear"). This
  covers `ω‖z‖₂`, the weighted norm `‖D z‖₂` with `D = diag(d) ≥ 0`, `ω‖z‖∞`, the zero function (weight 0) and the
  indicator of `{0}` (weight `∞`).
- `Ω(x) = Σ_k N_k(x_{g_k})`, and the proximal problem with a scalar step `t ≥ 0`:
  `prox_{tΩ}(y) = argmin_x ½‖x − y‖² + t Ω(x)`.
- **Group step** `P_k`: the proximal operator of `x ↦ t N_k(x_{g_k})`; it changes `x_{g_k}` alone. For `‖D z‖₂`
  at `v = x_{g_k}`: entries with `d_i = 0` are unchanged; with `t = 0` the step is the identity; with `t > 0` the
  entries with `d_i > 0` are set to zero when `Σ_{d_i>0} v_i²/d_i² ≤ t²`, and otherwise multiplied by `s/(s + d_i²)`,
  where `s > 0` solves `Σ_{d_i>0} d_i² v_i²/(s + d_i²)² = t²` (Bien, Bunea and Xiao, arXiv:1405.6210v1, Appendix B).
  An infinite weight stands for the indicator of `{0}` for each step `t ≥ 0` (`t · ∞ = ∞`, including `t = 0`), as on
  master (`groupnorml2.jl:44`).
- **Inclusion order:** an order in which `g_i ⊊ g_j` implies that `i` comes first. One pass is `x⁰ = y`,
  `x^k = P_{σ(k)}(x^{k−1})`; its dual blocks are `ξ_k = x^{k−1} − x^k`. The papers write this order differently (Jenatton
  et al., Corollary 6; Yu, NIPS 2013, Corollary 3; Bien, Bunea and Xiao count groups "outside-in"; #965 says
  "outer-to-inner"); all of them process the smallest groups first.

## 3. Inventory

Rows are the types, functions, operators and wrappers that take groups, index sets or a structured linear map, found by a
search of both trees (`git grep` of `struct`, `groups`, `prox_block` and the group kernels). The citation check
`check_inventory.py` reads each `path:line` at the pinned commit.

### 3.1 master (`13891ccfd`): all rows active

| Symbol | Location | Penalty or role | Layout | Oracles |
|---|---|---|---|---|
| `GroupNormL2` | `src/functions/groupnorml2.jl:52` | `λ₀ Σ_k w_k ‖x[G_k]‖₂` | nonempty disjoint index lists; weights in `[0, ∞]` | value, `prox!`, `value_and_prox!`, `prox_block!`, `kkt_violation` |
| `GroupNormL2` (default weights) | `src/functions/groupnorml2.jl:77` | `w_k` = square root of the size of `G_k` | as above | constructor |
| `prox!` | `src/functions/groupnorml2.jl:189` | one scalar group shrink per group | scalar step (`:15`); in place allowed (`:41`) | prox |
| `value_and_prox!` | `src/functions/groupnorml2.jl:191` | fused value at the output | as above | fused |
| `prox_block!` | `src/functions/groupnorml2.jl:228` | shrink of one compact group | `out === u` allowed | block prox |
| `kkt_violation` | `src/functions/groupnorml2.jl:308` | largest group score | array or `BlockIterate` point | optimality |
| `BlockIterate` (from a penalty) | `src/functions/groupnorml2.jl:86` | iterate with the penalty's groups | groups must partition `1:p` | constructor |
| `_symmetric_prox_error` | `src/functions/groupnorml2.jl:314` | refusal of `SymmetricSparseIterate` for `NormL2` and `GroupNormL2` | none | error |
| `NormL2` | `src/functions/norml2.jl:57` | `λ‖x‖₂`, one group | whole array | value, prox, fused, KKT |
| `NormLinf` | `src/functions/normlinf.jl:60` | `λ max_i x_i` in magnitude, one group | whole array | value, prox (clip), fused, KKT |
| `BlockIterate` | `src/block_iterate.jl:53` | vector with group coordinates | partition of `1:p` | storage |
| `_check_one_based_groups`, `_own_groups`, `_group_of`, `_check_partition` | `src/block_iterate.jl:109`, `src/block_iterate.jl:120`, `src/block_iterate.jl:128`, `src/block_iterate.jl:146` | axes, owned copy, nonempty disjoint groups, cover of `1:p` | lists or ranges | validation |
| `prox_block` | `src/block_prox.jl:2` | allocating block prox | compact block | block prox |
| `_threshold_product`, `_l2_zero_group`, `_l2_shrink!`, `_l2_shrink_fixed!` | `src/numerical_kernels.jl:103`, `src/numerical_kernels.jl:902`, `src/numerical_kernels.jl:1001`, `src/numerical_kernels.jl:1042` | threshold `γλw`; zero test; scalar group shrink (`y === x` allowed); zero and infinite weight, level and step | index set | kernels |
| `_l2_term`, `_l2_fused_term`, `_l2_value`, `_l2_expsum_add` | `src/numerical_kernels.jl:1083`, `src/numerical_kernels.jl:1108`, `src/numerical_kernels.jl:1140`, `src/numerical_kernels.jl:1173` | value terms `λw‖x[G]‖₂`, from the shrink, of `NormL2`, out of range | index set or array | kernels |
| `_l2_screen`, `_l2_score` | `src/numerical_kernels.jl:1184`, `src/numerical_kernels.jl:1233` | `‖g[G]‖₂ − λw` and the point-aware group score | index set | kernels |
| `SparseIterate` | `src/iterate_storage.jl:41` | working set of linear entries | linear indices | storage |
| `SymmetricSparseIterate` | `src/iterate_storage.jl:99` | `p(p+1)/2` lower-triangle variables, not an `AbstractArray` | lower triangle | storage |
| `tril_index` | `src/utils.jl:227` | `(r, c)` to a lower-triangle coordinate | index map | utility |
| `Scaled` | `src/functions/scaled.jl:100` | `α f`, folds the level of `GroupNormL2` | wrapper | forwards |
| `LeastSquaresLoss` | `src/functions/leastsquares.jl:44` | a loss with a design matrix, not a group penalty | matrix | smooth oracles |

No overlap, nested, tree or latent group penalty exists on master.

### 3.2 `legacy/full` (`eff21b6d7`): all rows preserved, none active

| Symbol | Location | Penalty or role | Layout | Oracles |
|---|---|---|---|---|
| `NormL2` | `src/functions/norms.jl:22` | `λ‖x‖₂`, with a symmetric owner path | whole array | value, prox |
| `_LowerGroupPosition` | `src/functions/norms.jl:126` | writes each symmetric entry once | lower triangle | helper |
| `NormLinf` | `src/functions/norms.jl:371` | `λ‖x‖∞` | whole array | value, prox |
| `AbstractGroupPenalty` | `src/functions/norms.jl:441` | supertype of the group penalties | disjoint groups | dispatch |
| `HomogeneousGroupNorm` | `src/functions/norms.jl:442` | convex group norms | disjoint groups | dispatch |
| `AbstractGroupNorm` | `src/functions/norms.jl:446` | alias kept for solvers | disjoint groups | dispatch |
| `GroupNormL2` | `src/functions/norms.jl:480` | `λ₀ Σ w_k ‖x[G_k]‖₂` | disjoint; symmetric owner output | value, prox, block, KKT, `restrict` |
| `GroupNormLinf` | `src/functions/norms.jl:508` | `λ₀ Σ w_k ‖x[G_k]‖∞` | disjoint | value, prox (clip), KKT |
| `_check_symmetric_groups` | `src/functions/norms.jl:808` | groups closed under transpose | symmetric output | validation |
| `restrict` | `src/functions/norms.jl:891` | penalty on a coordinate subset | disjoint groups | restriction |
| `value_and_prox!` | `src/functions/norms.jl:955` | group penalties on `BlockIterate` | partition | fused |
| `NormL1Fused` | `src/functions/norms.jl:1038` | `λ₁ Σ x_k` in magnitude plus all-pairs fusion | vector or fibers | value, prox |
| `SparseGroupNormL2` | `src/functions/sparsegroup.jl:41` | `λ₁‖x‖₁ + λ₂ Σ w_k ‖x[G_k]‖₂`, one-pass prox | disjoint groups | value, prox, block, KKT |
| `value_and_prox!` | `src/functions/sparsegroup.jl:265` | soft threshold, then group shrink | disjoint groups | fused |
| `prox_block!` | `src/block_prox.jl:35` | generic checked block prox | compact block | block prox |
| `_prox_block!` | `src/block_prox.jl:71` | sparse-group block step | compact block | block prox |
| `BlockIterate` | `src/block_iterate.jl:48` | group coordinates | partition | storage |
| `GroupMCP`, `GroupSCAD`, `IndBallGroupL0` | `src/functions/nonconvex.jl:381`, `src/functions/nonconvex.jl:405`, `src/functions/nonconvex.jl:705` | group MCP, group SCAD, at most `s` nonzero groups (nonconvex) | disjoint | value, prox |
| `SliceSum`, `FiberSum` | `src/functions/fibers.jl:58`, `src/functions/fibers.jl:182` | `Σ_k f_k(x_k)` over slices; `Σ_ij f(x[i,j,:])`, optional symmetric fibers | slices; fibers (disjoint) | per slice or fiber |
| `TotalVariation1D`, `FusedLasso`, `SLOPE` | `src/functions/ordered.jl:23`, `src/functions/ordered.jl:187`, `src/functions/ordered.jl:518` | 1-D total variation, `ℓ1` plus 1-D fusion, sorted `ℓ1` | ordered vector | value, prox |
| `IndPattern`, `IndBanded`, `IndToeplitz` | `src/functions/matrixsets.jl:30`, `src/functions/matrixsets.jl:124`, `src/functions/matrixsets.jl:224` | support in a pattern, banded and Toeplitz matrices | index pattern, bandwidths, diagonals | prox |
| `SeparableSum`, `Sum` | `src/calculus.jl:2129`, `src/calculus.jl:2224` | blocks of a tuple; `Σ f_k(x)`, "no proximal operator in general" | tuple; whole argument | blockwise; smooth oracles |
| `IndIntersection` | `src/functions/intersection.jl:53` | Dykstra; `converged` is not a certificate (`:22`–`:30`) | sets | iterative prox |
| `Precompose`, `LinearImage`, `IndGraph`, `IndAffine`, `IndPolyhedron`, `LinearConstrained`, `AffineConstrained` | `src/composition.jl:61`, `src/image.jl:85`, `src/functions/graph.jl:62`, `src/functions/affine.jl:38`, `src/functions/polyhedron.jl:58`, `src/constrained.jl:114`, `src/affineconstrained.jl:40` | wrappers and sets defined by a linear map | linear map or matrix | value, prox or projection |
| `IndSOC`, `IndRotatedSOC`, `IndExpPrimal`, `IndExpDual` | `src/functions/cones.jl:32`, `src/functions/cones.jl:200`, `src/functions/cones.jl:377`, `src/functions/cones.jl:413` | second-order, rotated and exponential cones, per group | disjoint groups | projection |
| `BallConstrained`, `SphereConstrained`, `_GroupReading` | `src/ballconstrained.jl:95`, `src/ballconstrained.jl:101`, `src/ballconstrained.jl:859` | `f` on an `ℓ2` ball or sphere, `GroupNormL2` included; its group reader | wrapper | prox |

Neither tree holds an arbitrary-overlap or latent group penalty; the rows from `SliceSum` on are outside the families.

## 4. Conditions for an exact one-pass proximal operator

**Lemma (composition).** Let the groups be laminar, each `N_k` sublinear, `t ≥ 0`, and `σ` an inclusion order. Suppose
that for each pair `k` before `j` with `g_k ⊆ g_j` and all `x`,

  (C) `∂N_k(P_j(x)_{g_k}) ⊇ ∂N_k(x_{g_k})`.

Then one pass returns `prox_{tΩ}(y)` for all `y`.

*Proof.* Write `M_k = t N_k`, sublinear (with `t · ∞ = ∞`); (C) for `N_k` gives it for `M_k` when `t > 0`, and for
`t = 0` `M_k` is `0` or the indicator of `{0}`, for which (C) holds because `P_j` keeps a zero subvector zero. The
optimality condition of step `k` gives `ξ_k ∈ ∂(M_k ∘ π_k)(x^k)`, with `π_k` the restriction to `g_k`.
A later step `j` either has `g_j ∩ g_k = ∅` and leaves `x_{g_k}` unchanged, or, by laminarity and the order,
`g_j ⊇ g_k`, and then (C) keeps `ξ_k` a subgradient. So `ξ_k ∈ ∂(M_k ∘ π_k)(x^K)` for each `k`, and
`y − x^K = Σ_k ξ_k ∈ ∂(tΩ)(x^K)` (a sum of subgradients is a subgradient of the sum). This is the optimality condition
of the strongly convex proximal problem, whose minimizer is unique. ∎ This is Yu (NIPS 2013), Theorem 1, applied group
by group; Jenatton et al., Lemma 4 and Proposition 5, is the same argument for scalar `ℓ2` or `ℓ∞` groups.

**Fact.** For a sublinear `N`, `∂N(c z) = ∂N(z)` for `c > 0` and `∂N(0) ⊇ ∂N(z)`. So (C) holds when `P_j` multiplies
`x_{g_k}` by a scalar `c ≥ 0`.

**Proposition 1 (disjoint groups).** Disjoint groups, any proper closed convex `N_k` (diagonal weights, any norm), scalar
step `t ≥ 0` or diagonal step. The problem separates over the groups and the unpenalized coordinates: the proximal point is
the group steps in any order, and coordinates in no group are unchanged. *Needed:* disjointness, for the separation argument (overlapping groups are Propositions 2–5).
Master's `GroupNormL2` implements this for scalar `ℓ2` weights; a zero weight copies the group and an infinite weight
or level fixes it at `+0` (`groupnorml2.jl:44`, kernel `numerical_kernels.jl:1042`). Probes D1 (recorded instance,
difference `4.44e-16` from the harness) and D2 (200 instances).

**Proposition 2 (laminar, scalar `ℓ2` on containing groups).** Laminar groups (forests, duplicates and unpenalized
coordinates allowed), an inclusion order, a scalar step `t ≥ 0`. Each group that contains another group of the
family, an equal group placed before it included, has `N_j = ω_j‖·‖₂` with `ω_j ∈ [0, ∞]`; the other groups have any
sublinear `N_k`. Then one pass is exact. *Proof.* `P_j` multiplies `x_{g_j}` by `c = (1 − tω_j/‖x_{g_j}‖)₊`:
`c = 0` for `ω_j = ∞` (for each `t`), `c = 1` for `ω_j = 0` or for `t = 0` with `ω_j < ∞`; the Fact and the Lemma
apply. ∎ With positive weights, one norm for all groups and a cover, this is Jenatton et al., Proposition 5 (`ℓ2`
case) and Yu, Corollary 3. *Hypotheses:*
- laminar: needed for the Lemma; without it see Section 10;
- the order: needed. The sparse-group penalty `|x₁|/2 + |x₂|/2 + ‖x‖₂/2` at `y = (1, 0.1)`, `t = 1`, has the proximal
  point `0`; the group step first, then the coordinates, gives `(0.0024814048950054326, 0)` (probe W2);
- scalar `ℓ2` on the containing groups: needed when the contained functions are arbitrary sublinear functions; an `ℓ∞`
  group containing an `ℓ2` group fails (Proposition 3), while `ℓ∞` inside `ℓ∞` (Proposition 3) and some diagonal
  weights (Proposition 4) are exact;
- zero and infinite weights, a group at its threshold (`‖v‖₂ = tω`), duplicates, forests, unpenalized coordinates and
  five nested groups are in the recorded instance L1; 300 random laminar families pass (L3).

*Corollary (two-level, sparse-group).* `Ω(x) = Σ_i c_i|x_i| + Σ_g ω_g‖x_g‖₂` with disjoint groups `g`, coordinate
weights `c_i ≥ 0` that may differ inside a group, `ω_g ∈ [0, ∞]`: the proximal point is the soft threshold at `t c_i`,
then the group shrink. The singletons are groups that contain no other. The legacy `SparseGroupNormL2` computes this
(`sparsegroup.jl:265`). Probes W1, W2.

**Proposition 3 (`ℓ∞` groups).** Laminar groups, an inclusion order, `N_k = ω_k‖·‖∞` with `ω_k ∈ [0, ∞]` for all
groups, scalar step: one pass is exact (Jenatton et al., Proposition 5, `ℓ∞` case). *Proof of (C).* `P_j` clips
`x_{g_j}` at a level `θ > 0`, sets it to zero, or leaves it unchanged (`ω_j = 0`, or `t = 0` with `ω_j < ∞`); clipping keeps the
signs and enlarges the set of coordinates of largest magnitude in `g_k`, so `∂‖·‖∞` does not shrink. ∎ Probe M1
(300 families). *One norm for all groups* (the setting of Lemma 4): an `ℓ2` group inside an `ℓ∞` group is not covered, and fails: groups `{1,2}` (`ℓ2`, weight 1) inside
`{1,2,3}` (`ℓ∞`, weight 1) at `y = (3, 1, 0)`, `t = 1`: one pass `(1.0513167019494862, 0.6837722339831621, 0)`, the
certified reference `(1.1055728090000843, 0.5527864045000421, 0)` (probe M3). An `ℓ2` group containing `ℓ∞` groups is
Proposition 2 (probe M2). For `ℓq`, `q ∉ {1, 2, ∞}`, Jenatton et al., Appendix C (Lemmas 12, 13), give a two-group
counterexample; not probed.

**Proposition 4 (diagonal weights inside a group).** Laminar groups, an inclusion order, scalar step `t ≥ 0`. Each group
that contains another group of the family, an equal group placed before it included, has `N_j(z) = ‖D_j z‖₂` with
`D_j = diag(d_j)`, `d_j ≥ 0` finite; the others any sublinear `N_k`.

- (a) If for each pair `k` before `j` with `g_k ⊆ g_j` the weights `d_j` are constant on `g_k` (zero allowed), one
  pass is exact. *Proof.* `P_j` multiplies `x_{g_k}` by `s/(s + a²)` when `d_j ≡ a` on `g_k`, by `0` when group `j`
  is set to zero and `a > 0`, and by `1` when `a = 0` or `t = 0`; the Fact and the Lemma apply. ∎ For a tree the
  condition reads: the weights of a group are constant on each child group and arbitrary on its own coordinates. Probe
  P1 (300 families, zero weights and duplicates included). The condition is sufficient, not necessary: an `ℓ1` group
  inside `‖diag(1, 3, 2) z‖₂` is exact (P5), since a positive diagonal scaling keeps the signs and zeros that `∂‖·‖₁`
  depends on. Without the duplicate clause a later duplicate could be any function: `‖z‖₂` then `‖z‖∞` on `{1,2}` at
  `y = (3, 1)`, `t = 1`, has the proximal point `(2 − 2/√5, 1 − 1/√5)`, which one pass misses (P6).
- (b) *Necessity for two groups.* Let `g_1 ⊆ g_2`, `N_1 = ‖D_1·‖₂`, `N_2 = ‖D_2·‖₂`, and `i_1, i_2 ∈ g_1` with
  `d_1(i_1), d_1(i_2) > 0` and `d_2(i_1) ≠ d_2(i_2)`. Then for each `t > 0` some `y` gives a one-pass point that is not
  the proximal point. *Proof.* Take `y = α u`, `u` supported on `{i_1, i_2}` with both entries nonzero, `α` large.
  Both steps and the proximal point keep zeros outside `{i_1, i_2}` (weighted `ℓ2` norms grow with each magnitude).
  For large `α` neither step sets its group to zero, so `x² = Λ x¹` with `Λ = diag(f_{i_1}, f_{i_2})`,
  `f_i = s/(s + d_2(i)²)`, `s ∈ (0, ∞)`, `f_{i_1} ≠ f_{i_2}`. Both norms are differentiable at `x²`, and one pass
  gives `y − x² = t∇N_1(x¹) + t∇N_2(x²)`; so `x²` is optimal if and only if `∇N_1(x¹) = ∇N_1(x²)`. With
  `∇N_1(z) = D_1² z/‖D_1 z‖₂` and `D_1` positive on `{i_1, i_2}`, this forces `x² = c x¹`, `c > 0`, which `Λ` and the
  two nonzero entries of `x¹` exclude. ∎ Probe P2 (`d_1 = (1, 2)`, `d_2 = (1, 3, 2)`, `y = (2, 3, 1)`, `t = 0.3`:
  refuted by the reference and by a gradient entry of the objective).
- (c) *A failure test for more groups.* Let all groups be weighted `ℓ2` norms, numbered in the order of the pass, and
  let `y` have `D_k y_{g_k} ≠ 0` for each `k`, so that each `N_k` is twice differentiable near `y` with Hessian `H_k`. Put
  `C(y) = Σ_k H_k(y) Σ_{j>k} ∇N_j(y)`. If `C(y) ≠ 0`, then for each `t > 0` some input gives a one-pass point
  that is not the proximal point. *Proof.* A step moves its input by at most `t max_i d_k(i)` (the Lipschitz constant
  of `t N_k`), so each iterate is within `O(t)` of `y`, where `D_k y_{g_k} ≠ 0`; for small `t` no step sets its group
  to zero and the objective is differentiable at `x^K`. Then `x^k = x^{k−1} − t∇N_k(x^k)` and `y − x^K = t Σ_k ∇N_k(x^k)`. The gradient of the objective at `x^K` is
  `G = t Σ_k (∇N_k(x^K) − ∇N_k(x^k))`, and `x^k − x^K = t Σ_{j>k} ∇N_j(y) + O(t²)`, so `G = −t² C(y) + O(t³)`,
  nonzero for small `t`. One pass and the proximal operator satisfy `T_t(y) = t T_1(y/t)` (each `N_k` is positively
  homogeneous), so a failure at `(y, t)` is a failure at `(c y/t, c)` for each `c > 0`. ∎ We do not prove that a pair
  violating (a) gives `C ≠ 0` in all families; Section 5(a) shows it for the chains of interest.
- *Duplicates* make the order matter: `g_1 = g_2 = {1,2,3}` with `d_1 = (1, 2, 3)` and the scalar weight `1.5` on
  `g_2` is exact with `g_1` first (P3) and refuted with `g_2` first (P4).

**Proposition 5 (diagonal step).** The problem `argmin_x ½ Σ_i (x_i − y_i)²/τ_i + Ω(x)`, `τ_i > 0`, with laminar
groups and scalar weights `ω_k ‖·‖₂`, `ω_k ∈ [0, ∞]`. With `z = x/√τ` it is the problem of Proposition 4 at `y/√τ` with the weights
`d_k = ω_k √τ` on `g_k`, and the group steps in `x` (each a weighted shrink with an ellipsoid root, not a closed-form
scalar shrink) correspond to those in `z`. So one pass is exact when `τ` is constant on each group contained in a group
of positive weight, and for two groups of positive finite weight it is needed (Proposition 4(b)); an infinite weight
sets its group to zero for each `τ`. For the sparse-group penalty the contained groups are singletons, so one pass is
exact for any `τ` (S1). Probes S2 (300 families) and S3 (refuted for
`τ = (1, 4, 1)` on `{1,2} ⊂ {1,2,3}`). Master's `GroupNormL2` takes scalar steps (`groupnorml2.jl:15`).

## 5. The general weights of Bien, Bunea and Xiao

Equation numbers are those of arXiv:1405.6210v1; the published version numbers the problem (3) as (2.1) and the
weights (6), (7), (8) as (2.4), (2.5), (2.6).

**Reduction to band norms.** For the banding penalty `λ Σ_ℓ (Σ_{m≤ℓ} w_{ℓm}² ‖X_{s_m}‖²)^{1/2}` on a `p × p` matrix,
with `s_m` the two subdiagonals `p − m` away from the diagonal and `g_ℓ = s_1 ∪ … ∪ s_ℓ`, the weights are constant on
each band. For fixed band norms `ρ_m` the distance term is minimized by `X_{s_m} = (ρ_m/r_m) S_{s_m}`, `r_m = ‖S_{s_m}‖₂`
(`X_{s_m} = 0` when `r_m = 0`), so the problem is the chain `{1} ⊂ {1,2} ⊂ … ⊂ {1, …, p−1}` in `ℝ^{p−1}` at `r ≥ 0` with
the diagonal weights `W[ℓ, 1:ℓ]` (its minimizer is nonnegative), and the diagonal of `S` is kept. At band-norm points
and dual blocks that vanish on the bands with `r_m = 0` (as one-pass and dual block coordinate descent iterates do),
the map to matrices keeps the primal and dual values, so their duality gaps transfer. Probe B1 (`p = 5`: matrix gap `1.78e-16` at the band-norm solution scaled back). The disjoint
control `w_{ℓm} = √(2ℓ) 1{ℓ = m}` is `GroupNormL2` on the bands (B2, difference `1.11e-16`).

**(a) The counterexample, for each `p ≥ 4`.** For the chain, Proposition 4(a) asks `w_{ℓm}` to be constant in `m < ℓ`.
The general weights give `w_{3m} = (√6/3, √6/2)` on bands 1, 2, so the condition fails once `ℓ = 3` exists, that is
`p ≥ 4`; for `p = 3` the contained group `{1}` is one band and one pass is exact.

*Claim.* Let `K = p − 1 ≥ 3`, `w₁₁ > 0`, and `w_{ℓ1}, w_{ℓ2} > 0` for `ℓ ≥ 2` with `w_{j2} > w_{j1}` for each `j ≥ 3`. Then for each
`λ > 0` some band norms `r` give a one-pass point that is not the proximal point. *Proof.* Apply Proposition 4(c) at
`y = (y₁, y₂, 0, …, 0)`, `y₁, y₂ > 0`; all iterates stay on bands 1, 2, and each group norm is nonzero. Group 1 is
`w₁₁|z₁|`, with zero Hessian near `y`. For `ℓ ≥ 2` the group acts on `{1, 2}` as `‖diag(w_{ℓ1}, w_{ℓ2}) z‖₂`, whose
Hessian at `y`, restricted to bands 1, 2, annihilates `y`, so it is `c_ℓ q qᵀ` with `q = (−y₂, y₁)` and `c_ℓ > 0`
(strict Cauchy–Schwarz, as `q` is not parallel to `y`); its other entries multiply components of
`Σ_{j>ℓ} ∇N_j(y)` that are zero. And `qᵀ∇N_j(y) = y₁y₂(w_{j2}² − w_{j1}²)/n_j`, `n_j = N_j(y)`. So
`C(y) = q Σ_{2≤ℓ<j≤K} c_ℓ y₁y₂ (w_{j2}² − w_{j1}²)/n_j`, a sum of positive terms, nonzero once `K ≥ 3`. ∎ The general
weights have `w_{j2}/w_{j1} = j/(j−1)` and the Yu–Bien weights `j²/(j−1)²`, both above 1, so both fail for each
`p ≥ 4` (Yu–Bien: rows with `K ≥ 3` penalized coordinates). Probe E checks `G/t² = −C(y)` in `BigFloat` at `t = 1e-40`
for `K = 3, 5, 10, 30` and `C = 0` at `K = 2`. The recorded instance (`p = 4`,
`r = (3.3908789763847373, 2.723510817894522, 1.102998740113383)`, `λ = 0.9768147919269893`), probes B3 and B4:

| | band norms | largest gradient entry |
|---|---|---|
| one pass (`Float64`) | `(0.880669128402111, 0.43980506851334233, 0.15773452285496928)` | `0.244` |
| reference (14 sweeps, BigFloat gap `8.02e-15`) | `(0.8339820745637797, 0.49606635020397205, 0.15980235543181953)` | `1.39e-07` |

The distance `0.0731` exceeds `2√(2 gap) = 2.53e-07`, all group norms are nonzero at both points, and the objective
excess of the one-pass point is `0.01022428370907346`. On 100 seeded instances per size, one pass is refuted in 81
(`p = 4`), 92 (`p = 6`) and 98 (`p = 10`) and accepted (BigFloat gap) in the others; it is accepted in 100 of 100 for
`p = 3` and for the basic weights `√(2ℓ)` and the disjoint control at `p = 4, 6, 10` (probe F).

**Zero bands.** Let `L*` be the number of leading zero bands of the proximal point `ρ*`, and `B°_ℓ` the dual ball of
group `ℓ` (`‖v_{1:ℓ}/W[ℓ, 1:ℓ]‖₂ ≤ 1`). Then `ρ*_{1:L} = 0` if `r_{1:L} ∈ λ Σ_{ℓ≤L} B°_ℓ`, and
`L* = max{L : r_{1:L} ∈ λ Σ_{ℓ≤L} B°_ℓ}`. *Proof.* If `r_{1:L}` has that form, let `ρ̃` minimize over the bands
`L+1, …` with the first `L` held at zero; at `(0, ρ̃)` each group `ℓ > L` has a subgradient that vanishes on bands
`1..L` (its gradient `λW_ℓ²ρ/n_ℓ`, or the restricted dual element padded with zeros), and the groups `ℓ ≤ L` absorb
`r_{1:L}`, so `(0, ρ̃)` satisfies the optimality condition. Conversely the gradients of the nonzero groups `ℓ > L*`
vanish on bands `1..L*`, so the optimality condition there reads `r_{1:L*} ∈ λ Σ_{ℓ≤L*} B°_ℓ`. ∎ If the one-pass step of
group `L` sets it to zero, its dual blocks `ξ_1, …, ξ_L` decompose `r_{1:L}` in that set, so `ρ*_{1:L} = 0`: each band
that one pass sets to zero is zero in the proximal point, and a band with `r_m = 0` is zero in both (the penalty does
not decrease as a band norm grows). The converse fails: for `p = 4`, `λ = 1`, band norms
`(2.05, 3, 0)`, one pass gives `(0.0014776018749914932, 0.0015847207086445254, 0)` while the proximal point is zero,
certified by a decomposition `r = ξ₁ + ξ₂ + ξ₃` with dual norms below `λ` (probe Z). So the bandwidth of the
one-pass map is at least that of the proximal point and can be larger. Read against arXiv v1: the proof of Theorem 3
(`K̂ ≤ K`, Appendix F) shows that one pass sets `g_L` to zero, so its conclusion holds for the minimizer as well;
Corollary 1 computes `K̂` from the one-pass `ν̂`, which gives the one-pass bandwidth.

**(b) The published text and related papers.**
- arXiv:1405.6210v1 (arXiv has this one version), Section 3: "only a single pass of BCD is required to reach convergence.
  This property is proved in Jenatton et al. (2011)"; proof of Theorem 2: "By Proposition 5 in Jenatton et al. (2011),
  we can get Σ̂ by a single pass as in Algorithm 1". Proposition 5 assumes one scalar weight `ω_g` per group (their Section 2).
- Published version (JASA 111(514), 834–845): we read the accepted author manuscript (PMC5199058, NIHMS801690) in a
  text form without formulas. It numbers the weights (2.4), (2.5), (2.6), the dual (3.1) and `h_ℓ` (3.2), and Section 3
  keeps the sentence "only a single pass of BCD is required to reach convergence. This property is proved in [citation
  elided in the text form] and is a direct consequence of the nested group structure". The typeset article and the
  supplementary materials (which hold the proof of Theorem 2, Section A.3) were not obtained.
- Yu and Bien (arXiv:1604.07451v3), Section 3 and Appendices C–D: the weighted prox of (10) with
  `w_{ℓm} = 1/(ℓ−m+1)²` by "a single pass as described in Algorithm 3", citing Jenatton et al. (proof of Theorem 8). In rows with three
  or more penalized coordinates (row `r ≥ 4`) these weights violate Proposition 4(a), and (a) above applies. Recorded instance `(2, 1.5, 1)`, step `0.5`: one pass
  `(1.4440890330680143, 0.980076615736529, 0.5573845210878831)`, reference
  `(1.4427610008814504, 0.981943801904041, 0.5574710002665637)` (probe Y); refuted in 76, 97 and 100 of 100 instances
  for 3, 5 and 9 penalized coordinates (F). Their prox runs inside an ADMM (Algorithm 1); the effect on that ADMM was not
  studied. The unweighted case (`w = 1`, Algorithm 2) satisfies Proposition 2. The published version was not read.
- Yan and Bien (Statistical Science 32(4), 2017; arXiv:1512.01631), Section 4, Algorithm 1: one pass for scalar
  weights `w_g > 0`, which Proposition 2 covers. Section 5 repeats that the convex banding estimators "can be
  evaluated in a single pass", and Section 4.2 gives the modified (general-weight) prox a cost of `O(p + D² log(n))`;
  both rest on the one-pass claim.

**(c) `hierband` 1.0** (CRAN source, read, not run).
- `R/hierband.R:79`: `subdiag.thresh` "Performs a single pass of BCD on a matrix R"; the loop over `l` is at lines
  108–126, the scaling of the band norms at 124 and the tapering of `R` at 127–128.
- `R/hierband.R:56`: without `delta`, `hierband` returns `subdiag.thresh(R=Sighat, lam=lam, w=w)`; with `delta`,
  line 62 calls it inside the eigenvalue loop.
- `R/hierband.R:50` and `src/subdiag.c:25`–`38`: default weights `sqrt(2l)/(l−m+1)`. A diagonal `w` goes to the
  exact disjoint `gpband` (lines 97–98); other zero weights are refused (99–100).
- `R/hierband.R:4` documents the objective without the factor `½` of the paper, while the zero test at line 110,
  `Σ (pp/ww)² ≤ lam²`, is that of `½‖·‖² + λΩ`: a factor 2 in `λ` to reconcile in any cross-check.
- `R/hierband.path.R:68`–`79`: `lam.max.hierband` is documented as the smallest `λ` with a diagonal result and
  returns `max ‖S_{s_ℓ}‖/w_ℓℓ` (also arXiv v1, Appendix C). It is an upper bound: for `p = 3` with band norms `(2, 0)`
  and `λ = 1 < 1.41` the proximal point is diagonal (probe L).

So `hierband` returns the one-pass point for its default weights; a comparison with it does not test exactness.

**What the paper's theory uses.** Read in arXiv v1:
- Theorem 6 (Frobenius rate) follows from Theorem 13, whose proof (Appendix G) uses the optimality conditions of (3);
  it concerns the minimizer. It uses the weights through `w_ℓℓ = √(2ℓ)` (Lemma 5),
  `Σ_ℓ w_ℓℓ² ≲ p²` and `w₀(0) = max_m Σ_{ℓ≥m} w_{ℓm}² ≤ 4p` (Appendix H). Remark 6 states a slower rate for the
  basic weights, for which `w₀(0)` is of order `p²` (our computation).
- Theorem 2 and Corollary 1 (tapering form, bandwidth), the proofs of Theorems 3–5 (Appendix F, "Referring to the proof
  of Theorem 2"), Theorem 9 (Appendix J, "By (12)", plus `Σ_{ℓ≥m} w_{ℓm}²/(2ℓ) ≲ 1`) and Theorem 10 (through
  Theorem 9) use the one-pass recursion (12). For the general weights and `p ≥ 4` these proofs, as written, describe the
  one-pass map, not the minimizer of (3). Apart from Theorem 3 (see Zero bands), whether their conclusions hold for the
  minimizer was not checked.

**(d) Is there an efficient exact evaluation?** One pass with weights in the condition of (a) costs a fixed number of
scalar root solves (one per group). For the general weights we found no method with that property; the methods below
reach the proximal point to a certified accuracy (a duality gap) after a number of iterations that depends on the instance.
- *Weights in the one-pass class.* For the chain, Proposition 4(a) leaves `w_{ℓm} = a_ℓ` for `m < ℓ` and
  `w_ℓℓ = b_ℓ` ("two-value" weights). One pass is then exact (accepted in 100 of 100 at `p = 4, 6, 10`, F), and after
  the `O(p²)` band norms it costs `p − 1` root solves of a two-term equation, keeping the running norm of the earlier
  bands. With `b_ℓ = √(2ℓ)` and `a_ℓ = √(2/ℓ)`: `w₀(0) = max_m (2m + Σ_{ℓ>m} 2/ℓ) = 2(p − 1)`, and
  `Σ_{ℓ≥m} w_{ℓm}²/(2ℓ) = 1 + Σ_{ℓ>m} 1/ℓ² ≤ π²/6`. We did not re-derive the proofs for such weights; they penalize
  all farther bands of a group alike, unlike (8), and their statistical behavior is not established here.
- *Dual block coordinate descent on the band norms*, stopped at a duality gap of `1e-12 (1 + ‖r‖²)`. Each sweep costs
  `p − 1` root solves, `O(p²)` operations per round of root iterations; the dual blocks take `p(p−1)/2` numbers. From
  zero (the first sweep is one pass), five instances per size, `p = 8, …, 128`, four families (probe C): 1–5 sweeps
  in families A and B (`λ = 1`), 9–356 in C (`λ = 2`), 1–7591 in D (`λ = 2.5`), where one instance at `p = 64` was
  still at a relative gap `2.2e-10` (distance bound `3.87e-3`) after 100000 sweeps. Each count above 300 comes with a
  smallest nonzero group norm below `0.005 λ w_ℓℓ` at the dual block coordinate descent point, but values near `1e-12` also occur with 9
  and 17 sweeps; the data show no cause. Within a family the counts do not grow steadily with `p`. One pass is off by
  up to `0.039 ‖r‖` on these instances.
- *A Newton-type method* (probe N, an illustration). Two dual sweeps from zero propose the zero prefix `L` (the leading
  zero bands of the iterate); Newton's method with a backtracking line search solves the bands `L+1, …, K`, where the
  objective is smooth while each group norm is nonzero; dual block coordinate descent on the first `L` groups supplies
  their dual blocks; the point is accepted when the duality gap of the whole problem, in `BigFloat`, is at most
  `1e-12 (1 + ‖r‖²)`. A failed attempt is reported and the proposal repeated with 20, 200 and 2000 sweeps. Each Newton
  iteration forms and factors a dense Hessian, `O(p³)` operations. Results on the same 100 instances: certified in 92,
  88 at the first attempt with at most 9 Newton iterations, 4 at the second with up to 34 iterations in all; the
  certificate of the zero prefix took at most 2 sweeps. The 8 failures, in family D at `p = 16, 64, 128`, are the
  instances whose dual block coordinate descent point has a nonzero group norm below `1e-9 λ w_ℓℓ`, the slow dual
  instance among them. A fifth attempt, which takes as the zero prefix the bands through the last group whose norm at
  the dual point of the last proposal is below `1e-6 λ w_ℓℓ` (a heuristic; the gap certificate decides acceptance),
  certified none of the 8. The probe does not establish global
  convergence, a rule for a group at its threshold (where the Hessian grows like `1/n_ℓ` and the prefix is not
  determined to the certificate's accuracy), `Float32` behavior, or a structured solve cheaper than `O(p³)`.
- *Structure of the optimality system.* With the zero bands `1, …, L` known, it is
  `ρ_m (1 + λ Σ_{ℓ≥m} w_{ℓm}²/n_ℓ) = r_m`, `n_ℓ = ‖W[ℓ, 1:ℓ] ρ_{1:ℓ}‖₂`, for `m, ℓ > L`. One pass solves it group by
  group under the condition of (a); for (8) each equation involves all larger groups. Yan and Bien (Algorithm 3,
  Lemma 3) give a finite exact prox for the latent penalty on a path, a different penalty (Section 10).

**(e) For #965.** The options, for the maintainer to choose:
1. the basic weights (2.5), arXiv (7): one pass exact (Proposition 2), `O(p²)`, but the slower rate of Remark 6;
2. two-value weights in the one-pass class: one pass exact; their theory is open;
3. the general weights (2.6), arXiv (8), evaluated to a certified gap, by dual block coordinate descent or a
   Newton-type method as above. On our probes dual block coordinate descent needed 1 to 7591 sweeps and had not
   reached the gap after 100000 in one instance; counts above 300 came with a group norm below `0.005 λ w_ℓℓ` at the
   dual point, but small norms also came with few sweeps. Newton failed on exactly the 8 instances with a group norm
   below `1e-9 λ w_ℓℓ` at the dual point. Under item 7 of the maintainer's scope for #963, an implementation issue would have to
   show, on sample covariances at the sizes of #965 (`p = 20, 100, 500`) along a `λ` path that includes the values where
   bands switch off: iterations, time and memory against one pass, the gap reached at the chosen tolerance, and the rule
   for groups at their thresholds;
4. one pass with (2.6) is a well-defined map, but not the proximal operator of (2.1), and its bandwidth can exceed that
   of the proximal point (Zero bands); under the policy it is not a ProximalBase proximal operator.

## 6. Comparison of the covered families

| | disjoint `ℓ2` (`GroupNormL2`) | sparse-group | laminar, scalar weights | chain with diagonal weights |
|---|---|---|---|---|
| Semantics | `Σ w_k‖x[G_k]‖₂` | `Σ c_i x_i` in magnitude plus disjoint group `ℓ2` | `Σ ω_g‖x_g‖₂`, nested zero patterns | `Σ_ℓ ‖D_ℓ x_{1:ℓ}‖₂` on band or row prefixes |
| Layout, weights | disjoint lists, `w ∈ [0, ∞]` | disjoint groups, `c_i ≥ 0`, `ω ∈ [0, ∞]` | laminar lists, `ω ∈ [0, ∞]` | prefixes, `W` triangular |
| Value | direct, `O(Σ G_k)` | direct | direct, `O(Σ g)` | direct, `O(p²)` with band norms |
| Prox | direct (Proposition 1) | direct (Corollary) | direct (Proposition 2), `O(Σ g)` | direct under Proposition 4(a); otherwise iterative, to a certified gap (Section 5) |
| Memory | `O(n + K)` | as disjoint | `O(Σ g)` for lists, `O(n + K)` for a parent map | `O(p)` band norms; `O(p²)` dual blocks if iterative |
| Contracts | value, prox, fused, block prox, KKT on master | as disjoint, plus an entry test | value, prox, fused; no group score for nested groups on master | value, prox, fused; no gradient (nonsmooth) |
| Workspaces, aliasing | none; `y === x`, `out === u` | none | none for one pass; in place in the order | band norms; in place by band scaling |
| `Float32`/`Float64` | master contract (`docs/src/numerics.md`) | soft threshold and shrink kernels | the same shrink kernel | root of `h(s)`: not measured here |
| `SparseIterate` | `prox!` takes an `AbstractArray` (`groupnorml2.jl:189`): indexed reads and writes | as disjoint | as disjoint | not applicable to bands |
| `BlockIterate` | blocks are the groups | blocks are the groups | blocks partition `1:p` (`block_iterate.jl:146`), so nested groups are not blocks | not applicable |
| `SymmetricSparseIterate` | refused on master (`groupnorml2.jl:314`) | not on master | not on master | bands are transpose-invariant; no owner path on master |

## 7. Matrix structures

| Representation | Memory | Validation | What the prox exploits | Symmetric storage |
|---|---|---|---|---|
| band summaries (#965) | `O(p)` band norms; weights `O(p²)` as a triangle or `O(1)` as a rule | square shape | reduction to `p − 1` band norms (Section 5) | bands are transpose-invariant: lower triangle with weight 2 off the diagonal |
| row-wise nested chains (Yu and Bien) | `O(1)` rule or `O(p²)` weights | lower-triangular shape | each row separately; scalar weights: one pass per row, `O(p²)` | not symmetric (a Cholesky factor) |
| fibers and slices (legacy `FiberSum`) | `O(1)` | array shape | disjoint fibers (Proposition 1), any fiber function | `symmetric = true` reads `i ≥ j` and copies (`fibers.jl:164`) |
| generic index lists (`GroupNormL2` layout) | `O(Σ G_k)`; nested bands as lists take `Σ ℓ(ℓ+1) = O(p³)` | disjointness `O(Σ G_k)`; laminarity after a sort by size, near linear | nothing beyond the lists | groups must be closed under transpose (legacy `norms.jl:808`) |

## 8. Designs (A) and (B)

| | (A) specialized penalties sharing kernels | (B) one nested/tree `GroupNormL2`-style type |
|---|---|---|
| Public names | one type per penalty: sparse-group, tree group `ℓ2`, the banding type of #965 | one type taking laminar lists and weights |
| Validation | each layout its own: disjoint, laminar, square matrix | laminarity, inclusion order and the Proposition 4(a) condition on generic lists |
| Aliasing | as `GroupNormL2`: in place allowed, other overlap rejected | the same |
| Workspaces | none (scalar), `O(p)` band norms | order and parent map; a list layout has no band norms |
| Tests | per type: Propositions 1–2 or 4 with certified gaps, degenerate layouts, `Float32`/`Float64` | all laminar layouts, refusal of non-laminar lists, the condition check, plus each estimator's layout |
| Estimators | sparse group lasso; hierarchical sparse coding (Jenatton et al.); convex banding; varband unweighted | the same, banding as `O(p³)` lists |
| Oracle cost | value and prox `O(Σ g)` or `O(p²)`; fused value from the shrink | `O(Σ g)`, `O(p³)` for banding |

Shared operations in (A): the scalar group shrink on an index set (`_l2_shrink!`), the soft threshold of `NormL1`, and,
for diagonal weights or steps, the projection onto an ellipsoid with a bracketed scalar root. Convex banding with the
basic weights and unweighted varband need the same operation (a scalar shrink of nested prefixes) on different layouts
(bands of a symmetric matrix, prefixes of the rows of a triangular one), so their reuse is at the kernel level, which (A)
provides. (B) would add generic validation, could not use band norms, and changing `GroupNormL2` to carry it conflicts
with the scope guard of #963 (no refactor or slowdown of the disjoint path).

**Recommendation: (A).** Evidence that would change it: two named estimators that need laminar groups on generic index
lists with the same weight contract, and a benchmark showing the generic layout no slower and no larger than the
specialized ones at their sizes. (C), a generic overlapping-group penalty with an inner solver, is excluded by the
policy and was not evaluated.

## 9. Penalties recommended for near-term implementation

**Sparse-group penalty.** Simon, Friedman, Hastie and Tibshirani (2013), the sparse group lasso, and the group
graphical lasso of Danaher, Wang and Witten (2014) as a `FiberSum`, both as cited in the legacy docstring
(`sparsegroup.jl:7`–`40`); the papers were not checked against the source. Penalty
`Σ_i c_i |x_i| + Σ_g ω_g ‖x_g‖₂`, disjoint groups. Operation: soft threshold, then scalar group shrink. Exact by the Corollary of Proposition 2.
- *Proposed issue:* "Restore `SparseGroupNormL2` on disjoint groups". Scope: type, value, `prox!`,
  `value_and_prox!`, `prox_block!`, `kkt_violation` with the entry test, on the `GroupNormL2` layout and kernels; the
  legacy file as a reference, unchanged. Evidence: the Corollary, probes W1, W2 and M2, the legacy implementation.

**Tree-structured group `ℓ2` penalty, scalar weights.** Jenatton, Mairal, Obozinski and Bach (2011), hierarchical sparse
coding; Yan and Bien (2017), the group lasso on the descendant sets of a tree. Penalty `Σ_g ω_g‖x_g‖₂` on laminar
groups, `ω_g ∈ [0, ∞]`. Operation: one pass of scalar group shrinks in an inclusion order, `O(Σ_g g)`, at most the depth
times `n` (Jenatton et al., Section 3.5). Exact by Proposition 2.
- *Proposed issue:* "Tree-structured group `ℓ2` penalty with scalar weights". Scope: laminar validation and the order at
  construction, value, prox, fused value; no diagonal weights; no `BlockIterate` blocks for nested groups. Evidence:
  Proposition 2, probes L1–L3, W2.

No issue is proposed for diagonal weights: the one penalty that needs them is #965, which exists (Section 5(e)).

## 10. Not supported

Groups with arbitrary overlap: for partially overlapping groups the Lemma does not apply, and the sources read give no
finite composition of group steps for them (Jenatton et al., Section 3.3, say that their dual block coordinate ascent
need not, in general, finish in finitely many iterations, before proving one pass for ordered tree-structured
groups); an exact polynomial algorithm exists for sums of `ℓ∞` norms (Mairal et al., JMLR 2011, abstract
and Propositions 6–7, by quadratic min-cost flow); for `ℓ2` that paper uses proximal splitting instead. The latent
penalty of Obozinski, Jacob and Vert (arXiv:1110.0413, Section 4.1 and Remark 4) is an infimal convolution, so its value is itself an optimization problem;
Yan and Bien (Section 4) note no closed-form prox on trees, with the path case (Algorithm 3) as the exception. Under the
policy neither is supported, and no probe or design covers them.
