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
| laminar | `D` not constant on a contained group | scalar | two groups: not exact for some input; chains below: not exact on recorded instances | Proposition 4(b) |
| laminar | scalar-weight `ℓ2` groups | diagonal `τ`, constant on each group contained in a penalized group | exact | Proposition 5 |

Nesting alone does not make one pass exact: the condition is on the weights inside a group.

**The general weights of Bien, Bunea and Xiao.** For their weights `w_{ℓm} = √(2ℓ)/(ℓ−m+1)` one pass is not exact
once `p ≥ 4`, and it is exact for `p ≤ 3`. The paper's one-pass claim cites a result for one scalar weight per group.
The accepted manuscript of the published version makes the same claim, and `hierband` 1.0 computes the one-pass point.
The weights of Yu and Bien, `1/(ℓ−m+1)²`, fail in the same way. An exact evaluation exists (dual block coordinate
descent on the `p − 1` band norms, stopped at a certified duality gap), but its sweep count depends on the instance, from
one to thousands on our probes. We found no finite exact algorithm. Section 5 lists the options for #965 and leaves the
choice to the maintainer.

**Recommended penalties.** The sparse-group penalty (`ℓ1` plus disjoint group `ℓ2`) and a tree-structured group `ℓ2`
penalty with one scalar weight per group: both have an exact one-pass proximal operator built from the scalar group
shrink that master already has. Section 9 proposes two follow-up issues.

**Design.** (A), specialized penalties that share small kernels, is the recommendation. The evidence that would change it
is in Section 8.

## 2. Setting and notation

- Coordinates `1, …, n`; groups `g_1, …, g_K ⊆ {1, …, n}`; `z_g` is the subvector of `z` on `g`. Groups may be equal
  (duplicates), and a coordinate may lie in no group.
- **Laminar:** any two groups are disjoint or one contains the other. A laminar family is a forest under inclusion
  (Jenatton et al., JMLR 2011, Definition 1, which also asks the groups to cover all coordinates; nothing below needs that).
- **Group function** `N_k` on `ℝ^{g_k}`: closed, convex, positively homogeneous, `N_k(0) = 0` ("sublinear"). This
  covers `ω‖z‖₂`, the weighted norm `‖D z‖₂` with `D = diag(d) ≥ 0`, `ω‖z‖∞`, the zero function (weight 0) and the
  indicator of `{0}` (weight `∞`).
- `Ω(x) = Σ_k N_k(x_{g_k})`, and the proximal problem with a scalar step `t ≥ 0`:
  `prox_{tΩ}(y) = argmin_x ½‖x − y‖² + t Ω(x)`.
- **Group step** `P_k`: the proximal operator of `x ↦ t N_k(x_{g_k})`; it changes `x_{g_k}` alone. For `‖D z‖₂` and a
  group not set to zero, `P_k` multiplies entry `i` by `s/(s + d_i²)`, where `s > 0` solves
  `Σ_i d_i² v_i²/(s + d_i²)² = t²` (sum over `d_i > 0`; entries with `d_i = 0` are unchanged). The group is set to zero
  when `Σ_i v_i²/d_i² ≤ t²` (Bien, Bunea and Xiao, arXiv:1405.6210v1, Appendix B).
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
| `_check_one_based_groups` | `src/block_iterate.jl:109` | axes of the group lists | lists | validation |
| `_own_groups` | `src/block_iterate.jl:120` | owned copy of the groups | lists or ranges | validation |
| `_group_of` | `src/block_iterate.jl:128` | nonempty, positive, disjoint groups | lists | validation |
| `_check_partition` | `src/block_iterate.jl:146` | groups cover `1:p` | lists | validation |
| `prox_block` | `src/block_prox.jl:2` | allocating block prox | compact block | block prox |
| `_threshold_product` | `src/numerical_kernels.jl:103` | threshold `γλw` rounded once | scalars | kernel |
| `_l2_zero_group` | `src/numerical_kernels.jl:902` | zero test of an index set | index set | kernel |
| `_l2_shrink!` | `src/numerical_kernels.jl:1001` | scalar group shrink, `y === x` allowed | index set | kernel |
| `_l2_shrink_fixed!` | `src/numerical_kernels.jl:1042` | zero and infinite weight, level and step | index set | kernel |
| `_l2_term` | `src/numerical_kernels.jl:1083` | value term `λw‖x[G]‖₂` | index set | kernel |
| `_l2_fused_term` | `src/numerical_kernels.jl:1108` | value term from the shrink | index set | kernel |
| `_l2_value` | `src/numerical_kernels.jl:1140` | value of `NormL2` | whole array | kernel |
| `_l2_expsum_add` | `src/numerical_kernels.jl:1173` | value recovery out of range | index set | kernel |
| `_l2_screen` | `src/numerical_kernels.jl:1184` | `‖g[G]‖₂ − λw` | index set | kernel |
| `_l2_score` | `src/numerical_kernels.jl:1233` | point-aware group score | index set | kernel |
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
| `GroupMCP` | `src/functions/nonconvex.jl:381` | group MCP, nonconvex | disjoint | value, prox |
| `GroupSCAD` | `src/functions/nonconvex.jl:405` | group SCAD, nonconvex | disjoint | value, prox |
| `IndBallGroupL0` | `src/functions/nonconvex.jl:705` | at most `s` nonzero groups, nonconvex | disjoint | value, prox |
| `SliceSum` | `src/functions/fibers.jl:58` | `Σ_k f_k(x_k)` over slices | last dimension | per slice |
| `FiberSum` | `src/functions/fibers.jl:182` | `Σ_ij f(x[i,j,:])`, optional symmetric fibers | fibers (disjoint) | per fiber |
| `TotalVariation1D` | `src/functions/ordered.jl:23` | 1-D total variation | ordered vector | value, prox |
| `FusedLasso` | `src/functions/ordered.jl:187` | `ℓ1` plus 1-D fusion | ordered vector | value, prox |
| `SLOPE` | `src/functions/ordered.jl:518` | sorted `ℓ1` | vector | value, prox |
| `IndPattern` | `src/functions/matrixsets.jl:30` | support in a pattern | index pattern | prox |
| `IndBanded` | `src/functions/matrixsets.jl:124` | banded matrices | bandwidths | prox |
| `IndToeplitz` | `src/functions/matrixsets.jl:224` | Toeplitz matrices | diagonals | prox |
| `SeparableSum` | `src/calculus.jl:2129` | sum over the blocks of a tuple | tuple | blockwise |
| `Sum` | `src/calculus.jl:2224` | `Σ f_k(x)`, "no proximal operator in general" | whole argument | smooth oracles |
| `IndIntersection` | `src/functions/intersection.jl:53` | Dykstra; `converged` is not a certificate (`:22`–`:30`) | sets | iterative prox |
| `Precompose` | `src/composition.jl:61` | `f(Ax)` | linear map | value, prox (special `A`) |
| `LinearImage` | `src/image.jl:85` | image of `f` under `L` | linear map | value, prox |
| `IndGraph` | `src/functions/graph.jl:62` | graph of a linear map | linear map | projection |
| `IndAffine` | `src/functions/affine.jl:38` | affine set | linear map | projection |
| `IndPolyhedron` | `src/functions/polyhedron.jl:58` | polyhedron | matrix | projection |
| `LinearConstrained` | `src/constrained.jl:114` | `f` on a hyperplane or slab | vector | prox |
| `AffineConstrained` | `src/affineconstrained.jl:40` | quadratic on an affine set | matrix | prox |
| `IndSOC` | `src/functions/cones.jl:32` | second-order cones, per group | disjoint groups | projection |
| `IndRotatedSOC` | `src/functions/cones.jl:200` | rotated cones, per group | disjoint groups | projection |
| `IndExpPrimal` | `src/functions/cones.jl:377` | exponential cones, per group | disjoint groups | projection |
| `IndExpDual` | `src/functions/cones.jl:413` | dual exponential cones, per group | disjoint groups | projection |
| `BallConstrained` | `src/ballconstrained.jl:95` | `f` on an `ℓ2` ball, `GroupNormL2` included | wrapper | prox |
| `SphereConstrained` | `src/ballconstrained.jl:101` | `f` on an `ℓ2` sphere | wrapper | prox |
| `_GroupReading` | `src/ballconstrained.jl:859` | group reader of the ball wrappers | disjoint groups | helper |

Neither tree holds an arbitrary-overlap or latent group penalty. The rows from `SliceSum` on are outside the covered
families and stay one line each.

## 4. Conditions for an exact one-pass proximal operator

**Lemma (composition).** Let the groups be laminar, each `N_k` sublinear, `t ≥ 0`, and `σ` an inclusion order. Suppose
that for each pair `k` before `j` with `g_k ⊆ g_j` and all `x`,

  (C) `∂N_k(P_j(x)_{g_k}) ⊇ ∂N_k(x_{g_k})`.

Then one pass returns `prox_{tΩ}(y)` for all `y`.

*Proof.* The optimality condition of step `k` gives `ξ_k ∈ t ∂(N_k ∘ π_k)(x^k)`, with `π_k` the restriction to `g_k`.
A later step `j` either has `g_j ∩ g_k = ∅` and leaves `x_{g_k}` unchanged, or, by laminarity and the order,
`g_j ⊇ g_k`, and then (C) keeps `ξ_k` a subgradient. So `ξ_k ∈ t ∂(N_k ∘ π_k)(x^K)` for each `k`, and
`y − x^K = Σ_k ξ_k ∈ t ∂Ω(x^K)` (a sum of subgradients is a subgradient of the sum). This is the optimality condition
of the strongly convex proximal problem, whose minimizer is unique. ∎ This is Yu (NIPS 2013), Theorem 1, applied group
by group; Jenatton et al., Lemma 4 and Proposition 5, is the same argument for scalar `ℓ2` or `ℓ∞` groups.

**Fact.** For a sublinear `N`, `∂N(c z) = ∂N(z)` for `c > 0` and `∂N(0) ⊇ ∂N(z)`. So (C) holds when `P_j` multiplies
`x_{g_k}` by a scalar `c ≥ 0`.

**Proposition 1 (disjoint groups).** Disjoint groups, any closed convex `N_k` (diagonal weights, any norm), scalar step
`t ≥ 0` or diagonal step. The problem separates over the groups and the unpenalized coordinates: the proximal point is
the group steps in any order, and coordinates in no group are unchanged. *Needed:* disjointness, for the separation.
Master's `GroupNormL2` implements this for scalar `ℓ2` weights; a zero weight copies the group and an infinite weight
or level fixes it at `+0` (`groupnorml2.jl:44`, kernel `numerical_kernels.jl:1042`). Probes D1 (recorded instance,
difference `4.44e-16` from the harness) and D2 (200 instances).

**Proposition 2 (laminar, scalar `ℓ2` on containing groups).** Laminar groups (forests, duplicates and unpenalized
coordinates allowed), an inclusion order, a scalar step `t ≥ 0`. Each group that contains another group of the
family, an equal group placed before it included, has `N_j = ω_j‖·‖₂` with `ω_j ∈ [0, ∞]`; the other groups have any
sublinear `N_k`. Then one pass is exact. *Proof.* `P_j` multiplies `x_{g_j}` by `c = (1 − tω_j/‖x_{g_j}‖)₊`
(`c = 1` for `ω_j = 0` or `t = 0`, `c = 0` for `ω_j = ∞`); the Fact and the Lemma apply. ∎ With positive weights, one
norm for all groups and a cover, this is Jenatton et al., Proposition 5 (`ℓ2` case) and Yu, Corollary 3.
*Hypotheses:*
- laminar: needed for the Lemma; without it see Section 10;
- the order: needed. The sparse-group penalty `|x₁|/2 + |x₂|/2 + ‖x‖₂/2` at `y = (1, 0.1)`, `t = 1`, has the proximal
  point `0`; the group step first, then the coordinates, gives `(0.0024814048950054326, 0)` (probe W2);
- scalar `ℓ2` on the containing groups: needed; an `ℓ∞` containing group fails (Proposition 3), diagonal weights are
  Proposition 4;
- zero and infinite weights, a group at its threshold (`‖v‖₂ = tω`), duplicates, forests, unpenalized coordinates and
  five nested groups are in the recorded instance L1; 300 random laminar families pass (L3).

*Corollary (two-level, sparse-group).* `Ω(x) = Σ_i c_i|x_i| + Σ_g ω_g‖x_g‖₂` with disjoint groups `g`, coordinate
weights `c_i ≥ 0` that may differ inside a group, `ω_g ∈ [0, ∞]`: the proximal point is the soft threshold at `t c_i`,
then the group shrink. The singletons are groups that contain no other. The legacy `SparseGroupNormL2` computes this
(`sparsegroup.jl:265`). Probes W1, W2.

**Proposition 3 (`ℓ∞` groups).** Laminar groups, an inclusion order, `N_k = ω_k‖·‖∞` with `ω_k ∈ [0, ∞]` for all
groups, scalar step: one pass is exact (Jenatton et al., Proposition 5, `ℓ∞` case). *Proof of (C).* `P_j` clips
`x_{g_j}` at a level `θ > 0`, sets it to zero, or leaves it unchanged (`ω_j = 0` or `t = 0`); clipping keeps the
signs and enlarges the set of coordinates of largest magnitude in `g_k`, so `∂‖·‖∞` does not shrink. ∎ Probe M1
(300 families). *One norm for all groups* (the setting of Lemma 4): an `ℓ2` group inside an `ℓ∞` group is not covered, and fails: groups `{1,2}` (`ℓ2`, weight 1) inside
`{1,2,3}` (`ℓ∞`, weight 1) at `y = (3, 1, 0)`, `t = 1`: one pass `(1.0513167019494862, 0.6837722339831621, 0)`, the
certified reference `(1.1055728090000843, 0.5527864045000421, 0)` (probe M3). An `ℓ2` group containing `ℓ∞` groups is
Proposition 2 (probe M2). For `ℓq`, `q ∉ {1, 2, ∞}`, Jenatton et al., Appendix C (Lemmas 12, 13), give a two-group
counterexample; not probed.

**Proposition 4 (diagonal weights inside a group).** Laminar groups, an inclusion order, scalar step `t ≥ 0`. Each group
that contains another has `N_j(z) = ‖D_j z‖₂` with `D_j = diag(d_j)`, `d_j ≥ 0` finite; the others any sublinear `N_k`.

- (a) If for each pair `k` before `j` with `g_k ⊆ g_j` the weights `d_j` are constant on `g_k` (zero allowed), one
  pass is exact. *Proof.* `P_j` multiplies `x_{g_k}` by `s/(s + a²)` when `d_j ≡ a` on `g_k`, by `0` when group `j`
  is set to zero and `a > 0`, and by `1` when `a = 0`; the Fact and the Lemma apply. ∎ For a tree the condition reads:
  the weights of a group are constant on each child group and arbitrary on its own coordinates. Probe P1 (300 families,
  zero weights and duplicates included).
- (b) *Necessity for two groups.* Let `g_1 ⊆ g_2`, `N_1 = ‖D_1·‖₂` with `d_1 > 0` on `g_1`, `N_2 = ‖D_2·‖₂`, and
  `d_2(i_1) ≠ d_2(i_2)` for some `i_1, i_2 ∈ g_1`. Then for each `t > 0` some `y` gives a one-pass point that is not
  the proximal point. *Proof.* Take `y = α u`, `u` supported on `{i_1, i_2}` with both entries nonzero, `α` large.
  Both steps and the proximal point keep zeros outside `{i_1, i_2}` (weighted `ℓ2` norms grow with each magnitude).
  For large `α` neither step sets its group to zero, so `x² = Λ x¹` with `Λ = diag(f_{i_1}, f_{i_2})`,
  `f_i = s/(s + d_2(i)²)`, `s ∈ (0, ∞)`, `f_{i_1} ≠ f_{i_2}`. Both norms are differentiable at `x²`, and one pass
  gives `y − x² = t∇N_1(x¹) + t∇N_2(x²)`; so `x²` is optimal if and only if `∇N_1(x¹) = ∇N_1(x²)`. With
  `∇N_1(z) = D_1² z/‖D_1 z‖₂` and `D_1` positive on `{i_1, i_2}`, this forces `x² = c x¹`, `c > 0`, which `Λ` and the
  two nonzero entries of `x¹` exclude. ∎ Probe P2 (`d_1 = (1, 2)`, `d_2 = (1, 3, 2)`, `y = (2, 3, 1)`, `t = 0.3`:
  refuted by the reference and by a gradient entry of the objective).
- (c) In a family with more groups we do not prove that a pair violating (a) forces failure for some input: other
  steps could compensate on special inputs. The chains of Section 5 fail on recorded instances.
- *Duplicates* make the order matter: `g_1 = g_2 = {1,2,3}` with `d_1 = (1, 2, 3)` and the scalar weight `1.5` on
  `g_2` is exact with `g_1` first (P3) and refuted with `g_2` first (P4).

**Proposition 5 (diagonal step).** The problem `argmin_x ½ Σ_i (x_i − y_i)²/τ_i + Ω(x)`, `τ_i > 0`, with laminar
groups and scalar weights `ω_k ‖·‖₂`. With `z = x/√τ` it is the problem of Proposition 4 at `y/√τ` with the weights
`d_k = ω_k √τ` on `g_k`, and the group steps in `x` (each a weighted shrink with an ellipsoid root, not a closed-form
scalar shrink) correspond to those in `z`. So one pass is exact when `τ` is constant on each group contained in a group
of positive weight, and for two groups of positive weight it is needed (Proposition 4(b)). For the sparse-group
penalty the contained groups are singletons, so one pass is exact for any `τ` (S1). Probes S2 (300 families) and S3 (refuted for
`τ = (1, 4, 1)` on `{1,2} ⊂ {1,2,3}`). Master's `GroupNormL2` takes scalar steps (`groupnorml2.jl:15`).

## 5. The general weights of Bien, Bunea and Xiao

Equation numbers are those of arXiv:1405.6210v1; the published version numbers the problem (3) as (2.1) and the
weights (6), (7), (8) as (2.4), (2.5), (2.6).

**Reduction to band norms.** For the banding penalty `λ Σ_ℓ (Σ_{m≤ℓ} w_{ℓm}² ‖X_{s_m}‖²)^{1/2}` on a `p × p` matrix,
with `s_m` the two subdiagonals `p − m` away from the diagonal and `g_ℓ = s_1 ∪ … ∪ s_ℓ`, the weights are constant on
each band. For fixed band norms `ρ_m` the distance term is minimized by `X_{s_m} = (ρ_m/r_m) S_{s_m}`, `r_m = ‖S_{s_m}‖₂`
(`X_{s_m} = 0` when `r_m = 0`), so the problem is the chain `{1} ⊂ {1,2} ⊂ … ⊂ {1, …, p−1}` in `ℝ^{p−1}` at `r ≥ 0` with
the diagonal weights `W[ℓ, 1:ℓ]` (its minimizer is nonnegative), and the diagonal of `S` is kept. Primal and dual values are equal under this map, so
duality gaps transfer. Probe B1 (`p = 5`: matrix gap `1.78e-16` at the band-norm solution scaled back). The disjoint
control `w_{ℓm} = √(2ℓ) 1{ℓ = m}` is `GroupNormL2` on the bands (B2, difference `1.11e-16`).

**(a) The counterexample.** For the chain, Proposition 4(a) asks `w_{ℓm}` to be constant in `m < ℓ`. The general
weights give `w_{3m} = (√6/3, √6/2)` on bands 1, 2, so the condition fails once `ℓ = 3` exists, that is `p ≥ 4`; for
`p = 3` the contained group `{1}` is one band and one pass is exact. The recorded instance (`p = 4`,
`r = (3.3908789763847373, 2.723510817894522, 1.102998740113383)`, `λ = 0.9768147919269893`), probes B3 and B4:

| | band norms | largest gradient entry |
|---|---|---|
| one pass (`Float64`) | `(0.880669128402111, 0.43980506851334233, 0.15773452285496928)` | `0.244` |
| reference (14 sweeps, BigFloat gap `8.02e-15`) | `(0.8339820745637797, 0.49606635020397205, 0.15980235543181953)` | `1.39e-07` |

The distance `0.0731` exceeds `2√(2 gap) = 2.53e-07`, all group norms are nonzero at both points, and the objective
excess of the one-pass point is `0.01022428370907346`. On 100 seeded instances per size, one pass is refuted in 81
(`p = 4`), 92 (`p = 6`) and 98 (`p = 10`), exact in the others, with none undecided; it is exact in 100 of 100 for
`p = 3` and for the basic weights `√(2ℓ)` and the disjoint control at `p = 4, 6, 10` (probe F).

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
  `w_{ℓm} = 1/(ℓ−m+1)²` by "a single pass as described in Algorithm 3", citing Jenatton et al. (proof of Theorem 8). For three or more
  coordinates in a row these weights violate Proposition 4(a). Recorded instance `(2, 1.5, 1)`, step `0.5`: one pass
  `(1.4440890330680143, 0.980076615736529, 0.5573845210878831)`, reference
  `(1.4427610008814504, 0.981943801904041, 0.5574710002665637)` (probe Y); refuted in 76, 97 and 100 of 100 instances
  for 3, 5 and 9 coordinates (F). Their prox runs inside an ADMM (Algorithm 1); the effect on that ADMM was not
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

So `hierband` returns the one-pass point for its default weights, and a comparison with it cannot detect the error.

**What the paper's theory uses.** Read in arXiv v1:
- Theorem 6 (Frobenius rate) follows from Theorem 13, whose proof (Appendix G) uses the optimality conditions of (3);
  it concerns the minimizer. It uses the weights through `w_ℓℓ = √(2ℓ)` (Lemma 5),
  `Σ_ℓ w_ℓℓ² ≲ p²` and `w₀(0) = max_m Σ_{ℓ≥m} w_{ℓm}² ≤ 4p` (Appendix H). Remark 6: the basic weights give `w₀` of order `p²` and a slower rate.
- Theorem 2 and Corollary 1 (tapering form, bandwidth), the proofs of Theorems 3–5 (Appendix F, "Referring to the proof
  of Theorem 2"), Theorem 9 (Appendix J, "By (12)", plus `Σ_{ℓ≥m} w_{ℓm}²/(2ℓ) ≲ 1`) and Theorem 10 (through
  Theorem 9) use the one-pass recursion (12). For the general weights and `p ≥ 4` these proofs, as written, describe the
  one-pass map, not the minimizer of (3). Whether their conclusions hold for the minimizer was not checked.

**(d) Is there an efficient exact evaluation?**
- *Weights in the one-pass class.* For the chain, Proposition 4(a) leaves `w_{ℓm} = a_ℓ` for `m < ℓ` and
  `w_ℓℓ = b_ℓ` ("two-value" weights). One pass is then exact (100 of 100 at `p = 4, 6, 10`, F), and after the
  `O(p²)` band norms it costs `O(p)` root solves, since the earlier bands share one factor per step. With
  `b_ℓ = √(2ℓ)`, `0 < a_ℓ ≤ b_ℓ`, the inequalities above can hold, for example `a_ℓ = √(2/ℓ)` gives
  `w₀(0) ≤ 2p + 2 ln p` and `Σ_{ℓ>m} a_ℓ²/(2ℓ) ≤ π²/6`. We did not re-derive the proofs for such weights, and they
  penalize all farther bands of a group alike, unlike (8); their statistical behavior is not established here.
- *Dual block coordinate descent on the band norms.* Exact in the limit, with the duality gap as a certificate. Each
  sweep costs `p − 1` root solves, each iteration of which costs `O(ℓ)`, so `O(p²)` per round of iterations; the dual blocks take `p(p−1)/2` numbers. Sweeps from zero
  (the first sweep is one pass) to a gap of `1e-12 (1 + ‖r‖²)`, five instances per size, `p = 8, …, 128` (probe C):
  1–5 sweeps when the factors `s/(s + w²)` of each step vary by at most `0.17` over its earlier nonzero bands
  (families A, B); 9–356 when they vary by up to `0.81` (C, `λ = 2`); up to 7591 when they vary by up to `0.965`, with
  one instance at `p = 64` still at a relative gap `2.2e-10` after 100000 sweeps (D, `λ = 2.5`). Across the families
  the larger counts come with the larger spreads; within a family they do not grow steadily with `p`. On these
  instances one pass is off by up to `0.039 ‖r‖`. A Newton-type method on the smooth part was not probed.
- *A finite exact algorithm.* With the zero bands `1, …, L` known, the optimality system is
  `ρ_m (1 + λ Σ_{ℓ≥m} w_{ℓm}²/n_ℓ) = r_m`, `n_ℓ = ‖W[ℓ, 1:ℓ] ρ_{1:ℓ}‖₂`, for `m, ℓ > L`. One pass solves it group by group
  under the condition of (a); for (8) each equation involves all larger groups. We found no finite algorithm, in the
  sources read or by our own analysis. Yan and Bien (Algorithm 3, Lemma 3) give a finite exact prox for the latent
  penalty on a path, a different penalty (Section 10).

**(e) For #965.** The options, for the maintainer to choose:
1. the basic weights (2.5), arXiv (7): one pass exact (Proposition 2), `O(p²)`, but the slower Frobenius rate of Remark 6;
2. two-value weights in the one-pass class: one pass exact; their theory is open;
3. the general weights (2.6), arXiv (8), with an exact iterative evaluation: under item 7 of the maintainer's scope for #963 this needs an implementation issue
   that reports dual BCD (or another method) to a certified gap on sample covariances at the sizes of #965
   (`p = 20, 100, 500`) along a `λ` path that includes the values where bands switch off: sweeps, time and memory against
   one pass, and the gap at the chosen tolerance;
4. one pass with (2.6) is a well-defined map, but not the proximal operator of (2.1); under the policy it is not a
   ProximalBase proximal operator.

## 6. Comparison of the covered families

| | disjoint `ℓ2` (`GroupNormL2`) | sparse-group | laminar, scalar weights | chain with diagonal weights |
|---|---|---|---|---|
| Semantics | `Σ w_k‖x[G_k]‖₂` | `Σ c_i x_i` in magnitude plus disjoint group `ℓ2` | `Σ ω_g‖x_g‖₂`, nested zero patterns | `Σ_ℓ ‖D_ℓ x_{1:ℓ}‖₂` on band or row prefixes |
| Layout, weights | disjoint lists, `w ∈ [0, ∞]` | disjoint groups, `c_i ≥ 0`, `ω ∈ [0, ∞]` | laminar lists, `ω ∈ [0, ∞]` | prefixes, `W` triangular |
| Value | direct, `O(Σ G_k)` | direct | direct, `O(Σ g)` | direct, `O(p²)` with band norms |
| Prox | direct (Proposition 1) | direct (Corollary) | direct (Proposition 2), `O(Σ g)` | direct under Proposition 4(a); otherwise iterative (Section 5) |
| Memory | `O(n + K)` | as disjoint | `O(Σ g)` for lists, `O(n + K)` for a parent map | `O(p)` band norms; `O(p²)` dual blocks if iterative |
| Contracts | value, prox, fused, block prox, KKT on master | as disjoint, plus an entry test | value, prox, fused; no group score for nested groups on master | value, prox, fused; no gradient (nonsmooth) |
| Workspaces, aliasing | none; `y === x`, `out === u` | none | none for one pass; in place in the order | band norms; in place by band scaling |
| `Float32`/`Float64` | master contract (`docs/src/numerics.md`) | soft threshold and shrink kernels | the same shrink kernel | root of `h(s)`: not measured here |
| `SparseIterate` | `prox!` takes an `AbstractArray` (`groupnorml2.jl:189`): indexed reads and writes | as disjoint | as disjoint | not applicable to bands |
| `BlockIterate` | blocks are the groups | blocks are the groups | blocks cannot be overlapping groups | not applicable |
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
| Workspaces | none (scalar), `O(p)` band norms | order and parent map; band norms cannot be used on lists |
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
  legacy file as a read-only reference. Evidence: the Corollary, probes W1, W2 and M2, the legacy implementation.

**Tree-structured group `ℓ2` penalty, scalar weights.** Jenatton, Mairal, Obozinski and Bach (2011), hierarchical sparse
coding; Yan and Bien (2017), the group lasso on the descendant sets of a tree. Penalty `Σ_g ω_g‖x_g‖₂` on laminar
groups, `ω_g ∈ [0, ∞]`. Operation: one pass of scalar group shrinks in an inclusion order, `O(Σ_g g)`, at most the depth
times `n` (Jenatton et al., Section 3.5). Exact by Proposition 2.
- *Proposed issue:* "Tree-structured group `ℓ2` penalty with scalar weights". Scope: laminar validation and the order at
  construction, value, prox, fused value; no diagonal weights; no `BlockIterate` blocks for nested groups. Evidence:
  Proposition 2, probes L1–L3, W2.

No issue is proposed for diagonal weights: the one penalty that needs them is #965, which exists (Section 5(e)).

## 10. Not supported

Groups with arbitrary overlap: for partially overlapping groups the Lemma does not apply, and Jenatton et al.
(Section 3.3) note that dual block coordinate ascent "is not guaranteed to solve exactly" the problem in finitely many
iterations in general; an exact polynomial algorithm exists for sums of `ℓ∞` norms (Mairal et al., JMLR 2011, abstract
and Propositions 6–7, by quadratic min-cost flow); for `ℓ2` that paper uses proximal splitting instead. The latent
penalty of Obozinski, Jacob and Vert (arXiv:1110.0413, Section 4.1 and Remark 4) is an infimal convolution, so its value is itself an optimization problem;
Yan and Bien (Section 4) note no closed-form prox on trees, with the path case (Algorithm 3) as the exception. Under the
policy neither is supported, and no probe or design covers them.
