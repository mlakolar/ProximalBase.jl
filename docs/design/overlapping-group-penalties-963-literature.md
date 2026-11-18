# Literature addendum to the note on nested and tree-structured group penalties (issue #963)

Status: research addendum to [`overlapping-group-penalties-963.md`](overlapping-group-penalties-963.md) (the note), no
implementation. Sections, propositions and probes named without a file are those of the note. New numbers come from
`docs/audit/overlapping-group-penalties-963/probe_literature.jl`; its output is recorded in that folder's `README.md`.

## 1. Purpose, scope and how the sources were read

Did the note miss results of the structured-sparsity and overlapping-group literature (Bach, Jenatton, Mairal,
Obozinski and later work) that change its exactness claims, its recommendations, the options for #965 or design (A)?
We read primary sources on trees, forests, DAGs and arbitrary overlap with `ℓ2` groups; network flows for `ℓ∞`
groups; latent groups and other norms with a direct proximal operator; algorithms for the weighted nested problem of
Bien, Bunea and Xiao; statistical applications. "In the sources read" means the sources of Section 10. Every
statement about an algorithm or a theorem carries one label:

- **[read]**: read with its proof in the primary source;
- **[stated]**: stated in the primary source; the proof is absent, deferred, or not checked by us;
- **[secondary]**: a survey, an abstract, a later paper's description, documentation, or code read but not run;
- **[proved here]**: ours, proved in this addendum;
- **[numerical]**: ours, numerical evidence only (a probe line).

Citations give the section, theorem, algorithm or equation and the page of the version read (journal pages for
published articles, printed pages for preprints). Third-party code (SPAMS, `varband`) was read, not run.

## 2. Summary

**Verdict.** In the sources read, no result contradicts Propositions 1–5 or the results of Section 5, and none gives
a finite algorithm for the proximal operator with the general weights of Bien, Bunea and Xiao. The
two near-term recommendations of Section 9 stand and are supported by more estimators than the note names
(Section 7). Design (A) stands (Section 8).

**What changes:** the rows marked changed, qualified or hypotheses added below; a fifth option for #965, the latent
penalty of Yan and Bien (Section 9.1); three more printed one-pass claims with per-coordinate weights (Section 5.3);
the new statements L1–L5 (Section 6). **What does not:** the other rows.

| Statement of the note | Literature | Effect |
|---|---|---|
| Propositions 1–5 | Liu and Ye prove Proposition 2 independently, with zero weights; we found no statement of the condition of Proposition 4 or of its necessity | confirmed |
| Section 5(a): one pass fails for the general weights | Yan and Bien (Appendix E.2), Yu and Bien (published), `varband` 0.9.0 repeat the claim | confirmed |
| Section 5(d): no fixed number of root solves | no finite algorithm; interior-point and Newton methods are generic engines | confirmed |
| Section 5(e) item 2: two-value theory open | Frobenius proofs carry over on our reading | changed (Frobenius part) |
| Section 9: sparse-group papers not checked | Simon et al. (12), Friedman et al. (11), narrower hypotheses | confirmed |
| Section 10: exact flow algorithm for `ℓ∞` sums | proved for one positive scalar weight per group, real arithmetic | hypotheses added |
| Section 10: latent value is an optimization problem | finite formula on a nested chain (L4) | qualified |
| Section 8: design (A) | named laminar estimators on general trees; no benchmark | (A) stands |

## 3. Results by structure (only what is new relative to the note)

**Trees and forests, `ℓ2` groups.** Liu and Ye (NIPS 2010) prove Proposition 2 for an index tree whose root is all
coordinates, weights `w ≥ 0` per node, `λ > 0`, one pass in reverse breadth-first order (Definition 1 and (2), p. 2;
Theorem 1, p. 4; we read the main argument, (17)–(22), not Lemmas 1–2) **[stated]**, at a stated cost `O(pd)` for
depth `d` (p. 3) **[stated]**. Yu (NIPS 2013, Theorem 4,
p. 5) proves `P_f(u) = c u`, `c ∈ [0, 1]`, for `f = h(‖·‖)` with `h` increasing **[read]**, the basis of L2.

**DAGs.** For the descendant-set group lasso every source read is iterative or generic: dual block coordinate
descent "while stopping criterion not reached" (Yan and Bien, Algorithm 1, p. 539) **[stated]**; an active set with a
duality-gap stop on the whole learning problem (Bach, arXiv:0809.1493v1, Section 3.2, p. 6) **[stated]**; an SOCP
(Jenatton, Audibert and Bach, Section 4, p. 2789) **[stated]**.

**Arbitrary overlap, `ℓ2` groups.** Mairal et al. (footnote 11, p. 2695): "We are not aware of any efficient
algorithm providing the exact solution of the proximal operator associated to a sum of ℓ2-norms"; they use ADMM on
the whole problem **[stated]**. Mosci et al. (NIPS 2010, p. 2) say a finite computation "is no longer possible in the
case of general overlap", without proof **[stated]**. For `λ₁‖x‖₁ + λ₂ Σ_i w_i ‖x_{G_i}‖₂`, `w_i ≥ 0`, Yuan, Liu and
Ye (NIPS 2011) prove that the soft threshold comes first (Theorem 1, p. 3), give a finite screen for zero groups
(Lemmas 2–3, p. 4) and a computable gap bounding the objective excess (Theorem 2, p. 6) **[read]**; the inner solver
is accelerated gradient on a smooth dual, whose smoothness proof is in a supplement we did not read, stopped at a gap
of `10⁻¹⁰`. Class: certified approximation. No source read proves that a finite method is impossible.

**Arbitrary overlap, `ℓ∞` groups (network flows).** Mairal et al. (JMLR 2011) define the problem for "positive
weights" `η_g`, one per group (Definition 3, p. 2689). Algorithm 1 (an `ℓ1`-ball projection, a max-flow, a split,
recursion) ends "in a finite and polynomial number of operations" (Proposition 6, p. 2709) **[read]** and returns the
proximal point (Proposition 7, pp. 2709–2712) **[stated]**: we read the proof in part, and the authors omit one half
("follows along similar arguments. We will therefore not detail it", p. 2712). The proofs use exact comparisons such
as `ξ_j ≠ γ_j`: real arithmetic. The bound `O(|V||E| log(|V|²/|E|))` for balanced cuts, "but we lose a factor |V|"
otherwise, is in footnote 10 (pp. 2692–2693) **[stated]**; the "improved projection step" (Section 3.4, p. 2693) has
no separate proof in the paper. SPAMS (Python port, commit `897314636898858f957ed82da7995eb27156150b`, read, not run)
decides splits with the absolute constant `EPSILON_MAXFLOW = 1e-10` (`project.h:25`, test at `:2172`), adds a missed
split to a local `flow_missed` that is not reported (`:2128`, `:2176`), and its FISTA parameters set `clever = true`
(`fista.h:147`, passed on at `:3347`), which selects the box-constrained projection (`project.h:2148`) **[secondary]**.
Submodular view: `Σ_g η_g ‖w_g‖∞ = f(abs(w))` with `f` the Lovász extension of `F(A) = Σ_g η_g 1{A ∩ g ≠ ∅}` (Bach,
FnTML, arXiv:1111.6453v2, Section 6.3, pp. 59–60); the divide-and-conquer algorithm of Section 9.1 (pp. 106–107)
solves `f(w) + Σ_j ψ_j(w_j)`, `ψ_j` strictly convex, with at most `p` submodular minimizations **[stated]**. The flow
operator of Obozinski and Bach (arXiv:1205.1240v1, Algorithm 1, p. 19) is for `Ω₂^{F∩}`, which "is not the weighted
ℓ1/ℓp norm with overlap" and has no simple closed form (Section 4, p. 12) **[stated]**.

*Assessment.* Exact under its hypotheses, but it needs a max-flow engine (warm-started preflows, components, a
recursion that can cost a factor `|V|` over one max-flow, footnote 10), and the released code decides splits by
absolute-tolerance tests and does not report a missed split. Under the policy it is not appropriate for ProximalBase.
What would change that: a named estimator needing `ℓ∞` groups with arbitrary overlap, a benchmark at its sizes, and a
duality gap returned with each evaluation.

**Latent groups.** Yan and Bien (p. 539): "there is no known algorithm that solves the proximal operator of Ω_LOG in
a closed form under a tree structure" **[stated]**. The methods for trees and DAGs in the sources read are iterative:
block coordinate descent and path-based block coordinate descent (Yan and Bien, Sections 4.1, 4.3); the identity
minus a projection onto an intersection of cylinders by cyclic projections or projected Newton (Villa et al.,
arXiv:1209.0368v1, Lemma 1); ADMM with a linear rate for a small step (Zhang et al., arXiv:2001.03322v2)
**[stated]**. Duplication (Obozinski, Jacob and Vert, Section 4.6, p. 14) gives a disjoint group lasso on a lifted
vector **[stated]** and needs nothing new from the library. The nested chain is the exception (L4).

## 4. Comparison table

Classes: finite (a fixed number of operations, sorts and scalar roots for a given size); finite-combinatorial;
iterative; certified approximation (with a computable distance bound); generic formulation (a general solver).

| Penalty | Structure | Weights | Algorithm, class | Complexity | Numerical requirements | Source, location | Label |
|---|---|---|---|---|---|---|---|
| `Σ ω_g ‖x_g‖₂` | tree, forest | scalar `≥ 0` | one pass; finite | `O(pd)` stated | scalar shrink | Liu and Ye, Theorem 1, p. 4 | stated (proof read in part) |
| `Σ ω_g ‖x_g‖₂ + h(‖x_G‖₂)`, `G` maximal | laminar | scalar `ℓ2` on containing groups | one pass; finite | one more scalar step | none new | L2; Yu, Theorem 4, p. 5 | proved here |
| `Σ η_k ‖D_k x‖∞` | laminar | `D_j` on `g_k` proportional to `D_k` | one pass; finite (sort, linear level) | `O(Σ size log size)` (ours) | breakpoint comparisons | L1 | proved here |
| `ω‖·‖∞` containing weighted `ℓ1` | laminar | scalar on `ℓ∞` groups | one pass; finite | as above | none new | L3; Yu, Corollary 8, p. 8 | proved here; Yu stated |
| `Σ ‖d^G ∘ x‖₂` | intervals, rectangles | per coordinate (W3) | active set and SOCP, or reweighting; generic | `O(p^3.5 + n_G^3.5)` per SOCP stated | solver tolerance | Jenatton, Audibert, Bach, Section 4, p. 2789 | stated |
| `λ₁‖x‖₁ + Σ w_i ‖x_{G_i}‖₂` | arbitrary | scalar `≥ 0` | screen, dual accelerated gradient, gap; certified approximation | not checked | gap below `1e-10` | Yuan, Liu, Ye, pp. 3–6 | read (dual smoothness stated) |
| `Σ ω_g ‖x_g‖₂` | arbitrary | scalar | no exact operator; ADMM on the problem; iterative | — | — | Mairal et al., footnote 11 | stated |
| descendant-set group lasso | DAG | scalar | dual block coordinate descent; iterative | `O(Σ size)` per cycle | stopping rule open | Yan and Bien, Algorithm 1 | stated |
| `Σ η_g ‖x_g‖∞` | arbitrary | one positive scalar per group | projection, max-flow, recursion; finite-combinatorial | "finite and polynomial" proved; footnote 10 stated | exact comparisons; SPAMS `1e-10` | Mairal et al., Propositions 6–7 | 6 read; 7 stated (half omitted by authors) |
| `f(abs(w)) + Σ ψ_j(w_j)` | submodular `F` | strictly convex `ψ_j` | divide and conquer; finite-combinatorial | at most `p` minimizations | — | Bach, FnTML, Section 9.1 | stated |
| `Ω₂^{F∩}` (not `Σ ‖x_g‖₂`) | arbitrary | per group | min-cuts and shrinks; finite-combinatorial | not checked | value needs the recursion | Obozinski and Bach, Algorithm 1 | stated |
| latent `ℓ2` | nested chain | scalar, increasing (any positive after L4(iii)) | Algorithm 3; finite | `O(p + Dm)` stated; `O(p + D)` (ours) | comparisons; any maximizer | Yan and Bien, Algorithm 3, Appendix C | read; L4 proved here |
| latent `ℓ2` | tree, DAG | scalar | block coordinate descent, projections, ADMM; iterative | — | stopping rules | Yan and Bien; Villa et al.; Zhang et al. | stated |
| BBX nested `ℓ2` | chain | per coordinate | interior point (certified), smoothing Newton (locally quadratic), dual block coordinate descent with a gap (certified) | `O(√N log(1/ε))` interior-point iterations stated; dual `O(1/k)` stated | strict complementarity for Newton; slow dual steps at thresholds | Section 5.1 | stated; numerical |
| sorted `ℓ1` (SLOPE, OWL) | none | nonincreasing | stack algorithm; finite | `O(p log p)` | — | Bogdan et al., Algorithms 3–4 | stated |
| group SLOPE | disjoint | nonincreasing | SLOPE step on group norms; finite | — | — | Brzyski et al., Appendix B.3 | stated |
| squared `k`-support, squared box norm | all `k`-subsets | none | sort and search; finite | `O(d(k + log d))`, `O(d log d)` stated | — | Argyriou et al., Algorithm 1; McDonald et al., Theorem 15 | stated |
| `ℓ1,∞` ball | rows | none | projection by sorting; finite | `O(n log n)` stated | — | Quattoni et al. | stated |

The last four rows are a different family (symmetric or sorted norms and their latent forms), not structured group
penalties in the note's sense; each would be a separate issue. The row operator of the weak hierarchical lasso (Bien,
Taylor and Tibshirani, problem (11), Appendix C, Algorithm 3: a soft threshold and the root of a nonincreasing
piecewise-linear function) is a finite candidate outside the group family **[stated]**.

## 5. The weighted nested problem of Bien, Bunea and Xiao

### 5.1 Algorithms for the general weights, classified

In the sources read no algorithm for these weights is finite.

- *Certified approximation with a worst-case bound that assumes no nondegeneracy.* The proximal problem is an SOCP
  with `N = p` cones (ours). For the primal-dual potential-reduction method Lobo et al. (Section 4, p. 222) state a
  gap below `ε` after at most `ν√(2N) log(η⁽⁰⁾/ε)/δ + ψ⁽⁰⁾/δ` iterations from a strictly feasible start, proof deferred
  to Nesterov and Nemirovskii **[stated]**; each iteration solves a dense system, `O(p³)` here (ours). Andersen et
  al. observe for their primal-dual method on sums of norms the same accuracy and "markedly higher" iteration counts
  when strict complementarity fails (Section 4, p. 255; experiments) **[stated]**.
- *Locally quadratic under strict complementarity.* Qi, Sun and Zhou prove Q-quadratic convergence of a smoothing
  Newton method for a sum of Euclidean norms when `‖y_i*‖ < 1` on the zero norms and `A_O` is nonsingular (Theorem
  5.2, p. 146) **[stated]**; the proximal problem fits after a reformulation (ours). The 8 failures of the Newton-type
  probe (Section 5(d)) have a nonzero group norm below `1e-9 λ w_ℓℓ` at the dual point, close to the case these
  hypotheses exclude (a zero norm whose dual block lies on the boundary).
- *Dual block coordinate descent.* Hong et al. state `f(x^r) − f* ≤ c/r` for cyclic exact block minimization
  (Theorem 6.1, p. 29) **[stated]**; as the dual excess bounds `½‖x − x*‖²`, this gives a primal error `O(r^{−1/2})`
  (ours). Linear rates need a local error bound (Hong et al., Section 1, p. 3) **[secondary]**; we did not prove that
  it fails at a threshold. Observation (line O): with every group of the `p = 4` chain at its threshold, the distance
  after 10, 100, 1000, 10000 sweeps is `0.00914, 0.00156, 0.000168, 1.7e-05`, close to `1/k`; just inside
  (`c = 0.999`) it is `0.00694, 0.000373, 1.15e-12, 7.41e-16` **[numerical]**.

*Inexact steps in outer solvers.* Schmidt, Le Roux and Bach (arXiv:1109.2415v2, (4), p. 5; Propositions 1–2,
p. 6) keep the `O(1/k)` rate of proximal gradient when `Σ √ε_k < ∞` and the `O(1/k²)` rate of the accelerated
method when `Σ k √ε_k < ∞`, `ε_k` the objective error of the `k`-th proximal step, and note that a duality gap gives
such bounds **[stated]**. A library whose operator is called by solvers it does not control cannot schedule `ε_k`.

### 5.2 The theory for weights in the one-pass class

Read in arXiv:1405.6210v1; the norms (35) use `w_ℓ = w_ℓℓ` (Property 1, p. 6).

- *Theorem 13* (p. 32, stated for "weights given by either (6) or (8)"). Its proof (Appendix G.1, pp. 32–38)
  **[read]** uses: Lemma 3 (p. 33), which needs `w_ℓℓ > 0` and `‖A‖_{2,1} ≤ ‖A‖*_{2,1}`, true for `w_ℓm ≥ 0` because
  `w_ℓℓ ‖A_{s_ℓ}‖` is one term under the `ℓ`-th root; Lemma 4 (pp. 33–34), `w_ℓm ≥ 0`; Proposition 1 (pp. 34–35), the
  optimality relation (10) of the minimizer; Proposition 2 (pp. 35–37) and (39)–(42) (pp. 37–38), Cauchy–Schwarz and
  `2λb ≤ aλ² + b²/a` with `a = 2w₀(L(B))`. The bound is written through `Σ_ℓ w_ℓ²`, `w₀(L(B))` and
  `‖S − Σ*‖_{2,∞}`; neither the choice (6) or (8) nor the value of `λ` enters the proof.
- *Theorem 6* (p. 12) and its proof (Appendix H, pp. 38–39) **[read]** add `w₀(L(B)) ≤ w₀(0) ≤ 4p` "for the given
  weights", `(Σ_ℓ w_ℓ²)^{1/2} ≲ p`, Lemma 5 (`‖S − Σ*‖_{2,∞} ≤ max_ij |S_ij − Σ*_ij| · max_ℓ √(2ℓ)/w_ℓ`), Theorem 12
  (no weights) and the event `A_x`. Corollary 2 (p. 13) follows from Theorem 6 with `B` banded.
- *Two-value weights* `w_ℓm = a_ℓ` (`m < ℓ`), `w_ℓℓ = √(2ℓ)`, `a_ℓ ≥ 0`, `Σ_ℓ a_ℓ² ≤ C p` **[proved here, a reading
  of the proofs]**: `w₀(0) = max_m (2m + Σ_{ℓ>m} a_ℓ²) ≤ (2 + C) p`, `Σ_ℓ w_ℓ² = p(p − 1)`, and the factor of Lemma 5
  is 1. So Theorem 13 holds, Theorem 6 holds with `16x²` replaced by `4(2 + C)x²` and other constants in `≲`, and so
  does Corollary 2. Both `a_ℓ = √(2/ℓ)` and `a_ℓ = √2` give `w₀(0) = 2(p − 1)`. The theorems state "(6) or (8)": the
  extension is ours, not the paper's, and is listed in Section 12.
- *Not re-derived.* Theorems 3–5 (pp. 10–11) are stated for `w_ℓ = √(2ℓ)` and any other weights (Theorem 5 adds
  `w_ℓℓ ≥ w_{ℓ+1,ℓ} > 0`); their proofs (Appendix F, p. 31) argue on the one-pass recursion (12), which for weights in
  the one-pass class is the minimizer (Proposition 4(a)). We read the proof of Theorem 3, which uses only
  `w_ℓℓ = √(2ℓ)`, and the first steps of Theorems 4–5. Theorem 9 needs `K̂ = K`, (12) and `Σ_{ℓ≥m} w_ℓm²/(2ℓ) ≲ 1`
  (Section 5 of the note): `a_ℓ = √(2/ℓ)` gives at most `π²/6`, `a_ℓ = √2` gives `1 + Σ_{ℓ>m} 1/ℓ`, of order `log p`.
  Theorem 10 goes through Theorem 9.

### 5.3 Further one-pass statements with per-coordinate weights in print

- Yan and Bien, Appendix E.2, Algorithm 5 (p. 553), weights `w_{ℓ,m} = √|s_ℓ|/(m − ℓ + 1)`: "By Jenatton et al.'s
  (2011) result, a single pass of BCD from gD to g1 will solve the dual problem" **[stated]**. Their Section 4.2
  (p. 541) rests the cost `O(p + D² log(n))` on it.
- Yu and Bien, published version (JMLR 18(42), 2017): weights (7) `1/(ℓ − m + 1)²` (p. 5), Algorithm 3 (one pass,
  p. 31), proof of Theorem 8: "By Jenatton et al. (2011), we can get a solution to (10) in a single pass as described
  in Algorithm 3" (p. 32) **[stated]**.
- `varband` 0.9.0 (CRAN archive, read, not run): `elliproj_w` (`src/varbandcpp.cpp:172`–`197`) is one loop over the
  groups with weights `1/(ℓ − m + 1)²` (`:184`–`185`), called inside the ADMM loop (`:256`) **[secondary]**.
- Jenatton, Audibert and Bach (JMLR 12): `Ω(w) = Σ_G ‖d^G ∘ w‖₂`, a variable "is allowed to be weighted differently"
  in two groups (eq. (1), p. 2780); (W3) `d^G_j = ρ^{|{H ∈ G : H ∋ j, H ⊂ G, H ≠ G}|}`, `ρ ∈ (0, 1)`, default `0.5`
  (p. 2798). On the chain `{1} ⊂ {1,2} ⊂ … ⊂ {1, …, K}` group `ℓ` has `d^ℓ_j = ρ^{ℓ−j}`, so `w₁₁ = 1`, all
  `w_ℓ1, w_ℓ2 > 0` and `w_j2/w_j1 = 1/ρ > 1`: the hypotheses of the Claim of Section 5(a), so for `K ≥ 3` and each
  `λ > 0` some input gives a one-pass point that is not the proximal point **[proved here]**. Line W3 (`ρ = 0.5`,
  `y = (2, 1.5, 1)`, `t = 0.5`): distance `0.0124 > 2√(2 gap) = 1.98e-07`, gradient entry `0.0129` **[numerical]**.
  They compute with an SOCP solver or reweighting (Section 4, p. 2789; Appendix H) and claim no proximal operator:
  not an error of theirs, but one pass with (W3) would not compute their estimator.

## 6. New claims with proofs

Notation of Section 2; the Lemma of Section 4 is applied through condition (C).

**L1 (weighted `ℓ∞` groups).** Laminar groups, an inclusion order, `t ≥ 0`, `N_k(z) = η_k ‖D_k z‖∞`, `D_k` positive
diagonal, `η_k ∈ [0, ∞]` (`∞` the indicator of `{0}`). If for each pair `k` before `j` with `g_k ⊆ g_j` and
`0 < η_j, η_k < ∞` the restriction of `D_j` to `g_k` is `c D_k` for some `c > 0`, one pass is exact **[proved here]**.
*Proof.* The dual norm of `‖D·‖∞` is `Σ_i |u_i|/d_i`, so `P_j` replaces `v = x_{g_j}` by `v` minus its projection on
`{u : Σ_i |u_i|/d_i ≤ tη_j}`: zero if `Σ_i |v_i|/d_i ≤ tη_j`, otherwise `x_i = sign(v_i) min(|v_i|, θ/d_i)` with
`θ > 0` solving `Σ_i (|v_i| − θ/d_i)₊/d_i = tη_j` (the projection's optimality condition). So `D_j x_{g_j}` is `D_j v`
clipped at `θ`, and `D_k x_{g_k} = c^{−1} D_j x_{g_k}` is `w = D_k v_{g_k}` clipped at `θ/c`. Clipping at a positive
level keeps signs and zeros and enlarges the set where `|w_i|` is largest (or leaves `w` unchanged), so `∂‖·‖∞` at
the clipped vector contains `∂‖·‖∞(w)`; with `∂N_k(z) = η_k D_k ∂‖·‖∞(D_k z)`, (C) holds. If `P_j` sets the group to
zero, is the identity (`η_j = 0` or `t = 0`) or `η_j = ∞`, the Fact applies; `η_k = 0` makes (C) trivial; for
`η_k = ∞`, `∂N_k` is empty off zero and `P_j` maps zero to zero. The Lemma gives the claim. ∎ One vector `D` shared by
all groups is the case `c = 1`. Lines L1: 300 families with proportional and 300 with shared weights accepted
(largest BigFloat gap/(1+‖y‖²) `1.29e-77`, `1.39e-77`) **[numerical]**. *Otherwise:* `‖z‖∞` on `{1,2}` inside
`‖diag(1,2,1) z‖∞` on `{1,2,3}`, `η = t = 1`, `y = (3, 2.9, 0)`: one pass gives `(107/50, 107/100, 0)` and the
directional derivative of the objective along `−e₁` there is `−7/50` (line L1, rational arithmetic), so the point is
not optimal. We claim no necessity beyond this instance.

**L2 (a radial function on a maximal group).** Laminar groups, an inclusion order, `t > 0`. Non-maximal groups that
contain another group carry `ω‖·‖₂`, `ω ∈ [0, ∞]` (as in Proposition 2), other non-maximal groups any group function.
A maximal group `g_j` (no group strictly contains it and no equal group comes after it) may carry `h_j(‖·‖₂)` with
`h_j : [0, ∞) → (−∞, ∞]` convex, nondecreasing, lower semicontinuous and finite at 0, for example `ωr + (μ/2)r²` or
the indicator of `[0, R]`. Then one pass is exact **[proved here]**. *Proof.* `t h_j(‖·‖)` is proper, closed and
convex, and by Yu's Theorem 4 (i ⇒ iii) its proximal map is `u ↦ c u`, `c ∈ [0, 1]`; so `P_j` multiplies each
contained `x_{g_k}` by `c` and the Fact gives (C). Groups after `j` are disjoint from `g_j` (laminarity, maximality,
order), so `ξ_j ∈ ∂(t h_j(‖·‖) ∘ π_j)(x^j)` stays a subgradient at `x^K`, and the argument of the Lemma gives
`y − x^K ∈ ∂(tΩ)(x^K)`. ∎ A ridge term `(μ/2)‖x‖²` on all coordinates is one maximal group containing all others:
one pass, then division by `1 + tμ` (Yu, Example 5, p. 6, for any gauge **[read]**). Lines L2: 300 families
accepted (`1.07e-77`); `0.8‖z‖₂` on `{1,2}` and `0.5r + r²` (`r = ‖z‖₂`) on `{1,2,3}`, `y = (2, 3, 1.5)`, `t = 0.3`,
accepted with gap 0; the radial function on the *contained* group `{1,2}` instead is refuted (largest gradient entry
`0.0984`, all group norms nonzero) **[numerical]**. In the sources read this is the one further `ℓ2`-based structure
with a direct proximal operator.

**L3 (weighted `ℓ1` inside `ℓ∞`).** Laminar groups, an inclusion order, `t ≥ 0`, `ω_k‖·‖∞` (`ω_k ∈ [0, ∞]`) on each
group that contains another, and `ω‖·‖∞` or `Σ_i c_i |z_i|` (`c_i ≥ 0`) on the others: one pass is exact **[proved
here]**. *Proof.* `Σ_i c_i |z_i|` is the sum of the singleton groups `c_i ‖z_i‖∞`, and its step is their steps
(Proposition 1). With the singletons in place of `g_k` and in its place in the order, the family is laminar with
scalar `ℓ∞` groups only, so Proposition 3 applies. ∎ Yu's Corollary 8 (p. 8) states the analogue for Ky Fan norms
with a sketch **[stated]**. Line L3: 300 families accepted (`4.47e-78`) **[numerical]**.

**L4 (latent penalty on a nested chain).** Groups `g_1 ⊂ … ⊂ g_D`, increments `s_i = g_i ∖ g_{i−1}` (`g_0 = ∅`),
weights `w_ℓ > 0`, `Ω(x) = min{Σ_ℓ w_ℓ ‖v^ℓ‖₂ : Σ_ℓ v^ℓ = x, supp v^ℓ ⊆ g_ℓ}`. Its dual norm is
`max_ℓ ‖u_{g_ℓ}‖/w_ℓ` (Obozinski, Jacob and Vert, Lemma 3, p. 10) **[stated]**.

- (iii) *Redundancy.* If `w_k ≥ w_j` for some `j > k`, then `‖u_{g_k}‖/w_k ≤ ‖u_{g_j}‖/w_j` for all `u`; removing
  group `k` (merging `s_k` into `s_{k+1}`) leaves the dual norm, hence the norm and its proximal operator, unchanged
  **[proved here]** (Obozinski, Jacob and Vert state it, p. 27). Afterwards the weights increase strictly, as
  Algorithm 3 requires.
- (i) *Hull form.* Let `W_j = w_j²`, `Z_j = ‖y_{g_j}‖²` (`W_0 = Z_0 = 0`), `c` the least concave majorant of the points
  `P_j = (W_j, Z_j)` and `σ_i` its slope on `[W_{i−1}, W_i]`. Then `prox_{λΩ}(y)_{s_i} = (1 − λ/√σ_i)₊ y_{s_i}`
  **[proved here]**, given Lemma 3 of Yan and Bien, whose proof (Appendix C, pp. 550–552: an explicit decomposition
  checked against the optimality conditions of Obozinski, Jacob and Vert, Lemma 11) we read in full **[read]**.
  *Proof.* `f(j, k)²` of Algorithm 3 is the slope of the chord `P_k P_j`. By induction over the knots, the line
  through a knot `P_k` with slope `s = max_{j>k} f(j, k)²` lies on or above all points: those right of `P_k` by
  maximality, those left of it because they lie under the previous line, whose slope is at least `s` (the maximizer
  `P_K` lies under it). Such a line through `P_k` and `P_K` coincides with `c` on `[W_k, W_K]`, so `σ_i = f(K, k)²`
  there, and after the last knot `c` has slope at most the stopping value `λ²`. On a segment Algorithm 3 returns
  `S_G(y_seg, λ√(ΔW)) = (1 − λ/a)₊ y_seg`, as `‖y_seg‖ = a√(ΔW)` with `a = f(K, k)`, and 0 after the last knot. ∎
  Any maximizer in line 5 gives a point of `c` (Appendix C uses only maximality); the last maximizer gives exactly
  the vertices of slope above `λ²`. A monotone stack over the points, sorted by `W`, finds the vertices in `O(D)`
  after the `O(p)` block norms (ours), against `O(p + Dm)` in Lemma 3.
- (ii) *Value.* `Ω(y) = Σ √(W_b − W_a) ‖y_{s_{a+1} ∪ … ∪ s_b}‖₂` over the segments `[P_a, P_b]` of `c` with positive
  slope **[proved here]**. *Proof.* Lower bound: `u_{s_i} = y_{s_i}/√σ_i` (0 where `σ_i = 0`, where `y_{s_i} = 0`) has
  `‖u_{g_ℓ}‖² ≤ W_ℓ` for all `ℓ` because `P_ℓ` lies under `c`, so `Ω(y) ≥ ⟨u, y⟩`, the formula. Upper bound: with
  vertices `0 = k_0 < … < k_m`, slopes `a_1² > … > a_m² > 0` and `A_j = Σ_{i≤j} y_{seg_i}/a_i` (`‖A_j‖² = W_{k_j}`),
  `v^{k_j} = (a_j − a_{j+1}) A_j`, `a_{m+1} = 0`, sums to `y` and costs
  `Σ_j W_{k_j}(a_j − a_{j+1}) = Σ_j a_j (W_{k_j} − W_{k_{j−1}})`, the same number. ∎ (This is the decomposition of
  Appendix C at `λ = 0`.) On a nested chain the latent norm thus has a finite formula.
- (a) *Check of Algorithm 3 as published* **[numerical]**, lines L4, in `BigFloat`, with a duality gap from
  `u = y − x` (`‖u_{g_ℓ}‖ ≤ λ w_ℓ` checked for all `ℓ`) and an explicit decomposition of `x` bounding its norm: 300
  chains with increasing weights (largest gap/(1+‖y‖²) `3.82e-77`); 300 with equal or decreasing weights reduced by
  (iii), certified against all original groups (`1.56e-77`); exact ties in line 5 with the first and the last
  maximizer (both certified, same point); the hull form equal to Algorithm 3 (difference `8.64e-77`); the value
  formula between the bounds (relative spread `3.91e-77`).

**L5 (`λ_max` of the banding prox).** By the zero-band characterization of Section 5 at `L = K`, the proximal point
is zero if and only if `r ∈ λ Σ_ℓ B°_ℓ`, and `Σ_ℓ B°_ℓ = ∂Ω(0)` is the unit ball of the dual norm `Ω°`; so the
smallest such `λ` is `Ω°(r) = min{max_ℓ ‖ξ_ℓ/W[ℓ, 1:ℓ]‖₂ : Σ_ℓ ξ_ℓ = r}`, a convex problem **[proved here]**. The
value `max_ℓ r_ℓ/w_ℓℓ` of `hierband` is that of the decomposition `ξ_ℓ = r_ℓ e_ℓ`, an upper bound. For `p = 3`,
`r = (2, 0)` (probe L) a primal point and a decomposition give equal bounds `0.828` (`2√2 − 2`, line L5), below `1.41`.

## 7. Applications

Each uses laminar groups with one scalar weight per group, so Proposition 2 makes one pass in an inclusion order
exact (ours, from the quoted definitions).

1. *Lag-order selection in vector autoregression.* Nicholson, Wilms, Bien and Matteson (JMLR 2020): penalties (5)–(7)
   (p. 9) on lag suffixes `Φ_i^{(ℓ:p)}`, unit weights; (6) gives each row the chain
   `Φ_i^{(ℓ:p)} ⊃ (Φ_{i,−i}^{(ℓ)}, Φ_i^{(ℓ+1:p)}) ⊃ Φ_i^{(ℓ+1:p)}`, (7) a chain per entry. The proximal problem (10) has
   "g1 ⊂ ··· ⊂ gH" and weights `w_h`, solved by Algorithm 2, one pass `h = 1, …, H` (pp. 12–13) **[stated]**; they
   decline "a more complicated weighting scheme" (p. 11). Hecq, Ternes and Wilms (mixed-frequency VAR,
   arXiv:2102.11780v2) use the chain (3) with weights `card(s_g^(1)) − card(s_g^(p)) + 1` (p. 13); their Appendix A
   (p. 39) describes the operator as groupwise soft thresholding without stating the order **[stated]**.
2. *Tree-guided multi-response regression.* Kim and Xing (AOAS 2012): the tree lasso (3.1) (p. 8),
   `Σ_j Σ_v w_v ‖β_j^{G_v}‖₂`, `G_v` the responses under node `v` (singletons at the leaves), the same tree for every
   covariate `j`, `w_v = g_v Π_{m∈Ancestors(v)} s_m` (internal) or `Π_{m∈Ancestors(v)} s_m` (leaf), `s_v = h_v`,
   `g_v = 1 − h_v` (p. 9). They use smoothing proximal gradient (Section 3.2, p. 11) **[stated]**, an approximate
   method; Proposition 2 gives the exact operator. Hierarchical sparse coding (Jenatton et al.) is a second user.
3. *Adaptive basis truncation, partially linear additive models.* Haris, Shojaie and Simon compute their estimator as
   (11), `Σ_k w_k ‖β̃_{k:K̃}‖₂` in QR coordinates, `w_k = k^m − (k − 1)^m` (eq. (5)), by Algorithm 1, one pass
   `k = K̃, …, 1` (Section 3.2) **[stated]**. SPLAM (Lou, Bien, Caruana, Gehrke):
   `Σ_j [α ‖β_j‖₂ + (1 − α) ‖β_{j,−1}‖₂]` (p. 3), dual solved inner group first, (7) (p. 4) **[stated]**.

Banded covariance estimation through the latent chain (Yan and Bien (20); graph-guided banding with a global
bandwidth, Bien, arXiv:1606.00451v2, Algorithm 3 at p. 7 and `w_b = |g_b|^{1/2}` "a good choice" at p. 6) is the
application of option 5 (Section 9.1). Sparse-group users: Simon et al. ((4), (12)) and the sparse lag penalties of
BigVAR with cardinality weights (arXiv:1702.07094v1, Table 1, rows (5)–(6), p. 3) **[stated]**.

## 8. Design (A) reassessed

Section 8 of the note: "Evidence that would change it: two named estimators that need laminar groups on generic index
lists with the same weight contract, and a benchmark showing the generic layout no slower and no larger than the
specialized ones at their sizes." The literature supplies the first half: the tree lasso of Kim and Xing and the
hierarchical sparse coding of Jenatton et al. need laminar groups on general trees with one scalar weight per group.
It does not supply the second: no source read benchmarks a generic list layout against specialized ones. The other
estimators of Section 7 are chains or two-level families. (A) stands, and its tree type should accept a general tree.

What the estimators imply for that type: zero weights (Kim and Xing's `w_v` is zero when `g_v = 0`); singleton and
duplicate groups (both allowed by Proposition 2); the same tree over the rows or fibers of a matrix (Kim and Xing over
covariates, HLag over rows); a compact chain or parent-map layout, since a chain of suffixes as lists takes `O(p²)`
indices for `p` coefficients, evaluated in `O(n + K)` by a post-order array, not by recursion; an optional ridge term
(L2). What they do not imply: an inner solver or engine, or per-coordinate `ℓ2` weights (HLag declines them; Kim and
Xing, HierBasis as computed and SPLAM use scalar weights).

## 9. Recommendations

### 9.1 For #965 (the options of Section 5(e), revised; the choice stays the maintainer's)

1. Basic weights (2.5): unchanged.
2. Two-value weights: computation unchanged (one pass, `p − 1` root solves after the band norms). Theory: Theorems 13
   and 6 and Corollary 2 hold on our reading of the proofs for `w_ℓℓ = √(2ℓ)`, `Σ_ℓ a_ℓ² ≤ C p` (Section 5.2), to be
   verified independently (Section 12); the proof of Theorem 3 applies; Theorems 4, 5, 9 and 10 were not re-derived,
   and Theorem 9 as written needs `Σ_{ℓ≥m} w_ℓm²/(2ℓ) ≲ 1` (`a_ℓ = √(2/ℓ)`, not `√2`).
3. General weights to a certified gap: unchanged. No finite method in the sources read; the best worst-case guarantee
   is an interior-point method, a generic engine (Section 5.1); an evaluation would take a tolerance per call and
   return its gap.
4. One pass with (2.6): unchanged; printed in three more places (Section 5.3).
5. *New:* the latent penalty of Yan and Bien ((20), p. 544) on the nested band unions `s_{1:m}` nearest the diagonal,
   weights `w_m = √|s_{1:m}|`, diagonal unpenalized. A different penalty from (2.1): latent groups, inner rather than
   outer unions, one scalar weight per group. Proximal operator finite and exact for strictly increasing weights (any
   positive weights after L4(iii); Algorithm 3,
   Lemma 3, Appendix C **[read]**), cost `O(p + Dm)` stated (Appendix D) or `O(p + D)` (L4), no root solve. Theorems
   1–3 (pp. 544–545) state `K̂ ≤ K`, `K̂ ≥ K` under (21) and `‖Σ̂ − Σ*‖_F² ≲ pK log p/n` with high probability
   **[stated; Appendices I–K not checked]**. On a path with unit group-lasso and `√i` latent weights, the ratio of the
   shrinkage at distance `h` decays exponentially in `h` for the group-lasso form and stays above a fixed `δ` for the
   latent form (Propositions 1–2, pp. 537–538) **[stated]**. Graph-guided banding with a global bandwidth uses the
   same operator (Bien, p. 7).

### 9.2 Sparse-group follow-up

Unchanged. Simon et al. ((12), p. 6) derive the soft threshold at `tαλ` followed by the group shrink for one `ℓ1`
coefficient on all coordinates, one scalar weight per group (`√p_l` in (4)) and every coordinate in a group
**[read]**; Friedman, Hastie and Tibshirani ((11), p. 5) give the formula for an orthonormal group design
**[stated]**. The Corollary of Proposition 2 is broader (per-coordinate `c_i`, coordinates in no group).

### 9.3 Tree-structured group `ℓ2` follow-up

Requirements: laminar validation and an inclusion order at construction; weights in `[0, ∞]`; singleton and duplicate
groups; one tree over the rows or fibers of a matrix; a parent-map or chain layout evaluated in `O(n + K)` by a
post-order array; optionally a ridge term (L2), with only a maximal group allowed a non-homogeneous function (the L2
counterexample). Named estimators: hierarchical sparse coding, the tree lasso of Kim and Xing, the HLag penalties,
the mixed-frequency penalty of Hecq et al., HierBasis as computed, SPLAM, the group lasso on descendant sets of a tree.

### 9.4 A possible third follow-up, if the maintainer admits it under the policy

"Latent group `ℓ2` penalty on a nested chain": Algorithm 3 as published (or the hull form), the redundancy rule
L4(iii), the value formula L4(ii), a certified probe as in lines L4. Users: option 5 and graph-guided banding with a
global bandwidth without the eigenvalue constraint.

### 9.5 Excluded, with the reason

Arbitrary overlap with `ℓ2` groups (no exact operator in the sources read; at best a certified approximation) or
`ℓ∞` groups (a flow engine); the DAG group lasso and latent groups on trees or DAGs (iterative); the vector weights of
Jenatton, Audibert and Bach (one pass fails); the general weights of Bien, Bunea and Xiao as an exact operator; the
sorted and symmetric norms (a separate family); weighted `ℓ∞` groups (L1: exact, but no named estimator found).

## 10. Annotated bibliography

*Trees and overlap with `ℓ2` groups.*
- Liu, Ye. Moreau-Yosida regularization for grouped tree structure learning. NIPS 23, 2010 (proceedings),
  https://papers.nips.cc/paper_files/paper/2010/file/d490d7b4576290fa60eb31b5fc917ad1-Paper.pdf. Proposition 2.
- Yu. On decomposing the proximal map. NIPS 26, 2013 (proceedings version, as in the note). Theorem 4, Example 5,
  Corollary 8: basis of L2 and L3.
- Jenatton, Audibert, Bach. Structured variable selection with sparsity-inducing norms. JMLR 12:2777–2824, 2011,
  https://jmlr.org/papers/v12/jenatton11b.html (published). Weights (W3); SOCP and reweighting.
- Yuan, Liu, Ye. Efficient methods for overlapping group lasso. NIPS 24, 2011 (proceedings),
  https://proceedings.neurips.cc/paper_files/paper/2011/file/03c6b06952c750899bb03d998e631860-Paper.pdf. Screen, gap.
- Mosci, Villa, Verri, Rosasco. A primal-dual algorithm for group sparse regularization with overlapping groups.
  NIPS 23, 2010 (proceedings),
  https://papers.neurips.cc/paper_files/paper/2010/file/8c6744c9d42ec2cb9e8885b54ff744d0-Paper.pdf.
- Bach. Exploring large feature spaces with hierarchical multiple kernel learning. arXiv:0809.1493v1. DAG norm.

*Flows and submodular functions.*
- Mairal, Jenatton, Obozinski, Bach. Convex and network flow optimization for structured sparsity. JMLR
  12:2681–2720, 2011, https://jmlr.org/papers/v12/mairal11a.html (published). Exact `ℓ∞` flow algorithm, footnotes.
- SPAMS, Python port, github.com/getspams/spams-python, commit `897314636898858f957ed82da7995eb27156150b`,
  `project.h`, `fista.h` (read, not run). Tolerances of the released flow code.
- Bach. Learning with submodular functions: a convex optimization perspective. FnTML 6(2–3), 2013,
  arXiv:1111.6453v2. Lovász extension, divide and conquer.
- Obozinski, Bach. Convex relaxation for combinatorial penalties. arXiv:1205.1240v1. Flow operator, other norm.

*Latent groups.*
- Obozinski, Jacob, Vert. Group lasso with overlaps: the latent group lasso approach. arXiv:1110.0413v1. Dual norm,
  optimality conditions, duplication, redundant groups.
- Yan, Bien. Hierarchical sparse modeling: a choice of two group lasso formulations. Statist. Sci. 32(4):531–560,
  2017, doi:10.1214/17-STS622 (published; arXiv:1512.01631). Algorithm 3 and proof, (20), Appendix E.2.
- Villa, Rosasco, Mosci, Verri. Proximal methods for the latent group lasso penalty. Comput. Optim. Appl.
  58:381–407, 2014, arXiv:1209.0368v1. Iterative general operator.
- Zhang, Liu, Davanloo Tajbakhsh. A first-order optimization algorithm for statistical learning with hierarchical
  sparsity structure. arXiv:2001.03322v2. ADMM on DAGs.
- Bien. Graph-guided banding of the covariance matrix. JASA 114, 2019, arXiv:1606.00451v2. Uses Algorithm 3.

*The weighted nested problem.*
- Bien, Bunea, Xiao. Convex banding of the covariance matrix. JASA 111(514):834–845, 2016, arXiv:1405.6210v1.
  Appendices F–H for Section 5.2.
- Yu, Bien. Learning local dependence in ordered data. JMLR 18(42):1–60, 2017,
  https://jmlr.org/papers/v18/16-198.html (published). One-pass claim, weights (7).
- `varband` 0.9.0, CRAN archive (read, not run). One-pass weighted operator inside ADMM.
- Lobo, Vandenberghe, Boyd, Lebret. Applications of second-order cone programming. Linear Algebra Appl.
  284:193–228, 1998, https://web.stanford.edu/~boyd/papers/pdf/socp.pdf (published text). Interior-point bound.
- Andersen, Christiansen, Conn, Overton. An efficient primal-dual interior-point method for minimizing a sum of
  Euclidean norms. SIAM J. Sci. Comput. 22(1):243–262, 2000, https://cs.nyu.edu/~overton/papers/pdffiles/sumnorms.pdf
  (published text). Behavior without strict complementarity.
- Qi, Sun, Zhou. A primal-dual algorithm for minimizing a sum of Euclidean norms. J. Comput. Appl. Math.
  138(1):127–150, 2002 (published text, authors' copy). Local quadratic rate.
- Hong, Wang, Razaviyayn, Luo. Iteration complexity analysis of block coordinate descent methods. Math. Program.
  163:85–114, 2017, arXiv:1310.6957v2. Sublinear rate.
- Schmidt, Le Roux, Bach. Convergence rates of inexact proximal-gradient methods for convex optimization. NIPS 24,
  2011, arXiv:1109.2415v2. Error conditions.

*Other norms with finite operators.*
- Bogdan, van den Berg, Sabatti, Su, Candès. SLOPE. AOAS 9(3):1103–1140, 2015, arXiv:1407.3824v2.
- Brzyski, Gossmann, Su, Bogdan. Group SLOPE. JASA 114:419–433, 2019, arXiv:1610.04960v1.
- Argyriou, Foygel, Srebro. Sparse prediction with the k-support norm. NIPS 25, 2012, arXiv:1204.5043v2.
- McDonald, Pontil, Stamos. New perspectives on k-support and cluster norms. JMLR 17(155), 2016, arXiv:1512.08204v1.
- Quattoni, Carreras, Collins, Darrell. An efficient projection for ℓ1,∞ regularization. ICML 2009, 857–864,
  https://people.csail.mit.edu/mcollins/papers/icml09.pdf.
- Bien, Taylor, Tibshirani. A lasso for hierarchical interactions. AoS 41(3):1111–1141, 2013, doi:10.1214/13-AOS1096,
  arXiv:1205.5050v3.

*Applications.*
- Nicholson, Wilms, Bien, Matteson. High dimensional forecasting via interpretable vector autoregression. JMLR
  21(166):1–52, 2020, https://jmlr.org/papers/v21/19-777.html (published). HLag chains.
- Hecq, Ternes, Wilms. Hierarchical regularizers for mixed-frequency vector autoregressions. JCGS 31(4), 2022,
  arXiv:2102.11780v2. Chain with cardinality-difference weights.
- Kim, Xing. Tree-guided group lasso for multi-response regression with structured sparsity, with an application to
  eQTL mapping. AOAS 6(3):1095–1117, 2012, doi:10.1214/12-AOAS549, arXiv:0909.1373v3. Tree lasso.
- Haris, Shojaie, Simon. Nonparametric regression with adaptive truncation via a convex hierarchical penalty.
  Biometrika 106(1), 2019, arXiv:1611.09972v4. Suffix chain.
- Lou, Bien, Caruana, Gehrke. Sparse partially linear additive models. JCGS 25(4), 2016, arXiv:1407.4729v3.
- Simon, Friedman, Hastie, Tibshirani. A sparse-group lasso. JCGS 22(2):231–245, 2013; authors' preprint,
  https://web.stanford.edu/~hastie/Papers/SGLpaper.pdf. Proximal step (12).
- Friedman, Hastie, Tibshirani. A note on the group lasso and a sparse group lasso. arXiv:1001.0736v1. (11).
- Nicholson, Matteson, Bien. BigVAR: tools for modeling sparse high-dimensional multivariate time series.
  arXiv:1702.07094v1. Sparse-group lag penalties.

## 11. Not obtained, not read, not checked

- Not obtained: the typeset article of Bien, Bunea and Xiao and its supplement; Hochbaum and Hong (1995) and Gallo,
  Grigoriadis and Tarjan (1989); Zhou, Toh and Sun (2003), cited for quadratic convergence without strict
  complementarity; Jacob, Obozinski and Vert (ICML 2009), whose extended version we read.
- Not checked: Appendices B and I–K of Yan and Bien; Appendix F of Bien, Bunea and Xiao beyond Theorem 3 and the first
  steps of Theorems 4–5, and Appendix J; the omitted half of Proposition 7 of Mairal et al.; the proofs behind Lobo et
  al., Andersen et al., Qi, Sun and Zhou, Hong et al., Schmidt, Le Roux and Bach, Bach's Section 9.1, Obozinski and
  Bach's Algorithm 1; the supplement of Yuan, Liu and Ye.
- Read as preprints: Kim and Xing, Haris et al., SPLAM, Simon et al., Villa et al., Hong et al., Schmidt et al., Hecq
  et al., Bien, McDonald et al. Not run: any package named here.

## 12. Claims that need independent proof or numerical verification

| Claim | Statement | Status |
|---|---|---|
| L1 | weighted `ℓ∞` groups with `D_j` on `g_k` proportional to `D_k`: one pass exact; fails on the recorded instance otherwise | proved here; lines L1 |
| L2 | `h(‖·‖₂)` on a maximal group, Proposition 2 below it: one pass exact; fails on a contained group (instance) | proved here; lines L2 |
| L3 | weighted `ℓ1` groups inside scalar `ℓ∞` groups: one pass exact | proved here (Propositions 1, 3); line L3 |
| L4(i) | Algorithm 3 equals the hull form; vertices in `O(D)` after the block norms | proved here, given Lemma 3 (read); line L4 |
| L4(ii) | finite value formula for the latent norm on a nested chain | proved here; line L4 |
| L4(iii) | groups with `w_k ≥ w_j`, `j > k`, are redundant | proved here; line L4 |
| L4(a) | Algorithm 3 as published is the proximal operator | numerical (independent of the proof); lines L4 |
| L5 | `λ_max` of the banding prox is `Ω°(r)`; `2√2 − 2` for `p = 3`, `r = (2, 0)` | proved here; line L5 |
| W3 | weights (W3) on a chain with `K ≥ 3`: one pass fails for each `λ > 0` | proved here (Claim of Section 5(a)); line W3 |
| R1 | Theorems 13, 6 and Corollary 2 of Bien, Bunea and Xiao for two-value weights, `w_ℓℓ = √(2ℓ)`, `Σ a_ℓ² ≤ C p` | ours, a reading of the proofs; needs an independent check |
| R2 | primal error `O(r^{−1/2})` of dual block coordinate descent | ours, from Hong et al. (stated) |
| O | distance close to `1/k` with every group at its threshold | numerical only (line O) |
