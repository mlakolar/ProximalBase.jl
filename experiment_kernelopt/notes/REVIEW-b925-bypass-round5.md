# Independent review: B925 small-array bypass, round 5

Reviewed 2026-11-16 on `usc-desktop`, Intel Core i5-13600K. Primary Julia 1.13.4, secondary Julia 1.10.13. This is a workstation screen; the external authoritative timing campaign was neither inspected nor incorporated. All retained scripts, copied test trees, local depot files, and outputs are inside `review-code/round5`; Pkg also used its normal temporary test sandboxes. The five supplied worktrees were read only.

## Verdicts

| Variant | Verdict | Reason |
|:--|:--|:--|
| B925s, `d59ed026` | **REJECT** | It bypasses the pass, but repeats validation and adds an out-of-line call before the generic kernel. Reproducible ordinary short-call regressions exceed 5%. It also does not restore master's structural-query behavior for custom arrays. |
| B925t, `048c151a` | **REJECT** | It removes the repeated validation on the short route, but the large method does not inline into `prox!` on either tested Julia. The extra call still produces widespread regressions above 5%. |
| B925u, `43a9468c` | **APPROVE WITH CHANGES** | Best implementation of the requested fixed cutoff: its short route has one length branch and the same direct generic-kernel call as master. Numerical and normal validation behavior passed differential testing. However, even this variant has reproducible ordinary regressions above 5% at very small lengths on the primary Julia. It needs an **explicit maintainer ruling accepting those regressions**, or a revised implementation and another screen, before proposal. Correct the equivalence wording and strengthen the evidence handling described below. |

**None should be proposed unconditionally under the stated rule.** B925u is the only candidate I would advance for a specific maintainer decision. Its extra helper controls inlining without changing the algorithm, the cutoff, or dispatch by alignment or architecture. This review supplies no waiver of the 5% rule.

## Findings, ordered by severity

### 1. Ordinary short calls still regress; source-level bypass does not mean master-speed execution

The supplied suite was audited before timing. Two complete fresh processes per Julia measured all lengths 1–17 plus 24, 32, 64, 256, and 1024, both element types, ordinary weighted/unweighted `prox!`, weighted/unweighted `value_and_prox!`, and three exceptional-step cases. Each process loaded master, B925m, B925s, B925t, B925u, and a second compilation of master (`MC`). The second process reversed source order and put the control first. Timing processes ran sequentially on logical CPU 2, with one Julia and one BLAS thread; CPU 2's sibling is CPU 3. The desktop was not reserved or frequency locked. Some supplemental probes ran on CPU 10 during parts of the full screen.

The full screen uses 12 rotating/reversing rounds, three batches per measurement, approximately 1 ms per batch, and the minimum batch time. The reported statistic is the median of paired per-round ratios **within each process**, followed by the range across processes. The A/A range is shown for every cell. “Reproducible >5%” below requires both process medians above 1.05 and both A/A medians in [0.98, 1.02]. Unclean-control cells are not evidence of safety.

| Julia | variant | ordinary cells | clean A/A cells | reproducible >5% | >5%, A/A unclean |
|:--|:--|--:|--:|--:|--:|
| 1.10 | B925m | 176 | 130 | 34 | 22 |
| 1.10 | B925s | 176 | 130 | 69 | 32 |
| 1.10 | B925t | 176 | 130 | 63 | 31 |
| 1.10 | B925u | 176 | 130 | 8 | 5 |
| 1.13 | B925m | 176 | 164 | 57 | 5 |
| 1.13 | B925s | 176 | 164 | 87 | 5 |
| 1.13 | B925t | 176 | 164 | 77 | 5 |
| 1.13 | B925u | 176 | 164 | 2 | 0 |


On primary Julia 1.13, two clean full-process examples for **B925u** are:

- Float64, unweighted `prox!`, n=1: **1.072–1.076 × master**.
- Float64, unweighted `prox!`, n=3: **1.057–1.059 × master**.

These are the two full-screen results; the focused confirmations below are deliberately reported separately. The absolute costs are small, fractions of a nanosecond on these inputs, but the ruling is relative and has no nanosecond exemption. B925s/t have much larger and more widespread short-call costs. Their extra calls are visible in generated code, not inferred just from timing.

I ran four additional fresh focused processes (two per Julia, the same six implementations and reversed orders) with 30 rounds and 2 ms batches, three samples. See `bash round5/run-focus.sh`. These compile a smaller subset of cases, so they are a separate experiment, not pooled into the full-screen ranges. All focused comparisons passed and recorded zero allocations.

| Julia | type | operation | n | B925u / master, focused | A/A, focused |
|:--|:--|:--|--:|--:|--:|
| 1.10 | Float64 | prox_g | 1 | 1.020–1.038 | 1.000–1.000 |
| 1.10 | Float64 | prox_g | 3 | 1.012–1.013 | 1.000–1.006 |
| 1.10 | Float32 | prox_gw | 6 | 1.005–1.009 | 0.951–1.001 |
| 1.10 | Float32 | valprox_g | 2 | 0.954–0.998 | 0.952–0.986 |
| 1.10 | Float32 | valprox_g | 6 | 0.936–0.999 | 0.925–0.997 |
| 1.10 | Float32 | valprox_g | 7 | 0.968–1.009 | 0.965–1.001 |
| 1.10 | Float64 | valprox_g | 15 | 0.958–1.002 | 0.955–1.037 |
| 1.10 | Float64 | valprox_g | 6 | 0.951–0.963 | 0.997–1.003 |
| 1.13 | Float64 | prox_g | 1 | 1.067–1.069 | 1.000–1.001 |
| 1.13 | Float64 | prox_g | 3 | 1.022–1.027 | 1.000–1.002 |
| 1.13 | Float32 | prox_gw | 6 | 1.012–1.012 | 1.000–1.000 |
| 1.13 | Float32 | valprox_g | 2 | 1.000–1.000 | 1.000–1.000 |
| 1.13 | Float32 | valprox_g | 6 | 1.003–1.003 | 0.998–1.001 |
| 1.13 | Float32 | valprox_g | 7 | 1.001–1.002 | 1.000–1.000 |
| 1.13 | Float64 | valprox_g | 15 | 0.998–1.001 | 1.000–1.000 |
| 1.13 | Float64 | valprox_g | 6 | 1.000–1.001 | 1.000–1.000 |

**The n=1 Float64 ordinary `prox!` regression on Julia 1.13 survives all four processes:** 1.072–1.076 in the full screen and 1.067–1.069 in the focused screen, with clean controls. That alone triggers the requested maintainer ruling. The n=3 primary regression does **not** survive the focused subset (1.022–1.027); do not describe it as an unavoidable branch cost. The full LTS screen has eight clean >5% B925u cells, including Float32 weighted `prox!` at n=6 (1.143–1.183) and Float64 unweighted `value_and_prox!` at n=6 (1.107–1.128). Those large LTS penalties also disappear in the focused subset; some focused LTS controls are themselves unclean. This is evidence of sensitivity to compilation/layout/workload, even when a particular A/A pair is clean. It strengthens the need for the external timing campaign and limits causal claims about any one extra instruction. It does not erase the consistently slow primary n=1 result.


Full tables, including every n=1–17, all requested operations, both types, and both Julias, appear in Appendix A. Larger-array exceptional fallback is tabulated separately in Appendix B, including direct paired ratios against B925m.

Evidence and reproduction:

```sh
bash round5/run-timings.sh
python3 round5/summarize.py
python3 round5/derived-tables.py
```

`run-timings.sh` contains all environment variables and exact driver commands. Its essential invocation is:

```sh
KO_SUITES=steps KO_TYPES=Float64,Float32 KO_ROUNDS=12 \
KO_BATCH_MS=1 KO_SAMPLES=3 KO_REF=M KO_TREES_ONLY=1 \
KO_FILTER='/n([1-9]|1[0-7]|24|32|64|256|1024)/' \
KO_TREES='M=.../base3;B925m=.../pr-B925;B925s=.../B925s;B925t=.../B925t;B925u=.../B925u' \
KO_OUTPUT=... taskset -c 2 julia +release --startup-file=no --threads=1 \
  --check-bounds=no --project=round5/env-v1.13 \
  /path/to/pbopt/harness/experiment_kernelopt/bench/run.jl
```

Master ns in the tables is the pooled median of the recorded reference times; ratios always use the paired per-process statistic above. The LTS command uses `+lts` and `env-v1.10`. `JULIA_DEPOT_PATH` puts `round5/depot` first, followed by the existing depot for reads; `JULIA_PKG_OFFLINE=true`. Copies of both supplied environments reside in `round5`.

```text
local-julia1.10-p1 308 cases × 6 implementations × 12 unique rounds = 22176 raw rows; all status ok; allocations 0; DONE
local-julia1.10-p2 308 cases × 6 implementations × 12 unique rounds = 22176 raw rows; all status ok; allocations 0; DONE
local-julia1.13-p1 308 cases × 6 implementations × 12 unique rounds = 22176 raw rows; all status ok; allocations 0; DONE
local-julia1.13-p2 308 cases × 6 implementations × 12 unique rounds = 22176 raw rows; all status ok; allocations 0; DONE
```

Full raw/meta/provenance files are in `round5/timing/local-julia{1.13,1.10}-p{1,2}`; `round5/supplied-digest-summary.txt` preserves the supplied digest run as an additional cross-check. No digest fixture data is mixed with real timings.

### 2. The cutoff does not repair the exceptional-step scan penalty at n ≥ 16

All three retain B925m's scan followed by the complete generic loop when the scan rejects the steps. A first exceptional step stops the scan **after its block of up to 256**, not after that entry. A last exceptional step scans the whole array. Thus n=16 and n=17 still incur an extra scan, and n=1024 with a last zero step still reads all 1024 steps twice. A first zero at n=1024 scans 256, then runs the full generic loop.

For example, unweighted last-zero fallback at n=16 on primary Julia/Float64 costs **1.443–1.445 × master** for u (B925m itself: 1.407–1.409). At n=1024, u costs 1.018–1.038 × master for a first zero across the primary type/process cells, versus 1.065–1.131 for a last zero. The two types differ; the exact per-type values are in Appendix B. On LTS the broad pattern is similar: a substantial small fallback cost, and a smaller first-block cost amortized over n=1024.

There is also a **full-screen timing difference relative to B925m**: primary Julia/Float64, unweighted last zero at n=32, u/B925m = **1.236–1.252**, with A/A 0.999–1.001. Its u/master range is 1.523–1.582. Thus “same fallback algorithm” must not be reported as “same fallback timing.” Most other LTS direct variant/B925m fallback ratios are near one (all observed process medians lie in 0.986–1.031 across the three variants and these fallback cells); primary results have more substantial outliers and cross-process variation.

I investigated those primary outliers in two more fresh focused fallback processes, 30 rounds, 2 ms batches (`bash round5/run-focus-fallback.sh`):

| operation | n | s / B925m | t / B925m | u / B925m | A/A |
|:--|--:|--:|--:|--:|--:|
| prox_gwx_last | 64 | 1.005–1.005 | 0.918–1.003 | 0.901–1.119 | 1.000–1.000 |
| prox_gx_first | 32 | 0.931–1.065 | 0.935–1.081 | 0.938–1.033 | 0.974–0.994 |
| prox_gx_last | 32 | 0.799–1.000 | 0.797–0.983 | 0.809–1.034 | 0.989–1.088 |

The n=32 extra u penalty does not persist in this subset, and one of its controls is unclean. The weighted n=64 comparisons also vary widely. I have not isolated the cause; code/data placement and compilation context are plausible explanations, not established diagnoses. The reliable structural conclusion is that every candidate retains the extra scan and generic fallback. The numerical tables expose rather than suppress the observed timing differences.


These are inherited costs, not a new mathematical defect in the cutoff. None of the variants removes them. The compiler/call-boundary differences can change timings relative to B925m; Appendix B reports those ratios instead of asserting timing identity. A cutoff-only revision cannot generally remove this cost while retaining the requested n ≥ 16 route. Treat it as a disclosed B925m tradeoff for the maintainer, not an optimization supplied by this revision.

### 3. The digest can label invalid or uncontrolled data as a successful result

In `bench/steps_digest.py`, the set `bad` records failed status or nonzero allocations, but only its **count is printed**. Those cases remain eligible for “faster” and “slower” classification. A missing A/A control is even replaced by `[1.0]`:

```python
k = per.get((c, ctl), [1.0])
```

I exercised the actual digest with two `DONE` directories containing a candidate marked `DIFF`, allocating 16 bytes, timing 0.9 × master, and **no control rows**:

```sh
python3 round5/digest-fixture.py
```

Actual output:

```text
## test-julia1.13 (2 processes; reference M; status or allocation problems: 1)
| 8 | 100.0 | 0.90–0.90 faster | 1.00–1.00 |
| U | 1-16 | prox_g | 1 | 0 | 1 | 0 |
```

This is a real digest validation defect. It did not invalidate this review's raw measurements: the completed full runs have all six implementations and all 12 rounds for each case, zero output differences, and zero recorded allocations. My summary reads raw/meta data, and the final coverage check validates completeness explicitly.

A separate fixture (`python3 round5/digest-policy-probe.py`) with a valid 1.20 candidate ratio and 1.03 control ratio produced `slower=0, control unclean=1`. The detailed digest labels such cells `(control)`, so it does expose uncertainty. **Zero “slower” cells is not a permission to propose a patch** when cells are unclean. Rerun them; do not turn the control gate into a waiver of the maintainer rule.

Other digest limitations: it does not require two processes, equal process coverage, complete rounds, or consistent references before classifying; a candidate missing from a process simply has fewer medians. Its `1-16` summary bucket includes n=16, which actually takes the scan route. Change that split to `1-15` and `16+`. Printing ratios to two decimals can also obscure proximity to 1.05; use at least three digits near the gate.

### 4. “Identical in every observable respect” is too strong for AbstractArray

For normal, stable array metadata, the tested numerical results, values, exception type/message, and validation-before-write behavior agree with the requested references. The generic scalar arithmetic is reached by `invoke` below 16. However, the **structural queries are observably different**. This matters to the literal question, especially since the tests themselves regard custom-array read counts as observable.

I used a valid-data `AbstractVector` that records `axes`, `size`, and `length` calls, with ordinary scalar `getindex`. For n=8, public `prox!` gives:

```text
TRACE M [:axes, :size, :axes, :length, :axes]
TRACE s [:axes, :size, :axes, :length, :length, :size, :axes, :length, :axes]
TRACE t [:axes, :length, :size, :axes, :length, :axes]
TRACE u [:axes, :length, :size, :axes, :length, :axes]
```

B925s repeats `_checkwrite`/`_checkparam`; t/u add the cutoff's early length query. With `axes` and `size` defined normally but `length(x)` throwing, and a wrong-sized output, master and s report `DimensionMismatch` first; t/u report the length-query `ErrorException` first. All leave the output unchanged:

```text
ORDER M DimensionMismatch ... output has size (9,) and the input (8,) unchanged=true
ORDER s DimensionMismatch ... output has size (9,) and the input (8,) unchanged=true
ORDER t ErrorException length queried unchanged=true
ORDER u ErrorException length queried unchanged=true
```

A second probe makes `length` throw only on its second call. It shows successful master calls becoming errors under repeated structural queries:

```text
steps M ok length_calls=1 y=[0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
steps s ErrorException: length queried twice length_calls=2 y=[7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]
steps t ok length_calls=1 y=[0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
steps u ok length_calls=1 y=[0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
input M ok length_calls=1 y=[0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
input s ErrorException: length queried twice length_calls=2 y=[7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]
input t ErrorException: length queried twice length_calls=2 y=[7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]
input u ErrorException: length queried twice length_calls=2 y=[7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]
```

The LTS run produced the same outcomes.

These custom methods deliberately make structural observations visible. They are not failures of ordinary Vector/view/SparseIterate numerical behavior, and a stable array normally implements pure, nonthrowing `length`. Nevertheless, the universal claim as asked is false for **every** candidate. At n ≥ 16, each candidate also introduces an extra `length(x)` observation relative to B925m; u changes stack frames as well. Restrict the compatibility claim to stable array metadata and the documented numerical/validation contract. If the literal requirement includes arbitrary structural-query side effects or identical exception stack traces, none of these revisions can be approved as equivalent.

Evidence: `round5/semantics.jl`, `round5/query-effects.jl`, and their release/LTS logs. Source evidence: s's cutoff follows its two checks; t's cutoff precedes them; u's inline selector calls either the generic kernel or the separate checked scan helper. Saved diffs are `round5/B925{s,t,u}-vs-{master,B925m}.diff`.

### 5. The tests cover the cutoff well, but do not prove all claimed equivalence

The shared added test file is byte-identical in s/t/u (SHA256 `b2b65823c3a28b6ef77429accdb5d6ee08606ca392f4bd80c85e4d49396436d4`). All three copied trees passed `scripts/test.jl infinite_weights` on both Julias. The four relevant final testsets report, per run:

```text
per-entry steps on arrays: the bits of prox_block (Float32) 26356 / 26356
per-entry steps on arrays: the bits of prox_block (Float64) 19420 / 19420
per-entry steps on other arrays (Float32)                    122 / 122
per-entry steps on other arrays (Float64)                    122 / 122
Testing ProximalBase tests passed
```

The added boundary lengths (14, 15, 16, 17, and 3×5/4×4 shapes) are appropriate. `CountedSteps` correctly expects n reads below 16, 2n for ordinary steps at/above 16, and `n + min(n,256)` for an exceptional **first** step at/above 16. The scan uses `ok &= ...`, evaluating every entry of the block before deciding to stop. n=300 exercises stopping at the first block. This is a useful route regression test, not a performance test: a side-effecting `getindex` constrains optimization differently from a plain Vector.

`_L1_STEP_PASS_MIN == 16` pins an explicit maintainer requirement and is justified. The exact 256-block count pins an existing implementation detail, which is reasonable for this narrow revision but would need updating after any intentional scan redesign. The allocation helper specializes its argument arities and warms the call before measuring; both `prox!` and `value_and_prox!` allocation checks are sound for the tested homogeneous Float32/Float64 configurations. Zero allocations on these cases is a sensible regression requirement, not proof for all AbstractArrays, element types, cold calls, or compiler releases.

Gaps and brittleness:

- The broad “other arrays” tests run at **n=600**, so by themselves they do not exercise the bypass for abstract element types, SparseIterate/BlockIterate, or views. The length-mismatch/no-write example is n=100. Add small versions at n=0, 1, 15, 16, and 17, especially competing validation failures and aliases.
- `same_bits` accepts **any NaN as equal to any NaN**; it does not compare NaN sign, payload, or signaling bits. It distinguishes signed zeros via `===`, but its name and the bit-for-bit claim overstate its NaN coverage. Either narrow the claim or compare `reinterpret`ed integers for Float32/Float64. The workstation harness's zero tolerance is also numerical equality, not a bitwise comparator.
- `CountedSteps` checks reads of step entries, not repeated metadata validation or the new early `length(x)` query. That is why all candidates pass the tests while the query-order probes differ.
- The method-count assertion `length(methods(...)) >= 3` is structurally brittle and proves little about correctness; the actual ambiguity check is useful. These assertions predate the cutoff-only test additions.

My extra probes compare raw floating bits, including noncanonical positive/negative quiet NaNs and signaling NaNs. They also cover Float16 and mixed/BigFloat levels as supplemental characterization, not a new support guarantee.

## Semantics: what was actually checked

Commands (all source imports are read-only; the scripts write logs only under `round5`):

```sh
JULIA_DEPOT_PATH="$PWD/round5/depot:/path/to/.julia" \
  julia +release --startup-file=no --project=round5/env-v1.13 \
  round5/semantics.jl > round5/semantics-release.log 2>&1
# Repeat with +lts and env-v1.10.
# Supplemental probes: special-bits.jl, abstract-output.jl, offsets.jl, query-effects.jl, inference.jl.
```

Actual results:

```text
release:
SEMANTICS checks=8136 failures=0
SPECIAL BITS checks=1008 failures=0
ABSTRACT OUTPUT/MIXED checks=144 failures=0
OFFSETS and public preflight ordering checks=60 failures=0
lts:
SEMANTICS checks=8136 failures=0
SPECIAL BITS checks=1008 failures=0
ABSTRACT OUTPUT/MIXED checks=144 failures=0
OFFSETS and public preflight ordering checks=60 failures=0
```

`semantics.jl` compares **each** variant with master for n<16 and B925m for n≥16. Lengths are 0–17, 32, and 600; weighted/unweighted, ordinary/last-zero steps, `prox!`/`value_and_prox!`; 17 cases cover ordinary arrays, y===x, y===γ, y===x===γ, strided and indexed views, reshapes with differently shaped same-length γ, abstract input/step element types, sparse/sparse and sparse/dense calls, BlockIterate including active metadata, wrong output size, wrong weights, wrong step length, competing real-type/size errors, and integer-output conversion errors. The compared record includes the output's raw Float64 bits, returned value or output identity, exception name/message, and stored/active metadata where applicable. The original tests supply the much wider exceptional floating-value/step grid for both Float32 and Float64.

The normal validation order is preserved on all variants for the tested built-in array types: public one-based checks and real-entry checks occur before the inner kernel; inside it, output/input size is checked before weight length, and weight length before step length. On the large route, the structural checks precede the pass, as in B925m. B925s repeats successful inner checks; t/u do them only in the generic kernel on the short route. Output remains unchanged on structural validation failure. A γ of a different shape but the same length is accepted and read linearly on both sides of the boundary.

SparseIterate→SparseIterate keeps its specialized stored-entry loop, bypassing the changed dense method entirely. Dense output from a SparseIterate takes the dense route. BlockIterate has no separate array-step prox route and uses the dense loop; activation metadata agreed with the reference. For the tested homogeneous floating destinations, empty arrays return the output object for `prox!` and the same zero value for `value_and_prox!`, with no step-entry reads. Supplemental abstract-destination calls also agree with the reference, including errors where its value accumulator is unsupported.

**“No write before a throw” must be limited to validation errors.** Master itself can write a prefix before a later scalar conversion error. With integer output initially all 7, x=[0.3,2,2,…], λ=0.3, and unit array steps, all five trees report:

```text
InexactError y=[0, 7, 7, 7, 7, 7, 7, 7]
```

The first converted result is 0; the next is 1.7 and throws. This is an uncovered case I ran, not a new regression, and it prevents an incorrect blanket claim about atomicity.

No numerical/reference mismatch was observed in these tests. This is evidence over the stated configurations, not a proof covering arbitrary user-defined Real arithmetic, arbitrary AbstractArray side effects, or every aliasing pattern.

## Generated code and B925u's compiler structure

Commands:

```sh
# Both orders, both Julia versions; source is loaded into separate modules.
julia +release --startup-file=no --project=round5/env-v1.13 round5/codegen.jl
ORDER=reverse julia +release --startup-file=no --project=round5/env-v1.13 round5/codegen.jl
# Repeat with +lts and env-v1.10; use the local-first JULIA_DEPOT_PATH above.
```

Optimized typed IR, LLVM IR, and native code for weighted/unweighted Float32/Float64 `prox!` and `value_and_prox!` are saved under `round5/codegen/{1.13.4,1.10.13}/{forward,reverse}`. These signatures accept a Vector of runtime length: the dump is not a compile-time n=8 specialization. The runtime short branch is therefore visible. Both compiler versions produce the following structure for the tested concrete signatures:

| Tree | Short public `prox!` execution | Validation / calls |
|:--|:--|:--|
| master | Public checks fold away for concrete Vectors; direct call of generic `_prox_sep_dense!`. | One generic kernel call, one set of inner checks and the scalar loop. |
| B925m | Direct call of new NormL1 dense method. | Checks, scan, ordinary loop if accepted; otherwise another call of the generic kernel. |
| B925s | New dense method **does not inline** into `prox!`. It validates, tests length, then invokes the generic kernel. | Extra call boundary; two successful sets of inner checks on the short route. |
| B925t | New dense method **does not inline** into `prox!`. It tests length, then invokes the generic kernel. | Extra call boundary; one set of checks in the generic kernel on the short route. |
| B925u | Selector **does inline** into `prox!`; `length(x)<16` selects a direct generic-kernel call. | Same number of kernel calls as master, plus the length load/comparison/branch. The generic kernel's checks and scalar loop remain. |

`_l1_prox_entries!` disappears as an ordinary helper call: its explicit Julia `invoke(_prox_sep_dense!, Tuple{AbstractArray,Any,AbstractArray,Any}, ...)` resolves to the **generic method**, rather than redispatching to the NormL1 specialization. The remaining LLVM call is statically resolved, not a dynamic invocation or recursive dispatch. For s/t the generic call sits inside the noninlined new dense method; for u it sits directly in `prox!`. The typed IR alone spells these all as `_prox_sep_dense!`, so its name is not enough to identify which method runs; inspect the source location/signature and LLVM callee.

Representative Julia 1.13 LLVM output:

```text
# master prox!: one call
%0 = call ... @j__prox_sep_dense!_2306(...)

# B925t prox!: call of its specialized dense method
%0 = call ... @j__prox_sep_dense!_3561(...)
# That method's short branch calls generic ..._3669 (weighted signature).

# B925u prox!: branch in the caller
%0 = icmp sgt i64 %x_length, 15
br i1 %0, label %long, label %short
short: call ... @j__prox_sep_dense!_4032(...)
long:  call ... @j__l1_prox_steps!_4033(...)
```

Native code for u includes `cmp qword ptr [rdx + 16], 15; jle ...`, followed by a call of the selected kernel. Julia 1.10 has the same selection/call structure, with its own array layout and LLVM syntax. This verifies that u is close to master on a short call; it is **not identical instruction-for-instruction**.

`value_and_prox!` has its existing out-of-line `_value_and_prox_sep!` wrapper for these signatures. The same selection/call differences occur inside that wrapper; the result is subsequently passed to the existing value route. There is no changed fused accumulation or returned-value algorithm.

B925u risk assessment:

- **Inference:** all recorded warmed Float32/Float64 signatures return `Vector{T}` from `prox!` and T from `value_and_prox!`, with zero runtime allocation. Supplemental abstract-element and view/sparse-input signatures retain concrete output/value inference. Moving the scan helper out of line did not introduce an observed inference loss.
- **Ambiguity/recursion:** relevant `Test.detect_ambiguities(...; recursive=true)` results are zero for all five trees, both versions. The explicit generic `invoke` cannot select the NormL1-specialized method with that Any-function signature, so there is no recursive dispatch cycle. Both short and long routes were executed repeatedly.
- **Invalidation:** the added specialization is restricted to NormL1 plus array steps, with fresh helper names; it does not overwrite unrelated methods or mutate methods at runtime. I did **not** count compiler invalidations or audit downstream packages that define additional methods. Package loading/method additions have the usual invalidation risk; there is no evidence here of a u-specific invalidation defect.
- **Inlining stability:** `@inline`/`@noinline` are compiler hints/contracts for code organization, not a portable timing guarantee. They produced the intended structure on both tested versions. Only `_l1_prox_steps!` is explicitly marked `@noinline`; the scan and ordinary-loop helpers are not both separately annotated. The ordinary loop can inline into the out-of-line scan helper. The generic loop remained out of line for these signatures. Describe the observed call structure accurately rather than claiming every loop has a noinline annotation.
- **Cold compilation:** inlining the length selector does **not** defer compilation of the long branch. A first call with n=8 still compiles scan/ordinary specializations, giving more cold latency than master. The source precompile workload exercises scalar-step NormL1, not these array-step signatures. The following are observed ranges across both fresh codegen processes, both types, and both weight modes. They exclude source-module loading; `@timed Base.invokelatest(...)` includes compilation of the first operation. The first `value_and_prox!` call follows `prox!` with the same arguments, so its prox kernel is already compiled; that column is not a fully independent cold invocation. These are illustrative observations, not a statistically controlled latency benchmark. n=16 in that script follows n=8 with identical argument types and is consequently warm, not a second cold measurement.

| Julia | tree | cold n=8 prox! ms | first value_and_prox! ms |
|:--|:--|--:|--:|
| 1.13.4 | master | 11.55–26.48 | 41.49–56.28 |
| 1.13.4 | B925m | 36.68–47.35 | 42.44–61.51 |
| 1.13.4 | B925s | 37.03–53.80 | 42.08–56.80 |
| 1.13.4 | B925t | 36.53–49.50 | 42.30–56.77 |
| 1.13.4 | B925u | 37.81–50.63 | 42.90–64.69 |
| 1.10.13 | master | 9.99–22.50 | 38.65–49.62 |
| 1.10.13 | B925m | 30.16–44.58 | 38.64–42.56 |
| 1.10.13 | B925s | 30.22–41.25 | 38.44–42.81 |
| 1.10.13 | B925t | 30.58–49.52 | 38.68–43.64 |
| 1.10.13 | B925u | 30.43–48.58 | 41.03–44.62 |

The Julia 1.13 source loaders emit a warning about accessing a freshly defined module binding in an older world. The **supplied benchmark driver** emits the same warning from `loadtree`; it is a harness construction issue, not a B925u method-inference or recursion warning. Correct that loader with an appropriate latest-world binding read if it is retained for future Julia versions. Timing closures use concrete function handles and timed kernels do not perform that module lookup.

## Benchmark suite audit

I read `cases_steps`, `l1_data`, `thunk`, `thunkref`, `outarr`, `outboth`, `Harness.run`, the driver, and the digest. Then I inspected and exercised the captured arguments:

```sh
julia +release --startup-file=no --project=round5/env-v1.13 round5/audit-suite.jl
```

Actual output:

```text
SUITE audit passed 490 captured tuples: lengths, weights, zero positions,
independent inputs, repeated-call immutability
```

The audit covers seven selected lengths including both sides of 16 and n=1024, both types, all seven operations, and all five supplied source trees. It checks the operation's captured tuple against its case label, weights present/absent, lengths, exceptional zero position, nonaliasing of input/output/steps, and unchanged input/steps/weights after repeated operations.

I found **no wrong-input, reassigned-closure, or input-mutation defect in `cases_steps`**:

- `l1_data` creates separate x, y, w, and γ arrays. Ordinary γ entries are in [0.5,1.5), with positive finite λ and weights; they take the intended ordinary-step route.
- Exceptional arrays are **copies** of γ, with one zero at index 1 or n. At n=1 the first/last cases coincide, as expected.
- `thunk`/`thunkref` receive the arrays and function object as arguments and capture a fresh tuple; they do not close directly over reassigned loop variables. Later loop iterations cannot retarget earlier cases.
- Implementations share each case's output y; multiple operation cases of one n also share y. That is valid because every tested dense operation overwrites all entries before its output is compared. Neither a preceding exceptional case nor value evaluation writes x, γ, or weights. The shared value Ref is read and deep-copied immediately after each operation.
- `Harness.run` checks outputs before timing, deep-copies snapshots, warms/calibrates each concrete closure, measures allocations, and rotates/reverses implementation order. The noinline batch wrapper prevents hoisting. Its fixed overhead is material at a few nanoseconds; these are library-call screens including that equal wrapper overhead, not an overhead-subtracted instruction benchmark.

Limitations: the suite's `expected=nothing` compares only with master, so shared bugs would pass; it ignores signed-zero/NaN-payload bit differences; exceptional `value_and_prox!` is not a suite case; weighted first-exception fallback is not a suite case; the scan's 256-entry block boundaries are not screened densely; inputs are hot and repeatedly reused; a single layout per case/process does not characterize all cache/address placements. Printing offsets makes placement visible; it does not control or average it. None of these is a reason to invent alignment-dependent dispatch, which the maintainer prohibited. Full correctness tests and independent differential probes provide separate evidence.

The digest defects and the Julia 1.13 loader warning above should be repaired before treating this campaign tooling as an automated acceptance gate.

## Concrete changes I would request

1. Start from **B925u**, retaining only the fixed `<16` test and its small inline selector/out-of-line checked scan helper. Do not select s or t merely because their source diff is shorter; their actual short-call route is more expensive on both compilers.
2. Present the exact tiny ordinary regressions and cold-call cost to the maintainer. Obtain an explicit acceptance ruling for the reproducible >5% cases, or revise and remeasure. Under the current rule, do not propose any candidate as already cleared. Do not tune a new threshold, branch on alignment, or change the scan architecture to work around this review.
3. Change comments to say “the short route skips the scan and calls the generic kernel,” rather than claiming identical execution/observability. State the stable-array-metadata assumption if universal AbstractArray compatibility is claimed. If arbitrary structural-query side effects truly are contractual, a separate design decision is needed; all three fail that literal condition.
4. Extend the existing test table with short view/reshape/abstract-element/iterate/alias cases and competing validation errors, checking the complete y buffer after validation failures. Add the integer-output partial-write example as characterization, without promising atomicity for runtime conversion failures. Strengthen or rename `same_bits` for NaNs. Keep the justified fixed-cutoff/read-count/allocation tests.
5. Make the digest reject or quarantine `DIFF`, nonzero-allocation, missing-control, missing-process, and incomplete-round cells. Require at least two complete processes for a reproducible classification. Mark unresolved controls as unresolved rather than safe, and fix the `<16` bucket. Correct the source loader's latest-world binding read. Keep shared-buffer thunks: the audited suite does not need a speculative closure/mutation rewrite.

## Checks and boundaries

The source comparison commands were, for each v in s/t/u:

```sh
git -C /path/to/pbopt/B925$v diff b8ddeaf3 HEAD -- src/functions/norml1.jl src/functions/separable.jl
git -C /path/to/pbopt/B925$v diff 5c9a6b21 HEAD -- src/functions/norml1.jl src/functions/separable.jl test/infinite_weights.jl
```

The supplied source trees resolved to b8ddeaf3, 5c9a6b21, d59ed026, 048c151a, and 43a9468c respectively, and had clean status at inspection. No change to `separable.jl` was found in these cutoff revisions. Tests ran from independent copies of `src`, `test`, `scripts`, and `Project.toml` inside `round5/test-{s,t,u}` because `Pkg.test` may resolve/write an environment. Example actual command:

```sh
JULIA_PKG_OFFLINE=true JULIA_DEPOT_PATH="$PWD/round5/depot:/path/to/.julia" \
  julia +release --startup-file=no round5/test-s/scripts/test.jl infinite_weights
```

The copied runner activates the copied package and uses the original file-test mechanism, including `--check-bounds=yes`. All six logs end in `Testing ProximalBase tests passed`. Timing uses `--check-bounds=no` as specified by the supplied benchmark driver; semantic probes use Julia's default bound checking and the copied tests force checking.

I did **not** run the full package test inventory, test GPU/distributed arrays or offset-array performance, exhaust arbitrary user-defined Real/AbstractArray implementations, promise every overlapping-view alias pattern, measure other CPUs/operating systems, reserve the workstation, run the external authoritative campaign, inspect every downstream extension, count invalidations, or measure package load/precompile cache behavior as a controlled latency experiment. The main timing screen stops at n=1024; the larger sizes present in `STEP_SIZES` were constructed by the driver but filtered out before checking/timing. Results for such sizes are not claimed.

Evidence is preserved under [round5](round5/): scripts, saved diffs, six file-test logs, semantic/codegen logs, generated IR/assembly, provenance, complete `raw.csv`/`meta.csv` files, digest fixtures, and summary scripts. Read the raw ratios and control columns when a rounded table cell sits near a decision boundary.

## Appendix A: full timing screen

Operation labels: `prox_g` = unweighted ordinary array steps; `prox_gw` = weighted ordinary steps; `valprox_g`/`valprox_gw` = corresponding `value_and_prox!`; `prox_gx_first`/`prox_gx_last` = unweighted one-zero-step cases; `prox_gwx_last` = weighted last-zero-step case. n=16 takes the scan route. Values greater than 1 are slower than master. These ranges summarize processes, not confidence intervals.

Ratios are per-process medians of paired per-round times; ranges are across fresh processes. M is master; MC is A/A. A value above 1 is slower. Each table includes B925m as well as the three revisions.

### Julia 1.10, Float64, `prox_g`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 3.20 | 2.188–2.257 | 1.357–1.443 | 1.341–1.456 | 1.020–1.031 | 1.000–1.000 |
| 2 | 5.00 | 1.722–1.959 | 1.133–1.351 | 1.130–1.284 | 0.945–1.039 | 0.884–0.981 |
| 3 | 5.33 | 1.759–1.830 | 1.214–1.324 | 1.214–1.287 | 1.027–1.027 | 1.000–1.000 |
| 4 | 6.49 | 1.271–1.276 | 1.182–1.210 | 1.159–1.197 | 1.010–1.010 | 1.000–1.000 |
| 5 | 8.24 | 1.172–1.220 | 1.119–1.119 | 1.119–1.164 | 1.000–1.000 | 0.924–0.929 |
| 6 | 9.07 | 1.179–1.188 | 1.170–1.236 | 1.161–1.260 | 0.999–1.000 | 0.971–0.976 |
| 7 | 10.09 | 1.136–1.186 | 1.088–1.103 | 1.081–1.094 | 0.999–1.002 | 1.000–1.002 |
| 8 | 11.38 | 0.922–0.922 | 1.084–1.087 | 1.081–1.085 | 1.000–1.002 | 1.000–1.001 |
| 9 | 12.61 | 0.958–1.036 | 1.080–1.082 | 1.074–1.075 | 1.005–1.006 | 0.998–1.000 |
| 10 | 13.75 | 1.008–1.018 | 1.069–1.075 | 1.067–1.072 | 1.003–1.006 | 1.000–1.002 |
| 11 | 15.09 | 0.932–1.017 | 1.063–1.064 | 1.060–1.087 | 1.007–1.008 | 1.000–1.001 |
| 12 | 16.24 | 0.752–0.764 | 1.062–1.072 | 1.058–1.065 | 1.008–1.008 | 0.999–1.000 |
| 13 | 17.58 | 0.792–0.831 | 1.055–1.060 | 1.050–1.054 | 1.005–1.005 | 1.000–1.000 |
| 14 | 18.84 | 0.842–0.881 | 1.051–1.052 | 1.049–1.049 | 1.002–1.003 | 1.000–1.000 |
| 15 | 19.94 | 0.850–0.901 | 1.051–1.051 | 1.048–1.048 | 1.004–1.004 | 1.000–1.000 |
| 16 | 21.29 | 0.509–0.509 | 0.510–0.510 | 0.510–0.510 | 0.516–0.517 | 1.000–1.001 |
| 17 | 22.64 | 0.538–0.538 | 0.539–0.539 | 0.539–0.539 | 0.539–0.539 | 1.000–1.000 |
| 24 | 31.43 | 0.471–0.473 | 0.473–0.476 | 0.475–0.476 | 0.472–0.474 | 1.000–1.000 |
| 32 | 41.25 | 0.387–0.392 | 0.388–0.395 | 0.389–0.394 | 0.388–0.394 | 1.000–1.000 |
| 64 | 81.43 | 0.338–0.338 | 0.337–0.337 | 0.337–0.337 | 0.337–0.337 | 1.000–1.000 |
| 256 | 329.74 | 0.284–0.294 | 0.284–0.294 | 0.284–0.294 | 0.284–0.294 | 1.000–1.000 |
| 1024 | 1291.69 | 0.293–0.293 | 0.293–0.293 | 0.293–0.293 | 0.293–0.293 | 1.000–1.000 |

### Julia 1.10, Float64, `prox_gw`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 3.43 | 2.287–2.288 | 1.441–1.449 | 1.392–1.400 | 1.029–1.029 | 1.010–1.010 |
| 2 | 4.96 | 1.828–1.933 | 1.180–1.388 | 1.136–1.347 | 0.957–1.138 | 0.917–0.999 |
| 3 | 5.75 | 1.815–1.865 | 1.240–1.276 | 1.211–1.239 | 0.996–1.069 | 0.984–1.018 |
| 4 | 6.94 | 1.719–1.723 | 1.217–1.237 | 1.214–1.245 | 1.034–1.089 | 1.024–1.024 |
| 5 | 8.17 | 1.636–1.640 | 1.183–1.204 | 1.189–1.251 | 1.035–1.035 | 1.024–1.034 |
| 6 | 9.61 | 1.459–1.536 | 1.107–1.170 | 1.091–1.144 | 0.981–1.043 | 0.970–1.035 |
| 7 | 11.04 | 1.440–1.471 | 1.117–1.156 | 1.118–1.142 | 1.008–1.039 | 1.034–1.035 |
| 8 | 12.47 | 0.915–0.922 | 1.111–1.147 | 1.098–1.132 | 1.009–1.058 | 1.043–1.044 |
| 9 | 13.92 | 0.927–0.941 | 1.095–1.128 | 1.083–1.125 | 1.004–1.037 | 1.034–1.035 |
| 10 | 15.20 | 0.937–0.946 | 1.085–1.118 | 1.074–1.111 | 1.005–1.030 | 1.028–1.029 |
| 11 | 16.62 | 0.925–0.932 | 1.077–1.111 | 1.067–1.102 | 1.005–1.033 | 1.031–1.032 |
| 12 | 18.00 | 0.758–0.760 | 1.071–1.104 | 1.062–1.092 | 1.001–1.030 | 1.029–1.034 |
| 13 | 19.53 | 0.769–0.774 | 1.068–1.092 | 1.059–1.087 | 1.002–1.025 | 1.026–1.034 |
| 14 | 20.83 | 0.785–0.787 | 1.063–1.091 | 1.056–1.084 | 1.003–1.029 | 1.028–1.028 |
| 15 | 22.16 | 0.799–0.802 | 1.061–1.089 | 1.053–1.084 | 1.003–1.034 | 1.032–1.032 |
| 16 | 23.74 | 0.525–0.525 | 0.529–0.530 | 0.530–0.531 | 0.529–0.530 | 1.032–1.036 |
| 17 | 25.19 | 0.563–0.564 | 0.563–0.564 | 0.565–0.565 | 0.564–0.565 | 1.031–1.032 |
| 24 | 35.01 | 0.476–0.478 | 0.476–0.477 | 0.478–0.482 | 0.477–0.480 | 1.034–1.034 |
| 32 | 46.01 | 0.393–0.397 | 0.390–0.397 | 0.393–0.399 | 0.393–0.399 | 1.030–1.030 |
| 64 | 91.35 | 0.328–0.328 | 0.327–0.327 | 0.330–0.330 | 0.330–0.330 | 1.031–1.032 |
| 256 | 362.17 | 0.285–0.285 | 0.285–0.285 | 0.285–0.285 | 0.286–0.286 | 1.031–1.031 |
| 1024 | 1448.87 | 0.282–0.283 | 0.282–0.283 | 0.282–0.283 | 0.283–0.283 | 1.029–1.030 |

### Julia 1.10, Float64, `valprox_g`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 5.89 | 1.600–1.600 | 1.167–1.167 | 1.167–1.167 | 1.000–1.000 | 1.000–1.000 |
| 2 | 7.06 | 1.556–1.717 | 1.167–1.318 | 1.138–1.148 | 1.108–1.139 | 0.999–1.040 |
| 3 | 10.20 | 1.403–1.452 | 1.115–1.230 | 1.096–1.104 | 1.012–1.058 | 1.000–1.000 |
| 4 | 9.85 | 1.158–1.160 | 1.097–1.156 | 1.077–1.078 | 1.011–1.045 | 0.990–1.001 |
| 5 | 11.77 | 1.112–1.145 | 1.083–1.152 | 1.059–1.085 | 1.055–1.058 | 0.948–0.952 |
| 6 | 12.27 | 1.156–1.190 | 1.194–1.201 | 1.109–1.126 | 1.107–1.128 | 0.999–1.018 |
| 7 | 13.73 | 1.106–1.132 | 1.069–1.143 | 1.087–1.087 | 1.043–1.056 | 1.000–1.001 |
| 8 | 15.36 | 0.913–0.928 | 1.061–1.100 | 1.058–1.067 | 1.000–1.009 | 0.999–1.000 |
| 9 | 16.88 | 0.964–1.036 | 1.049–1.128 | 1.054–1.054 | 1.014–1.018 | 1.000–1.000 |
| 10 | 18.34 | 1.024–1.038 | 1.060–1.100 | 1.030–1.059 | 1.034–1.063 | 0.970–1.000 |
| 11 | 19.68 | 0.967–1.092 | 1.048–1.067 | 1.052–1.052 | 1.011–1.016 | 0.999–1.001 |
| 12 | 21.19 | 0.800–0.876 | 1.042–1.076 | 1.043–1.046 | 1.020–1.039 | 0.999–1.003 |
| 13 | 22.77 | 0.801–0.841 | 1.046–1.051 | 1.046–1.048 | 1.004–1.006 | 0.999–1.000 |
| 14 | 24.26 | 0.824–0.889 | 1.044–1.053 | 1.038–1.038 | 1.015–1.047 | 1.000–1.000 |
| 15 | 25.46 | 0.852–0.908 | 1.038–1.076 | 1.046–1.047 | 1.053–1.065 | 1.000–1.001 |
| 16 | 25.09 | 0.565–0.566 | 0.565–0.567 | 0.566–0.566 | 0.572–0.572 | 0.999–1.000 |
| 17 | 26.66 | 0.591–0.594 | 0.593–0.593 | 0.592–0.593 | 0.598–0.598 | 1.000–1.001 |
| 24 | 36.86 | 0.538–0.540 | 0.542–0.543 | 0.542–0.542 | 0.543–0.543 | 1.000–1.000 |
| 32 | 45.37 | 0.434–0.442 | 0.431–0.438 | 0.431–0.438 | 0.433–0.441 | 1.000–1.001 |
| 64 | 86.45 | 0.363–0.364 | 0.362–0.363 | 0.363–0.363 | 0.364–0.364 | 1.000–1.000 |
| 256 | 335.97 | 0.318–0.318 | 0.318–0.318 | 0.318–0.318 | 0.318–0.318 | 0.999–1.000 |
| 1024 | 1346.21 | 0.317–0.322 | 0.315–0.319 | 0.314–0.318 | 0.318–0.322 | 0.999–1.001 |

### Julia 1.10, Float64, `valprox_gw`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 5.89 | 1.734–1.933 | 1.250–1.267 | 1.233–1.233 | 1.000–1.018 | 1.000–1.000 |
| 2 | 7.61 | 1.522–1.777 | 1.198–1.211 | 1.112–1.184 | 1.026–1.097 | 0.951–1.016 |
| 3 | 11.97 | 1.398–1.473 | 1.132–1.134 | 1.090–1.132 | 1.011–1.015 | 0.994–1.021 |
| 4 | 10.85 | 1.452–1.513 | 1.134–1.147 | 1.124–1.133 | 1.018–1.032 | 1.021–1.021 |
| 5 | 12.61 | 1.395–1.439 | 1.119–1.126 | 1.102–1.117 | 1.021–1.023 | 1.015–1.020 |
| 6 | 14.50 | 1.343–1.372 | 1.089–1.106 | 1.081–1.101 | 1.017–1.018 | 1.012–1.013 |
| 7 | 16.38 | 1.299–1.307 | 1.100–1.115 | 1.100–1.106 | 1.002–1.032 | 1.026–1.027 |
| 8 | 18.45 | 0.937–0.943 | 1.072–1.093 | 1.069–1.093 | 1.004–1.032 | 1.025–1.029 |
| 9 | 20.70 | 0.939–0.944 | 1.056–1.068 | 1.054–1.078 | 0.994–1.022 | 1.010–1.015 |
| 10 | 22.47 | 0.947–0.974 | 1.062–1.073 | 1.044–1.070 | 1.002–1.019 | 1.014–1.017 |
| 11 | 24.48 | 0.952–0.963 | 1.056–1.070 | 1.043–1.068 | 1.006–1.020 | 1.017–1.018 |
| 12 | 26.49 | 0.832–0.846 | 1.047–1.064 | 1.039–1.060 | 1.002–1.018 | 1.016–1.017 |
| 13 | 28.61 | 0.849–0.852 | 1.044–1.064 | 1.036–1.059 | 1.005–1.023 | 1.022–1.025 |
| 14 | 30.59 | 0.867–0.876 | 1.039–1.058 | 1.037–1.054 | 1.001–1.019 | 1.016–1.017 |
| 15 | 32.60 | 0.863–0.866 | 1.037–1.056 | 1.030–1.052 | 1.000–1.022 | 1.020–1.022 |
| 16 | 34.71 | 0.681–0.685 | 0.681–0.682 | 0.687–0.688 | 0.683–0.687 | 1.019–1.023 |
| 17 | 36.71 | 0.707–0.708 | 0.705–0.706 | 0.711–0.711 | 0.709–0.711 | 1.025–1.027 |
| 24 | 50.99 | 0.642–0.645 | 0.644–0.644 | 0.645–0.647 | 0.645–0.647 | 1.020–1.021 |
| 32 | 66.99 | 0.580–0.587 | 0.580–0.586 | 0.582–0.588 | 0.582–0.590 | 1.017–1.020 |
| 64 | 134.42 | 0.542–0.543 | 0.543–0.543 | 0.543–0.545 | 0.544–0.545 | 1.019–1.022 |
| 256 | 555.63 | 0.533–0.536 | 0.533–0.536 | 0.533–0.536 | 0.533–0.536 | 1.014–1.020 |
| 1024 | 2243.64 | 0.537–0.538 | 0.537–0.537 | 0.537–0.537 | 0.537–0.538 | 1.018–1.018 |

### Julia 1.10, Float64, `prox_gx_first`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 4.49 | 1.892–2.017 | 1.267–1.342 | 1.265–1.284 | 1.018–1.082 | 1.000–1.007 |
| 2 | 5.79 | 1.747–1.856 | 1.206–1.266 | 1.208–1.231 | 1.023–1.033 | 0.946–1.005 |
| 3 | 6.66 | 1.708–1.861 | 1.209–1.295 | 1.235–1.240 | 1.007–1.092 | 1.000–1.053 |
| 4 | 7.73 | 1.659–1.803 | 1.167–1.244 | 1.192–1.268 | 1.008–1.066 | 1.008–1.040 |
| 5 | 8.81 | 1.590–1.738 | 1.132–1.205 | 1.136–1.177 | 1.007–1.025 | 1.000–1.026 |
| 6 | 9.92 | 1.563–1.732 | 1.135–1.233 | 1.159–1.207 | 1.011–1.046 | 1.000–1.082 |
| 7 | 11.18 | 1.516–1.600 | 1.105–1.112 | 1.084–1.093 | 1.004–1.006 | 0.998–0.999 |
| 8 | 12.49 | 1.506–1.510 | 1.089–1.089 | 1.079–1.079 | 1.005–1.007 | 0.999–0.999 |
| 9 | 13.74 | 1.487–1.504 | 1.079–1.080 | 1.067–1.069 | 1.005–1.005 | 0.999–1.000 |
| 10 | 14.83 | 1.443–1.548 | 1.073–1.112 | 1.071–1.108 | 1.005–1.006 | 1.002–1.005 |
| 11 | 16.18 | 1.461–1.466 | 1.067–1.069 | 1.060–1.060 | 1.005–1.006 | 0.999–1.001 |
| 12 | 17.41 | 1.447–1.461 | 1.066–1.075 | 1.058–1.060 | 1.005–1.006 | 1.000–1.001 |
| 13 | 18.73 | 1.417–1.436 | 1.059–1.061 | 1.052–1.053 | 1.004–1.005 | 1.000–1.000 |
| 14 | 19.96 | 1.431–1.431 | 1.056–1.056 | 1.050–1.050 | 1.003–1.004 | 0.999–0.999 |
| 15 | 21.08 | 1.409–1.442 | 1.054–1.054 | 1.072–1.082 | 1.004–1.005 | 1.000–1.000 |
| 16 | 22.50 | 1.252–1.252 | 1.253–1.253 | 1.253–1.253 | 1.253–1.254 | 1.000–1.001 |
| 17 | 23.85 | 1.257–1.257 | 1.257–1.257 | 1.257–1.257 | 1.257–1.259 | 0.999–1.001 |
| 24 | 32.62 | 1.250–1.251 | 1.250–1.253 | 1.252–1.252 | 1.254–1.254 | 1.000–1.000 |
| 32 | 42.57 | 1.160–1.162 | 1.158–1.163 | 1.160–1.162 | 1.163–1.164 | 0.999–0.999 |
| 64 | 82.68 | 1.117–1.117 | 1.117–1.118 | 1.118–1.118 | 1.118–1.119 | 0.998–1.000 |
| 256 | 331.04 | 1.057–1.094 | 1.057–1.095 | 1.057–1.095 | 1.058–1.095 | 0.999–0.999 |
| 1024 | 1297.27 | 1.021–1.031 | 1.024–1.032 | 1.020–1.029 | 1.030–1.034 | 0.996–0.997 |

### Julia 1.10, Float64, `prox_gx_last`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 4.49 | 1.883–2.050 | 1.262–1.338 | 1.297–1.300 | 1.015–1.068 | 1.000–1.004 |
| 2 | 5.53 | 1.810–1.953 | 1.282–1.349 | 1.298–1.402 | 1.012–1.120 | 1.000–1.047 |
| 3 | 6.64 | 1.726–1.836 | 1.225–1.268 | 1.251–1.330 | 1.010–1.055 | 0.994–1.034 |
| 4 | 7.75 | 1.646–1.765 | 1.148–1.152 | 1.132–1.134 | 1.009–1.012 | 1.000–1.038 |
| 5 | 9.03 | 1.568–1.695 | 1.109–1.178 | 1.123–1.194 | 1.000–1.022 | 0.977–1.022 |
| 6 | 10.60 | 1.482–1.581 | 1.087–1.093 | 1.111–1.128 | 0.993–1.000 | 0.933–0.944 |
| 7 | 11.15 | 1.496–1.603 | 1.103–1.104 | 1.088–1.094 | 1.001–1.024 | 0.999–1.001 |
| 8 | 12.42 | 1.476–1.499 | 1.082–1.083 | 1.079–1.081 | 0.999–1.028 | 0.999–1.000 |
| 9 | 13.66 | 1.461–1.498 | 1.081–1.081 | 1.079–1.133 | 1.002–1.004 | 1.000–1.000 |
| 10 | 14.86 | 1.429–1.487 | 1.065–1.068 | 1.064–1.064 | 1.002–1.007 | 1.000–1.000 |
| 11 | 16.19 | 1.420–1.428 | 1.060–1.062 | 1.060–1.061 | 1.002–1.002 | 0.999–1.000 |
| 12 | 17.30 | 1.408–1.450 | 1.058–1.059 | 1.056–1.057 | 1.001–1.002 | 1.000–1.000 |
| 13 | 18.60 | 1.405–1.424 | 1.053–1.060 | 1.052–1.057 | 0.999–1.004 | 0.999–1.000 |
| 14 | 19.85 | 1.408–1.424 | 1.051–1.052 | 1.049–1.050 | 1.000–1.000 | 1.000–1.000 |
| 15 | 21.09 | 1.397–1.404 | 1.050–1.051 | 1.047–1.049 | 1.004–1.108 | 1.000–1.001 |
| 16 | 22.80 | 1.231–1.257 | 1.230–1.259 | 1.232–1.260 | 1.231–1.258 | 0.976–1.000 |
| 17 | 23.73 | 1.260–1.260 | 1.260–1.260 | 1.261–1.261 | 1.262–1.264 | 1.000–1.001 |
| 24 | 32.52 | 1.245–1.246 | 1.245–1.247 | 1.247–1.247 | 1.249–1.249 | 0.999–1.000 |
| 32 | 42.36 | 1.170–1.170 | 1.170–1.173 | 1.171–1.172 | 1.172–1.173 | 0.999–0.999 |
| 64 | 82.54 | 1.120–1.121 | 1.121–1.122 | 1.121–1.122 | 1.122–1.122 | 0.998–1.000 |
| 256 | 330.90 | 1.058–1.095 | 1.058–1.096 | 1.058–1.096 | 1.059–1.096 | 0.999–1.000 |
| 1024 | 1296.20 | 1.093–1.109 | 1.093–1.104 | 1.095–1.112 | 1.095–1.105 | 1.002–1.008 |

### Julia 1.10, Float64, `prox_gwx_last`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 5.23 | 1.838–1.838 | 1.287–1.289 | 1.256–1.256 | 1.012–1.013 | 1.000–1.000 |
| 2 | 6.34 | 1.763–1.764 | 1.237–1.242 | 1.212–1.216 | 1.011–1.058 | 1.005–1.010 |
| 3 | 7.46 | 1.685–1.751 | 1.202–1.211 | 1.182–1.196 | 1.026–1.085 | 1.009–1.009 |
| 4 | 8.70 | 1.624–1.624 | 1.173–1.184 | 1.154–1.165 | 1.008–1.019 | 1.012–1.012 |
| 5 | 9.86 | 1.573–1.591 | 1.152–1.167 | 1.138–1.152 | 1.009–1.024 | 1.019–1.020 |
| 6 | 11.11 | 1.538–1.564 | 1.129–1.155 | 1.120–1.150 | 1.018–1.028 | 1.023–1.025 |
| 7 | 12.52 | 1.479–1.513 | 1.110–1.143 | 1.103–1.134 | 1.008–1.033 | 1.031–1.033 |
| 8 | 14.02 | 1.453–1.493 | 1.100–1.129 | 1.089–1.115 | 1.004–1.037 | 1.030–1.030 |
| 9 | 15.43 | 1.429–1.461 | 1.095–1.123 | 1.084–1.119 | 1.009–1.038 | 1.027–1.034 |
| 10 | 16.74 | 1.386–1.457 | 1.074–1.111 | 1.065–1.102 | 1.002–1.031 | 1.025–1.027 |
| 11 | 18.12 | 1.390–1.449 | 1.072–1.107 | 1.067–1.096 | 1.000–1.032 | 1.028–1.028 |
| 12 | 19.44 | 1.384–1.439 | 1.075–1.099 | 1.064–1.093 | 1.008–1.029 | 1.027–1.027 |
| 13 | 20.82 | 1.378–1.415 | 1.067–1.101 | 1.063–1.097 | 1.006–1.036 | 1.032–1.036 |
| 14 | 22.22 | 1.383–1.403 | 1.067–1.089 | 1.057–1.083 | 1.004–1.028 | 1.024–1.026 |
| 15 | 23.69 | 1.357–1.395 | 1.061–1.083 | 1.056–1.080 | 1.003–1.027 | 1.026–1.027 |
| 16 | 25.20 | 1.235–1.277 | 1.240–1.276 | 1.240–1.279 | 1.239–1.279 | 1.030–1.030 |
| 17 | 26.66 | 1.247–1.273 | 1.246–1.274 | 1.249–1.273 | 1.248–1.274 | 1.030–1.031 |
| 24 | 36.45 | 1.221–1.269 | 1.223–1.268 | 1.223–1.268 | 1.224–1.269 | 1.031–1.032 |
| 32 | 47.59 | 1.163–1.196 | 1.163–1.198 | 1.162–1.197 | 1.163–1.196 | 1.025–1.027 |
| 64 | 92.76 | 1.116–1.141 | 1.116–1.142 | 1.116–1.141 | 1.116–1.142 | 1.028–1.028 |
| 256 | 363.69 | 1.087–1.118 | 1.088–1.118 | 1.088–1.118 | 1.088–1.119 | 1.030–1.030 |
| 1024 | 1446.83 | 1.088–1.122 | 1.088–1.119 | 1.088–1.118 | 1.088–1.116 | 1.033–1.035 |

### Julia 1.10, Float32, `prox_g`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 3.17 | 2.165–2.181 | 1.359–1.459 | 1.318–1.361 | 1.050–1.051 | 0.999–1.000 |
| 2 | 4.67 | 1.896–1.948 | 1.260–1.366 | 1.213–1.286 | 1.082–1.141 | 0.947–1.046 |
| 3 | 5.30 | 1.722–1.795 | 1.216–1.333 | 1.191–1.259 | 1.074–1.074 | 1.000–1.000 |
| 4 | 6.58 | 1.680–1.883 | 1.191–1.265 | 1.159–1.206 | 1.059–1.101 | 0.968–1.039 |
| 5 | 7.63 | 1.643–1.649 | 1.209–1.210 | 1.154–1.168 | 1.030–1.036 | 1.000–1.010 |
| 6 | 8.80 | 1.351–1.496 | 1.126–1.183 | 1.138–1.144 | 0.988–1.030 | 0.962–1.016 |
| 7 | 10.03 | 1.342–1.540 | 1.087–1.174 | 1.079–1.135 | 1.037–1.038 | 0.999–1.018 |
| 8 | 11.30 | 0.818–0.832 | 1.081–1.083 | 1.078–1.083 | 0.999–0.999 | 0.999–1.000 |
| 9 | 12.56 | 0.831–0.967 | 1.078–1.082 | 1.076–1.079 | 1.003–1.005 | 1.000–1.000 |
| 10 | 13.71 | 0.857–0.952 | 1.069–1.074 | 1.068–1.069 | 1.005–1.008 | 1.000–1.000 |
| 11 | 15.07 | 0.849–0.958 | 1.067–1.067 | 1.061–1.062 | 1.002–1.006 | 1.000–1.000 |
| 12 | 16.22 | 0.851–0.908 | 1.062–1.062 | 1.058–1.058 | 1.005–1.005 | 0.999–1.001 |
| 13 | 17.47 | 0.866–0.910 | 1.057–1.058 | 1.053–1.055 | 1.004–1.005 | 1.000–1.000 |
| 14 | 18.68 | 0.863–0.956 | 1.054–1.055 | 1.051–1.052 | 1.005–1.005 | 0.999–1.000 |
| 15 | 20.04 | 0.878–0.938 | 1.049–1.051 | 1.046–1.048 | 1.004–1.005 | 1.001–1.001 |
| 16 | 21.33 | 0.591–0.634 | 0.594–0.628 | 0.583–0.592 | 0.598–0.607 | 0.999–1.001 |
| 17 | 22.56 | 0.607–0.656 | 0.614–0.665 | 0.612–0.620 | 0.621–0.629 | 1.000–1.000 |
| 24 | 31.39 | 0.523–0.537 | 0.524–0.536 | 0.517–0.528 | 0.526–0.529 | 1.000–1.000 |
| 32 | 41.35 | 0.263–0.263 | 0.266–0.266 | 0.266–0.267 | 0.266–0.266 | 1.000–1.000 |
| 64 | 81.32 | 0.200–0.200 | 0.200–0.201 | 0.199–0.200 | 0.201–0.201 | 0.999–1.000 |
| 256 | 323.28 | 0.157–0.157 | 0.156–0.157 | 0.156–0.157 | 0.157–0.157 | 1.000–1.000 |
| 1024 | 1289.53 | 0.154–0.154 | 0.154–0.154 | 0.154–0.154 | 0.153–0.154 | 1.002–1.002 |

### Julia 1.10, Float32, `prox_gw`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 3.34 | 2.416–2.471 | 1.475–1.477 | 1.406–1.457 | 1.020–1.059 | 1.000–1.000 |
| 2 | 4.62 | 1.993–2.092 | 1.261–1.447 | 1.317–1.429 | 1.005–1.196 | 1.054–1.072 |
| 3 | 5.59 | 1.929–1.942 | 1.280–1.298 | 1.239–1.333 | 1.012–1.087 | 1.000–1.000 |
| 4 | 6.83 | 1.768–1.787 | 1.226–1.293 | 1.191–1.341 | 1.011–1.120 | 0.999–1.000 |
| 5 | 8.08 | 1.656–1.678 | 1.189–1.358 | 1.156–1.239 | 1.010–1.116 | 1.000–1.001 |
| 6 | 9.47 | 1.555–1.571 | 1.144–1.202 | 1.222–1.226 | 1.143–1.183 | 0.997–1.002 |
| 7 | 10.98 | 1.444–1.457 | 1.117–1.181 | 1.103–1.198 | 1.004–1.075 | 0.998–1.001 |
| 8 | 12.33 | 0.859–0.869 | 1.110–1.112 | 1.093–1.097 | 1.002–1.004 | 0.998–1.000 |
| 9 | 13.83 | 0.858–0.898 | 1.095–1.098 | 1.082–1.106 | 1.000–1.007 | 1.000–1.001 |
| 10 | 15.13 | 0.894–0.900 | 1.089–1.091 | 1.072–1.077 | 1.002–1.002 | 1.000–1.001 |
| 11 | 16.62 | 0.876–0.883 | 1.078–1.081 | 1.063–1.070 | 1.001–1.004 | 0.997–1.000 |
| 12 | 17.96 | 0.895–0.905 | 1.073–1.074 | 1.060–1.063 | 1.002–1.004 | 0.999–1.002 |
| 13 | 19.33 | 0.911–0.917 | 1.068–1.068 | 1.058–1.059 | 1.002–1.012 | 1.001–1.001 |
| 14 | 20.68 | 0.929–0.970 | 1.065–1.083 | 1.056–1.064 | 1.003–1.021 | 0.999–1.000 |
| 15 | 22.30 | 0.925–0.945 | 1.060–1.060 | 1.051–1.058 | 1.001–1.002 | 1.000–1.000 |
| 16 | 23.65 | 0.584–0.591 | 0.596–0.606 | 0.593–0.597 | 0.605–0.611 | 1.000–1.000 |
| 17 | 25.07 | 0.623–0.631 | 0.626–0.632 | 0.625–0.629 | 0.642–0.652 | 0.999–1.001 |
| 24 | 34.70 | 0.503–0.511 | 0.508–0.510 | 0.503–0.510 | 0.515–0.526 | 1.000–1.000 |
| 32 | 46.25 | 0.272–0.272 | 0.272–0.272 | 0.272–0.272 | 0.272–0.272 | 1.000–1.000 |
| 64 | 90.51 | 0.200–0.202 | 0.200–0.202 | 0.201–0.202 | 0.201–0.202 | 1.000–1.000 |
| 256 | 362.05 | 0.153–0.153 | 0.153–0.153 | 0.153–0.153 | 0.153–0.153 | 1.000–1.000 |
| 1024 | 1447.60 | 0.149–0.149 | 0.149–0.149 | 0.149–0.149 | 0.150–0.150 | 1.003–1.003 |

### Julia 1.10, Float32, `valprox_g`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 5.89 | 1.633–1.633 | 1.200–1.200 | 1.167–1.167 | 1.000–1.000 | 1.000–1.000 |
| 2 | 7.06 | 1.626–1.656 | 1.169–1.250 | 1.152–1.222 | 1.122–1.139 | 0.986–1.014 |
| 3 | 10.22 | 1.410–1.427 | 1.096–1.167 | 1.115–1.116 | 0.994–1.057 | 0.931–1.017 |
| 4 | 9.82 | 1.518–1.610 | 1.142–1.179 | 1.148–1.157 | 1.081–1.117 | 0.966–1.028 |
| 5 | 11.15 | 1.419–1.479 | 1.128–1.172 | 1.129–1.169 | 1.019–1.071 | 0.949–1.056 |
| 6 | 12.37 | 1.288–1.359 | 1.102–1.132 | 1.097–1.146 | 1.062–1.082 | 0.989–1.015 |
| 7 | 13.70 | 1.330–1.400 | 1.097–1.136 | 1.099–1.140 | 1.096–1.102 | 0.998–1.002 |
| 8 | 15.28 | 0.889–0.894 | 1.060–1.067 | 1.061–1.063 | 1.018–1.021 | 1.000–1.000 |
| 9 | 16.80 | 0.950–0.973 | 1.064–1.087 | 1.065–1.068 | 1.025–1.072 | 1.000–1.000 |
| 10 | 18.29 | 0.936–0.965 | 1.011–1.060 | 1.021–1.100 | 1.039–1.042 | 0.970–1.000 |
| 11 | 19.82 | 0.954–0.962 | 1.052–1.053 | 1.048–1.059 | 1.023–1.037 | 1.000–1.000 |
| 12 | 21.28 | 0.933–0.933 | 1.036–1.039 | 1.037–1.046 | 1.024–1.033 | 0.999–1.000 |
| 13 | 22.69 | 0.941–0.954 | 1.050–1.050 | 1.046–1.047 | 1.006–1.009 | 1.000–1.000 |
| 14 | 24.21 | 0.983–0.995 | 1.050–1.075 | 1.046–1.100 | 1.036–1.045 | 0.999–1.001 |
| 15 | 25.84 | 0.947–0.998 | 1.033–1.061 | 1.030–1.075 | 1.022–1.026 | 0.997–1.000 |
| 16 | 27.25 | 0.717–0.748 | 0.723–0.796 | 0.721–0.740 | 0.738–0.747 | 0.999–1.000 |
| 17 | 28.73 | 0.722–0.745 | 0.732–0.824 | 0.735–0.751 | 0.763–0.774 | 1.000–1.000 |
| 24 | 41.29 | 0.648–0.662 | 0.648–0.682 | 0.646–0.650 | 0.655–0.658 | 0.999–1.003 |
| 32 | 45.19 | 0.315–0.316 | 0.318–0.318 | 0.319–0.319 | 0.322–0.322 | 1.000–1.000 |
| 64 | 85.88 | 0.244–0.246 | 0.244–0.245 | 0.243–0.246 | 0.242–0.245 | 1.000–1.000 |
| 256 | 329.95 | 0.174–0.174 | 0.174–0.174 | 0.174–0.174 | 0.174–0.174 | 1.000–1.000 |
| 1024 | 1316.22 | 0.168–0.169 | 0.168–0.168 | 0.168–0.168 | 0.168–0.168 | 1.000–1.005 |

### Julia 1.10, Float32, `valprox_gw`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 5.85 | 1.767–1.830 | 1.267–1.272 | 1.211–1.236 | 1.005–1.020 | 0.984–1.016 |
| 2 | 7.67 | 1.593–1.637 | 1.138–1.224 | 1.160–1.221 | 1.028–1.092 | 0.950–1.053 |
| 3 | 11.78 | 1.438–1.443 | 1.134–1.182 | 1.113–1.127 | 1.008–1.075 | 1.000–1.000 |
| 4 | 10.75 | 1.479–1.503 | 1.149–1.151 | 1.115–1.156 | 1.007–1.060 | 0.999–1.001 |
| 5 | 12.53 | 1.412–1.423 | 1.123–1.128 | 1.102–1.142 | 1.001–1.016 | 0.999–1.000 |
| 6 | 14.34 | 1.368–1.370 | 1.098–1.109 | 1.091–1.099 | 1.004–1.017 | 0.999–1.015 |
| 7 | 16.35 | 1.304–1.324 | 1.086–1.087 | 1.075–1.079 | 1.007–1.045 | 1.000–1.001 |
| 8 | 18.40 | 0.906–0.916 | 1.071–1.074 | 1.063–1.066 | 0.997–1.000 | 1.000–1.001 |
| 9 | 20.56 | 0.910–0.939 | 1.067–1.069 | 1.055–1.059 | 1.006–1.009 | 1.000–1.001 |
| 10 | 22.34 | 0.933–0.954 | 1.063–1.064 | 1.051–1.052 | 1.004–1.005 | 0.999–1.001 |
| 11 | 24.40 | 0.930–0.959 | 1.061–1.061 | 1.045–1.045 | 1.007–1.011 | 1.001–1.001 |
| 12 | 26.44 | 0.931–0.966 | 1.053–1.054 | 1.041–1.044 | 1.005–1.006 | 0.998–1.002 |
| 13 | 28.55 | 0.945–0.966 | 1.041–1.042 | 1.035–1.035 | 0.998–0.998 | 1.000–1.001 |
| 14 | 30.34 | 0.954–0.958 | 1.047–1.049 | 1.040–1.043 | 1.004–1.005 | 1.000–1.002 |
| 15 | 32.88 | 0.937–0.958 | 1.037–1.038 | 1.023–1.025 | 0.997–0.999 | 0.999–1.001 |
| 16 | 34.88 | 0.716–0.741 | 0.736–0.750 | 0.716–0.750 | 0.718–0.748 | 1.000–1.001 |
| 17 | 36.76 | 0.743–0.763 | 0.742–0.766 | 0.746–0.766 | 0.750–0.770 | 0.999–1.001 |
| 24 | 50.53 | 0.667–0.668 | 0.664–0.665 | 0.665–0.666 | 0.670–0.672 | 1.000–1.001 |
| 32 | 67.19 | 0.494–0.495 | 0.496–0.497 | 0.495–0.495 | 0.495–0.496 | 1.000–1.001 |
| 64 | 133.72 | 0.458–0.459 | 0.460–0.462 | 0.460–0.460 | 0.490–0.492 | 1.000–1.003 |
| 256 | 554.85 | 0.449–0.449 | 0.449–0.449 | 0.449–0.449 | 0.449–0.449 | 1.000–1.007 |
| 1024 | 2244.82 | 0.450–0.451 | 0.451–0.452 | 0.450–0.451 | 0.451–0.451 | 0.998–1.003 |

### Julia 1.10, Float32, `prox_gx_first`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 4.15 | 1.936–1.937 | 1.275–1.448 | 1.264–1.322 | 1.087–1.092 | 1.000–1.000 |
| 2 | 5.38 | 1.768–1.879 | 1.220–1.315 | 1.236–1.276 | 1.076–1.094 | 0.995–1.034 |
| 3 | 6.41 | 1.694–1.741 | 1.231–1.273 | 1.211–1.226 | 1.057–1.119 | 0.946–1.057 |
| 4 | 7.66 | 1.634–1.692 | 1.135–1.222 | 1.138–1.177 | 1.044–1.054 | 0.951–1.050 |
| 5 | 8.63 | 1.621–1.657 | 1.175–1.200 | 1.102–1.156 | 1.044–1.062 | 0.973–1.022 |
| 6 | 9.81 | 1.555–1.561 | 1.111–1.167 | 1.130–1.143 | 1.037–1.099 | 0.916–1.041 |
| 7 | 10.84 | 1.546–1.571 | 1.099–1.160 | 1.125–1.126 | 1.035–1.089 | 0.984–1.017 |
| 8 | 12.13 | 1.494–1.497 | 1.092–1.093 | 1.082–1.084 | 1.005–1.005 | 1.000–1.001 |
| 9 | 13.37 | 1.472–1.475 | 1.084–1.088 | 1.074–1.078 | 1.003–1.004 | 0.999–0.999 |
| 10 | 14.71 | 1.460–1.500 | 1.076–1.118 | 1.066–1.084 | 1.003–1.036 | 0.992–1.015 |
| 11 | 15.87 | 1.460–1.468 | 1.070–1.072 | 1.061–1.063 | 1.003–1.005 | 0.999–1.000 |
| 12 | 17.11 | 1.440–1.442 | 1.066–1.066 | 1.057–1.058 | 1.004–1.004 | 1.000–1.001 |
| 13 | 18.43 | 1.438–1.451 | 1.062–1.062 | 1.054–1.055 | 1.003–1.004 | 1.000–1.001 |
| 14 | 19.63 | 1.424–1.424 | 1.058–1.059 | 1.050–1.051 | 1.003–1.004 | 0.999–1.001 |
| 15 | 20.90 | 1.409–1.425 | 1.055–1.058 | 1.048–1.051 | 1.001–1.003 | 1.000–1.001 |
| 16 | 22.24 | 1.385–1.406 | 1.404–1.413 | 1.392–1.411 | 1.428–1.430 | 1.000–1.001 |
| 17 | 23.46 | 1.388–1.400 | 1.392–1.392 | 1.382–1.404 | 1.400–1.400 | 0.999–1.000 |
| 24 | 32.19 | 1.364–1.373 | 1.368–1.368 | 1.364–1.371 | 1.375–1.376 | 1.000–1.000 |
| 32 | 42.24 | 1.136–1.136 | 1.137–1.138 | 1.137–1.139 | 1.138–1.139 | 1.000–1.000 |
| 64 | 82.13 | 1.088–1.089 | 1.088–1.088 | 1.088–1.088 | 1.088–1.089 | 0.999–1.001 |
| 256 | 324.20 | 1.053–1.053 | 1.053–1.053 | 1.053–1.053 | 1.053–1.053 | 1.000–1.000 |
| 1024 | 1299.60 | 1.012–1.013 | 1.013–1.015 | 1.014–1.014 | 1.012–1.013 | 1.000–1.006 |

### Julia 1.10, Float32, `prox_gx_last`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 4.16 | 1.934–1.937 | 1.276–1.438 | 1.275–1.323 | 1.078–1.085 | 1.000–1.000 |
| 2 | 5.39 | 1.790–1.796 | 1.208–1.312 | 1.226–1.245 | 1.070–1.072 | 0.985–0.995 |
| 3 | 6.47 | 1.748–1.791 | 1.188–1.289 | 1.187–1.252 | 1.062–1.115 | 1.005–1.018 |
| 4 | 7.49 | 1.685–1.764 | 1.150–1.232 | 1.103–1.180 | 1.036–1.049 | 0.982–1.020 |
| 5 | 9.01 | 1.603–1.669 | 1.075–1.204 | 1.062–1.159 | 0.994–1.045 | 0.964–1.047 |
| 6 | 10.08 | 1.561–1.583 | 1.139–1.171 | 1.085–1.195 | 1.001–1.039 | 0.967–1.017 |
| 7 | 10.88 | 1.530–1.561 | 1.090–1.127 | 1.084–1.104 | 1.007–1.066 | 0.983–1.030 |
| 8 | 12.07 | 1.470–1.471 | 1.081–1.085 | 1.078–1.080 | 0.998–1.002 | 0.999–1.000 |
| 9 | 13.31 | 1.449–1.473 | 1.080–1.085 | 1.075–1.079 | 0.999–1.001 | 0.999–1.000 |
| 10 | 14.53 | 1.439–1.469 | 1.068–1.081 | 1.066–1.067 | 1.000–1.000 | 1.000–1.000 |
| 11 | 15.87 | 1.408–1.413 | 1.061–1.062 | 1.060–1.061 | 0.999–1.001 | 0.999–1.000 |
| 12 | 17.06 | 1.415–1.427 | 1.059–1.060 | 1.055–1.056 | 1.000–1.001 | 1.000–1.000 |
| 13 | 18.31 | 1.407–1.412 | 1.057–1.058 | 1.054–1.055 | 1.000–1.002 | 1.000–1.002 |
| 14 | 19.57 | 1.402–1.408 | 1.054–1.057 | 1.052–1.052 | 1.000–1.002 | 1.000–1.001 |
| 15 | 20.85 | 1.393–1.411 | 1.050–1.052 | 1.049–1.050 | 0.999–1.002 | 0.999–1.000 |
| 16 | 22.16 | 1.373–1.390 | 1.387–1.387 | 1.373–1.384 | 1.389–1.393 | 1.000–1.000 |
| 17 | 23.45 | 1.378–1.402 | 1.400–1.402 | 1.371–1.391 | 1.391–1.392 | 0.999–1.001 |
| 24 | 32.15 | 1.356–1.357 | 1.361–1.362 | 1.356–1.363 | 1.361–1.365 | 0.999–1.001 |
| 32 | 42.14 | 1.132–1.132 | 1.133–1.133 | 1.132–1.133 | 1.133–1.134 | 1.000–1.000 |
| 64 | 82.06 | 1.087–1.088 | 1.087–1.087 | 1.087–1.087 | 1.087–1.088 | 0.999–1.001 |
| 256 | 324.24 | 1.052–1.052 | 1.052–1.052 | 1.052–1.052 | 1.052–1.052 | 1.000–1.001 |
| 1024 | 1289.57 | 1.052–1.052 | 1.052–1.056 | 1.052–1.053 | 1.054–1.057 | 1.000–1.000 |

### Julia 1.10, Float32, `prox_gwx_last`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 4.74 | 1.856–1.904 | 1.297–1.332 | 1.272–1.283 | 1.031–1.042 | 0.975–1.000 |
| 2 | 5.89 | 1.803–1.844 | 1.264–1.271 | 1.280–1.290 | 1.039–1.041 | 0.978–1.089 |
| 3 | 7.29 | 1.590–1.755 | 1.113–1.264 | 1.200–1.267 | 0.950–1.099 | 0.910–1.011 |
| 4 | 8.19 | 1.654–1.725 | 1.192–1.264 | 1.164–1.295 | 1.008–1.102 | 1.000–1.000 |
| 5 | 9.39 | 1.591–1.642 | 1.167–1.175 | 1.144–1.211 | 1.044–1.060 | 1.000–1.000 |
| 6 | 10.99 | 1.664–1.713 | 1.134–1.264 | 1.087–1.248 | 0.996–1.092 | 0.965–0.993 |
| 7 | 12.02 | 1.498–1.582 | 1.126–1.158 | 1.104–1.194 | 1.005–1.057 | 0.998–1.002 |
| 8 | 13.50 | 1.460–1.465 | 1.103–1.107 | 1.090–1.091 | 1.002–1.002 | 1.000–1.000 |
| 9 | 14.91 | 1.439–1.441 | 1.093–1.096 | 1.083–1.083 | 0.998–1.002 | 0.999–1.000 |
| 10 | 16.23 | 1.428–1.429 | 1.086–1.090 | 1.072–1.077 | 1.005–1.006 | 0.999–1.003 |
| 11 | 17.72 | 1.401–1.408 | 1.077–1.079 | 1.067–1.068 | 1.002–1.002 | 1.000–1.000 |
| 12 | 19.04 | 1.379–1.391 | 1.074–1.075 | 1.064–1.065 | 1.002–1.002 | 0.998–1.001 |
| 13 | 20.39 | 1.376–1.393 | 1.066–1.069 | 1.058–1.061 | 1.000–1.003 | 0.997–1.000 |
| 14 | 21.88 | 1.374–1.409 | 1.063–1.067 | 1.054–1.056 | 1.001–1.002 | 0.999–1.000 |
| 15 | 23.40 | 1.384–1.386 | 1.058–1.061 | 1.052–1.058 | 1.002–1.006 | 1.000–1.000 |
| 16 | 24.72 | 1.365–1.376 | 1.355–1.380 | 1.366–1.385 | 1.372–1.387 | 0.998–1.000 |
| 17 | 26.20 | 1.352–1.354 | 1.355–1.360 | 1.350–1.360 | 1.350–1.359 | 1.000–1.001 |
| 24 | 35.78 | 1.323–1.343 | 1.329–1.335 | 1.326–1.333 | 1.328–1.334 | 1.000–1.000 |
| 32 | 47.35 | 1.125–1.126 | 1.126–1.132 | 1.127–1.127 | 1.126–1.128 | 1.000–1.000 |
| 64 | 91.61 | 1.084–1.085 | 1.080–1.085 | 1.084–1.085 | 1.086–1.087 | 1.000–1.000 |
| 256 | 363.36 | 1.046–1.048 | 1.047–1.048 | 1.047–1.047 | 1.047–1.048 | 1.000–1.000 |
| 1024 | 1446.99 | 1.047–1.064 | 1.047–1.058 | 1.052–1.052 | 1.047–1.053 | 1.000–1.004 |

### Julia 1.13, Float64, `prox_g`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 2.94 | 2.133–2.133 | 1.400–1.400 | 1.377–1.380 | 1.072–1.076 | 1.000–1.000 |
| 2 | 3.70 | 1.887–1.970 | 1.275–1.327 | 1.243–1.274 | 1.045–1.088 | 0.960–1.000 |
| 3 | 4.45 | 1.852–1.856 | 1.279–1.279 | 1.235–1.258 | 1.057–1.059 | 1.000–1.000 |
| 4 | 5.20 | 1.473–1.476 | 1.207–1.210 | 1.206–1.207 | 1.017–1.019 | 1.000–1.000 |
| 5 | 5.98 | 1.478–1.478 | 1.186–1.188 | 1.181–1.181 | 1.016–1.017 | 0.999–0.999 |
| 6 | 6.73 | 1.457–1.461 | 1.166–1.166 | 1.137–1.145 | 1.017–1.017 | 0.998–1.000 |
| 7 | 7.50 | 1.446–1.446 | 1.153–1.153 | 1.132–1.134 | 1.015–1.015 | 0.998–0.999 |
| 8 | 8.28 | 1.045–1.046 | 1.140–1.140 | 1.116–1.117 | 1.012–1.014 | 1.000–1.000 |
| 9 | 9.10 | 1.078–1.078 | 1.124–1.124 | 1.105–1.105 | 1.011–1.012 | 0.998–1.000 |
| 10 | 9.97 | 1.082–1.083 | 1.097–1.099 | 1.092–1.093 | 1.009–1.011 | 0.999–1.000 |
| 11 | 10.85 | 1.091–1.092 | 1.085–1.085 | 1.086–1.087 | 1.010–1.010 | 1.000–1.000 |
| 12 | 11.75 | 0.827–0.827 | 1.075–1.076 | 1.073–1.073 | 1.007–1.007 | 1.000–1.001 |
| 13 | 12.64 | 0.854–0.856 | 1.079–1.080 | 1.075–1.077 | 1.007–1.007 | 0.999–1.000 |
| 14 | 13.56 | 0.870–0.878 | 1.074–1.074 | 1.061–1.061 | 1.005–1.005 | 0.999–0.999 |
| 15 | 14.47 | 0.886–0.889 | 1.073–1.074 | 1.062–1.063 | 1.006–1.007 | 1.000–1.001 |
| 16 | 15.37 | 0.665–0.666 | 0.664–0.677 | 0.677–0.681 | 0.676–0.677 | 0.999–1.000 |
| 17 | 16.25 | 0.713–0.714 | 0.724–0.725 | 0.713–0.714 | 0.724–0.725 | 0.999–0.999 |
| 24 | 22.56 | 0.567–0.567 | 0.574–0.575 | 0.565–0.567 | 0.575–0.576 | 1.000–1.000 |
| 32 | 29.74 | 0.451–0.452 | 0.456–0.462 | 0.456–0.456 | 0.456–0.457 | 1.000–1.000 |
| 64 | 61.25 | 0.374–0.406 | 0.374–0.402 | 0.375–0.407 | 0.376–0.408 | 0.996–1.082 |
| 256 | 236.06 | 0.332–0.333 | 0.331–0.333 | 0.331–0.332 | 0.332–0.334 | 1.002–1.007 |
| 1024 | 927.38 | 0.327–0.329 | 0.325–0.329 | 0.326–0.327 | 0.328–0.329 | 1.001–1.006 |

### Julia 1.13, Float64, `prox_gw`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 3.14 | 2.313–2.313 | 1.437–1.438 | 1.312–1.313 | 1.032–1.061 | 1.000–1.000 |
| 2 | 4.01 | 2.051–2.106 | 1.286–1.323 | 1.268–1.274 | 1.000–1.051 | 0.976–1.001 |
| 3 | 4.78 | 1.974–1.976 | 1.274–1.274 | 1.233–1.233 | 1.021–1.027 | 1.000–1.000 |
| 4 | 5.59 | 1.513–1.518 | 1.228–1.228 | 1.193–1.193 | 1.017–1.018 | 1.000–1.000 |
| 5 | 6.42 | 1.561–1.571 | 1.193–1.193 | 1.162–1.162 | 1.015–1.015 | 1.000–1.000 |
| 6 | 7.26 | 1.524–1.525 | 1.163–1.164 | 1.140–1.141 | 1.012–1.012 | 1.000–1.000 |
| 7 | 8.16 | 1.476–1.476 | 1.157–1.159 | 1.116–1.117 | 1.007–1.008 | 0.998–0.999 |
| 8 | 9.15 | 1.030–1.030 | 1.122–1.122 | 1.111–1.112 | 1.007–1.009 | 1.000–1.000 |
| 9 | 10.24 | 1.079–1.081 | 1.098–1.102 | 1.079–1.082 | 1.008–1.008 | 0.999–1.000 |
| 10 | 11.26 | 1.063–1.073 | 1.092–1.096 | 1.073–1.074 | 1.003–1.005 | 1.000–1.000 |
| 11 | 12.32 | 1.051–1.066 | 1.081–1.089 | 1.068–1.071 | 1.001–1.005 | 0.999–1.001 |
| 12 | 13.33 | 0.781–0.784 | 1.080–1.085 | 1.064–1.067 | 1.004–1.005 | 0.999–1.000 |
| 13 | 14.34 | 0.835–0.839 | 1.073–1.075 | 1.056–1.058 | 1.005–1.005 | 1.000–1.001 |
| 14 | 15.30 | 0.848–0.855 | 1.070–1.071 | 1.056–1.056 | 1.003–1.005 | 1.000–1.000 |
| 15 | 16.35 | 0.860–0.862 | 1.065–1.066 | 1.051–1.051 | 1.002–1.003 | 1.000–1.000 |
| 16 | 17.34 | 0.647–0.652 | 0.663–0.665 | 0.660–0.662 | 0.656–0.656 | 1.000–1.000 |
| 17 | 18.39 | 0.713–0.713 | 0.720–0.721 | 0.728–0.728 | 0.720–0.721 | 1.000–1.000 |
| 24 | 25.60 | 0.548–0.550 | 0.554–0.555 | 0.551–0.562 | 0.555–0.557 | 1.000–1.000 |
| 32 | 33.68 | 0.440–0.440 | 0.443–0.444 | 0.444–0.449 | 0.444–0.444 | 1.000–1.000 |
| 64 | 66.32 | 0.371–0.372 | 0.376–0.377 | 0.371–0.372 | 0.375–0.375 | 1.000–1.001 |
| 256 | 268.21 | 0.316–0.316 | 0.317–0.318 | 0.316–0.317 | 0.317–0.317 | 1.000–1.000 |
| 1024 | 1056.49 | 0.313–0.313 | 0.313–0.313 | 0.313–0.313 | 0.313–0.313 | 0.999–1.000 |

### Julia 1.13, Float64, `valprox_g`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 5.51 | 1.603–1.605 | 1.210–1.212 | 1.211–1.213 | 1.032–1.034 | 1.000–1.000 |
| 2 | 6.48 | 1.572–1.575 | 1.177–1.181 | 1.147–1.151 | 1.026–1.030 | 0.997–1.000 |
| 3 | 7.46 | 1.530–1.537 | 1.158–1.158 | 1.131–1.132 | 1.000–1.032 | 0.999–1.000 |
| 4 | 12.16 | 0.903–0.907 | 1.048–1.050 | 1.052–1.056 | 1.002–1.004 | 1.002–1.003 |
| 5 | 12.57 | 0.983–0.985 | 1.042–1.044 | 1.039–1.042 | 0.997–1.000 | 0.998–1.001 |
| 6 | 12.92 | 1.033–1.035 | 1.053–1.054 | 1.054–1.055 | 1.000–1.001 | 0.999–1.001 |
| 7 | 13.52 | 1.075–1.076 | 1.058–1.058 | 1.044–1.045 | 1.002–1.004 | 1.000–1.001 |
| 8 | 14.38 | 0.835–0.837 | 1.049–1.050 | 1.048–1.050 | 0.999–1.000 | 1.000–1.000 |
| 9 | 14.99 | 0.893–0.893 | 1.044–1.044 | 1.041–1.043 | 1.000–1.000 | 1.000–1.000 |
| 10 | 15.42 | 0.942–0.944 | 1.045–1.045 | 1.043–1.044 | 0.998–0.999 | 0.999–1.001 |
| 11 | 15.90 | 0.988–0.992 | 1.042–1.043 | 1.030–1.042 | 0.993–0.999 | 1.000–1.000 |
| 12 | 17.09 | 0.771–0.773 | 1.032–1.033 | 1.033–1.033 | 0.996–0.997 | 0.999–1.001 |
| 13 | 17.48 | 0.832–0.833 | 1.035–1.035 | 1.036–1.036 | 1.002–1.002 | 1.000–1.000 |
| 14 | 17.96 | 0.876–0.878 | 1.044–1.044 | 1.039–1.040 | 0.998–0.999 | 0.999–1.000 |
| 15 | 18.87 | 0.895–0.898 | 1.047–1.049 | 1.044–1.046 | 1.001–1.002 | 0.999–1.000 |
| 16 | 19.71 | 0.698–0.700 | 0.707–0.717 | 0.717–0.717 | 0.707–0.708 | 1.000–1.000 |
| 17 | 20.71 | 0.758–0.759 | 0.758–0.759 | 0.759–0.761 | 0.758–0.760 | 1.000–1.001 |
| 24 | 27.60 | 0.637–0.779 | 0.636–0.637 | 0.636–0.639 | 0.636–0.638 | 1.000–1.000 |
| 32 | 34.50 | 0.533–0.533 | 0.535–0.539 | 0.539–0.540 | 0.536–0.537 | 1.000–1.005 |
| 64 | 66.47 | 0.418–0.444 | 0.424–0.452 | 0.424–0.450 | 0.427–0.454 | 1.006–1.069 |
| 256 | 250.30 | 0.371–0.373 | 0.371–0.373 | 0.371–0.372 | 0.372–0.373 | 1.000–1.007 |
| 1024 | 983.89 | 0.363–0.366 | 0.361–0.363 | 0.361–0.362 | 0.363–0.364 | 0.995–1.002 |

### Julia 1.13, Float64, `valprox_gw`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 5.90 | 1.700–1.704 | 1.231–1.233 | 1.165–1.168 | 1.031–1.033 | 0.999–1.000 |
| 2 | 7.26 | 1.621–1.622 | 1.189–1.190 | 1.162–1.162 | 1.027–1.027 | 1.000–1.000 |
| 3 | 8.44 | 1.559–1.561 | 1.163–1.163 | 1.139–1.142 | 1.023–1.023 | 1.000–1.000 |
| 4 | 12.09 | 0.992–0.996 | 1.062–1.065 | 1.045–1.045 | 1.018–1.020 | 0.999–1.002 |
| 5 | 12.88 | 1.101–1.101 | 1.074–1.074 | 1.059–1.060 | 1.016–1.017 | 0.999–1.000 |
| 6 | 14.11 | 1.114–1.117 | 1.059–1.060 | 1.049–1.051 | 1.015–1.016 | 0.999–1.000 |
| 7 | 14.91 | 1.148–1.156 | 1.056–1.058 | 1.039–1.040 | 1.009–1.010 | 1.000–1.001 |
| 8 | 14.82 | 0.914–0.917 | 1.066–1.066 | 1.044–1.047 | 1.015–1.018 | 1.000–1.000 |
| 9 | 15.78 | 1.000–1.005 | 1.039–1.043 | 1.029–1.032 | 1.003–1.012 | 1.000–1.000 |
| 10 | 16.75 | 1.020–1.029 | 1.053–1.054 | 1.039–1.040 | 1.012–1.014 | 1.000–1.000 |
| 11 | 18.13 | 1.024–1.033 | 1.042–1.046 | 1.036–1.037 | 1.008–1.009 | 0.999–1.000 |
| 12 | 18.46 | 0.809–0.810 | 1.043–1.044 | 1.032–1.033 | 1.006–1.008 | 1.000–1.001 |
| 13 | 19.93 | 0.861–0.861 | 1.039–1.040 | 1.031–1.032 | 1.008–1.009 | 1.000–1.000 |
| 14 | 21.27 | 0.880–0.886 | 1.042–1.044 | 1.033–1.035 | 1.002–1.007 | 0.999–1.000 |
| 15 | 22.59 | 0.893–0.900 | 1.046–1.047 | 1.037–1.040 | 1.008–1.009 | 0.999–1.001 |
| 16 | 23.31 | 0.702–0.703 | 0.709–0.709 | 0.713–0.714 | 0.709–0.710 | 1.000–1.000 |
| 17 | 24.78 | 0.755–0.757 | 0.761–0.761 | 0.769–0.772 | 0.767–0.774 | 1.000–1.000 |
| 24 | 33.38 | 0.635–0.635 | 0.637–0.639 | 0.638–0.643 | 0.643–0.644 | 1.000–1.000 |
| 32 | 41.62 | 0.550–0.551 | 0.554–0.555 | 0.549–0.555 | 0.554–0.556 | 1.000–1.000 |
| 64 | 79.04 | 0.477–0.478 | 0.479–0.480 | 0.477–0.478 | 0.478–0.479 | 1.000–1.001 |
| 256 | 307.08 | 0.413–0.416 | 0.413–0.417 | 0.413–0.416 | 0.413–0.416 | 0.996–0.999 |
| 1024 | 1216.68 | 0.402–0.404 | 0.403–0.405 | 0.402–0.405 | 0.403–0.405 | 0.991–1.002 |

### Julia 1.13, Float64, `prox_gx_first`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 4.01 | 1.984–2.057 | 1.203–1.253 | 1.238–1.349 | 0.961–0.977 | 0.999–1.023 |
| 2 | 4.71 | 1.845–2.000 | 1.138–1.209 | 1.217–1.219 | 0.904–1.021 | 0.923–1.000 |
| 3 | 5.39 | 1.966–2.014 | 1.200–1.231 | 1.201–1.256 | 0.988–1.044 | 0.970–0.994 |
| 4 | 6.07 | 1.872–1.901 | 1.162–1.180 | 1.180–1.226 | 1.000–1.016 | 0.984–0.999 |
| 5 | 6.75 | 1.862–1.863 | 1.163–1.164 | 1.162–1.163 | 1.013–1.014 | 0.998–0.999 |
| 6 | 7.50 | 1.810–1.813 | 1.150–1.152 | 1.125–1.126 | 1.012–1.013 | 0.997–1.000 |
| 7 | 8.27 | 1.783–1.784 | 1.142–1.143 | 1.120–1.120 | 1.010–1.012 | 0.996–0.999 |
| 8 | 9.07 | 1.627–1.630 | 1.129–1.130 | 1.113–1.113 | 1.009–1.010 | 0.997–0.998 |
| 9 | 9.87 | 1.614–1.622 | 1.121–1.121 | 1.104–1.107 | 1.006–1.008 | 0.998–0.999 |
| 10 | 10.72 | 1.601–1.603 | 1.096–1.097 | 1.090–1.096 | 1.004–1.007 | 0.999–1.001 |
| 11 | 11.62 | 1.588–1.589 | 1.085–1.089 | 1.089–1.093 | 1.003–1.003 | 0.998–1.000 |
| 12 | 12.48 | 1.480–1.489 | 1.086–1.088 | 1.082–1.083 | 1.002–1.005 | 0.999–1.000 |
| 13 | 13.36 | 1.481–1.485 | 1.087–1.089 | 1.068–1.076 | 1.009–1.009 | 0.999–1.001 |
| 14 | 14.23 | 1.478–1.478 | 1.084–1.085 | 1.065–1.069 | 1.008–1.008 | 1.000–1.000 |
| 15 | 15.15 | 1.477–1.482 | 1.079–1.080 | 1.067–1.073 | 1.006–1.008 | 0.999–1.000 |
| 16 | 16.05 | 1.397–1.399 | 1.405–1.405 | 1.395–1.398 | 1.421–1.422 | 0.999–1.000 |
| 17 | 16.95 | 1.404–1.407 | 1.415–1.415 | 1.408–1.410 | 1.423–1.426 | 0.999–0.999 |
| 24 | 23.23 | 1.337–1.346 | 1.345–1.349 | 1.341–1.349 | 1.353–1.383 | 0.998–0.999 |
| 32 | 30.47 | 1.237–1.254 | 1.240–1.358 | 1.237–1.251 | 1.291–1.387 | 1.000–1.001 |
| 64 | 64.34 | 1.157–1.162 | 1.161–1.165 | 1.160–1.164 | 1.155–1.163 | 0.994–1.002 |
| 256 | 237.71 | 1.136–1.142 | 1.134–1.138 | 1.136–1.138 | 1.138–1.139 | 0.997–1.002 |
| 1024 | 931.23 | 1.031–1.041 | 1.030–1.035 | 1.032–1.035 | 1.031–1.038 | 1.000–1.005 |

### Julia 1.13, Float64, `prox_gx_last`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 4.03 | 1.955–2.053 | 1.192–1.263 | 1.238–1.339 | 0.954–0.976 | 1.000–1.023 |
| 2 | 4.81 | 1.958–1.989 | 1.204–1.243 | 1.227–1.285 | 1.071–1.103 | 1.068–1.083 |
| 3 | 5.39 | 1.930–2.024 | 1.179–1.237 | 1.195–1.219 | 1.058–1.092 | 1.030–1.083 |
| 4 | 5.99 | 1.817–1.902 | 1.128–1.180 | 1.161–1.180 | 1.033–1.075 | 0.993–1.023 |
| 5 | 6.74 | 1.864–1.867 | 1.164–1.165 | 1.164–1.164 | 1.041–1.047 | 0.999–1.000 |
| 6 | 7.50 | 1.807–1.811 | 1.152–1.152 | 1.126–1.130 | 1.013–1.021 | 0.999–0.999 |
| 7 | 8.27 | 1.784–1.787 | 1.140–1.142 | 1.119–1.120 | 1.011–1.012 | 0.997–0.998 |
| 8 | 9.03 | 1.632–1.634 | 1.131–1.133 | 1.109–1.110 | 1.010–1.017 | 0.998–0.999 |
| 9 | 9.80 | 1.624–1.628 | 1.123–1.126 | 1.103–1.106 | 1.008–1.010 | 0.998–1.000 |
| 10 | 10.58 | 1.615–1.618 | 1.097–1.103 | 1.098–1.099 | 1.008–1.011 | 0.998–0.999 |
| 11 | 11.39 | 1.610–1.612 | 1.100–1.104 | 1.104–1.106 | 1.008–1.009 | 0.999–1.000 |
| 12 | 12.27 | 1.493–1.495 | 1.092–1.093 | 1.093–1.097 | 1.006–1.006 | 1.000–1.000 |
| 13 | 13.15 | 1.487–1.488 | 1.091–1.092 | 1.084–1.090 | 1.006–1.006 | 0.997–0.999 |
| 14 | 14.05 | 1.476–1.484 | 1.091–1.091 | 1.072–1.072 | 1.005–1.005 | 1.000–1.000 |
| 15 | 14.95 | 1.477–1.478 | 1.083–1.084 | 1.072–1.074 | 1.005–1.006 | 1.000–1.001 |
| 16 | 15.86 | 1.407–1.409 | 1.412–1.420 | 1.404–1.406 | 1.443–1.445 | 1.000–1.000 |
| 17 | 16.77 | 1.409–1.409 | 1.422–1.424 | 1.409–1.411 | 1.440–1.445 | 0.999–0.999 |
| 24 | 23.05 | 1.347–1.348 | 1.350–1.357 | 1.345–1.346 | 1.373–1.407 | 0.999–1.001 |
| 32 | 30.25 | 1.256–1.268 | 1.260–1.565 | 1.259–1.259 | 1.523–1.582 | 0.999–1.001 |
| 64 | 68.97 | 1.152–1.171 | 1.146–1.155 | 1.154–1.159 | 1.149–1.157 | 0.992–1.012 |
| 256 | 242.53 | 1.131–1.140 | 1.131–1.133 | 1.134–1.134 | 1.133–1.133 | 0.998–1.003 |
| 1024 | 938.07 | 1.129–1.130 | 1.129–1.132 | 1.131–1.132 | 1.130–1.131 | 0.996–0.996 |

### Julia 1.13, Float64, `prox_gwx_last`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 4.34 | 1.991–2.091 | 1.212–1.273 | 1.179–1.275 | 1.002–1.008 | 1.003–1.045 |
| 2 | 5.69 | 1.773–1.893 | 1.073–1.154 | 1.053–1.271 | 0.896–0.984 | 1.037–1.100 |
| 3 | 6.09 | 1.860–1.911 | 1.157–1.196 | 1.097–1.207 | 0.969–1.006 | 1.031–1.066 |
| 4 | 6.87 | 1.772–1.772 | 1.172–1.172 | 1.143–1.143 | 0.991–1.000 | 1.028–1.028 |
| 5 | 7.52 | 1.801–1.801 | 1.174–1.174 | 1.148–1.148 | 1.013–1.013 | 1.017–1.017 |
| 6 | 8.34 | 1.765–1.772 | 1.153–1.153 | 1.129–1.129 | 1.011–1.012 | 1.000–1.000 |
| 7 | 9.18 | 1.735–1.745 | 1.134–1.135 | 1.119–1.119 | 1.011–1.011 | 0.999–1.000 |
| 8 | 10.08 | 1.590–1.592 | 1.114–1.114 | 1.104–1.106 | 1.008–1.008 | 1.000–1.000 |
| 9 | 10.95 | 1.575–1.577 | 1.118–1.120 | 1.099–1.103 | 1.014–1.016 | 1.002–1.002 |
| 10 | 12.03 | 1.534–1.540 | 1.103–1.108 | 1.082–1.085 | 1.004–1.004 | 0.999–1.000 |
| 11 | 13.08 | 1.542–1.543 | 1.093–1.097 | 1.083–1.083 | 1.003–1.005 | 0.999–1.000 |
| 12 | 14.12 | 1.454–1.454 | 1.079–1.080 | 1.071–1.072 | 1.003–1.003 | 1.000–1.000 |
| 13 | 15.10 | 1.438–1.439 | 1.074–1.076 | 1.061–1.062 | 1.002–1.003 | 0.999–1.000 |
| 14 | 16.01 | 1.434–1.437 | 1.070–1.071 | 1.059–1.060 | 1.008–1.009 | 1.000–1.000 |
| 15 | 17.05 | 1.428–1.430 | 1.071–1.072 | 1.052–1.054 | 1.007–1.007 | 1.001–1.003 |
| 16 | 18.06 | 1.350–1.352 | 1.372–1.372 | 1.369–1.372 | 1.358–1.359 | 1.000–1.000 |
| 17 | 19.07 | 1.368–1.376 | 1.378–1.379 | 1.378–1.379 | 1.377–1.383 | 1.000–1.001 |
| 24 | 26.33 | 1.301–1.303 | 1.313–1.317 | 1.306–1.310 | 1.306–1.310 | 1.000–1.000 |
| 32 | 34.40 | 1.229–1.231 | 1.240–1.240 | 1.235–1.235 | 1.233–1.236 | 1.000–1.000 |
| 64 | 67.07 | 1.161–1.298 | 1.166–1.166 | 1.163–1.306 | 1.163–1.164 | 1.000–1.000 |
| 256 | 274.36 | 1.107–1.122 | 1.119–1.121 | 1.110–1.120 | 1.121–1.122 | 0.998–1.000 |
| 1024 | 1062.47 | 1.106–1.120 | 1.119–1.119 | 1.107–1.119 | 1.119–1.120 | 1.000–1.000 |

### Julia 1.13, Float32, `prox_g`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 2.91 | 2.137–2.182 | 1.403–1.432 | 1.403–1.432 | 1.034–1.067 | 1.006–1.007 |
| 2 | 3.63 | 2.053–2.059 | 1.294–1.297 | 1.297–1.370 | 1.024–1.052 | 0.999–1.000 |
| 3 | 4.38 | 1.925–1.932 | 1.254–1.254 | 1.253–1.298 | 1.030–1.030 | 1.000–1.000 |
| 4 | 5.15 | 1.521–1.525 | 1.216–1.221 | 1.188–1.217 | 1.015–1.018 | 0.999–1.000 |
| 5 | 5.89 | 1.531–1.533 | 1.199–1.201 | 1.165–1.167 | 1.017–1.018 | 0.998–1.000 |
| 6 | 6.67 | 1.505–1.506 | 1.178–1.178 | 1.148–1.148 | 1.011–1.015 | 1.000–1.000 |
| 7 | 7.45 | 1.476–1.477 | 1.162–1.164 | 1.135–1.136 | 1.011–1.012 | 0.997–0.998 |
| 8 | 8.21 | 1.047–1.052 | 1.122–1.135 | 1.121–1.126 | 1.007–1.012 | 0.995–1.000 |
| 9 | 9.03 | 1.087–1.088 | 1.117–1.120 | 1.112–1.112 | 1.012–1.013 | 1.000–1.000 |
| 10 | 9.89 | 1.095–1.096 | 1.097–1.098 | 1.084–1.090 | 1.012–1.012 | 0.999–1.000 |
| 11 | 10.78 | 1.093–1.093 | 1.092–1.092 | 1.072–1.076 | 1.011–1.012 | 0.999–1.001 |
| 12 | 11.67 | 0.824–0.824 | 1.084–1.085 | 1.067–1.068 | 1.009–1.010 | 1.000–1.001 |
| 13 | 12.62 | 0.854–0.860 | 1.084–1.090 | 1.065–1.069 | 1.008–1.008 | 1.000–1.000 |
| 14 | 13.49 | 0.879–0.881 | 1.077–1.112 | 1.067–1.101 | 1.007–1.007 | 1.000–1.000 |
| 15 | 14.39 | 0.887–0.888 | 1.068–1.073 | 1.064–1.068 | 1.007–1.007 | 1.000–1.000 |
| 16 | 15.29 | 0.681–0.681 | 0.680–0.684 | 0.694–0.695 | 0.684–0.684 | 1.000–1.001 |
| 17 | 16.20 | 0.720–0.720 | 0.716–0.717 | 0.728–0.729 | 0.723–0.723 | 1.000–1.000 |
| 24 | 22.49 | 0.543–0.544 | 0.543–0.543 | 0.550–0.551 | 0.544–0.546 | 0.998–1.000 |
| 32 | 29.74 | 0.350–0.351 | 0.349–0.352 | 0.351–0.352 | 0.351–0.353 | 1.000–1.000 |
| 64 | 63.35 | 0.213–0.215 | 0.213–0.216 | 0.213–0.215 | 0.214–0.216 | 0.919–0.928 |
| 256 | 236.65 | 0.173–0.173 | 0.172–0.173 | 0.172–0.173 | 0.173–0.174 | 0.996–0.998 |
| 1024 | 930.59 | 0.167–0.168 | 0.167–0.168 | 0.167–0.168 | 0.167–0.168 | 0.995–1.000 |

### Julia 1.13, Float32, `prox_gw`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 3.15 | 2.247–2.426 | 1.353–1.404 | 1.294–1.340 | 1.000–1.032 | 0.926–1.000 |
| 2 | 4.04 | 2.008–2.181 | 1.279–1.361 | 1.213–1.261 | 1.000–1.034 | 0.923–1.001 |
| 3 | 4.72 | 1.923–2.011 | 1.240–1.292 | 1.200–1.250 | 1.000–1.021 | 0.960–1.000 |
| 4 | 5.53 | 1.562–1.562 | 1.242–1.242 | 1.207–1.207 | 1.018–1.018 | 1.000–1.000 |
| 5 | 6.35 | 1.608–1.609 | 1.205–1.205 | 1.174–1.174 | 1.016–1.016 | 1.000–1.000 |
| 6 | 7.20 | 1.563–1.563 | 1.177–1.177 | 1.149–1.150 | 1.011–1.012 | 1.000–1.000 |
| 7 | 8.11 | 1.505–1.508 | 1.147–1.148 | 1.126–1.128 | 1.009–1.009 | 1.001–1.001 |
| 8 | 9.10 | 1.035–1.036 | 1.142–1.143 | 1.102–1.103 | 1.007–1.007 | 1.000–1.001 |
| 9 | 10.19 | 1.079–1.082 | 1.107–1.111 | 1.096–1.101 | 1.004–1.008 | 0.997–1.003 |
| 10 | 11.23 | 1.072–1.073 | 1.098–1.099 | 1.082–1.083 | 1.007–1.009 | 1.001–1.002 |
| 11 | 12.33 | 1.038–1.066 | 1.066–1.090 | 1.050–1.073 | 0.994–1.003 | 0.998–0.999 |
| 12 | 13.18 | 0.790–0.790 | 1.092–1.096 | 1.071–1.072 | 1.003–1.003 | 1.000–1.000 |
| 13 | 14.35 | 0.838–0.840 | 1.088–1.089 | 1.068–1.070 | 1.004–1.006 | 1.001–1.001 |
| 14 | 15.28 | 0.853–0.859 | 1.080–1.162 | 1.062–1.150 | 1.003–1.003 | 1.000–1.000 |
| 15 | 16.30 | 0.853–0.860 | 1.075–1.075 | 1.061–1.063 | 1.002–1.005 | 1.000–1.000 |
| 16 | 17.31 | 0.645–0.649 | 0.657–0.662 | 0.645–0.660 | 0.658–0.661 | 0.999–1.000 |
| 17 | 18.33 | 0.699–0.707 | 0.711–0.717 | 0.723–0.725 | 0.714–0.719 | 1.000–1.000 |
| 24 | 25.58 | 0.510–0.512 | 0.518–0.518 | 0.511–0.521 | 0.522–0.523 | 1.000–1.000 |
| 32 | 33.91 | 0.339–0.342 | 0.341–0.345 | 0.342–0.345 | 0.346–0.346 | 1.000–1.000 |
| 64 | 66.83 | 0.213–0.227 | 0.215–0.229 | 0.214–0.226 | 0.216–0.230 | 0.940–1.063 |
| 256 | 266.88 | 0.166–0.168 | 0.166–0.168 | 0.166–0.168 | 0.166–0.169 | 0.984–1.011 |
| 1024 | 1048.84 | 0.160–0.162 | 0.161–0.162 | 0.161–0.162 | 0.161–0.162 | 0.988–1.011 |

### Julia 1.13, Float32, `valprox_g`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 5.49 | 1.643–1.643 | 1.214–1.214 | 1.182–1.266 | 1.000–1.000 | 1.000–1.000 |
| 2 | 6.49 | 1.564–1.578 | 1.170–1.182 | 1.140–1.160 | 1.000–1.034 | 0.990–1.000 |
| 3 | 7.46 | 1.506–1.530 | 1.141–1.157 | 1.113–1.138 | 0.999–1.021 | 0.983–1.000 |
| 4 | 11.58 | 0.950–0.954 | 1.023–1.025 | 1.023–1.027 | 0.988–0.994 | 1.001–1.002 |
| 5 | 11.74 | 1.055–1.055 | 1.051–1.051 | 1.053–1.055 | 0.998–0.998 | 0.996–0.999 |
| 6 | 12.28 | 1.102–1.105 | 1.064–1.068 | 1.059–1.060 | 1.004–1.004 | 1.000–1.000 |
| 7 | 12.93 | 1.139–1.139 | 1.052–1.053 | 1.055–1.056 | 1.000–1.001 | 1.000–1.000 |
| 8 | 13.98 | 0.857–0.858 | 1.059–1.060 | 1.055–1.056 | 0.997–0.998 | 0.999–1.001 |
| 9 | 14.67 | 0.910–0.911 | 1.036–1.036 | 1.044–1.045 | 1.003–1.003 | 1.000–1.000 |
| 10 | 15.12 | 0.961–0.961 | 1.047–1.049 | 1.045–1.045 | 0.998–0.998 | 1.000–1.001 |
| 11 | 15.55 | 1.010–1.012 | 1.032–1.032 | 1.031–1.032 | 0.989–0.995 | 1.000–1.000 |
| 12 | 16.58 | 0.793–0.794 | 1.032–1.033 | 1.032–1.565 | 1.000–1.000 | 0.999–1.000 |
| 13 | 16.94 | 0.858–0.858 | 1.041–1.042 | 1.036–1.037 | 0.999–1.000 | 1.000–1.000 |
| 14 | 17.84 | 0.881–0.881 | 1.046–1.047 | 1.034–1.036 | 0.995–0.999 | 1.000–1.000 |
| 15 | 18.87 | 0.891–0.903 | 1.036–1.048 | 1.026–1.040 | 1.002–1.005 | 0.988–1.000 |
| 16 | 19.60 | 0.723–0.728 | 0.731–0.770 | 0.732–0.740 | 0.731–0.732 | 1.000–1.000 |
| 17 | 20.54 | 0.756–0.756 | 0.764–0.764 | 0.765–0.765 | 0.765–0.765 | 1.000–1.000 |
| 24 | 27.09 | 0.603–0.603 | 0.609–0.645 | 0.610–0.611 | 0.609–0.610 | 1.001–1.001 |
| 32 | 34.20 | 0.402–0.416 | 0.406–0.419 | 0.401–0.414 | 0.406–0.420 | 0.963–1.000 |
| 64 | 67.37 | 0.272–0.275 | 0.272–0.274 | 0.270–0.272 | 0.274–0.276 | 0.935–0.949 |
| 256 | 245.49 | 0.203–0.203 | 0.203–0.203 | 0.202–0.202 | 0.203–0.203 | 0.995–0.997 |
| 1024 | 955.88 | 0.191–0.192 | 0.191–0.192 | 0.191–0.192 | 0.192–0.192 | 0.997–1.001 |

### Julia 1.13, Float32, `valprox_gw`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 5.89 | 1.747–1.769 | 1.217–1.232 | 1.189–1.199 | 1.020–1.033 | 0.988–0.999 |
| 2 | 7.26 | 1.633–1.635 | 1.189–1.190 | 1.136–1.140 | 1.001–1.002 | 1.000–1.000 |
| 3 | 8.44 | 1.566–1.566 | 1.163–1.163 | 1.139–1.139 | 1.023–1.024 | 1.000–1.000 |
| 4 | 11.41 | 1.069–1.071 | 1.077–1.079 | 1.056–1.056 | 1.017–1.020 | 1.000–1.001 |
| 5 | 12.48 | 1.151–1.155 | 1.085–1.087 | 1.061–1.064 | 1.013–1.017 | 1.000–1.002 |
| 6 | 13.81 | 1.146–1.152 | 1.039–1.049 | 1.029–1.034 | 1.004–1.007 | 0.999–1.002 |
| 7 | 14.42 | 1.205–1.206 | 1.061–1.062 | 1.050–1.051 | 1.007–1.011 | 1.001–1.001 |
| 8 | 14.31 | 0.937–0.938 | 1.057–1.061 | 1.045–1.046 | 1.011–1.012 | 1.000–1.000 |
| 9 | 15.17 | 1.028–1.036 | 1.061–1.063 | 1.051–1.052 | 1.010–1.010 | 1.000–1.000 |
| 10 | 16.48 | 1.037–1.037 | 1.060–1.061 | 1.044–1.044 | 1.002–1.004 | 1.000–1.000 |
| 11 | 17.83 | 1.035–1.036 | 1.055–1.056 | 1.043–1.044 | 1.002–1.009 | 0.999–1.000 |
| 12 | 18.00 | 0.822–0.822 | 1.051–1.051 | 1.043–1.044 | 1.008–1.008 | 0.999–1.001 |
| 13 | 19.60 | 0.865–0.868 | 1.046–1.047 | 1.039–1.040 | 1.005–1.006 | 1.000–1.001 |
| 14 | 20.94 | 0.875–0.883 | 1.046–1.061 | 1.031–1.048 | 1.002–1.012 | 0.998–0.999 |
| 15 | 22.31 | 0.886–0.892 | 1.044–1.046 | 1.036–1.037 | 1.002–1.003 | 0.999–1.000 |
| 16 | 22.57 | 0.708–0.711 | 0.713–0.719 | 0.707–0.719 | 0.723–0.726 | 0.999–1.002 |
| 17 | 24.07 | 0.756–0.760 | 0.760–0.765 | 0.757–0.760 | 0.768–0.775 | 0.998–1.000 |
| 24 | 31.62 | 0.592–0.593 | 0.593–0.594 | 0.590–0.596 | 0.600–0.601 | 0.999–1.000 |
| 32 | 40.38 | 0.413–0.417 | 0.414–0.418 | 0.413–0.416 | 0.420–0.425 | 1.000–1.000 |
| 64 | 75.35 | 0.302–0.315 | 0.302–0.314 | 0.302–0.312 | 0.304–0.317 | 0.957–1.042 |
| 256 | 286.60 | 0.237–0.237 | 0.237–0.237 | 0.236–0.237 | 0.237–0.237 | 0.998–1.009 |
| 1024 | 1122.56 | 0.224–0.224 | 0.224–0.224 | 0.224–0.224 | 0.224–0.224 | 0.999–1.015 |

### Julia 1.13, Float32, `prox_gx_first`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 3.93 | 1.883–2.120 | 1.091–1.333 | 1.137–1.295 | 0.955–0.982 | 0.909–0.968 |
| 2 | 4.71 | 2.000–2.031 | 1.167–1.227 | 1.227–1.229 | 0.994–1.042 | 0.953–0.966 |
| 3 | 5.35 | 1.981–2.038 | 1.164–1.208 | 1.176–1.200 | 1.019–1.036 | 0.938–0.971 |
| 4 | 6.04 | 1.871–1.933 | 1.129–1.167 | 1.161–1.167 | 1.000–1.032 | 0.952–0.983 |
| 5 | 6.56 | 1.914–1.915 | 1.167–1.167 | 1.167–1.167 | 1.012–1.018 | 0.998–1.002 |
| 6 | 7.34 | 1.845–1.853 | 1.149–1.154 | 1.123–1.127 | 1.011–1.012 | 0.996–0.999 |
| 7 | 8.06 | 1.831–1.831 | 1.146–1.147 | 1.120–1.121 | 1.012–1.016 | 1.001–1.001 |
| 8 | 8.85 | 1.642–1.642 | 1.132–1.133 | 1.112–1.113 | 1.010–1.012 | 1.000–1.001 |
| 9 | 9.64 | 1.633–1.634 | 1.121–1.121 | 1.106–1.107 | 1.008–1.009 | 0.998–0.998 |
| 10 | 10.45 | 1.621–1.622 | 1.093–1.093 | 1.100–1.101 | 1.003–1.008 | 0.999–1.000 |
| 11 | 11.33 | 1.602–1.609 | 1.083–1.086 | 1.090–1.093 | 1.004–1.008 | 0.999–1.000 |
| 12 | 12.17 | 1.486–1.487 | 1.085–1.085 | 1.081–1.081 | 1.008–1.008 | 1.000–1.000 |
| 13 | 13.08 | 1.481–1.481 | 1.082–1.084 | 1.074–1.075 | 1.007–1.010 | 0.999–1.000 |
| 14 | 14.01 | 1.477–1.508 | 1.078–1.111 | 1.066–1.100 | 1.005–1.005 | 0.999–1.000 |
| 15 | 14.89 | 1.486–1.489 | 1.075–1.075 | 1.071–1.072 | 1.006–1.006 | 0.999–1.000 |
| 16 | 15.79 | 1.396–1.396 | 1.397–1.398 | 1.405–1.406 | 1.409–1.412 | 1.000–1.000 |
| 17 | 16.69 | 1.395–1.397 | 1.399–1.400 | 1.407–1.407 | 1.409–1.411 | 0.999–1.000 |
| 24 | 23.01 | 1.293–1.295 | 1.293–1.295 | 1.302–1.304 | 1.296–1.297 | 0.999–0.999 |
| 32 | 31.91 | 1.196–1.211 | 1.191–1.313 | 1.186–1.211 | 1.191–1.192 | 0.977–1.047 |
| 64 | 64.51 | 1.110–1.120 | 1.112–1.121 | 1.113–1.118 | 1.115–1.119 | 0.991–0.992 |
| 256 | 237.13 | 1.073–1.076 | 1.074–1.078 | 1.075–1.076 | 1.074–1.076 | 1.002–1.002 |
| 1024 | 928.72 | 1.019–1.021 | 1.022–1.022 | 1.019–1.025 | 1.018–1.020 | 1.004–1.005 |

### Julia 1.13, Float32, `prox_gx_last`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 3.90 | 2.050–2.148 | 1.200–1.364 | 1.250–1.338 | 0.993–1.038 | 1.000–1.013 |
| 2 | 4.69 | 1.975–2.046 | 1.164–1.277 | 1.305–1.344 | 1.024–1.038 | 0.945–0.955 |
| 3 | 5.04 | 2.088–2.104 | 1.247–1.300 | 1.286–1.323 | 1.032–1.069 | 1.022–1.052 |
| 4 | 5.79 | 1.966–1.966 | 1.218–1.220 | 1.195–1.268 | 1.017–1.085 | 1.017–1.051 |
| 5 | 6.56 | 1.913–1.916 | 1.166–1.170 | 1.168–1.223 | 1.012–1.048 | 0.997–1.003 |
| 6 | 7.31 | 1.850–1.852 | 1.153–1.154 | 1.127–1.180 | 1.013–1.020 | 0.998–1.000 |
| 7 | 8.08 | 1.823–1.829 | 1.142–1.145 | 1.121–1.138 | 1.010–1.010 | 0.998–1.002 |
| 8 | 8.83 | 1.648–1.648 | 1.135–1.135 | 1.113–1.114 | 1.011–1.011 | 1.000–1.000 |
| 9 | 9.62 | 1.621–1.635 | 1.115–1.125 | 1.096–1.105 | 1.008–1.010 | 0.998–1.000 |
| 10 | 10.39 | 1.629–1.629 | 1.098–1.098 | 1.098–1.098 | 1.010–1.013 | 1.000–1.000 |
| 11 | 11.26 | 1.602–1.617 | 1.086–1.092 | 1.083–1.092 | 1.004–1.009 | 0.998–1.001 |
| 12 | 12.08 | 1.492–1.493 | 1.092–1.093 | 1.071–1.079 | 1.007–1.010 | 1.000–1.001 |
| 13 | 13.01 | 1.470–1.479 | 1.080–1.086 | 1.069–1.074 | 1.007–1.010 | 0.999–1.000 |
| 14 | 13.87 | 1.479–1.509 | 1.085–1.122 | 1.070–1.099 | 1.005–1.007 | 1.000–1.000 |
| 15 | 14.81 | 1.472–1.479 | 1.079–1.081 | 1.062–1.067 | 1.007–1.008 | 1.000–1.001 |
| 16 | 15.68 | 1.390–1.391 | 1.390–1.391 | 1.404–1.405 | 1.402–1.408 | 0.999–0.999 |
| 17 | 16.60 | 1.390–1.392 | 1.397–1.398 | 1.406–1.407 | 1.404–1.406 | 0.999–0.999 |
| 24 | 22.87 | 1.294–1.295 | 1.293–1.294 | 1.304–1.385 | 1.296–1.301 | 1.000–1.000 |
| 32 | 30.21 | 1.279–1.305 | 1.227–1.307 | 1.225–1.292 | 1.182–1.333 | 0.992–1.000 |
| 64 | 68.24 | 1.121–1.127 | 1.124–1.130 | 1.125–1.135 | 1.115–1.126 | 1.011–1.016 |
| 256 | 241.96 | 1.072–1.074 | 1.075–1.080 | 1.074–1.076 | 1.074–1.075 | 1.003–1.005 |
| 1024 | 937.21 | 1.066–1.066 | 1.065–1.069 | 1.066–1.070 | 1.065–1.066 | 1.001–1.001 |

### Julia 1.13, Float32, `prox_gwx_last`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | A/A |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 4.19 | 2.001–2.213 | 1.304–1.328 | 1.279–1.344 | 0.978–1.080 | 0.896–1.082 |
| 2 | 5.03 | 1.726–2.123 | 1.265–1.279 | 1.224–1.257 | 0.976–1.085 | 0.813–1.020 |
| 3 | 6.01 | 1.758–2.023 | 1.183–1.221 | 1.186–1.199 | 1.000–1.081 | 0.869–1.012 |
| 4 | 6.56 | 1.722–1.888 | 1.194–1.219 | 1.158–1.195 | 1.000–1.036 | 0.912–1.005 |
| 5 | 7.36 | 1.747–1.839 | 1.180–1.189 | 1.162–1.179 | 0.999–1.014 | 0.949–1.000 |
| 6 | 8.12 | 1.769–1.797 | 1.166–1.167 | 1.141–1.164 | 1.000–1.012 | 0.981–1.000 |
| 7 | 8.92 | 1.786–1.788 | 1.153–1.155 | 1.130–1.135 | 1.011–1.023 | 1.000–1.000 |
| 8 | 9.83 | 1.603–1.605 | 1.141–1.143 | 1.120–1.120 | 1.005–1.008 | 0.999–0.999 |
| 9 | 10.80 | 1.575–1.576 | 1.119–1.120 | 1.106–1.107 | 1.007–1.007 | 0.999–1.000 |
| 10 | 11.81 | 1.557–1.560 | 1.114–1.116 | 1.088–1.091 | 1.005–1.008 | 1.000–1.000 |
| 11 | 12.79 | 1.545–1.549 | 1.112–1.119 | 1.096–1.098 | 1.002–1.004 | 0.999–1.002 |
| 12 | 13.76 | 1.464–1.466 | 1.102–1.103 | 1.088–1.090 | 1.004–1.005 | 1.000–1.001 |
| 13 | 14.84 | 1.460–1.464 | 1.092–1.094 | 1.077–1.081 | 1.004–1.004 | 1.000–1.000 |
| 14 | 15.85 | 1.456–1.528 | 1.083–1.175 | 1.068–1.153 | 1.002–1.002 | 0.999–1.001 |
| 15 | 16.92 | 1.448–1.453 | 1.076–1.078 | 1.064–1.065 | 1.004–1.005 | 0.999–1.000 |
| 16 | 17.91 | 1.368–1.369 | 1.368–1.374 | 1.372–1.374 | 1.374–1.376 | 1.000–1.000 |
| 17 | 18.91 | 1.362–1.364 | 1.376–1.379 | 1.371–1.373 | 1.377–1.377 | 0.999–1.000 |
| 24 | 26.20 | 1.267–1.268 | 1.278–1.279 | 1.280–1.281 | 1.278–1.279 | 1.000–1.000 |
| 32 | 34.29 | 1.210–1.213 | 1.205–1.206 | 1.209–1.211 | 1.214–1.218 | 1.000–1.000 |
| 64 | 71.13 | 0.997–1.127 | 1.115–1.126 | 0.997–1.128 | 0.999–1.130 | 0.882–1.128 |
| 256 | 272.18 | 1.067–1.082 | 1.066–1.069 | 1.066–1.081 | 1.066–1.082 | 0.985–1.014 |
| 1024 | 1054.35 | 1.063–1.078 | 1.062–1.070 | 1.063–1.077 | 1.062–1.078 | 0.987–1.015 |


## Appendix B: exceptional fallback against both master and B925m

### Exceptional-step fallback, n ≥ 16

Each ratio range spans both full processes. Columns against B925m use paired raw ratios directly. The three operations are unweighted first zero, unweighted last zero, and weighted last zero.


Julia 1.10, Float64, `prox_gx_first`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | s / m | t / m | u / m | A/A |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 16 | 22.50 | 1.252–1.252 | 1.253–1.253 | 1.253–1.253 | 1.253–1.254 | 1.000–1.001 | 1.001–1.002 | 1.001–1.002 | 1.000–1.001 |
| 17 | 23.85 | 1.257–1.257 | 1.257–1.257 | 1.257–1.257 | 1.257–1.259 | 1.000–1.001 | 1.000–1.000 | 1.000–1.001 | 0.999–1.001 |
| 32 | 42.57 | 1.160–1.162 | 1.158–1.163 | 1.160–1.162 | 1.163–1.164 | 0.998–1.001 | 1.000–1.001 | 1.001–1.002 | 0.999–0.999 |
| 64 | 82.68 | 1.117–1.117 | 1.117–1.118 | 1.118–1.118 | 1.118–1.119 | 0.999–1.001 | 1.000–1.001 | 1.001–1.002 | 0.998–1.000 |
| 256 | 331.04 | 1.057–1.094 | 1.057–1.095 | 1.057–1.095 | 1.058–1.095 | 0.999–1.001 | 1.000–1.001 | 1.001–1.001 | 0.999–0.999 |
| 1024 | 1297.27 | 1.021–1.031 | 1.024–1.032 | 1.020–1.029 | 1.030–1.034 | 1.000–1.001 | 0.997–0.998 | 1.002–1.008 | 0.996–0.997 |

Julia 1.10, Float64, `prox_gx_last`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | s / m | t / m | u / m | A/A |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 16 | 22.80 | 1.231–1.257 | 1.230–1.259 | 1.232–1.260 | 1.231–1.258 | 1.000–1.002 | 1.001–1.002 | 1.000–1.001 | 0.976–1.000 |
| 17 | 23.73 | 1.260–1.260 | 1.260–1.260 | 1.261–1.261 | 1.262–1.264 | 1.000–1.000 | 1.000–1.001 | 1.002–1.003 | 1.000–1.001 |
| 32 | 42.36 | 1.170–1.170 | 1.170–1.173 | 1.171–1.172 | 1.172–1.173 | 1.000–1.003 | 1.001–1.002 | 1.002–1.003 | 0.999–0.999 |
| 64 | 82.54 | 1.120–1.121 | 1.121–1.122 | 1.121–1.122 | 1.122–1.122 | 1.000–1.001 | 1.000–1.001 | 1.001–1.002 | 0.998–1.000 |
| 256 | 330.90 | 1.058–1.095 | 1.058–1.096 | 1.058–1.096 | 1.059–1.096 | 1.000–1.001 | 1.000–1.000 | 1.001–1.001 | 0.999–1.000 |
| 1024 | 1296.20 | 1.093–1.109 | 1.093–1.104 | 1.095–1.112 | 1.095–1.105 | 0.995–1.001 | 1.001–1.002 | 0.996–1.002 | 1.002–1.008 |

Julia 1.10, Float64, `prox_gwx_last`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | s / m | t / m | u / m | A/A |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 16 | 25.20 | 1.235–1.277 | 1.240–1.276 | 1.240–1.279 | 1.239–1.279 | 0.999–1.005 | 1.002–1.005 | 1.002–1.003 | 1.030–1.030 |
| 17 | 26.66 | 1.247–1.273 | 1.246–1.274 | 1.249–1.273 | 1.248–1.274 | 1.000–1.001 | 1.000–1.002 | 1.001–1.002 | 1.030–1.031 |
| 32 | 47.59 | 1.163–1.196 | 1.163–1.198 | 1.162–1.197 | 1.163–1.196 | 1.001–1.002 | 1.000–1.001 | 1.001–1.001 | 1.025–1.027 |
| 64 | 92.76 | 1.116–1.141 | 1.116–1.142 | 1.116–1.141 | 1.116–1.142 | 1.000–1.001 | 1.000–1.001 | 1.000–1.001 | 1.028–1.028 |
| 256 | 363.69 | 1.087–1.118 | 1.088–1.118 | 1.088–1.118 | 1.088–1.119 | 1.000–1.001 | 1.000–1.001 | 1.000–1.000 | 1.030–1.030 |
| 1024 | 1446.83 | 1.088–1.122 | 1.088–1.119 | 1.088–1.118 | 1.088–1.116 | 0.995–1.000 | 0.998–1.000 | 0.996–1.000 | 1.033–1.035 |

Julia 1.10, Float32, `prox_gx_first`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | s / m | t / m | u / m | A/A |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 16 | 22.24 | 1.385–1.406 | 1.404–1.413 | 1.392–1.411 | 1.428–1.430 | 0.998–1.019 | 1.005–1.005 | 1.017–1.031 | 1.000–1.001 |
| 17 | 23.46 | 1.388–1.400 | 1.392–1.392 | 1.382–1.404 | 1.400–1.400 | 0.993–1.001 | 0.995–1.000 | 0.997–1.006 | 0.999–1.000 |
| 32 | 42.24 | 1.136–1.136 | 1.137–1.138 | 1.137–1.139 | 1.138–1.139 | 1.001–1.001 | 1.001–1.002 | 1.002–1.002 | 1.000–1.000 |
| 64 | 82.13 | 1.088–1.089 | 1.088–1.088 | 1.088–1.088 | 1.088–1.089 | 1.000–1.000 | 0.999–1.001 | 1.000–1.001 | 0.999–1.001 |
| 256 | 324.20 | 1.053–1.053 | 1.053–1.053 | 1.053–1.053 | 1.053–1.053 | 1.000–1.000 | 1.000–1.000 | 1.000–1.000 | 1.000–1.000 |
| 1024 | 1299.60 | 1.012–1.013 | 1.013–1.015 | 1.014–1.014 | 1.012–1.013 | 1.000–1.001 | 1.000–1.001 | 1.000–1.000 | 1.000–1.006 |

Julia 1.10, Float32, `prox_gx_last`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | s / m | t / m | u / m | A/A |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 16 | 22.16 | 1.373–1.390 | 1.387–1.387 | 1.373–1.384 | 1.389–1.393 | 0.998–1.010 | 0.997–0.999 | 0.998–1.013 | 1.000–1.000 |
| 17 | 23.45 | 1.378–1.402 | 1.400–1.402 | 1.371–1.391 | 1.391–1.392 | 1.010–1.012 | 0.993–0.994 | 0.990–1.011 | 0.999–1.001 |
| 32 | 42.14 | 1.132–1.132 | 1.133–1.133 | 1.132–1.133 | 1.133–1.134 | 1.001–1.001 | 1.000–1.001 | 1.001–1.001 | 1.000–1.000 |
| 64 | 82.06 | 1.087–1.088 | 1.087–1.087 | 1.087–1.087 | 1.087–1.088 | 1.000–1.000 | 1.000–1.001 | 1.000–1.000 | 0.999–1.001 |
| 256 | 324.24 | 1.052–1.052 | 1.052–1.052 | 1.052–1.052 | 1.052–1.052 | 1.000–1.000 | 1.000–1.000 | 1.000–1.000 | 1.000–1.001 |
| 1024 | 1289.57 | 1.052–1.052 | 1.052–1.056 | 1.052–1.053 | 1.054–1.057 | 1.000–1.003 | 1.000–1.001 | 1.002–1.006 | 1.000–1.000 |

Julia 1.10, Float32, `prox_gwx_last`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | s / m | t / m | u / m | A/A |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 16 | 24.72 | 1.365–1.376 | 1.355–1.380 | 1.366–1.385 | 1.372–1.387 | 0.986–1.011 | 1.000–1.004 | 0.999–1.013 | 0.998–1.000 |
| 17 | 26.20 | 1.352–1.354 | 1.355–1.360 | 1.350–1.360 | 1.350–1.359 | 1.001–1.006 | 0.998–1.004 | 0.997–1.005 | 1.000–1.001 |
| 32 | 47.35 | 1.125–1.126 | 1.126–1.132 | 1.127–1.127 | 1.126–1.128 | 1.001–1.005 | 1.001–1.001 | 1.001–1.002 | 1.000–1.000 |
| 64 | 91.61 | 1.084–1.085 | 1.080–1.085 | 1.084–1.085 | 1.086–1.087 | 0.995–1.001 | 0.999–1.001 | 1.001–1.002 | 1.000–1.000 |
| 256 | 363.36 | 1.046–1.048 | 1.047–1.048 | 1.047–1.047 | 1.047–1.048 | 1.000–1.002 | 0.999–1.000 | 1.001–1.001 | 1.000–1.000 |
| 1024 | 1446.99 | 1.047–1.064 | 1.047–1.058 | 1.052–1.052 | 1.047–1.053 | 0.994–1.000 | 0.987–1.004 | 0.990–0.999 | 1.000–1.004 |

Julia 1.13, Float64, `prox_gx_first`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | s / m | t / m | u / m | A/A |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 16 | 16.05 | 1.397–1.399 | 1.405–1.405 | 1.395–1.398 | 1.421–1.422 | 1.005–1.006 | 0.999–0.999 | 1.016–1.017 | 0.999–1.000 |
| 17 | 16.95 | 1.404–1.407 | 1.415–1.415 | 1.408–1.410 | 1.423–1.426 | 1.006–1.006 | 1.000–1.004 | 1.013–1.013 | 0.999–0.999 |
| 32 | 30.47 | 1.237–1.254 | 1.240–1.358 | 1.237–1.251 | 1.291–1.387 | 1.003–1.057 | 0.999–1.001 | 1.031–1.128 | 1.000–1.001 |
| 64 | 64.34 | 1.157–1.162 | 1.161–1.165 | 1.160–1.164 | 1.155–1.163 | 1.003–1.003 | 1.002–1.003 | 0.999–1.001 | 0.994–1.002 |
| 256 | 237.71 | 1.136–1.142 | 1.134–1.138 | 1.136–1.138 | 1.138–1.139 | 0.997–0.999 | 0.997–1.001 | 0.998–1.001 | 0.997–1.002 |
| 1024 | 931.23 | 1.031–1.041 | 1.030–1.035 | 1.032–1.035 | 1.031–1.038 | 0.995–0.999 | 0.995–1.000 | 0.998–1.000 | 1.000–1.005 |

Julia 1.13, Float64, `prox_gx_last`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | s / m | t / m | u / m | A/A |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 16 | 15.86 | 1.407–1.409 | 1.412–1.420 | 1.404–1.406 | 1.443–1.445 | 1.004–1.008 | 0.998–0.998 | 1.025–1.025 | 1.000–1.000 |
| 17 | 16.77 | 1.409–1.409 | 1.422–1.424 | 1.409–1.411 | 1.440–1.445 | 1.010–1.011 | 1.001–1.001 | 1.023–1.026 | 0.999–0.999 |
| 32 | 30.25 | 1.256–1.268 | 1.260–1.565 | 1.259–1.259 | 1.523–1.582 | 1.002–1.202 | 0.992–1.000 | 1.236–1.252 | 0.999–1.001 |
| 64 | 68.97 | 1.152–1.171 | 1.146–1.155 | 1.154–1.159 | 1.149–1.157 | 0.985–0.995 | 0.990–1.003 | 0.988–0.997 | 0.992–1.012 |
| 256 | 242.53 | 1.131–1.140 | 1.131–1.133 | 1.134–1.134 | 1.133–1.133 | 0.994–1.000 | 0.994–1.001 | 0.994–1.000 | 0.998–1.003 |
| 1024 | 938.07 | 1.129–1.130 | 1.129–1.132 | 1.131–1.132 | 1.130–1.131 | 1.000–1.002 | 1.000–1.002 | 1.000–1.000 | 0.996–0.996 |

Julia 1.13, Float64, `prox_gwx_last`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | s / m | t / m | u / m | A/A |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 16 | 18.06 | 1.350–1.352 | 1.372–1.372 | 1.369–1.372 | 1.358–1.359 | 1.014–1.016 | 1.013–1.017 | 1.005–1.005 | 1.000–1.000 |
| 17 | 19.07 | 1.368–1.376 | 1.378–1.379 | 1.378–1.379 | 1.377–1.383 | 1.003–1.007 | 1.003–1.007 | 1.006–1.006 | 1.000–1.001 |
| 32 | 34.40 | 1.229–1.231 | 1.240–1.240 | 1.235–1.235 | 1.233–1.236 | 1.006–1.009 | 1.003–1.005 | 1.003–1.004 | 1.000–1.000 |
| 64 | 67.07 | 1.161–1.298 | 1.166–1.166 | 1.163–1.306 | 1.163–1.164 | 0.898–1.005 | 0.896–1.125 | 0.896–1.003 | 1.000–1.000 |
| 256 | 274.36 | 1.107–1.122 | 1.119–1.121 | 1.110–1.120 | 1.121–1.122 | 0.999–1.010 | 0.989–1.011 | 0.999–1.013 | 0.998–1.000 |
| 1024 | 1062.47 | 1.106–1.120 | 1.119–1.119 | 1.107–1.119 | 1.119–1.120 | 0.999–1.012 | 0.988–1.012 | 1.000–1.012 | 1.000–1.000 |

Julia 1.13, Float32, `prox_gx_first`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | s / m | t / m | u / m | A/A |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 16 | 15.79 | 1.396–1.396 | 1.397–1.398 | 1.405–1.406 | 1.409–1.412 | 1.000–1.000 | 1.007–1.007 | 1.009–1.012 | 1.000–1.000 |
| 17 | 16.69 | 1.395–1.397 | 1.399–1.400 | 1.407–1.407 | 1.409–1.411 | 1.001–1.002 | 1.007–1.008 | 1.009–1.009 | 0.999–1.000 |
| 32 | 31.91 | 1.196–1.211 | 1.191–1.313 | 1.186–1.211 | 1.191–1.192 | 0.924–1.044 | 0.943–0.983 | 0.955–0.968 | 0.977–1.047 |
| 64 | 64.51 | 1.110–1.120 | 1.112–1.121 | 1.113–1.118 | 1.115–1.119 | 1.002–1.003 | 0.999–1.002 | 0.999–1.005 | 0.991–0.992 |
| 256 | 237.13 | 1.073–1.076 | 1.074–1.078 | 1.075–1.076 | 1.074–1.076 | 1.001–1.002 | 1.000–1.002 | 1.000–1.001 | 1.002–1.002 |
| 1024 | 928.72 | 1.019–1.021 | 1.022–1.022 | 1.019–1.025 | 1.018–1.020 | 1.001–1.002 | 1.000–1.001 | 1.000–1.000 | 1.004–1.005 |

Julia 1.13, Float32, `prox_gx_last`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | s / m | t / m | u / m | A/A |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 16 | 15.68 | 1.390–1.391 | 1.390–1.391 | 1.404–1.405 | 1.402–1.408 | 1.000–1.000 | 1.009–1.010 | 1.008–1.013 | 0.999–0.999 |
| 17 | 16.60 | 1.390–1.392 | 1.397–1.398 | 1.406–1.407 | 1.404–1.406 | 1.004–1.005 | 1.010–1.011 | 1.010–1.010 | 0.999–0.999 |
| 32 | 30.21 | 1.279–1.305 | 1.227–1.307 | 1.225–1.292 | 1.182–1.333 | 0.939–0.960 | 0.958–0.999 | 0.816–1.005 | 0.992–1.000 |
| 64 | 68.24 | 1.121–1.127 | 1.124–1.130 | 1.125–1.135 | 1.115–1.126 | 1.002–1.004 | 1.006–1.009 | 0.995–1.000 | 1.011–1.016 |
| 256 | 241.96 | 1.072–1.074 | 1.075–1.080 | 1.074–1.076 | 1.074–1.075 | 1.003–1.005 | 1.001–1.002 | 1.000–1.002 | 1.003–1.005 |
| 1024 | 937.21 | 1.066–1.066 | 1.065–1.069 | 1.066–1.070 | 1.065–1.066 | 1.000–1.002 | 1.000–1.004 | 0.999–1.000 | 1.001–1.001 |

Julia 1.13, Float32, `prox_gwx_last`

| n | master ns | B925m / M | B925s / M | B925t / M | B925u / M | s / m | t / m | u / m | A/A |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 16 | 17.91 | 1.368–1.369 | 1.368–1.374 | 1.372–1.374 | 1.374–1.376 | 1.000–1.004 | 1.002–1.004 | 1.003–1.006 | 1.000–1.000 |
| 17 | 18.91 | 1.362–1.364 | 1.376–1.379 | 1.371–1.373 | 1.377–1.377 | 1.010–1.011 | 1.007–1.008 | 1.010–1.011 | 0.999–1.000 |
| 32 | 34.29 | 1.210–1.213 | 1.205–1.206 | 1.209–1.211 | 1.214–1.218 | 0.994–0.995 | 0.998–0.999 | 1.003–1.003 | 1.000–1.000 |
| 64 | 71.13 | 0.997–1.127 | 1.115–1.126 | 0.997–1.128 | 0.999–1.130 | 0.999–1.119 | 1.000–1.000 | 1.002–1.002 | 0.882–1.128 |
| 256 | 272.18 | 1.067–1.082 | 1.066–1.069 | 1.066–1.081 | 1.066–1.082 | 0.988–0.998 | 0.998–0.999 | 0.998–1.000 | 0.985–1.014 |
| 1024 | 1054.35 | 1.063–1.078 | 1.062–1.070 | 1.063–1.077 | 1.062–1.078 | 0.992–1.000 | 1.000–1.000 | 1.000–1.000 | 0.987–1.015 |

