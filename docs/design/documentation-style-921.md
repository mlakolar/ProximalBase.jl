# Documentation audit and editorial style

This is the Stage 0 audit and review record for issue #921. It describes an
editorial refactor of the minimal 0.6 documentation, not a new numerical
contract. The baseline is commit
`18b2714955c224300f8924aeb3a205b84c62be57`. Runtime behavior, supported signatures,
failure behavior and approved precision guarantees remain with their source,
tests and current manual pages.

## Baseline and scope

At the pinned audit baseline, the public inventory contained **51 exported
bindings**, **14 qualified public bindings**, and the module binding: 66 explicit
`@docs` entries and 65 operation/type coverage rows. Immediately after #937 retired the
scalar observation-loss API, the core had 51 exports, 8 qualified public
bindings and the module (60 reference bindings, 59 operation/type rows).
Later #957 and #961 added `NormLinf` and `SqrtLeastSquaresLoss`. These are not
part of the pinned baseline; `docs/check_api.jl` checks the live inventory
on every build rather than treating the old counts as current. Qualified public
names remain usable on Julia 1.10 even though the language-level `public`
marker starts in Julia 1.11.

At the audit baseline, the concrete array-function families were
`ZeroFunction`, `LinearFunction`, `SqrNormL2`, `QuadraticFunction`,
`LeastSquaresLoss`, `NormL1`, `ElasticNet`, `NormL2` and `GroupNormL2`;
`Scaled` was their composition wrapper. The current core additionally has
`NormLinf` and `SqrtLeastSquaresLoss`.
At the pinned baseline, the separate scalar interface still contained
`ProximalBase.Squared`; #937 subsequently removed that interface, its six
docstrings and the six qualified public names. The retained iterates are
`SparseIterate`, `SymmetricSparseIterate` and `BlockIterate`. The archived full
package is not a source of additional current APIs.

The reported issue baseline predates recent merges. At the pinned commit
there were **71** source docstrings; immediately after #937 there were **65**.
The newer functions added more docstrings. The table below measures only
the pinned baseline, not today's moving master:

| Measure | Reproduced result |
|:--|--:|
| Source docstrings using a multiline triple-quoted block | 71 |
| Words, including signatures and example text | 10,934 |
| Public/module/selected Base-method docstrings | 70 |
| Private docstrings | 1 (`_model_precision`, 67 words) |
| Source docstrings longer than 250 words | 14 |
| Source docstrings containing a fenced example | 5 |
| Source docstrings containing a `jldoctest` | 0 |
| Source docstrings containing `@ref` | 4 |
| Authored manual `@example` blocks | 78 |
| Authored manual `jldoctest` blocks | 0 |
| Isolated Literate example files in the build | 9 |

The four existing source `@ref` users are the least-squares predictor/residual
oracles. The old statement that source docstrings contain no checked links is
therefore no longer correct.

| Long docstring | Baseline words |
|:--|--:|
| `GroupNormL2` | 918 |
| `ElasticNet` | 636 |
| `SparseIterate` | 536 |
| `BlockIterate` | 495 |
| `SymmetricSparseIterate` | 486 |
| `LinearFunction` | 445 |
| `NormL2` | 425 |
| Symmetric-owner three-factor product | 409 |
| Sparse-iterate three-factor product | 406 |
| `QuadraticFunction` | 391 |
| `value_from_predictor` | 382 |
| `value_and_gradient_from_predictor!` | 355 |
| `NormL1` | 332 |
| `SqrNormL2` | 257 |

### How to reproduce the counts

These are editorial measurements, not parser-certified coverage. The heuristic
matches standalone opening/closing triple quotes at the same indentation and
splits the enclosed text on whitespace. It includes signatures, formulas and
code. Manual counts include authored `docs/src` Markdown only, not generated
Literate/contract pages. Each source match was inspected: all 71 are attached
docstrings, and `_model_precision` is the one private binding.

Run the following from a repository with the baseline commit available:

```python
import re
import subprocess

base = "18b2714955c224300f8924aeb3a205b84c62be57"
paths = subprocess.check_output(
    ["git", "ls-tree", "-r", "--name-only", base, "src", "docs/src"],
    text=True,
).splitlines()
pattern = re.compile(r'^([ \t]*)(?:raw)?"""\n(.*?)^\1"""', re.M | re.S)
rows = []
examples = doctests = 0
for path in paths:
    text = subprocess.check_output(["git", "show", f"{base}:{path}"], text=True)
    if path.startswith("src/") and path.endswith(".jl"):
        for match in pattern.finditer(text):
            body = match.group(2)
            rows.append((path, len(body.split()), "```" in body,
                         "jldoctest" in body, "@ref" in body))
    elif path.endswith(".md"):
        examples += text.count("```@example")
        doctests += text.count("```jldoctest")
print("docstrings, words:", len(rows), sum(row[1] for row in rows))
print("over 250 words:", sum(row[1] > 250 for row in rows))
print("fences, doctests, links:", *(sum(row[i] for row in rows) for i in (2, 3, 4)))
print("manual examples, doctests:", examples, doctests)
for row in sorted(rows, key=lambda row: row[1], reverse=True)[:14]:
    print(row[:2])
```

Do not turn this heuristic into a blocking source linter. It has not been
validated as a Julia parser and would need fixtures for multiline signatures,
raw strings, nested markup and private documentation before enforcement.
The existing explicit API inventory, strict Documenter build and rendered
link checker remain the acceptance gates. A new formatter or linter is optional
follow-up work, not a dependency of readable help.

## Reviewed style

Use this order where it helps the reader:

1. Four-space-indented signatures.
2. One sentence saying what the object or operation computes.
3. Its mathematical definition, with each parameter explained.
4. Essential input, ownership, output and mutation semantics.
5. A deterministic executable example.
6. A small set of critical restrictions and checked links to detailed rules.

Keep most quick-reference entries within about 250 words. This is a review
prompt, not a limit that justifies deleting a condition. A reader must be able
to choose the correct argument shape, avoid unsafe aliases, understand what a
mutating call returns and identify unsupported operations from the help entry.
Do not repeat a long exception table in every constructor.

Use `jldoctest` for stable short transcripts, preferably scalar values,
Boolean comparisons or deliberately exact arrays. The source-docstring setup
in `make.jl` loads `ProximalBase` and `LinearAlgebra`. Manual doctests need their
own setup or explicit imports. Keep `@example` where shared setup is useful;
existing `@assert` statements already verify their results.

Use checked symbol links such as ``[`prox!`](@ref)`` and explicit page anchors
such as `[Numerical contract](@ref numerics)`. Add a canonical destination
before replacing the only explanation with a link. Keep the dispatch signatures
in the explicit API entries for method-specific Base multiplication docs and
the two `dropzeros!` methods. `checkdocs=:exports` plus the live qualified-public
inventory already covers the intended surface; a switch to another mode is
unnecessary for this refactor.

Issue numbers, timing campaigns, old alternatives and historical rule labels
belong in developer/design history. A source comment should explain the current
invariant, arithmetic decision or performance reason. Removing history must
not remove the reason a recovery or zero check exists. A label such as `M3`
can be a processor name, and `L1` can name a norm; automated label deletion
would be unsafe.

## Before/after review samples

The initial sample review compared the complete baseline help with the proposed
short help and the detailed manual destinations. These excerpts record the
editorial decision; the executable versions live beside their declarations.

### `GroupNormL2`

**Before:** the 918-word constructor help combined the definition and input
rules with constructor representation, compact block orchestration, every KKT
endpoint, a full non-finite-input table, recovery implementation thresholds and
historical issue references. Its example used `randn`.

**After:** lead with the disjoint-group penalty, default weights and unpenalized
entries. Retain nonnegative/non-NaN parameters, copied constructor inputs, real
one-based storage, the scalar-step restriction, compact group ordering, exact
input/output aliasing and independent iterate output. Keep the infinite-group
constraint at zero step visible and link to the canonical endpoints. Use an
exact small example:

```julia
f = GroupNormL2(1.0, [1:2], [1.0])
f([3.0, 4.0])                         # 5.0
prox_block(f, [3.0, 4.0], 1, 5.0)     # [0.0, 0.0]
```

The sample review specifically retained the absence of a symmetric-owner prox,
`restrict`, per-group steps and a `Scaled` compact-block method. A copied input
array does not imply that mutating internal fields is supported. An infinite
group weight fixes even non-finite inputs; this must not be generalized to an
infinite `NormL1` coordinate weight, whose guarantee is narrower.

### `SparseIterate`

**Before:** the 536-word constructor help introduced the type, working-set
semantics, indexing and alias behavior, then described QR/LQ/Hessenberg
representations, copying and numerical products in detail.

**After:** lead with array-compatible working-set storage, linear coordinates,
constant-time entry access and storage proportional to logical length. Show
how nonzero writes activate and zero writes preserve an active coordinate.
Keep the difference between a state copy and a value write, live-view
invalidation, bulk complex rejection and possible partial conversion failure
visible. Link to the factorization and product sections for the method-specific
rules. A short example can expose the active-zero lifecycle directly:

```julia
x = SparseIterate(3)
x[2] = 2.0
x[2] = 0.0
nactive(x)                 # 1
dropzeros!(x)
nactive(x)                 # 0
```

The corresponding symmetric help must continue to say that its owner is not
an `AbstractArray`; callers choose `coordinates` or `matrixview` explicitly.

## Preservation map

| Detail removed from repeated quick help | Canonical rendered destination |
|:--|:--|
| Homogeneous `Float32`/`Float64` guarantees and mixed-precision limits | `numerics.md` and `numerical_usage.md` |
| Infinite levels, zero steps, non-finite data and endpoint precedence | `numerics.md`, with short critical warnings retained in constructor help |
| Group ownership, compact block ordering, unsupported combinations and KKT screen | `group_penalty.md` |
| Norm, threshold and value recovery limits | `numerics.md` |
| Least-squares shapes, borrowed data, cache invalidation and predictor/residual limitations | `smooth.md` |
| Indefinite quadratic matrices, convexity trait and admissible proximal steps | `smooth.md` and `function_objects.md`; preserve the implemented policy |
| Sparse/symmetric replacement copies, factorization inputs and product domains | `sparse_iterate.md`, `symmetric_sparse_iterate.md` and `iterates.md` |
| Ordered group partitions, activity and block aliases | `block_iterate.md` |
| View lifetime, aliases, conversion failures and atomicity limits | `numerical_usage.md` and `developer/views_aliasing.md` |
| Value/gradient/prox return conventions and raw extension boundaries | `operations.md`, `extending.md` and `iterate_oracles.md` |
| Historical alternatives, issue evidence and machine-specific measurements | `docs/design/` and immutable issue/PR records |

This map is checked against each edited docstring, not used as permission to
drop a fact merely because a broadly related page exists. Missing constructor
ownership/type details or failure rules are added at their destination before
the repeated text is removed.

## Navigation and correctness findings

The audit identified current information that the rewrite must make easier
to find:

- Add a short function/oracle selection path for first use. Distinguish a
  residual penalty, a regression loss and a general quadratic without
  recommending the retired scalar observation-loss interface.
- Make numerical usage and the current contract discoverable to users while
  keeping design history separate from normative behavior.
- Repair the unindented leading `prox_block` and `kkt_violation` signatures.
- Reconcile statements that compact block prox has no built-in implementation:
  `GroupNormL2` now has one. Distinguish its group-norm KKT screen and explicit
  layout/gradient checks from the generic maximum of scalar screens.
- Update the source map for restored norm/elastic-net functions and the new
  least-squares predictor oracles. Update stale eight-example counts to nine.
- Keep one-based rejection, symmetry-model preconditions, indefinite-quadratic
  support and function-specific exceptional steps as currently implemented.
  This task does not decide a different policy for any of them.

## Validation and independent review

The work is organized as functions/oracles, iterates, and navigation/polish.
Independent review checks the concise samples before broad edits and checks
the completed diff for lost conditions and unsupported promises. Production
Julia expressions must be unchanged: source edits are docstrings and comments.

Acceptance uses the strict Documenter build, doctests, isolated Literate
examples, explicit API coverage, authored Markdown check and rendered HTML/link
check. Review the rendered quick reference and first-use pages as well as the
Markdown diff. No numerical performance campaign is needed for an editorial
change; ordinary repository CI remains required. Report executed validation
separately from any check that was unavailable in the editing environment.

This sentence is a CI probe and is never merged.
