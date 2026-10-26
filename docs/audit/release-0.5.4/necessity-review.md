# Necessity review of the release repairs

This pass starts from PR #128 at `5839b4e` and incorporates master `4dc00b0`
(PR #127). The first audit remains a historical record. This pass asks which
code and API surface are required, rather than accepting a rationale for every
existing implementation. Coverage percentages are not an acceptance target.

## Removed complexity and exposure

| Change | Evidence and resulting contract |
| --- | --- |
| Delete scalar-domain and conjugate-slope inference | Its two consumers only rejected improper box compositions. The partial tables could not infer arbitrary custom domains and required wrapper propagation and precision policy. A nonempty `dom(f) ∩ box` is now a uniform caller precondition. No rejection guarantee is made for an improper composition. This deliberately withdraws the first audit's proposed policy; valid compositions retain their proximal behavior. |
| Remove workspace injection from `Conjugate` | Only the one-argument constructor is documented or used by known downstream callers. It now owns its `Scratch`; the undocumented two-argument constructors are removed. Deep copies retain separate workspace ownership. |
| Stop promoting internal group hierarchy types | The architecture still names internal types, but the migration guide no longer recommends subtyping or dispatching on them. The existing `AbstractGroupNorm` alias remains for actual downstream compatibility. |
| Remove unused QR metadata and outer mutability | `QRWorkspace.m/n` were written but never read. Its remaining fields are never reassigned; the arrays and status reference remain mutable numerical buffers. |
| Require the spectral reconstruction buffer | Every production caller already supplies it. The allocating default and its keyword surface were unnecessary; the internal buffer is a required positional argument. |
| Remove the three-argument `_lq_root` adapter | Production uses the four-argument kernel. The only old-arity callers were tests; they now compare the public scalar proximal operation with an independent reference. |
| Remove a GPU nonzero reduction and transfer | Stable group norms already distinguish zero groups. Group-L0 selection uses those norms directly; zero-budget and exceptional-value behavior has regression coverage. |
| Keep generated evidence out of tracked package sources | Raw receipts, repeated hash maps, recovered Page JSON and PNG mirrors are preserved locally and at the immutable first-audit commit. Markdown/Mermaid sources and the audit ledger remain reviewable without shipping those generated copies. |

## Removed computational work

GPU grouped `prox!` previously called the combined value/proximal routine and
discarded its value. That performed host-valued reductions and possible device
synchronization. A private compile-time flag now shares the proximal algorithm
while eliminating those value reductions. It does not add a public keyword.

For sparse-group penalties with a zero group term, only masked soft thresholding
is needed. This path now skips group gathers, norm reductions and factor
buffers. Entries outside penalty groups retain their existing behavior.

## Additional correctness defects

The first audit missed conjugate paths that still squared unscaled norms,
quadratic conjugate energies with overflowing or underflowing intermediates,
and a dual-ball tolerance addition that could overflow at `floatmax`.
These now use the existing robust norm calculation, scaled energies and a
stable comparison, with independent BigFloat references.

Grouped thresholds also formed `γ * λ0 * w` in an unsafe order. A large overall
level and compensating small weight could erase a coefficient; GPU zero-weight
groups could be erased or become `NaN` after `γλ0` overflowed. One private
three-factor product helper preserves the ordinary path and reorders exceptional
products. CPU and shared device tests cover these cases. Group values apply
the overall level before an unscaled subtotal can overflow.

## Compatibility boundary

No documented export was removed. Absence of a current workspace caller does
not make an established library operation private. The documented ScalarLoss
and Testing interfaces remain. Legacy 0.4 names retain their existing removal
policy; deleting them is separate 0.6 work. Private downstream uses of `_entry`
and `_checkparams` are coupling to remove in those callers, not APIs to export.

## Validation and limits

Meaningful regressions target the confirmed mathematical and ownership defects.
The standard `scripts/test.jl` workflow uses `Pkg.test()` with bounds checking;
the obsolete cross-package bootstrap stays removed. Final execution counts,
the exact source head and immutable CI links are recorded in PR #128 and its
delivery receipt. Local Julia runs remain sequential.

Before/after JLArrays diagnostics measure avoided work on a CPU reference
backend. They do not establish physical GPU compilation or performance.
Exhaustive optional-version, nonfinite-input, alias and concurrency coverage
remains outside this pass. The review does not prove absence of bugs.
