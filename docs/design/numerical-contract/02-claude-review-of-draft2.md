# Review of the reader's edition (Draft 2)

*Claude, 2026-10-31. Reviews `01-chatgpt-readers-edition.md` against the
code and the issue tracker at master `3f90a29`.*

## Worth keeping

- **Guarantees per result.** Draft 2 attaches promises to the value, each
  output entry and each zero or threshold decision, not to whole functions.
  This matches #366's per-oracle guarantees (C1).
- **Acceptance cases built from exact powers of two.** Run locally at
  `3f90a29`:

  | case | result at `3f90a29` |
  |:---|:---|
  | V1, V2 (values whose intermediates overflow or underflow) | correct |
  | V3 (`SqrNormL2(2^160)`, step `2^-160`, `Float32[1]`) | `NaN`; Draft 2 wants `0.5f0` |
  | V4, V6 (ridge factor too small to store) | `0`; Draft 2 wants `2^-60` and `2^-700` |
  | V5 (ridge gradient with `λ = 2^160` on Float32) | gradient `Inf`; Draft 2 wants `2^80` |
  | V7 (centered ridge, `x - c` overflows) | gradient `Inf`; Draft 2 wants `[4.0]` |
  | V8 (`NormL1(1 - 2^-30)` at `Float32[1]`) | `0`; Draft 2 wants `2^-30` |
  | V9 (value at the stored Float32 point) | `2^-57`, as Draft 2 wants |
  | S1, S4, S7, S8, logistic tail at `η = 40` | as Draft 2 wants |

- **A bug nothing tracks.** `prox(NormL2(1.0), [NaN, 4.0], 1.0)` returns
  `[0, 0]`, because the NaN norm fails the `norm > threshold` test and the
  code takes the zero branch. `GroupNormL2` returns `[0, 3]`.
- **The solver section, and the test method:** high-precision references
  evaluated on the stored binary operands, and exact checks for zeros.

## Problems

1. **It reverses #238 without saying so.** Its promise to work "with the
   numbers you actually supplied", and its case V8, need a Float64 λ to be
   treated as exact on Float32 data. #238 and #366 (C2) say the parameter
   converted to the working precision defines the problem, so V8 returns `0`
   on purpose today.
2. **It is not one rule.** It keeps the spectral NaN-to-`Inf` value as a
   special exception, so `SqrNormL2(0)` at NaN gives `0` while
   `NuclearNorm(0)` gives `Inf`; master does exactly that. Its stronger
   accuracy promise ("tier B") covers four named functions and leaves the
   rest undefined. Both conflict with the package-wide rule the maintainer
   wants.
3. **It leaves out most of #366:**
   - Float16;
   - accumulators and SIMD reassociation;
   - certified decisions in the multiplier searches (#151, #257);
   - tolerance policy, including the fixed 1e-10 and 1e-8 defaults;
   - the error scale of LAPACK results;
   - wider destinations (#362);
   - BigFloat caches;
   - diagonal metrics that do not fit the working type.

   These are what the paused pull requests wait on.
4. **It restates rules owned by open issues without linking them:** #311
   (steps), #367 (aliasing), #349 (indicators at non-finite inputs). It calls
   the ridge limits at `γ = 0` and `γ = Inf` "proposed", but master already
   implements them and the interface page documents "a limit only where it is
   unique".
5. **It mixes four kinds of text:** a tutorial, a design contract, a brief for
   a planning agent (Part IV), and a map of rule codes (N1–B5) defined in a
   Draft 1 that is not available. As written, it cannot serve as a
   documentation page.

## Trackers

- #283: overflow in values, with parts #356–#360.
- #361: proximal outputs that lose a representable result.
- #311 steps; #367 aliasing; #349 and #350 non-finite inputs; #366 precision.

Nothing tracks:

- the `NormL2` and `GroupNormL2` NaN-to-zeros results;
- `SqrNormL2` cases V3–V7;
- `SqrNormL2(0.0)` returning a NaN gradient at `[NaN]` while its value is
  `0`;
- the LAPACK-based proximal operators checking only the output's element
  type.

Draft 3 (`03-claude-draft3.md`) is the revision that answers these points.
