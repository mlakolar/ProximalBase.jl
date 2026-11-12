# Independent review record

An independent Codex GPT-6.1-Sol agent was explicitly requested at high effort,
with no inherited conversation history. It received the frozen baseline and
candidate paths, the issue constraints, tests and measurement harness. Review
was read-only; CPU-heavy checks were held during benchmark windows.

Findings addressed during development:

* A development candidate included the concurrent #861 changes from the shared
  checkout. That campaign was excluded. Eligible candidate trees were rebuilt
  exclusively from the frozen master source, and the production diff was
  checked to exclude whole-iterate state copying.
* The initial analyzer could silently drop missing source pairs. It now requires
  complete requested role/round/common-case coverage, matching fixture seeds,
  source identities/hashes, Julia version/binary and compiler/thread settings,
  and recomputed raw sample medians. The runner retains the executed harness
  snapshot and checks source immutability.
* Custom index arrays can run callbacks during index or alias-metadata access.
  They retain Base's copy path, including singleton copies. Independent tests
  cover throwing index reads, partial writes and alias-metadata exceptions.
* A scalar-prefix path must respect a view's first index. An offset-axis dense
  reference test protects this behavior; custom descriptors use Base's path.
* The bulk loop's aliasing and SIMD proof was checked: same-type Float32/64
  Vector reads and stores cannot run user conversions; `Base.unalias` precedes
  mutation; stored group indices are valid and unique; OR reduction correctly
  handles signed zero, NaN and Inf. An inactive untouched suffix is already
  zero, so prefix nonzero detection is sufficient.
* Zero-commit full BCD cases were added alongside isolated zero-copy sweeps,
  with exact-zero assertions in addition to numerical reference tolerances.

Independent generated-code review confirmed that residual-update arithmetic
remains vectorized in both baseline and bulk versions. The general bulk loop
expands the caller's code, but the heavy main-vector-body spill annotations
cannot be assigned to b8, which instead runs the four-lane epilogue. Listings
support a code-shape tradeoff, not a uniquely identified executed spill/cache
cause. The outlined-helper variant changes only the executable inlining
annotation and measures that combined compiler-boundary effect.

The singleton-only fallback candidate was independently reviewed with no
correctness or scope finding: it uses one existing owner-backed scalar write,
while longer buffers and custom descriptors retain Base's copy path. The final confirmation rejects all three candidates under the declared gate.
Independent verification reproduced all 105 process files, 60,480 observations,
4,837,587 timing samples and every exported bootstrap endpoint. The singleton
fallback caller retains per-group length guards; both versions make one copy
call on the b8 path. No unique allocation/spill cause is asserted. Final
artifact and CI receipts are linked from PR #867.
