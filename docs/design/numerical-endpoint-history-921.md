# Numerical endpoint history

This is historical design and performance evidence, not a new numerical
contract. Current behavior is documented in `docs/src/numerics.md`, under
“Infinite penalties at zero step” and the function-specific endpoint table.

The infinite-level `NormL1` zero-step projection was selected in
[#665](https://github.com/mlakolar/ProximalBase.jl/issues/665). The previous
zero-step identity changed only for that endpoint. Its test is on the
exceptional kernel branch, so ordinary finite steps retain their existing loop.

The endpoint extensions were deferred to protect ordinary proximal throughput:
PR [#874](https://github.com/mlakolar/ProximalBase.jl/pull/874) measured
slowdowns of **+5.9% (`Float32`) and +9.9% (`Float64`)** for weighted array steps
at 65,536 entries after optimization, and reported **+19–31%** for weighted
scalar steps on a newer local CPU. Unlike finite-input overflow/underflow
recovery, these cases require a particular convention for a genuinely
non-finite input. No additional checks or scans are authorized in the hot
loop for #853 or #866. Reconsider #866 if a practically important finite-input
failure is demonstrated or an implementation preserves ordinary-path speed.
