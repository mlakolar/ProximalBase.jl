# Comparing full-point NormLinf scores

Use a Julia project with the frozen package dependencies instantiated. Each command
accepts a filesystem checkout. To pin the experiment, record each checkout's
full Git SHA, Julia version, architecture, thread count and optimization flags
beside the output. Examples below use an existing Julia binary on PATH.

```sh
julia --startup-file=no --project="$BASE" experiment981/check_oracle.jl
julia --startup-file=no --project="$BASE" experiment981/accuracy.jl base "$BASE" base.tsv
julia -O0 --startup-file=no --project="$BASE" experiment981/accuracy.jl base "$BASE" base-o0.tsv
julia --startup-file=no --project="$BASE" experiment981/timing.jl base "$BASE" other "$OTHER" pair.tsv 6
julia --startup-file=no --project="$BASE" experiment981/timing.jl base-A "$BASE" base-B "$BASE" aa.tsv 6
julia --startup-file=no --project="$BASE" experiment981/reads.jl other "$OTHER" reads.tsv
```

## Accuracy

Score references use the public working type to establish the exact face and
gradient entries. For a nonzero point, the oracle groups the signed values
`u = -sign(xᵢ)vᵢ` on the face, and adds off-face squared residuals. At zero,
it groups `abs(vᵢ)` and handles containment in the ball. Every subsequent
arithmetic operation uses exact binary rationals through the squared distance.
For positive radius outside the ball or on a face, the root of
`Σ kᵢ max(uᵢ-t,0) = c` lies between `min(u)-c` and `max(u)`. The oracle
checks breakpoints and bisects this interval, accepting a chord intersection
only after exact substitution into the mass equation. Finitely many breakpoints
and a piecewise affine mass guarantee finite termination. Grouping duplicates
keeps the million-entry calculation inexpensive. The independent self-check
compares with exhaustive face-support minimization on 224 small problems
and checks the round-3 distance against its analytic expression and published
decimal.

The TSV includes the exact rational square, its 256-bit square root, and the
absolute error divided by `norm(v,2)+effective_radius`. For finite references
the empirical target is `32 eps(W) (norm(v,2)+effective_radius) + 2δ`, with
`δ = nextfloat(zero(W))`. A value outside this target is marked `accuracy`
or, if its normalized error reaches 0.001, `material`. These thresholds
describe the measurement. A separate ratio uses the frozen PR's first-order
expression `(2n+√n+5) eps(W)/2 (norm(v,2)+effective_radius) + 4nδ`.
Its use on exceptional paths or outside the assumptions of that expression is
descriptive. A finite result at an exactly stationary point within the target
is flagged `roundoff_at_zero` if positive. A `pass` for a subnormal still
includes its actual error: for example the aggregate 100δ case differs by
1.5625δ if an implementation returns zero. The known range limit of the lazy
single-factor division is marked separately as `documented_range`. Contract
rows include the required exception type, return type and exceptional-value
expectation.

The count-rise family, round-3 decimal and clustered input recipe come from
the [PR discussion](https://github.com/mlakolar/ProximalBase.jl/pull/981#issuecomment-6976459290).
The two-entry count-rise case and Float64 rounded-count cases come from the
frozen regression suite and the
[review comments](https://github.com/mlakolar/ProximalBase.jl/pull/981/files).
For one million ties the public description specifies the family
`y0 = (1-c)/(1+ρm²u)` and rounded observed/reference scores, with eleven
unspecified ρ values. Here `c=.1`, `u=2^-24`, and `ρ=.06` yield
`Float32(0.00025158789092684895)`; the exact-reference distance rounds to
0.9345033240098486 and the frozen full-face call gives about 0.249 on our
host. The case ID explicitly identifies this reconstruction. Four additional
input distributions come directly from the published
[prox reproducers](https://github.com/mlakolar/ProximalBase.jl/issues/1003).
The harness evaluates scores on those distributions.

## Paired timing

Both checkouts load under their original module name in separate parent
namespaces. Each pair shares the same concrete input vectors. The first call
per checkout and element type measures dispatch and initial compilation after
source loading. Steady-state calls exclude module loading, input generation,
construction of the penalty and compilation warmup. A pilot determines a
common repeat count for both implementations; their order alternates within
pairs, and the starting side alternates across cases. Use an even repetition
count to balance the order. The checksum accumulates results inside each timed
batch, and allocated bytes measure a separate warm call. `phase=warm` gives
raw samples; `median` gives each side's median wall time in ns per call. Use
matching replicate numbers for paired ratios, and use A/A to estimate noise
for the same workload. A/A uses two independent loads of the same checkout.

`N981_SIZES` accepts a comma-separated subset of the default sizes
`2,8,32,256,4096,100000,1000000`. `N981_TARGET_NS` sets the target per batch
(250000 by default). Longer batches improve stability for short calls. First-call
timings include Julia compilation and can depend on which side runs first.

## Input traversal

The separate reads script counts public gradient and point scalar reads using
instrumented AbstractVector inputs. Columns `v_reads_per_n` and `x_reads_per_n`
give complete-input-scan equivalents. Reads from scratch buffers and operations
on an already materialized face are invisible here. This measures input traffic,
and does not give an internal threshold iteration count. The wrapper can change
specialization, accumulation order and decisions. The TSV includes the ordinary
and instrumented values plus an equality check to expose such differences.
