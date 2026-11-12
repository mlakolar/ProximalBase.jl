# Block value updates (#860)

This extends the retained #852 harness. It reuses its layout helpers, scalar
residual BCD kernel, soft threshold and reference tolerances. Each source loads
unchanged in its own fresh Julia process. `residual_bulk` is a separately labeled
complete BCD workload using the existing documented buffer/copy idiom. It is
checked against independent dense block algebra and the residual identity.
`residual_bcd` retains #852's interleaved scalar coefficient/residual updates.
The `_zero` variants use a high penalty so complete BCD commits zero blocks;
they test whether zero-copy overhead affects full residual workloads.

Run on Linux / Julia 1.13.3 with one Julia and BLAS thread. Prepare the locked
environment and immutable source trees from a full clone. The experimental
patches failed the acceptance gate; apply them only to isolated source trees:

```sh
JULIA=/absolute/path/to/julia-1.13.3/bin/julia
export JULIA_DEPOT_PATH=/tmp/pb860-depot
"$JULIA" --project=benchmark/audit852 -e 'using Pkg; Pkg.instantiate()'
mkdir -p /tmp/pb860-baseline
git archive eb74a914d779371a02e4a6637eaa724aeb66fdf5 src | tar -x -C /tmp/pb860-baseline
for variant in singleton bulk-inline bulk-outlined; do
  mkdir -p "/tmp/pb860-$variant"
  cp -a /tmp/pb860-baseline/src "/tmp/pb860-$variant/"
  patch -d "/tmp/pb860-$variant" -p1 < "docs/audit/block-updates-860/patches/$variant.patch"
done
python3 benchmark/audit860/run.py --julia "$JULIA" --cpu 2 \
  --output /tmp/pb860-check --check --rounds 1 \
  --role base=/tmp/pb860-baseline --role candidate=/tmp/pb860-singleton \
  --role outlined=/tmp/pb860-bulk-outlined --role bulk=/tmp/pb860-bulk-inline
python3 benchmark/audit860/run.py --julia "$JULIA" --cpu 2 \
  --output /tmp/pb860-broad --rounds 7 \
  --role base=/tmp/pb860-baseline --role baseAA=/tmp/pb860-baseline \
  --role candidate=/tmp/pb860-singleton --role outlined=/tmp/pb860-bulk-outlined
python3 benchmark/audit860/analyze.py /tmp/pb860-broad --output /tmp/pb860-broad.csv
python3 benchmark/audit860/run.py --julia "$JULIA" --cpu 2 \
  --output /tmp/pb860-confirm --rounds 21 --start 101 \
  --role base=/tmp/pb860-baseline --role baseAA=/tmp/pb860-baseline \
  --role candidate=/tmp/pb860-singleton --role outlined=/tmp/pb860-bulk-outlined \
  --role bulk=/tmp/pb860-bulk-inline
python3 benchmark/audit860/analyze.py /tmp/pb860-confirm --output /tmp/pb860-confirm.csv
```

Use an allowed CPU and keep it fixed within a campaign. Source and case order
rotate; deterministic fixture seeds change each round and match across roles.
Seeded allocation padding is held through each fixture, and storage/design/
residual/buffer pointers are recorded for placement controls.
Both designs are constructed before numerical-kernel warmup. Source loading,
fixture compilation and prior fixtures can still affect allocator state;
padding and pointers expose that sensitivity rather than guarantee placement.
The runner snapshots its harness, records source/harness hashes and machine
information, and checks source immutability at the end. The analyzer requires
all requested role/round/case pairs and matching seed/compiler/source provenance,
and recomputes medians from individual samples. It reports the median of paired
process ratios and descriptive, unadjusted 95% round-bootstrap intervals.
Run A/A analysis with `--numerator baseAA --denominator base`. Analyze the
other strategies with `--numerator outlined` or `--numerator bulk`.
Seven-round estimates are exploratory; confirmation
uses 21 new fixture seeds and processes for the entire matrix, including every
initial >5% flag. Do not pool those rounds silently.

Every timed mutation uses `evals=1`, up to 80 samples / 0.1 seconds after three
warmups and a separate wrapper warmup. Complete storage, activity arrays and
residuals are restored outside timing, independently of production state-copy
methods. Setup, reference calculations, layouts, designs and buffers are not
timed. Copy sweeps distinguish already active, first activation, inactive zero,
active written zero and pre-existing active zeros. Raw/once sweeps are diagnostic
controls: raw bypasses activity; once validates pre-existing activity per block.
Neither is a public API proposal or an equivalent inactive-group implementation.

The matrix contains Float32/64, n8192/65536, b1/8/64 and contiguous, strided,
arbitrary and reversed layouts at 10% of groups. It is synthetic fixed-schedule
BCD, with no claim about solver convergence or full ProximalSolvers solve time.
Allocation observations count Julia heap allocation, not native allocation/RSS.
The host is not exclusively reserved or frequency locked; report A/A dispersion.

After timing, capture typed IR, LLVM and native code for both sources:

```sh
"$JULIA" --startup-file=no --threads=1 --check-bounds=no -O2 \
  --project=benchmark/audit852 benchmark/audit860/codegen.jl \
  /tmp/pb860-baseline /tmp/pb860-code-base
"$JULIA" --startup-file=no --threads=1 --check-bounds=no -O2 \
  --project=benchmark/audit852 benchmark/audit860/codegen.jl \
  /tmp/pb860-bulk-outlined /tmp/pb860-code-outlined
"$JULIA" --project=. -e 'using Pkg; Pkg.test(; julia_args=["--check-bounds=yes"])'
```

The historical original can be supplied as another role after archiving
bd0740d0ebbebcc76fe208e4318fe6fc5451d34b. The harness uses its existing raw
AtomIterate constructor and cached vector views, prepared outside timing,
because its convenience constructor lacks Float32/arbitrary-layout support.
Original activity checks are explicitly unavailable. Keep historical diagnostic
ratios separate from current baseline/candidate acceptance results.
