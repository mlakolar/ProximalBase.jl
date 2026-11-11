# Historical iterate audit #852

This directory is independent of the existing benchmark harness and frozen
baselines. The [specification](../../docs/audit/iterate-performance-852/history-and-spec.md)
records source selection, compatibility and prior infrastructure.

Use Linux, Python >=3.11 and **Julia 1.13.3**. A regular Git clone must contain
the four commits in spec.py. Run from the repository root. For example:

```sh
export JULIA_DEPOT_PATH=/tmp/pb852-depot
JULIA=/absolute/path/to/julia-1.13.3/bin/julia
"$JULIA" --startup-file=no --project=benchmark/audit852 -e 'using Pkg; Pkg.instantiate()'
python3 benchmark/audit852/run.py --julia "$JULIA" --output /tmp/pb852-check --check --rounds 1 --roles original,historical05,post848,post850
python3 benchmark/audit852/run.py --julia "$JULIA" --output /tmp/pb852-broad --rounds 7
python3 benchmark/audit852/analyze.py /tmp/pb852-broad /tmp/pb852-broad-summary --verify
python3 benchmark/audit852/run.py --julia "$JULIA" --output /tmp/pb852-followup --rounds 21 --start-round 8 --cases-file /tmp/pb852-broad-summary/suspicious.txt
python3 benchmark/audit852/analyze.py /tmp/pb852-followup /tmp/pb852-followup-summary --verify
```

`--cpu N` selects one allowed logical CPU (default: lowest allowed). All workers
run sequentially on that CPU, including compilation, and BLAS/Julia use one
thread. Do not run unrelated CPU-heavy jobs during timing. `--sources DIR`
controls the archive location (default /tmp/pb852-frozen); a stored SHA marker
prevents silently reusing a directory labeled as another source. The runner
records source digests; for an independent replay use a fresh sources directory.
Source trees are loaded unchanged in separate Main.ProximalBase processes with
identical -O2, startup and bounds flags; correctness uses --check-bounds=yes,
runtime --check-bounds=no. No historical dependency resolution is required:
only unchanged source is loaded under the common locked harness environment.

The broad run creates six fresh workers **per round**. OriginalAA and post850AA
are independent same-source controls. Version/case order alternates; each
round draws deterministic input IDs shared by all versions. The reported follow-up starts six fresh workers in each of 21 rounds and
reconstructs every fixture with seeded ID schedules and held allocation padding.
The optional --persistent mode is exploratory and was not used for these
headline follow-up estimates. `--filter REGEX` and
--cases-file select operations **before** building fixtures. No output campaign
can be overwritten. N/A cases have an explicit status and reason; unexpected
correctness failures are retained and fail the campaign.

Every operation has two independent-reference checks before timing. Three
operation warmups and a preliminary trial compile its timing wrapper. A timed
trial requests at most 80 samples / 8 ms, evals=1, with state restoration outside
timing (BenchmarkTools may exceed its time budget to complete an evaluation).
80 is a cap, not a minimum. Each trial retains individual time/GC samples and
median/minimum/p90, heap bytes/counts. Allocation estimates are BenchmarkTools'
trial estimates; reachable object size is measured separately by summarysize.
The warm_seconds and source_load_seconds fields are first-use/warmup diagnostics,
not pure compilation costs. Timer calibration is retained at process start/end.

analyze.py uses only the Python standard library. It reports the median of
paired per-round ratios and a seeded 10,000-resample percentile bootstrap 95%
interval, resampling rounds rather than individual timing samples. Seven rounds
provide limited tail resolution. Intervals are descriptive, unadjusted for the
many comparisons; A/A, absolute costs, sequence timings and repeated controls
are needed for interpretation. Every initial >5% median (primary or extraction)
is selected for follow-up without a significance filter. `--verify` recomputes
trial medians/minima from individual samples and checks record uniqueness,
compatibility consistency and complete version pairing. No old-machine timings
enter new ratios. Work counts describe logical updates/reads or active product
multiply-adds; constructor work counts describe problem size.

For plots, optionally install requirements-plot.txt in a virtual environment and
pass --plots to analyze.py. It writes standalone SVG and PNG scaling figures.
Raw observations, provenance, logs, fixture pointers and individual samples
are packaged by pack.py; SHA256SUMS pins the archive and source artifacts.

Supplemental shape-adapted and rebuilding workloads use a generated copy:

```sh
python3 benchmark/audit852/supplement.py --harness /tmp/pb852-supplement-harness --sources /tmp/pb852-frozen
python3 /tmp/pb852-supplement-harness/run.py --julia "$JULIA" --sources /tmp/pb852-frozen --output /tmp/pb852-supplement-check --check --rounds 1 --roles original,historical05,post848,post850
python3 /tmp/pb852-supplement-harness/run.py --julia "$JULIA" --sources /tmp/pb852-frozen --output /tmp/pb852-supplement --rounds 7
python3 /tmp/pb852-supplement-harness/analyze.py /tmp/pb852-supplement /tmp/pb852-supplement-summary --verify
```

The generator prearchives sources using the actual Git checkout before copying
its harness outside the repository. It adds original strided row-block matrix
value operations, copyto_shared using similar(src), and four repeated complete
zero-write/compact/reconstruct cycles. The original matrix and current vector
shapes differ explicitly; no timed conversion or fabricated activity metadata
is used. Native vector products are not compared across those shapes.

For common scalar cache conditioning, put selected primary scalar case IDs in
selection.txt and run restore_control.py with --julia, --sources, --cases-file,
--output and --rounds (default 21). It copies the harness and changes only untimed
scalar state restoration: all three full-capacity arrays are copied in the same
order, then nnz is set. The generated harness and hashes are retained alongside
its campaign. Use its analyze.py for the control's raw-to-summary replay. This
is a setup diagnostic, not a new package API measurement.

For the end-to-end alternative, use the dependency setup, correctness and broad
run commands above, then this driver instead of the manual analysis/follow-up
commands. Continue with the full control sequence below exactly once:

```sh
python3 benchmark/audit852/finish_campaign.py --broad /tmp/pb852-broad --prefix /tmp/pb852-final --julia "$JULIA" --sources /tmp/pb852-frozen
```

The supplement_controls.py command in the next block runs 28 fresh-process
native-index supplemental controls (seven plus 21 confirmation rounds), then seven hot scalar
controls. It applies direct v[1,j] access to original row-block matrix views,
while vector views retain v[j]. Initial linear-index supplemental results are
kept as a distinct API context. No timed shape conversion is added. Shared
full-state restoration and the 1-second/80-sample budget remain explicit.

restore_control.py accepts --harness-source for supplemental workloads,
--native-block-indexing for the corrected row-block access and --restore-mode
active for identical active-prefix scalar setup. Active-prefix and full-array
controls bracket hot/cold conditioning; report them separately. Raw allocation
isolation uses one constructor/import operation per fresh process, with A/A
controls and a longer budget; no equivalence claim follows from A/B near one
when same-source A/A varies strongly.

To reproduce the full set of controls after finish_campaign.py, continue with:

```sh
python3 benchmark/audit852/supplement_controls.py --prefix /tmp/pb852-final --julia "$JULIA" --sources /tmp/pb852-frozen
python3 benchmark/audit852/confirm_controls.py --prefix /tmp/pb852-final --julia "$JULIA" --sources /tmp/pb852-frozen
python3 benchmark/audit852/raw_blocks.py --prefix /tmp/pb852-final --julia "$JULIA" --sources /tmp/pb852-frozen
python3 benchmark/audit852/static_adapter_control.py --prefix /tmp/pb852-final --julia "$JULIA" --sources /tmp/pb852-frozen
python3 benchmark/audit852/backing_blas_control.py --prefix /tmp/pb852-final --julia "$JULIA" --sources /tmp/pb852-frozen
python3 benchmark/audit852/finalize_data.py --prefix /tmp/pb852-final --broad /tmp/pb852-broad --julia "$JULIA" --sources /tmp/pb852-frozen
```

The commands are sequential. --wait is available for the dependent drivers to
wait for their predecessor's verified outputs, without overlapping Julia work.
Use a fresh prefix and output tree for a new campaign. Each driver refuses to
reuse existing measurement directories. Install the plotting requirements into
the Python environment running the final driver, or use --skip-plots there and
run plot_report.py separately afterward.

The raw block variant addresses original convenience-constructor exclusions:
its existing AtomIterate(storage, atoms) holds native vector SubArray views for
both precisions and all three layouts. Storage/views are prepared outside
timing. Construction is intentionally excluded, because a prebuilt raw field
constructor would not be comparable to a full modern factory. Original group
activation and compaction remain unsupported. Raw-native, convenience-vector
and native-row-matrix contexts are labeled separately; never pool their times.

finalize_data.py waits for all timing/confirmations, then runs tighter Float64
references, nonuniform all-index layout fingerprints, generated-code and
compilation diagnostics, independent p90/provenance checks, seven/21 supplemental
partitions, the regression ledger and report exports/figures. It does not
publish, merge, or modify src/. Heavy diagnostics/plotting/packing must occur
after all measurement processes finish. Adapter_diagnostic.jl tests whether
runtime versus static API selection affects historical inference, separately
from source-kernel attribution.

Run static_adapter_control.py after raw_blocks.py and before finalize_data.py,
with the same --prefix/--julia/--sources arguments. It diagnoses all four adapter
paths, then measures corrected historical05 block attribution with static API
selection and equal retained-zero copy preparation. --wait serializes it after raw block
measurements. All original observations remain available; source_attribution
in the exported summaries/all-comparisons.csv and adapter-exclusions.csv identify
superseded historical block contrasts, including unequal native-clear copy
preparation. This diagnostic is required before final
export, and finalization also waits for its completion marker.

backing_blas_control.py then compares the original's existing storage-Vector
BLAS products with current sparse-aware owner products, without conversion or
new kernels. Current storage can also use BLAS; this available-strategy control
is separate from default-owner-API results. It uses the same seven-plus-21
protocol and must finish before finalize_data.py; --wait enforces serialization.


Published data are in [the audit artifacts](../../docs/audit/iterate-performance-852/artifacts/README.md).
That page gives checksum verification, extraction and raw-to-summary replay
commands. The [report](../../docs/audit/iterate-performance-852/report.md)
contains source selection, corrected adapter attribution, preparation contexts,
results, figures and limitations. `runtime_manifest.py JULIA OUTPUT.json`
hashes the entire installed Julia distribution (including the system image and
shared libraries) for comparison with the published runtime manifest.
