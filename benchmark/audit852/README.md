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
python3 benchmark/audit852/run.py --julia "$JULIA" --output /tmp/pb852-followup --rounds 21 --start-round 8 --persistent --cases-file /tmp/pb852-broad-summary/suspicious.txt
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
round draws deterministic input IDs shared by all versions. The follow-up keeps
six workers alive for 21 rounds while reconstructing every fixture with fresh
seeded ID schedules and held allocation padding. Repeat selected cases without
--persistent for additional fresh-process investigation. `--filter REGEX` and
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
