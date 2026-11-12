# Shared `dropzeros!` loop (#834)

Paired campaign for `dropzeros!` on `SparseIterate` and `SymmetricSparseIterate`, head against
pinned master. Each round starts three fresh sequential Julia processes, pinned to one CPU: the
baseline, the candidate, and an independently started second baseline (the A/A control). The role
order rotates so every role takes every position once in three rounds, the case order reverses
every round, and seeds vary by round and match across roles. Each case's generator is seeded with
the first 8 bytes (big-endian) of SHA-256(`dropzeros834|SEED|case`), never with Julia's `hash`.
Every output row carries a SHA-256 fingerprint of the case's fixtures (coordinate IDs and the exact
bits of the values). `run.py` stops the campaign when the roles of a round differ in any
fingerprint, and `analyze.py` checks again. Fingerprints are comparable only within one Julia version,
whose random streams they reflect. Source trees are loaded explicitly
from archived `src` directories; `campaign.json` records their digests, the harness hashes, the
Julia binary hash, the benchmark manifest hash and `lscpu`.

Cases: both iterate types, Float32 and Float64, 64, 1024 and 16384 active coordinates out of eight
times as many coordinates, and

- `zeros0`, `zeros10`, `zeros50`, `zeros90`, `zeros100`: that share of the working set holds ±0
  (0 is the zero-free call, 100 the all-zero clear);
- `fill_drop`: `fill!(coordinates(x), 0); dropzeros!(x)` from a zero-free working set;
- `maintain10` (s = 1024 and 16384): one cycle of working-set maintenance through the public API:
  a write to every active coordinate (10% become ±0), nonzero writes that activate 10% new
  coordinates, then `dropzeros!`. The time inside `dropzeros!` is also accumulated with
  `time_ns`, so its share of the cycle is reported;
- `reset`: the restoration alone, which runs untimed before every other sample.

Every working set is a random subset of the coordinates in random order. Each case cycles through
32 patterns, so successive calls see different zero patterns. One sample times
`max(1, 4096 ÷ s)` calls on separate iterates (one evaluation per sample), and times are reported
per call; restoration is outside the timed region. The check mode runs bounds-checked and compares
every call with an independent copy of the previous loop (count, live values and IDs in order, the
reverse map).

`check_fixtures.py` is the regression check of the fixtures. It generates them in four fresh
processes (baseline, candidate, baseline, candidate) for two seeds, requires one fingerprint per
seed and case and a different one for the other seed, and recomputes the seed derivation in Python.

```sh
mkdir -p /tmp/dz834-base /tmp/dz834-head
git archive BASE_SHA src | tar -x -C /tmp/dz834-base
git archive HEAD_SHA src | tar -x -C /tmp/dz834-head
JULIA=/path/to/julia
python3 benchmark/dropzeros834/run.py --julia "$JULIA" --baseline /tmp/dz834-base --candidate /tmp/dz834-head --output /tmp/dz834-check --check --rounds 1 --cpu 8
python3 benchmark/dropzeros834/check_fixtures.py --julia "$JULIA" --project /tmp/dz834-check/env --baseline /tmp/dz834-base --candidate /tmp/dz834-head
python3 benchmark/dropzeros834/run.py --julia "$JULIA" --baseline /tmp/dz834-base --candidate /tmp/dz834-head --output /tmp/dz834-time --rounds 12 --cpu 8
python3 benchmark/dropzeros834/analyze.py /tmp/dz834-time --output /tmp/dz834-summary
```

`analyze.py` reports per case the median over rounds of the per-round ratio of per-call medians,
with a seeded 10,000-resample 95% bootstrap interval over rounds, for head/base and for the A/A
control, the absolute times, the dispersion over rounds and the allocations. A case is flagged
`investigate` when its median ratio exceeds 1.05 and the interval's lower bound exceeds 1.0.

`diagnose.jl SOURCE OUTPUT_DIR` writes the optimized LLVM IR of `dropzeros!` for each type and
element type, with type names and debug information removed, so that two trees can be diffed.

`attribution.jl BASE_SRC HEAD_SRC OUTDIR ORDER ROUNDS` is the bounded attribution of a flagged
cell. It loads both trees into one process (`ORDER` is `base-first` or `head-first`), times them
interleaved on the same fixtures, and writes the native code of `dropzeros!` for both trees.
When LinuxPerf.jl is in the environment, it also writes hardware counters: 2000 restore + call
cycles minus 2000 restores.
