# Corrected-state PB-only follow-up

All runs here use the corrected driver: values are zeroed, `dropzeros!` clears
activity, group 2 is activated, and group 1 is activated only for active probes.
Assertions outside timing validate the starting state after repeated writes.
Earlier `inactive_nonzero` records elsewhere in this archive are superseded;
older inactive-zero/active-zero findings remain credible.

The `active-cutoff-110` trial at PB `1c2205da` uses an active-only writer with
cutoff 16. Ten pairs with A/A found fallback overhead; that trial is rejected.
Baseline source is `a9f53e6e`. Per-run manifests pin exact candidate commits.
Raw files are gzip-compressed losslessly; manifests hash the uncompressed data.
Decompress TSV files into a separate directory before running the summarizer.
Only PB projects are loaded; no ProximalSolvers dependency or execution occurs.
