#!/bin/bash
# Regenerates the tables of REPORT.md sections 8.7 and 8.8 from results/m3/probe{1,2,3,5,6}.
cd "$(dirname "$0")/.."
python3 bench/offset_summary.py results/m3/probe2 results/m3/probe5 > notes/tables/probe-offsets.md
python3 bench/offset64.py results/m3 > notes/tables/probe-offset64.md
python3 bench/offset64.py results/m3 --report | sed -n '/CPU, Julia/,$p' > notes/tables/report-offset64.md
python3 bench/offset_band.py results/m3/probe2 results/m3/probe5 > notes/tables/probe-band.md
python3 bench/sparse_digest.py results/m3/probe3 > notes/tables/probe-sparse-own-arrays.txt
python3 bench/sparse_digest.py results/m3/probe6 > notes/tables/probe-sparse-shared-arrays.txt
python3 bench/offset_self.py results/m3/probe5 > notes/tables/report-offset-self.md
python3 bench/offset_band.py results/m3/probe7 --families=l1g > notes/tables/probe-band-b925.md
