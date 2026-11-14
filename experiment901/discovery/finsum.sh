#!/bin/bash
# finsum.sh <Y label> <X label>   (run dirs fin/runs/final-<Y>-vs-<X>-rep{1,2}-{fwd,rev}; fwd = X first)
H=/home/mkolar/projects/juliaPkg/pb901x/harness/experiment901
# FIN: directory holding runs/ (fetch_run.sh output); summaries go to $FIN/conv
Y=$1; X=$2; cd "${FIN:-$(dirname "$0")/fin}"; mkdir -p conv
for h in cold norms l1inf l1under linsparse core; do
  runs=()
  for r in rep1 rep2; do for d in fwd rev; do
    P=runs/final-$Y-vs-$X-$r-$d
    [ -d $P ] || continue
    if [ $h = cold ] || [ $h = norms ]; then dirs=$(ls -d $P/${h}_s* | grep -v '\.log$' | paste -sd,); else dirs=$P/$h; fi
    if [ $d = fwd ]; then A=$X; B=$Y; else A=$Y; B=$X; fi
    o=conv/$Y-$X-$h-$r-$d; python3 -I $H/cold2screen.py "$dirs" $A $B $o; runs+=($o)
  done; done
  python3 -I $H/summarize901.py --num $Y --den $X --also "$X#aa,$Y#aa" --csv conv/$Y-$X-$h.csv --top 15 "${runs[@]}" > conv/$Y-$X-$h.txt
done
