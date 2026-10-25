#!/usr/bin/env python3
"""Compare matching seeded runs of two baseline directories, within Julia version."""
import argparse
import csv
import glob
import json
import math
from pathlib import Path
import statistics
import tomllib
from validate import MATCH_SETTINGS, validate

def load(root):
    rows, configs = {}, set()
    versions = sorted({p.name.removeprefix("julia-").split("-rep")[0]
                       for p in Path(root).glob("*-metadata.toml")})
    if not versions:
        raise ValueError("No process metadata")
    checked = validate(root, versions=versions)
    if not checked["valid"]:
        raise ValueError(f"Invalid baseline {root}: {checked['issues']}")
    for path in glob.glob(str(Path(root)/"*-metadata.toml")):
        with open(path,"rb") as stream:
            m=tomllib.load(stream)
        configs.add(tuple(str(m[key]) for key in MATCH_SETTINGS))
    for path in glob.glob(str(Path(root)/"*-metrics.tsv")):
        version=Path(path).name.split("-rep")[0]
        with open(path,newline="") as stream:
            for r in csv.DictReader(stream,delimiter="\t"):
                key=(version,r["id"],int(r["rep"]))
                if key in rows: raise ValueError(f"duplicate measurement: {key}")
                rows[key]=r
    return rows, configs

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before");parser.add_argument("after");parser.add_argument("output")
    args=parser.parse_args()
    before,old_configs=load(args.before);after,new_configs=load(args.after)
    if old_configs!=new_configs:
        raise SystemExit("Benchmark configuration differs; establish comparable settings before claiming a speedup")
    groups={}
    for key in sorted(before.keys() & after.keys()):
        old,new=before[key],after[key]
        # Same scenario and repetition within the same Julia version: paired data.
        groups.setdefault(key[:2],[]).append((int(old["rep"]),float(old["mean_ns"]),float(new["mean_ns"])))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    fields=["julia","id","paired_runs","before_mean_ns","after_mean_ns","paired_geometric_speedup","ci95_low","ci95_high"]
    with open(out/"comparison.csv","w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader()
        for (version,case),pairs in sorted(groups.items()):
            if {rep for rep, _, _ in pairs} != set(range(1, 11)):
                raise ValueError(f"Incomplete paired repetitions: {version}/{case}")
            log_ratios=[math.log(old/new) for _,old,new in pairs]
            mu=statistics.mean(log_ratios)
            half=2.262*statistics.stdev(log_ratios)/math.sqrt(10)
            writer.writerow(dict(julia=version,id=case,paired_runs=10,
                before_mean_ns=statistics.mean(old for _,old,_ in pairs),
                after_mean_ns=statistics.mean(new for _,_,new in pairs),
                paired_geometric_speedup=math.exp(mu),ci95_low=math.exp(mu-half),ci95_high=math.exp(mu+half)))
    (out/"coverage.json").write_text(json.dumps({"shared_measurements":len(before.keys() & after.keys()),
        "before_only":len(before.keys()-after.keys()),"after_only":len(after.keys()-before.keys()),
        "note":"Ratios above one favor after; interval is paired on log ratios; no automatic cross-hardware or cross-Julia comparison"},indent=2)+"\n")
