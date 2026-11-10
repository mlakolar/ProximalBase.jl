"""Summarize paired PB block diagnostics without treating warm-up as evidence.

Intervals describe the collected log ratios conditional on the process cohorts;
report each cohort and A/A separately. They are not population guarantees over
machines/compiler layouts. Confirm flagged cases in a separate focused run.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def interval(ratios):
    logs = [math.log(r) for r in ratios]
    center = statistics.mean(logs)
    # Campaigns use at least ten pairs. 2.262 is t(9,.975), conservative
    # for campaigns with more observations; keep an interval unavailable below 10.
    radius = 2.262 * statistics.stdev(logs) / math.sqrt(len(logs)) if len(logs) >= 10 else None
    return [math.exp(center-radius), math.exp(center+radius)] if radius is not None else None


def geometric(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def load(path):
    with path.open() as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    result = {row["case"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate case in {path}")
    return result


def summarize(directory):
    run = json.loads((directory / "run.json").read_text())
    if run.get("exit_code") != 0:
        raise ValueError("campaign is not complete")
    pairs = []
    for pair in run["pairs"]:
        rows = {label: load(directory / name) for label, name in pair["files"].items()}
        if not (rows["before"].keys() == rows["after"].keys() == rows["aa"].keys()):
            raise ValueError("case inventories differ")
        pairs.append((pair["cohort"], rows))
    if len(pairs) < 10:
        raise ValueError("acceptance summary needs at least ten pairs")
    result = []
    for key in pairs[0][1]["before"]:
        values = {label: [float(rows[label][key]["median_ns"]) for _, rows in pairs]
                  for label in ("before", "after", "aa")}
        ratios = [b/a for a,b in zip(values["before"], values["after"])]
        aa = [b/a for a,b in zip(values["before"], values["aa"])]
        vs_control = [b/a for a,b in zip(values["aa"], values["after"])]
        cohorts = {str(c): geometric([ratio for (cohort,_),ratio in zip(pairs,ratios) if cohort == c])
                   for c in sorted({c for c,_ in pairs})}
        ci, control_ci = interval(ratios), interval(vs_control)
        flag = (ci[0] > 1.05 and control_ci[0] > 1.05
                and all(r > 1.05 for r in cohorts.values())
                and sum(r > 1.05 for r in ratios) >= math.ceil(.8*len(ratios)))
        result.append(dict(case=key, pairs=len(pairs),
            before_median_ns=statistics.median(values["before"]),
            after_median_ns=statistics.median(values["after"]),
            median_paired_ratio=statistics.median(ratios), geometric_paired_ratio=geometric(ratios),
            conditional_log_t_interval95=ci, aa_geometric_ratio=geometric(aa),
            aa_conditional_interval95=interval(aa), candidate_vs_control_ratio=geometric(vs_control),
            candidate_vs_control_interval95=control_ci, cohort_geometric_ratios=cohorts,
            paired_ratio_range=[min(ratios),max(ratios)], paired_ratios=ratios, aa_ratios=aa,
            candidate_bytes=max(int(rows["after"][key]["bytes"]) for _,rows in pairs),
            candidate_allocations=max(int(rows["after"][key]["allocations"]) for _,rows in pairs),
            repeatable_regression_candidate=flag))
    return dict(method=__doc__, source_run="run.json", paired_repetitions=len(pairs),
                case_count=len(result), flagged_cases=sum(r["repeatable_regression_candidate"] for r in result),
                cases=result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.directory)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['case_count']} cases, {report['paired_repetitions']} pairs; "
          f"{report['flagged_cases']} cases require focused confirmation")
