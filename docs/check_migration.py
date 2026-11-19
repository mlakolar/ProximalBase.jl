#!/usr/bin/env python3
"""Run six named preservation checks; report-only until the migration locks.

Examples:
  python3 docs/check_migration.py --init --baseline-html /tmp/pb-baseline-html
  python3 docs/check_migration.py --check all --build docs/build
  python3 docs/check_migration.py --blocking --scope-base origin/master
"""
from pathlib import Path
import argparse
import json
import sys

import migration_checks as m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", action="store_true", help="create an all-stay ledger; refuses overwrite")
    parser.add_argument("--baseline-html", type=Path)
    parser.add_argument("--build", type=Path, default=m.ROOT / "docs/build")
    parser.add_argument("--check", choices=("all", "blocks", "tokens", "sentences", "anchors", "citations", "examples"), default="all")
    parser.add_argument("--blocking", action="store_true", help="return nonzero for findings (default: report-only)")
    parser.add_argument("--scope-base", help="PR merge base for docs-only src/test scope; never use the frozen docs baseline for this")
    parser.add_argument("--apply-moves", type=Path, help="apply explicit unique frozen-source spans from a JSON move manifest")
    args = parser.parse_args()
    if args.init:
        if args.baseline_html is None:
            parser.error("--init requires --baseline-html from a successful unchanged build")
        print(json.dumps(m.initialize(args.baseline_html), indent=2))
        _, citations = m.citations_check()
        m.write_tsv(m.AUDIT / "source-citations.tsv", ("path", "line", "heading"), citations)
    metadata = json.loads((m.AUDIT / "baseline.json").read_text())
    baseline = metadata["baseline"]
    if args.apply_moves:
        count = m.apply_moves(json.loads(args.apply_moves.read_text()), baseline=baseline)
        print(f"Recorded {count} exact moved baseline blocks.")
    base_blocks = m.baseline_blocks(baseline=baseline)
    rows = m.read_tsv(m.AUDIT / "ledger.tsv")
    errors, mappings = m.coverage(base_blocks, rows)
    checks = {
        "blocks": lambda: errors,
        "tokens": lambda: m.accounting(base_blocks, mappings, "tokens"),
        "sentences": lambda: m.accounting(base_blocks, mappings, "sentences"),
        "anchors": lambda: m.anchor_check(m.read_tsv(m.AUDIT / "anchors.tsv"), args.build),
        "citations": lambda: m.citations_check(baseline=baseline, scope_base=args.scope_base)[0],
        "examples": lambda: m.examples_check(base_blocks, mappings, metadata),
    }
    results = {}
    for name, check in checks.items():
        if args.check not in ("all", name):
            continue
        findings = check()
        results[name] = {"status": "FAIL" if findings else "PASS", "findings": findings}
    print(json.dumps(dict(mode="blocking" if args.blocking else "report-only", baseline=baseline,
                         baseline_blocks=len(base_blocks), ledger_rows=len(rows), checks=results), indent=2))
    failed = any(result["findings"] for result in results.values())
    return 1 if args.blocking and failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, KeyError) as error:
        # Configuration failures cannot produce a credible report, even during
        # report-only migration. Actual preservation findings are printed above.
        print(f"Migration checker could not run: {error}", file=sys.stderr)
        sys.exit(2)
