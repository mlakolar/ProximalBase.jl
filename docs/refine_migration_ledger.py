#!/usr/bin/env python3
"""Prepare reviewed parent-to-sentence mappings without overwriting the ledger.

Input JSON is a list of {block_id, destinations} records. Destinations are in
source sentence order and include retained sentences as well as moved ones.
Output gives replacement root rows and any historical upstream migration bases.
A real independent review identity@revision is mandatory; this helper does not
supply or authenticate one.
"""
import argparse
import json
import sys

import migration_checks as m


def prepare(plan, reviewer, *, root=m.ROOT, baseline=m.BASELINE):
    identity, _, revision = reviewer.partition("@")
    if not identity or identity == "mechanical" or not revision:
        raise ValueError("an actual independent review identity@revision is required")
    originals = m.baseline_blocks(root, baseline)
    by_id = {block.id: block for block in originals}
    audit = root / "docs/audit/docs-tracks-1019"
    rows = {row["block_id"]: row for row in m.read_tsv(audit / "ledger.tsv")}
    prior_bases = m.receipt_records(root, "migration-bases.json", "bases")
    upstream = m.upstream_records(list(rows.values()), root, prior_bases)
    refined = m.validated_refinements(originals, root)
    result, bases, children = [], [], []
    seen = set()
    for record in plan:
        identifier = record["block_id"]
        if identifier in seen:
            raise ValueError(f"duplicate refinement plan root: {identifier}")
        seen.add(identifier)
        old, row = by_id[identifier], rows[identifier]
        if identifier not in refined:
            raise ValueError(f"root is not in the frozen refinement map: {identifier}")
        basis = [old]
        if row["action"] == "upstream" or identifier in prior_bases:
            receipt = upstream[identifier]
            basis = m.validate_upstream(old, receipt, receipt, root, current=False)
            bases.append(dict(block_id=old.id, baseline_sha=old.sha,
                              upstream_sha=m.receipt_digest(receipt),
                              source_shas=[block.sha for block in basis],
                              evidence="Exact historical terminal span validated by the retained upstream receipt; subsequent sentence moves use ordinary preservation accounting."))
        elif row["action"] not in {"stay", "move", "split"}:
            raise ValueError(f"refine an unchanged or reviewed upstream source, not {row['action']}: {identifier}")
        pieces = [child for block in basis for child in m.sentence_children(block)]
        destinations = record["destinations"]
        if (not isinstance(destinations, list) or len(destinations) != len(pieces) or
                not all(isinstance(dest, str) and dest for dest in destinations)):
            raise ValueError(f"one destination per ordered source sentence is required: {identifier}")
        for dest in destinations:
            m.split_destination(dest)
        update = dict(row, action="split", dest=";".join(destinations),
                      dest_sha=";".join(piece.sha for piece in pieces), reviewer=reviewer)
        result.append(update)
        children.extend(dict(parent_id=identifier, child_id=piece.id, sha=piece.sha, dest=dest)
                        for piece, dest in zip(pieces, destinations))
    return dict(schema=1, baseline=baseline, ledger_rows=result,
                migration_bases=bases, children=children)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", help="JSON list with root block_id and ordered destinations")
    parser.add_argument("--reviewer", required=True, help="actual independent identity@reviewed-revision")
    args = parser.parse_args()
    with open(args.plan, encoding="utf-8") as stream:
        plan = json.load(stream)
    print(json.dumps(prepare(plan, args.reviewer), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (KeyError, ValueError, FileNotFoundError) as error:
        print(f"Could not prepare refinement: {error}", file=sys.stderr)
        sys.exit(2)
