#!/usr/bin/env python3
"""Independent summaries from the raw experiment tables."""
import argparse
import csv
import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', newline='') as inp:
        yield from csv.DictReader(inp, delimiter='\t')


def numerical(paths):
    summary = defaultdict(lambda: defaultdict(int))
    for path in paths:
        for row in read(path):
            state, sign, mode = row['sign_state'], row['exact_sign'], row['design']
            if row['group'] == 'range':
                scopes = ('range', f"range/{row['type']}/{row['case']}/{row['route']}")
            else:
                scopes = ('ordinary', f"ordinary/{row['type']}",
                          f"n={row['n']}/{row['type']}", f"scale={row['scale']}/{row['type']}",
                          f"n={row['n']}/scale={row['scale']}/{row['type']}",
                          f"route={row['route']}/{row['type']}")
                if abs(float(row['margin'])) <= 1e-4 * float(row['r']):
                    scopes = (*scopes, 'near-boundary',
                              f"near-boundary/{row['type']}",
                              f"near-boundary/n={row['n']}/{row['type']}")
            for scope in scopes:
                a = summary[(scope, mode)]
                a['rows'] += 1
                a['unavailable'] += state == 'unavailable'
                a['cross'] += state == 'cross'
                a['cap_cross'] += row['cap_state'] == 'cross'
                a['feasible'] += state == 'le'
                a['infeasible'] += state == 'gt'
                a['false_feasible'] += state == 'le' and sign == 'positive'
                a['false_infeasible'] += state == 'gt' and sign in ('negative', 'zero')
                a['boundary'] += sign == 'zero'
                a['pb_false_feasible'] += row['pb_feasible'] == 'true' and sign == 'positive'
                a['pb_false_infeasible'] += row['pb_feasible'] == 'false' and sign in ('negative','zero')
                a['two_scans'] += row['scans'] == '2'
                if state != 'unavailable':
                    a['max_width'] = max(a['max_width'], float(row['width']))
                    a['max_relative_width'] = max(a['max_relative_width'], float(row['relative_width']))
            assert not (state == 'le' and sign == 'positive' or state == 'gt' and sign in ('negative', 'zero')), row
    return [{"scope": scope, "design": mode, **a} for (scope, mode), a in sorted(summary.items())]


def quantile(values, q):
    values = sorted(values)
    where = q*(len(values)-1)
    i = math.floor(where)
    return values[i] if i == len(values)-1 else values[i] + (where-i)*(values[i+1]-values[i])


def distribution(values):
    return [quantile(values, q) for q in (0.10, 0.5, 0.90)]


def timings(paths, expected, by):
    cells = defaultdict(list)
    for path in paths:
        name = path.name.removesuffix('.gz').removesuffix('.tsv')
        for row in read(path):
            key = (name, *(row[k] for k in by))
            cells[key].append(row)
    result = []
    for (name, *key), rows in sorted(cells.items()):
        controls = [float(r['aa1_ns'])/float(r['aa0_ns']) for r in rows]
        paired = [float(r['b_ns'])/float(r['a_ns']) for r in rows]
        ab = [v for v,r in zip(paired, rows) if r['order'] == 'AB']
        ba = [v for v,r in zip(paired, rows) if r['order'] == 'BA']
        assert len(ab) == len(ba) == expected, (key, len(ab), len(ba))
        a, b, delta = (distribution(xs) for xs in ([float(r['a_ns']) for r in rows],
                        [float(r['b_ns']) for r in rows],
                        [float(r['b_ns']) - float(r['a_ns']) for r in rows]))
        stable_regression = min(statistics.median(ab), statistics.median(ba)) > max(1.05, quantile(controls, .90))
        data = {"file": name, **dict(zip(by,key)), "a_ns": a, "b_ns": b, "delta_ns": delta,
                "paired": distribution(paired), "aa": distribution(controls),
                "order_medians": [statistics.median(ab), statistics.median(ba)], "flag_gt5pct": stable_regression}
        for stat in ('bytes', 'allocs'):
            for mode in ('a', 'b', 'aa'):
                data[f'{mode}_{stat}'] = sorted(set(float(r[f'{mode}_{stat}']) for r in rows))
        for column in ('prox_calls', 'strict_matched', 'active', 'sign_refine', 'cap_refine', 'parity'):
            if column in rows[0]:
                data[column] = sorted(set(r[column] for r in rows))
        result.append(data)
    return result


def replay(paths):
    result = defaultdict(lambda: defaultdict(int))
    for path in paths:
        for r in read(path):
            for k in ((path.name.split('-replay')[0],r['family'],r['case'],r['type'],r['method'],r['n']),
                      ('combined',r['family'],'all',r['type'],r['method'],r['n']),
                      ('combined',r['family'],'all','any','any','any'),
                      ('combined','all','all','any','any','any')):
                a = result[k]
                a[r['kind']] += 1
                if r['kind'] != 'trial' or r['active'] != 'true':
                    continue
                a['active_trials'] += 1
                a['pb_reject'] += r['pb_feasible'] != 'true'
                a['positive_gap'] += float(r['margin']) > 0
                a['excluded_cap20'] += r['exact_pass20'] != 'true'
                a['excluded_cap40'] += r['exact_pass40'] != 'true'
                a['legacy_refines'] += r['legacy_refine20'] == 'true'
                a['legacy_refuses_after_check'] += r['legacy_pass20'] != 'true'
                for query in ('sign','cap20','cap40'):
                    for ar in 'ABC':
                        state = r[ar + '_' + query]
                        a[f'{ar}_{query}_{state}'] += 1
    return [{"file": k[0], "family": k[1], "case": k[2], "type": k[3], "method": k[4], "n": k[5], **v}
            for k,v in sorted(result.items())]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['numerical','micro','solve','replay'])
    p.add_argument('file', type=Path, nargs='+')
    p.add_argument('--pairs-per-order', type=int, default=8)
    a = p.parse_args()
    if a.mode == 'numerical':
        data = numerical(a.file)
    elif a.mode == 'replay':
        data = replay(a.file)
    elif a.mode == 'micro':
        data = timings(a.file, a.pairs_per_order, ['type','n','case','pair'])
    else:
        data = timings(a.file, a.pairs_per_order, ['family','type','n','case','method','pair'])
    print(json.dumps(data, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
