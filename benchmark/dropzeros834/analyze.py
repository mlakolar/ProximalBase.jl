#!/usr/bin/env python3
"""Paired analysis of #834 campaigns: per-case head/base and A/A ratios over rounds, with a seeded bootstrap."""
import argparse, csv, math, random, statistics
from collections import defaultdict
from pathlib import Path

ORDINARY = ('zeros10', 'zeros50', 'zeros90')

def bootstrap(ratios, rng, n=10000):
    k = len(ratios)
    meds = sorted(statistics.median(rng.choices(ratios, k=k)) for _ in range(n))
    return meds[int(0.025 * n)], meds[int(0.975 * n) - 1]

def verdict(med, lo):
    if med <= 1.05:
        return 'pass'
    return 'investigate' if lo > 1.0 else 'inconclusive'

def gmean(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('campaigns', nargs='+', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    obs = defaultdict(dict)       # (case, campaign, round) -> role -> row
    for c in a.campaigns:
        for row in csv.DictReader((c / 'observations.csv').open()):
            assert row['status'] == 'pass', row
            key = (row['case'], c.name, int(row['round']))
            assert row['role'] not in obs[key], ('duplicate', key, row['role'])
            obs[key][row['role']] = row
    cases = sorted({k[0] for k in obs})
    rng = random.Random(834)
    out = []
    for case in cases:
        keys = sorted(k for k in obs if k[0] == case)
        rows = [obs[k] for k in keys]
        assert all(set(r) == {'baseline', 'candidate', 'baselineAA'} for r in rows), case
        # the roles of each round timed the same fixtures (absent only in campaigns before fingerprints)
        fingerprints = [{x.get('fingerprint') for x in r.values()} for r in rows]
        assert all(len(f) == 1 and None not in f and '' not in f for f in fingerprints), ('fingerprint', case)
        t = {role: [float(r[role]['median_ns']) for r in rows] for role in ('baseline', 'candidate', 'baselineAA')}
        head = [c / b for c, b in zip(t['candidate'], t['baseline'])]
        aa = [c / b for c, b in zip(t['baselineAA'], t['baseline'])]
        hlo, hhi = bootstrap(head, rng)
        alo, ahi = bootstrap(aa, rng)
        allocs = {role: max(int(r[role]['allocs']) for r in rows) for role in t}
        direct = {role: max(int(r[role]['direct_bytes']) for r in rows) for role in t}
        cv = {role: statistics.stdev(v) / statistics.mean(v) if len(v) > 1 else 0.0 for role, v in t.items()}
        kind, T, s, op = case.split('/')
        drop = {role: [float(r[role]['drop_mean_ns']) for r in rows if r[role].get('drop_mean_ns')] for role in t}
        out.append(dict(case=case, kind=kind, T=T, s=s, op=op, rounds=len(rows),
                        base_ns=statistics.median(t['baseline']), head_ns=statistics.median(t['candidate']),
                        aa_ns=statistics.median(t['baselineAA']),
                        head_ratio=statistics.median(head), head_lo=hlo, head_hi=hhi,
                        aa_ratio=statistics.median(aa), aa_lo=alo, aa_hi=ahi,
                        base_cv=cv['baseline'], head_cv=cv['candidate'],
                        head_verdict=verdict(statistics.median(head), hlo),
                        aa_verdict=verdict(statistics.median(aa), alo),
                        allocs_base=allocs['baseline'], allocs_head=allocs['candidate'],
                        direct_bytes_base=direct['baseline'], direct_bytes_head=direct['candidate'],
                        drop_base_ns=statistics.median(drop['baseline']) if drop['baseline'] else '',
                        drop_head_ns=statistics.median(drop['candidate']) if drop['candidate'] else ''))
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output / 'cases.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    lines = ['# #834 dropzeros! campaign', '',
             'Ratios are medians over rounds of the per-round ratio of per-call medians; [lo, hi] is a seeded',
             '10,000-resample 95% bootstrap interval over rounds. Verdict: pass if the median ratio is at most',
             '1.05; investigate if it exceeds 1.05 and the lower bound exceeds 1.0; otherwise inconclusive.', '']
    lines += ['## Groups (geometric mean over the measured cells of each group; n = number of cells)', '',
              '| kind | case | n | head/base | A/A | head verdicts | A/A verdicts |', '|---|---|---|---|---|---|---|']
    for kind in ('sparse', 'symmetric'):
        for op in ('zeros0', 'zeros10', 'zeros50', 'zeros90', 'zeros100', 'fill_drop', 'maintain10', 'reset'):
            g = [r for r in out if r['kind'] == kind and r['op'] == op]
            if not g:
                continue
            hv = ', '.join(f'{v} {sum(r["head_verdict"] == v for r in g)}' for v in ('pass', 'inconclusive', 'investigate'))
            av = ', '.join(f'{v} {sum(r["aa_verdict"] == v for r in g)}' for v in ('pass', 'inconclusive', 'investigate'))
            lines.append(f'| {kind} | {op} | {len(g)} | {gmean([r["head_ratio"] for r in g]):.3f} | '
                         f'{gmean([r["aa_ratio"] for r in g]):.3f} | {hv} | {av} |')
    lines += ['', '## Cases', '',
              '| case | base ns/call | head ns/call | head/base [95%] | A/A [95%] | CV base/head | allocs b/h | verdict |',
              '|---|---|---|---|---|---|---|---|']
    for r in out:
        lines.append(f'| {r["case"]} | {r["base_ns"]:.1f} | {r["head_ns"]:.1f} | '
                     f'{r["head_ratio"]:.3f} [{r["head_lo"]:.3f}, {r["head_hi"]:.3f}] | '
                     f'{r["aa_ratio"]:.3f} [{r["aa_lo"]:.3f}, {r["aa_hi"]:.3f}] | '
                     f'{r["base_cv"]:.3f}/{r["head_cv"]:.3f} | {r["allocs_base"]}/{r["allocs_head"]} | {r["head_verdict"]} |')
    work = [r for r in out if r['op'] == 'maintain10']
    if work:
        lines += ['', '## Maintenance workload: cycle time and the part inside dropzeros! (medians over rounds)', '',
                  '| case | cycle base ns | cycle head ns | head/base [95%] | dropzeros! base ns | dropzeros! head ns | head - base, ns | approx. share of base cycle in dropzeros! (mean inner / median cycle) |',
                  '|---|---|---|---|---|---|---|---|']
        for r in work:
            lines.append(f'| {r["case"]} | {r["base_ns"]:.0f} | {r["head_ns"]:.0f} | {r["head_ratio"]:.3f} [{r["head_lo"]:.3f}, {r["head_hi"]:.3f}] | '
                         f'{r["drop_base_ns"]:.0f} | {r["drop_head_ns"]:.0f} | {r["head_ns"] - r["base_ns"]:+.0f} | {r["drop_base_ns"] / r["base_ns"]:.2f} |')
    flagged = [r for r in out if r['op'] != 'reset' and r['head_verdict'] != 'pass']
    lines += ['', f'Non-pass head cells (reset excluded): {len(flagged)}',
              f'Non-pass A/A cells (reset excluded): {sum(r["aa_verdict"] != "pass" for r in out if r["op"] != "reset")}',
              f'Any allocation: base {max(r["allocs_base"] for r in out)}, head {max(r["allocs_head"] for r in out)}; '
              f'direct bytes: base {max(r["direct_bytes_base"] for r in out)}, head {max(r["direct_bytes_head"] for r in out)}']
    (a.output / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
