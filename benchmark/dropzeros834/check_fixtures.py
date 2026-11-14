#!/usr/bin/env python3
"""Regression check: separate Julia processes, loading either source tree, generate identical fixtures.

Runs the worker in fingerprint mode in four fresh processes (baseline, candidate, baseline, candidate)
for two campaign seeds and requires the same nonempty set of cases, without duplicates, in all eight
outputs; per seed and case, one fingerprint across the four processes; a different fingerprint for
the other seed; and the generator seed derivation documented in worker.jl.
"""
import argparse, csv, hashlib, subprocess, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent

def case_seed(seed, case):
    return int.from_bytes(hashlib.sha256(f'dropzeros834|{seed}|{case}'.encode()).digest()[:8], 'big')

def verify(outputs):
    """Check {(seed, process): {case: fingerprint}} for seeds 834001/834002 and processes 0-3."""
    assert set(outputs) == {(seed, n) for seed in (834001, 834002) for n in range(4)}, 'missing process output'
    # every one of the eight process outputs must hold the same, nonempty set of cases
    inventories = {frozenset(rows) for rows in outputs.values()}
    assert len(inventories) == 1 and len(next(iter(inventories))) > 0, 'process outputs differ in their cases'
    cases = sorted(next(iter(inventories)))
    bad = [(seed, c) for seed in (834001, 834002) for c in cases
           if len({outputs[(seed, n)][c] for n in range(4)}) != 1]
    same = [c for c in cases if outputs[(834001, 0)][c] == outputs[(834002, 0)][c]]
    return cases, bad, same

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--julia', type=Path, required=True)
    p.add_argument('--project', type=Path, required=True, help='an instantiated copy of benchmark/Project.toml')
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    a = p.parse_args()
    outputs = {}                                 # (seed, process) -> case -> fingerprint
    with tempfile.TemporaryDirectory() as tmp:
        for seed in (834001, 834002):
            for n, source in enumerate((a.baseline, a.candidate, a.baseline, a.candidate)):
                out = Path(tmp) / f'{seed}-{n}.csv'
                subprocess.run([str(a.julia), '--startup-file=no', f'--project={a.project}', str(HERE / 'worker.jl'),
                                str(source.resolve()), f'process{n}', str(out), str(seed), 'fingerprint'], check=True)
                rows = {}
                for row in csv.DictReader(out.open()):
                    assert row['case'] not in rows, ('duplicate row', seed, n, row['case'])
                    assert int(row['seed']) == case_seed(seed, row['case']), ('seed derivation', row['case'])
                    assert row['fingerprint'], ('empty fingerprint', seed, n, row['case'])
                    rows[row['case']] = row['fingerprint']
                outputs[(seed, n)] = rows
    cases, bad, same = verify(outputs)
    print(f'{len(cases)} cases x 2 seeds x 4 processes (every case in all 8 outputs): '
          f'{len(bad)} with differing fingerprints, {len(same)} identical across seeds')
    if bad or same:
        sys.exit(1)

if __name__ == '__main__':
    main()
