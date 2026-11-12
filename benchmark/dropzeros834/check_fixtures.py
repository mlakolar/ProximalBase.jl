#!/usr/bin/env python3
"""Regression check: separate Julia processes, loading either source tree, generate identical fixtures.

Runs the worker in fingerprint mode in four fresh processes (baseline, candidate, baseline, candidate)
for two campaign seeds and requires, per seed and case, one fingerprint across the processes, a
different fingerprint for the other seed, and the generator seed SHA-256 documents in worker.jl.
"""
import argparse, csv, hashlib, subprocess, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent

def case_seed(seed, case):
    return int.from_bytes(hashlib.sha256(f'dropzeros834|{seed}|{case}'.encode()).digest()[:8], 'big')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--julia', type=Path, required=True)
    p.add_argument('--project', type=Path, required=True, help='an instantiated copy of benchmark/Project.toml')
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    a = p.parse_args()
    prints = {}
    with tempfile.TemporaryDirectory() as tmp:
        for seed in (834001, 834002):
            for n, source in enumerate((a.baseline, a.candidate, a.baseline, a.candidate)):
                out = Path(tmp) / f'{seed}-{n}.csv'
                subprocess.run([str(a.julia), '--startup-file=no', f'--project={a.project}', str(HERE / 'worker.jl'),
                                str(source.resolve()), f'process{n}', str(out), str(seed), 'fingerprint'], check=True)
                for row in csv.DictReader(out.open()):
                    assert int(row['seed']) == case_seed(seed, row['case']), ('seed derivation', row['case'])
                    prints.setdefault((seed, row['case']), set()).add(row['fingerprint'])
    cases = sorted({c for _, c in prints})
    bad = [k for k, v in prints.items() if len(v) != 1]
    same = [c for c in cases if prints[(834001, c)] == prints[(834002, c)]]
    print(f'{len(cases)} cases x 2 seeds x 4 processes: {len(bad)} with differing fingerprints, '
          f'{len(same)} identical across seeds')
    if bad or same:
        sys.exit(1)

if __name__ == '__main__':
    main()
