#!/usr/bin/env python3
"""#834 paired protocol: fresh processes per role and round, rotated role order, baseline A/A control."""
import argparse, csv, hashlib, json, os, platform, shutil, subprocess, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def tree_digest(root):
    files = sorted(p for p in (root / 'src').rglob('*.jl'))
    lines = [f'{p.relative_to(root)}:{digest(p)}' for p in files]
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--julia', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True, help='archived src of the pinned master')
    p.add_argument('--candidate', type=Path, required=True, help='archived src of the head')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=11)
    p.add_argument('--start-round', type=int, default=1)
    p.add_argument('--check', action='store_true', help='bounds-checked correctness run, no timing')
    p.add_argument('--filter', default='')
    p.add_argument('--cpu', type=int, default=min(os.sched_getaffinity(0)))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    # a fresh environment per campaign from the checked-in benchmark project
    env_dir = a.output / 'env'
    env_dir.mkdir()
    shutil.copy(HERE.parent / 'Project.toml', env_dir / 'Project.toml')
    subprocess.run([str(a.julia), '--startup-file=no', f'--project={env_dir}', '-e',
                    'using Pkg; Pkg.instantiate(); using BenchmarkTools, SHA'], check=True)
    roles = ['baseline', 'candidate', 'baselineAA']
    sources = {'baseline': a.baseline, 'candidate': a.candidate, 'baselineAA': a.baseline}
    julia_version = subprocess.check_output([str(a.julia), '--startup-file=no', '-e', 'print(VERSION)'], text=True)
    provenance = {
        'argv': {k: str(v) for k, v in vars(a).items()}, 'platform': platform.platform(),
        'julia_version': julia_version, 'julia_sha256': digest(a.julia.resolve()),
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'source_digest': {r: tree_digest(s.resolve()) for r, s in sources.items()},
        'harness_sha256': {f.name: digest(f) for f in sorted(HERE.iterdir()) if f.suffix in ('.jl', '.py', '.md')},
        'benchmark_project_sha256': digest(env_dir / 'Project.toml'),
        'benchmark_manifest_sha256': digest(env_dir / 'Manifest.toml'),
        'lscpu': subprocess.check_output(['lscpu'], text=True),
        'cpu': a.cpu,
    }
    (a.output / 'campaign.json').write_text(json.dumps(provenance, indent=2) + '\n')
    failures = []
    fields = ['round', 'order', 'role', 'seed', 'loadavg', 'case', 'status', 'median_ns', 'min_ns', 'mean_ns',
              'p90_ns', 'allocs', 'memory', 'direct_bytes', 'samples', 'batch']
    with (a.output / 'observations.csv').open('w') as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for r in range(a.start_round, a.start_round + a.rounds):
            order = roles[r % 3:] + roles[:r % 3]        # each role takes each position once per 3 rounds
            seed = 834000 + r                            # same seed for every role of a round
            for ordinal, role in enumerate(order, 1):
                target = a.output / f'round-{r:02}-{role}.csv'
                load = os.getloadavg()[0]
                cmd = ['taskset', '-c', str(a.cpu), str(a.julia), '--startup-file=no', '--threads=1', '-O2',
                       '--check-bounds=' + ('yes' if a.check else 'no'), f'--project={env_dir}',
                       str(HERE / 'worker.jl'), str(sources[role].resolve()), role, str(target), str(seed),
                       'check' if a.check else 'time', a.filter]
                env = os.environ | {'JULIA_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
                with (a.output / f'round-{r:02}-{role}.log').open('w') as log:
                    subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
                for row in csv.DictReader(target.open()):
                    writer.writerow(dict(round=r, order=ordinal, role=role, seed=seed, loadavg=load) | row)
                    if row['status'] != 'pass':
                        failures.append((r, role, row['case']))
                out.flush()
                print(f'round {r} {role} (position {ordinal}, load {load:.2f}) failures={len(failures)}', flush=True)
    provenance['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (a.output / 'campaign.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (a.output / 'failures.json').write_text(json.dumps(failures, indent=2) + '\n')
    assert not failures, failures[:3]

if __name__ == '__main__':
    main()
