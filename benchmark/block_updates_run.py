"""Run PB-only warmed A/B/A-A diagnostics in isolated Julia processes.

Only one worker measures at a time. Two cohorts restart all workers; order
alternates within each cohort. The first complete timing round is warm-up.
Source projects must pin immutable checkouts before acceptance measurements.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Worker:
    def __init__(self, binary, project, driver, output, label, args):
        self.output = output
        self.label = label
        self.stderr = (output / f"{label}.stderr").open("w")
        env = dict(os.environ, JULIA_DEPOT_PATH=args.depot, JULIA_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", PB_BLOCK_FILTER=args.filter,
                   PB_BLOCK_SAMPLES=str(args.samples), PB_BLOCK_SECONDS=str(args.seconds),
                   PB_BLOCK_METADATA=str(output / f"{label}.metadata.tsv"))
        self.command = ["taskset", "-c", str(args.cpu), binary, "--startup-file=no",
                        "--compiled-modules=no", "--check-bounds=no", f"--project={project}",
                        str(driver), "--server"]
        self.process = subprocess.Popen(self.command, env=env, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.stderr,
                                        text=True, bufsize=1)
        ready = self.process.stdout.readline().strip()
        if not ready.startswith("READY "):
            raise RuntimeError(f"{label} did not start: {ready}; see {self.stderr.name}")
        self.count = int(ready.split()[1])
        print(f"Warmed {label}: {self.count} probes", flush=True)
        self.measure("warmup")

    def measure(self, repetition):
        path = self.output / f"{self.label}-{repetition}.tsv"
        self.process.stdin.write(str(path) + "\n")
        self.process.stdin.flush()
        done = self.process.stdout.readline().strip()
        if done != f"DONE {path}":
            raise RuntimeError(f"{self.label} failed: {done}; see {self.stderr.name}")
        return path

    def stop(self):
        if self.process.poll() is None:
            self.process.stdin.write("STOP\n")
            self.process.stdin.flush()
            self.process.stdin.close()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)
        self.stderr.close()
        if self.process.returncode:
            raise RuntimeError(f"{self.label} exited {self.process.returncode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--julia", required=True)
    parser.add_argument("--before", required=True, help="baseline Julia project")
    parser.add_argument("--after", required=True, help="candidate Julia project")
    parser.add_argument("--driver", default=str(Path(__file__).with_name("block_updates.jl")))
    parser.add_argument("--output", required=True)
    parser.add_argument("--depot", required=True)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--filter", default="")
    parser.add_argument("--cohorts", type=int, default=2)
    parser.add_argument("--pairs", type=int, default=5, help="paired rounds per cohort")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seconds", type=float, default=.02)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    driver = Path(args.driver).resolve()
    record = dict(vars(args), driver_sha256=sha(driver),
                  controller_sha256=sha(__file__), pairs=[],
                  method="two independent process cohorts; alternating A/B/control order; single evaluations; first round excluded")
    if args.pairs * args.cohorts < 10:
        record["preliminary"] = True
    workers = []
    try:
        for cohort in range(1, args.cohorts + 1):
            projects = {"before": args.before, "after": args.after, "aa": args.before}
            order = ["before", "after", "aa"] if cohort % 2 else ["aa", "after", "before"]
            live = {}
            for label in order:
                worker = Worker(args.julia, projects[label], driver, output,
                                f"cohort{cohort}-{label}", args)
                workers.append(worker)
                live[label] = worker
            counts = {worker.count for worker in live.values()}
            metadata = [sha(output / f"cohort{cohort}-{label}.metadata.tsv") for label in live]
            if len(counts) != 1 or len(set(metadata)) != 1:
                raise RuntimeError("candidate/baseline numerical metadata or case inventory differ")
            for repetition in range(1, args.pairs + 1):
                order = ["before", "after", "aa"] if (cohort + repetition) % 2 == 0 else ["aa", "after", "before"]
                paths = {}
                for label in order:
                    paths[label] = live[label].measure(f"pair{repetition}").name
                record["pairs"].append(dict(cohort=cohort, repetition=repetition,
                                             order=order, files=paths))
                (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
                print(f"Completed cohort {cohort} pair {repetition}: {','.join(order)}", flush=True)
            for worker in live.values():
                worker.stop()
            workers.clear()
    finally:
        for worker in workers:
            worker.stop()
    record["exit_code"] = 0
    (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
