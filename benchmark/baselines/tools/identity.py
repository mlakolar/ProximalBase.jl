import sys, glob, tomllib, collections
B = sys.argv[1]
keys = ["harness_hash","source_sha","source_hash","source_dirty","suite","groups","julia","julia_threads","blas_threads","cpu_model","cpu","check_bounds","os","cases","failures","host","benchmarktools","samples_target","seconds_limit","slurm_job_id"]
seen = collections.defaultdict(lambda: collections.defaultdict(set))
for d in ["base-groups","base-layouts","aa-groups","aa-layouts"]:
    files = sorted(glob.glob(f"{B}/{d}/*-metadata.toml"))
    print(d, "metadata files:", len(files), "reps:", sorted(tomllib.load(open(f,'rb'))["rep"] for f in files))
    for f in files:
        m = tomllib.load(open(f,'rb'))
        for k in keys:
            seen[d][k].add(str(m.get(k)))
for k in keys:
    print(f"{k:16s}", {d: sorted(seen[d][k]) for d in seen})
