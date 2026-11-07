using ProximalBase, Random, Statistics
function fixture(T, n, s; seed)
    rng = Xoshiro(seed); ids = randperm(rng, n)[1:s]
    vals = [T((rand(rng) < 0.5 ? -1 : 1) * (1 + rand(rng))) for _ in 1:s]
    x = SparseIterate(T, n); for (i, v) in zip(ids, vals); x[i] = v; end
    x, ids
end
act!(x, batch, v) = (@inbounds for k in batch; x[k] = v; end; nothing)
function probe(nb, restore; reps=400)
    x, ids = fixture(Float64, 1_000_000, 1000; seed=11)
    active = Set(ids); rng = Xoshiro(5); batch = Int[]
    while length(batch) < 256*nb
        k = rand(rng, 1:1_000_000); (k in active || k in batch) && continue; push!(batch, k)
    end
    snap = copy(x)
    act!(x, batch, 1.0)
    restore == :dropzeros ? (for k in batch; x[k] = 0.0; end; dropzeros!(x)) : copyto!(x, snap)
    t = Float64[]
    for _ in 1:reps
        t0 = time_ns(); act!(x, batch, 1.0); t1 = time_ns(); push!(t, (t1-t0)/length(batch))
        restore == :dropzeros ? (for k in batch; x[k] = 0.0; end; dropzeros!(x)) : copyto!(x, snap)
    end
    median(t)
end
for nb in (1, 16), r in (:dropzeros, :copyto)
    println("nb=$nb restore=$r median ns/coord = ", round(probe(nb, r), digits=3))
end
