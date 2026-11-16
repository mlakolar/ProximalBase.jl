# ElasticNet prox! on a sparse iterate with a scalar step (the case enet/sparse_prox!/scalar of benchmark/ridge910.jl:
# 65,536 entries, a tenth of them stored), with the stored values replaced before every call by one of K value sets.
# K = 1 is the benchmark's situation, the same input in every call, which a branch predictor can learn; larger K
# shows the loop on inputs it has not just seen. The copy of the values is inside the timed region for every tree.
# The trees share the arrays of the input and of the output (probe3 was run with a first version in which every tree
# had its own arrays; KO_SPARSE_OWN=1 reproduces it).
#   julia --threads=1 --check-bounds=no bench/sparse_probe.jl <reference root> <candidate root> <output csv> [rounds]
# Row: element type, operation, K, implementation, minimum and median over rounds of one call (ns), copy included,
# and the time of the copy alone.
using Random, Printf, Statistics
function loadtree(name, root)
  m = Core.eval(Main, :(module $(Symbol("Tree_", name)) end))
  Base.include(m, joinpath(abspath(root), "src", "ProximalBase.jl"))
  getfield(m, :ProximalBase)
end
const REF = loadtree("ref", ARGS[1]); const CAND = loadtree("cand", ARGS[2]); const REFC = loadtree("refC", ARGS[1])
const ROUNDS = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 11
@noinline fill_values!(dst, V, k, m) = copyto!(dst, 1, V[k], 1, m)
@noinline function sweep(p::P, y, f, x, γ, V, m, calls) where {P}
  K = length(V)
  for c in 1:calls
    fill_values!(x.nzval, V, mod1(c, K), m)
    p(y, f, x, γ)
  end
end
@noinline function sweep_copy(x, V, m, calls)
  K = length(V)
  for c in 1:calls; fill_values!(x.nzval, V, mod1(c, K), m); end
end
function best(f::F) where {F}
  t = typemax(UInt64)
  for _ in 1:3; t0 = time_ns(); f(); t = min(t, time_ns() - t0); end
  t
end
open(ARGS[3], "w") do io
  println(io, "T,op,K,impl,min_ns,med_ns,copy_ns")
  n = 65536
  for T in (Float32, Float64)
    rng = MersenneTwister(910)
    xs = zeros(T, n); stored = randperm(rng, n)[1:(n ÷ 10)]; xs[stored] .= T(2) .* randn(rng, T, length(stored))
    mods = (("ref", REF), ("cand", CAND), ("refC", REFC))
    for op in ("enet", "l1", "enet_w")
      for K in (1, 2, 8, 64, 512)
        # one input and one output, shared by the trees: every tree wraps the same index and value arrays in its own
        # SparseIterate type (KO_SPARSE_OWN=1 gives every tree its own arrays, as PB's harness does)
        sp0 = REF.SparseIterate(xs); out0 = copy(sp0)
        data = map(mods) do (name, M)
          if get(ENV, "KO_SPARSE_OWN", "0") == "1"
            sp = M.SparseIterate(xs); out = copy(sp)
          else
            sp = M.SparseIterate(sp0.nzval, sp0.nzval2ind, sp0.ind2nzval, sp0.nnz)
            out = M.SparseIterate(out0.nzval, out0.nzval2ind, out0.ind2nzval, out0.nnz)
          end
          f = op == "l1" ? M.NormL1(T(0.3)) : op == "enet" ? M.ElasticNet(T(0.3), T(0.5)) : M.ElasticNet(T(0.3), T(0.5), T(0.5) .+ rand(MersenneTwister(7), T, n))
          (name, M.prox!, sp, out, f)
        end
        m = data[1][3].nnz
        V = [T(2) .* randn(MersenneTwister(k), T, m) for k in 1:K]
        K == 1 && copyto!(V[1], 1, data[1][3].nzval, 1, m)
        calls = 4 * max(K, 32)
        γ = T(0.7)
        for (_, p, sp, out, f) in data; sweep(p, out, f, sp, γ, V, m, calls); end
        t = [Float64[] for _ in data]; tc = Float64[]
        for r in 1:ROUNDS
          order = circshift(collect(1:length(data)), (r - 1) ÷ 2); isodd(r) || reverse!(order)
          for k in order
            (_, p, sp, out, f) = data[k]
            push!(t[k], best(() -> sweep(p, out, f, sp, γ, V, m, calls)) / calls)
          end
          push!(tc, best(() -> sweep_copy(data[1][3], V, m, calls)) / calls)
        end
        for (k, d) in enumerate(data)
          @printf(io, "%s,%s,%d,%s,%.1f,%.1f,%.1f\n", T, op, K, d[1], minimum(t[k]), median(t[k]), median(tc))
        end
        flush(io)
      end
    end
  end
end
println("done")
