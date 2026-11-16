# Interleaved in-process timing harness of the kernel-optimization study.
#
# A case is one mathematical operation on one input; its implementations do the same work on the same
# arrays (inputs and outputs are shared, so array placement does not differ between them). Every round
# times every implementation once, in an order that rotates and reverses from round to round; a timing is
# the minimum of a few batches, each calibrated to about BATCH_MS milliseconds. The raw table has one row
# per (case, implementation, round); summarize.jl forms the paired per-round ratios.
module Harness

using Printf, Statistics, Dates, Random, LinearAlgebra

struct Impl
  name::String
  level::Int          # 1 inner kernel, 2 library API, 3 solver
  op::Any             # () -> nothing: the timed operation
  out::Any            # () -> the output to compare (run after op)
  note::String
end

mutable struct Case
  id::String
  family::String
  ref::String          # the implementation that the ratios are taken against
  impls::Vector{Impl}
  expected::Any        # independent reference output (nothing: compare with ref only)
  rtol::Float64        # allowed relative difference of the outputs
end
Case(id, family, ref; expected = nothing, rtol = 0.0) = Case(id, family, ref, Impl[], expected, rtol)

add!(c::Case, name, level, op, out; note = "") = (push!(c.impls, Impl(name, level, op, out, note)); c)

# The call is opaque to the batch loop, so that a pure kernel on unchanged inputs is not hoisted out of it.
@noinline invoke_op(op::F) where {F} = (op(); nothing)
function runbatch(op::F, evals::Int) where {F}
  t0 = time_ns()
  for _ in 1:evals
    invoke_op(op)
  end
  time_ns() - t0
end

function calibrate(op, batch_ns)
  runbatch(op, 1); runbatch(op, 1)
  evals = 1
  while true
    t = runbatch(op, evals)
    t >= batch_ns / 4 && return max(1, round(Int, evals * batch_ns / max(t, 1)))
    evals *= 4
  end
end

# The largest relative difference of two outputs (numbers, arrays or tuples of them). NaN matches NaN and an
# infinity matches the same infinity; any other difference of class (a NaN or an infinity on one side only), another
# shape or another tuple length is Inf, so that it cannot pass a tolerance.
reldiff(a::Tuple, b::Tuple) = length(a) == length(b) ? maximum(reldiff(x, y) for (x, y) in zip(a, b)) : Inf
function reldiff(a::Number, b::Number)
  (isnan(a) || isnan(b)) && return (isnan(a) && isnan(b)) ? 0.0 : Inf
  a == b && return 0.0
  (isinf(a) || isinf(b)) && return Inf
  Float64(abs(big(a) - big(b)) / max(abs(big(b)), big(floatmin(Float64))))
end
function reldiff(a::AbstractArray, b::AbstractArray)
  size(a) == size(b) || return Inf
  isempty(a) && return 0.0
  # relative to the largest finite magnitude of the reference, so that entries shrunk to zero count
  s = max(maximum(x -> isfinite(x) ? abs(big(x)) : big(0.0), b), big(floatmin(Float64)))
  m = 0.0
  for (x, y) in zip(a, b)
    if isnan(x) || isnan(y)
      (isnan(x) && isnan(y)) || return Inf
    elseif x != y
      (isinf(x) || isinf(y)) && return Inf
      m = max(m, Float64(abs(big(x) - big(y)) / s))
    end
  end
  m
end
# the entries that are exactly zero in one output and not in the other (the active sets of a proximal point)
zeromismatch(a::Tuple, b::Tuple) = sum(zeromismatch(x, y) for (x, y) in zip(a, b))
zeromismatch(a::Number, b::Number) = Int(iszero(a) != iszero(b))
zeromismatch(a::AbstractArray, b::AbstractArray) = size(a) == size(b) ? count(i -> iszero(a[i]) != iszero(b[i]), eachindex(a, b)) : -1
zeromismatch(a, b) = 0

function provenance(io, trees)
  println(io, "date = ", now())
  println(io, "julia = ", VERSION)
  println(io, "llvm = ", Base.libllvm_version)
  println(io, "cpu_name = ", Sys.CPU_NAME)
  println(io, "cpu_model = ", Sys.cpu_info()[1].model)
  println(io, "cpu_threads = ", Sys.CPU_THREADS)
  println(io, "julia_threads = ", Threads.nthreads())
  println(io, "blas_threads = ", BLAS.get_num_threads())
  println(io, "blas_config = ", BLAS.get_config())
  println(io, "blas_vendor_libs = ", join([basename(l.libname) for l in BLAS.get_config().loaded_libs], ","))
  try
    println(io, "openblas_corename = ", unsafe_string(ccall((:openblas_get_corename64_, BLAS.libblastrampoline), Cstring, ())))
  catch
  end
  println(io, "hostname = ", gethostname())
  for k in ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE",
            "JULIA_CPU_TARGET", "OPENBLAS_NUM_THREADS", "KO_CONTROL_FIRST", "KO_TREES", "KO_REF", "KO_TREES_ONLY", "KO_SUITES", "KO_ROUNDS", "KO_BATCH_MS", "KO_SAMPLES", "KO_FILTER", "KO_TYPES")
    haskey(ENV, k) && println(io, k, " = ", ENV[k])
  end
  println(io, "check_bounds = ", Base.JLOptions().check_bounds)
  println(io, "opt_level = ", Base.JLOptions().opt_level)
  println(io, "startup_file = ", Base.JLOptions().startupfile)
  println(io, "bigfloat_precision = ", precision(BigFloat))
  println(io, "harness_version = 2")
  for f in ("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "/sys/kernel/mm/transparent_hugepage/enabled", "/sys/devices/system/cpu/smt/active")
    isfile(f) && println(io, f, " = ", strip(read(f, String)))
  end
  for (name, path, sha) in trees
    println(io, "tree.", name, " = ", path, " @ ", sha)
  end
end

"""
    run(cases, outdir; rounds, batch_ms, samples)

Check and time the cases, writing `raw.csv` (case, impl, round, pos, ns) and `meta.csv` (one row per
implementation: level, evals, allocated bytes, the relative difference from the reference implementation and
from the independent reference) to `outdir`.
"""
function run(cases::Vector{Case}, outdir; rounds = 21, batch_ms = 2.0, samples = 3, check_only = false)
  mkpath(outdir)
  batch_ns = batch_ms * 1e6
  raw = open(joinpath(outdir, "raw.csv"), "w")
  meta = open(joinpath(outdir, "meta.csv"), "w")
  println(raw, "case,impl,round,pos,ns")
  println(meta, "case,family,ref,impl,level,evals,alloc_bytes,diff_ref,diff_expected,rtol,status,note,zero_mismatch")
  nfail = 0
  for (ci, c) in enumerate(cases)
    names = [i.name for i in c.impls]
    refi = findfirst(==(c.ref), names)
    refi === nothing && error("case $(c.id): no reference implementation $(c.ref)")
    # outputs: every implementation runs once and is compared with the reference and the expected value
    outs = Any[]
    for im in c.impls
      invoke_op(im.op)
      push!(outs, deepcopy(im.out()))
    end
    evals = Int[]; allocs = Int[]
    for im in c.impls
      e = check_only ? 1 : calibrate(im.op, batch_ns)
      push!(evals, e)
      invoke_op(im.op)
      push!(allocs, @allocated invoke_op(im.op))
    end
    for (k, im) in enumerate(c.impls)
      dref = reldiff(outs[k], outs[refi])
      dexp = c.expected === nothing ? NaN : reldiff(outs[k], c.expected)
      bad = !(dref <= c.rtol) || (c.expected !== nothing && !(dexp <= c.rtol))
      zm = zeromismatch(outs[k], outs[refi])
      nfail += bad
      @printf(meta, "%s,%s,%s,%s,%d,%d,%d,%.3g,%.3g,%.3g,%s,%s,%d\n", c.id, c.family, c.ref, im.name, im.level, evals[k], allocs[k],
              dref, dexp, c.rtol, bad ? "DIFF" : "ok", replace(im.note, ',' => ';'), zm)
      bad && @printf(stderr, "DIFF %s %s: vs ref %.3g, vs expected %.3g (rtol %.3g)\n", c.id, im.name, dref, dexp, c.rtol)
    end
    flush(meta)
    check_only && continue
    m = length(c.impls)
    GC.gc()
    for r in 1:rounds
      # rounds 2k-1 and 2k are rotation k-1 of the implementations and its reversal: over m pairs every
      # implementation has held every position, in both directions (also for two implementations)
      order = circshift(collect(1:m), (r - 1) ÷ 2)
      isodd(r) || reverse!(order)
      for (pos, k) in enumerate(order)
        op = c.impls[k].op
        t = typemax(UInt64)
        for _ in 1:samples
          t = min(t, runbatch(op, evals[k]))
        end
        @printf(raw, "%s,%s,%d,%d,%.4f\n", c.id, names[k], r, pos, t / evals[k])
      end
    end
    flush(raw)
    ci % 20 == 0 && @printf(stderr, "[%s] %d/%d cases\n", Dates.format(now(), "HH:MM:SS"), ci, length(cases))
  end
  close(raw); close(meta)
  nfail
end

end # module
