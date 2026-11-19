include("common.jl")
include("norm-cases.jl")

@noinline function run_scores(score::S, f, v, x, calls) where {S}
    total = zero(eltype(v))
    for _ in 1:calls
        total += score(f, v, x)
    end
    total
end

@noinline measured_bytes(score::S, f, v, x) where {S} = @allocated score(f, v, x)

function elapsed(args, calls)
    start = time_ns()
    total = run_scores(args..., calls)
    time_ns() - start, total
end

function measure(io, modules, v, x, c, exact, name, ::Type{T}, reps, first_order) where {T}
    args = map(pb -> (pb.kkt_violation, pb.NormLinf(c), v, x), modules)
    outputs = map(args) do a
        score, f, v, x = a
        score(f, v, x)
    end
    @assert isapprox(outputs[3], T(exact); rtol = 8 * eps(T)) || length(v) <= 1024
    alloc = map(a -> measured_bytes(a...), args)
    foreach(a -> run_scores(a..., 1), args)
    slow = maximum(a -> first(elapsed(a, 1)), args)
    calls = clamp(div(1_200_000, max(slow, 1)), 1, 4096)
    for rep in 1:reps
        order = isodd(rep) ? first_order : reverse(first_order)
        for second in (isodd(rep) ? (3, 2) : (2, 3))
            for (position, side) in enumerate(order)
                i = side == 1 ? 1 : second
                ns, checksum = elapsed(args[i], calls)
                @assert isfinite(checksum)
                tsv(io, VERSION, T, name, length(v), ("PR", "AA", "pairwise")[second],
                    side, rep, position, calls, ns, alloc[i], outputs[i], exact)
            end
        end
    end
end

function main(pr, replica, repaired, output, reps)
    open(output, "w") do io
        tsv(io, "julia", "type", "case", "n", "right", "side", "rep", "position",
            "calls", "ns", "bytes", "observed", "reference")
        cases = Iterators.product((Float32, Float64),
            ("zero_penalty", "zero_outside", "singleton_face", "signed_face"),
            (512, 1024, 1025, 4096, 100001, 1 << 20, 1 << 22))
        for (index, (T, path, n)) in enumerate(cases)
            v, x, c, exact = norm_case(T, n, path)
            first_order = isodd(index) ? (1, 2) : (2, 1)
            measure(io, (pr, replica, repaired), v, x, c, exact, path, T, reps, first_order)
        end
    end
end

pr, candidate, output, reps = ARGS
Base.invokelatest(main, load_checkout(pr, :PRFocused), load_checkout(pr, :ReplicaFocused),
                  load_checkout(candidate, :CandidateFocused), output, parse(Int, reps))
