include("solver.jl")

@noinline solve_bytes(args) = @allocated solve!(args...)

function main(pr, replica, repaired, folder, reps)
    mkpath(folder)
    open(joinpath(folder, "solves.tsv"), "w") do timing
        open(joinpath(folder, "allocations.tsv"), "w") do alloc
            tsv(timing, "julia", "problem", "type", "stride", "left", "right", "side", "rep", "position",
                "ns", "steps", "scores", "reported", "reference")
            tsv(alloc, "julia", "problem", "type", "stride", "label", "bytes", "steps", "scores", "same_solution")
            cases = Iterators.product(("dense", "diagonal"), (Float32, Float64), (1, 10))
            for (caseidx, (name, T, stride)) in enumerate(cases)
                loss, step, c, tol = problem(T, name)
                args = map((pr, replica, repaired)) do pb
                    (workspace(loss), loss, pb.NormLinf(c), step, tol, stride, pb.prox!, pb.kkt_violation, no_capture)
                end
                endpoints = map(a -> warm_endpoint(a, c, tol), args)
                for (i, label) in enumerate(("PR", "AA", "pairwise"))
                    bytes = solve_bytes(args[i])
                    tsv(alloc, VERSION, name, T, stride, label, bytes, endpoints[i][1].steps,
                        endpoints[i][1].scores, endpoints[1][3] == endpoints[i][3])
                end
                first_order = isodd(caseidx) ? (1, 2) : (2, 1)
                last_order = reverse(first_order)
                # Interleave the A/B and A/A blocks across rounds.
                for rep in 1:reps
                    order = isodd(rep) ? first_order : last_order
                    for second in (isodd(rep) ? (3, 2) : (2, 3))
                        label = second == 3 ? "pairwise" : "AA"
                        for (position, side) in enumerate(order)
                            i = side == 1 ? 1 : second
                            start = time_ns()
                            result = solve!(args[i]...)
                            elapsed = time_ns() - start
                            @assert result == endpoints[i][1]
                            tsv(timing, VERSION, name, T, stride, "PR", label, side, rep, position,
                                elapsed, result.steps, result.scores, result.residual, endpoints[i][2])
                        end
                    end
                end
                flush(timing)
                flush(alloc)
                println(stderr, "paired ", VERSION, " ", name, " ", T, " stride=", stride)
            end
        end
    end
end

pr, candidate, folder, reps = ARGS
Base.invokelatest(main, load_checkout(pr, :PRBench), load_checkout(pr, :ReplicaBench),
                  load_checkout(candidate, :CandidateBench), folder, parse(Int, reps))
