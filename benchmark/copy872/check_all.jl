# Run check.jl for every candidate tree in fresh processes with bounds checking.
# usage: julia --project=benchmark/audit852 check_all.jl SOURCES_DIR [NTRIALS]
# The negative control X (an unconditional dense copy) must fail.
srcdir = abspath(ARGS[1])
ntr = length(ARGS) > 1 ? ARGS[2] : "4000"
jl(args...) = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) --check-bounds=yes $(joinpath(@__DIR__, "check.jl")) $args`
run(ignorestatus(jl(joinpath(srcdir, "A"), 1, 1)))      # compile package images for these flags once
names = sort!(filter(n -> !(n in ("AA", "O")), readdir(srcdir)))
status = Dict(n => run(ignorestatus(jl(joinpath(srcdir, n), ntr, 872))).exitcode for n in names)
println("exit codes: ", sort!(collect(status)))
ok = all(v == 0 for (n, v) in status if n != "X") && get(status, "X", 1) != 0
println(ok ? "CHECK_ALL OK (negative control X rejected)" : "CHECK_ALL FAILED")
exit(ok ? 0 : 1)
