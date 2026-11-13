# Run check.jl in fresh bounds-checked processes for tree A and for every tree that
# make_sources.py builds from candidates/*.jl. AA (a copy of A), O (the original
# implementation) and any other directory are not checked. Passes only when every
# checked tree but the negative control X passes, and X runs to completion and fails.
# usage: julia --project=benchmark/audit852 check_all.jl SOURCES_DIR [NTRIALS]
srcdir = abspath(ARGS[1])
ntr = length(ARGS) > 1 ? ARGS[2] : "4000"
names = ["A"; sort!([splitext(f)[1] for f in readdir(joinpath(@__DIR__, "candidates")) if endswith(f, ".jl")])]
"X" in names || error("candidates/X.jl, the negative control, is missing")
absent = filter(n -> !isdir(joinpath(srcdir, n, "src")), names)
isempty(absent) || error("missing source trees: $(join(absent, ", "))")
jl(args...) = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) --check-bounds=yes $(joinpath(@__DIR__, "check.jl")) $args`
run(ignorestatus(jl(joinpath(srcdir, "A"), 1, 1)))      # compile package images for these flags once
results = Dict{String,Tuple{Int,String}}()
for n in names
  out = IOBuffer()
  p = run(pipeline(ignorestatus(jl(joinpath(srcdir, n), ntr, 872)); stdout = out, stderr = out))
  text = String(take!(out)); print(text)
  results[n] = (p.exitcode, text)
end
println("exit codes: ", sort!([n => r[1] for (n, r) in results]))
# X must finish its checks and report failures; an error before checking does not count
xran = occursin(r"checks=\d+ failures=[1-9]", results["X"][2])
ok = all(r[1] == 0 for (n, r) in results if n != "X") && results["X"][1] != 0 && xran
println(ok ? "CHECK_ALL OK (negative control X ran and failed its checks)" : "CHECK_ALL FAILED")
exit(ok ? 0 : 1)
