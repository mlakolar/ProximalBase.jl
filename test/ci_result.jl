module CIResultTest

# The reader of a test job's log (scripts/ci_result.jl): the files the runner selected, what became
# of each, the failures with a signature, and the check that a job ran the whole catalog of its tier.
# The fixtures are cut from the failed log of a real run (a macOS and a Windows job that both failed
# test/three_factor_product.jl), in the shape `gh run view --log-failed` prints.

using Test

include(joinpath(dirname(@__DIR__), "scripts", "ci_result.jl"))
using .CIResult: parse_log, check_inventory, normalize_signature, test_file, clean_line, error_line

const ROOT = dirname(@__DIR__)
const SCRIPT = joinpath(ROOT, "scripts", "ci_result.jl")
const MANIFEST = CIResult.default_manifest()
const TS = "2026-11-07T16:14:14.9228380Z"
const TESTSET = "allocation does not grow with the number of stored coefficients"

# The log of one job as `gh run view --log-failed` prints it: job, step and timestamp, a tab apart.
ghshape(job, lines) = join(["$job\tRun package tests\t$TS $l" for l in lines], '\n')

const MAC_JOB = "Julia 1 - macos-latest - aarch64 - full"
const MAC_LINES = [
  "ProximalBase tests: tier full: 183 files; Julia 1.13.3 on arm64-apple-darwin27.0.0 (Apple M1 (Virtual), 3 threads), commit 45e3b75334d62a3e07d741a3bc028f07ecd5f4ac",
  "  active_handle affine allocations blockchecks_steps gpu precision_mixed three_factor_product",
  "Running 183 tests using 2 parallel jobs. To change the number of jobs, specify the `--jobs=N` argument to the tests, or set the `PTR_NUM_JOBS` environment variable.",
  "Available memory: \e[1m3.310 GiB\e[22m; Max worker RSS: \e[1m1.953 GiB\e[22m",
  "Test                             (Worker) │ time (s) │ time (s) │   (%)   │ GC (s) │ GC % │ Alloc (MB) │ RSS (MB) │",
  "three_factor_product                  (1) │         started at 2026-11-07T16:13:51.683\e[22m",
  "blockchecks_steps                     (2) │         started at 2026-11-07T16:13:51.715\e[22m",
  "\e[31mthree_factor_product                  (1) │    21.68 │     3.83 │    failed at 2026-11-07T16:14:14.899\e[39m",
  "gpu                                   (1) │         started at 2026-11-07T16:14:14.921\e[22m",
  "blockchecks_steps                     (2) │   330.04 │     3.39 │   98.67 │   8.04 │  2.4 │   21658.44 │  1094.73 │",
  "precision_mixed                       (2) │         started at 2026-11-07T16:19:23.396\e[22m",
  "gpu                                   (1) │   319.63 │     0.43 │   96.15 │  10.32 │  3.2 │   25258.26 │  1258.69 │",
  "precision_mixed                       (2) │  1500.12 │     2.10 │   97.20 │  12.00 │  0.8 │   30000.00 │  1500.00 │",
  "\e[0m\e[1mTest Summary:                                                                    | \e[22m\e[32m\e[1m   Pass  \e[22m\e[39m\e[91m\e[1mFail  \e[22m\e[39m",
  "    three_factor_product                                                         | \e[32m  31002  \e[39m\e[91m   1  \e[39m   21.7s",
  "    \e[31m\e[1mFAILURE\e[39m\e[22m",
  "",
  "Error in testset $TESTSET:",
  "\e[91m\e[1mTest Failed\e[22m\e[39m at \e[39m\e[1m/Users/runner/work/ProximalBase.jl/ProximalBase.jl/test/three_factor_product.jl:973\e[22m",
  "  Expression: a1 <= sizeof(T) * (p * p + 3p) + 1024",
  "   Evaluated: 9184 <= 8944",
  "\e[91m\e[1mERROR: \e[22m\e[39mLoadError: \e[91mTest run finished with errors\e[39m",
  "in expression starting at /Users/runner/work/ProximalBase.jl/ProximalBase.jl/test/runtests.jl:86",
]
const WIN_JOB = "Julia 1 - windows-latest - x64 - full"
const WIN_LINES = [
  "ProximalBase tests: tier full: 183 files; Julia 1.13.3 on x86_64-w64-mingw32 (INTEL(R) XEON(R) PLATINUM 8573C, 4 threads), commit 45e3b75334d62a3e07d741a3bc028f07ecd5f4ac",
  "  active_handle affine allocations blockchecks_steps gpu precision_mixed three_factor_product",
  "three_factor_product                  (1) │\e[2m         started at 2026-11-07T16:14:49.349\e[22m",
  "gpu                                   (3) │\e[2m         started at 2026-11-07T16:14:50.478\e[22m",
  "\e[31mthree_factor_product                  (1) │    34.91 │     5.86 │    failed at 2026-11-07T16:15:26.922\e[39m",
  "gpu                                   (3) │   400.10 │     0.55 │   95.00 │  10.00 │  2.5 │   20000.00 │  900.00 │",
  "Error in testset $TESTSET:",
  "\e[91m\e[1mTest Failed\e[22m\e[39m at \e[39m\e[1mD:\\a\\ProximalBase.jl\\ProximalBase.jl\\test\\three_factor_product.jl:972\e[22m",
  "  Expression: a1 == a2 == a0",
  "   Evaluated: 4279 == 4263 == 4263",
  "\e[91m\e[1mERROR: \e[22m\e[39mLoadError: \e[91mTest run finished with errors\e[39m",
  "in expression starting at D:\\a\\ProximalBase.jl\\ProximalBase.jl\\test\\runtests.jl:86",
]

# The same lines in the other shapes a log comes in.
plain(lines) = [replace(l, r"\e\[[0-9;]*[A-Za-z]" => "") for l in lines]
stamped(lines) = ["$TS $l" for l in lines]
rawlog(lines) = join(lines, '\n')

# What matters of a parsed log, comparable with ==.
digest(log) = (log.recognized, log.header, log.label, log.count, log.files, log.status,
               [(f.file, f.testset, f.kind, f.signature, f.excerpt) for f in log.failures])

@testset "the macOS job of the real run" begin
  log = parse_log(ghshape(MAC_JOB, MAC_LINES))
  @test log.recognized && log.header && log.headers == 1
  @test log.label == "tier full" && log.count == 183
  @test log.files == ["active_handle", "affine", "allocations", "blockchecks_steps", "gpu", "precision_mixed", "three_factor_product"]
  @test log.status == Dict("three_factor_product" => :failed, "blockchecks_steps" => :passed, "gpu" => :passed,
                           "precision_mixed" => :passed)
  f = only(log.failures)
  @test f.file == "three_factor_product" && f.testset == TESTSET && f.kind == :test
  @test f.signature == "a# <= sizeof(T) * (p * p + #p) + #"
  @test startswith(f.excerpt, "Test Failed at /Users/runner/work/ProximalBase.jl/ProximalBase.jl/test/three_factor_product.jl:973\n")
  @test occursin("  Expression: a1 <= sizeof(T) * (p * p + 3p) + 1024\n   Evaluated: 9184 <= 8944", f.excerpt)
  # the block ends before the next error line
  @test !occursin("LoadError", f.excerpt)
end

@testset "the Windows job of the real run" begin
  log = parse_log(ghshape(WIN_JOB, WIN_LINES))
  @test log.recognized && log.count == 183
  @test log.status == Dict("three_factor_product" => :failed, "gpu" => :passed)
  f = only(log.failures)
  # the backslashes of D:\a\...\test\three_factor_product.jl do not matter
  @test f.file == "three_factor_product" && f.testset == TESTSET && f.kind == :test
  @test f.signature == "a# == a# == a#"
  @test startswith(f.excerpt, "Test Failed at D:\\a\\ProximalBase.jl\\ProximalBase.jl\\test\\three_factor_product.jl:972\n")
  @test occursin("   Evaluated: 4279 == 4263 == 4263", f.excerpt)
end

@testset "every shape of a log gives the same result" begin
  for (lines, job) in ((MAC_LINES, MAC_JOB), (WIN_LINES, WIN_JOB))
    want = digest(parse_log(ghshape(job, lines)))
    @test digest(parse_log(rawlog(plain(lines)))) == want                       # raw text
    @test digest(parse_log(rawlog(lines))) == want                              # colors
    @test digest(parse_log(rawlog(stamped(lines)))) == want                     # Actions API timestamps
    @test digest(parse_log(rawlog(stamped(plain(lines))))) == want
    @test digest(parse_log(ghshape(job, plain(lines)))) == want                 # gh, no colors
    @test digest(parse_log(join(lines, "\r\n"))) == want                        # CRLF
    @test digest(parse_log(join(stamped(lines), "\r\n") * "\r\n")) == want
    @test digest(parse_log(Vector{UInt8}(codeunits(rawlog(lines))))) == want    # the bytes of a file
    @test digest(parse_log("\ufeff" * rawlog(stamped(lines)))) == want          # a byte order mark
  end
  @test clean_line("\ufeff2026-11-07T16:14:14.9228380Z \e[31mred\e[39m text\r") == "red text"
  @test clean_line("job - name\tstep\t2026-11-07T16:14:14Z   indented") == "  indented"
  # a tab in the text of a raw line is not a column
  @test clean_line("a\tb\tc") == "a\tb\tc"
end

@testset "an exception, a test that threw, and a crashed file" begin
  lines = [
    "ProximalBase tests: tier full: 3 files; Julia 1.10.12 on x86_64-linux-gnu (cpu, 4 threads)",
    "  conformance/kernels reductions solver_cache",
    "Test                             (Worker) │ time (s) │",
    "conformance/kernels                   (1) │         started at 2026-11-07T16:14:49.349",
    "reductions                            (2) │         started at 2026-11-07T16:14:49.400",
    "solver_cache                          (3) │         started at 2026-11-07T16:14:49.500",
    "conformance/kernels                   (1) │    12.00 │     3.00 │    failed at 2026-11-07T16:15:26.922",
    "reductions                            (2) │    14.00 │     3.00 │    failed at 2026-11-07T16:15:27.922",
    "solver_cache                          (3) │ crashed at 2026-11-07T16:20:00.123",
    "Error in testset kernels:",
    "Error During Test at /home/runner/work/ProximalBase.jl/ProximalBase.jl/test/conformance/kernels.jl:57",
    "  Got exception outside of a @test",
    "  MethodError: no method matching foo(::Int64, ::Vector{Float64})",
    "  Closest candidates are:",
    "    foo(::Int64) at /home/runner/work/ProximalBase.jl/ProximalBase.jl/src/x.jl:3",
    "  Stacktrace:",
    "   [1] top-level scope",
    "     @ /home/runner/work/ProximalBase.jl/ProximalBase.jl/test/conformance/kernels.jl:57",
    "",
    "Error in testset reductions:",
    "Error During Test at /home/runner/work/ProximalBase.jl/ProximalBase.jl/test/reductions.jl:12",
    "  Test threw exception",
    "  Expression: f(x)",
    "  ErrorException: boom 0x00007f3a1c2b4d10 after 12 s",
    "  Stacktrace:",
  ]
  log = parse_log(rawlog(lines))
  @test log.status == Dict("conformance/kernels" => :failed, "reductions" => :failed, "solver_cache" => :crashed)
  @test length(log.failures) == 3
  crash = only(f for f in log.failures if f.kind == :crash)
  @test crash.file == "solver_cache" && crash.testset == "" && crash.signature == "crashed at #-#-#T#:#:#.#"
  @test occursin("crashed at 2026-11-07T16:20:00.123", crash.excerpt)
  ex = only(f for f in log.failures if f.file == "conformance/kernels")
  @test ex.kind == :exception && ex.testset == "kernels"
  # the first line of the exception, without numbers, paths and addresses
  @test ex.signature == "MethodError: no method matching foo(::Int#, ::Vector{Float#})"
  # the stack trace is part of the excerpt, cut at 12 lines
  @test startswith(ex.excerpt, "Error During Test at ") && occursin("Closest candidates are:", ex.excerpt)
  @test length(split(ex.excerpt, '\n')) <= 12
  # an exception inside a test names the exception, not the expression
  thrown = only(f for f in log.failures if f.file == "reductions")
  @test thrown.kind == :exception && thrown.testset == "reductions" && thrown.signature == "ErrorException: boom after # s"
end

@testset "the excerpt is bounded" begin
  long = ["  line $i " * "x" ^ 200 for i in 1:40]
  log = parse_log(rawlog(["Error in testset t:", "Test Failed at /w/ProximalBase.jl/test/a.jl:5", "  Expression: x", long...]))
  f = only(log.failures)
  @test length(split(f.excerpt, '\n')) <= 12 && length(f.excerpt) <= 1000
  @test f.file == "a" && f.signature == "x"
end

@testset "a file that started and never finished" begin
  # a log cut by a time limit: no summary, no failure block
  lines = [
    "ProximalBase tests: tier full: 183 files; Julia 1.13.3 on arm64-apple-darwin27.0.0 (Apple M1 (Virtual), 3 threads)",
    "  gpu precision_mixed three_factor_product",
    "gpu                                   (1) │         started at 2026-11-07T16:14:14.921",
    "three_factor_product                  (2) │         started at 2026-11-07T16:14:14.925",
    "three_factor_product                  (2) │    21.68 │     3.83 │   98.67 │   8.04 │  2.4 │   2165.44 │  1094.73 │",
    "precision_mixed                       (2) │         started at 2026-11-07T16:19:23.396",
  ]
  log = parse_log(rawlog(lines))
  @test log.recognized
  @test log.status == Dict("gpu" => :unfinished, "three_factor_product" => :passed, "precision_mixed" => :unfinished)
  @test isempty(log.failures)
end

@testset "a log that is not a test log" begin
  for text in ("", "\n\n", "Run actions/checkout@v7\nFetching the repository\n", "│ │ │", "started at 2026-11-07T16:14:14.921")
    log = parse_log(text)
    @test !log.recognized && !log.header && isempty(log.files) && isempty(log.status) && isempty(log.failures)
    @test log.count === nothing && log.headers == 0
  end
  # bytes that are not UTF-8 do not make it fail
  log = parse_log(UInt8[0xff, 0xfe, UInt8('\n'), 0x80, UInt8('x')])
  @test !log.recognized
  # failures are found without the header or the table
  log = parse_log(rawlog(plain(MAC_LINES[17:end])))
  @test !log.recognized && !log.header && isempty(log.files)
  f = only(log.failures)
  @test f.file == "three_factor_product" && f.testset == TESTSET && f.signature == "a# <= sizeof(T) * (p * p + #p) + #"
  # a header without names: the selection is empty, the header is known
  log = parse_log("ProximalBase tests: tier full: 5 files; Julia 1.13.3 on x\nnext line")
  @test log.header && log.recognized && log.count == 5 && isempty(log.files)
  # two headers: the first one is read, and the count of them is kept
  log = parse_log(rawlog([MAC_LINES[1:2]; MAC_LINES[1:2]]))
  @test log.headers == 2 && log.count == 183
end

@testset "the file of a failure and the signature" begin
  @test test_file("/Users/runner/work/ProximalBase.jl/ProximalBase.jl/test/three_factor_product.jl") == "three_factor_product"
  @test test_file("D:\\a\\ProximalBase.jl\\ProximalBase.jl\\test\\conformance\\x.jl") == "conformance/x"
  @test test_file("C:\\Users\\runner\\test\\a\\b\\test\\conformance\\x.jl") == "conformance/x"
  @test test_file("/home/runner/work/ProximalBase.jl/ProximalBase.jl/test/conformance/gpu_arrays.jl") == "conformance/gpu_arrays"
  @test test_file("test/utils.jl") == "utils"
  @test test_file("/home/u/.julia/packages/Foo/abc/src/foo.jl") == "foo"
  @test normalize_signature("a1 <= sizeof(T) * (p * p + 3p) + 1024") == "a# <= sizeof(T) * (p * p + #p) + #"
  @test normalize_signature("  x   ==\t y  ") == "x == y"
  @test normalize_signature("Error at C:\\Users\\x\\foo.jl:12 and /home/u/a/b.jl:3 ptr @0x00007f12") == "Error at and ptr"
  # division and fractions are not paths
  @test normalize_signature("a/b == x / y == (1)/2") == "a/b == x / y == (#)/#"
  # numbers and addresses vary from night to night; the rest does not
  @test normalize_signature("Evaluated: 9184 <= 8944 at 0xdeadbeef") == normalize_signature("Evaluated: 100 <= 99 at 0x1f")
end

# The text a runner prints for a selection.
headerlines(names; count = length(names), tier = "full") =
  ["ProximalBase tests: tier $tier: $count files; Julia 1.10.12 on x86_64-linux-gnu (cpu, 4 threads)", "  " * join(names, " ")]
header(names; kw...) = rawlog(headerlines(names; kw...))

@testset "the inventory of a tier" begin
  quick = CIResult.TestTiers.select(MANIFEST, "quick")
  full = CIResult.TestTiers.select(MANIFEST, "full")
  @test length(quick) < length(full) && "ci_result" in quick
  @test isempty(check_inventory(full, "full"; manifest = MANIFEST))
  @test isempty(check_inventory(reverse(full), "full"; manifest = MANIFEST)) # the order does not matter
  @test isempty(check_inventory(parse_log(header(full)), "full"; manifest = MANIFEST))
  @test isempty(check_inventory(parse_log(ghshape(MAC_JOB, plain(headerlines(full)))), "full"; manifest = MANIFEST))
  # a file that did not run
  p = check_inventory(full[2:end], "full"; manifest = MANIFEST)
  @test length(p) == 1 && occursin("1 test files of tier full did not run: $(full[1])", only(p))
  p = check_inventory(parse_log(header(full[3:end])), "full"; manifest = MANIFEST)
  @test any(occursin("2 test files of tier full did not run: $(full[1]), $(full[2])", s) for s in p)
  # a file the tier does not select
  p = check_inventory([full; "no_such_file"], "full"; manifest = MANIFEST)
  @test length(p) == 1 && occursin("1 test files ran that tier full does not select: no_such_file", only(p))
  p = check_inventory(parse_log(header(full)), "quick"; manifest = MANIFEST)
  @test length(p) == 1 && occursin("ran that tier quick does not select", only(p))
  # the count of the header and the names disagree
  p = check_inventory(parse_log(header(full; count = length(full) + 1)), "full"; manifest = MANIFEST)
  @test length(p) == 1 && occursin("header says $(length(full) + 1) files, but it lists $(length(full)) names", only(p))
  @test !isempty(check_inventory(full, "full"; count = length(full) - 1, manifest = MANIFEST))
  # a name twice makes the count right and the inventory wrong
  p = check_inventory(parse_log(header([full; full[1]])), "full"; manifest = MANIFEST)
  @test any(occursin("more than once: $(full[1])", s) for s in p)
  # no header, two headers, or a header without names
  p = check_inventory(parse_log("Run package tests\nnothing here"), "full"; manifest = MANIFEST)
  @test length(p) == 1 && occursin("no line `ProximalBase tests:", only(p))
  p = check_inventory(parse_log(header(full) * "\n" * header(full)), "full"; manifest = MANIFEST)
  @test length(p) == 1 && occursin("has 2 lines", only(p))
  p = check_inventory(parse_log("ProximalBase tests: tier full: $(length(full)) files; Julia 1.10.12 on x\n"), "full"; manifest = MANIFEST)
  @test any(occursin("did not run", s) for s in p) && any(occursin("header says", s) for s in p)
  # the extra files of a focus run join the tier, and only then
  extras = ["gpu"]
  @test "gpu" in full && !("gpu" in quick)
  @test isempty(check_inventory(quick, "quick"; manifest = MANIFEST))
  @test isempty(check_inventory([quick; extras], "quick"; extra = extras, manifest = MANIFEST))
  @test !isempty(check_inventory(quick, "quick"; extra = extras, manifest = MANIFEST))
  @test !isempty(check_inventory([quick; extras], "quick"; manifest = MANIFEST))
  @test isempty(check_inventory(parse_log(header([quick; extras]; tier = "quick")), "quick"; extra = extras, manifest = MANIFEST))
  @test_throws ArgumentError check_inventory(quick, "huge"; manifest = MANIFEST)
  # a long list is cut in the message
  p = check_inventory(String[], "full"; manifest = MANIFEST)
  @test any(occursin("... ($(length(full)) in all)", s) for s in p)
end

@testset "the command line" begin
  full = CIResult.TestTiers.select(MANIFEST, "full")
  quick = CIResult.TestTiers.select(MANIFEST, "quick")
  function run_main(args)
    io = IOBuffer()
    status = redirect_stderr(devnull) do
      CIResult.main(args; io)
    end
    return status, String(take!(io))
  end
  mktempdir() do dir
    logfile(name, text) = (path = joinpath(dir, name); write(path, text); path)
    good = logfile("good.log", ghshape(WIN_JOB, plain(headerlines(quick; tier = "quick"))))
    status, out = run_main(["check-inventory", good, "--tier=quick"])
    @test status == 0 && !occursin("::error::", out) && occursin("every test file of tier quick", out)
    # a missing file, an extra file and a count: one error line each, percent-encoded commands
    bad = logfile("bad.log", header([quick[2:end]; "no_such_file"]; count = length(quick)))
    status, out = run_main(["check-inventory", bad, "--tier=quick"])
    errors = filter(!isempty, split(out, '\n'))
    @test status == 1 && length(errors) == 2 && all(startswith("::error::"), errors)
    @test occursin("1 test files of tier quick did not run: $(quick[1])", errors[1]) && occursin("no_such_file", errors[2])
    # the extra files
    withgpu = logfile("gpu.log", header([quick; "gpu"]; tier = "quick with gpu"))
    @test first(run_main(["check-inventory", withgpu, "--tier=quick", "--extra=gpu"])) == 0
    @test first(run_main(["check-inventory", withgpu, "--tier=quick"])) == 1
    @test first(run_main(["check-inventory", good, "--tier=quick", "--extra=gpu"])) == 1
    # no header is a problem
    nohdr = logfile("none.log", "Run package tests\nno header at all\n")
    status, out = run_main(["check-inventory", nohdr, "--tier=quick"])
    @test status == 1 && startswith(out, "::error::the test log has no line `ProximalBase tests:") && count(==('\n'), out) == 1
    # usage errors and unreadable input are status 2, not a problem of the log
    @test first(run_main(String[])) == 2
    @test first(run_main(["check-log", good])) == 2
    @test first(run_main(["check-inventory", good])) == 2
    @test first(run_main(["check-inventory", "--tier=quick"])) == 2
    @test first(run_main(["check-inventory", good, nohdr, "--tier=quick"])) == 2
    @test first(run_main(["check-inventory", good, "--tier=quick", "--bogus"])) == 2
    @test first(run_main(["check-inventory", joinpath(dir, "absent.log"), "--tier=quick"])) == 2
    @test first(run_main(["check-inventory", good, "--tier=huge"])) == 2
    # the script itself: its exit status is what the workflow step sees
    exitcode(args) = run(pipeline(ignorestatus(`$(Base.julia_cmd()) --startup-file=no $SCRIPT $args`); stdout = devnull, stderr = devnull)).exitcode
    @test exitcode(["check-inventory", good, "--tier=quick"]) == 0
    @test exitcode(["check-inventory", bad, "--tier=quick"]) == 1
    @test exitcode(["check-inventory"]) == 2
  end
  @test error_line("a%b\r\nc") == "::error::a%25b%0D%0Ac"
end

end
