module TierSelectionTest

# The tier manifest, test selection and the CI risk policy (test/tiers.jl, test/tiers.toml).

using Test
import ParallelTestRunner

include(joinpath(@__DIR__, "tiers.jl"))
using .TestTiers: TestTiers, TIERS, ManifestError

const DIR = @__DIR__
const MANIFEST = TestTiers.load(joinpath(DIR, "tiers.toml"), joinpath(dirname(DIR), ".github", "tier-rules.toml"))

@testset "the manifest covers every test file" begin
  found = TestTiers.discover(DIR)
  @test Set(found) == Set(keys(ParallelTestRunner.find_tests(DIR)))
  @test TestTiers.check(MANIFEST, found) === nothing
  # full runs every file except the helpers and the separate environments
  tests = filter(n -> !(n in MANIFEST.helpers || TestTiers.isenvironment(MANIFEST, n)), found)
  @test TestTiers.select(MANIFEST, "full") == sort(tests)
  @test any(startswith("cuda/"), found)  # cuda/instantiate.jl; runtests.jl files are never found
  @test !any(n -> startswith(n, "cuda/") || startswith(n, "oracles/"), TestTiers.select(MANIFEST, "full"))
end

@testset "tiers are cumulative and each adds files" begin
  selections = [TestTiers.select(MANIFEST, t) for t in TIERS]
  for (smaller, larger) in zip(selections, selections[2:end])
    @test issubset(smaller, larger)
    @test length(smaller) < length(larger)
  end
  @test !isempty(first(selections))
  @test_throws ArgumentError TestTiers.select(MANIFEST, "fast")
  @test_throws ArgumentError TestTiers.select(MANIFEST, "")
end

const SMALL = """
  [tiers]
  quick = ["a"]
  standard = ["b"]
  thorough = ["c"]
  full = ["d/e"]
  [not_tests]
  helpers = ["h"]
  environments = ["env"]
  """

problems(f) = try
  f()
  String[]
catch err
  err isa ManifestError || rethrow()
  err.problems
end

@testset "manifest problems are reported" begin
  m = TestTiers.parse_manifest(SMALL)
  @test TestTiers.select(m, "quick") == ["a"]
  @test TestTiers.select(m, "thorough") == ["a", "b", "c"]
  @test TestTiers.select(m, "full") == ["a", "b", "c", "d/e"]
  @test TestTiers.check(m, ["a", "b", "c", "d/e", "h", "env/runtests2", "env/x/y"]) === nothing
  # a new file without a tier
  p = problems(() -> TestTiers.check(m, ["a", "b", "c", "d/e", "h", "new"]))
  @test length(p) == 1 && occursin("`test/new.jl` has no tier", p[1])
  # a listed file that does not exist, and a missing helper
  p = problems(() -> TestTiers.check(m, ["a", "b", "c"]))
  @test any(occursin("`d/e` is in tier `full`", s) for s in p)
  @test any(occursin("helper `h` does not exist", s) for s in p)
  # a file in two tiers, an empty tier, an unknown tier
  p = problems(() -> TestTiers.parse_manifest(replace(SMALL, "standard = [\"b\"]" => "standard = [\"a\"]")))
  @test any(occursin("`a` is listed in both `quick` and `standard`", s) for s in p)
  p = problems(() -> TestTiers.parse_manifest(replace(SMALL, "thorough = [\"c\"]" => "thorough = []")))
  @test any(occursin("tier `thorough` lists no test files", s) for s in p)
  p = problems(() -> TestTiers.parse_manifest(SMALL * "fast = [\"z\"]\n"))
  @test isempty(p)  # a key after [not_tests] belongs to that table
  p = problems(() -> TestTiers.parse_manifest(replace(SMALL, "[not_tests]" => "fast = [\"z\"]\n[not_tests]")))
  @test any(occursin("unknown tier `fast`", s) for s in p)
  p = problems(() -> TestTiers.parse_manifest(SMALL * "[[risk]]\ntier = \"fast\"\npaths = [\"x\"]\n"))
  @test any(occursin("risk rule 1 has tier `fast`", s) for s in p)
  p = problems(() -> TestTiers.parse_manifest(replace(SMALL, "helpers = [\"h\"]" => "helpers = [\"a\"]")))
  @test any(occursin("`a` is both a helper and in tier `quick`", s) for s in p)
  # a focus file that no tier lists, a bad timeout
  p = problems(() -> TestTiers.parse_manifest(SMALL * "[[risk]]\ntier = \"quick\"\npaths = [\"x\"]\nfocus = [\"zz\"]\n"))
  @test any(occursin("focus `zz`, which is in no tier", s) for s in p)
  p = problems(() -> TestTiers.parse_manifest(SMALL * "[[risk]]\ntier = \"standard\"\nnarrow = \"standard\"\npaths = [\"x\"]\n"))
  @test any(occursin("narrow tier `standard`; use a tier below", s) for s in p)
  p = problems(() -> TestTiers.parse_manifest(SMALL * "[timeouts]\nquick = 0\n"))
  @test any(occursin("timeouts.quick", s) for s in p)
  @test MANIFEST.timeouts == Dict("quick" => 20, "standard" => 30, "thorough" => 45, "full" => 90)
end

# SMALL with more helpers, and with the test files that run two of them.
const HELPERS = """
  [tiers]
  quick = ["a"]
  standard = ["b"]
  thorough = ["c"]
  full = ["d/e"]
  [not_tests]
  helpers = ["h", "h2", "h3", "x/h"]
  environments = ["env"]
  """
const RUN_THROUGH = HELPERS * "run_through = { h = [\"a\"], h2 = [\"c\", \"a\"] }\n"

@testset "the test files that run each helper" begin
  @test isempty(TestTiers.parse_manifest(SMALL).run_through)
  @test TestTiers.parse_manifest(RUN_THROUGH).run_through == Dict("h" => ["a"], "h2" => ["c", "a"])
  runs(text) = problems(() -> TestTiers.parse_manifest(HELPERS * "[not_tests.run_through]\n" * text))
  # a helper under a directory is a quoted key
  @test isempty(runs("h = [\"a\"]\n\"x/h\" = [\"d/e\"]\n"))
  @test TestTiers.parse_manifest(HELPERS * "[not_tests.run_through]\n\"x/h\" = [\"d/e\"]\n").run_through == Dict("x/h" => ["d/e"])
  # a key that is no helper, a runner that is no test
  @test only(runs("zz = [\"a\"]\n")) == "not_tests.run_through has `zz`, which is not in not_tests.helpers"
  @test only(runs("h = [\"zz\"]\n")) == "not_tests.run_through: helper `h` runs through `zz`, which is in no tier"
  @test only(runs("h = [\"h2\"]\n")) == "not_tests.run_through: helper `h` runs through `h2`, which is a helper, not a test"
  for name in ("env/x", "env")
    @test only(runs("h = [\"$name\"]\n")) ==
          "not_tests.run_through: helper `h` runs through `$name`, which is in a directory with an environment of its own"
  end
  # an empty list, a string, a list with something other than a name
  for value in ("[]", "\"a\"", "[\"a\", 1]")
    @test only(runs("h = $value\n")) == "not_tests.run_through: helper `h` needs a non-empty list of test names"
  end
  @test only(runs("h = [\"a\", \"a\"]\n")) == "not_tests.run_through: helper `h` lists `a` more than once"
  p = problems(() -> TestTiers.parse_manifest(HELPERS * "run_through = [\"a\"]\n"))
  @test only(p) == "not_tests.run_through must be a table of helper = [test names]"
  # a key of [not_tests] written after the subtable belongs to the subtable
  misplaced = replace(HELPERS, "environments = [\"env\"]\n" => "") * "[not_tests.run_through]\nh = [\"a\"]\nenvironments = [\"env\"]\n"
  @test only(problems(() -> TestTiers.parse_manifest(misplaced))) ==
        "not_tests.run_through has `environments`, which is not in not_tests.helpers"
end

# What filter_tests! keeps for these names; check_selectors must agree with it on what matches.
function kept(names, positionals)
  suite = Dict{String, Any}(n => nothing for n in names)
  ParallelTestRunner.filter_tests!(suite, ParallelTestRunner.parse_args(positionals))
  return sort!(collect(keys(suite)))
end

@testset "test names that select no test file, exclusions that remove none" begin
  m = TestTiers.parse_manifest(RUN_THROUGH)
  runnable = ["a", "b", "c", "d/e"]
  available = ["a", "b"] # the tier left after the tier and --extra
  check(args...) = TestTiers.check_selectors(m, runnable, available, collect(args); scope = "tier standard")
  # names that match: nothing to say
  @test check() == (String[], String[])
  @test check("a") == check("a", "b", "!b") == check("!a") == check("") == (String[], String[])
  # a typo is not hidden by a name that matches; every one is listed once, in the order given
  errors, warnings = check("a", "ab", "zz", "ab")
  @test isempty(warnings) && length(errors) == 2
  @test startswith(errors[1], "`ab`: no test file starts with it") && startswith(errors[2], "`zz`:")
  @test length(check("zz")[1]) == 1 # and alone
  # a helper is no test, and the message names the test files listed for it, sorted, marking those
  # outside the tier; a helper without an entry names none
  @test only(check("a", "h")[1]) == "`h`: a helper that test files include, not a test; it runs through a"
  @test only(check("!h")[2]) == "`!h`: a helper that test files include, not a test; it runs through a"
  @test only(check("h2")[1]) == "`h2`: a helper that test files include, not a test; it runs through a, c (outside tier standard)"
  @test only(check("h3")[1]) == "`h3`: a helper that test files include, not a test"
  # a prefix of a helper is not the helper
  @test only(check("h_")[1]) == "`h_`: no test file starts with it"
  # a test outside the tier: an error that points to --tier and --extra, not a silent no match
  errors, _ = check("c", "d/")
  @test length(errors) == 2
  @test occursin("matches only tests outside tier standard (c)", errors[1]) && occursin("--tier", errors[1]) && occursin("--extra", errors[1])
  @test occursin("(d/e)", errors[2])
  # an exclusion that matches nothing warns and is no error
  errors, warnings = check("a", "!zz", "!c")
  @test isempty(errors) && length(warnings) == 2
  @test startswith(warnings[1], "`!zz`: no test file starts with it") && occursin("outside tier standard (c)", warnings[2])
  @test check("!zz") == (String[], ["`!zz`: no test file starts with it"])
  # an exclusion that matches an available test, but none that the included names select, removes nothing
  @test check("a", "!b") == (String[], ["`!b`: it matches b, which the included names (a) do not select; it removes nothing"])
  @test check("a", "!b", "!a", "!b") == (String[], ["`!b`: it matches b, which the included names (a) do not select; it removes nothing"])
  @test check("a", "b", "!b") == check("!b") == check("a", "!a") == (String[], String[]) # it removes a selected test, or all of them
  @test check("b", "a", "!b", "!a") == (String[], String[])
  # when every included name is an error the exclusions keep today's reasons: no new warning buries the errors
  errors, warnings = check("zz", "!b", "!c", "!yy")
  @test only(errors) == "`zz`: no test file starts with it" && length(warnings) == 2
  @test occursin("outside tier standard (c)", warnings[1]) && startswith(warnings[2], "`!yy`: no test file starts with it")
  # an --extra test is available, and is dropped unless an included name selects it
  extra = ["a", "b", "d/e"]
  check_extra(args...) = TestTiers.check_selectors(m, runnable, extra, collect(args); scope = "tier standard")
  @test check_extra("a", "!d/") == (String[], ["`!d/`: it matches d/e, which the included names (a) do not select; it removes nothing"])
  @test check_extra("a", "d/", "!d/") == check_extra("!d/") == (String[], String[])
  # the matches and the included names are cut to three
  wide = ["a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4"]
  errors, warnings = TestTiers.check_selectors(m, wide, wide, ["b1", "b2", "b3", "b4", "!a"])
  @test isempty(errors) && only(warnings) ==
        "`!a`: it matches a1, a2, a3, ..., which the included names (b1, b2, b3, ...) do not select; it removes nothing"
  # more than three tests outside the tier are cut short
  errors, _ = TestTiers.check_selectors(m, ["a", "t1", "t2", "t3", "t4"], ["a"], ["t"])
  @test occursin("(t1, t2, t3, ...)", only(errors))
end

@testset "test names of the real catalog" begin
  runnable = TestTiers.select(MANIFEST, "full")
  quick = TestTiers.select(MANIFEST, "quick")
  check(args...; available = runnable) = TestTiers.check_selectors(MANIFEST, runnable, available, collect(args); scope = "tier quick")
  # the reported case: a helper next to a test that runs it
  errors, warnings = check("second_order", "gpu_smoke", "gpu_suite_basics")
  @test isempty(warnings) && only(errors) == "`gpu_suite_basics`: a helper that test files include, not a test; it runs through gpu_smoke"
  # every helper: the test files listed for it in tiers.toml, with the whole catalog available
  # (first) and with tier quick (second), where a test outside the tier is marked
  messages = [
    "conformance/common" => ("`conformance/common`: a helper that test files include, not a test",
                             "`conformance/common`: a helper that test files include, not a test"),
    "gpu_ambiguities" => ("`gpu_ambiguities`: a helper that test files include, not a test; it runs through gpu_extension_ambiguities, gpu_smoke",
                          "`gpu_ambiguities`: a helper that test files include, not a test; it runs through gpu_extension_ambiguities, gpu_smoke (outside tier quick)"),
    "gpu_prelude" => ("`gpu_prelude`: a helper that test files include, not a test; it runs through gpu, gpu_smoke",
                      "`gpu_prelude`: a helper that test files include, not a test; it runs through gpu (outside tier quick), gpu_smoke (outside tier quick)"),
    "gpu_suite" => ("`gpu_suite`: a helper that test files include, not a test; it runs through gpu",
                    "`gpu_suite`: a helper that test files include, not a test; it runs through gpu (outside tier quick)"),
    "gpu_suite_basics" => ("`gpu_suite_basics`: a helper that test files include, not a test; it runs through gpu_smoke",
                           "`gpu_suite_basics`: a helper that test files include, not a test; it runs through gpu_smoke (outside tier quick)"),
    "testutils" => ("`testutils`: a helper that test files include, not a test",
                    "`testutils`: a helper that test files include, not a test"),
    "tiers" => ("`tiers`: a helper that test files include, not a test; it runs through tier_selection",
                "`tiers`: a helper that test files include, not a test; it runs through tier_selection"),
  ]
  @test sort(first.(messages)) == sort(MANIFEST.helpers)
  for (helper, (whole, inquick)) in messages
    @test only(check(helper)[1]) == whole
    @test only(check(helper; available = quick)[1]) == inquick
  end
  # a directory with an environment of its own
  for name in ("cuda/runtests", "cuda", "oracles/")
    @test only(check(name)[1]) == "`$name`: a directory with an environment of its own that is run by hand, not a test"
  end
  # --core has other advice than a higher tier, which it cannot be combined with
  errors, _ = TestTiers.check_selectors(MANIFEST, runnable, quick, ["second_order"]; scope = "--core", hint = "run without --core")
  @test only(errors) == "`second_order`: it matches only tests outside --core (second_order); run without --core"
  # a typo and a name that the tier drops
  @test startswith(only(check("second_order", "second_ordr")[1]), "`second_ordr`: no test file starts")
  errors, _ = check("second_order", "lapack"; available = quick)
  @test startswith(only(errors), "`second_order`: it matches only tests outside tier quick (second_order)")
  # the device-array suites are outside tier quick, and the message says so
  errors, _ = check("gpu_smoke", "lapack"; available = quick)
  @test startswith(only(errors), "`gpu_smoke`: it matches only tests outside tier quick (gpu_smoke)")
  @test check("!gpu_smoek") == (String[], ["`!gpu_smoek`: no test file starts with it"])
  @test check("!gpu_smok") == (String[], String[]) # a prefix of gpu_smoke: it excludes that file
  # prefixes keep selecting: a directory, a short name, an exact name, with an exclusion
  @test check("conformance/", "second", "gpu_smoke", "!gpu_smoke") == (String[], String[])
  @test check("conformance/", "!conformance/second_order") == (String[], String[])
  # an exclusion that removes nothing the included names selected
  @test check("allocations_basics", "!lapack") ==
        (String[], ["`!lapack`: it matches lapack, which the included names (allocations_basics) do not select; it removes nothing"])
  @test check("allocations_basics", "!lapack"; available = quick) == check("allocations_basics", "!lapack")
  @test only(check("conformance/", "!lapack")[2]) ==
        "`!lapack`: it matches lapack, which the included names (conformance/) do not select; it removes nothing"
  @test check("conformance/", "!conformance/second_order") == check("", "!lapack") == check("!lapack") == (String[], String[])
  @test check("gpu_smoke", "!gpu_smoke") == (String[], String[]) # it removes every selected test: the runner stops on that
  # every included name an error: the errors come first and the exclusions keep their reasons
  errors, warnings = check("zz", "!lapack", "!zy")
  @test only(errors) == "`zz`: no test file starts with it" && only(warnings) == "`!zy`: no test file starts with it"
  # an --extra test is dropped unless an included name selects it
  errors, warnings = check("allocations_basics", "!second_order"; available = [quick; "second_order"])
  @test isempty(errors) && only(warnings) ==
        "`!second_order`: it matches second_order, which the included names (allocations_basics) do not select; it removes nothing"
  # the check agrees with filter_tests! on which names match: an include is flagged exactly when the
  # real function keeps nothing for it, an exclusion when it removes nothing that the includes
  # (every name without `!`, the erroring ones too) keep; if every include is an error, when it
  # removes nothing from all the tests
  for names in (["gpu_smoke", "second_order"], ["gpu_suite_basics", "second_ordr", "conformance/"], ["!gpu_smoek", "lapack"],
                ["tier_sel", "x"], ["!tier_selection", "!"], ["!!zz", "!!lapack", "", "!gpu_smok"], ["!x", "!lapack", "conformance"],
                ["gpu_smoke", "!lapack"], ["conformance/", "!conformance/second_order"], ["gpu_smoke", "!gpu_smoke"], ["zz", "!lapack"],
                ["!", "gpu_smoke"], ["!!lapack", "gpu_smoke"], ["", "!lapack"], ["gpu_smoke", "!lapack", "!lapack"])
    for available in (runnable, quick)
      errors, warnings = check(names...; available)
      flagged(msgs) = [first(split(m, "`:"; limit = 2))[2:end] for m in msgs]
      @test flagged(errors) == [n for n in unique(names) if !startswith(n, "!") && isempty(kept(available, [n]))]
      includes = filter(n -> !startswith(n, "!"), names)
      removes_nothing(n) = isempty(kept(available, includes)) ? kept(available, [n]) == sort(available) :
                           kept(available, [includes; n]) == kept(available, includes)
      @test flagged(warnings) == [n for n in unique(names) if startswith(n, "!") && removes_nothing(n)]
    end
  end
end

# The test files recorded in tiers.toml for each helper against the sources, both ways; `source(name)`
# is the text of test/<name>.jl, a test or a helper, and `loaded` the helpers that the runner loads
# for every test file. A file is named by the quoted literal `"<basename>.jl"`. Forward: a recorded
# test names the helper's file, or names a helper whose source names it (one level); this checks
# only that the literal appears, so a literal in a comment or a same-named file elsewhere also
# passes. Reverse: a test that names a helper's file is recorded for it, so a false match fails the
# test and never reaches a message. A test that includes a helper through another path form, such
# as "dir/h.jl" or "../h.jl", is not seen by the reverse check. Returns one string per mismatch.
function run_through_mismatches(m, source, loaded)
  literal(helper) = "\"$(basename(helper)).jl\""
  found = String[]
  for helper in sort!(collect(keys(m.run_through))), name in m.run_through[helper]
    text = source(name)
    through = [source(h) for h in m.helpers if occursin(literal(h), text)]
    occursin(literal(helper), text) || any(s -> occursin(literal(helper), s), through) ||
      push!(found, "`$name` is recorded for helper `$helper` but names neither its file nor a helper that names it")
  end
  for name in sort!(collect(keys(m.tier))), helper in setdiff(m.helpers, loaded)
    occursin(literal(helper), source(name)) && !(name in get(m.run_through, helper, String[])) &&
      push!(found, "`$name` names the file of helper `$helper` but is not recorded for it")
  end
  return found
end

@testset "the recorded test files that run each helper match the sources" begin
  source(name) = read(joinpath(DIR, name * ".jl"), String)
  # the runner loads these for every test file, so no single test runs them
  loaded = ["conformance/common", "testutils"]
  @test all(h -> occursin("\"$(basename(h)).jl\"", source("runtests")), loaded)
  @test !isempty(MANIFEST.run_through) && isempty(intersect(keys(MANIFEST.run_through), loaded))
  @test run_through_mismatches(MANIFEST, source, loaded) == String[]
  # synthetic sources: t_chain runs h2 only through mid, empty.jl is an empty helper, and the
  # runner loads loaded.jl
  sources = Dict("t_direct" => "include(\"h.jl\")\n", "t_chain" => "include(\"mid.jl\")\n",
                 "t_empty" => "include(\"empty.jl\")\n", "t_loaded" => "include(\"loaded.jl\")\n",
                 "t_other" => "# no include\n", "h" => "\n", "mid" => "include(\"h2.jl\")\n", "h2" => "\n",
                 "empty" => "", "loaded" => "\n")
  synthetic(entries) = TestTiers.parse_manifest("""
    [tiers]
    quick = ["t_direct", "t_chain"]
    standard = ["t_empty"]
    thorough = ["t_loaded"]
    full = ["t_other"]
    [not_tests]
    helpers = ["empty", "h", "h2", "loaded", "mid"]
    [not_tests.run_through]
    empty = ["t_empty"]
    mid = ["t_chain"]
    """ * entries)
  mismatches(entries, loaded = ["loaded"]) = run_through_mismatches(synthetic(entries), n -> sources[n], loaded)
  @test mismatches("h = [\"t_direct\"]\nh2 = [\"t_chain\"]\n") == String[]
  # a helper that the runner does not load is reported where a test names it without an entry
  @test mismatches("h = [\"t_direct\"]\nh2 = [\"t_chain\"]\n", String[]) ==
        ["`t_loaded` names the file of helper `loaded` but is not recorded for it"]
  # an empty helper file does not make a test run every helper
  @test mismatches("h = [\"t_direct\"]\nh2 = [\"t_chain\", \"t_empty\"]\n") ==
        ["`t_empty` is recorded for helper `h2` but names neither its file nor a helper that names it"]
  # a recorded test that names nothing, and a test that names a helper without an entry
  @test mismatches("h = [\"t_direct\", \"t_other\"]\nh2 = [\"t_chain\"]\n") ==
        ["`t_other` is recorded for helper `h` but names neither its file nor a helper that names it"]
  @test mismatches("h2 = [\"t_chain\"]\n") == ["`t_direct` names the file of helper `h` but is not recorded for it"]
end

# runtests.jl itself, as a process: the check runs before any worker starts and leaves --list alone.
# The selections are chosen so that none of them would run a test file if it were accepted.
@testset "runtests.jl rejects names that match no test file" begin
  julia = `$(Base.julia_cmd()) --startup-file=no --project=$(dirname(Base.active_project()))`
  function run_selection(args...)
    out, err = IOBuffer(), IOBuffer()
    p = run(pipeline(ignorestatus(`$julia $(joinpath(DIR, "runtests.jl")) $(collect(args))`); stdout = out, stderr = err))
    return success(p), String(take!(out)), String(take!(err))
  end
  # a name that matches next to one that does not: an error, and no run starts
  ok, out, err = run_selection("--tier=quick", "allocations_basics", "gpu_suite_basics", "second_ordr")
  @test !ok && !occursin("ProximalBase tests:", out)
  @test occursin("`gpu_suite_basics`: a helper that test files include, not a test; it runs through gpu_smoke (outside tier quick)", err)
  @test occursin("`second_ordr`: no test file starts with it", err) && occursin("scripts/test.jl --list", err)
  # an exclusion that removes nothing warns; the typo still stops the run
  ok, out, err = run_selection("--tier=quick", "allocations_basics", "!lapack", "second_ordr")
  @test !ok && !occursin("ProximalBase tests:", out)
  @test occursin("test exclusions that remove no test file:", err)
  @test occursin("`!lapack`: it matches lapack, which the included names (allocations_basics, second_ordr) do not select; it removes nothing", err)
  @test occursin("`second_ordr`: no test file starts with it", err)
  # a name that the tier drops
  ok, _, err = run_selection("--tier=quick", "allocations_basics", "second_order")
  @test !ok && occursin("`second_order`: it matches only tests outside tier quick", err)
  ok, _, err = run_selection("--tier=quick", "gpu_smoke")
  @test !ok && occursin("`gpu_smoke`: it matches only tests outside tier quick (gpu_smoke)", err)
  # --core has its own advice
  ok, _, err = run_selection("--core", "precision_mixed")
  @test !ok && occursin("outside --core (precision_mixed); run without --core", err)
  # --list shows the catalog, whatever the names
  ok, out, err = run_selection("--tier=quick", "--list", "zz", "gpu_suite_basics")
  @test ok && occursin("tier_selection", out) && !occursin("select no test file", err)
end

@testset "path patterns" begin
  matches(pattern, path) = occursin(TestTiers.glob_regex(pattern), path)
  @test matches("src/**", "src/functions/norms.jl")
  @test matches("src/*.jl", "src/utils.jl")
  @test !matches("src/*.jl", "src/functions/norms.jl")
  @test matches("**/*.md", "README.md") && matches("**/*.md", "test/oracles/README.md")
  @test !matches("**/*.md", "README.mdx")
  @test matches("docs/**", "docs/src/index.md") && !matches("docs/**", "docsx/a")
  @test matches("test/gpu_suite.jl", "test/gpu_suite.jl") && !matches("test/gpu_suite.jl", "test/gpu_suiteXjl")
end

classify(paths...) = TestTiers.classify(MANIFEST, collect(paths))[1:2]

# The grids files (test/TESTING.md) are in the full tier. A pull request adds one as a focus file
# for a change to the source of its own family (the table below); the tests check the focus of the
# listed and the named unrelated paths, not the selection for every source path. A later pull
# request adds a row to the table; the expectations below are derived from it.
#   grids:     the grids file (full tier);
#   base:      the file whose cases it repeats, and base_tier, the tier that file is in;
#   shared:    whether the rule for the code shared by every function (src/utils.jl, ...) lists it;
#   paths:     the source paths of its own family, each with the tier the rules give a change to it.
# The rows with base_tier "thorough" are listed by the two thorough rules, the rule for src/calculus.jl
# and the shared rule, and by no standard-tier rule: a pull request that changes only a standard-tier
# path did not run their base files, so the grids file would run the Float32 repeats without the Float64 cases they repeat.
const GRIDS = [
  (grids = "atom_iterate_grids", base = "atom_iterate", base_tier = "quick", shared = false,
   paths = ["src/atom_iterate.jl" => "standard", "src/functions/norms.jl" => "standard",
            "src/functions/sparsegroup.jl" => "standard", "src/functions/nonconvex.jl" => "standard"]),
  (grids = "conjugate_basics_grids", base = "conjugate_basics", base_tier = "quick", shared = true,
   paths = ["src/calculus.jl" => "thorough", "src/functions/losses.jl" => "standard"]),
  (grids = "constrained_basics_grids", base = "constrained_basics", base_tier = "quick", shared = false,
   paths = ["src/constrained.jl" => "standard"]),
  (grids = "lmo_atoms_grids", base = "lmo_atoms", base_tier = "quick", shared = true,
   paths = ["src/lmo.jl" => "standard", "src/lmo_atoms.jl" => "standard", "src/functions/singular.jl" => "standard"]),
  (grids = "nested_scaled_grids", base = "nested_scaled", base_tier = "quick", shared = true,
   paths = ["src/calculus.jl" => "thorough", "src/functions/singular.jl" => "standard"]),
  (grids = "scaled_infinite_level_underflow_grids", base = "scaled_infinite_level_underflow", base_tier = "thorough",
   shared = true, paths = ["src/calculus.jl" => "thorough"]),
  (grids = "sparse_iterate_basics_grids", base = "sparse_iterate_basics", base_tier = "quick", shared = false,
   paths = ["src/sparse_iterate.jl" => "standard"]),
  (grids = "symmetric_groups_grids", base = "symmetric_groups", base_tier = "thorough",
   shared = true, paths = ["src/calculus.jl" => "thorough"]),
  (grids = "zero_level_infinite_step_grids", base = "zero_level_infinite_step", base_tier = "thorough",
   shared = true, paths = ["src/calculus.jl" => "thorough"])]
# the grids files that the rule for the code shared by every function lists, and representatives of
# the paths of that rule: each gets SHARED_GRIDS
const SHARED_GRIDS = sort!([row.grids for row in GRIDS if row.shared])
const SHARED_GRIDS_PATHS = ["src/metric.jl", "src/utils.jl", "src/functions/spectral.jl"]
const ALL_GRIDS = sort!([row.grids for row in GRIDS])
# source path => (its tier, its grids focus files): the direct-family paths of the rows, and the
# representatives of the shared rule
const GRIDS_OF = let d = Dict{String,Tuple{String,Vector{String}}}()
  for row in GRIDS, (path, tier) in row.paths
    old = get(d, path, (tier, String[]))
    d[path] = (tier, sort!([old[2]; row.grids]))
    @assert old[1] == tier
  end
  for path in SHARED_GRIDS_PATHS
    d[path] = ("thorough", SHARED_GRIDS)
  end
  d
end

@testset "risk policy" begin
  # documentation needs no package tests
  @test classify("docs/src/index.md", "CONTRIBUTING.md") == ("none", String[])
  @test classify("benchmark/run.jl", "test/oracles/runtests.jl") == ("none", String[])
  # README examples are tests
  @test classify("README.md") == ("quick", ["readme"])
  # a changed test file runs beside the tier
  @test classify("test/norms.jl") == ("quick", ["norms"])
  @test classify("test/gpu_suite.jl") == ("quick", ["gpu", "gpu_smoke"])
  # a self-contained family with a test change: quick and its own tests
  @test classify("src/functions/entropy.jl", "test/entropy.jl") == ("quick", ["entropy"])
  @test classify("src/functions/cones.jl", "test/soc.jl") == ("quick", ["expcone", "rotatedsoc", "soc"])
  # package code without any test change: at least standard
  @test classify("src/functions/entropy.jl") == ("standard", ["entropy"])
  @test classify("src/functions/norms.jl", "test/norms.jl") == ("standard", ["atom_iterate_grids", "norms"])
  # a family whose own tests are in the full tier runs them beside standard
  @test classify("src/functions/sets.jl", "test/sets.jl") == ("standard", ["hyperslab", "sets"])
  @test classify("src/composition.jl") == ("standard", ["composition", "precompose_array"])
  # shared code
  @test classify("src/utils.jl")[1] == "thorough"
  @test classify("src/calculus.jl", "test/calculus.jl") ==
        ("thorough", sort!(["calculus", "stepchecks", GRIDS_OF["src/calculus.jl"][2]...]))
  @test classify("src/interface.jl", "docs/src/interface.md")[1] == "thorough"
  # the conformance suite and its coverage map
  # (its gpu_arrays reports are written beside the tier, so the check has the complete inventory)
  @test classify("test/conformance/kernels.jl") == ("thorough", ["conformance/gpu_arrays", "conformance/kernels"])
  @test classify("docs/design/numerical-contract/coverage.md") == ("thorough", ["conformance/gpu_arrays"])
  # infrastructure, platform code and paths no rule knows
  @test classify("src/lapack.jl")[1] == "full"
  @test classify("ext/ProximalBaseGPUArraysCoreExt.jl", "test/gpu.jl")[1] == "full"
  @test classify("test/runtests.jl")[1] == "full"
  @test classify(".github/tier-rules.toml")[1] == "full"
  @test classify("test/tiers.jl")[1] == "full"
  @test classify("scripts/ci_gate.jl")[1] == "full"
  @test classify("scripts/ci_verdict.jl")[1] == "full"
  # the tier lists decide what every tier runs, so any change to them gets full, also the line
  # of a new test file
  @test classify("test/tiers.toml")[1] == "full"
  @test classify("test/tiers.toml", "test/norms.jl") == ("full", ["norms"])
  # the API pages have @docs entries that test/docstrings.jl checks; prose needs no tests
  @test classify("docs/src/api/functions.md")[1] == "quick"
  @test classify("docs/src/numerics.md")[1] == "none"
  @test classify(".github/workflows/CI.yml")[1] == "full"
  @test classify("Project.toml")[1] == "full"
  @test classify("a/new/file.txt")[1] == "full"
  @test classify(".github/dependabot.yml")[1] == "full"
  # a deleted or unlisted test file is no focus file; the runner rejects an unlisted one
  @test classify("test/removed_file.jl") == ("quick", String[])
  # every rule decides the path it is written for
  for rule in MANIFEST.rules, pattern in rule.patterns
    occursin('*', pattern) && continue
    t, = TestTiers.classify(MANIFEST, [pattern])
    i = findfirst(r -> any(re -> occursin(re, pattern), r.regexes), MANIFEST.rules)
    @test MANIFEST.rules[i] === rule
  end
end


# The tier and focus files a PR gets, as the plan job reports them.
pr(paths...; labels = String[]) = TestTiers.pr_tier(MANIFEST, collect(paths), labels)[1:2]
const NARROW = [TestTiers.NARROW_LABEL]
const ATOM = ["atom_iterate_grids", "block_step", "infinite_group_weights", "norms", "real_iterates", "sparsegroup"]
const SPARSE = ["block_step", "conformance/failures_storage", "real_iterates", "sparse_iterate", "sparse_iterate_basics_grids",
                "sparse_value_cost", "symmetric_argument", "symmetric_mul"]
const SYMPARAMS = ["symmetric_parameters_extensions", "symmetric_parameters_merge", "symmetric_parameters_paths",
                   "symmetric_parameters_trait", "symmetric_parameters_writes"]
instandard(names) = filter(n -> !(n in TestTiers.select(MANIFEST, "standard")), names)

@testset "iterate policy" begin
  # a narrow change to one implementation, declared by the label: quick and its focus files,
  # also without a test change
  @test pr("src/atom_iterate.jl"; labels = NARROW) == ("quick", ATOM)
  @test pr("src/atom_iterate.jl", "test/atom_iterate.jl"; labels = NARROW) == ("quick", ATOM)
  @test pr("src/sparse_iterate.jl"; labels = NARROW) == ("quick", SPARSE)
  @test pr("src/symmetric_parameters.jl"; labels = NARROW) == ("quick", SYMPARAMS)
  # without the label: standard and the same focus files (those standard lacks)
  @test pr("src/atom_iterate.jl") == ("standard", instandard(ATOM))
  @test pr("src/sparse_iterate.jl", "test/sparse_iterate.jl") == ("standard", instandard(SPARSE))
  @test pr("src/symmetric_parameters.jl") == ("standard", SYMPARAMS)
  # two implementations: the label does not apply; standard and both groups
  @test pr("src/sparse_iterate.jl", "src/atom_iterate.jl"; labels = NARROW) ==
        ("standard", instandard(sort!(union(SPARSE, ATOM))))
  @test pr("src/sparse_iterate.jl", "src/symmetric_parameters.jl"; labels = NARROW)[1] == "standard"
  # a shared or architectural change gets test:thorough by hand; test:full always wins
  @test pr("src/sparse_iterate.jl"; labels = ["test:thorough"])[1] == "thorough"
  @test pr("src/sparse_iterate.jl"; labels = [NARROW; "test:thorough"])[1] == "thorough"
  @test pr("src/atom_iterate.jl"; labels = [NARROW; "test:full"]) == ("full", String[])
  @test pr("src/atom_iterate.jl"; labels = [NARROW; "test:standard"]) == ("standard", instandard(ATOM))
  # the label never lowers another path: shared code, infrastructure, unknown paths
  @test pr("src/sparse_iterate.jl", "src/utils.jl"; labels = NARROW)[1] == "thorough"
  @test pr("src/atom_iterate.jl", "src/interface.jl"; labels = NARROW)[1] == "thorough"
  @test pr("src/atom_iterate.jl", "Project.toml"; labels = NARROW)[1] == "full"
  @test pr("src/atom_iterate.jl", ".github/workflows/CI.yml"; labels = NARROW)[1] == "full"
  @test pr("src/atom_iterate.jl", "a/new/file.txt"; labels = NARROW)[1] == "full"
  # nor package code without tests: other source still needs standard, with or without a test
  # change elsewhere
  @test pr("src/atom_iterate.jl", "src/functions/entropy.jl"; labels = NARROW)[1] == "standard"
  @test pr("src/atom_iterate.jl", "src/functions/norms.jl", "test/norms.jl"; labels = NARROW)[1] == "standard"
  @test pr("src/functions/entropy.jl", "test/entropy.jl"; labels = NARROW) == ("quick", String[])
  @test pr("src/functions/norms.jl"; labels = NARROW) == pr("src/functions/norms.jl") == ("standard", ["atom_iterate_grids"])
  # the label only adds information: the same paths without it never get a lower tier, and the
  # focus files are the same
  for paths in (["src/atom_iterate.jl"], ["src/sparse_iterate.jl", "src/utils.jl"], ["src/functions/sets.jl"],
                ["docs/src/index.md"], ["src/atom_iterate.jl", "test/norms.jl"])
    t1, f1, _ = TestTiers.classify(MANIFEST, paths; narrow = true)
    t2, f2, _ = TestTiers.classify(MANIFEST, paths)
    @test TestTiers.rank(t1) <= TestTiers.rank(t2) && f1 == f2
  end
  # documentation keeps no tests with the label; an unknown test label is an error
  @test pr("docs/src/index.md"; labels = NARROW) == ("none", String[])
  @test_throws ArgumentError pr("src/atom_iterate.jl"; labels = ["test:iterate"])
  # the decision table says when and why the label applies
  _, _, d = TestTiers.classify(MANIFEST, ["src/atom_iterate.jl"]; narrow = true)
  @test occursin("narrow change", d[1][3])
  _, _, d = TestTiers.classify(MANIFEST, ["src/atom_iterate.jl", "src/sparse_iterate.jl"]; narrow = true)
  @test d[1][1] == "(label test:iterate-narrow)" && occursin("more than one iterate implementation", d[1][3])
  # the result does not depend on the order of the paths
  paths = ["src/sparse_iterate.jl", "test/norms.jl", "src/atom_iterate.jl", "README.md"]
  @test TestTiers.classify(MANIFEST, paths)[1:2] == TestTiers.classify(MANIFEST, reverse(paths))[1:2]
  # every focus file runs in full, and the iterate rules are the only ones the label can lower
  full = TestTiers.select(MANIFEST, "full")
  for rule in MANIFEST.rules
    @test issubset(rule.focus, full)
    isempty(rule.narrow) || @test all(p -> startswith(p, "src/") && occursin("iterate", p) || p == "src/symmetric_parameters.jl", rule.patterns)
  end
end

const GPU = ["conformance/gpu_arrays", "gpu", "gpu_smoke"]

@testset "the device-array suites run in the full tier and beside changes to the GPU files" begin
  # outside the default tiers, inside full, each listed once (the manifest checks reject a duplicate)
  for tier in ("quick", "standard", "thorough")
    @test isdisjoint(TestTiers.select(MANIFEST, tier), GPU)
  end
  @test issubset(GPU, TestTiers.select(MANIFEST, "full")) && all(n -> MANIFEST.tier[n] == "full", GPU)
  # the ambiguity guard for the GPU extension is a quick test file, run by every GPU run through its helper
  @test MANIFEST.tier["gpu_extension_ambiguities"] == "quick" && "gpu_ambiguities" in MANIFEST.helpers
  @test MANIFEST.run_through["gpu_ambiguities"] == ["gpu_extension_ambiguities", "gpu_smoke"]
  include_helper = "include(joinpath(@__DIR__, \"gpu_ambiguities.jl\"))"
  guard = read(joinpath(DIR, "gpu_extension_ambiguities" * ".jl"), String)
  @test occursin("Base.get_extension(ProximalBase, :ProximalBaseGPUArraysCoreExt) !== nothing", guard)
  @test occursin(include_helper, guard)
  @test occursin(include_helper, read(joinpath(DIR, "gpu_suite_basics" * ".jl"), String))
  # a change to a GPU test file, a GPU helper, the GPU extension, the coverage map or the conformance
  # scripts runs the GPU files that it can affect
  @test pr("test/gpu_smoke.jl") == ("quick", ["gpu_smoke"])
  @test pr("test/gpu.jl") == ("quick", ["gpu"])
  @test pr("test/conformance/gpu_arrays.jl") == ("thorough", ["conformance/gpu_arrays"])
  for helper in ("gpu_ambiguities", "gpu_prelude", "gpu_suite", "gpu_suite_basics")
    @test pr("test/$helper.jl") == ("quick", ["gpu", "gpu_smoke"])
  end
  @test pr("test/gpu_extension_ambiguities.jl") == ("quick", String[]) # a quick file itself
  @test pr("ext/ProximalBaseGPUArraysCoreExt.jl") == ("full", String[]) # full runs every file
  for path in ("docs/design/numerical-contract/coverage.md", "scripts/conformance_map.jl", "scripts/conformance_run.jl",
               "test/conformance/kernels.jl")
    @test pr(path) == ("thorough", ["conformance/gpu_arrays"])
  end
  # unrelated changes leave them to the nightly run
  for path in ("src/functions/norms.jl", "src/utils.jl", "src/composition.jl", "test/norms.jl", "README.md")
    @test isdisjoint(pr(path)[2], GPU)
  end
  @test pr("src/functions/norms.jl") == ("standard", ["atom_iterate_grids"])
  # the label test:gpu adds the three files to any change; labels never lower the tier
  @test pr("src/functions/norms.jl"; labels = ["test:gpu"]) == ("standard", ["atom_iterate_grids"; GPU])
  @test pr("src/utils.jl"; labels = ["test:gpu"]) == ("thorough", sort!([GPU; SHARED_GRIDS]))
  @test pr("src/utils.jl"; labels = ["test:gpu", "test:standard"]) == ("thorough", sort!([GPU; SHARED_GRIDS]))
  @test pr("src/functions/norms.jl"; labels = ["test:gpu", "test:thorough"]) == ("thorough", ["atom_iterate_grids"; GPU])
  @test pr("src/functions/norms.jl"; labels = ["test:full", "test:gpu"]) == ("full", String[])
  @test pr("src/functions/norms.jl"; labels = ["test:gpu", NARROW...]) == ("standard", ["atom_iterate_grids"; GPU])
  # a change that needs no package tests runs quick with the files (a plan with tier none runs nothing)
  _, _, _, notes = TestTiers.pr_tier(MANIFEST, ["docs/src/index.md"], ["test:gpu"])
  @test pr("docs/src/index.md"; labels = ["test:gpu"]) == ("quick", GPU)
  @test any(occursin("test:gpu", n) && occursin("gpu_smoke", n) for n in notes)
  @test pr("docs/src/index.md") == ("none", String[])
  @test pr("docs/src/index.md", "test/gpu_smoke.jl"; labels = ["test:gpu"]) == ("quick", GPU)
  @test pr(; labels = ["test:gpu"]) == ("quick", GPU) # no changed files: quick
  # the label names are labels: SubStrings, as split gives them, work, and an unknown label next to
  # test:gpu is still an error that lists test:gpu
  @test pr("src/utils.jl"; labels = split("test:gpu,bug", ',')) == ("thorough", sort!([GPU; SHARED_GRIDS]))
  err = try pr("src/utils.jl"; labels = ["test:gpu", "test:gpuu"]); nothing catch e; e end
  @test err isa ArgumentError && occursin("`test:gpuu` names no tier", err.msg) && occursin("test:gpu", err.msg)
  # the label table is part of the manifest: names, their focus files in tiers
  @test MANIFEST.label_focus == Dict("test:gpu" => ["gpu_smoke", "gpu", "conformance/gpu_arrays"])
  @test all(n -> haskey(MANIFEST.tier, n), reduce(vcat, values(MANIFEST.label_focus)))
end

@testset "label_focus in the manifest" begin
  manifest_with(text) = SMALL * text
  m = TestTiers.parse_manifest(manifest_with("[label_focus]\n\"test:x\" = [\"d/e\", \"a\"]\n"))
  @test m.label_focus == Dict("test:x" => ["d/e", "a"])
  @test isempty(TestTiers.parse_manifest(SMALL).label_focus)
  # the label adds its files, the tier and the other labels are as before
  @test TestTiers.pr_tier(m, ["src/x.jl"], ["test:x"])[1:2] == ("full", String[]) # no rule: full runs every file
  @test TestTiers.pr_tier(m, String[], ["test:x"])[1:2] == ("quick", ["d/e"])    # quick runs a
  problems_of(text) = problems(() -> TestTiers.parse_manifest(manifest_with(text)))
  @test only(problems_of("[label_focus]\n\"test:x\" = [\"zz\"]\n")) ==
        "label_focus: `test:x` has focus `zz`, which is in no tier"
  # a label of a tier, the narrow or the benchmark label, or without the prefix has its own meaning or none
  for label in ("test:quick", "test:standard", "test:thorough", "test:full", "test:iterate-narrow", "test:benchmark")
    @test occursin("label of a tier, of test:iterate-narrow or of test:benchmark", only(problems_of("[label_focus]\n\"$label\" = [\"a\"]\n")))
  end
  for label in ("gpu", "test:")
    @test occursin("is not a label of the form test:<name>", only(problems_of("[label_focus]\n\"$label\" = [\"a\"]\n")))
  end
  for value in ("[]", "\"a\"", "[\"a\", 1]")
    @test only(problems_of("[label_focus]\n\"test:x\" = $value\n")) == "label_focus: `test:x` needs a non-empty list of test names"
  end
  @test only(problems_of("[label_focus]\n\"test:x\" = [\"a\", \"a\"]\n")) == "label_focus: `test:x` lists `a` more than once"
  @test only(problems(() -> TestTiers.parse_manifest("label_focus = [\"a\"]\n" * SMALL))) ==
        "label_focus must be a table of label = [test names]"
  # a helper is no test file, so it is no focus file
  @test only(problems_of("[label_focus]\n\"test:x\" = [\"h\"]\n")) == "label_focus: `test:x` has focus `h`, which is in no tier"
end

@testset "additive changes to the tier lists" begin
  # the edits below work on LF text; a Windows checkout has CRLF line endings
  old = replace(read(joinpath(DIR, "tiers.toml"), String), "\r\n" => "\n")
  add(text, tier, name) = replace(text, "\n$tier = [\n" => "\n$tier = [\n  \"$name\",\n"; count = 1)
  # new files only: their names, which then run as changed test files
  @test TestTiers.manifest_additions(old, add(old, "quick", "iterate_new"), ["iterate_new"]) == (["iterate_new"], String[])
  # line endings do not matter
  @test TestTiers.manifest_additions(old, replace(add(old, "quick", "iterate_new"), "\n" => "\r\n"), ["iterate_new"]) ==
        (["iterate_new"], String[])
  two = add(add(old, "full", "b_new"), "standard", "a_new")
  @test TestTiers.manifest_additions(old, two, ["a_new", "b_new"]) == (["a_new", "b_new"], String[])
  # comments and order alone change nothing
  @test TestTiers.manifest_additions(old, replace(old, "# iterates:" => "# the iterates:"), String[]) == (String[], String[])
  swapped = replace(old, "  \"aqua\",\n  \"ci_gate\",\n" => "  \"ci_gate\",\n  \"aqua\",\n")
  @test swapped != old && TestTiers.manifest_additions(old, swapped, String[]) == (String[], String[])
  # anything else needs full
  a, p = TestTiers.manifest_additions(old, add(old, "quick", "iterate_new"), String[])
  @test a === nothing && occursin("not a new file of this change", only(p))
  moved = add(replace(old, "  \"utils\",\n" => ""; count = 1), "standard", "utils")
  a, p = TestTiers.manifest_additions(old, moved, String[])
  @test a === nothing && only(p) == "`utils` moves from `quick` to `standard`"
  a, p = TestTiers.manifest_additions(old, replace(old, "  \"utils\",\n" => ""; count = 1), String[])
  @test a === nothing && only(p) == "`utils` is removed from tier `quick`"
  a, p = TestTiers.manifest_additions(old, add(replace(old, "  \"utils\",\n" => ""; count = 1), "quick", "utils2"), ["utils2"])
  @test a === nothing && only(p) == "`utils` is removed from tier `quick`" # a rename
  a, p = TestTiers.manifest_additions(old, replace(old, "\"testutils\", \"tiers\"]" => "\"testutils\", \"tiers\", \"new_helper\"]"), String[])
  @test a === nothing && occursin("outside [tiers]", only(p))
  a, p = TestTiers.manifest_additions(old, replace(old, "environments = [\"cuda\", \"oracles\"]" => "environments = [\"cuda\"]"), String[])
  @test a === nothing && occursin("outside [tiers]", only(p))
  a, p = TestTiers.manifest_additions(old, replace(old, "tiers = [\"tier_selection\"]" => "tiers = [\"tier_selection\", \"gpu\"]"), String[])
  @test a === nothing && only(p) == "something outside [tiers] changed (helpers, environments or run_through)"
  a, p = TestTiers.manifest_additions(old, old * "\n[tiers.extra]\n", String[])
  @test a === nothing
  a, p = TestTiers.manifest_additions(old, "not toml [", String[])
  @test a === nothing && occursin("could not be compared", only(p))
  # the plan: additions give quick and run the new file; otherwise the tier lists need full
  mktempdir() do dir
    path = joinpath(dir, "tiers.toml")
    write(path, add(old, "full", "iterate_new"))
    m = TestTiers.load(path, joinpath(dirname(DIR), ".github", "tier-rules.toml"))
    paths = ["test/iterate_new.jl", "test/tiers.toml"]
    @test TestTiers.pr_tier(m, paths, String[]; manifest_additions = ["iterate_new"])[1:2] == ("quick", ["iterate_new"])
    @test TestTiers.pr_tier(m, paths, String[])[1:2] == ("full", String[])
    @test TestTiers.pr_tier(m, [paths; "src/atom_iterate.jl"], NARROW; manifest_additions = ["iterate_new"])[1:2] ==
          ("quick", sort!([ATOM; "iterate_new"]))
    @test TestTiers.pr_tier(m, [paths; "src/utils.jl"], String[]; manifest_additions = ["iterate_new"])[1] == "thorough"
    # an unlisted test file still stops the plan
    @test_throws ManifestError TestTiers.check(m, [TestTiers.discover(DIR); "unlisted_new"])
  end
end


# The planner's benchmark decision (CIPlan.benchmark_check, [events.*] benchmark and
# [benchmark_check] in .github/tier-rules.toml): a harness path, a label, or the event decides;
# never documentation only. test/ci_plan.jl tests the whole plan.
module CIPlan
include(joinpath(dirname(@__DIR__), "scripts", "ci_plan.jl"))
end

@testset "benchmark check plan" begin
  bc(paths; event = "pull_request", labels = String[], inputs = Dict{String, String}(), tier = nothing) =
    CIPlan.benchmark_check(event, paths, labels, inputs; tier)
  # a pull request: the harness or its environment, or the workflow file that defines the jobs
  for path in ("benchmark/run.jl", "benchmark/cases.jl", "benchmark/check_only.jl", "benchmark/setup.jl",
               "benchmark/check_only_test.jl", "benchmark/Project.toml", ".github/workflows/CI.yml")
    @test bc([path]) && bc(["docs/src/index.md", path])
  end
  # package tests alone do not start them, and documentation never does
  for path in ("src/sparse_iterate.jl", "benchmark/README.md", "benchmark/baselines/iterate-family-0.5.csv", "benchmark/compare.py",
               "benchmark/gpu_solver/analysis.py", "docs/src/index.md", "README_extra.md")
    @test !bc([path])
  end
  @test !bc(String[]) && !bc(nothing)
  # the labels test:benchmark and test:full ask for them; others do not
  @test bc(["src/utils.jl"]; labels = ["test:benchmark"]) && bc(nothing; labels = ["test:benchmark"])
  @test bc(["src/utils.jl"]; labels = ["test:full"]) && !bc(["src/utils.jl"]; labels = ["test:thorough", "bug"])
  # a push to master: the harness paths only; a tag and the nightly run: always
  @test bc(["benchmark/run.jl"]; event = "push") && !bc(["src/utils.jl"]; event = "push") && !bc(nothing; event = "push")
  @test bc(nothing; event = "tag") && bc(["docs/src/index.md"]; event = "tag") && bc(nothing; event = "schedule")
  # a manual run: its input, or the full tier
  @test !bc(nothing; event = "workflow_dispatch", tier = "quick") && bc(nothing; event = "workflow_dispatch", tier = "full")
  @test bc(nothing; event = "workflow_dispatch", tier = "quick", inputs = Dict("benchmark" => "true"))
  @test !bc(nothing; event = "workflow_dispatch", tier = "quick", inputs = Dict("benchmark" => "false"))
  # the rules file the workflow reads has the section, and the tiers agree: these paths need no package tests
  @test first(classify("benchmark/cases.jl")) == "none" && first(classify("benchmark/Project.toml")) == "none"
  @test ".github/workflows/CI.yml" in CIPlan.benchmark_paths()
end

@testset "the grids files" begin
  allgrids = ALL_GRIDS
  @test allunique(allgrids)
  thorough = TestTiers.select(MANIFEST, "thorough")
  full = TestTiers.select(MANIFEST, "full")
  for row in GRIDS
    # in the full tier and in no lower one
    @test MANIFEST.tier[row.grids] == "full"
    @test !(row.grids in thorough)
    # its base file is in the tier base_tier, and its name starts with the base file's name
    @test MANIFEST.tier[row.base] == row.base_tier
    @test row.base in TestTiers.select(MANIFEST, row.base_tier) && startswith(row.grids, row.base)
  end
  # master pushes (thorough) run none of them, the nightly run (full) all
  @test isdisjoint(allgrids, thorough) && issubset(allgrids, full)
  # the listed source paths get their grids files as focus (the shared rule lists other paths too,
  # such as src/interface.jl, with the same six shared grids files), and the tier the path had before
  for (path, (want_tier, names)) in GRIDS_OF
    tier, focus = classify(path)
    @test sort(filter(in(allgrids), focus)) == names
    @test tier == want_tier
  end
  @test pr("src/constrained.jl") == ("standard", ["constrained_basics_grids"])
  # the narrow label on a change to one iterate source still gives quick, with the iterate's own grids file
  @test pr("src/atom_iterate.jl"; labels = NARROW) == ("quick", ATOM)
  @test "atom_iterate_grids" in pr("src/atom_iterate.jl"; labels = NARROW)[2]
  @test "sparse_iterate_basics_grids" in pr("src/sparse_iterate.jl"; labels = NARROW)[2]
  @test pr("src/sparse_iterate.jl"; labels = NARROW)[1] == "quick"
  # the policy, as representative checks: these unrelated source paths get no grids file as focus
  for path in ("src/functions/entropy.jl", "src/deprecated.jl", "src/functions/sets.jl", "src/composition.jl",
               "src/symmetric_parameters.jl", "src/functions/cones.jl", "src/functions/smooth.jl")
    @test isdisjoint(classify(path)[2], allgrids)
  end
  # the grids files of a thorough-tier base file: the rules that list them are thorough-tier rules,
  # the rule for src/calculus.jl and the shared rule, and the standard paths of the other rows get none
  thorough_grids = [row.grids for row in GRIDS if row.base_tier == "thorough"]
  @test length(thorough_grids) == 3
  for rule in MANIFEST.rules, grids in thorough_grids
    grids in rule.focus && @test rule.tier == "thorough"
  end
  for path in ("src/atom_iterate.jl", "src/functions/norms.jl", "src/functions/sparsegroup.jl", "src/functions/nonconvex.jl",
               "src/sparse_iterate.jl", "src/constrained.jl", "src/lmo.jl", "src/lmo_atoms.jl", "src/functions/singular.jl",
               "src/functions/losses.jl")
    @test isdisjoint(classify(path)[2], thorough_grids)
  end
  @test pr("src/functions/norms.jl") == ("standard", ["atom_iterate_grids"])
  # a pull request that changes src/calculus.jl or a shared source runs all three, beside thorough
  for path in ("src/calculus.jl", "src/utils.jl", "src/interface.jl", "src/ProximalBase.jl")
    @test issubset(thorough_grids, pr(path)[2])
  end
  # a changed grids test file selects itself; test:full gives the full tier
  for grids in allgrids
    @test classify("test/$grids.jl") == ("quick", [grids])
    @test pr("test/$grids.jl") == ("quick", [grids])
  end
  @test pr("src/functions/entropy.jl"; labels = ["test:full"]) == ("full", String[])
  # the planner: a master push of package code runs the thorough tier and the nightly run the full tier,
  # both without focus files
  # (the planner reads the manifest with its own copy of test/tiers.jl)
  pm, policy = CIPlan.default_manifest(), CIPlan.load_policy()
  planned(event; kw...) = CIPlan.decide(event; manifest = pm, policy, kw...)[1]
  r = planned("push"; ref_type = "branch", paths = ["src/atom_iterate.jl"])
  @test r.tier == "thorough" && isempty(r.focus) && isdisjoint(allgrids, CIPlan.TestTiers.select(pm, r.tier))
  r = planned("schedule")
  @test r.tier == "full" && issubset(allgrids, CIPlan.TestTiers.select(pm, r.tier))
  r = planned("pull_request"; paths = ["src/atom_iterate.jl"], labels = ["test:full"])
  @test r.tier == "full" && issubset(allgrids, CIPlan.TestTiers.select(pm, r.tier))
  r = planned("pull_request"; paths = ["src/atom_iterate.jl"])
  @test r.tier == "standard" && "atom_iterate_grids" in split(r.focus, ',')
end

# Labels that are not tiers (test:benchmark, test:iterate-narrow) leave the tier of a PR alone.
@testset "labels that are not tiers" begin
  for paths in (["src/utils.jl"], ["docs/src/index.md"], ["test/norms.jl"], ["src/functions/sets.jl"], ["Project.toml"])
    base = pr(paths...)
    @test pr(paths...; labels = ["test:benchmark"]) == base
    @test pr(paths...; labels = ["bug", "test:benchmark", "documentation"]) == base
  end
  @test pr("docs/src/index.md"; labels = ["test:benchmark"]) == ("none", String[])
  @test pr("test/norms.jl"; labels = ["test:benchmark", "test:thorough"])[1] == "thorough"
  _, _, _, notes = TestTiers.pr_tier(MANIFEST, ["src/utils.jl"], ["test:benchmark"])
  @test !any(occursin("test:benchmark", n) for n in notes)
  # a label that names no tier is still an error, and says which labels exist
  @test_throws ArgumentError pr("src/utils.jl"; labels = ["test:fast"])
  e = try pr("src/utils.jl"; labels = ["test:fast"]) catch err; err end
  @test occursin("test:benchmark", e.msg) && occursin("test:iterate-narrow", e.msg)
end

end
