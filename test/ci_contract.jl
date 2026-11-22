module CIContractTest
using Test, TOML, LinearAlgebra
const root=dirname(@__DIR__)
const TT=Main.TestTiers
# A Windows checkout can convert line endings to CRLF; the job-block patterns below expect LF.
normalize_newlines(text)=replace(text,"\r\n"=>"\n")
workflow=normalize_newlines(read(joinpath(root,".github","workflows","CI.yml"),String))

# The text of one top-level job, up to the next job or comment at the same indent.
function job_block(name)
    m=match(Regex("^  "*name*":\\n(.*?)(?=^  \\S|\\z)","ms"),workflow)
    m === nothing ? "" : m.captures[1]
end

pos(needle,text)=(r=findfirst(needle,text); r===nothing ? nothing : first(r))

parts(names,w,count)=[TT.shard_files(names,i,count;weights=w) for i in 1:count]
function check_partition(names,w,count)
    p=parts(names,w,count)
    combined=reduce(vcat,p)
    @test all(!isempty,p)
    @test sort(combined)==sort(names)
    @test length(unique(combined))==length(names)
    @test p==parts(names,w,count)
    @test all(s -> s==filter(in(s),names),p)
end

@testset "minimal required CI contract" begin
    @test normalize_newlines(replace(workflow,"\n"=>"\r\n"))==workflow
    @test occursin("name: PR required jobs",workflow)
    @test occursin("name: PR merge gate",workflow)
    @test occursin("version: '1.13'",workflow)
    @test !occursin("version: '1'",workflow)
    @test occursin("--check-bounds=yes",workflow)
    @test occursin("shard: [1, 2, 3]",workflow)
    @test occursin("needs: [changes, test, benchmark, docs]",workflow)
    @test occursin("BENCHMARK_RESULT:",workflow)
    @test occursin("DOCS_RESULT:",workflow)
    @test occursin("PB_CHECK_ONLY: '1'",workflow)
    @test occursin("docs/make.jl",workflow)
    @test !occursin("continue-on-error",workflow)
    @test !occursin("pull_request_target:",workflow)
    @test !occursin("paths-ignore",workflow)
    @test occursin("cancel-in-progress: \${{ github.event_name == 'pull_request' }}",workflow)
    @test occursin("group: minimal-ci-\${{ github.workflow }}-\${{ github.event_name == 'pull_request' && github.ref || github.run_id }}",workflow)
    rules=TOML.parsefile(joinpath(root,".github","tier-rules.toml"))
    @test Set(keys(rules["platforms"]))==Set(["linux"])
    @test rules["platforms"]["linux"]["julia"]=="1.13"
end
@testset "documentation-only classification fails closed" begin
    @test occursin("name: Change classification / Linux",workflow)
    for job in ("changes","test","benchmark","docs","required","gate","deploy-docs")
        @test !isempty(job_block(job))
    end
    required=job_block("required")
    @test occursin("name: PR required jobs",required)
    @test occursin("if: always()",required)
    @test occursin("needs: [changes, test, benchmark, docs]",required)
    @test occursin("CHANGES_RESULT: \${{ needs.changes.result }}",required)
    @test occursin("DOCS_ONLY: \${{ needs.changes.outputs.docs_only }}",required)
    @test occursin("TEST_RESULT: \${{ needs.test.result }}",required)
    @test occursin("BENCHMARK_RESULT: \${{ needs.benchmark.result }}",required)
    @test occursin("DOCS_RESULT: \${{ needs.docs.result }}",required)
    @test occursin("test \"\$CHANGES_RESULT\" = success",required)
    @test occursin("test \"\$DOCS_RESULT\" = success",required)
    @test occursin("true)  test \"\$TEST_RESULT\" = skipped; test \"\$BENCHMARK_RESULT\" = skipped ;;",required)
    @test occursin("false) test \"\$TEST_RESULT\" = success; test \"\$BENCHMARK_RESULT\" = success ;;",required)
    @test occursin("*)     echo \"unclassified change: '\$DOCS_ONLY'\"; exit 1 ;;",required)
    gate=job_block("gate")
    @test occursin("name: PR merge gate",gate)
    @test occursin("if: always()",gate)
    @test occursin("needs: [required]",gate)
    @test occursin("REQUIRED_RESULT: \${{ needs.required.result }}",gate)
    @test occursin("test \"\$REQUIRED_RESULT\" = success",gate)
    # The test and benchmark jobs each depend on `changes` and run when the classification is 'false'.
    for job in ("test","benchmark")
        section=job_block(job)
        @test occursin("needs: [changes]",section)
        @test occursin("if: needs.changes.outputs.docs_only == 'false'",section)
    end
    changes=job_block("changes")
    @test occursin("fetch-depth: 2",changes)
    @test occursin("docs_only: \${{ steps.classify.outputs.docs_only }}",changes)
    @test occursin("BASE: \${{ steps.classify.outputs.base }}",changes)
    @test occursin("check_repo_hygiene.py --base \"\${BASE:-HEAD^1}\"",changes)
    @test occursin("test_repo_hygiene.py",changes)
    @test !occursin("hygiene",job_block("benchmark"))
    @test !isnothing(pos("id: classify",changes)) && !isnothing(pos("check_repo_hygiene.py",changes)) &&
          pos("id: classify",changes) < pos("check_repo_hygiene.py",changes)
    deploy=job_block("deploy-docs")
    @test occursin("needs: [gate, docs]",deploy)
    @test occursin("id: tip",deploy)
    @test occursin("git fetch --no-tags origin refs/heads/master",deploy)
    @test occursin("tip=\"\$(git rev-parse FETCH_HEAD)\"",deploy)
    @test occursin("\"\$tip\" == \"\$GITHUB_SHA\"",deploy)
    # Every step after the freshness check, including the publication, is conditional on it.
    after=deploy[something(pos("id: tip",deploy),1):end]
    @test count("if: steps.tip.outputs.current == 'true'",after)==5
    @test occursin("docs/deploy.jl",after)
    publish=filter(step -> occursin("docs/deploy.jl",step),split(deploy,r"^      - "m))
    @test length(publish)==1
    @test all(step -> occursin("if: steps.tip.outputs.current == 'true'",step),publish)
    @test !isnothing(pos("id: tip",deploy)) && !isnothing(pos("docs/deploy.jl",deploy)) &&
          pos("id: tip",deploy) < pos("docs/deploy.jl",deploy)
    @test count("docs_only == 'false'",workflow)==2
    @test !occursin("docs_only == 'true'",workflow)
end
@testset "shards cover the complete supported inventory exactly once" begin
    inv=TT.inventory()
    selected=TT.select(inv,"full")
    @test !isempty(selected)
    @test Set(keys(inv.weights))==Set(selected)
    @test all(w -> w isa Real && !(w isa Bool) && isfinite(w) && w > 0, values(inv.weights))
    check_partition(selected,inv.weights,3)
    check_partition(selected,inv.weights,2)
    check_partition(selected[1:2:end],inv.weights,3)
    @test parts(selected,inv.weights,1)==[selected]
    @test_throws "shard index must lie in 1:3" TT.shard_files(selected,0,3;weights=inv.weights)
    @test_throws "shard index must lie in 1:3" TT.shard_files(selected,4,3;weights=inv.weights)
    @test_throws "shard count must be positive" TT.shard_files(selected,1,0;weights=inv.weights)
    @test_throws "more shards than selected test files" TT.shard_files(selected,1,length(selected)+1;weights=inv.weights)
end
@testset "scheduler places the heaviest files on the lightest shard" begin
    # Equal weights are placed in lexical order. The final tie goes to shard 1.
    @test parts(["c","a","b"],Dict("a"=>1,"b"=>1,"c"=>1),2)==[["c","a"],["b"]]
    # Shard 1 holds h alone (10). Shard 2 holds m+n+o (9) in input order.
    @test parts(["o","h","n","m"],Dict("h"=>10,"m"=>3.0,"n"=>3,"o"=>3),2)==[["h"],["o","n","m"]]
    # Ties between shards go to the lowest index: a,b,c fill 1,2,3, then d -> 1 and e -> 2.
    @test parts(["a","b","c","d","e"],Dict("a"=>2,"b"=>2,"c"=>2,"d"=>1,"e"=>1),3)==[["a","d"],["b","e"],["c"]]
    # Loads are exact: Float64 rounds 2^60 + 1 back to 2^60 and sees a tie; exactly, d goes to the lighter shard 2.
    @test parts(["a","b","c","d"],Dict("a"=>2.0^60,"b"=>2.0^60,"c"=>1.0,"d"=>1.0),2)==[["a","c"],["b","d"]]
    # Float64 loads would overflow to Inf and tie; exact loads put f on shard 2 (2fm against 2.5fm).
    fm=floatmax()
    big=Dict("a"=>fm,"b"=>fm,"c"=>fm,"d"=>fm,"e"=>fm/2,"f"=>fm/2)
    @test parts(["a","b","c","d","e","f"],big,2)==[["a","c","e"],["b","d","f"]]
end
@testset "tier selection is cumulative and shards use only selected weights" begin
    mktempdir() do dir
        for f in ("q","s","t","f","helper")
            write(joinpath(dir,f*".jl"),"nothing\n")
        end
        write(joinpath(dir,"tiers.toml"),"""
        [tiers]
        quick = ["q"]
        standard = ["s"]
        thorough = ["t"]
        full = ["f"]
        [not_tests]
        files = ["helper"]
        [weights]
        q = 1
        s = 5
        t = 1
        f = 1
        """)
        inv=TT.inventory(dir)
        @test TT.select(inv,"quick")==["q"]
        @test TT.select(inv,"standard")==["q","s"]
        @test TT.select(inv,"thorough")==["q","s","t"]
        @test TT.select(inv,"full")==["f","q","s","t"]
        @test_throws "unknown test tier" TT.select(inv,"nightly")
        standard=TT.select(inv,"standard")
        @test parts(standard,inv.weights,2)==[["s"],["q"]]
    end
end
@testset "runner filters by prefix before sharding" begin
    function list_output(args)
        # Load the list-only runner in its own module with its own arguments.
        runner = Module(gensym(:ListRunner))
        Core.eval(runner, :(const ARGS = $args))
        Core.eval(runner, :(include(path) = Base.include($runner, path)))
        mktemp() do _, output
            redirect_stdout(output) do
                threads = BLAS.get_num_threads()
                try
                    Base.include(runner, joinpath(@__DIR__, "runtests.jl"))
                finally
                    BLAS.set_num_threads(threads)
                end
            end
            seekstart(output)
            read(output, String)
        end
    end
    inv = TT.inventory()
    selected = filter(n -> startswith(n, "indball") || startswith(n, "kkt"),
                      TT.select(inv, "quick"))
    cases = ["--list", "--tier=quick", "indball", "indballl", "kkt"]
    threads = BLAS.get_num_threads()
    for i in 1:2
        output = list_output([cases; "--shard=$i/2"])
        @test !occursin("PB_TEST_", output)
        @test split(last(split(chomp(output), '\n'))) == TT.shard_files(selected, i, 2; weights=inv.weights)
    end
    @test BLAS.get_num_threads() == threads
    @test_throws "no test file matches no_matching_prefix" list_output([cases; "no_matching_prefix"])
    @test BLAS.get_num_threads() == threads
end
@testset "manifest rejects unclassified tests" begin
    empty_catalog="[tiers]\nquick=[]\nstandard=[]\nthorough=[]\nfull=[]\n[not_tests]\nfiles=[]\n"
    mktempdir() do dir
        write(joinpath(dir,"tiers.toml"),empty_catalog)
        @test_throws "[weights] table" TT.inventory(dir)
        write(joinpath(dir,"tiers.toml"),empty_catalog*"[weights]\n")
        inv=TT.inventory(dir)
        @test isempty(inv.weights)
        @test isempty(TT.select(inv,"full"))
        write(joinpath(dir,"unexpected.jl"),"nothing\n")
        @test_throws "inventory mismatch" TT.inventory(dir)
        write(joinpath(dir,"tiers.toml"),empty_catalog)
        @test_throws "inventory mismatch" TT.inventory(dir)
    end
    mktempdir() do dir
        write(joinpath(dir,"a.jl"),"nothing\n")
        single="[tiers]\nquick=[\"a\"]\nstandard=[]\nthorough=[]\nfull=[]\n[not_tests]\nfiles=[]\n"
        write(joinpath(dir,"tiers.toml"),single)
        @test_throws "[weights] table" TT.inventory(dir)
        write(joinpath(dir,"tiers.toml"),single*"[weights]\na=2.5\n")
        inv=TT.inventory(dir)
        @test inv.weights==Dict("a"=>2.5)
        @test parts(TT.select(inv,"full"),inv.weights,1)==[["a"]]
        overlap=replace(single,"files=[]"=>"files=[\"a\"]")
        write(joinpath(dir,"tiers.toml"),overlap*"[weights]\na=2.5\n")
        @test_throws "declared as both tests and helpers" TT.inventory(dir)
    end
end
@testset "manifest weights must match the declared tests and be finite positive" begin
    catalog="""
    [tiers]
    quick = ["a"]
    standard = ["b"]
    thorough = []
    full = []
    [not_tests]
    files = ["helper"]
    """
    mktempdir() do dir
        for f in ("a","b","helper")
            write(joinpath(dir,f*".jl"),"nothing\n")
        end
        load(text)=(write(joinpath(dir,"tiers.toml"),text); TT.inventory(dir))
        inv=load(catalog*"[weights]\na = 1.5\nb = 2\n")
        @test inv.weights==Dict("a"=>1.5,"b"=>2)
        @test parts(TT.select(inv,"standard"),inv.weights,2)==[["b"],["a"]]
        @test_throws "[weights] table" load(catalog)
        @test_throws "[weights] table" load("weights = 3\n"*catalog)
        @test_throws "[weights] table" load("weights = []\n"*catalog)
        @test_throws "[weights] table" load(catalog*"[[weights]]\na = 1\nb = 1\n")
        for (label, entries) in (
            ("missing key", "a = 1\n"),
            ("extra key", "a = 1\nb = 1\nc = 1\n"),
            ("stale key", "a = 1\nold = 1\n"),
            ("helper key", "a = 1\nb = 1\nhelper = 1\n"),
        )
            @testset "$label" begin
                @test_throws "weight mismatch" load(catalog*"[weights]\n"*entries)
            end
        end
        for value in ("true", "\"1\"", "[1]", "{ x = 1 }", "0", "0.0", "-1", "nan", "inf", "-inf")
            @testset "invalid weight $value" begin
                @test_throws "finite positive" load(catalog*"[weights]\na = $value\nb = 1\n")
            end
        end
        @testset "duplicate key" begin
            @test_throws TOML.ParserError load(catalog*"[weights]\na = 1\na = 2\nb = 1\n")
        end
    end
end
end
