# Tests of the check-only driver: synthetic cases, and the read-only input checks on real cases built
# from the smallest groups that contain them (quick-suite IDs); then cases_test.jl: the sparse
# operands of the `products` cases and their reference checks for repetitions 1-10.
#
#   julia --project=<benchmark environment> benchmark/check_only_test.jl
#
# CI runs it in the benchmark reference-check job before the quick suite.
include("check_only.jl")
using Test

mutable struct Calls
    prepare::Int
    op::Int
    validate::Int
end

# A synthetic case whose callbacks count their calls; `fail_validate_at = k` throws on the
# k-th call of validate.
function counting_case(id; evals, fail_validate_at = 0)
    n = Calls(0, 0, 0)
    validate() = (n.validate += 1; n.validate == fail_validate_at && error("validate failed at call $(n.validate)"); nothing)
    meta = (; id)
    PerfCase(meta, () -> (n.op += 1; nothing), () -> (n.prepare += 1; nothing), validate, evals, false), n
end

# What `check!` runs for one case (see `measure!` in run.jl): validate, three warm-up rounds and
# twelve allocation probes (prepare and op each), then, for an automatic evaluation count,
# CHECK_SAMPLES samples of prepare and CHECK_EVALS ops and a second validate.
const WARM = 3 + 12

@testset "check!" begin
    @testset "a failing reference check throws before the operation runs" begin
        c, n = counting_case("synthetic/validate_throws"; evals = nothing, fail_validate_at = 1)
        @test_throws ErrorException("validate failed at call 1") check!(c)
        @test (n.validate, n.prepare, n.op) == (1, 0, 0)
    end
    @testset "automatic evaluation count: repeated evaluations, then validate again" begin
        c, n = counting_case("synthetic/auto"; evals = nothing)
        check!(c)
        @test n.validate == 2
        @test n.prepare == WARM + CHECK_SAMPLES
        @test n.op == WARM + CHECK_SAMPLES * CHECK_EVALS
        # State corrupted by the repeated evaluations is caught by the second validate.
        c, n = counting_case("synthetic/auto_post"; evals = nothing, fail_validate_at = 2)
        @test_throws ErrorException("validate failed at call 2") check!(c)
        @test n.op == WARM + CHECK_SAMPLES * CHECK_EVALS
    end
    @testset "fixed evaluation count: validated once, as measure! does" begin
        c, n = counting_case("synthetic/fixed"; evals = 1, fail_validate_at = 2)
        check!(c)          # a second validate would throw; a fixed-evals case has none
        @test (n.validate, n.prepare, n.op) == (1, WARM, WARM)
    end
end

@testset "check_cases!" begin
    good, ngood = counting_case("synthetic/good"; evals = nothing)
    bad, _ = counting_case("synthetic/bad"; evals = 1, fail_validate_at = 1)
    late, nlate = counting_case("synthetic/late"; evals = nothing, fail_validate_at = 2)
    io = IOBuffer()
    failed = redirect_stderr(devnull) do
        check_cases!(Any[bad, good, late], io)
    end
    @test failed == ["synthetic/bad", "synthetic/late"]
    @test ngood.validate == 2 && nlate.validate == 2     # the cases after a failure still run
    rows = split.(split(chomp(String(take!(io))), '\n'), '\t')
    @test rows[1] == ["id", "exception", "stack"]
    @test [r[1] for r in rows[2:end]] == failed
    @test all(length(r) == 3 for r in rows)
    @test occursin("validate failed at call 1", rows[2][2]) && occursin("validate failed at call 2", rows[3][2])
    @test all(occursin("check!", r[3]) for r in rows[2:end])   # the full stack trace, not a summary
    # Streaming: each row is flushed when its case fails, before the next case runs.
    path = tempname()
    seen = Ref("")
    bad, _ = counting_case("synthetic/bad"; evals = 1, fail_validate_at = 1)
    spy = PerfCase((; id = "synthetic/spy"), () -> (seen[] = read(path, String); nothing), noop, noop, 1, false)
    redirect_stderr(devnull) do
        open(io -> check_cases!(Any[bad, spy], io), path, "w")
    end
    @test occursin("synthetic/bad", seen[])
end

# ---- read-only inputs ----

# the message of the exception that f throws, or "" when it returns
message(f) = try f(); "" catch err; sprint(showerror, err) end
# a case with another operation; the inputs are forwarded
with_operation(c, op) = PerfCase(c.meta, op, c.prepare, c.validate, c.evals, c.profile, c.inputs)
# the six-argument form, which declares nothing: the negative control
without_inputs(c, op) = PerfCase(c.meta, op, c.prepare, c.validate, c.evals, c.profile)
# the operation of c that runs corrupt!() on its k-th call first
function on_call(c, k, corrupt!)
    n, op = Ref(0), c.operation
    () -> (n[] += 1; n[] == k && corrupt!(); op())
end
during(name, c) = "read-only input `$name` changed during the trial of $(c.meta.id)"
after_build(name) = "read-only input `$name` changed after the case was built"

@testset "read-only inputs: synthetic cases" begin
    @test_throws UndefKeywordError addcase!(PerfCase[], "synthetic", "no_inputs", noop, noop)
    # a fixed evaluation count: check! runs the 15 calls of the warm-up and the probes
    a = zeros(3); n = Ref(0)
    c = PerfCase((; id = "synthetic/fixed_input"), () -> (n[] += 1; n[] == 10 && (a[2] = 1); nothing), noop, noop, 1, false, (; a))
    @test occursin(during("a", c), message(() -> check!(c)))
    # the input check runs before the post-trial validate, which here restores the input
    a = zeros(3); n = Ref(0)
    c = PerfCase((; id = "synthetic/auto_input"), () -> (n[] += 1; n[] == 20 && (a[2] = 1); nothing), noop,
                 () -> (fill!(a, 0); nothing), nothing, false, (; a))
    @test occursin(during("a", c), message(() -> check!(c)))
    # the digests are taken after the pre-trial validate, which may change shared state
    a = zeros(3)
    c = PerfCase((; id = "synthetic/validate_changes"), noop, noop, () -> (a[1] += 1; nothing), 1, false, (; a))
    @test check!(c) === nothing
end

@testset "logical digest" begin
    v = zeros(10^6); h = digest(v)
    v[500_001] = -0.0; @test digest(v) != h          # the sign of a zero, in the middle of 10⁶ entries
    v[500_001] = 1e-300; @test digest(v) != h
    v[500_001] = 0.0; @test digest(v) == h
    @test digest([NaN]) == digest([reinterpret(Float64, 0x7ff8000000000001)])   # isequal: NaNs are equal
    s1 = SparseIterate(Float64, 100); s1[3] = 1.0; s1[7] = 2.0
    s2 = SparseIterate(Float64, 100); s2[7] = 2.0; s2[3] = 1.0
    @test digest(s1) == digest(s2)                   # the slot order is not part of the state
    activate!(s1, 50)                                # a stored zero: the dense values are unchanged
    @test Array(s1) == Array(s2) && digest(s1) != digest(s2)
    x = AtomIterate(Float64, 12, [1:3, 4:6, 7:9, 10:12]); x[1] = 1.0; h = digest(x)
    activate!(x, 3)                                  # an active group of zeros
    @test Vector(x)[7:9] == zeros(3) && digest(x) != h
    M = randn(3, 4)
    @test digest(transpose(M)) == digest(transpose(copy(M))) && digest(transpose(M)) != digest(M)
    # digestable agrees with what digest supports: element types without a digest word are rejected
    # when the case is built, and every accepted input can be digested
    for a in (Float16[1, 2], big.([1.0, 2.0]), [Float16(1)], (zeros(2), Float16[1]))
        @test !digestable(a)
        @test_throws MethodError digest(a)
        @test occursin("has no logical digest", message(() -> addcase!(PerfCase[], "synthetic", "f16", noop, noop; inputs = (; a))))
    end
    for a in (zeros(Float32, 3), [1, 2], [true, false], 1:4, [1:2, 3:4], [[1, 2], [3]], Symmetric(randn(3, 3)),
              Diagonal([1.0, 2.0]), sparse([1.0 0; 0 2]), view(randn(6), 1:2:6), nothing, 2.5, (zeros(2), (ones(3),)))
        @test digestable(a) && digest(a) isa UInt64
    end
end

# the real cases: quick-suite IDs, from the smallest builders that contain them
Random.seed!(1)
BLAS.set_num_threads(1)
const CS = PerfCase[]
separable_cases!(CS, 1; quick = true)
smooth_cases!(CS, 1; quick = true)
matrix_cases!(CS, 1; quick = true)
product_cases!(CS, 1; quick = true)
owner_cases!(CS, 1; quick = true)
sparse_vector_cases!(CS, 1, Float64, 10^4, 10^2, Dict{Any,Any}())
let T = Float64, n = 1000, rng = scenario_rng(1, "blocks/flat/$T/$n")      # as block_cases! builds blocks/flat
    addblockprox!(CS, "flat", SeparableSum(NormL1(T(0.1)), SqrNormL2()), (randn(rng, T, n), randn(rng, T, n)))
end
const GROUPS8 = atom_groups(8 * 10^4, 8, false)
atom_scenario_cases!(CS, 1, Float64, 8 * 10^4, 8, false, 0.1, "10", GROUPS8, (Dict{Any,Any}(), Dict{Any,Any}(), Dict{Any,Any}()))
atom_layout_cases!(CS, 8 * 10^4, 8, false, GROUPS8)
byid(id) = only(filter(c -> c.meta.id == id, CS))

@testset "sabotage and recovery: x changed after construction" begin
    for id in ("separable/NormL1/prox!/Float64/256/dense/warm",
               "separable/NormL1/value_and_prox!/Float64/256/dense/warm",
               "smooth/least_squares/gradient!/Float64/32/dense/warm",
               "least_squares/tall/Float64/32/dense/construct_and_first_prox")
        c = byid(id)
        x = c.inputs.x; x0 = copy(x)
        x[1] += 1
        @test occursin(after_build("x"), message(c.validate))
        copyto!(x, x0)
        @test message(c.validate) == ""
    end
    c = byid("blocks/flat/prox!/Float64/2x1000/dense/warm")
    b = first(c.inputs.x); b0 = copy(b)
    b .+= 1
    @test occursin(after_build("x"), message(c.validate))
    copyto!(b, b0)
    @test message(c.validate) == ""
end

@testset "the frozen reference: data held by a function object changed after construction" begin
    # The center of Translate is created inside the function object, so it is not a declared input;
    # the reference computed when the case was built catches the change, which a reference computed at
    # check time from the live object would follow.
    c = byid("compositions/Translate/value_and_gradient!/Float64/32/dense/warm")
    f = c.operation.f
    @test !any(v -> v === f.b, values(c.inputs))
    @test message(c.validate) == ""
    f.b .+= 1
    @test occursin("value/gradient objective mismatch", message(c.validate))
    f.b .-= 1
    @test message(c.validate) == ""
end

@testset "repeated evaluations change a declared input" begin
    # automatic evaluation count: the post-trial validate restores the shared iterate, so only the input
    # check sees the change (call 20 is in a stand-in sample)
    c = byid("sparse/read_x[i]/Float64/n10000_s100/active/warm")
    x = c.inputs.x; i = first(activecoordinates(x))
    corrupt!() = (x[i] += 1)
    @test check!(without_inputs(c, on_call(c, 20, corrupt!))) === nothing
    @test occursin(during("x", c), message(() -> check!(with_operation(c, on_call(c, 20, corrupt!)))))
    c.validate()                                     # restores the shared iterate
    # fixed evaluation count: no post-trial validate at all (call 10 is among the probes)
    c = byid("products/strided_vector/mul!/Float64/512x256/sparse_0.01/warm")
    x = c.inputs.x; i = first(activecoordinates(x)); x0 = x[i]
    corrupt2!() = (x[i] += 1)
    @test check!(without_inputs(c, on_call(c, 10, corrupt2!))) === nothing
    x[i] = x0
    @test occursin(during("x", c), message(() -> check!(with_operation(c, on_call(c, 10, corrupt2!)))))
    x[i] = x0
end

@testset "aliases and stateful cases pass" begin
    for id in ("separable/NormL1/prox!_alias/Float64/256/dense/warm",
               "separable/NormL1/value_and_prox!/Float64/256/dense/warm",
               "partial/nuclear/prox!/Float64/128x64/dense/warm",
               "sparse/first_activation/Float64/n10000_s100/absent_distinct_batch/first_activation",
               "owner/write_X[i,j]_active/Float64/p500_s100/active/warm",
               # the second sweeps the iterate that the first leaves changed
               "owner/sweep_prox_coordinate!/Float64/p500_s100/NormL1/warm",
               "owner/sweep_activecoordinates/Float64/p500_s100/trivial_update/warm")
        @test message(() -> check!(byid(id))) == ""
    end
end

@testset "representation changes that the output comparison does not see" begin
    # a stored zero in the SparseIterate input of a proximal map
    c = byid("sparse_penalties/NormL1/prox!/Float64/256/sparse_1pct/warm")
    x = c.inputs.x; i = findfirst(k -> !isactive(x, k), 1:length(x)); d0 = Array(x)
    @test occursin(during("x", c), message(() -> check!(with_operation(c, on_call(c, 20, () -> activate!(x, i))))))
    @test Array(x) == d0
    @test occursin(after_build("x"), message(c.validate))
    dropzeros!(x)
    @test message(c.validate) == ""
    # an active group of zeros in the shared AtomIterate; the validate restores the iterate first
    c = byid("atom/dot_vx/Float64/p80000_g8_contiguous_act10/z0/warm")
    x = c.inputs.x; k = findfirst(k -> !isactive(x, k), 1:ncoordinates(x))
    @test check!(without_inputs(c, on_call(c, 20, () -> activate!(x, k)))) === nothing
    @test occursin(during("x", c), message(() -> check!(with_operation(c, on_call(c, 20, () -> activate!(x, k))))))
    c.validate()
    # a stored zero added to the owner X50 during a fixed-evaluation trial: no validate follows it (the
    # fixture check of the validate before the trial would catch one added earlier, so the negative control
    # cannot run after it); this owner is not used afterwards
    c = byid("owner/mul!/Float64/p500_s100/owner_z50/warm")
    X50 = c.inputs.X50; k = findfirst(k -> !isactive(X50, k), 1:ncoordinates(X50))
    @test occursin(during("X50", c), message(() -> check!(with_operation(c, on_call(c, 10, () -> activate!(X50, k))))))
    # a changed group array: the shared call is a pointer comparison, and the validate accepts it
    c = byid("atom/samegroups/Float64/p80000_g8_contiguous/shared/warm")
    if haskey(c.inputs, :f_groups)
        G = c.inputs.f_groups
        swap!() = begin G[3], G[4] = G[4], G[3]; nothing end
        swap!(); @test message(c.validate) == ""; swap!()
        @test occursin(during("x_shared", c), message(() -> check!(with_operation(c, on_call(c, 20, swap!)))))
        swap!()
        @test message(() -> check!(c)) == ""
    end
end

include("cases_test.jl")
include("case_selection_test.jl")
