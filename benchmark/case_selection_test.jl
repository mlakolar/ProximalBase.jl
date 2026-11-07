# Tests of the selection of benchmark cases before their fixtures are built (`PB_CASES`, `build_cases(...; selector)`,
# `selected_cases`): the ID spelling and the selection rule, a test-only walker over the captured state of the
# cases, and a differential oracle that builds a selection and compares it, case by case, with the same cases of
# the unfiltered build (IDs and order, metadata, declared inputs, captured state, validation, and the state after
# running the case). The unfiltered catalog is pinned to the digests it had before the selection existed.
#
# This file assumes that check_only.jl (hence run.jl and cases.jl) and Test are loaded, as in check_only_test.jl,
# which includes it. The extended suite is never built unfiltered here: its branches (Float32, 256 and 1024 rows,
# `full`, the `x$nbatch` labels, the z50 variants, the Float64 conversions) are tested at the level of one
# scenario.

# ---- the pinned unfiltered catalog ---------------------------------------------------------------------------
#
# The quick `atom` group, unfiltered, at the revision before selection existed (118 cases; the same for
# repetitions 1, 2 and 10). `catalog` is the SHA-1 of the sorted IDs joined by newlines; `order` the SHA-1 of the
# rows in build order: the metadata keys and values, the evaluation count and the profile flag of every case.
# A change that alters this catalog on purpose updates these values and says why.
const ATOM_QUICK_CASES = 118
const ATOM_QUICK_CATALOG_SHA1 = "3d34cb6c7097496992359bfdda543fb2c77c9cf8"
const ATOM_QUICK_ORDER_SHA1 = "76f5ab38e0617715ca06dd85ea621a8ac8524820"

sel_catalog_sha1(cases) = bytes2hex(sha1(join(sort!([c.meta.id for c in cases]), '\n')))
function sel_order_sha1(cases)
    io = IOBuffer()
    for c in cases
        join(io, (keys(c.meta)...,), ','); write(io, '\t')
        join(io, (values(c.meta)...,), '\t')
        write(io, '\t', c.evals === nothing ? "auto" : string(c.evals), '\t', string(c.profile), '\n')
    end
    bytes2hex(sha1(take!(io)))
end

# ---- the walker ----------------------------------------------------------------------------------------------
#
# `case_fingerprint(cases...)` hashes the logical state that the `operation`, `prepare` and `validate` of each
# case reach: the captured fields of closures and callable structs (their names and, for closures, their field
# types, so that a `Core.Box` or a `Union{Nothing,...}` field changes the result), nested closures, `Core.Box`
# and `Ref` contents, numbers, symbols, strings, `nothing`, arrays by content (a dense array by its elements, a
# sparse array through its storage arrays), iterates through the harness digest, and penalty and layout objects
# field by field. Every mutable object is numbered at its first visit, with the numbering shared by the cases of
# one call, so that aliasing between cases is part of the result; a cycle ends at the number. Raw pointers,
# addresses and the names of anonymous closure types never enter it; named types do. Only the layout caches and
# the scratch workspaces of the penalties are skipped (`SEL_SKIPPED_FIELDS`): the calls that the cases measure
# fill them, and a workspace starts as uninitialized memory, so neither is part of what a case holds. The walk
# stops with an error beyond a budget of nodes.
const SEL_SKIPPED_FIELDS = (:layout_cache, :scratch)
const SEL_NODE_BUDGET = Ref(5_000_000)

mutable struct SelWalker
    h::UInt64
    memo::IdDict{Any,Int}
    nodes::Int
    trace::Union{Nothing,Vector{String}}
end
SelWalker(; trace=nothing) = SelWalker(0x0, IdDict{Any,Int}(), 0, trace)

# a type, without the names of anonymous closure types
sel_type_token(T) = replace(string(T), r"#[#A-Za-z0-9_!]*" => "#")
sel_anonymous(T::DataType) = startswith(string(nameof(T)), "#")
sel_anonymous(T) = false
sel_flat(T) = isbitstype(T) || (T <: Array && isbitstype(eltype(T)))

function sel_mix!(w::SelWalker, path, x)
    w.h = hash(x, w.h)
    w.trace === nothing || push!(w.trace, string(path, " = ", x))
    nothing
end

function sel_walk!(w::SelWalker, x, path::String)
    (w.nodes += 1) > SEL_NODE_BUDGET[] && error("the case fingerprint exceeded its budget of $(SEL_NODE_BUDGET[]) nodes at $path")
    T = typeof(x)
    if x isa Union{Nothing,Missing}
        return sel_mix!(w, path, :nothing)
    elseif x isa Union{Number,Symbol,AbstractString,AbstractChar}
        sel_mix!(w, path * "::type", sel_type_token(T))
        return sel_mix!(w, path, x)
    elseif x isa Union{Type,Module,Ptr}
        return sel_mix!(w, path, x isa Ptr ? "pointer" : string(x))
    end
    if ismutable(x)
        n = get(w.memo, x, 0)
        if n != 0
            return sel_mix!(w, path, "same object as #$n")
        end
        w.memo[x] = n = length(w.memo) + 1
        sel_mix!(w, path, "object #$n")
    end
    if x isa Union{AtomIterate,SparseIterate,SymmetricSparseIterate} && digestable(x)
        sel_mix!(w, path * "::type", sel_type_token(T))
        return sel_mix!(w, path * " digest", digest(x))
    elseif x isa Core.Box
        return isdefined(x, :contents) ? sel_walk!(w, x.contents, path * ".contents") : sel_mix!(w, path, :undef)
    elseif x isa Base.RefValue
        return sel_walk!(w, x[], path * "[]")
    elseif x isa Array
        sel_mix!(w, path * "::type", sel_type_token(T))
        sel_mix!(w, path * " size", size(x))
        if sel_flat(eltype(x))
            if digestable(x)
                return sel_mix!(w, path * " digest", digest(x))
            end
            w.nodes += length(x)
            return sel_mix!(w, path * " elements", [hash(v) for v in x])
        end
        for (i, v) in enumerate(x)
            sel_walk!(w, v, string(path, "[", i, "]"))
        end
        return nothing
    elseif x isa AbstractDict
        for (k, v) in sort!(collect(x); by=pair -> hash(pair.first))
            sel_walk!(w, k, path * ".key"); sel_walk!(w, v, path * "[key]")
        end
        return nothing
    elseif x isa AbstractSet
        for k in sort!(collect(x); by=hash)
            sel_walk!(w, k, path * ".element")
        end
        return nothing
    end
    closure = x isa Function || sel_anonymous(T)
    sel_mix!(w, path * "::type", sel_anonymous(T) ? "closure" : sel_type_token(T))
    for i in 1:fieldcount(T)
        name = T <: Tuple ? i : fieldname(T, i)
        name in SEL_SKIPPED_FIELDS && continue
        fpath = string(path, ".", name)
        closure && sel_mix!(w, fpath * "::fieldtype", sel_type_token(fieldtype(T, i)))
        isdefined(x, i) ? sel_walk!(w, getfield(x, i), fpath) : sel_mix!(w, fpath, :undef)
    end
    nothing
end

# one hash per case; `trace` (a vector) receives the visited items, for locating a difference
function case_fingerprint(cases::PerfCase...; trace=nothing)
    w = SelWalker(; trace)
    map(cases) do c
        w.h = 0x0
        for (name, f) in ((:operation, c.operation), (:prepare, c.prepare), (:validate, c.validate))
            sel_mix!(w, string(c.meta.id, "/", name), name)
            sel_walk!(w, f, string(name))
        end
        w.h
    end
end
sel_fingerprint(x; trace=nothing) = (w = SelWalker(; trace); sel_walk!(w, x, "x"); w.h)

# ---- the differential oracle ---------------------------------------------------------------------------------
# the captured variables that the compiler boxed (a `Core.Box` field costs every timed call a dynamic dispatch)
function sel_boxed(cases)
    trace = String[]
    case_fingerprint(cases...; trace)
    filter(t -> occursin("::fieldtype = Core.Box", t), trace)
end
sel_error(f) = try f(); nothing catch err; err end
sel_message(f) = (err = sel_error(f); err === nothing ? "" : sprint(showerror, err))

# `unfiltered`: a function that builds the unfiltered catalog (run cases leave their destinations changed, so the
# oracle starts from a fresh build), or, with `run=false`, an unfiltered build that was never run; `build_selected(selector)`
# builds with the selector. The cases of the selection must agree item by item, and so must the state after running
# them (`run`).
function sel_compare(unfiltered, build_selected, selection::AbstractString; targeted=true, run=true)
    run && !(unfiltered isa Function) && error("an unfiltered build that is reused must not be run")
    all_cases = unfiltered isa Function ? unfiltered() : unfiltered
    selector = Regex(selection)
    expected = filter(c -> occursin(selector, c.meta.id), all_cases)
    built = build_selected(selector)
    actual = filter(c -> occursin(selector, c.meta.id), built)
    @test !isempty(expected)
    targeted && @test length(built) < length(all_cases)
    @test [c.meta.id for c in actual] == [c.meta.id for c in expected]
    length(actual) == length(expected) || return nothing
    differences = Dict{String,Vector{String}}()
    note(what, c) = push!(get!(differences, what, String[]), c.meta.id)
    fe, fa = case_fingerprint(expected...), case_fingerprint(actual...)
    for (i, (e, a)) in enumerate(zip(expected, actual))
        e.meta == a.meta || note("meta", e)
        e.evals === a.evals && e.profile == a.profile || note("evals or profile", e)
        input_digests(e.inputs) == input_digests(a.inputs) || note("input digests", e)
        fe[i] == fa[i] || note("state before", e)
        run || continue
        sel_message(e.validate) == sel_message(a.validate) || note("validate", e)
        for c in (e, a)
            c.prepare(); c.operation()
        end
        only(case_fingerprint(e)) == only(case_fingerprint(a)) || note("state after", e)
        sel_message(e.validate) == sel_message(a.validate) || note("validate after", e)
    end
    @test isempty(differences)
    isempty(differences) || println(stderr, "selection ", repr(selection), " differs: ", differences)
    nothing
end

@testset "case IDs and the selection rule" begin
    @test case_id("atom", "isactive", Float64, "p8_g2", "half_active", "warm") == "atom/isactive/Float64/p8_g2/half_active/warm"
    @test case_id("a", "b", Float32, "", "dense", "construct") == "a/b/Float32//dense/construct"
    c = addcase!(PerfCase[], "f", "n", noop, noop; inputs=(;), T=Float32, shape="s", layout="l", phase="p")
    @test c[1].meta.id == case_id("f", "n", Float32, "s", "l", "p")
    @test wants(nothing, "anything") && wants(r"^any", "anything") && !wants(r"^thing", "anything")
    @test wants(r"thing$", "anything") && !wants(r"x", "anything")
    # the spelling of an enumerated ID is the registered one, for every kind of enumerator
    @test only(atom_construct_ids(Float64, 80, 8, true)) ==
          ("construct", "atom/construct/Float64/p80_g8_strided/empty/construct")
    @test [id for (_, id) in atom_layout_ids(80, 8, false)] ==
          ["atom/samegroups/Float64/p80_g8_contiguous/$l/warm" for l in ("shared", "distinct_first_seen", "alternating")]
    @test [id for (_, id) in atom_symmetric_ids(Float32, 64)] ==
          ["atom/mul!/Float32/p1024_g64_contiguous_act10/symmetric_$z/warm" for z in ("z0", "z50")]
end

mutable struct SelNode
    next::Any
end

@testset "the case walker" begin
    a = [1.0, 2.0, 3.0]
    @test sel_fingerprint(() -> a) == sel_fingerprint(() -> a)
    # a changed array element, scalar and box content
    f0 = sel_fingerprint(() -> a)
    a[2] = 5.0
    @test sel_fingerprint(() -> a) != f0
    a[2] = 2.0
    @test sel_fingerprint(() -> a) == f0
    mk(s) = () -> s
    @test sel_fingerprint(mk(1)) != sel_fingerprint(mk(2))
    @test sel_fingerprint(mk(1.0)) != sel_fingerprint(mk(1))                  # the type of a number
    @test sel_fingerprint(mk(:a)) != sel_fingerprint(mk(:b)) && sel_fingerprint(mk("a")) != sel_fingerprint(mk("b"))
    @test sel_fingerprint(mk(nothing)) != sel_fingerprint(mk(0))
    b1, b2 = Core.Box(1), Core.Box(1)
    @test sel_fingerprint((b1,)) == sel_fingerprint((b2,))
    b2.contents = 2
    @test sel_fingerprint((b1,)) != sel_fingerprint((b2,))
    r1, r2 = Ref(1.5), Ref(1.5)
    @test sel_fingerprint(r1) == sel_fingerprint(r2)
    r2[] = 2.5
    @test sel_fingerprint(r1) != sel_fingerprint(r2)
    # a captured variable that is boxed differs from one that is not, with the same value
    boxed = let n = 1
        g = () -> n
        n = 2
        g
    end
    unboxed = let n = 2
        () -> n
    end
    @test boxed() == unboxed() && any(i -> fieldtype(typeof(boxed), i) === Core.Box, 1:fieldcount(typeof(boxed)))
    @test sel_fingerprint(boxed) != sel_fingerprint(unboxed)
    # aliasing: one object seen twice differs from two equal objects; rebuilt identical objects agree
    shared, twin = [1.0], [1.0]
    @test sel_fingerprint((shared, shared)) != sel_fingerprint((shared, twin))
    build() = (v = [1.0, 2.0]; w = sparse([1, 3], [1, 2], [1.0, 2.0]); () -> (v, w, view(v, 1:2)))
    @test sel_fingerprint(build()) == sel_fingerprint(build())
    s1 = sparse([1, 3], [1, 2], [1.0, 2.0]); s2 = copy(s1); s2.nzval[1] = 9.0
    @test sel_fingerprint(() -> s1) != sel_fingerprint(() -> s2)
    # a cycle ends
    cyc = Ref{Any}(nothing); cyc[] = cyc
    @test sel_fingerprint(cyc) isa UInt64
    n1 = SelNode(nothing); n2 = SelNode(n1); n1.next = n2
    @test sel_fingerprint(n1) isa UInt64
    # the walk is bounded
    budget = SEL_NODE_BUDGET[]
    SEL_NODE_BUDGET[] = 10
    try
        @test occursin("exceeded its budget of 10 nodes", sel_message(() -> sel_fingerprint(Any[string(i) for i in 1:100])))
    finally
        SEL_NODE_BUDGET[] = budget
    end
end

# ---- the regex matrix on the quick atom group ----------------------------------------------------------------
const SEL_PRIMARY = (raw"^atom/isactive/",
                     raw"^atom/isactive/Float64/p80000_g8_contiguous_act10/half_active/warm$",
                     raw"atom/write_x\[i\]_active",
                     raw"^atom/(read_x\[i\]|write_x\[i\]_active)/",
                     raw"^atom/mul!/",
                     raw"^atom/first_activation/",
                     raw"^atom/samegroups/")
const SEL_BROAD = ("Float64", raw"/warm$", "act10")
sel_all(rep, suite, groups) = build_cases(rep; suite, groups)

@testset "the unfiltered catalog is unchanged" begin
    for rep in (1, 2, 10)
        Random.seed!(rep)
        cases = build_cases(rep; suite="quick", groups=["atom"])
        @test length(cases) == ATOM_QUICK_CASES
        @test sel_catalog_sha1(cases) == ATOM_QUICK_CATALOG_SHA1
        @test sel_order_sha1(cases) == ATOM_QUICK_ORDER_SHA1
    end
    # an empty selection is the whole catalog, in order
    all_cases = build_cases(1; suite="quick", groups=["atom"])
    @test [c.meta.id for c in selected_cases(1, "quick", ["atom"], "")] == [c.meta.id for c in all_cases]
end

@testset "selection of the quick atom group" begin
    for rep in (1, 2, 10)
        unfiltered() = build_cases(rep; suite="quick", groups=["atom"])
        selected(selector) = build_cases(rep; suite="quick", groups=["atom"], selector)
        @testset "primary rep $rep: $selection" for selection in SEL_PRIMARY
            sel_compare(unfiltered, selected, selection)
        end
        if rep == 1
            @testset "the symmetric mul! alone and the constructions" for selection in (raw"atom/mul!/.*symmetric", raw"^atom/construct/")
                sel_compare(unfiltered, selected, selection)
            end
            # nearly every case is selected: compared before anything runs, on one build that is never run
            pristine = unfiltered()
            @testset "selected_cases: $selection" for selection in (SEL_PRIMARY[1], SEL_PRIMARY[4], SEL_BROAD[2])
                selector = Regex(selection)
                @test [c.meta.id for c in selected_cases(rep, "quick", ["atom"], selection)] ==
                      [c.meta.id for c in pristine if occursin(selector, c.meta.id)]
            end
            @testset "broad: $selection" for selection in SEL_BROAD
                sel_compare(pristine, selected, selection; targeted=false, run=false)
            end
            pristine = nothing
        end
        GC.gc()
    end
    # an alternation across families, with another group
    groups = ["atom", "vectors"]
    # (the destination of a `separable/.../prox!` case is uninitialized memory, so those cases are not compared)
    unfiltered_two() = build_cases(1; suite="quick", groups)
    sel_compare(unfiltered_two, selector -> build_cases(1; suite="quick", groups, selector), raw"^atom/isactive/|^separable/NormL1/value/")
    GC.gc()
end

@testset "empty, unmatched and invalid selections" begin
    @test [c.meta.id for c in selected_cases(1, "quick", ["atom"], "")] == [c.meta.id for c in build_cases(1; suite="quick", groups=["atom"])]
    @test sel_message(() -> selected_cases(1, "quick", ["atom"], "no such case anywhere")) == "PB_CASES selects no case"
    # an invalid expression fails at once, with the exception of Regex, before the groups are even checked
    bad = sel_error(() -> Regex("("))
    @test bad !== nothing
    err = sel_error(() -> selected_cases(1, "quick", ["no such group"], "("))
    @test typeof(err) == typeof(bad) && sprint(showerror, err) == sprint(showerror, bad)
    @test sel_message(() -> selected_cases(1, "quick", ["no such group"], "x")) != sprint(showerror, bad)
end

@testset "the draws of the picks stream and the global generator" begin
    all_cases = sel_all(1, "quick", ["atom"])
    full(id) = only(filter(c -> c.meta.id == id, all_cases))
    ids = [c.meta.id for c in all_cases if c.meta.name == "write_x[i]_active"]
    @test length(ids) == 3
    write_only = build_cases(1; suite="quick", groups=["atom"], selector=r"atom/write_x\[i\]_active")
    combined = build_cases(1; suite="quick", groups=["atom"], selector=r"^atom/(read_x\[i\]|write_x\[i\]_active)/")
    for built in (write_only, combined), id in ids
        a, e = only(filter(c -> c.meta.id == id, built)), full(id)
        @test a.inputs.wids == e.inputs.wids && a.inputs.wvals == e.inputs.wvals
    end
    @test length(write_only) < length(combined) < length(all_cases)
    # no builder draws from the global generator
    Random.seed!(1); control = rand(UInt64, 4)
    for selector in (nothing, r"^atom/isactive/", r"atom/write_x\[i\]_active")
        Random.seed!(1)
        build_cases(1; suite="quick", groups=["atom"], selector)
        @test rand(UInt64, 4) == control
    end
end

@testset "the enumerated IDs are complete: quick grid" begin
    all_cases = sel_all(1, "quick", ["atom"])
    scenarios = atom_scenarios(true)
    @test length(scenarios) == 3
    registered = Set(c.meta.id for c in all_cases)
    enumerated = Set{String}()
    for s in scenarios
        ids = Set(id for (_, id) in atom_scenario_ids(s.T, s.p, s.g, s.strided, s.label; s.full, s.rows))
        shape = atom_shape(s.p, s.g, s.strided, s.label)
        mine = Set(c.meta.id for c in all_cases if c.meta.shape == shape)
        @test mine == ids
        union!(enumerated, ids)
    end
    grid = atom_grid(true)
    for T in grid.types, p in grid.sizes, (g, strided) in grid.layouts
        union!(enumerated, id for (_, id) in atom_construct_ids(T, p, g, strided))
    end
    for p in grid.sizes, (g, strided) in grid.layouts
        union!(enumerated, id for (_, id) in atom_layout_ids(p, g, strided))
    end
    for T in grid.types, g in grid.symmetric_groups
        union!(enumerated, id for (_, id) in atom_symmetric_ids(T, g))
    end
    @test enumerated == registered
    @test isempty(sel_boxed(all_cases))
end

# ---- the extended branches, one scenario at a time -----------------------------------------------------------
const SEL_SCENARIOS = [(T=Float64, p=8 * 10^4, g=8, strided=false, label="10", full=true, rows=(256,)),
                       (T=Float32, p=8 * 10^4, g=64, strided=true, label="1", full=true, rows=(256,)),
                       (T=Float64, p=8 * 10^4, g=1, strided=true, label="10", full=true, rows=()),
                       (T=Float64, p=8 * 10^4, g=1, strided=false, label="1", full=false, rows=(256,))]
const SEL_SCENARIO_REGEXES = (raw"/isactive/", raw"write_x\[i\]_active", raw"/read_x\[i\]/", raw"mul!.*z50", raw"mul!/.*rows256_z0",
                              raw"/dot_vx/.*z50", raw"dot_xy/.*atom_sparse_z50", raw"dot_xy/.*atom_atom_z0", raw"/_restartdot/.*private_fields",
                              raw"/_setvalues!/.*accessors", raw"/_nvalues/", raw"copyto!_convert/.*to_Int", raw"copyto!_convert/",
                              raw"/dropzeros!/.*zeros50", raw"/prox_coordinate!_staged/.*GroupNormLinf", raw"prox_coordinate!_5arg/.*NormL1/",
                              raw"absent_distinct_batch_x\d+/", raw"/activate_distinct_batch/", raw"rmul!_inf", raw"rmul!_finite/.*/x\d+/",
                              raw"/replace_staged/", raw"/replace_cheap/|/copyto!_dense/", raw"/broadcast_assign/|/copyto!_state/",
                              raw"/copy/|/copyto!_same_kind/", raw"/copyto!_unrelated/|/broadcast_self/", raw"/fill!_zero",
                              raw"/atomindices_view/|/atom_dot/", raw"/sweep_active")
sel_scenario_build(s; selector=nothing) = (cases = PerfCase[];
    atom_scenario_cases!(cases, 1, s.T, s.p, s.g, s.strided, atom_label_density(s.label), s.label,
                         atom_groups(s.p, s.g, s.strided), (Dict{Any,Any}(), Dict{Any,Any}(), Dict{Any,Any}());
                         s.full, s.rows, selector);
    cases)

@testset "extended scenarios: $(s.T) g=$(s.g) strided=$(s.strided) act$(s.label) full=$(s.full) rows=$(s.rows)" for s in SEL_SCENARIOS
    pristine = sel_scenario_build(s)
    ids = Set(id for (_, id) in atom_scenario_ids(s.T, s.p, s.g, s.strided, s.label; s.full, s.rows))
    @test Set(c.meta.id for c in pristine) == ids              # no ID is missing, and none is spurious
    applicable = [r for r in SEL_SCENARIO_REGEXES if any(c -> occursin(Regex(r), c.meta.id), pristine)]
    @test length(applicable) >= (s.full ? 18 : 6)
    @test isempty(sel_boxed(pristine))
    pristine = nothing
    for selection in applicable
        sel_compare(() -> sel_scenario_build(s), selector -> sel_scenario_build(s; selector), selection)
    end
    # a selection that matches no ID builds nothing
    @test isempty(sel_scenario_build(s; selector=r"^no such ID$"))
    GC.gc()
end
