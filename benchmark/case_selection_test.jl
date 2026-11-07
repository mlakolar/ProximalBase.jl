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

# The quick `sparse` group likewise (117 cases; the same for repetitions 1, 2 and 10).
const SPARSE_QUICK_CASES = 117
const SPARSE_QUICK_CATALOG_SHA1 = "d0a29c2a31041ec99b75ae531eef7d469d1c2365"
const SPARSE_QUICK_ORDER_SHA1 = "ed770d220b57af5f6b76c95b6a5c62f8ebc5c86e"

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
const SEL_TOKENS = IdDict{Any,String}()
sel_type_token(T) = get!(() -> replace(string(T), r"#[#A-Za-z0-9_!]*" => "#"), SEL_TOKENS, T)
sel_anonymous(T::DataType) = startswith(string(nameof(T)), "#")
sel_anonymous(T) = false
sel_flat(T) = isbitstype(T)           # an array of immutable bits goes by digest; any other is walked element by element

# `sel_mix!` takes scalars (numbers, symbols, strings, small tuples of those) and nothing else: `Base.hash` of an
# array samples a large one, so an array reaches the walker's hash only through `sel_digest`, a loop over every
# element (the harness digest where the element type has a digest word, a fold of the elements' own hashes else).
sel_mix!(w::SelWalker, path, x::AbstractArray) = error("an array reached the walker's hash at $path; use sel_digest")
function sel_digest(A::AbstractArray)
    digestable(A) && return digest(A)
    h = digest_size(DIGEST_SEED, A)
    for v in A
        h = digest_step(h, hash(v))
    end
    digest_mix(h)
end
function sel_mix!(w::SelWalker, path, @nospecialize(x))
    w.h = hash(x, w.h)
    w.trace === nothing || push!(w.trace, string(path, " = ", x))
    nothing
end

# the sharing number of a mutable object that is not walked into: the object itself and nothing of its contents
function sel_share!(w::SelWalker, @nospecialize(x), path)
    n = get(w.memo, x, 0)
    n != 0 && return sel_mix!(w, path, "same object as #$n")
    w.memo[x] = n = length(w.memo) + 1
    sel_mix!(w, path, "object #$n")
end

# the label of an item, built only when a trace is kept: the walk visits tens of thousands of group arrays
sel_p(w::SelWalker, path, parts...) = w.trace === nothing ? "" : string(path, parts...)

# `x` is not specialized on: the closures of the cases have a type each, and a method per type is mostly compile time
function sel_walk!(w::SelWalker, @nospecialize(x), path::String)
    (w.nodes += 1) > SEL_NODE_BUDGET[] && error("the case fingerprint exceeded its budget of $(SEL_NODE_BUDGET[]) nodes at $path")
    T = typeof(x)
    if x isa Union{Nothing,Missing}
        return sel_mix!(w, path, :nothing)
    elseif x isa Union{Number,Symbol,AbstractString,AbstractChar}
        sel_mix!(w, sel_p(w, path, "::type"), sel_type_token(T))
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
        sel_mix!(w, sel_p(w, path, "::type"), sel_type_token(T))
        sel_mix!(w, sel_p(w, path, " digest"), digest(x))
        # the digest sorts the active coordinates: the live state in its order is added, through the accessors
        # that exist on 0.5.x and 0.6 (the live part only: a stale tail of a backing array can be uninitialized)
        ids = collect(activecoordinates(x))
        sel_mix!(w, sel_p(w, path, " active order"), sel_digest(ids))
        if x isa AtomIterate
            indices = group_indices_by()
            sel_mix!(w, sel_p(w, path, " active values"), sel_digest([x[i] for k in ids for i in indices(x, k)]))
        else
            sel_mix!(w, sel_p(w, path, " active values"), sel_digest(collect(activevalues(x))))
        end
        # the sharing of every mutable backing array, by number alone: never hashed beyond the live entries
        for name in fieldnames(T)
            name in SEL_SKIPPED_FIELDS && continue
            f = getfield(x, name)
            f isa AbstractArray && ismutable(f) && sel_share!(w, f, sel_p(w, path, ".", name))
        end
        return nothing
    elseif x isa Core.Box
        return isdefined(x, :contents) ? sel_walk!(w, x.contents, sel_p(w, path, ".contents")) : sel_mix!(w, path, :undef)
    elseif x isa Base.RefValue
        return sel_walk!(w, x[], sel_p(w, path, "[]"))
    elseif x isa Array
        sel_mix!(w, sel_p(w, path, "::type"), sel_type_token(T))
        sel_mix!(w, sel_p(w, path, " size"), size(x))
        if sel_flat(eltype(x))
            if digestable(x)
                return sel_mix!(w, sel_p(w, path, " digest"), digest(x))
            end
            w.nodes += length(x)
            return sel_mix!(w, sel_p(w, path, " elements"), sel_digest(x))
        end
        for (i, v) in enumerate(x)
            sel_walk!(w, v, sel_p(w, path, "[", i, "]"))
        end
        return nothing
    elseif x isa AbstractDict
        for (k, v) in sort!(collect(x); by=pair -> hash(pair.first))
            sel_walk!(w, k, sel_p(w, path, ".key")); sel_walk!(w, v, sel_p(w, path, "[key]"))
        end
        return nothing
    elseif x isa AbstractSet
        for k in sort!(collect(x); by=hash)
            sel_walk!(w, k, sel_p(w, path, ".element"))
        end
        return nothing
    end
    closure = x isa Function || sel_anonymous(T)
    sel_mix!(w, sel_p(w, path, "::type"), sel_anonymous(T) ? "closure" : sel_type_token(T))
    for i in 1:fieldcount(T)
        name = T <: Tuple ? i : fieldname(T, i)
        name in SEL_SKIPPED_FIELDS && continue
        fpath = sel_p(w, path, ".", name)
        closure && sel_mix!(w, sel_p(w, fpath, "::fieldtype"), sel_type_token(fieldtype(T, i)))
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
# them (`run`). `sel_differences` returns the dictionary of differences found (empty when there is none), with the
# builds, for `sel_compare`, which asserts on them.
function sel_differences_full(unfiltered, build_selected, selection::AbstractString; run=true)
    run && !(unfiltered isa Function) && error("an unfiltered build that is reused must not be run")
    all_cases = unfiltered isa Function ? unfiltered() : unfiltered
    selector = Regex(selection)
    expected = filter(c -> occursin(selector, c.meta.id), all_cases)
    built = build_selected(selector)
    actual = filter(c -> occursin(selector, c.meta.id), built)
    differences = Dict{String,Vector{String}}()
    note(what, c) = push!(get!(differences, what, String[]), c.meta.id)
    if [c.meta.id for c in actual] == [c.meta.id for c in expected]
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
    else
        differences["ids"] = [c.meta.id for c in actual]
    end
    (; all_cases, expected, built, actual, differences)
end
sel_differences(unfiltered, build_selected, selection; run=true, targeted=true) =
    sel_differences_full(unfiltered, build_selected, selection; run).differences

function sel_compare(unfiltered, build_selected, selection::AbstractString; targeted=true, run=true)
    r = sel_differences_full(unfiltered, build_selected, selection; run)
    @test !isempty(r.expected)
    targeted && @test length(r.built) < length(r.all_cases)
    @test [c.meta.id for c in r.actual] == [c.meta.id for c in r.expected]
    @test isempty(r.differences)
    isempty(r.differences) || println(stderr, "selection ", repr(selection), " differs: ", r.differences)
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
    @test only(sparse_construct_ids(Float32, 1000)) == ("construct", "sparse/construct/Float32/n1000/empty/construct")
    @test [id for (_, id) in sparse_symmetric_ids(Float64, 256, 0.03)] ==
          ["sparse/mul!/Float64/p256_s8/symmetric_$z/warm" for z in ("z0", "z50")]
    # the branch of a conversion follows 4 * (2k) < length
    @test [id for (_, id) in sparse_convert_ids((1000, 64), 12800, "1000x64_d20")] ==
          ["sparse/copyto!_convert/Float64/1000x64_d20/to_$(Td)_dense_branch/warm" for Td in (Float64, Float32, Int)]
    @test [id for (_, id) in sparse_convert_ids((10^4,), 100, "n10000_s100")] ==
          ["sparse/copyto!_convert/Float64/n10000_s100/to_$(Td)_sparse_branch/warm" for Td in (Float64, Float32, Int)]
    # a vector scenario spells the mul! shape with the rows, and the first-activation layouts with the batch count
    ids = sparse_vector_ids(Float32, 10^4, 10^2; rows=(256,))
    @test ("mul!", "sparse/mul!/Float32/n10000_s100_m256/transposed_z50/warm") in ids
    @test ("first_activation", "sparse/first_activation/Float32/n10000_s100/activate_distinct_batch_x16/first_activation") in ids
    @test !any(pair -> occursin("copyto!_convert", pair[2]), ids)
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
    # the order of the active coordinates, which the harness digest sorts away
    g = [1:2, 3:4]
    ax = build_atom(Float64, 4, g, [1, 2], [1.0, 2.0, 3.0, 4.0])
    ay = build_atom(Float64, 4, g, [2, 1], [3.0, 4.0, 1.0, 2.0])
    @test digest(ax) == digest(ay) && Vector(ax) == Vector(ay)
    @test sel_fingerprint(ax) != sel_fingerprint(ay)
    @test sel_fingerprint(ax) == sel_fingerprint(build_atom(Float64, 4, g, [1, 2], [1.0, 2.0, 3.0, 4.0]))
    sx, sy = SparseIterate(Float64, 4), SparseIterate(Float64, 4)
    sx[1] = 1.0; sx[2] = 2.0; sy[2] = 2.0; sy[1] = 1.0
    @test digest(sx) == digest(sy) && Array(sx) == Array(sy)
    @test sel_fingerprint(sx) != sel_fingerprint(sy)
    # an array of mutable elements is walked element by element: one array twice against two equal arrays
    a1 = [1.0]
    @test sel_fingerprint([a1, a1]) != sel_fingerprint([[1.0], [1.0]])
    @test sel_fingerprint([[1.0], [1.0]]) == sel_fingerprint([[1.0], [1.0]])
    # the backing array of an iterate: shared with a captured one, or an equal copy
    @test sel_fingerprint((ax, ax.storage)) != sel_fingerprint((ax, copy(ax.storage)))
    @test sel_fingerprint((ax, ax.storage)) == sel_fingerprint((ax, ax.storage))
    @test sel_fingerprint((sx, sx.nzval)) != sel_fingerprint((sx, copy(sx.nzval)))
    # the walk is bounded
    budget = SEL_NODE_BUDGET[]
    SEL_NODE_BUDGET[] = 10
    try
        @test occursin("exceeded its budget of 10 nodes", sel_message(() -> sel_fingerprint(Any[string(i) for i in 1:100])))
    finally
        SEL_NODE_BUDGET[] = budget
    end
end

# the oracle reports a difference that it should see, and none between independently rebuilt identical builds
@testset "the oracle sees order and sharing" begin
    g = [1:2, 3:4]
    atom_pair(order) = (x = build_atom(Float64, 4, g, order, order == [1, 2] ? [1.0, 2.0, 3.0, 4.0] : [3.0, 4.0, 1.0, 2.0]);
                        c = PerfCase[];
                        addcase!(c, "t", "order", () -> first(activecoordinates(x)),
                                 () -> first(activecoordinates(x)) == 1 || error("the first active group is not 1");
                                 inputs=(; x));
                        c)
    sparse_pair(order) = (x = SparseIterate(Float64, 4); foreach(i -> x[i] = Float64(i), order);
                          c = PerfCase[];
                          addcase!(c, "t", "order", () -> first(activecoordinates(x)),
                                   () -> first(activecoordinates(x)) == 1 || error("the first active coordinate is not 1");
                                   inputs=(; x));
                          c)
    arrays_pair(shared) = (a = [1.0]; arrs = shared ? [a, a] : [[1.0], [1.0]]; c = PerfCase[];
                           addcase!(c, "t", "same", () -> arrs[1] === arrs[2],
                                    () -> arrs[1] === arrs[2] || error("two arrays");
                                    inputs=(;));
                           c)
    storage_pair(shared) = (x = build_atom(Float64, 4, g, [1, 2], [1.0, 2.0, 3.0, 4.0]); st = shared ? x.storage : copy(x.storage);
                            c = PerfCase[];
                            addcase!(c, "t", "same", () -> st === x.storage,
                                     () -> st === x.storage || error("not the storage");
                                     inputs=(; x));
                            c)
    differences(a, b) = sel_differences(() -> a(), selector -> b(), "^t/")
    for (a, b) in ((() -> atom_pair([1, 2]), () -> atom_pair([2, 1])),
                   (() -> sparse_pair([1, 2]), () -> sparse_pair([2, 1])),
                   (() -> arrays_pair(true), () -> arrays_pair(false)),
                   (() -> storage_pair(true), () -> storage_pair(false)))
        @test !isempty(differences(a, b))
        @test isempty(differences(a, a))
        @test isempty(differences(b, b))
    end
end

# Base.hash samples a large array: a difference at one position of a long array, in the order of the active
# coordinates, in their values, in the sharing of elements and in a type without a digest word must still be seen.
# The validators are no-ops, so that only the captured state can tell the builds apart.
@testset "the walker digests every live order and value of a long array" begin
    n = 10_000
    function swapped(v, i=3, j=5)
        w = copy(v)
        w[i], w[j] = w[j], w[i]
        w
    end
    function sparse_state(ids, vals)
        z = build_iterate(Float64, (2n,), ids, vals)
        c = PerfCase[]
        addcase!(c, "t", "order", () -> collect(activecoordinates(z))[3], noop; inputs=(; z))
        c
    end
    atom_groups_20000 = [(2i - 1):(2i) for i in 1:n]
    function atom_state(aids, vals)
        z = build_atom(Float64, 2n, atom_groups_20000, aids, vals)
        c = PerfCase[]
        addcase!(c, "t", "order", () -> collect(activecoordinates(z))[3], noop; inputs=(; z))
        c
    end
    function arrays_state(shared)
        arrs = [[Float64(i)] for i in 1:n]
        arrs[5] = shared ? arrs[3] : [3.0]        # equal contents, one object or two
        c = PerfCase[]
        addcase!(c, "t", "same", () -> arrs[3] === arrs[5], noop; inputs=(;))
        c
    end
    half_state(T, pos) = (a = fill(T(1), n); pos > 0 && (a[pos] = T(2)); c = PerfCase[];
                          addcase!(c, "t", "flat", () -> sum(a), noop; inputs=(;)); c)
    ids, ids_swapped = collect(1:n), swapped(collect(1:n))
    ones_n, distinct = ones(n), Float64.(1:n)
    pairs = [("sparse order", () -> sparse_state(ids, ones_n), () -> sparse_state(ids_swapped, ones_n)),
             ("atom order", () -> atom_state(ids, ones(2n)), () -> atom_state(ids_swapped, ones(2n))),
             ("sparse values", () -> sparse_state(ids, distinct), () -> sparse_state(ids, swapped(distinct))),
             ("atom values", () -> atom_state(ids, Float64.(1:2n)), () -> atom_state(ids, swapped(Float64.(1:2n)))),
             ("sparse order and values", () -> sparse_state(ids, distinct), () -> sparse_state(ids_swapped, swapped(distinct))),
             ("shared elements", () -> arrays_state(true), () -> arrays_state(false))]
    for pos in (2, 3, 5, 4999, 5001, 7777)
        push!(pairs, ("flat Float16 at $pos", () -> half_state(Float16, pos), () -> half_state(Float16, 0)))
    end
    @testset "$name" for (name, a, b) in pairs
        ca, cb = a(), b()
        @test only(case_fingerprint(ca...)) != only(case_fingerprint(cb...))
        @test !isempty(sel_differences(a, selector -> b(), "^t/"))
        @test isempty(sel_differences(a, selector -> a(), "^t/"))
    end
    # a changed element of a long flat array that has no digest word, directly
    u, v = fill(Float16(1), n), fill(Float16(1), n)
    v[5001] = Float16(2)
    @test sel_fingerprint(() -> u) != sel_fingerprint(() -> v)
    # an array never reaches the hash
    @test occursin("sel_digest", sel_message(() -> sel_mix!(SelWalker(), "x", [1.0])))
end

# ---- the regex matrix on the quick atom group ----------------------------------------------------------------
const SEL_PRIMARY = (raw"^atom/isactive/",
                     raw"^atom/isactive/Float64/p80000_g8_contiguous_act10/half_active/warm$",
                     raw"atom/write_x\[i\]_active",
                     raw"^atom/(read_x\[i\]|write_x\[i\]_active)/",
                     raw"^atom/mul!/",
                     raw"^atom/first_activation/",
                     raw"^atom/samegroups/")
# the expressions whose cases draw from keyed streams (the picks, the batches): the only ones repeated at repetitions 2 and 10
const SEL_KEYED = (SEL_PRIMARY[1], SEL_PRIMARY[3], SEL_PRIMARY[6])
const SEL_BROAD = (raw"/warm$", "act10")
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
        @testset "primary rep $rep: $selection" for selection in (rep == 1 ? SEL_PRIMARY : SEL_KEYED)
            sel_compare(unfiltered, selected, selection)
        end
        if rep == 1
            @testset "the symmetric mul! alone and the constructions" for selection in (raw"atom/mul!/.*symmetric", raw"^atom/construct/")
                sel_compare(unfiltered, selected, selection)
            end
            # nearly every case is selected: compared before anything runs, on one build that is never run
            pristine = unfiltered()
            @testset "selected_cases: $selection" for selection in (SEL_PRIMARY[1], SEL_PRIMARY[4], SEL_BROAD[1])
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
    # nor do the sparse builders: every draw of a scenario has its own keyed stream
    for selector in (nothing, r"^sparse/isactive/", r"^sparse/first_activation/", r"^sparse/dot_xy/", r"^sparse/mul!_matrix/",
                     r"^sparse/copyto!_convert/")
        Random.seed!(1)
        build_cases(1; suite="quick", groups=["sparse"], selector)
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
# one expression of each kind for the scenarios other than the first (the full list) and the one without `full`
const SEL_SCENARIO_REDUCED = (raw"/isactive/", raw"write_x\[i\]_active", raw"mul!.*z50", raw"/dot_vx/.*z50", raw"dot_xy/.*atom_sparse_z50",
                              raw"/_restartdot/.*private_fields", raw"copyto!_convert/.*to_Int", raw"/dropzeros!/.*zeros50",
                              raw"prox_coordinate!_5arg/.*NormL1/", raw"absent_distinct_batch_x\d+/", raw"rmul!_inf", raw"/replace_staged/",
                              raw"/copy/|/copyto!_same_kind/", raw"/atomindices_view/|/atom_dot/")
sel_scenario_build(s; selector=nothing, rep=1) = (cases = PerfCase[];
    atom_scenario_cases!(cases, rep, s.T, s.p, s.g, s.strided, atom_label_density(s.label), s.label,
                         atom_groups(s.p, s.g, s.strided), (Dict{Any,Any}(), Dict{Any,Any}(), Dict{Any,Any}());
                         s.full, s.rows, selector);
    cases)

@testset "extended scenarios: $(s.T) g=$(s.g) strided=$(s.strided) act$(s.label) full=$(s.full) rows=$(s.rows)" for s in SEL_SCENARIOS
    pristine = sel_scenario_build(s)
    ids = Set(id for (_, id) in atom_scenario_ids(s.T, s.p, s.g, s.strided, s.label; s.full, s.rows))
    @test Set(c.meta.id for c in pristine) == ids              # no ID is missing, and none is spurious
    regexes = s === SEL_SCENARIOS[1] || !s.full ? SEL_SCENARIO_REGEXES : SEL_SCENARIO_REDUCED
    applicable = [r for r in regexes if any(c -> occursin(Regex(r), c.meta.id), pristine)]
    @test length(applicable) >= (!s.full ? 6 : s === SEL_SCENARIOS[1] ? 18 : 10)
    @test isempty(sel_boxed(pristine))
    pristine = nothing
    for selection in applicable
        sel_compare(() -> sel_scenario_build(s), selector -> sel_scenario_build(s; selector), selection)
    end
    # a selection that matches no ID builds nothing
    @test isempty(sel_scenario_build(s; selector=r"^no such ID$"))
    GC.gc()
end

# ---- the sparse group: the quick group, through build_cases --------------------------------------------------
const SEL_SPARSE_PRIMARY = (raw"^sparse/isactive/",
                            raw"^sparse/dot_vx/",
                            raw"^sparse/dot_xy/",
                            raw"^sparse/mul!/",
                            raw"^sparse/mul!_matrix/",
                            raw"^sparse/first_activation/",
                            raw"^sparse/copyto!_convert/",
                            raw"^sparse/construct/",
                            raw"^sparse/isactive/Float64/n10000_s100/half_active/warm$")
# the sparse expressions whose cases draw from keyed streams, repeated at repetitions 2 and 10 (the sparse reads and
# writes draw nothing; the exact ID and the constructions are at repetition 1 only)
const SEL_SPARSE_KEYED = (SEL_SPARSE_PRIMARY[1], SEL_SPARSE_PRIMARY[2], SEL_SPARSE_PRIMARY[3], SEL_SPARSE_PRIMARY[6])
const SEL_SPARSE_BROAD = (raw"/warm$", "n10000_s100")

@testset "the unfiltered sparse catalog is unchanged" begin
    for rep in (1, 2, 10)
        Random.seed!(rep)
        cases = build_cases(rep; suite="quick", groups=["sparse"])
        @test length(cases) == SPARSE_QUICK_CASES
        @test sel_catalog_sha1(cases) == SPARSE_QUICK_CATALOG_SHA1
        @test sel_order_sha1(cases) == SPARSE_QUICK_ORDER_SHA1
    end
    @test [c.meta.id for c in selected_cases(1, "quick", ["sparse"], "")] ==
          [c.meta.id for c in build_cases(1; suite="quick", groups=["sparse"])]
end

@testset "selection of the quick sparse group" begin
    for rep in (1, 2, 10)
        unfiltered() = build_cases(rep; suite="quick", groups=["sparse"])
        selected(selector) = build_cases(rep; suite="quick", groups=["sparse"], selector)
        @testset "primary rep $rep: $selection" for selection in (rep == 1 ? SEL_SPARSE_PRIMARY : SEL_SPARSE_KEYED)
            sel_compare(unfiltered, selected, selection)
        end
        if rep == 1
            # nearly every case is selected: compared before anything runs, on one build that is never run
            pristine = unfiltered()
            @testset "selected_cases: $selection" for selection in (SEL_SPARSE_PRIMARY[1], SEL_SPARSE_PRIMARY[5], SEL_SPARSE_BROAD[1])
                selector = Regex(selection)
                @test [c.meta.id for c in selected_cases(rep, "quick", ["sparse"], selection)] ==
                      [c.meta.id for c in pristine if occursin(selector, c.meta.id)]
            end
            @testset "broad: $selection" for selection in SEL_SPARSE_BROAD
                sel_compare(pristine, selected, selection; targeted=false, run=false)
            end
            pristine = nothing
        end
        GC.gc()
    end
    # an alternation across families, with another group
    groups = ["sparse", "atom"]
    unfiltered_two() = build_cases(1; suite="quick", groups)
    sel_compare(unfiltered_two, selector -> build_cases(1; suite="quick", groups, selector), raw"^sparse/isactive/|^atom/isactive/")
    GC.gc()
end

@testset "the enumerated sparse IDs are complete: quick grid" begin
    all_cases = sel_all(1, "quick", ["sparse"])
    scenarios = sparse_scenarios(true)
    @test (length(scenarios.constructs), length(scenarios.vectors), length(scenarios.matrices),
           length(scenarios.symmetrics), length(scenarios.converts)) == (2, 2, 2, 1, 1)
    registered = Set(c.meta.id for c in all_cases)
    enumerated = Set{String}()
    mine(shapes...) = Set(c.meta.id for c in all_cases if c.meta.shape in shapes)
    for s in scenarios.constructs
        ids = Set(id for (_, id) in sparse_construct_ids(s.T, s.n))
        @test ids == Set(c.meta.id for c in all_cases if c.meta.name == "construct" && c.meta.shape == "n$(s.n)")
        union!(enumerated, ids)
    end
    for s in scenarios.vectors
        ids = Set(id for (_, id) in sparse_vector_ids(s.T, s.n, s.s; s.rows))
        shape = sparse_vector_shape(s.n, s.s)
        @test ids == mine(shape, (shape * "_m$m" for m in s.rows)...)
        union!(enumerated, ids)
    end
    for s in scenarios.matrices
        ids = Set(id for (_, id) in sparse_matrix_ids(s.T, s.dims, s.label, s.rows))
        @test ids == mine(sparse_matrix_shape(s.dims, s.label))
        union!(enumerated, ids)
    end
    for s in scenarios.symmetrics
        ids = Set(id for (_, id) in sparse_symmetric_ids(s.T, s.p, s.density))
        @test ids == mine(sparse_symmetric_shape(s.p, round(Int, s.density * s.p)))
        union!(enumerated, ids)
    end
    for dims in scenarios.converts
        ids = Set(id for (_, id) in sparse_matrix_convert_ids(dims))
        @test ids == mine(sparse_convert_shape(dims))
        union!(enumerated, ids)
    end
    @test enumerated == registered
    @test isempty(sel_boxed(all_cases))
end

# ---- the sparse extended branches, one scenario at a time ----------------------------------------------------
const SEL_SPARSE_VECTORS = [(T=Float32, n=10^4, s=10^2, rows=(256, 1024)),
                            (T=Float64, n=10^4, s=10^2, rows=(256, 1024)),
                            (T=Float64, n=10^6, s=10^3, rows=()),
                            (T=Float32, n=10^6, s=10^4, rows=())]
const SEL_SPARSE_VECTOR_REGEXES = (raw"/isactive/", raw"write_x\[i\]_active", raw"/read_x\[i\]/", raw"/first_activation/.*_x16",
                                   raw"/activate_distinct_batch/", raw"/sweep_activevalues/|/dot_x/", raw"/sweep_full/",
                                   raw"/dot_vx/.*z50", raw"/dot_xy/.*z0", raw"mul!/.*strided_z0", raw"_m1024/.*transposed_z50",
                                   raw"/rmul!_inf", raw"/rmul!_finite", raw"/fill!_coordinates_zero", raw"/prox_coordinate!_5arg",
                                   raw"/_restartdot/.*private_fields", raw"/_setvalues!/.*accessors", raw"/_values!/",
                                   raw"/copyto!_convert/", raw"copyto!_convert/.*to_Int", raw"/replace_staged/",
                                   raw"/copy/|/copyto!_same_kind/", raw"/copyto!_unrelated/|/broadcast_self/|/copyto!_activevalues/",
                                   raw"/fill!_zero", raw"/broadcast_assign/|/copyto!_state/|/copyto!_dense/",
                                   raw"/dropzeros!/.*zeros50", raw"/replace_cheap/")
const SEL_SPARSE_F32_REGEXES = (raw"/isactive/", raw"write_x\[i\]_active", raw"/first_activation/.*_x16", raw"/sweep_activevalues/|/dot_x/",
                                raw"/dot_vx/.*z50", raw"/dot_xy/.*z0", raw"mul!/.*strided_z0", raw"_m1024/.*transposed_z50", raw"/rmul!_inf",
                                raw"/prox_coordinate!_5arg", raw"/_restartdot/.*private_fields", raw"/_values!/",
                                raw"/copyto!_unrelated/|/broadcast_self/|/copyto!_activevalues/", raw"/replace_staged/",
                                raw"/dropzeros!/.*zeros50")
const SEL_SPARSE_LARGE_REGEXES = (raw"/isactive/", raw"/dot_vx/.*z50", raw"/sweep_activevalues/|/dot_x/", raw"/dropzeros!/.*zeros10",
                                  raw"/replace_cheap/", raw"/copyto!_convert/", raw"/rmul!_finite/", raw"/first_activation/.*_x16")
sel_sparse_vector_build(s; selector=nothing, rep=1) = (cases = PerfCase[];
    sparse_vector_cases!(cases, rep, s.T, s.n, s.s, Dict{Any,Any}(); s.rows, selector);
    cases)

@testset "sparse extended scenarios: $(s.T) n=$(s.n) s=$(s.s) rows=$(s.rows)" for s in SEL_SPARSE_VECTORS
    large = s.n >= 10^6
    pristine = sel_sparse_vector_build(s)
    ids = Set(id for (_, id) in sparse_vector_ids(s.T, s.n, s.s; s.rows))
    @test Set(c.meta.id for c in pristine) == ids              # no ID is missing, and none is spurious
    regexes = large ? SEL_SPARSE_LARGE_REGEXES : s.T == Float32 ? SEL_SPARSE_F32_REGEXES : SEL_SPARSE_VECTOR_REGEXES
    applicable = [r for r in regexes if any(c -> occursin(Regex(r), c.meta.id), pristine)]
    @test length(applicable) >= (large ? (s.T == Float64 ? 8 : 7) : s.T == Float64 ? 27 : 15)
    @test isempty(sel_boxed(pristine))
    pristine = nothing
    for selection in applicable
        sel_compare(() -> sel_sparse_vector_build(s), selector -> sel_sparse_vector_build(s; selector), selection)
    end
    # a selection that matches no ID builds nothing
    @test isempty(sel_sparse_vector_build(s; selector=r"^no such ID$"))
    GC.gc()
end

sel_sparse_density(label) = only(d for (d, l) in SPARSE_DENSITIES if l == label)
const SEL_SPARSE_MATRICES = [(T=Float64, dims=(1000, 64), label="0.5", rows=(256,)),
                             (T=Float32, dims=(1000, 64), label="20", rows=(256,)),
                             (T=Float64, dims=(256, 1000), label="3", rows=(256, 1000))]
const SEL_SPARSE_MATRIX_REGEXES = (raw"/sweep_via_coordinates/", raw"/sweep_via_x/", raw"/mul!_matrix/",
                                   raw"mul!_matrix/.*rows1000_z0", raw"/dot_aXb/", raw"dot_aXb/.*z50")
sel_sparse_matrix_build(s; selector=nothing, rep=1) = (cases = PerfCase[];
    sparse_matrix_cases!(cases, rep, s.T, s.dims, sel_sparse_density(s.label), s.label, s.rows, Dict{Any,Any}(); selector);
    cases)

@testset "sparse extended matrices: $(s.T) $(s.dims) d$(s.label) rows=$(s.rows)" for s in SEL_SPARSE_MATRICES
    pristine = sel_sparse_matrix_build(s)
    @test Set(c.meta.id for c in pristine) == Set(id for (_, id) in sparse_matrix_ids(s.T, s.dims, s.label, s.rows))
    applicable = [r for r in SEL_SPARSE_MATRIX_REGEXES if any(c -> occursin(Regex(r), c.meta.id), pristine)]
    @test length(applicable) >= (length(s.rows) == 2 ? 6 : 5)
    @test isempty(sel_boxed(pristine))
    pristine = nothing
    for selection in applicable
        sel_compare(() -> sel_sparse_matrix_build(s), selector -> sel_sparse_matrix_build(s; selector), selection)
    end
    @test isempty(sel_sparse_matrix_build(s; selector=r"^no such ID$"))
    GC.gc()
end

# the functions that are guarded as a whole: the Symmetric operands and the conversions of the 20% matrices
@testset "sparse extended whole-function scenarios: enumerated IDs" begin
    for (T, p, label) in ((Float64, 256, "3"), (Float32, 1024, "20"), (Float64, 1024, "3"))
        cases = PerfCase[]
        sparse_symmetric_cases!(cases, 1, T, p, sel_sparse_density(label), label, Dict{Any,Any}())
        @test Set(c.meta.id for c in cases) == Set(id for (_, id) in sparse_symmetric_ids(T, p, sel_sparse_density(label)))
        @test length(cases) == 2
    end
    for dims in ((1000, 64), (256, 1000))
        cases = PerfCase[]
        sparse_matrix_convert_cases!(cases, 1, dims)
        @test Set(c.meta.id for c in cases) == Set(id for (_, id) in sparse_matrix_convert_ids(dims))
        @test all(c -> occursin("_dense_branch/", c.meta.id), cases) && length(cases) == 3
    end
end

# the repetitions other than 1 at scenario level: one atom and one sparse extended scenario
@testset "extended scenarios at repetitions 2 and 10" for rep in (2, 10)
    a = SEL_SCENARIOS[1]
    for selection in (raw"/isactive/", raw"/first_activation/.*_x\d+")
        sel_compare(() -> sel_scenario_build(a; rep), selector -> sel_scenario_build(a; selector, rep), selection)
    end
    v = SEL_SPARSE_VECTORS[2]
    for selection in (raw"/isactive/", raw"mul!/.*_m1024/.*transposed_z50", raw"/dot_xy/.*z50", raw"/first_activation/.*_x16")
        sel_compare(() -> sel_sparse_vector_build(v; rep), selector -> sel_sparse_vector_build(v; selector, rep), selection)
    end
    GC.gc()
end
