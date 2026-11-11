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
# #844 intentionally renames the retired coordinate-prox cases to block_prox_write
# and block_prox_staged_write. The PB-only storage rename changes the family
# to block/ and its view/index case names, retaining all 118 quick cases.
# These digests pin that migrated catalog.
# The quick `block` group, unfiltered (118 cases; the same for
# repetitions 1, 2 and 10). `catalog` is the SHA-1 of the sorted IDs joined by newlines; `order` the SHA-1 of the
# rows in build order: the metadata keys and values, the evaluation count and the profile flag of every case.
# A change that alters this catalog on purpose updates these values and says why.
const BLOCK_QUICK_CASES = 118
const BLOCK_QUICK_CATALOG_SHA1 = "0e3b5693a88061569e3a60c5656032c61ce88ed5"
const BLOCK_QUICK_ORDER_SHA1 = "d916bf345b4315db34599a8574750593ca8d96cd"

# The quick `sparse` group likewise (117 cases; the same for repetitions 1, 2 and 10).
const SPARSE_QUICK_CASES = 117
const SPARSE_QUICK_CATALOG_SHA1 = "1a2530cf8db96e564f86171dcf749dc222f3abe9"
const SPARSE_QUICK_ORDER_SHA1 = "6d65d157e790d2e1f8cef55f6c86783d68a7c753"

@testset "function blocks and iterate blocks coexist in the catalog" begin
    cases = build_cases(1; suite="quick", groups=["blocks", "block"])
    ids = [c.meta.id for c in cases]
    @test length(unique(ids)) == length(ids)
    @test any(id -> startswith(id, "blocks/"), ids)
    @test any(id -> startswith(id, "block/"), ids)
end

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
# case reach. The encoding is typed and self-delimiting: every value starts with its kind and its full type
# (`sel_type_token`: the names of anonymous closure types are left out, named types stay) and ends with a closing
# marker; a container carries its length (an array its size) and every dictionary or set entry its begin and end
# markers; a closure or a struct carries its field count, its field names and, for a closure, its field types (so
# that a `Core.Box` or a `Union{Nothing,...}` field changes the result). `nothing` and `missing` are two kinds, a
# `Ref{T}` carries T, and a `Core.Box` is its own kind. Dense arrays go by content (a bit-element array through the
# harness digest or a loop over every element, an array of anything else element by element), a sparse array through
# its storage, the iterates through the harness digest plus their live order and values, and penalty and layout
# objects field by field. Every mutable object is numbered at its first visit, with the numbering shared by the
# cases of one call, so that aliasing between cases is part of the result; a cycle ends at the number. For an
# iterate the sharing graph also has an edge to every initialized child array of an array-of-arrays field (the
# groups of a `BlockIterate`): the identity of the child, never its contents. Raw addresses and the names of
# anonymous closure types never enter it. Only the layout caches and the scratch workspaces of the penalties are
# skipped (`SEL_SKIPPED_FIELDS`): the calls that the cases measure fill them, and a workspace starts as
# uninitialized memory, so neither is part of what a case holds.
#
# The walk is closed: a value whose kind is not in `SEL_SUPPORTED` makes it throw, with the case ID (added by
# `case_fingerprint`), the capture path and the kind. The walk stops with an error beyond a budget of nodes per case.
const SEL_SKIPPED_FIELDS = (:layout_cache, :scratch)
const SEL_NODE_BUDGET = Ref(5_000_000)

# the supported kinds of captured value, each with the reason that the cases need it; `sel_kind` assigns them
const SEL_SUPPORTED = [
    :nothing => "an absent option or reference",
    :missing => "a missing marker, distinct from nothing",
    :number => "scalars: step sizes, sizes, expected values (the type is part of the value)",
    :symbol => "names and option flags",
    :string => "labels and messages",
    :char => "the triangle of a Symmetric operand",
    :type => "element types and types passed as arguments",
    :module => "a module held as a value, by name",
    :flat_array => "dense arrays of immutable bits: iterates' inputs, operands, references, destinations",
    :array => "arrays of anything else, element by element: vectors of arrays, unions, layouts",
    :range => "index ranges and group ranges",
    :wrapper => "array wrappers walked by their fields: views, a BitMatrix mask, transposes, Symmetric, Diagonal, sparse matrices by storage",
    :tuple => "tuples of captured values",
    :namedtuple => "named tuples of captured values",
    :dict => "dictionaries, entries in a deterministic order",
    :set => "sets, elements in a deterministic order",
    :ref => "a Ref, with its element type",
    :box => "a Core.Box of a closure variable that the compiler boxed",
    :function => "closures and function objects, by their fields",
    :struct => "callable structs, penalties, layouts and other structs of this package or of the harness",
    :iterate => "the iterates, through the harness digest, their live order and values, and their sharing graph"]
const SEL_SUPPORTED_KINDS = Set(first.(SEL_SUPPORTED))
# modules whose structs the walker accepts, by the root of the defining module; of Base only the named wrappers
const SEL_STRUCT_ROOTS = (:Main, :ProximalBase, :LinearAlgebra, :SparseArrays)
const SEL_BASE_STRUCTS = (:SubArray, :BitArray)

mutable struct SelWalker
    h::UInt64
    memo::IdDict{Any,Int}
    nodes::Int
    trace::Union{Nothing,Vector{String}}
    census::Union{Nothing,IdDict{Any,Set{Symbol}}}        # the kinds met per type, instead of a hash
end
SelWalker(; trace=nothing, census=nothing) = SelWalker(0x0, IdDict{Any,Int}(), 0, trace, census)

# the path of a value, built from the links of the walk and rendered only when it is needed
struct SelPath
    parent::Union{Nothing,SelPath}
    sep::String
    label::Any
end
SelPath(label) = SelPath(nothing, "", label)
function Base.show(io::IO, p::SelPath)
    p.parent === nothing || show(io, p.parent)
    p.label isa Integer ? print(io, '[', p.label, ']') : print(io, p.sep, p.label)
end
sel_at(p::SelPath, sep, label) = SelPath(p, sep, label)

# a type, without the names of anonymous closure types
const SEL_TOKENS = IdDict{Any,String}()
function sel_type_token(T)
    s = get(SEL_TOKENS, T, nothing)
    s === nothing || return s
    SEL_TOKENS[T] = replace(string(T), r"#[#A-Za-z0-9_!]*" => "#")
end
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

# the label of an item, built only when a trace is kept
sel_p(w::SelWalker, path, parts...) = w.trace === nothing ? "" : string(path, parts...)

# the sharing number of a mutable object: positive at its first visit (the object is registered), negative at a
# later one. The object itself and nothing of its contents is mixed in; the result tells whether it was new.
function sel_share!(w::SelWalker, @nospecialize(x), path, idx=0)
    n = get(w.memo, x, 0)
    seen = n != 0
    seen || (w.memo[x] = n = length(w.memo) + 1)
    n = seen ? -n : n
    if w.trace === nothing
        w.h = hash(n, w.h)
    else
        sel_mix!(w, string(path, idx == 0 ? "" : "[$idx]", seen ? " same object as" : " new object"), n)
    end
    !seen
end

# a mutable array that an iterate or a layout holds
sel_object_array(f) = f isa AbstractArray && ismutable(f)

sel_struct_ok(T::DataType) =
    (m = T.name.module; nameof(Base.moduleroot(m)) in SEL_STRUCT_ROOTS || (m === Base && nameof(T) in SEL_BASE_STRUCTS))

# the kind of a value; a value of no supported kind is `:unsupported`
function sel_kind(@nospecialize(x))
    x === nothing && return :nothing
    x === missing && return :missing
    x isa Array && return isbitstype((typeof(x)::DataType).parameters[1]) ? :flat_array : :array
    x isa Number && return :number
    x isa Symbol && return :symbol
    x isa AbstractString && return :string
    x isa AbstractChar && return :char
    x isa Type && return :type
    x isa Module && return :module
    x isa Core.Box && return :box
    x isa Base.RefValue && return :ref
    x isa Union{BlockIterate,SparseIterate,SymmetricSparseIterate} && return :iterate
    x isa AbstractRange && return :range
    x isa AbstractDict && return :dict
    x isa AbstractSet && return :set
    x isa Tuple && return :tuple
    x isa NamedTuple && return :namedtuple
    x isa AbstractArray && return sel_struct_ok(typeof(x)) ? :wrapper : :unsupported
    x isa Function && return :function
    isstructtype(typeof(x)) && sel_struct_ok(typeof(x)) ? :struct : :unsupported
end

# a deterministic order for the entries of a dictionary or set that depends on nothing but the keys' content (never
# on `Base.hash` of an array); a key of a kind that cannot be ordered this way is an error
function sel_keyhash(@nospecialize(k))::UInt64
    if k isa Union{Nothing,Missing,Number,Symbol,AbstractString,AbstractChar}
        hash(k, hash(sel_type_token(typeof(k))))
    elseif k isa Type
        hash(sel_type_token(k), UInt64(1))
    elseif k isa Module
        hash(string(k), UInt64(2))
    elseif k isa Union{Tuple,NamedTuple}
        h = hash(sel_type_token(typeof(k)), UInt64(3))
        for e in k
            h = hash(sel_keyhash(e), h)
        end
        h
    elseif k isa Array && sel_flat(eltype(k))
        hash(sel_digest(k), hash((sel_type_token(typeof(k)), size(k)), UInt64(4)))
    else
        error("a key of kind $(sel_type_token(typeof(k))) in a dictionary or set cannot be ordered deterministically")
    end
end
function sel_entries(x::AbstractDict)
    es = collect(x)
    es[sortperm([sel_keyhash(e.first) for e in es]; alg=MergeSort)]
end
function sel_entries(x::AbstractSet)
    es = collect(x)
    es[sortperm([sel_keyhash(e) for e in es]; alg=MergeSort)]
end

# `x` is not specialized on: the closures of the cases have a type each, and a method per type is mostly compile time
function sel_walk!(w::SelWalker, @nospecialize(x), path::SelPath)
    (w.nodes += 1) > SEL_NODE_BUDGET[] && error("the case fingerprint exceeded its budget of $(SEL_NODE_BUDGET[]) nodes at $path")
    T = typeof(x)
    kind = sel_kind(x)
    token = sel_type_token(T)
    w.census === nothing || push!(get!(Set{Symbol}, w.census, T), kind)
    if kind === :unsupported
        w.census === nothing || return nothing
        error("unsupported captured value of kind $token at $path")
    end
    sel_mix!(w, sel_p(w, path, "::kind"), kind)
    sel_mix!(w, sel_p(w, path, "::type"), token)
    if kind === :nothing || kind === :missing
    elseif kind === :number || kind === :symbol || kind === :string || kind === :char
        sel_mix!(w, path, x)
    elseif kind === :type
        sel_mix!(w, path, sel_type_token(x))
    elseif kind === :module
        sel_mix!(w, path, string(x))
    else
        if ismutable(x)
            if !sel_share!(w, x, path)
                return sel_mix!(w, sel_p(w, path, "::end"), kind)
            end
        end
        sel_body!(w, x, path, Val(kind), T)
    end
    sel_mix!(w, sel_p(w, path, "::end"), kind)
end

sel_body!(w, @nospecialize(x), path, ::Val{:box}, T) =
    isdefined(x, :contents) ? sel_walk!(w, x.contents, sel_at(path, ".", "contents")) : sel_mix!(w, path, :undef)
sel_body!(w, @nospecialize(x), path, ::Val{:ref}, T) =
    isdefined(x, 1) ? sel_walk!(w, x[], sel_at(path, "", "[]")) : sel_mix!(w, path, :undef)

function sel_body!(w, @nospecialize(x), path, ::Val{:iterate}, T)
    digestable(x) || error("unsupported captured value of kind $(sel_type_token(T)) at $path: the element type has no digest word")
    w.census === nothing || return nothing
    sel_mix!(w, sel_p(w, path, " digest"), digest(x))
    # the digest sorts the active coordinates: the live state in its order is added, through the accessors
    # that exist on 0.5.x and 0.6 (the live part only: a stale tail of a backing array can be uninitialized)
    ids = collect(activecoordinates(x))
    sel_mix!(w, sel_p(w, path, " active order"), sel_digest(ids))
    if x isa BlockIterate
        indices = group_indices_by()
        sel_mix!(w, sel_p(w, path, " active values"), sel_digest([x[i] for k in ids for i in indices(x, k)]))
    else
        sel_mix!(w, sel_p(w, path, " active values"), sel_digest(collect(activevalues(x))))
    end
    # the sharing graph: every mutable backing array by its number, never hashed beyond the live entries, and, for
    # an array of arrays (the groups), an edge to every initialized child array when the layout array is new
    for name in fieldnames(T)
        name in SEL_SKIPPED_FIELDS && continue
        f = getfield(x, name)
        sel_object_array(f) || continue
        fpath = sel_at(path, ".", name)
        sel_mix!(w, sel_p(w, fpath, "::field"), name)
        sel_share!(w, f, fpath) && !sel_flat(eltype(f)) || continue
        w.nodes += length(f)
        sel_children!(w, f, fpath)
    end
    nothing
end
# the edges to the initialized child arrays of an array of arrays (specialized on the element type: the groups of
# a long layout are tens of thousands of arrays)
function sel_children!(w::SelWalker, f::AbstractArray, fpath)
    for i in eachindex(f)
        isassigned(f, i) || continue
        c = f[i]
        c isa AbstractArray && ismutable(c) && sel_share!(w, c, fpath, i)
    end
end

function sel_body!(w, @nospecialize(x), path, ::Val{:flat_array}, T)
    sel_mix!(w, sel_p(w, path, " size"), size(x))
    dig = digestable(x)
    dig || (w.nodes += length(x))
    w.census === nothing && sel_mix!(w, sel_p(w, path, " digest"), dig ? digest(x) : sel_digest(x))
    nothing
end

function sel_body!(w, @nospecialize(x), path, ::Val{:array}, T)
    sel_mix!(w, sel_p(w, path, " size"), size(x))
    for i in eachindex(x)
        ipath = sel_at(path, "", i)
        isassigned(x, i) ? sel_walk!(w, x[i], ipath) : sel_mix!(w, sel_p(w, ipath, ""), :undef)
    end
    nothing
end

function sel_body!(w, @nospecialize(x), path, ::Val{:range}, T)
    sel_mix!(w, sel_p(w, path, " length"), length(x))
    # an empty range is a value too (its `first`, `step` and endpoint are observable): they are encoded for every
    # range. The endpoint is the stored stop where the type has one; any other range type (a `StepRangeLen`, whose
    # `last` is out of bounds when it is empty) gets `last` when it is not empty.
    sel_walk!(w, first(x), sel_at(path, ".", "first"))
    sel_walk!(w, step(x), sel_at(path, ".", "step"))
    if x isa Union{UnitRange,StepRange,Base.OneTo,LinRange}
        sel_walk!(w, x.stop, sel_at(path, ".", "stop"))
    elseif !isempty(x)
        sel_walk!(w, last(x), sel_at(path, ".", "last"))
    end
    nothing
end

function sel_body!(w, @nospecialize(x), path, ::Val{:dict}, T)
    sel_mix!(w, sel_p(w, path, " length"), length(x))
    for (k, v) in sel_entries(x)
        sel_mix!(w, sel_p(w, path, " entry"), :begin)
        sel_walk!(w, k, sel_at(path, ".", "key")); sel_walk!(w, v, sel_at(path, "", "[key]"))
        sel_mix!(w, sel_p(w, path, " entry"), :end)
    end
    nothing
end

function sel_body!(w, @nospecialize(x), path, ::Val{:set}, T)
    sel_mix!(w, sel_p(w, path, " length"), length(x))
    for k in sel_entries(x)
        sel_mix!(w, sel_p(w, path, " entry"), :begin)
        sel_walk!(w, k, sel_at(path, ".", "element"))
        sel_mix!(w, sel_p(w, path, " entry"), :end)
    end
    nothing
end

function sel_body!(w, @nospecialize(x), path, ::Val{:tuple}, T)
    sel_mix!(w, sel_p(w, path, " length"), length(x))
    for i in 1:length(x)
        sel_walk!(w, x[i], sel_at(path, "", i))
    end
    nothing
end

function sel_body!(w, @nospecialize(x), path, ::Val{:namedtuple}, T)
    sel_mix!(w, sel_p(w, path, " names"), keys(x))
    for name in keys(x)
        sel_walk!(w, x[name], sel_at(path, ".", name))
    end
    nothing
end

# closures, function objects, structs and array wrappers: by their fields
function sel_body!(w, @nospecialize(x), path, ::Val, T)
    closure = x isa Function
    sel_mix!(w, sel_p(w, path, " fields"), fieldcount(T))
    for i in 1:fieldcount(T)
        name = fieldname(T, i)
        fpath = sel_at(path, ".", name)
        sel_mix!(w, sel_p(w, fpath, "::field"), name)
        name in SEL_SKIPPED_FIELDS && continue
        closure && sel_mix!(w, sel_p(w, fpath, "::fieldtype"), sel_type_token(fieldtype(T, i)))
        isdefined(x, i) ? sel_walk!(w, getfield(x, i), fpath) : sel_mix!(w, fpath, :undef)
    end
    nothing
end

# one hash per case; `trace` (a vector) receives the visited items, for locating a difference. The cases are not
# specialized on (each has a type of its own, and a method per case type is compile time): their callbacks are read
# through `getfield`, and the result is a vector of hashes.
function sel_root!(w::SelWalker, id, name::Symbol, @nospecialize(f))
    sel_mix!(w, string(id, "/", name), name)
    sel_walk!(w, f, SelPath(string(name)))
end
function sel_case!(w::SelWalker, @nospecialize(c))
    id = getfield(c, :meta).id
    w.h = 0x0
    w.nodes = 0
    try
        sel_root!(w, id, :operation, getfield(c, :operation))
        sel_root!(w, id, :prepare, getfield(c, :prepare))
        sel_root!(w, id, :validate, getfield(c, :validate))
    catch err
        err isa ErrorException ? throw(ErrorException("case $id: $(err.msg)")) : rethrow()
    end
    w.h
end
function case_fingerprint(@nospecialize(cases::PerfCase...); trace=nothing)
    w = SelWalker(; trace)
    UInt64[sel_case!(w, c) for c in cases]
end
sel_fingerprint(x; trace=nothing) = (w = SelWalker(; trace); sel_walk!(w, x, SelPath("x")); w.h)

# the (kind, type) pairs that the walker meets in the captured state of `cases`: the same traversal, with the same
# memo and budget, without the digests of array contents, and without an error for an unsupported kind
function sel_census(cases)
    w = SelWalker(; census=IdDict{Any,Set{Symbol}}())
    foreach(c -> sel_case!(w, c), cases)
    Set((kind, sel_type_token(T)) for (T, kinds) in w.census for kind in kinds)
end

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
    @test case_id("block", "isactive", Float64, "p8_g2", "half_active", "warm") == "block/isactive/Float64/p8_g2/half_active/warm"
    @test case_id("a", "b", Float32, "", "dense", "construct") == "a/b/Float32//dense/construct"
    c = addcase!(PerfCase[], "f", "n", noop, noop; inputs=(;), T=Float32, shape="s", layout="l", phase="p")
    @test c[1].meta.id == case_id("f", "n", Float32, "s", "l", "p")
    @test wants(nothing, "anything") && wants(r"^any", "anything") && !wants(r"^thing", "anything")
    @test wants(r"thing$", "anything") && !wants(r"x", "anything")
    # the spelling of an enumerated ID is the registered one, for every kind of enumerator
    @test only(block_construct_ids(Float64, 80, 8, true)) ==
          ("construct", "block/construct/Float64/p80_g8_strided/empty/construct")
    @test [id for (_, id) in block_layout_ids(80, 8, false)] ==
          ["block/samegroups/Float64/p80_g8_contiguous/$l/warm" for l in ("shared", "distinct_first_seen", "alternating")]
    @test [id for (_, id) in block_symmetric_ids(Float32, 64)] ==
          ["block/mul!/Float32/p1024_g64_contiguous_act10/symmetric_$z/warm" for z in ("z0", "z50")]
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
    ax = build_block(Float64, 4, g, [1, 2], [1.0, 2.0, 3.0, 4.0])
    ay = build_block(Float64, 4, g, [2, 1], [3.0, 4.0, 1.0, 2.0])
    @test digest(ax) == digest(ay) && Vector(ax) == Vector(ay)
    @test sel_fingerprint(ax) != sel_fingerprint(ay)
    @test sel_fingerprint(ax) == sel_fingerprint(build_block(Float64, 4, g, [1, 2], [1.0, 2.0, 3.0, 4.0]))
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
    # the encoding is typed and self-delimiting
    @test sel_fingerprint(Any[nothing]) != sel_fingerprint(Any[missing])
    @test sel_fingerprint(Any[Ref{Int}(1)]) != sel_fingerprint(Any[Ref{Any}(1)])
    @test sel_fingerprint(Any[Ref{Any}(1)]) != sel_fingerprint(Any[Core.Box(1)])
    @test sel_fingerprint(Any[Dict{Int,Int}()]) != sel_fingerprint(Any[Dict{Symbol,Int}()])
    @test sel_fingerprint(Any[Set{Int}()]) != sel_fingerprint(Any[Set{Symbol}()])
    @test sel_fingerprint(Any[Dict{Int,Int}()]) != sel_fingerprint(Any[Set{Int}()])
    @test sel_fingerprint((("ab", "c"),)) != sel_fingerprint((("a", "bc"),))
    @test sel_fingerprint(((1, 2), 3)) != sel_fingerprint((1, (2, 3)))
    @test sel_fingerprint((a = 1, b = 2)) != sel_fingerprint((b = 1, a = 2))
    @test sel_fingerprint(Any[[1.0], [2.0]]) != sel_fingerprint(Any[[1.0, 2.0]])
    @test sel_fingerprint(Any[1:3]) != sel_fingerprint(Any[1:4]) && sel_fingerprint(Any[1:3]) != sel_fingerprint(Any[1:2:5])
    @test sel_fingerprint(Any[Dict(:a => 1, :b => 2)]) == sel_fingerprint(Any[Dict(:b => 2, :a => 1)])      # an entry order that depends on nothing else
    @test sel_fingerprint(Dict{Any,Int}([1.0] => 1, [2.0] => 2)) == sel_fingerprint(Dict{Any,Int}([2.0] => 2, [1.0] => 1))
    # an undefined element is a marker, not an error
    @test sel_fingerprint(Vector{Any}(undef, 2)) != sel_fingerprint(Any[nothing, nothing])
    # a value of another kind is an error that names the path and the kind, and a key that cannot be ordered too
    @test occursin("unsupported captured value of kind Ptr{Nothing} at x[2]", sel_message(() -> sel_fingerprint(Any[1, Ptr{Nothing}(8)])))
    @test occursin("cannot be ordered deterministically", sel_message(() -> sel_fingerprint(Dict(Ref(1) => 1))))
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
    block_pair(order) = (x = build_block(Float64, 4, g, order, order == [1, 2] ? [1.0, 2.0, 3.0, 4.0] : [3.0, 4.0, 1.0, 2.0]);
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
    storage_pair(shared) = (x = build_block(Float64, 4, g, [1, 2], [1.0, 2.0, 3.0, 4.0]); st = shared ? x.storage : copy(x.storage);
                            c = PerfCase[];
                            addcase!(c, "t", "same", () -> st === x.storage,
                                     () -> st === x.storage || error("not the storage");
                                     inputs=(; x));
                            c)
    differences(a, b) = sel_differences(() -> a(), selector -> b(), "^t/")
    for (a, b) in ((() -> block_pair([1, 2]), () -> block_pair([2, 1])),
                   (() -> sparse_pair([1, 2]), () -> sparse_pair([2, 1])),
                   (() -> arrays_pair(true), () -> arrays_pair(false)),
                   (() -> storage_pair(true), () -> storage_pair(false)))
        @test !isempty(differences(a, b))
        @test isempty(differences(a, a))
        @test isempty(differences(b, b))
    end
end

# structs of the degenerate-instance pairs below (a struct cannot be defined inside a test set)
struct SelEmptyA end
struct SelEmptyB end
struct SelCallA end
struct SelCallB end
(::SelCallA)(x) = x
(::SelCallB)(x) = -x
mutable struct SelPartial
    a
    SelPartial() = new()
    SelPartial(a) = new(a)
end

# Pairs of states that the validators cannot tell apart (all no-ops), that a callback does tell apart, through the
# complete oracle: each pair shows that the fault fired (the callbacks' results differ between the two sides), that the
# oracle reports a difference, and that independently rebuilt equal states agree.
@testset "the oracle sees what the encoding must tell apart" begin
    # the probe holds its state and callback through `Any` references: one closure type serves every pair (a
    # method per pair type is compile time)
    probe(@nospecialize(z), @nospecialize(f)) = (r = Ref{Any}((z, f)); PerfCase((; id="probe"), () -> r[][2](r[][1]), noop, noop, nothing, false, (;)))
    function check_pair(@nospecialize(a), @nospecialize(b), @nospecialize(f))
        @test !isequal(f(a()), f(b()))
        @test !isempty(sel_differences(() -> [probe(a(), f)], sel -> [probe(b(), f)], "probe"))
        @test isempty(sel_differences(() -> [probe(a(), f)], sel -> [probe(a(), f)], "probe"))
        @test isempty(sel_differences(() -> [probe(b(), f)], sel -> [probe(b(), f)], "probe"))
    end
    @testset "nested layout sharing" begin
        p(shared) = (x = build_block(Float64, 8, block_groups(8, 2, true), [1, 2], [1., 2., 3., 4.]);
                     (x, shared ? x.groups[1] : copy(x.groups[1])))
        check_pair(() -> p(true), () -> p(false), z -> z[2] === z[1].groups[1])
    end
    @testset "nested dictionaries" begin
        ks = sort!([:a, :b, :c, :d]; by=hash)
        k1, k2, k3 = ks[1:3]
        a() = Dict{Symbol,Any}(k1 => Dict{Symbol,Any}(k2 => [1.0]), k3 => [2.0])
        b() = Dict{Symbol,Any}(k1 => Dict{Symbol,Any}(k2 => [1.0], k3 => [2.0]))
        check_pair(a, b, length)
    end
    @testset "nothing and missing" begin
        check_pair(() -> Any[nothing], () -> Any[missing], z -> z[1] === nothing)
    end
    @testset "the type parameter of a Ref" begin
        check_pair(() -> Any[Ref{Int}(1)], () -> Any[Ref{Any}(1)], z -> typeof(z[1]))
    end
    @testset "a Ref and a Core.Box" begin
        check_pair(() -> Any[Ref{Any}(1)], () -> Any[Core.Box(1)], z -> typeof(z[1]))
    end
    @testset "empty containers of other element types" begin
        check_pair(() -> Any[Dict{Int,Int}()], () -> Any[Dict{Symbol,Int}()], z -> typeof(z[1]))
        check_pair(() -> Any[Set{Int}()], () -> Any[Set{Symbol}()], z -> typeof(z[1]))
        check_pair(() -> Any[Dict{Int,Int}()], () -> Any[Set{Int}()], z -> typeof(z[1]))
    end
    @testset "sets of arrays and dictionaries with tuple keys" begin
        q(shared) = (v = [1.0, 2.0]; (Set{Any}([v]), shared ? v : copy(v)))
        check_pair(() -> q(true), () -> q(false), z -> any(e -> e === z[2], z[1]))
        check_pair(() -> Dict((:a, 1) => 1, (:b, 2) => 2), () -> Dict((:a, 2) => 1, (:b, 1) => 2), z -> get(z, (:a, 1), 0))
        check_pair(() -> Dict{Any,Int}([1.0] => 1), () -> Dict{Any,Int}([2.0] => 1), z -> only(keys(z)))
    end
    # degenerate instances of the supported kinds: empty or undefined, still observable by a callback
    @testset "empty ranges of every range type" begin
        check_pair(() -> 1:2:0, () -> 3:2:2, first)                       # the same StepRange{Int64,Int64}, length zero, step two
        check_pair(() -> 1:2:0, () -> 1:3:0, step)
        check_pair(() -> 1:0, () -> 5:4, first)
        check_pair(() -> Base.OneTo(0), () -> 5:4, first)
        check_pair(() -> Base.OneTo(0), () -> 1:0, typeof)
        check_pair(() -> StepRangeLen(1.0, 2.0, 0), () -> StepRangeLen(3.0, 2.0, 0), first)
        check_pair(() -> StepRangeLen(1.0, 2.0, 0), () -> StepRangeLen(1.0, 3.0, 0), step)
        check_pair(() -> LinRange(1.0, 2.0, 0), () -> LinRange(3.0, 4.0, 0), first)
        check_pair(() -> LinRange(1, 2, 0), () -> LinRange(1, 3, 0), step)
        check_pair(() -> 1:3, () -> 1:4, last)                           # a nonempty range: the endpoint
    end
    @testset "empty arrays" begin
        check_pair(() -> Int[], () -> Float64[], typeof)
        check_pair(() -> zeros(0, 3), () -> zeros(3, 0), size)
        check_pair(() -> Vector{Vector{Int}}(), () -> Vector{Vector{Float64}}(), typeof)
        check_pair(() -> Any[Int[]], () -> Any[Float64[]], z -> typeof(z[1]))
        check_pair(() -> Any[], () -> Int[], typeof)
    end
    @testset "empty tuples, named tuples and strings" begin
        check_pair(() -> (), () -> NamedTuple(), typeof)
        check_pair(() -> Any[()], () -> Any[NamedTuple()], z -> typeof(z[1]))
        check_pair(() -> (a=1,), () -> (b=1,), keys)
        check_pair(() -> "", () -> SubString("a", 1, 0), typeof)
        check_pair(() -> "", () -> Symbol(""), typeof)
    end
    @testset "empty dictionaries and sets: key and value types" begin
        check_pair(() -> Dict{Int,Int}(), () -> Dict{Int,Symbol}(), typeof)
        check_pair(() -> Set{Int}(), () -> Set{Int8}(), typeof)
    end
    @testset "zero-field structs and callable structs without captures" begin
        check_pair(() -> SelEmptyA(), () -> SelEmptyB(), typeof)
        check_pair(() -> SelCallA(), () -> SelCallB(), z -> z(1))
        @test isempty(sel_differences(() -> [probe(SelEmptyA(), typeof)], sel -> [probe(SelEmptyA(), typeof)], "probe"))
        @test isempty(sel_differences(() -> [probe(SelCallA(), f -> f(1))], sel -> [probe(SelCallA(), f -> f(1))], "probe"))
    end
    @testset "undefined fields and elements" begin
        check_pair(() -> SelPartial(), () -> SelPartial(1), z -> isdefined(z, :a))
        check_pair(() -> Ref{Any}(), () -> Ref{Any}(1), isassigned)
        check_pair(() -> Core.Box(), () -> Core.Box(1), z -> isdefined(z, :contents))
        check_pair(() -> Vector{Any}(undef, 1), () -> Any[nothing], z -> isassigned(z, 1))
        check_pair(() -> Vector{Vector{Int}}(undef, 1), () -> [Int[]], z -> isassigned(z, 1))
    end
    @testset "empty closures" begin
        # a closure without captures is its type and no fields: two instances of one definition agree, and it is
        # not a zero-field struct (its code is the same in every build and is not state)
        e() = () -> 1
        @test isempty(sel_differences(() -> [probe(e(), f -> f())], sel -> [probe(e(), f -> f())], "probe"))
        check_pair(() -> e(), () -> SelEmptyA(), z -> z isa Function)
    end
end

# a value of a kind that the walker does not encode is an error with the case ID, the path and the kind, never a
# placeholder; the census reports it instead
@testset "the walker fails closed" begin
    for (what, value, kind) in (("ptr", Ptr{Cvoid}(UInt(8)), "Ptr"), ("channel", Channel{Int}(1), "Channel"),
                                ("task", Task(() -> nothing), "Task"), ("io", IOBuffer(), "(IOBuffer|GenericIOBuffer)"),
                                ("weakref", WeakRef([1.0]), "WeakRef"), ("expr", :(a + b), "Expr"))
        t = (1, value)
        c = PerfCase((; id="probe/$what"), () -> t, noop, noop, nothing, false, (;))
        msg = sel_message(() -> case_fingerprint(c))
        @test occursin(Regex("case probe/$what: unsupported captured value of kind (Base\\.)?$kind"), msg)
        @test occursin("at operation.t[2]", msg)
        @test (:unsupported, sel_type_token(typeof(value))) in sel_census([c])
        # a nested closure and a field of a struct give the path of the field
        inner = let v = value
            () -> v
        end
        d = PerfCase((; id="probe/nested/$what"), () -> inner, noop, noop, nothing, false, (;))
        @test occursin("at operation.inner.v", sel_message(() -> case_fingerprint(d)))
    end
    # an iterate whose element type has no digest word is not walked into its backing arrays
    @test occursin("unsupported captured value", sel_message(() -> sel_fingerprint(BlockIterate(Float16, 4, [1:2, 3:4]))))
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
        z = build_iterate(Float64, (2length(ids),), ids, vals)
        c = PerfCase[]
        addcase!(c, "t", "order", () -> collect(activecoordinates(z))[3], noop; inputs=(; z))
        c
    end
    function block_state(aids, vals)
        m = length(aids)
        z = build_block(Float64, 2m, [(2i - 1):(2i) for i in 1:m], aids, vals)
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
    half_state(mk, pos) = (a = mk(); pos > 0 && (a[pos] += oftype(a[pos], 1)); c = PerfCase[];
                           addcase!(c, "t", "flat", () -> sum(a), noop; inputs=(;)); c)
    # A position that the running Julia's `hash` of the vector does not read at all, among the first thousand (changing
    # the element there leaves the hash alone), so that a difference there is invisible to a sampled hash and must come
    # from the walker's own digest; none where `hash` reads every element, and then that sub-test is skipped. Julia 1.13
    # reads every element of a vector of 10,000 and skips the first of one of 100,000; Julia 1.10 skips it in both. (The
    # order tests at positions 3 and 5 below fail with the sampled hash of Julia 1.10 either way.)
    function unsampled(A, change)
        h, B = hash(A), copy(A)
        for i in first(eachindex(A)):min(last(eachindex(A)), 1000)
            B[i] = change(A[i])
            hash(B) == h && return i
            B[i] = A[i]
        end
        nothing
    end
    ids, ids_swapped = collect(1:n), swapped(collect(1:n))
    ones_n, distinct = ones(n), Float64.(1:n)
    pairs = [("sparse order", () -> sparse_state(ids, ones_n), () -> sparse_state(ids_swapped, ones_n)),
             ("atom order", () -> block_state(ids, ones(2n)), () -> block_state(ids_swapped, ones(2n))),
             ("sparse order and values", () -> sparse_state(ids, distinct), () -> sparse_state(ids_swapped, swapped(distinct))),
             ("shared elements", () -> arrays_state(true), () -> arrays_state(false))]
    # the values of an iterate, and a flat Float16 array, in vectors long enough that `hash` of the current Julia skips
    # an element (the values of 100,000 coordinates for the sparse iterate, of 200,000 for the atom)
    long = 100_000
    ids_long, values_long, values_atom = collect(1:long), Float64.(1:long), Float64.(1:2long)
    bump(A, i) = (B = copy(A); B[i] += 0.5; B)
    pv, pa = unsampled(values_long, x -> x + 0.5), unsampled(values_atom, x -> x + 0.5)
    pv === nothing || push!(pairs, ("sparse values at unsampled position $pv", () -> sparse_state(ids_long, values_long),
                                    () -> sparse_state(ids_long, bump(values_long, pv))))
    pa === nothing || push!(pairs, ("atom values at unsampled position $pa", () -> block_state(ids_long, values_atom),
                                    () -> block_state(ids_long, bump(values_atom, pa))))
    ones_half() = fill(Float16(1), n)
    long_half() = Float16.(mod.(1:long, 251))
    for pos in (2, 3, 5, 4999, 5001, 7777)
        push!(pairs, ("flat Float16 at $pos", () -> half_state(ones_half, pos), () -> half_state(ones_half, 0)))
    end
    ph = unsampled(long_half(), x -> x + Float16(1))
    ph === nothing || push!(pairs, ("flat Float16 at unsampled position $ph", () -> half_state(long_half, ph), () -> half_state(long_half, 0)))
    @testset "$name" for (name, a, b) in pairs
        ca, cb = a(), b()
        @test only(case_fingerprint(ca...)) != only(case_fingerprint(cb...))
        @test !isempty(sel_differences(a, selector -> b(), "^t/"))
        @test isempty(sel_differences(a, selector -> a(), "^t/"))
    end
    # a changed element of a long flat array that has no digest word, directly
    u, v = ones_half(), ones_half()
    v[5001] = Float16(2)
    @test sel_fingerprint(() -> u) != sel_fingerprint(() -> v)
    # an array never reaches the hash
    @test occursin("sel_digest", sel_message(() -> sel_mix!(SelWalker(), "x", [1.0])))
end

# ---- the regex matrix on the quick atom group ----------------------------------------------------------------
const SEL_PRIMARY = (raw"^block/isactive/",
                     raw"^block/isactive/Float64/p80000_g8_contiguous_act10/half_active/warm$",
                     raw"block/write_x\[i\]_active",
                     raw"^block/(read_x\[i\]|write_x\[i\]_active)/",
                     raw"^block/mul!/",
                     raw"^block/first_activation/",
                     raw"^block/samegroups/")
# the expressions whose cases draw from keyed streams (the picks, the batches): the only ones repeated at repetitions 2 and 10
const SEL_KEYED = (SEL_PRIMARY[1], SEL_PRIMARY[3], SEL_PRIMARY[6])
const SEL_BROAD = (raw"/warm$", "act10")
sel_all(rep, suite, groups) = build_cases(rep; suite, groups)

# the (kind, type) pairs met in the captured state of every catalog that this file builds for a comparison
const SEL_SEEN = Set{Tuple{Symbol,String}}()
sel_note!(cases) = (union!(SEL_SEEN, sel_census(cases)); cases)

@testset "the unfiltered catalog is unchanged" begin
    for rep in (1, 2, 10)
        Random.seed!(rep)
        cases = build_cases(rep; suite="quick", groups=["block"])
        @test length(cases) == BLOCK_QUICK_CASES
        @test sel_catalog_sha1(cases) == BLOCK_QUICK_CATALOG_SHA1
        @test sel_order_sha1(cases) == BLOCK_QUICK_ORDER_SHA1
    end
    # an empty selection is the whole catalog, in order
    all_cases = build_cases(1; suite="quick", groups=["block"])
    @test [c.meta.id for c in selected_cases(1, "quick", ["block"], "")] == [c.meta.id for c in all_cases]
end

@testset "selection of the quick atom group" begin
    for rep in (1, 2, 10)
        unfiltered() = build_cases(rep; suite="quick", groups=["block"])
        selected(selector) = build_cases(rep; suite="quick", groups=["block"], selector)
        @testset "primary rep $rep: $selection" for selection in (rep == 1 ? SEL_PRIMARY : SEL_KEYED)
            sel_compare(unfiltered, selected, selection)
        end
        if rep == 1
            @testset "the symmetric mul! alone and the constructions" for selection in (raw"block/mul!/.*symmetric", raw"^block/construct/")
                sel_compare(unfiltered, selected, selection)
            end
            # nearly every case is selected: compared before anything runs, on one build that is never run
            pristine = unfiltered()
            @testset "selected_cases: $selection" for selection in (SEL_PRIMARY[1], SEL_PRIMARY[4], SEL_BROAD[1])
                selector = Regex(selection)
                @test [c.meta.id for c in selected_cases(rep, "quick", ["block"], selection)] ==
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
    groups = ["block", "vectors"]
    # (the destination of a `separable/.../prox!` case is uninitialized memory, so those cases are not compared)
    unfiltered_two() = sel_note!(build_cases(1; suite="quick", groups))
    sel_compare(unfiltered_two, selector -> build_cases(1; suite="quick", groups, selector), raw"^block/isactive/|^separable/NormL1/value/")
    GC.gc()
end

@testset "empty, unmatched and invalid selections" begin
    @test [c.meta.id for c in selected_cases(1, "quick", ["block"], "")] == [c.meta.id for c in build_cases(1; suite="quick", groups=["block"])]
    @test sel_message(() -> selected_cases(1, "quick", ["block"], "no such case anywhere")) == "PB_CASES selects no case"
    # an invalid expression fails at once, with the exception of Regex, before the groups are even checked
    bad = sel_error(() -> Regex("("))
    @test bad !== nothing
    err = sel_error(() -> selected_cases(1, "quick", ["no such group"], "("))
    @test typeof(err) == typeof(bad) && sprint(showerror, err) == sprint(showerror, bad)
    @test sel_message(() -> selected_cases(1, "quick", ["no such group"], "x")) != sprint(showerror, bad)
end

@testset "the draws of the picks stream and the global generator" begin
    all_cases = sel_all(1, "quick", ["block"])
    full(id) = only(filter(c -> c.meta.id == id, all_cases))
    ids = [c.meta.id for c in all_cases if c.meta.name == "write_x[i]_active"]
    @test length(ids) == 3
    write_only = build_cases(1; suite="quick", groups=["block"], selector=r"block/write_x\[i\]_active")
    combined = build_cases(1; suite="quick", groups=["block"], selector=r"^block/(read_x\[i\]|write_x\[i\]_active)/")
    for built in (write_only, combined), id in ids
        a, e = only(filter(c -> c.meta.id == id, built)), full(id)
        @test a.inputs.wids == e.inputs.wids && a.inputs.wvals == e.inputs.wvals
    end
    @test length(write_only) < length(combined) < length(all_cases)
    # no builder draws from the global generator
    Random.seed!(1); control = rand(UInt64, 4)
    for selector in (nothing, r"^block/isactive/", r"block/write_x\[i\]_active")
        Random.seed!(1)
        build_cases(1; suite="quick", groups=["block"], selector)
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
    all_cases = sel_note!(sel_all(1, "quick", ["block"]))
    scenarios = block_scenarios(true)
    @test length(scenarios) == 3
    registered = Set(c.meta.id for c in all_cases)
    enumerated = Set{String}()
    for s in scenarios
        ids = Set(id for (_, id) in block_scenario_ids(s.T, s.p, s.g, s.strided, s.label; s.full, s.rows))
        shape = block_shape(s.p, s.g, s.strided, s.label)
        mine = Set(c.meta.id for c in all_cases if c.meta.shape == shape)
        @test mine == ids
        union!(enumerated, ids)
    end
    grid = block_grid(true)
    for T in grid.types, p in grid.sizes, (g, strided) in grid.layouts
        union!(enumerated, id for (_, id) in block_construct_ids(T, p, g, strided))
    end
    for p in grid.sizes, (g, strided) in grid.layouts
        union!(enumerated, id for (_, id) in block_layout_ids(p, g, strided))
    end
    for T in grid.types, g in grid.symmetric_groups
        union!(enumerated, id for (_, id) in block_symmetric_ids(T, g))
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
                              raw"/dot_vx/.*z50", raw"dot_xy/.*block_sparse_z50", raw"dot_xy/.*block_block_z0", raw"/_restartdot/.*private_fields",
                              raw"/_setvalues!/.*accessors", raw"/_nvalues/", raw"copyto!_convert/.*to_Int", raw"copyto!_convert/",
                              raw"/dropzeros!/.*zeros50", raw"/block_prox_staged_write/.*GroupNormLinf", raw"block_prox_write/.*NormL1/",
                              raw"absent_distinct_batch_x\d+/", raw"/activate_distinct_batch/", raw"rmul!_inf", raw"rmul!_finite/.*/x\d+/",
                              raw"/replace_staged/", raw"/replace_cheap/|/copyto!_dense/", raw"/broadcast_assign/|/copyto!_state/",
                              raw"/copy/|/copyto!_same_kind/", raw"/copyto!_unrelated/|/broadcast_self/", raw"/fill!_zero",
                              raw"/blockindices_view/|/block_dot/", raw"/sweep_active")
# one expression of each kind for the scenarios other than the first (the full list) and the one without `full`
const SEL_SCENARIO_REDUCED = (raw"/isactive/", raw"write_x\[i\]_active", raw"mul!.*z50", raw"/dot_vx/.*z50", raw"dot_xy/.*block_sparse_z50",
                              raw"/_restartdot/.*private_fields", raw"copyto!_convert/.*to_Int", raw"/dropzeros!/.*zeros50",
                              raw"block_prox_write/.*NormL1/", raw"absent_distinct_batch_x\d+/", raw"rmul!_inf", raw"/replace_staged/",
                              raw"/copy/|/copyto!_same_kind/", raw"/blockindices_view/|/block_dot/")
sel_scenario_build(s; selector=nothing, rep=1) = (cases = PerfCase[];
    block_scenario_cases!(cases, rep, s.T, s.p, s.g, s.strided, block_label_density(s.label), s.label,
                         block_groups(s.p, s.g, s.strided), (Dict{Any,Any}(), Dict{Any,Any}(), Dict{Any,Any}());
                         s.full, s.rows, selector);
    cases)

@testset "extended scenarios: $(s.T) g=$(s.g) strided=$(s.strided) act$(s.label) full=$(s.full) rows=$(s.rows)" for s in SEL_SCENARIOS
    pristine = sel_note!(sel_scenario_build(s))
    ids = Set(id for (_, id) in block_scenario_ids(s.T, s.p, s.g, s.strided, s.label; s.full, s.rows))
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
    groups = ["sparse", "block"]
    unfiltered_two() = build_cases(1; suite="quick", groups)
    sel_compare(unfiltered_two, selector -> build_cases(1; suite="quick", groups, selector), raw"^sparse/isactive/|^block/isactive/")
    GC.gc()
end

@testset "the enumerated sparse IDs are complete: quick grid" begin
    all_cases = sel_note!(sel_all(1, "quick", ["sparse"]))
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
                                   raw"/rmul!_inf", raw"/rmul!_finite", raw"/fill!_coordinates_zero", raw"/block_prox_write",
                                   raw"/_restartdot/.*private_fields", raw"/_setvalues!/.*accessors", raw"/_values!/",
                                   raw"/copyto!_convert/", raw"copyto!_convert/.*to_Int", raw"/replace_staged/",
                                   raw"/copy/|/copyto!_same_kind/", raw"/copyto!_unrelated/|/broadcast_self/|/copyto!_activevalues/",
                                   raw"/fill!_zero", raw"/broadcast_assign/|/copyto!_state/|/copyto!_dense/",
                                   raw"/dropzeros!/.*zeros50", raw"/replace_cheap/")
const SEL_SPARSE_F32_REGEXES = (raw"/isactive/", raw"write_x\[i\]_active", raw"/first_activation/.*_x16", raw"/sweep_activevalues/|/dot_x/",
                                raw"/dot_vx/.*z50", raw"/dot_xy/.*z0", raw"mul!/.*strided_z0", raw"_m1024/.*transposed_z50", raw"/rmul!_inf",
                                raw"/block_prox_write", raw"/_restartdot/.*private_fields", raw"/_values!/",
                                raw"/copyto!_unrelated/|/broadcast_self/|/copyto!_activevalues/", raw"/replace_staged/",
                                raw"/dropzeros!/.*zeros50")
const SEL_SPARSE_LARGE_REGEXES = (raw"/isactive/", raw"/dot_vx/.*z50", raw"/sweep_activevalues/|/dot_x/", raw"/dropzeros!/.*zeros10",
                                  raw"/replace_cheap/", raw"/copyto!_convert/", raw"/rmul!_finite/", raw"/first_activation/.*_x16")
sel_sparse_vector_build(s; selector=nothing, rep=1) = (cases = PerfCase[];
    sparse_vector_cases!(cases, rep, s.T, s.n, s.s, Dict{Any,Any}(); s.rows, selector);
    cases)

@testset "sparse extended scenarios: $(s.T) n=$(s.n) s=$(s.s) rows=$(s.rows)" for s in SEL_SPARSE_VECTORS
    large = s.n >= 10^6
    pristine = sel_note!(sel_sparse_vector_build(s))
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
    pristine = sel_note!(sel_sparse_matrix_build(s))
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

# ---- the census ----------------------------------------------------------------------------------------------
# The kinds of captured value that the walker meets, over the catalogs that this file compares (the quick `blockview`,
# `sparse` and `vectors` groups and the extended scenarios), must all be supported, and the supported list holds nothing
# else but the kinds that a self-test needs and the kinds that only groups the oracle never compares reach. The census of
# the whole quick catalog (`sel_census(build_cases(1))`) is not part of this test: it needs the builds of all
# eleven groups, about two minutes cold, and belongs to the one-off audit.
const SEL_SELF_TEST_KINDS = (:missing, :symbol, :string, :module, :box, :dict, :set)
const SEL_NOT_COMPARED = [:ref => "constraints, matrices, owner"]

@testset "the census of the captured state" begin
    kinds = Set(first.(SEL_SEEN))
    @test !isempty(SEL_SEEN)
    @test :unsupported ∉ kinds
    @test issubset(kinds, SEL_SUPPORTED_KINDS)
    @test length(SEL_SUPPORTED_KINDS) == length(SEL_SUPPORTED)             # no kind is listed twice
    @test setdiff(SEL_SUPPORTED_KINDS, kinds) == Set(SEL_SELF_TEST_KINDS) ∪ Set(first.(SEL_NOT_COMPARED))
    @test isdisjoint(kinds, Set(SEL_SELF_TEST_KINDS) ∪ Set(first.(SEL_NOT_COMPARED)))
    # every kind of the list is assigned by `sel_kind`, for a value of that kind
    samples = (nothing, missing, 1, :a, "a", 'a', Int, Base, Core.Box(1), Ref(1), BlockIterate(2, [1:1, 2:2]), [1.0], Any[1], 1:2,
               view([1.0, 2.0], 1:1), (1,), (a = 1,), Dict(1 => 1), Set([1]), () -> 1, SelNode(nothing))
    @test Set(map(sel_kind, samples)) == SEL_SUPPORTED_KINDS
    @test sel_kind(Ptr{Nothing}(0)) === :unsupported && sel_kind(Channel{Int}(1)) === :unsupported
end
