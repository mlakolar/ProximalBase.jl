# Case definitions are independent of the scheduler and of measurement settings.
# Random data use stable scenario keys, so adding a scenario does not change others.
struct PerfCase{F,S,V,M,I<:NamedTuple}
    meta::M
    operation::F
    prepare::S
    validate::V
    evals::Union{Int,Nothing}
    profile::Bool
    # the read-only inputs: what the operation or prepare reads and must leave logically unchanged
    inputs::I
end
PerfCase(meta, operation, prepare, validate, evals, profile) =
    PerfCase(meta, operation, prepare, validate, evals, profile, (;))

function scenario_rng(rep, key)
    seed = UInt64(0xcbf29ce484222325) ⊻ UInt64(rep)
    for b in codeunits(key)
        seed = (seed ⊻ UInt64(b)) * UInt64(0x100000001b3)
    end
    Xoshiro(seed)
end

# Number of nonzeros of a sparse operand with N entries at the given density: density * N
# rounded to the nearest integer, at least 1.
function fixed_count(N::Integer, density::Real)
    N >= 1 || throw(ArgumentError("the number of entries must be at least 1, got $N"))
    0 < density <= 1 || throw(ArgumentError("density must be in (0, 1], got $density"))
    clamp(round(Int, density * N), 1, N)
end

# k standard normal values of type T, each redrawn from rng until it is not zero.
function nonzero_randn(rng::AbstractRNG, ::Type{T}, k::Integer) where {T}
    v = randn(rng, T, k)
    for i in eachindex(v)
        while iszero(v[i])
            v[i] = randn(rng, T)
        end
    end
    v
end

# A sparse vector or matrix with exactly fixed_count(N, density) nonzeros: distinct positions
# drawn without replacement from rng, then normal values (never zero) from the same rng.
function sprandn_fixed(rng::AbstractRNG, ::Type{T}, n::Integer, density::Real) where {T}
    k = fixed_count(n, density)
    ind = sort!(randperm(rng, n)[1:k])
    SparseVector(Int(n), ind, nonzero_randn(rng, T, k))
end

function sprandn_fixed(rng::AbstractRNG, ::Type{T}, m::Integer, n::Integer, density::Real) where {T}
    k = fixed_count(m * n, density)
    lin = sort!(randperm(rng, m * n)[1:k])
    sparse(mod1.(lin, m), cld.(lin, m), nonzero_randn(rng, T, k), Int(m), Int(n))
end

noop() = nothing
tolerance(::Type{Float32}) = 5e-4
tolerance(::Type{Float64}) = 2e-8
dense(x) = Array(x)

function close_enough(x, y)
    t = tolerance(float(eltype(x)))
    isapprox(dense(x), dense(y); atol=t, rtol=t) || error("output differs from dense reference")
end

# ---- logical digests of the read-only inputs ----
#
# A 64-bit digest of the logical record of an input, taken by a loop over every stored element (never
# Base.hash of an array, which samples large arrays). Elements compare with isequal semantics: -0.0 and
# 0.0 differ, any two NaNs are equal. Each step is the round of xxHash64, a bijection of the state for a
# given word, so one changed word always changes the digest; a final murmur3 fmix64 spreads it.
# the element types with a digest word; digestable rejects the others (Float16, BigFloat, …) when the case
# is built, as no input uses them
const DigestReal = Union{Float32,Float64,Integer}
const DIGEST_SEED = 0x243f6a8885a308d3
const DIGEST_P1 = 0x9e3779b185ebca87
const DIGEST_P2 = 0xc2b2ae3d27d4eb4f
@inline digest_step(h::UInt64, u::Integer) = bitrotate(h + (u % UInt64) * DIGEST_P2, 31) * DIGEST_P1
@inline function digest_mix(h::UInt64)
    h ⊻= h >> 33
    h *= 0xff51afd7ed558ccd
    h ⊻= h >> 33
    h *= 0xc4ceb9fe1a85ec53
    h ⊻ (h >> 33)
end
@inline digest_word(v::Float64) = isnan(v) ? 0x7ff8000000000000 : reinterpret(UInt64, v)
@inline digest_word(v::Float32) = isnan(v) ? UInt64(0x7fc00000) : UInt64(reinterpret(UInt32, v))
@inline digest_word(v::Integer) = v % UInt64
# one tag word per kind of record, so that, e.g., a Transpose and its parent differ
const DIGEST_TAGS = (; nothing=1, number=2, array=3, range=4, transpose=5, symmetric=6, diagonal=7, csc=8,
                     stored=9, atom=10, tuple=11, container=12)
@inline digest_tag(h::UInt64, tag::Symbol) = digest_step(h, getfield(DIGEST_TAGS, tag))
function digest_size(h::UInt64, A)
    for n in size(A)
        h = digest_step(h, n)
    end
    h
end
# the elements in linear order; four interleaved chains for a long array
function digest_elements(h::UInt64, A)
    n = length(A)
    if n < 16
        for v in A
            h = digest_step(h, digest_word(v))
        end
        return h
    end
    l1, l2, l3, l4 = h + DIGEST_P1, h + DIGEST_P2, h ⊻ DIGEST_P1, h ⊻ DIGEST_P2
    i = firstindex(A)
    last = i + n - 4
    if IndexStyle(A) isa IndexLinear
        @inbounds while i <= last
            l1 = digest_step(l1, digest_word(A[i]))
            l2 = digest_step(l2, digest_word(A[i+1]))
            l3 = digest_step(l3, digest_word(A[i+2]))
            l4 = digest_step(l4, digest_word(A[i+3]))
            i += 4
        end
        @inbounds while i < firstindex(A) + n
            l1 = digest_step(l1, digest_word(A[i]))
            i += 1
        end
    else
        k = 0
        for v in A
            u = digest_word(v)
            r = k & 3
            r == 0 ? (l1 = digest_step(l1, u)) : r == 1 ? (l2 = digest_step(l2, u)) :
                r == 2 ? (l3 = digest_step(l3, u)) : (l4 = digest_step(l4, u))
            k += 1
        end
    end
    digest_step(digest_step(digest_step(digest_step(h, l1), l2), l3), l4)
end
digest!(h::UInt64, ::Nothing) = digest_tag(h, :nothing)
digest!(h::UInt64, x::DigestReal) = digest_step(digest_tag(h, :number), digest_word(x))
digest!(h::UInt64, A::AbstractArray{<:DigestReal}) = digest_elements(digest_size(digest_tag(h, :array), A), A)
digest!(h::UInt64, r::AbstractRange{<:Integer}) =
    digest_step(digest_step(digest_step(digest_tag(h, :range), first(r)), step(r)), length(r))
digest!(h::UInt64, A::Union{Transpose{<:DigestReal},Adjoint{<:DigestReal}}) = digest!(digest_tag(h, :transpose), parent(A))
# only the referenced triangle is meaningful; the whole parent is digested
digest!(h::UInt64, A::Symmetric{<:DigestReal}) = digest!(digest_step(digest_tag(h, :symmetric), UInt8(A.uplo)), parent(A))
digest!(h::UInt64, A::Diagonal{<:DigestReal}) = digest!(digest_tag(h, :diagonal), A.diag)
digest!(h::UInt64, A::SparseMatrixCSC{<:DigestReal}) =
    digest!(digest!(digest!(digest_size(digest_tag(h, :csc), A), A.colptr), A.rowval), A.nzval)
# the stored (coordinate, value) pairs, stored zeros included, sorted by coordinate: the slot order is not
# part of the state. The accessors exist on 0.5.x and 0.6 and stop at the stored count, so a stale storage
# tail is not read. The coordinates of the owner are lower-triangle IDs.
function digest!(h::UInt64, x::Union{SparseIterate{<:DigestReal},SymmetricSparseIterate{<:DigestReal}})
    h = digest_step(digest_step(digest_size(digest_tag(h, :stored), x), ncoordinates(x)), nactive(x))
    ids = collect(activecoordinates(x))
    vals = activevalues(x)
    @inbounds for j in sortperm(ids)
        h = digest_step(digest_step(h, ids[j]), digest_word(vals[j]))
    end
    h
end
# the active groups (sorted), every entry, and the group layout, which is part of the state; the indices
# through the build-time spelling of atomindices (defined with the atom group below), never the cache
function digest!(h::UInt64, x::AtomIterate{<:DigestReal})
    h = digest_step(digest_step(digest_tag(h, :atom), length(x)), ncoordinates(x))
    h = digest!(h, sort!(collect(activecoordinates(x))))
    h = digest_elements(h, x)
    digest_groups(h, x, group_indices_by())
end
function digest_groups(h::UInt64, x, indices::F) where {F}
    for k in 1:ncoordinates(x)
        G = indices(x, k)
        h = G isa AbstractRange ? digest!(h, G) : digest_elements(digest_step(h, length(G)), G)
    end
    h
end
# a tuple is not specialized on: its leaves dispatch at run time, and the many tuple types of the block
# cases would otherwise each compile their own method
function digest!(h::UInt64, @nospecialize(xs::Union{Tuple,NamedTuple}))
    h = digest_step(digest_tag(h, :tuple), length(xs))
    for x in xs
        h = digest!(h, x)
    end
    h
end
# a container of arrays (a group layout): specialized, so that each element is a direct call
function digest!(h::UInt64, xs::AbstractArray{<:AbstractArray})
    h = digest_size(digest_tag(h, :container), xs)
    for x in xs
        h = digest!(h, x)
    end
    h
end
function digest!(h::UInt64, @nospecialize(xs::AbstractArray))
    h = digest_size(digest_tag(h, :container), xs)
    for x in xs
        h = digest!(h, x)
    end
    h
end
digest(x) = digest_mix(digest!(DIGEST_SEED, x))
digestable(x) = false
digestable(::Union{Nothing,DigestReal}) = true
digestable(::AbstractArray{<:DigestReal}) = true
digestable(@nospecialize(xs::AbstractArray)) = all(digestable, xs)
digestable(@nospecialize(xs::Union{Tuple,NamedTuple})) = all(digestable, xs)
# the digests of the inputs, in their order
input_digests(@nospecialize(inputs::NamedTuple)) = UInt64[digest(inputs[k]) for k in keys(inputs)]
# throws, naming the first input whose digest differs
function check_inputs(@nospecialize(inputs::NamedTuple), digests::Vector{UInt64}, when::AbstractString)
    for (j, name) in enumerate(keys(inputs))
        digest(inputs[name]) == digests[j] || error("read-only input `$name` changed $when")
    end
    nothing
end
# the reference computed when a case is built: a copy where it could share memory with the input
unaliased(r, x) = x isa AbstractArray && r isa AbstractArray && Base.mightalias(r, x) ? copy(r) : r
const AFTER_BUILD = "after the case was built"

# The canonical spelling of a case ID: the registration and the cheap ID enumerators of the guarded builders
# below share it. `wants(selector, id)` is the selection rule of `build_cases(...; selector)`: `nothing` selects
# every case, a Regex the IDs it matches (`occursin`); the selector is never inspected or parsed.
case_id(family, name, T, shape, layout, phase) = join((family, name, string(T), shape, layout, phase), "/")
wants(::Nothing, id::AbstractString) = true
wants(selector::Regex, id::AbstractString) = occursin(selector, id)
const DEFAULT_PHASE = "warm"

# `inputs` is required: what the operation, or prepare, reads and must leave logically unchanged (not the
# destinations, the state that a case changes, workspaces, caches, scalars or function objects). The
# harness digests them after the pre-trial validate and compares after the trial.
function addcase!(cases, family, name, op, validate; inputs::NamedTuple, T=Float64, shape="", layout="dense",
                  phase=DEFAULT_PHASE, prepare=noop, evals=prepare===noop ? nothing : 1, profile=false, work=0,
                  zero_alloc=phase == DEFAULT_PHASE)
    id = case_id(family, name, T, shape, layout, phase)
    all(digestable, inputs) || error("an input of $id has no logical digest")
    meta = (;id, family, name, dtype=string(T), shape, layout, phase, work, zero_alloc)
    push!(cases, PerfCase(meta, op, prepare, validate, evals, profile, inputs))
end

# The builders compute their references once, when the case is built, from the untouched input (read, not
# changed; a reference that could share its memory is copied) and from a copy of the function object (so
# building does not run the measured object), digest their inputs, and
# start every validation by comparing the inputs with those digests. `inputs` adds arrays of the call site
# that the function object holds; `reference` passes a reference already computed for the same call.
# Reference construction and checks are deliberately outside all measurements. The reference is returned.
function addprox!(cases, family, name, f, x; γ=eltype(x)(0.7), layout="dense",
                  prepare=noop, profile=false, alias=false, value=true, reference_function=f,
                  inputs=(;), reference=nothing)
    T = eltype(x)
    y = similar(x)
    declared = merge((; x), γ isa AbstractArray ? (; γ) : (;), inputs)
    built = input_digests(declared)
    # an Array input is not copied (prox does not change its argument); another input is made dense
    r = reference === nothing ? unaliased(prox(deepcopy(reference_function), x isa Array ? x : dense(x), γ), x) : reference
    function check!()
        check_inputs(declared, built, AFTER_BUILD)
        prepare(); prox!(y, f, x, γ)
        close_enough(y, r)
        all(isfinite, y) || error("nonfinite proximal output")
        nothing
    end
    addcase!(cases, family, name * "/prox!", () -> prox!(y, f, x, γ), check!;
             inputs=declared, T, shape=join(size(x), "x"), layout, prepare, profile, work=length(x))
    if value
        function checkvalue!()
            check!()
            prepare(); v = value_and_prox!(y, f, x, γ)
            close_enough(y, r)
            expected = f(y)
            t = tolerance(T)
            (v == expected || isapprox(v, expected; atol=t, rtol=t)) ||
                error("returned value differs from f(prox): $v versus $expected")
        end
        addcase!(cases, family, name * "/value_and_prox!", () -> value_and_prox!(y, f, x, γ), checkvalue!;
                 inputs=declared, T, shape=join(size(x), "x"), layout, prepare, work=length(x))
    end
    if alias
        z = copy(x)
        prep_alias!() = copyto!(z, x)
        check_alias!() = (check_inputs(declared, built, AFTER_BUILD); prep_alias!(); prox!(z, f, z, γ); close_enough(z, r))
        addcase!(cases, family, name * "/prox!_alias", () -> prox!(z, f, z, γ), check_alias!;
                 inputs=declared, T, shape=join(size(x), "x"), layout, prepare=prep_alias!, work=length(x))
    end
    r
end

function addcold!(cases, family, name, makef, x; γ=eltype(x)(0.7), inputs=(;))
    y = similar(x)
    op() = prox!(y, makef(), x, γ)
    declared = merge((; x), inputs)
    built = input_digests(declared)
    reference = unaliased(prox(deepcopy(makef()), x isa Array ? x : dense(x), γ), x)
    check!() = (check_inputs(declared, built, AFTER_BUILD); op(); close_enough(y, reference))
    addcase!(cases, family, name, op, check!; inputs=declared, T=eltype(x), shape=join(size(x), "x"),
             phase="construct_and_first_prox", profile=eltype(x)==Float64, work=length(x))
end

function addgrad!(cases, family, name, f, x; profile=false, second=true, inputs=(;))
    T = eltype(x)
    g = zeros(T, size(x))
    v = fill(T(inv(sqrt(length(x)))), size(x))
    hv = similar(g)
    declared = merge((; x), inputs)
    declared_v = merge(declared, (; v))
    built = input_digests(declared_v)
    # the input is read, not changed, while the references are computed
    f0, xd = deepcopy(f), x isa Array ? x : dense(x)
    ref_value = f0(x)
    δfd = 1e-5
    ref_fd = T == Float64 ? (f0(xd .+ δfd .* v) - f0(xd .- δfd .* v)) / (2δfd) : nothing
    function checkgrad!()
        check_inputs(declared_v, built, AFTER_BUILD)
        val = value_and_gradient!(g, f, x)
        t = tolerance(T)
        isapprox(val, ref_value; atol=t, rtol=t) || error("value/gradient objective mismatch")
        all(isfinite, g) || error("nonfinite gradient")
        if T == Float64
            isapprox(dot(g, v), ref_fd; atol=1e-5, rtol=1e-4) || error("directional gradient check failed")
        end
    end
    shape = join(size(x), "x")
    addcase!(cases, family, name * "/value", () -> f(x), checkgrad!; inputs=declared, T, shape, work=length(x))
    addcase!(cases, family, name * "/gradient!", () -> gradient!(g, f, x), checkgrad!; inputs=declared, T, shape, profile, work=length(x))
    addcase!(cases, family, name * "/value_and_gradient!", () -> value_and_gradient!(g, f, x), checkgrad!;
             inputs=declared, T, shape, work=length(x))
    if second && hashessian(f)
        δh = T == Float32 ? T(0.01) : T(1e-5)
        huber = f isa GLMLoss && f.loss isa ProximalBase.Huber
        ref_hv = if huber
            η = f0.X*xd
            f0.offset===nothing || (η .+= f0.offset)
            hweights = T.(abs.(η .- f0.y).<=f0.loss.δ)
            f0.weights===nothing || (hweights .*= f0.weights)
            f0.X'*(hweights.*(f0.X*v))
        else
            (gradient(f0, xd .+ δh .* v) .- gradient(f0, xd .- δh .* v)) ./ (2δh)
        end
        function checkhess!()
            check_inputs(declared_v, built, AFTER_BUILD)
            hessian_vec!(hv, f, x, v)
            if huber
                close_enough(hv, ref_hv)
                return
            end
            isapprox(hv, ref_hv; atol=T==Float32 ? 0.02 : 1e-5,
                     rtol=T==Float32 ? 0.02 : 1e-4) || error("Hessian direction check failed")
        end
        addcase!(cases, family, name * "/hessian_vec!", () -> hessian_vec!(hv, f, x, v), checkhess!;
                 inputs=declared_v, T, shape, work=length(x))
        q = quadratic_model(f, x)
        ref_grad = unaliased(gradient(f0, x), x)
        checkmodel!() = (check_inputs(declared_v, built, AFTER_BUILD); quadratic_model!(q, f, x);
                         close_enough(gradient(q, x), ref_grad))
        addcase!(cases, family, name * "/quadratic_model!", () -> quadratic_model!(q, f, x), checkmodel!;
                 inputs=declared, T, shape, work=length(x))
    end
end

# Tuple references descend to array blocks instead of calling tuple oracles.
function close_enough(x::Tuple, y::Tuple)
    length(x)==length(y) || error("block counts differ")
    foreach(close_enough,x,y)
    nothing
end

block_similar(x::Tuple) = map(block_similar,x)
block_similar(x) = similar(x)
block_type(x::Tuple) = block_type(first(x))
block_type(x) = eltype(x)
block_count(x::Tuple) = sum(block_count,x)
block_count(x) = 1
block_work(x::Tuple) = sum(block_work,x)
block_work(x) = length(x)
block_finite(x::Tuple) = all(block_finite,x)
block_finite(x) = all(isfinite,x)

block_functions(f) = f
block_functions(f::SeparableSum) = map(block_functions,f.fs)
block_wrap(w, fs::Tuple) = map(f -> block_wrap(w,f),fs)
block_wrap(w, f) = w(f)
block_functions(f::Scaled) = block_wrap(g -> Scaled(g,f.α),block_functions(f.f))
block_functions(f::Conjugate) = block_wrap(Conjugate,block_functions(f.f))
block_prox(fs::Tuple, x::Tuple, γ) = map((f,z) -> block_prox(f,z,γ),fs,x)
block_prox(f, x, γ) = prox(f,x,γ)
block_prox_value(fs::Tuple, x::Tuple, γ) = sum(map((f,z) -> block_prox_value(f,z,γ),fs,x))
# The leaf oracle also supplies values for conjugates without an explicit value method.
block_prox_value(f, x, γ) = value_and_prox!(similar(x),f,x,γ)
block_value(fs::Tuple, x::Tuple) = sum(map(block_value,fs,x))
block_value(f, x) = f(x)
block_gradient(fs::Tuple, x::Tuple) = map(block_gradient,fs,x)
block_gradient(f, x) = gradient(f,x)
block_hessian(fs::Tuple, x::Tuple, v::Tuple) = map(block_hessian,fs,x,v)
block_hessian(f, x, v) = hessian_vec(f,x,v)

function block_close_value(v, expected, T)
    t=tolerance(T)
    (v==expected || isapprox(v,expected;atol=t,rtol=t)) || error("block objective mismatch: $v versus $expected")
end

function addblockprox!(cases, name, f, x; value=true, operation="prox!")
    T=block_type(x); γ=T(0.7); y=block_similar(x)
    shape="$(block_count(x))x$(length(first_leaf(x)))"; work=block_work(x)
    fs=block_functions(f)
    declared=(; x); built=input_digests(declared)
    fs0,xc=deepcopy(fs),deepcopy(x)
    reference=block_prox(fs0,xc,γ)
    function check!()
        check_inputs(declared,built,AFTER_BUILD)
        prox!(y,f,x,γ)
        close_enough(y,reference)
        block_finite(y) || error("nonfinite proximal output")
    end
    addcase!(cases,"blocks",name*"/"*operation,()->prox!(y,f,x,γ),check!;
             inputs=declared,T,shape,work)
    if value
        expected=block_prox_value(fs0,xc,γ)
        function checkvalue!()
            check_inputs(declared,built,AFTER_BUILD)
            val=value_and_prox!(y,f,x,γ)
            close_enough(y,reference)
            block_finite(y) || error("nonfinite proximal output")
            block_close_value(val,expected,T)
        end
        addcase!(cases,"blocks",name*"/value_and_prox!",()->value_and_prox!(y,f,x,γ),checkvalue!;
                 inputs=declared,T,shape,work)
    end
end

first_leaf(x::Tuple) = first_leaf(first(x))
first_leaf(x) = x

function addblockgrad!(cases, name, f, x, v; full=true)
    T=block_type(x); g=block_similar(x); hv=block_similar(x)
    shape="$(block_count(x))x$(length(first_leaf(x)))"; work=block_work(x)
    fs=block_functions(f)
    declared=(; x); declared_v=(; x, v); built=input_digests(declared_v)
    fs0,xc,vc=deepcopy(fs),deepcopy(x),deepcopy(v)
    ref_value,ref_grad,ref_hv=block_value(fs0,xc),block_gradient(fs0,xc),block_hessian(fs0,xc,vc)
    verify()=check_inputs(declared_v,built,AFTER_BUILD)
    checkvalue!() = (verify(); block_close_value(f(x),ref_value,T))
    function checkgrad!()
        verify()
        gradient!(g,f,x)
        close_enough(g,ref_grad)
    end
    function checkvaluegrad!()
        verify()
        val=value_and_gradient!(g,f,x)
        close_enough(g,ref_grad)
        block_close_value(val,ref_value,T)
    end
    function checkhess!()
        verify()
        hessian_vec!(hv,f,x,v)
        close_enough(hv,ref_hv)
    end
    if full
        addcase!(cases,"blocks",name*"/value",()->f(x),checkvalue!;inputs=declared,T,shape,work)
        addcase!(cases,"blocks",name*"/gradient!",()->gradient!(g,f,x),checkgrad!;inputs=declared,T,shape,work)
    end
    addcase!(cases,"blocks",name*"/value_and_gradient!",()->value_and_gradient!(g,f,x),checkvaluegrad!;
             inputs=declared,T,shape,work)
    addcase!(cases,"blocks",name*"/hessian_vec!",()->hessian_vec!(hv,f,x,v),checkhess!;
             inputs=declared_v,T,shape,work)
    if full
        q=quadratic_model(f,x)
        function checkmodel!()
            verify()
            quadratic_model!(q,f,x)
            block_close_value(q(x),ref_value,T)
            close_enough(gradient(q,x),ref_grad)
            close_enough(hessian_vec(q,x,v),ref_hv)
        end
        addcase!(cases,"blocks",name*"/quadratic_model!",()->quadratic_model!(q,f,x),checkmodel!;
                 inputs=declared,T,shape,work)
    end
end

function block_cases!(cases, rep; quick=false)
    # Both suites retain the same small-block overhead and long-block work cases.
    for T in (Float64,Float32), n in (2,1000)
        l1=NormL1(T(0.1)); l2=SqrNormL2(); l22=SqrNormL2(T(2))
        flat=SeparableSum(l1,l2)
        fixtures=(("flat",flat,:flat,false),
                  ("flat_smooth",SeparableSum(l2,l22),:flat,true),
                  ("nested",SeparableSum(flat,SeparableSum(l22,NormL1(T(0.05)))),:nested,false),
                  ("mixed",SeparableSum(l1,SeparableSum(l2,l1)),:mixed,false),
                  ("nested_smooth",SeparableSum(SeparableSum(l2,l22),SeparableSum(SqrNormL2(T(3)),SqrNormL2(T(4)))),:nested,true),
                  ("mixed_smooth",SeparableSum(l2,SeparableSum(l22,SqrNormL2(T(3)))),:mixed,true),
                  ("scaled",Scaled(flat,T(2)),:flat,false),
                  ("conjugate_known",Conjugate(flat),:flat,false),
                  ("conjugate_fallback",Conjugate(SeparableSum(ElasticNet(T(0.1),T(0.2)),ElasticNet(T(0.3),T(0.1)))),:flat,false),
                  ("conjugate_indicator",Conjugate(SeparableSum(IndBox(T(-1),T(1)),IndBox(T(-0.5),T(2)))),:flat,false),
                  ("conjugate_scaled",Conjugate(Scaled(flat,T(2))),:flat,false))
        for (name,f,shape,smooth) in fixtures
            rng=scenario_rng(rep,"blocks/$name/$T/$n")
            makeblocks() = shape==:nested ? ((randn(rng,T,n),randn(rng,T,n)),(randn(rng,T,n),randn(rng,T,n))) :
                           shape==:mixed ? (randn(rng,T,n),(randn(rng,T,n),randn(rng,T,n))) :
                           (randn(rng,T,n),randn(rng,T,n))
            x=makeblocks()
            if smooth
                addblockgrad!(cases,name,f,x,makeblocks())
            else
                addblockprox!(cases,name,f,x)
            end
        end
    end
    for T in (Float64,Float32), k in (8,31,32,64,128), n in (k in (8,32) ? (2,1000) : (2,))
        name="n$k"
        rng=scenario_rng(rep,"blocks/$name/$T/$n")
        x=ntuple(_ -> randn(rng,T,n),k)
        f=SeparableSum(ntuple(i -> isodd(i) ? NormL1(T(0.1)) : SqrNormL2(),k))
        addblockprox!(cases,name,f,x)
        addblockprox!(cases,name,Conjugate(f),x;value=false,operation="conjugate_prox!")
        rngsmooth=scenario_rng(rep,"blocks/$(name)_smooth/$T/$n")
        xs=ntuple(_ -> randn(rngsmooth,T,n),k)
        v=ntuple(_ -> randn(rngsmooth,T,n),k)
        smooth=SeparableSum(ntuple(i -> SqrNormL2(T(isodd(i) ? 1 : 2)),k))
        addblockgrad!(cases,name,smooth,xs,v;full=false)
    end
end

# references of the ad hoc value checks, computed when the case is built, from copies
entrywise_value(f, z) = (f0 = deepcopy(f); z0 = copy(z); sum(value_entry(f0, z0[i], i) for i in eachindex(z0)))
function entrywise_steps(f, x, γ)
    f0, x0, γ0 = deepcopy(f), copy(x), copy(γ)
    r = [prox_entry(f0, x0[i], i, γ0[i]) for i in eachindex(x0)]
    r isa Vector{eltype(x)} || error("the entrywise reference has element type $(eltype(r))")
    r
end
function ordered_reference(f, x)
    tv = sum(abs, diff(x))
    f isa TotalVariation1D ? f.λ * tv : f.λ1 * sum(abs, x) + f.λ2 * tv
end
function slope_reference(f, x)
    m = sort(abs.(x); rev = true)
    sum(k -> f.λ[k] * m[k], 1:length(x))
end

function separable_cases!(cases, rep; quick=false)
    for T in (quick ? (Float64,) : (Float32, Float64)), n in (quick ? (256,) : (256, 100_000))
        rng = scenario_rng(rep, "separable/$T/$n")
        x, w = randn(rng, T, n), rand(rng, T, n) .+ T(0.2)
        fs = (NormL1(T(0.3)), NormL1(T(0.3), w), NormL1Nonneg(T(0.3)),
              ElasticNet(T(0.3), T(0.4)), IndBox(T(-1), T(1)), IndNonnegative(),
              NormL0(T(0.1)), MCP(T(0.3)), SCAD(T(0.3)), NegEntropy(T(0.2)),
              KLDivergence(w; λ=T(0.2)), NegLog(T(0.2)), Berhu(T(0.3), T(1)),
              CheckLoss(T(0.3)), HingeLoss(), EpsInsensitiveLoss(T(0.2)),
              CappedL1(T(0.3), T(1)), LogPenalty(T(0.3), T(0.5)), NormLq(T(0.3), T(0.5)),
              NormL1Nonneg(T(0.3), w))
        for (k, f) in enumerate(fs)
            quick && !(k in (1, 4)) && continue
            weighted = f isa Union{NormL1,NormL1Nonneg} && f.w !== nothing
            name = string(nameof(typeof(f)), weighted ? "_weighted" : "")
            # the weights are held by the function object
            addprox!(cases, "separable", name, f, x; profile=T==Float64 && n==100_000 && k in (1,9,10,19), alias=k in (1,4) || weighted,
                     inputs=weighted || f isa KLDivergence ? (; w) : (;))
            if weighted
                z = f isa NormL1Nonneg ? abs.(x) : x
                ref_weighted = entrywise_value(f, z)
                checkvalue() = isapprox(f(z), ref_weighted;
                                       rtol=tolerance(T), atol=tolerance(T)) || error("weighted value differs from entrywise sum")
                addcase!(cases, "separable", name * "/value", () -> f(z), checkvalue;
                         inputs=(; z, w), T, shape=string(n), work=n)
            end
        end
        if !quick
            wz = copy(w)
            wz[1:10:end] .= zero(T)
            for F in (NormL1, NormL1Nonneg)
                z = F === NormL1Nonneg ? abs.(x) : x
                for (suffix, level, weights) in (("zero_weights", T(0.3), wz), ("zero_level", zero(T), w))
                    f = F(level, weights)
                    ref_zero = entrywise_value(f, z)
                    checkzero() = isapprox(f(z), ref_zero;
                                          rtol=tolerance(T), atol=tolerance(T)) || error("zero-factor value differs from entrywise sum")
                    addcase!(cases, "separable", string(nameof(F), "_", suffix, "/value"), () -> f(z), checkzero;
                             inputs=(; z, weights), T, shape=string(n), work=n)
                end
            end
        end
        sx = SparseIterate(sprandn(rng, T, n, 0.01))
        for f in (NormL1(T(0.3)), ElasticNet(T(0.3), T(0.4)), NormL0(T(0.1)), MCP(T(0.3)))
            quick && !(f isa NormL1) && continue
            addprox!(cases, "sparse_penalties", string(nameof(typeof(f))), f, sx; layout="sparse_1pct", profile=T==Float64 && n==100_000)
        end
        steps = T(0.2) .+ rand(rng, T, n)
        for f in (NormL1(T(0.3)), ElasticNet(T(0.3), T(0.4)))
            addprox!(cases, "metric", string(nameof(typeof(f))), f, x; γ=steps, layout="diagonal_step")
        end
        if !quick
            wi = copy(w)
            wi[1], wi[2] = zero(T), T(Inf)
            failed_steps = copy(steps)
            failed_steps[1] = zero(T)
            for (suffix, weights, γ) in (("ordinary", w, steps), ("zero_infinite_weights", wi, steps),
                                         ("uncertified_step", w, failed_steps))
                for f in (ElasticNet(T(0.3), T(0.4), weights), Berhu(T(0.3), one(T), weights),
                          CheckLoss(T(0.3), weights), EpsInsensitiveLoss(T(0.2), weights),
                          HingeLoss(nothing, weights), CappedL1(T(0.3), one(T), weights))
                    y = similar(x)
                    # the comparison is exact (isequal), so the digest of the reference stands for it
                    ref_steps = digest(entrywise_steps(f, x, γ))
                    function checksteps!()
                        prox!(y, f, x, γ)
                        digest(y) == ref_steps || error("weighted steps differ from entrywise reference")
                    end
                    addcase!(cases, "metric", string(nameof(typeof(f)), "_weighted_", suffix, "/prox!"),
                             () -> prox!(y, f, x, γ), checksteps!;
                             inputs=(; x, γ, weights), T, shape=string(n), layout="diagonal_step", work=n)
                end
            end
        end
        xv = view(randn(rng, T, 2n), 1:2:2n)
        addprox!(cases, "layouts", "NormL1", NormL1(T(0.3)), xv; layout="stride2_input", alias=true)
        for f in (NormL2(T(0.3)), NormLinf(T(0.3)), IndBallL0(max(1,n÷10)),
                  IndBallL1(T(n/10)), IndBallL2(T(1)), IndSphereL2(T(1)), IndSimplex(T(1)),
                  IndHyperplane(one(T), zero(T)), IndHalfspace(one(T), zero(T)))
            quick && !(f isa Union{NormL2,IndSimplex}) && continue
            addprox!(cases, "norms_sets", string(nameof(typeof(f))), f, x;
                     profile=T==Float64 && n==100_000 && f isa Union{NormLinf,IndSimplex})
        end
        for f in (TotalVariation1D(T(0.4)), FusedLasso(T(0.2),T(0.4)), IndIsotonic(),
                  SLOPE(collect(range(T(0.5),T(0.1);length=n))))
            if f isa Union{TotalVariation1D,FusedLasso}
                ref_ordered = ordered_reference(f, copy(x))
                function checkvalue!()
                    isapprox(f(x),ref_ordered;atol=tolerance(T),rtol=tolerance(T)) ||
                        error("ordered value differs from direct sum")
                end
                addcase!(cases,"ordered",string(nameof(typeof(f)),"/value"),()->f(x),checkvalue!;
                         inputs=(; x),T,shape=string(n),profile=T==Float64 && n==100_000,work=n)
            end
            quick && !(f isa Union{TotalVariation1D,SLOPE}) && continue
            addprox!(cases, "ordered", string(nameof(typeof(f))), f, x; profile=T==Float64 && n==100_000)
        end
        groups = [j:min(j+7,n) for j in 1:8:n]
        weights = ones(T, length(groups))
        for f in (GroupNormL2(T(0.3),groups,weights), GroupNormLinf(T(0.3),groups,weights),
                  SparseGroupNormL2(T(0.2),T(0.3),groups,weights),
                  GroupMCP(T(0.3),groups,weights), GroupSCAD(T(0.3),groups,weights))
            quick && !(f isa GroupNormL2) && continue
            # the group weights are held by the penalty
            addprox!(cases, "groups", string(nameof(typeof(f))), f, x; profile=T==Float64 && n==100_000 && f isa SparseGroupNormL2,
                     inputs=(; weights))
            if !quick
                steps = T(0.2) .+ rand(rng, T, length(groups))
                addprox!(cases, "groups", string(nameof(typeof(f))), f, x; γ=steps, layout="group_steps", inputs=(; weights))
            end
            xa = copy(x); xa[1:n÷2] .= 0
            atomx = AtomIterate(xa,f)
            addprox!(cases, "groups", string(nameof(typeof(f))), f, atomx; layout="atom_half_active", inputs=(; weights))
            z = copy(atomx)
            step = T(0.7)
            group_index = length(groups)
            prep!() = copyto!(z,atomx)
            check!() = (prep!(); prox_coordinate!(z,f,group_index,step); all(isfinite,z) || error("nonfinite group update"))
            addcase!(cases, "coordinates", string(nameof(typeof(f))), () -> prox_coordinate!(z,f,group_index,step), check!;
                     inputs=(; atomx), T, shape=string(n), layout="atom", prepare=prep!, work=length(last(groups)))
        end
    end
    if !quick
        # SLOPE value at sizes beyond the (256, 100_000) loop above; inputs come from their own
        # seeded streams so the draws of every other case stay unchanged.
        for T in (Float32, Float64), n in (256, 512, 1024, 4096, 8192, 16384)
            rng = scenario_rng(rep, "ordered/SLOPE/value/$T/$n")
            x = randn(rng, T, n)
            f = SLOPE(collect(range(T(0.5), T(0.1); length = n)))
            ref_slope = slope_reference(f, copy(x))
            function checkslope!()
                isapprox(f(x), ref_slope; atol = tolerance(T), rtol = tolerance(T)) ||
                    error("SLOPE value differs from sorted-magnitude sum")
            end
            addcase!(cases, "ordered", "SLOPE/value", () -> f(x), checkslope!;
                     inputs = (; x), T, shape = string(n), work = n)
        end
    end
end

function group_layout_cases!(cases, rep)
  for T in (Float32, Float64), K in (1_000, 10_000), indices in (false, true)
    ranges = [8k-7:8k for k in 1:K]
    groups = indices ? collect.(ranges) : ranges
    p = 8K
    rng = scenario_rng(rep, "group_layouts/$T/$K/$indices")
    v = zeros(T, p)
    v[1:64] .= T(2) .* randn(rng, T, 64)
    grad = ones(T, p)
    step, k = T(0.7), 4
    for F in (GroupNormL2, GroupNormLinf, SparseGroupNormL2)
      f = F === SparseGroupNormL2 ? F(T(0.2), T(0.3), groups) : F(T(0.3), groups)
      name = string(nameof(typeof(f)))
      reference = prox(f, v, step)
      reference_value = f(v)
      reference_prox_value = f(reference)
      reference_kkt = kkt_violation(f, grad, zeros(T, p), first(groups[k]))
      coordinate_reference = copy(v)
      coordinate_reference[groups[k]] = reference[groups[k]]
      for shared in (true, false)
        layout = string(indices ? "indices" : "ranges", shared ? "_shared" : "_distinct")
        source = shared ? f : groups
        x = AtomIterate(v, source)
        y = AtomIterate(T, p, source)
        z = AtomIterate(v, source)
        f(x); f(y); f(z)
        prepare!() = copyto!(z, x)
        function checkcoordinate!()
          prepare!()
          prox_coordinate!(z, f, k, step)
          close_enough(z, coordinate_reference)
        end
        checkkkt!() = isapprox(kkt_violation(f, grad, x, k), reference_kkt;
                              atol=tolerance(T), rtol=tolerance(T)) || error("KKT differs from dense reference")
        checkvalue!() = isapprox(f(x), reference_value; atol=tolerance(T), rtol=tolerance(T)) ||
          error("value differs from dense reference")
        checkprox!() = (prox!(y, f, x, step); close_enough(y, reference))
        function checkvprox!()
          value = value_and_prox!(y, f, x, step)
          close_enough(y, reference)
          isapprox(value, reference_prox_value; atol=tolerance(T), rtol=tolerance(T)) ||
            error("proximal value differs from dense reference")
        end
        addcase!(cases, "group_layouts", name * "/coordinate", () -> prox_coordinate!(z, f, k, step), checkcoordinate!;
                 inputs=(; x), T, shape=string(K), layout, prepare=prepare!, work=8)
        addcase!(cases, "group_layouts", name * "/kkt", () -> kkt_violation(f, grad, x, k), checkkkt!;
                 inputs=(; x, grad), T, shape=string(K), layout, work=8)
        addcase!(cases, "group_layouts", name * "/value", () -> f(x), checkvalue!;
                 inputs=(; x), T, shape=string(K), layout, work=64)
        addcase!(cases, "group_layouts", name * "/prox!", () -> prox!(y, f, x, step), checkprox!;
                 inputs=(; x), T, shape=string(K), layout, work=64)
        addcase!(cases, "group_layouts", name * "/value_and_prox!", () -> value_and_prox!(y, f, x, step), checkvprox!;
                 inputs=(; x), T, shape=string(K), layout, work=64)
        if !shared
          other = F === SparseGroupNormL2 ? F(T(0.2), T(0.3), groups) : F(T(0.3), groups)
          function checkalternating!()
            checkvalue!()
            isapprox(other(x), other(v); atol=tolerance(T), rtol=tolerance(T)) ||
              error("alternating value differs from dense reference")
          end
          addcase!(cases, "group_layouts", name * "/value_alternating", () -> (f(x), other(x)), checkalternating!;
                   inputs=(; x), T, shape=string(K), layout, work=128)
        end
      end
    end
  end
end

# Values of QuadraticFunction and KronQuadratic (#520): the ordinary path of well-scaled inputs, which
# pays only the underflow check (layout = dense, sparse, diagonal, square, rectangular), and, as
# separate `layout=recovery` cases, calls whose intermediates underflow and take the recomputation.
function underflow_cases!(cases, rep; quick=false)
    for T in (quick ? (Float64,) : (Float32,Float64)), n in (quick ? (4,64) : (4,32,256,1024))
        rng = scenario_rng(rep,"quadratic_underflow/$T/$n")
        X = randn(rng,T,2n,n); H = Matrix(Symmetric(X'X/T(2n)+I)); b = randn(rng,T,n); x = randn(rng,T,n)
        S = sprandn(rng,T,n,n,min(1.0,5/n)); Asp = S'S + I
        for (layout,A) in (("dense",H),("sparse",Asp),("diagonal",Diagonal(diag(H))))
            f = QuadraticFunction(A,b,zero(T)); g = zeros(T,n); Ad = Matrix(A)
            ref_quadratic = dot(x,Ad*x)/2 + dot(b,x)
            checkv!() = isapprox(f(x),ref_quadratic;rtol=100tolerance(T),atol=100tolerance(T)) || error("value differs from reference")
            checkg!() = isapprox(value_and_gradient!(g,f,x),ref_quadratic;rtol=100tolerance(T),atol=100tolerance(T)) || error("value differs from reference")
            # the function holds A and b
            addcase!(cases,"quadratic_underflow","quadratic/value",() -> f(x),checkv!;inputs=(; x, A, b),T,shape=string(n),layout,work=n*n)
            addcase!(cases,"quadratic_underflow","quadratic/value_and_gradient!",() -> value_and_gradient!(g,f,x),checkg!;
                     inputs=(; x, A, b),T,shape=string(n),layout,work=n*n)
        end
    end
    for T in (quick ? (Float64,) : (Float32,Float64)), (p,q) in (quick ? ((4,4),(1,16)) : ((4,4),(32,32),(128,128),(1,64),(64,4)))
        rng = scenario_rng(rep,"kron_underflow/$T/$p/$q")
        X = randn(rng,T,2p,p); Y = randn(rng,T,2q,q)
        A = Matrix(Symmetric(X'X/T(2p)+I)); B = Matrix(Symmetric(Y'Y/T(2q)+I))
        Θ = randn(rng,T,p,q); C = randn(rng,T,p,q)
        for (layout,f) in (("plain",KronQuadratic(A,B)),("linear",KronQuadratic(A,B,C)))
            G = zeros(T,p,q)
            ref_kron = dot(Θ,A*Θ*B)/2 + (layout=="linear" ? dot(C,Θ) : zero(T))
            checkv!() = isapprox(f(Θ),ref_kron;rtol=100tolerance(T),atol=100tolerance(T)) || error("value differs from reference")
            checkg!() = isapprox(value_and_gradient!(G,f,Θ),ref_kron;rtol=100tolerance(T),atol=100tolerance(T)) || error("value differs from reference")
            addcase!(cases,"quadratic_underflow","kron/value",() -> f(Θ),checkv!;inputs=(; Θ),T,shape="$(p)x$q",layout,work=p*q*(p+q))
            addcase!(cases,"quadratic_underflow","kron/value_and_gradient!",() -> value_and_gradient!(G,f,Θ),checkg!;
                     inputs=(; Θ),T,shape="$(p)x$q",layout,work=p*q*(p+q))
        end
    end
    # a value that is exactly zero at a nonzero argument (A = 0, or an argument in the null space with normal
    # products): below the worst-case threshold, so the exponent test of the trigger decides (supplemental gate)
    for T in (quick ? (Float64,) : (Float32,Float64)), n in (quick ? (32,) : (32,256))
        x = ones(T,n)
        Z = zeros(T,n,n)
        N = zeros(T,n,n)
        for i in 1:2:n-1
            N[i,i] = N[i+1,i+1] = 1; N[i,i+1] = N[i+1,i] = -1
        end
        for (layout,A) in (("zero",Z),("null",N))
            f = QuadraticFunction(A); g = zeros(T,n)
            checkv!() = f(x) == 0 || error("value differs from reference")
            checkg!() = value_and_gradient!(g,f,x) == 0 || error("value differs from reference")
            addcase!(cases,"quadratic_underflow","quadratic/value",() -> f(x),checkv!;inputs=(; x, A),T,shape=string(n),layout,work=n*n)
            addcase!(cases,"quadratic_underflow","quadratic/value_and_gradient!",() -> value_and_gradient!(g,f,x),checkg!;
                     inputs=(; x, A),T,shape=string(n),layout,work=n*n)
        end
    end
    for T in (quick ? (Float64,) : (Float32,Float64)), (p,q) in (quick ? ((4,4),) : ((4,4),(32,32)))
        rng = scenario_rng(rep,"kron_zero/$T/$p/$q")
        Y = randn(rng,T,2q,q); B = Matrix(Symmetric(Y'Y/T(2q)+I)); Θ = randn(rng,T,p,q)
        f = KronQuadratic(zeros(T,p,p),B); G = zeros(T,p,q)
        checkv!() = f(Θ) == 0 || error("value differs from reference")
        checkg!() = value_and_gradient!(G,f,Θ) == 0 || error("value differs from reference")
        addcase!(cases,"quadratic_underflow","kron/value",() -> f(Θ),checkv!;inputs=(; Θ),T,shape="$(p)x$q",layout="zero",work=p*q*(p+q))
        addcase!(cases,"quadratic_underflow","kron/value_and_gradient!",() -> value_and_gradient!(G,f,Θ),checkg!;
                 inputs=(; Θ),T,shape="$(p)x$q",layout="zero",work=p*q*(p+q))
    end
    # the recomputation: every call takes it (diagnostic; reported apart from the ordinary cases)
    for T in (quick ? (Float64,) : (Float32,Float64))
        rng = scenario_rng(rep,"underflow_recovery/$T")
        sa, sb, sx = T===Float32 ? (-110,-125,80) : (-900,-1000,800)
        n = 32
        X = randn(rng,T,2n,n); H = Matrix(Symmetric(X'X/T(2n)+I))
        k = T===Float32 ? -140 : -1060
        Aq = ldexp.(H,k); xq = ldexp.(randn(rng,T,n),T===Float32 ? 30 : 40)
        fq = QuadraticFunction(Aq); gq = zeros(T,n)
        ref_q = Float64(dot(big.(xq),big.(Aq)*big.(xq))/2)
        checkq!() = isapprox(Float64(fq(xq)),ref_q;rtol=1e-4) || error("recovered value differs from BigFloat reference")
        addcase!(cases,"quadratic_underflow","quadratic/value",() -> fq(xq),checkq!;inputs=(; xq, Aq),T,shape=string(n),layout="recovery",
                 work=n*n,zero_alloc=false)
        p, q = 8, 8
        Ak = ldexp.(H[1:p,1:p],sa); Bk = ldexp.(H[1:q,1:q],sb); Θk = ldexp.(randn(rng,T,p,q),sx)
        fk = KronQuadratic(Ak,Bk)
        ref_k = Float64(dot(big.(Θk),big.(Ak)*big.(Θk)*big.(Bk))/2)
        checkk!() = isapprox(Float64(fk(Θk)),ref_k;rtol=1e-4) || error("recovered value differs from BigFloat reference")
        addcase!(cases,"quadratic_underflow","kron/value",() -> fk(Θk),checkk!;inputs=(; Θk),T,shape="$(p)x$q",layout="recovery",
                 work=p*q*(p+q),zero_alloc=false)
    end
end

function smooth_cases!(cases, rep; quick=false)
    underflow_cases!(cases, rep; quick)
    for T in (quick ? (Float64,) : (Float32,Float64)), (n,p) in (quick ? ((128,32),) : ((128,32),(1024,256)))
        rng = scenario_rng(rep,"smooth/$T/$n/$p")
        X,y,x = randn(rng,T,n,p),randn(rng,T,n),T(0.03).*randn(rng,T,p)
        for (name,f) in (("least_squares",LeastSquaresLoss(y,X)), ("logistic",LogisticLoss(rand(rng,0:1,n),X)),
                  ("poisson",PoissonLoss(rand(rng,0:3,n),X)), ("huber",HuberLoss(y,X,T(0.5))),
                  ("logistic_weighted_offset",LogisticLoss(rand(rng,0:1,n),X;weights=rand(rng,T,n),offset=randn(rng,T,n))))
            quick && !(name in ("least_squares", "logistic")) && continue
            # the losses hold X (and y where they use it; the other labels are inline)
            addgrad!(cases,"smooth",name,f,x; profile=T==Float64 && n==1024,
                     inputs=name in ("least_squares","huber") ? (; X, y) : (; X))
        end
        ls = LeastSquaresLoss(y,X)
        ref_lipschitz = opnorm(X)^2
        check_global_curvature!() = begin
            bound = lipschitz(ls)
            isfinite(bound) && bound >= ref_lipschitz * (1 - tolerance(T)) ||
                error("invalid global curvature bound")
        end
        addcase!(cases,"setup","least_squares_lipschitz",() -> lipschitz(ls),check_global_curvature!;
                 inputs=(; X, y),T,shape="$(n)x$p",phase="setup",zero_alloc=false,work=n*p)
        addprox!(cases,"least_squares","tall",ls,x;profile=T==Float64 && n==1024,inputs=(; X, y))
        addcold!(cases,"least_squares","tall",() -> LeastSquaresLoss(y,X),x;inputs=(; X, y))
        B = randn(rng,T,p,3); Y = randn(rng,T,n,3)
        addgrad!(cases,"least_squares_multi","three_responses",LeastSquaresLoss(Y,X),B;inputs=(; X, Y))
        L = zeros(T,p)
        ref_curvatures = vec(sum(abs2,X;dims=1))
        checkL!() = (coordinate_lipschitz!(L,ls); close_enough(L,ref_curvatures))
        addcase!(cases,"coordinates","least_squares_curvatures",() -> coordinate_lipschitz!(L,ls),checkL!;
                 inputs=(; X, y),T,shape="$(n)x$p",work=n*p)
        G = zeros(T,p); penalty=NormL1(T(0.3)); step=T(0.1)/sum(abs2,X)
        checkG!() = (gradient_mapping!(G,ls,penalty,x,step);all(isfinite,G) || error("nonfinite gradient mapping"))
        addcase!(cases,"calculus","gradient_mapping!",() -> gradient_mapping!(G,ls,penalty,x,step),checkG!;
                 inputs=(; x, X, y),T,shape="$(n)x$p",work=n*p)
        H=Matrix(Symmetric(X'X+I)); q=QuadraticFunction(H,randn(rng,T,p),zero(T))
        addgrad!(cases,"quadratic","dense",q,x;inputs=(; H))
        addprox!(cases,"quadratic","dense",q,x;inputs=(; H))
        addcold!(cases,"quadratic","dense",() -> QuadraticFunction(H,zeros(T,p),zero(T)),x;inputs=(; H))
        for f in (Regularize(ls,T(0.3)), Precompose(SqrNormL2(T(0.3)),T(-2)),
                  Translate(SqrNormL2(T(0.3)),ones(T,p)), Sum(ls,SqrNormL2(T(0.3))))
            addgrad!(cases,"compositions",string(nameof(typeof(f))),f,x;
                     inputs=f isa Union{Regularize,Sum} ? (; X, y) : (;))
        end
        # An actual fixed-work solver trace, with reset outside the measured block.
        β=zeros(T,p); grad=similar(β); r=copy(y); a=vec(sum(abs2,X;dims=1))
        prep_cd!() = (fill!(β,0);copyto!(r,y))
        function cdtrace!()
            for sweep in 1:5, j in 1:p
                old=β[j]; col=view(X,:,j)
                new=prox_entry(penalty,old+dot(col,r)/a[j],j,inv(a[j]))
                β[j]=new; axpy!(old-new,col,r)
            end
            β
        end
        # the objective bound is fixed here; the residual depends on the output β, so X and y are first
        # compared with their digests at construction
        cd_inputs = (; X, y, a)
        cd_built = input_digests(cd_inputs)
        cd_bound = sum(abs2,y)/2+1e-3
        checkcd!() = (check_inputs(cd_inputs,cd_built,AFTER_BUILD); prep_cd!();cdtrace!(); close_enough(r,y-X*β);
                      ls(β)+penalty(β)<=cd_bound || error("CD objective increased"))
        addcase!(cases,"solver_traces","lasso_5_sweeps",cdtrace!,checkcd!;inputs=cd_inputs,T,shape="$(n)x$p",prepare=prep_cd!,
                 profile=T==Float64 && n==1024,work=5n*p)
    end
    for T in (quick ? (Float64,) : (Float32,Float64)), (n,p) in (quick ? ((32,128),) : ((32,128),(128,1024)))
        rng=scenario_rng(rep,"wide/$T/$n/$p")
        X,y,x=randn(rng,T,n,p),randn(rng,T,n),randn(rng,T,p)
        f=LeastSquaresLoss(y,X)
        addprox!(cases,"least_squares","wide",f,x;profile=T==Float64 && p==1024,inputs=(; X, y))
        addcold!(cases,"least_squares","wide",() -> LeastSquaresLoss(y,X),x;inputs=(; X, y))
        sparsex=SparseIterate(sprandn(rng,T,p,0.01))
        addgrad!(cases,"sparse_smooth","least_squares",f,sparsex;second=false,inputs=(; X, y))
        A=randn(rng,T,4,p); b=zeros(T,4); C=IndAffine(A,b)
        constrained=Constrained(f,C)
        # IndAffine holds A (not b)
        addprox!(cases,"affine_constrained","least_squares",constrained,x;inputs=(; X, y, A))
        addcold!(cases,"affine_constrained","least_squares",() -> Constrained(LeastSquaresLoss(y,X),C),x;inputs=(; X, y, A))
    end
end

function matrix_cases!(cases, rep; quick=false)
    for T in (quick ? (Float64,) : (Float32,Float64)), p in (quick ? (32,) : (32,256))
        rng=scenario_rng(rep,"matrices/$T/$p")
        M=randn(rng,T,p,p÷2); A=Matrix(Symmetric(randn(rng,T,p,p)))
        for f in (NuclearNorm(T(0.5)), IndBallRank(4), SingularValue(NormL1(T(0.5))))
            addprox!(cases,"svd",string(nameof(typeof(f))),f,M;profile=T==Float64 && p==256)
        end
        for f in (IndPSD(),IndPSD(rank=4),TracePSD(T(0.2)),Spectral(NormL1(T(0.2))))
            addprox!(cases,"eigen",string(nameof(typeof(f)),f isa IndPSD && f.rank !== nothing ? "_rank4" : ""),f,A;
                     profile=T==Float64 && p==256)
        end
        for f in (IndPattern(rand(rng,p,p).>0.8), IndBanded(3,p), IndToeplitz(symmetric=true))
            addprox!(cases,"matrix_sets",string(nameof(typeof(f))),f,A;profile=T==Float64 && p==256)
        end
        B=randn(rng,T,2p,p); S=Matrix(Symmetric(B'B/(2p)+I)); P=Matrix(T(1)I,p,p)
        addgrad!(cases,"logdet","positive_definite",LogDetLoss(S),P;profile=T==Float64 && p==256)
        addprox!(cases,"logdet","positive_definite",LogDetLoss(S),P)
        kron=KronQuadratic(S,Matrix(T(1)I,4,4),randn(rng,T,p,4))
        addgrad!(cases,"quadratic","kron",kron,randn(rng,T,p,4);profile=T==Float64 && p==256)
        sp=SymmetricSparseIterate(A .* (abs.(A).>T(1.5)))
        addprox!(cases,"sparse_penalties","symmetric_NormL1",NormL1(T(0.3)),sp;layout="symmetric_sparse")
    end
    for T in (quick ? (Float64,) : (Float32,Float64)), p in (quick ? (128,) : (128,512))
        rng=scenario_rng(rep,"partial/$T/$p")
        U=Matrix(qr(randn(rng,T,p,4)).Q)[:,1:4]
        V=Matrix(qr(randn(rng,T,p÷2,4)).Q)[:,1:4]
        M=U*Diagonal(T[20,16,13,10])*V' + T(0.001).*randn(rng,T,p,p÷2)
        A=U*Diagonal(T[20,16,13,10])*U' + T(0.001).*Matrix(Symmetric(randn(rng,T,p,p)))
        for (name,f,x) in (("nuclear",NuclearNorm(T(0.5);partial=true),M),
                           ("rank4",IndBallRank(4;partial=true),M),
                           ("psd_rank4",IndPSD(rank=4,partial=true),A),
                           ("psd",IndPSD(partial=true),A - T(0.5)*I),
                           ("trace",TracePSD(T(0.5);partial=true),A))
            full = name=="nuclear" ? NuclearNorm(T(0.5)) : name=="rank4" ? IndBallRank(4) :
                   name=="psd_rank4" ? IndPSD(rank=4) : name=="psd" ? IndPSD() : TracePSD(T(0.5))
            previous=x .+ T(1e-5).*randn(rng,T,size(x)); scratch=similar(x)
            # The warm call always starts from a nearby point, not its own answer.
            prep!() = prox!(scratch,f,previous,T(0.7))
            # prepare reads `previous`; the two cases share the reference prox(full, x)
            r = addprox!(cases,"partial",name,f,x;prepare=prep!,profile=T==Float64 && p==512,reference_function=full,
                         inputs=(; previous))
            addprox!(cases,"partial",name*"_full",full,x;value=false,reference=r)
            makef = name=="nuclear" ? () -> NuclearNorm(T(0.5);partial=true) :
                    name=="rank4" ? () -> IndBallRank(4;partial=true) :
                    name=="psd_rank4" ? () -> IndPSD(rank=4,partial=true) :
                    name=="psd" ? () -> IndPSD(partial=true) : () -> TracePSD(T(0.5);partial=true)
            addcold!(cases,"partial",name,makef,x)
        end
        if !quick && p == 128
            # A high-rank input exercises the full-decomposition fallback.
            fallback_input = randn(rng,T,64,32)
            fallback_function = NuclearNorm(T(0.01);partial=true)
            fallback_scratch = similar(fallback_input)
            addprox!(cases,"partial","high_rank_fallback",fallback_function,fallback_input;
                     prepare=() -> prox!(fallback_scratch,fallback_function,fallback_input,T(0.7)),
                     reference_function=NuclearNorm(T(0.01)))
        end
    end
end

function constrained_cases!(cases,rep; quick=false)
    for T in (quick ? (Float64,) : (Float32,Float64)), n in (quick ? (256,) : (256,100_000))
        rng=scenario_rng(rep,"constrained/$T/$n"); x=randn(rng,T,n)
        fs=(Constrained(NormL1(T(0.3)),IndBox(T(-1),T(1))),
            Constrained(SCAD(T(0.3)),IndBox(T(-1),T(1))),
            Constrained(NormL1(T(0.3)),IndHyperplane(one(T),zero(T))),
            Constrained(IndBox(zero(T),one(T)),IndHyperplane(one(T),T(n÷2))),
            Constrained(ElasticNet(T(0.3),T(0.4)),IndSimplex(T(n/10))),
            Constrained(NormL1(T(0.3)),IndBallL2(one(T))),
            Constrained(NormL1(T(0.3)),IndSphereL2(one(T))))
        for (name,f) in zip(("l1_box","scad_box","zero_sum_lasso","capped_simplex","simplex_elastic_net","l1_ball","l1_sphere"),fs)
            previous=x .+ T(0.001).*randn(rng,T,n); scratch=similar(x)
            prep!() = prox!(scratch,f,previous,T(0.7))
            addprox!(cases,"constrained",name,f,x;prepare=prep!,profile=T==Float64 && n==100_000,inputs=(; previous))
            # Reset the multiplier, retaining allocated workspace, before a cold call.
            if name in ("zero_sum_lasso", "capped_simplex", "simplex_elastic_net")
                cold!() = (f.last[] = T(NaN); nothing)
                addprox!(cases,"constrained",name*"_cold",f,x;prepare=cold!)
            end
        end
        for (name,f) in (("scaled",T(2)*NormL1(T(0.3))),
                         ("translated",Translate(NormL1(T(0.3)),ones(T,n))),
                         ("tilted",Tilt(NormL1(T(0.3)),fill(T(0.1),n))),
                         ("conjugate",Conjugate(NormL2(T(0.3)))),
                         ("regularize",Regularize(NormL1(T(0.3)),T(0.4))),
                         ("precompose",Precompose(NormL1(T(0.3)),T(-2))),
                         ("envelope",MoreauEnvelope(NormL1(T(0.3)),T(0.5))))
            addprox!(cases,"compositions",name,f,x;profile=T==Float64 && n==100_000 && name=="envelope")
        end
        # Dykstra has an independent specialized reference for this intersection.
        if n == 256
            C=IndIntersection(IndNonnegative(),IndHyperplane(one(T),T(10));warmstart=false)
            out=similar(x)
            step=T(0.7)
            ref_dykstra=prox(IndSimplex(T(10)),copy(x))
            check!() = (prox!(out,C,x,step);C.converged || error("Dykstra failed to converge");close_enough(out,ref_dykstra))
            addcase!(cases,"intersection","simplex_cold_corrections",()->prox!(out,C,x,step),check!;
                     inputs=(; x),T,shape=string(n),profile=T==Float64,work=n)
        end
    end
end

function tensor_cases!(cases,rep; quick=false)
    for T in (quick ? (Float64,) : (Float32,Float64)), p in (quick ? (8,) : (8,48))
        rng=scenario_rng(rep,"tensors/$T/$p"); X=randn(rng,T,p,p,4)
        X=(X+permutedims(X,(2,1,3)))./2
        for (name,f) in (("fiber_l2",FiberSum(NormL2(T(0.3)))),
                         ("fiber_tv",FiberSum(TotalVariation1D(T(0.3)))),
                         ("fiber_symmetric_l2",FiberSum(NormL2(T(0.3));symmetric=true,diagonal=nothing)),
                         ("slice_l1",SliceSum([NormL1(T(0.3)) for _ in 1:4])))
            addprox!(cases,"tensors",name,f,X;profile=T==Float64 && p==48)
        end
    end
end

# The sparse operand of a product case, drawn from its own subkey; args are (n, density) or
# (m, n, density).
product_operand(rep, key, ::Type{T}, args...) where {T} = sprandn_fixed(scenario_rng(rep, key), T, args...)

# The coefficient matrix of a switch_matrix case: the operand S and a SparseIterate x with the
# same entries, stored in a random order drawn from the same subkey after S.
function switch_operand(rep, ::Type{T}, m, density; p=1024, q=4) where {T}
    rng = scenario_rng(rep, "products/switch/$T/$m/$density/x")
    S = sprandn_fixed(rng, T, p, q, density)
    I, J, V = findnz(S)
    x = SparseIterate(T, p, q)
    for k in randperm(rng, length(I)); x[I[k], J[k]] = V[k]; end
    x, S
end

# Sparse operands hold exactly fixed_count(N, density) nonzeros, so `work` is the same in
# every repetition; references are built from the operand itself.
function product_cases!(cases,rep; quick=false)
    for T in (quick ? (Float64,) : (Float32,Float64)), p in (quick ? (256,) : (256,1024)), density in (quick ? (0.01,) : (0.01,0.2)), triangle in (quick ? (:U,) : (:U,:L))
        rng=scenario_rng(rep,"products/$T/$p/$density/$triangle")
        A=Symmetric(randn(rng,T,p,p),triangle)
        S=product_operand(rep,"products/$T/$p/$density/$triangle/xs",T,p,density)
        xs=SparseIterate(S)
        xd=Array(S)
        groups=[j:min(j+7,p) for j in 1:8:p]
        xa=AtomIterate(xd,GroupNormL2(T(0.3),groups))
        for (layout,x) in (("sparse",xs),("atom",xa))
            y=zeros(T,p); ref=A*xd
            check!()=(mul!(y,A,x);close_enough(y,ref))
            addcase!(cases,"products","symmetric_$triangle",()->mul!(y,A,x),check!;
                     inputs=(; A, x),T,shape=string(p),layout="$(layout)_$density",evals=10,
                     profile=T==Float64 && p==1024 && density==0.01 && triangle==:U,work=nnz(xs))
        end
    end
    n = 512
    for T in (quick ? (Float64,) : (Float32,Float64)), p in (quick ? (256,) : (256,1024)), density in (quick ? (0.01,) : (0.01,0.2))
        key="products/strided/$T/$p/$density"
        rng=scenario_rng(rep,key)
        A=randn(rng,T,n,p)
        Sxs=product_operand(rep,key*"/xs",T,p,density)
        Sxm=product_operand(rep,key*"/xm",T,p,4,density)
        Sxt=product_operand(rep,key*"/xt",T,n,density)
        xs=SparseIterate(Sxs)
        xm=SparseIterate(Sxm)
        groups=[j:min(j+7,p) for j in 1:8:p]
        xa=AtomIterate(Array(Sxs),GroupNormL2(T(0.3),groups))
        xt=SparseIterate(Sxt)
        for (name,B,x,S) in (("strided_vector",A,xs,Sxs),("strided_matrix",A,xm,Sxm),
                             ("strided_atom",A,xa,Sxs),("transposed_vector",transpose(A),xt,Sxt))
            shape="$(n)x$p" * (x === xm ? "x4" : "")
            layout="$(x isa AtomIterate ? "atom" : "sparse")_$density"
            xd=Array(S)
            out=x isa AbstractVector ? zeros(T,size(B,1)) : zeros(T,size(B,1),size(x,2))
            reference=B*xd
            work=size(B,1)*count(!iszero,xd)
            check!()=(mul!(out,B,x);close_enough(out,reference))
            addcase!(cases,"products",name*"/mul!",()->mul!(out,B,x),check!;
                     inputs=(; B, x),T,shape,layout,evals=10,work)
            original=fill(T(0.25),size(out))
            α,β=T(1.5),T(0.5)
            prepare!()=copyto!(out,original)
            expected_scaled=mul!(copy(original),B,xd,α,β)
            function checkscaled!()
                prepare!(); mul!(out,B,x,α,β)
                close_enough(out,expected_scaled)
            end
            addcase!(cases,"products",name*"/mul!_scaled",()->mul!(out,B,x,α,β),checkscaled!;
                     inputs=(; B, x, original),T,shape,layout,prepare=prepare!,evals=1,work)
        end
    end
    # Matrix coefficients with entries stored in random order, at densities on both
    # sides of the cutoff of the five-argument product (1/32), for short and long columns.
    quick && return
    for T in (Float32,Float64), m in (64,2048), density in (0.02,0.05,0.1)
        rng=scenario_rng(rep,"products/switch/$T/$m/$density")
        p,q=1024,4
        A=randn(rng,T,m,p)
        x,S=switch_operand(rep,T,m,density;p,q)
        xd=Array(S)
        out=zeros(T,m,q)
        reference=A*xd
        work=m*count(!iszero,xd)
        check!()=(mul!(out,A,x);close_enough(out,reference))
        addcase!(cases,"products","switch_matrix/mul!",()->mul!(out,A,x),check!;
                 inputs=(; A, x),T,shape="$(m)x$(p)x$q",layout="random_$density",evals=10,work)
    end
end

# SymmetricSparseIterate owner cases (issue #391, Phase 0; acceptance guide section 4).
# They use only calls that exist both in 0.5.x and in 0.6 (X[i, j], activecoordinates,
# prox_coordinate! with a lower-triangle coordinate ID, mul! and Matrix on the matrix
# view when there is one), so one harness run on each revision gives the before/after
# comparison. No case here depends on the owner being an AbstractMatrix.
owner_matrix(X) = isdefined(ProximalBase, :matrixview) ? ProximalBase.matrixview(X) : X

function owner_fixture(rng, T, p, s)
    q = p * (p + 1) ÷ 2
    ids = randperm(rng, q)[1:s]               # distinct lower-triangle coordinate IDs
    vals = T.(randn(rng, s)) .+ T(2)          # bounded away from zero
    X = SymmetricSparseIterate(T, p)
    for (k, v) in zip(ids, vals)
        r, c = ProximalBase.tril_cartesian(p, k)
        X[r, c] = v
    end
    I = Int[]; J = Int[]; V = T[]
    for (k, v) in zip(ids, vals)
        r, c = ProximalBase.tril_cartesian(p, k)
        push!(I, r); push!(J, c); push!(V, v)
        r == c || (push!(I, c); push!(J, r); push!(V, v))
    end
    (; X, ids, vals, S=sparse(I, J, V, p, p))
end

function owner_cases!(cases, rep; quick=false)
    T = Float64
    # (p, s): s = 100 with p growing, then p = 1000 with s growing
    grid = quick ? [(500, 100), (1000, 100), (1000, 1000)] :
           [(p, 100) for p in (500, 1000, 2000, 4000)] ∪ [(1000, s) for s in (10^3, 10^4, 10^5)]
    npos = 1024
    for (p, s) in grid
        rng = scenario_rng(rep, "owner/$T/$p/$s")
        fx = owner_fixture(rng, T, p, s)
        X, S = fx.X, fx.S
        shape = "p$(p)_s$(s)"
        q = p * (p + 1) ÷ 2
        active = [ProximalBase.tril_cartesian(p, fx.ids[1 + (i - 1) % s]) for i in 1:npos]   # cycles when s < npos
        activeset = Set(fx.ids)
        # distinct, initially inactive coordinate IDs (never repeated)
        absent_ids = Int[]; seen = Set{Int}()
        while length(absent_ids) < npos
            k = rand(rng, 1:q)
            (k in activeset || k in seen) && continue
            push!(seen, k); push!(absent_ids, k)
        end
        absent = [ProximalBase.tril_cartesian(p, k) for k in absent_ids]
        # four read cases of nread reads each: active and absent coordinates, each read through
        # X[r, c] with r >= c and through X[c, r]; each is checked against the CSC reference
        nread = 256
        for (kind, base) in (("active", active), ("absent", absent)), (side, flip) in (("lower", false), ("upper", true))
            pos = [flip ? (c, r) : (r, c) for (r, c) in base[1:nread]]
            expected_read = sum(S[r, c] for (r, c) in pos)
            read_op() = (acc = zero(T); @inbounds for (r, c) in pos; acc += X[r, c]; end; acc)
            check_read!() = isapprox(read_op(), expected_read; rtol=1e-12) || error("owner reads differ from the CSC reference")
            addcase!(cases, "owner", "read_X[i,j]", read_op, check_read!; inputs=(; X, pos), T, shape, layout="$(kind)_$(side)", work=nread)
        end
        # writes to already active coordinates, with a changing value
        wpos = active
        Xw = copy(fx.X)                      # this case's own owner
        wstate = Ref(zero(T))
        write_op() = (val = wstate[] += one(T); @inbounds for (r, c) in wpos; Xw[r, c] = val; end; nothing)
        function check_write!()
            write_op()
            val = wstate[]
            all(Xw[r, c] == val for (r, c) in wpos) || error("owner writes were not stored")
            length(collect(activecoordinates(Xw))) == s || error("a write to an active coordinate changed the working set")
        end
        # Xw is not an input: every call writes a new value into it
        addcase!(cases, "owner", "write_X[i,j]_active", write_op, check_write!; inputs=(; wpos), T, shape, layout="active", work=length(wpos))
        # first activation: always an absent coordinate; restored outside the timed region
        Xa = copy(fx.X)
        # a batch of 256 distinct, initially inactive coordinates per call: one activation is below the timer floor
        batch_ids = absent_ids[1:256]
        batch = absent[1:256]
        activate_op() = (@inbounds for (r, c) in batch; Xa[r, c] = one(T); end; nothing)
        function restore!()               # outside the timed region
            for (r, c) in batch; Xa[r, c] = zero(T); end
            dropzeros!(Xa)
            any(k -> isactive(Xa, k), batch_ids) && error("a batch coordinate is still active: not a first activation")
            nothing
        end
        function check_activate!()
            restore!()
            length(unique(batch_ids)) == length(batch_ids) || error("batch coordinates are not distinct")
            activate_op()
            all(k -> isactive(Xa, k), batch_ids) || error("activation did not add every coordinate")
            length(collect(activecoordinates(Xa))) == s + length(batch_ids) || error("unexpected working set")
            restore!()
            length(collect(activecoordinates(Xa))) == s || error("restore did not recover the working set")
        end
        addcase!(cases, "owner", "first_activation", activate_op, check_activate!; inputs=(; batch, batch_ids), T, shape, layout="absent_distinct_batch",
                 phase="first_activation", prepare=restore!, evals=1, zero_alloc=true, work=length(batch))
        # sweeps over the active coordinates
        Xs = copy(fx.X)
        sweep_ids() = (acc = 0; for k in activecoordinates(Xs); acc += k; end; acc)
        expected_sum = sum(fx.ids)
        sorted_owner_ids = sort(fx.ids)
        addcase!(cases, "owner", "sweep_activecoordinates", sweep_ids,
                 function ()
                     sweep_ids() == expected_sum || error("sweep visited the wrong coordinates")
                     sort!(collect(activecoordinates(Xs))) == sorted_owner_ids || error("the active coordinates are not the fixture's (a sum can agree)")
                 end;
                 inputs=(; Xs), T, shape, layout="trivial_update", work=s)
        f = NormL1(T(0.1))
        step = T(0.5)
        prox_sweep() = (for k in activecoordinates(Xs); prox_coordinate!(Xs, f, k, step); end; nothing)
        vstate = copy(fx.vals)
        prep_sweep!() = (for (k, v) in zip(fx.ids, fx.vals); r, c = ProximalBase.tril_cartesian(p, k); Xs[r, c] = v; end; nothing)
        function check_sweep!()
            prep_sweep!(); prox_sweep()
            ok = all(zip(fx.ids, fx.vals)) do (k, v)
                r, c = ProximalBase.tril_cartesian(p, k)
                isapprox(Xs[r, c], sign(v) * max(abs(v) - step * T(0.1), zero(T)); atol=1e-12)
            end
            ok || error("coordinate sweep gave the wrong soft threshold")
            prep_sweep!()
        end
        addcase!(cases, "owner", "sweep_prox_coordinate!", prox_sweep, check_sweep!; inputs=(; fx_ids=fx.ids, fx_vals=fx.vals),
                 T, shape, layout="NormL1",
                 prepare=prep_sweep!, evals=1, work=s)
        # matrix product with a dense vector

        v = randn(rng, T, p); y = zeros(T, p); ref = S * v
        A = owner_matrix(X)
        mul_op() = (mul!(y, A, v); nothing)
        check_mul!() = (mul_op(); isapprox(y, ref; rtol=1e-10, atol=1e-10) || error("owner product differs from the CSC product"))
        addcase!(cases, "owner", "mul!", mul_op, check_mul!; inputs=(; X, v), T, shape, layout="owner", evals=10, work=s)
        # baselines in the same session
        ycsc = zeros(T, p)
        addcase!(cases, "owner", "mul!", () -> (mul!(ycsc, S, v); nothing),
                 () -> (mul!(ycsc, S, v); isapprox(ycsc, ref; rtol=1e-10, atol=1e-10) || error("CSC product mismatch"));
                 inputs=(; S, v), T, shape, layout="csc_baseline", evals=10, work=s)
        if p <= 2000
            D = Symmetric(Matrix(S), :L); yd = zeros(T, p)
            addcase!(cases, "owner", "mul!", () -> (mul!(yd, D, v); nothing),
                     () -> (mul!(yd, D, v); isapprox(yd, ref; rtol=1e-10, atol=1e-10) || error("dense product mismatch"));
                     inputs=(; D, v), T, shape, layout="dense_symmetric_baseline", evals=10, work=p * p)
        end
        # conversion to a dense matrix
        if p <= 2000 || s == 100
            addcase!(cases, "owner", "Matrix", () -> (Matrix(A); nothing),
                     () -> (M = Matrix(A); sparse(M) == S || error("Matrix(owner) differs from the CSC reference"));
                     inputs=(; X), T, shape, layout="owner", phase="allocating", zero_alloc=false, evals=1, work=p * p)
        end
    end
    # construction (empty owner), quadratic in p
    for p in (quick ? (500, 1000) : (500, 1000, 2000, 4000))
        addcase!(cases, "owner", "construct", () -> (SymmetricSparseIterate(T, p); nothing),
                 function ()
                     X = SymmetricSparseIterate(T, p); q = p * (p + 1) ÷ 2
                     size(X, 1) == p || error("wrong size")
                     # retained memory: one T array and two Int arrays of length q, plus headers
                     abs(Base.summarysize(X) - (sizeof(T) + 2 * sizeof(Int)) * q) <= 1024 || error("unexpected retained size")
                 end;
                 # the operation reads only scalars
                 inputs=(;), T, shape="p$p", layout="empty", phase="construct", zero_alloc=false, evals=1, work=p * (p + 1) ÷ 2)
    end
    owner_extra_cases!(cases, rep; quick)
end

# Owner cases added by the iterate-family baseline (issue #420), appended to the owner group: the
# product with stored zeros (FD-9), the three clears (FP-7) and the momentum loops (FD-8). They use
# the (p, s) grid and the shapes of owner_cases!, helpers defined below, and random inputs of their
# own (keys "owner_extra/..."), so every case above keeps its ID and its inputs; only the owner_z50
# product takes the coordinates of the owner row of its shape. Only calls that exist in 0.5.x and 0.6.

# an owner with `vals` stored at the coordinates `ids`, in this slot order; an exact zero is stored
# with activate!, as zeros are not stored by assignment
function build_owner(T, p, ids, vals)
    X = SymmetricSparseIterate(T, p)
    for (k, v) in zip(ids, vals)
        if iszero(v)
            activate!(X, k)
        else
            r, c = ProximalBase.tril_cartesian(p, k)
            X[r, c] = v
        end
    end
    X
end

# the symmetric CSC matrix of the nonzero entries, both triangles: the reference of the products
function owner_csc(T, p, ids, vals)
    I = Int[]; J = Int[]; V = T[]
    for (k, v) in zip(ids, vals)
        iszero(v) && continue
        r, c = ProximalBase.tril_cartesian(p, k)
        push!(I, r); push!(J, c); push!(V, v)
        r == c || (push!(I, c); push!(J, r); push!(V, v))
    end
    sparse(I, J, V, p, p)
end

# the three clears of FP-7, as functions of the owner
clear_coordinates!(X) = (fill!(coordinates(X), 0); nothing)
clear_coordinates_dropzeros!(X) = (fill!(coordinates(X), 0); dropzeros!(X); nothing)
clear_owner!(X) = (fill!(X, 0); nothing)

function owner_extra_cases!(cases, rep; quick=false)
    T = Float64
    # (p, s): the grid of owner_cases!
    grid = quick ? [(500, 100), (1000, 100), (1000, 1000)] :
           [(p, 100) for p in (500, 1000, 2000, 4000)] ∪ [(1000, s) for s in (10^3, 10^4, 10^5)]
    for (p, s) in grid
        key = "owner_extra/$T/$p/$s"
        shape = "p$(p)_s$(s)"
        q = p * (p + 1) ÷ 2
        fx = owner_fixture(scenario_rng(rep, key * "/fixture"), T, p, s)
        X = fx.X
        snap = copy(X)
        coords = [ProximalBase.tril_cartesian(p, k) for k in fx.ids]
        restore_state!() = (copyto!(X, snap); nothing)
        guarded(check) = restored(restore_state!, check)

        # FD-9: a product with 50% of the stored coordinates exact zeros, at random slots. The coordinates
        # and values are those of the owner row of this shape (the first draw of its generator), so that
        # the pair of rows differs only in the zeros; the slots of the zeros have a key of their own. The
        # reference is the CSC matrix of the nonzero entries; the output starts from a sentinel, so that
        # a call that does nothing cannot pass.
        ids50, base_vals = let base = owner_fixture(scenario_rng(rep, "owner/$T/$p/$s"), T, p, s)
            base.ids, base.vals
        end
        slots = zero_slots(scenario_rng(rep, key * "/zeros"), s, 0.5)
        vals50 = with_zeros(base_vals, slots)
        X50 = build_owner(T, p, ids50, vals50)
        v = randn(scenario_rng(rep, key * "/v"), T, p)
        y = zeros(T, p)
        ref = owner_csc(T, p, ids50, vals50) * v
        A50 = owner_matrix(X50)
        function check_mul50()
            (nactive(X50) == s && count(iszero, activevalues(X50)) == length(slots)) || error("the explicit zeros are not stored")
            fill!(y, T(1000))
            mul!(y, A50, v)
            isapprox(y, ref; rtol=1e-10, atol=1e-10) || error("owner product with stored zeros differs from the CSC reference")
        end
        addcase!(cases, "owner", "mul!", () -> (mul!(y, A50, v); nothing), check_mul50;
                 inputs=(; X50, v), T, shape, layout="owner_z50", evals=10, zero_alloc=true, work=s)

        # FP-7: the clears. The clear through coordinates(X) writes a value into each of the q
        # coordinates and keeps the working set; fill!(X, 0) is O(s) and clears the working set. Only the
        # values are checked, as the working set differs between the clears and between 0.5.x and 0.6.
        function check_clear(clear!)
            collect(activecoordinates(X)) == fx.ids || error("the owner is not the fixture")
            all(X[r, c] == v for ((r, c), v) in zip(coords, fx.vals)) || error("the owner does not hold the fixture")
            clear!(X)
            all(iszero, activevalues(X)) || error("a stored value is not zero after the clear")
            all(iszero(X[r, c]) for (r, c) in coords) || error("a fixture coordinate is not zero after the clear")
        end
        addcase!(cases, "owner", "fill!_coordinates_zero", () -> clear_coordinates!(X),
                 guarded(() -> check_clear(clear_coordinates!));
                 inputs=(; snap), T, shape, layout="owner", prepare=restore_state!, zero_alloc=true, work=q)
        addcase!(cases, "owner", "fill!_coordinates_zero_dropzeros", () -> clear_coordinates_dropzeros!(X),
                 guarded(() -> check_clear(clear_coordinates_dropzeros!));
                 inputs=(; snap), T, shape, layout="owner", prepare=restore_state!, zero_alloc=true, work=q)
        addcase!(cases, "owner", "fill!_zero", () -> clear_owner!(X), guarded(() -> check_clear(clear_owner!));
                 inputs=(; snap), T, shape, layout="owner", prepare=restore_state!, zero_alloc=false, work=s)

        # FD-8: the momentum value loops over the stored values, through the accessor and through the
        # private array; the reference is the fixture's values in slot order
        ybuf = randn(scenario_rng(rep, key * "/momentum_y"), T, s)
        xcur = randn(scenario_rng(rep, key * "/momentum_cur"), T, s)
        xprev = randn(scenario_rng(rep, key * "/momentum_prev"), T, s)
        buf = zeros(T, s)
        rd_total, rd_magnitude = restart_reference(fx.vals, ybuf, xcur, xprev)
        for (layout, getvals) in (("accessors", activevalues), ("private_fields", stored_values_by_field(X)))
            w = copy(fx.vals)
            values_op, setvalues_op, dot_op = momentum_ops(getvals, X, buf, w, ybuf, xcur, xprev, s)
            function check_values()
                collect(activecoordinates(X)) == fx.ids || error("the owner is not in the fixture's slot order")
                fill!(buf, zero(T))
                values_op()
                buf == fx.vals || error("_values! differs from the fixture's values")
            end
            function check_setvalues()
                fill!(activevalues(X), zero(T))
                setvalues_op()
                activevalues(X) == fx.vals || error("_setvalues! differs from the fixture's values")
                all(X[r, c] == v for ((r, c), v) in zip(coords, fx.vals)) || error("_setvalues! did not write the fixture's values")
            end
            check_restartdot() = close_scalar(dot_op(), rd_total, rd_magnitude, T)
            for (name, op, check, inputs) in (("_values!", values_op, check_values, (; X)),
                                              ("_setvalues!", setvalues_op, check_setvalues, (; X, w)),
                                              ("_restartdot", dot_op, check_restartdot, (; X, ybuf, xcur, xprev)))
                addcase!(cases, "owner", name, op, guarded(check); inputs, T, shape, layout, zero_alloc=true, work=s)
            end
        end
    end
end

# The BigFloat solution of the ridge problem on the hyperplane a'z = b with per-entry steps, in the form
# that the check reads: a pair of Float64 per entry (hi + lo is within 2⁻¹⁰⁶ of the BigFloat value, far
# below the tolerance) and the scale of the tolerance. Computed when the case is built.
# Entry by entry (two passes), so that no BigFloat array is held.
function hyperplane_reference(::Type{T}, λ, x, a, b, cen, steps) where {T}
    function pq(i)
        s = big(T(steps[i]))                       # the steps as the search converts them
        t = s * big(λ)
        c = cen === nothing ? big(0) : big(cen[i])
        ((big(x[i]) + t * c) / (1 + t), s * big(a[i]) / (1 + t))
    end
    ap, aq = big(0), big(0)
    for i in eachindex(x)
        p, q = pq(i)
        ap += big(a[i]) * p
        aq += big(a[i]) * q
    end
    μ = (ap - big(b)) / aq
    hi, lo = zeros(Float64, length(x)), zeros(Float64, length(x))
    rmax, xmax = big(0), big(0)
    for i in eachindex(x)
        p, q = pq(i)
        r = p - μ * q
        hi[i] = Float64(r)
        lo[i] = Float64(r - big(hi[i]))
        rmax = max(rmax, abs(r))
        xmax = max(xmax, abs(big(x[i])))
    end
    (; hi, lo, scale = rmax + xmax)
end

# Ridge: ordinary calls in the three precision mixes, a coordinate loop, and the
# recovery path (parameters or centers outside the model's range) as separate cases.
function ridge_cases!(cases, rep; quick=false)
    pow2(T, k) = ldexp(one(T), k)
    sizes = quick ? (8, 1000) : (8, 1000, 100_000)
    # (label, data type, level, step, center type or nothing)
    mixes = (("all32", Float32, Float32(0.7), Float32(0.3), Float32),
             ("all64", Float64, 0.7, 0.3, Float64),
             ("data32_params64", Float32, 0.7, 0.3, Float64),
             ("data32_params64_center32", Float32, 0.7, 0.3, Float32))
    for n in sizes, (label, T, λ, γ, C) in mixes, centered in (false, true)
        rng = scenario_rng(rep, "ridge/$label/$centered/$n")
        x = randn(rng, T, n)
        c = centered ? randn(rng, C, n) : nothing
        f = SqrNormL2(λ, c)
        name = string(label, centered ? "/centered" : "/uncentered")
        shape = string(n)
        z = similar(x)
        g = similar(x)
        cf = c === nothing ? zeros(T, n) : T.(c)
        refprox = [(x[i] + big(γ) * big(λ) * big(cf[i])) / (1 + big(γ) * big(λ)) for i in 1:n]
        refgrad = [big(λ) * (big(x[i]) - big(cf[i])) for i in 1:n]
        scale = [max(abs(big(x[i])), abs(big(cf[i])), big(eps(T))) for i in 1:n]
        close_ref(y, r) = all(i -> abs(big(y[i]) - r[i]) <= 4 * eps(T) * (scale[i] * (iszero(λ) ? 1 : max(1, big(λ)))), eachindex(y)) ||
            error("output differs from the reference")
        # the center is held by the function (nothing when uncentered)
        addcase!(cases, "ridge", name * "/prox!", () -> prox!(z, f, x, γ),
                 () -> (prox!(z, f, x, γ); close_ref(z, refprox)); inputs=(; x, c), T, shape, work=n)
        # independent entries, as a vectorizable sweep, and a dependent chain, as in
        # coordinate descent (each update feeds the next argument)
        addcase!(cases, "ridge", name * "/prox_entry_loop",
                 () -> begin @inbounds for i in eachindex(x) z[i] = prox_entry(f, x[i], i, γ) end end,
                 () -> (for i in eachindex(x); z[i] = prox_entry(f, x[i], i, γ); end; close_ref(z, refprox));
                 inputs=(; x, c), T, shape, layout="coordinate", work=n)
        chain() = (v = zero(T); @inbounds for i in eachindex(x) v = prox_entry(f, x[i] + T(0.5) * v, i, γ) end; v)
        addcase!(cases, "ridge", name * "/prox_entry_chain", chain,
                 () -> isfinite(chain()) || error("nonfinite chain"); inputs=(; x, c), T, shape, layout="coordinate", work=n)
        addcase!(cases, "ridge", name * "/value_and_prox!", () -> value_and_prox!(z, f, x, γ),
                 () -> (v = value_and_prox!(z, f, x, γ); close_ref(z, refprox);
                        isapprox(v, f(z); rtol=100 * eps(T), atol=100 * eps(T)) || error("value differs from f(prox)"));
                 inputs=(; x, c), T, shape, work=n)
        addcase!(cases, "ridge", name * "/gradient!", () -> gradient!(g, f, x),
                 () -> (gradient!(g, f, x); close_ref(g, refgrad)); inputs=(; x, c), T, shape, work=n)
        ref_fx = deepcopy(f)(copy(x))
        addcase!(cases, "ridge", name * "/value_and_gradient!", () -> value_and_gradient!(g, f, x),
                 () -> (v = value_and_gradient!(g, f, x); close_ref(g, refgrad);
                        isapprox(v, ref_fx; rtol=100 * eps(T), atol=100 * eps(T)) || error("value mismatch"));
                 inputs=(; x, c), T, shape, work=n)
        refvalue = big(λ) / 2 * sum(abs2, big.(x) .- big.(cf))
        addcase!(cases, "ridge", name * "/value", () -> f(x),
                 () -> isapprox(big(f(x)), refvalue; rtol=1000 * eps(T)) || error("value differs from the reference");
                 inputs=(; x, c), T, shape, work=n)
    end
    # Recovery path: separate from the ordinary calls above.
    let
      n = quick ? 1000 : 10_000
      rng = scenario_rng(rep, "ridge/recovery/$n")
      x = randn(rng, Float32, n)
      z = similar(x); g = similar(x)
      big_level = pow2(Float64, 160)
      tiny_step = pow2(Float64, -160)
      c64 = rand(rng, Float64, n)
      far = fill(pow2(Float64, 200), n)
      f0 = SqrNormL2(big_level)
      fc = SqrNormL2(big_level, Float32.(c64))
      ff = SqrNormL2(pow2(Float64, -400), far)          # Float32 data, centers outside Float32
      finite(y) = all(isfinite, y) || error("nonfinite recovery output")
      half = x ./ 2
      addcase!(cases, "ridge", "recovery/coefficient_in_band_after_product/prox!", () -> prox!(z, f0, x, tiny_step),
               () -> (prox!(z, f0, x, tiny_step); all(i -> z[i] == half[i], 1:n) || error("unexpected prox")); inputs=(; x), T=Float32, layout="recovery", shape=string(n), work=n)
      addcase!(cases, "ridge", "recovery/coefficient_scaled/prox!", () -> prox!(z, f0, x, 1.0),
               () -> (prox!(z, f0, x, 1.0); finite(z)); inputs=(; x), T=Float32, layout="recovery", shape=string(n), work=n)
      addcase!(cases, "ridge", "recovery/coefficient_scaled_centered/prox!", () -> prox!(z, fc, x, 1.0),
               () -> (prox!(z, fc, x, 1.0); finite(z)); inputs=(; x), T=Float32, layout="recovery", shape=string(n), work=n)
      addcase!(cases, "ridge", "recovery/level_outside/gradient!", () -> gradient!(g, SqrNormL2(big_level), x .* pow2(Float32, -100)),
               () -> (gradient!(g, SqrNormL2(big_level), x .* pow2(Float32, -100)); finite(g)); inputs=(; x), T=Float32, layout="recovery", shape=string(n), work=n, zero_alloc=false)
      addcase!(cases, "ridge", "recovery/center_outside/prox!", () -> prox!(z, ff, x, 1.0),
               () -> (prox!(z, ff, x, 1.0); finite(z)); inputs=(; x, far), T=Float32, layout="recovery", shape=string(n), work=n)
      addcase!(cases, "ridge", "recovery/center_outside/value", () -> ff(x),
               () -> isfinite(ff(x)) || error("nonfinite value"); inputs=(; x, far), T=Float32, layout="recovery", shape=string(n), work=n, zero_alloc=false)
    end
    # Constrained ridge on a hyperplane with per-entry steps (extended suite only): repeated
    # calls start from the last multiplier; "cold" calls start from zero; the scalar step is
    # the control. Validated against the closed-form multiplier, outside the timing.
    quick && return
    for n in sizes, (label, T, λ, F, C) in (("all32", Float32, Float32(0.7), Float32, Float32),
                                            ("all64", Float64, 0.7, Float64, Float64),
                                            ("data32_params64", Float32, 0.7, Float64, Float64)),
        centered in (false, true)
        rng = scenario_rng(rep, "ridge/constrained/$label/$centered/$n")
        x = randn(rng, T, n)
        a = rand(rng, T, n) .+ T(0.1)
        b = T(n / 8)
        γs = F.(0.1 .+ rand(rng, n))
        γ = F(0.3)
        cen = centered ? randn(rng, C, n) : nothing
        c = Constrained(SqrNormL2(λ, cen), IndHyperplane(a, b))
        z = similar(x)
        name = string("constrained/", label, centered ? "/centered" : "/uncentered")
        function check_hyperplane(steps, ref)
            prox!(z, c, x, steps)
            r = big.(ref.hi) .+ big.(ref.lo)
            maximum(abs.(big.(z) .- r)) <= 64 * eps(T) * ref.scale ||
                error("constrained output differs from the reference")
        end
        # the vector steps and the cold calls share one reference; the scalar step is checked through
        # the equal vector of steps, as before
        γfill = fill(γ, n)
        ref_vector = hyperplane_reference(T, λ, x, a, b, cen, γs)
        ref_scalar = hyperplane_reference(T, λ, x, a, b, cen, γfill)
        addcase!(cases, "ridge", name * "/vector_steps/prox!", () -> prox!(z, c, x, γs),
                 () -> check_hyperplane(γs, ref_vector); inputs=(; x, γs, a, cen), T, shape=string(n), layout="constrained", work=n)
        addcase!(cases, "ridge", name * "/vector_steps_cold/prox!", () -> prox!(z, c, x, γs),
                 () -> check_hyperplane(γs, ref_vector); inputs=(; x, γs, a, cen), T, shape=string(n), layout="constrained", work=n,
                 prepare=() -> (c.last[] = NaN; c.slope[] = NaN))
        addcase!(cases, "ridge", name * "/scalar_step/prox!", () -> prox!(z, c, x, γ),
                 () -> check_hyperplane(γfill, ref_scalar); inputs=(; x, a, cen), T, shape=string(n), layout="constrained", work=n)
    end
    # one step below the band: the search keeps the tested entries
    let
      n = 10_000
      rng = scenario_rng(rep, "ridge/constrained/recovery/$n")
      x = randn(rng, Float32, n)
      a = rand(rng, Float32, n) .+ 0.1f0
      γs = 0.1f0 .+ rand(rng, Float32, n)
      γs[1] = ldexp(1f0, -130)
      c = Constrained(SqrNormL2(0.7f0, randn(rng, Float32, n)), IndHyperplane(a, Float32(n / 8)))
      z = similar(x)
      addcase!(cases, "ridge", "recovery/constrained_step_outside/prox!", () -> prox!(z, c, x, γs),
               () -> (prox!(z, c, x, γs); all(isfinite, z) || error("nonfinite constrained output"));
               inputs=(; x, γs, a), T=Float32, layout="recovery", shape=string(n), work=n)
    end
end

# ---- helpers shared by the sparse and atom groups ----
#
# One working iterate and one immutable snapshot serve all the cases of a scenario; the cases run
# one at a time. Every validation starts and ends by restoring the working iterate with a state copy
# (copyto! from the snapshot, which is a state copy on 0.5.x and on 0.6), so that it does not matter
# what the case that ran before it left behind. A case that changes state restores it in `prepare`.

# k values ±(1 + u), u ~ U[0, 1), with a random sign: bounded away from zero.
function fixture_values(rng, T, k)
    T[(rand(rng, Bool) ? one(T) : -one(T)) * T(1 + rand(rng)) for _ in 1:k]
end

# k items of `src` in order, starting again from the first when `src` has fewer
cycled(src, k) = [src[1 + (j - 1) % length(src)] for j in 1:k]

# the values with those slots set to exact zeros
function with_zeros(vals, slots)
    v = copy(vals)
    v[slots] .= zero(eltype(v))
    v
end

# the slots of a fraction of the stored values, at random places in storage order
zero_slots(rng, s, fraction) = randperm(rng, s)[1:round(Int, fraction * s)]

# an iterate with `vals` stored at the linear indices `ids`, in this slot order; an exact zero is
# stored with activate!, as zeros are not stored by assignment
function build_iterate(T, dims, ids, vals)
    x = SparseIterate(T, dims...)
    for (i, v) in zip(ids, vals)
        iszero(v) ? activate!(x, i) : (x[i] = v)
    end
    x
end

# the dense array the iterate must equal: the reference of every check
function dense_array(T, dims, ids, vals)
    D = zeros(T, dims...)
    D[ids] = vals
    D
end

# a validation that starts and ends with the shared iterate restored
restored(restore!, check) = function ()
    restore!()
    check()
    restore!()
    nothing
end

# |a - b| <= tolerance(T) * scale, for sums whose rounding depends on the order of the terms
function close_scalar(a, b, scale, T)
    abs(a - b) <= tolerance(T) * scale || error("scalar differs from the dense reference: $a versus $b")
    nothing
end

# Version-dependent spelling is resolved once, when the case is built, with the 0.5.x spelling as the
# fallback, and handed to a function that builds the timed closures: they see a concrete function
# type, and nothing is tested inside the timed call. The catalog of case IDs does not depend on it.
private_nzval(x) = x.nzval
stored_values_by_field(x) = hasfield(typeof(x), :nzval) ? private_nzval : activevalues

# a repetition count for an operation of `units` steps, so that a call takes about 2 µs on the pinned source
repeats_for(units, target) = max(1, cld(target, units))

# the reference of the restart dot products: Σ (vals[j] - y[j]) (xcur[j] - xprev[j]) and the sum of the
# magnitudes of the terms, in this order; computed when the case is built
function restart_reference(vals, y, xcur, xprev)
    total, magnitude = zero(eltype(y)), zero(eltype(y))
    for j in eachindex(vals)
        term = (vals[j] - y[j]) * (xcur[j] - xprev[j])
        total += term
        magnitude += abs(term)
    end
    total, magnitude
end

# ---- the sparse group: SparseIterate vectors and matrices ----
#
# Fixtures: vectors (n, s) with s distinct random coordinates; matrices with exactly round(d·L)
# stored entries (so that a repetition never crosses the 1/32 switch of the matrix mul!); dense
# operands built once per (T, shape) and shared. Only calls that exist on 0.5.x and on 0.6 are used.

const SPARSE_DENSITIES = ((0.005, "0.5"), (0.03, "3"), (0.2, "20"))

# the loops of the cases are functions of their arguments, so that the timed closures are thin
function sum_reads(x, ids, ::Type{T}) where {T}
    acc = zero(T)
    @inbounds for k in ids
        acc += x[k]
    end
    acc
end
function write_values!(x, ids, vals)
    @inbounds for j in eachindex(ids)
        x[ids[j]] = vals[j]
    end
    nothing
end
function activate_all!(x, ids)
    @inbounds for k in ids
        activate!(x, k)
    end
    nothing
end
function count_active(x, ids)
    c = 0
    @inbounds for k in ids
        c += isactive(x, k)
    end
    c
end
# isactive against the stored membership and not against the values. On a private small iterate, a write of
# zero to an active coordinate keeps it active (0.5.x and 0.6 agree; a write of zero to an inactive one does
# nothing), so the nonzero count of the dense vector is smaller than the working set and an `isactive` that
# tests the value fails here. Run in the check of the isactive cases, outside the timed region.
function check_isactive_stored_zero(::Type{T}) where {T}
    members = collect(1:3:60)
    zeroed = members[1:2:end]
    xz = SparseIterate(T, 64)
    for i in members; xz[i] = one(T); end
    for i in zeroed; xz[i] = zero(T); end
    count(!iszero, Array(xz)) == length(members) - length(zeroed) || error("the stored-zero fixture is wrong")
    all(i -> isactive(xz, i) == (i in members), 1:64) || error("isactive differs from the stored membership (explicit zeros)")
    nactive(xz) == length(members) || error("explicit zeros changed the working set")
    nothing
end
function check_isactive_stored_zero_atom(::Type{T}) where {T}
    groups = [(3j - 2):(3j) for j in 1:8]
    members = [1, 3, 5, 8]
    zeroed = [1, 5]
    xz = AtomIterate(T, 24, groups)
    for k in members; xz[first(groups[k])] = one(T); end
    for k in zeroed, i in groups[k]; xz[i] = zero(T); end
    v = Vector(xz)
    count(k -> any(!iszero, v[groups[k]]), 1:8) == length(members) - length(zeroed) || error("the stored-zero fixture is wrong")
    all(k -> isactive(xz, k) == (k in members), 1:8) || error("isactive differs from the stored membership (explicit zeros)")
    nactive(xz) == length(members) || error("explicit zeros changed the working set")
    nothing
end
# The trivial update of the sweeps adds a zero: `x[i] = x[i]` alone is removed by the compiler (the
# store of a value just loaded; a 10⁴-entry sweep took 10 ns on Julia 1.13), and the sweep would
# measure nothing. Adding +0.0 is not an identity for -0.0, so the compiler keeps it; no stored value
# is -0.0, so the iterate is unchanged.
function full_sweep!(x)
    z = zero(eltype(x))
    @inbounds for i in 1:ncoordinates(x)
        x[i] = x[i] + z
    end
    nothing
end
# the sweep returns the number of entries it touched
function active_sweep!(x)
    z = zero(eltype(x))
    touched = 0
    @inbounds for k in activecoordinates(x)
        x[k] = x[k] + z
        touched += 1
    end
    touched
end
function sum_active_ids(x)
    acc = 0
    for k in activecoordinates(x)
        acc += k
    end
    acc
end
function sum_active_values(x, ::Type{T}) where {T}
    acc = zero(T)
    for v in activevalues(x)
        acc += v
    end
    acc
end
function coordinate_sweep!(x)
    c = coordinates(x)
    z = zero(eltype(c))
    @inbounds for k in 1:length(c)
        c[k] = c[k] + z
    end
    nothing
end
function index_sweep!(x)
    z = zero(eltype(x))
    @inbounds for k in 1:length(x)
        x[k] = x[k] + z
    end
    nothing
end
function rmul_repeat!(x, a, r)
    for _ in 1:r
        rmul!(x, a)
    end
    nothing
end
function prox_batch!(x, f, ids, us, γ)
    @inbounds for j in eachindex(ids)
        prox_coordinate!(x, f, ids[j], us[j], γ)
    end
    nothing
end

# the three value loops of the momentum history (acceptance guide §3.4), over the stored values
# through `getvals`: the accessor of 0.6, or the private array that the downstream code reads today
momentum_values!(getvals::G, buf, x, n) where {G} = copyto!(buf, 1, getvals(x), 1, n)
momentum_setvalues!(getvals::G, x, buf, n) where {G} = copyto!(getvals(x), 1, buf, 1, n)
function momentum_dot(getvals::G, x, y, xcur, xprev, n) where {G}
    av = getvals(x)
    d = zero(eltype(y))
    @inbounds for j in 1:n
        d += (av[j] - y[j]) * (xcur[j] - xprev[j])
    end
    d
end
# the function barrier: the closures are built where `getvals` has a concrete type
function momentum_ops(getvals::G, x, buf, w, y, xcur, xprev, n) where {G}
    (() -> momentum_values!(getvals, buf, x, n),
     () -> momentum_setvalues!(getvals, x, w, n),
     () -> momentum_dot(getvals, x, y, xcur, xprev, n))
end

function replace_cheap!(x, A)
    copyto!(dropzeros!(fill!(x, 0)), A)
    nothing
end
function replace_staged!(x, A)
    t = similar(x)
    for i in eachindex(t)
        t[i] = A[i]
    end
    copyto!(x, t)
    nothing
end

# FW-4: a state copy with a conversion, on dedicated iterates. The source holds integral values and is
# built from scratch, so that its whole `nzval` array (the stale tail included, which the 0.5.x dense
# branch converts too) converts to Int; it is never changed. Only the success path is measured.
function sparse_convert_cases!(cases, rep, dims, ids, key, shape)
    k = length(ids)
    rng = scenario_rng(rep, key * "/integral")
    iv = Float64[(rand(rng, Bool) ? 1 : -1) * rand(rng, 1:7) for _ in 1:k]
    src = build_iterate(Float64, dims, ids, iv)
    dsrc = dense_array(Float64, dims, ids, iv)
    # the branch of the copy kernel once source and destination hold the same entries (4(nnz + nnz) < length)
    branch = 4 * (2 * k) < prod(dims) ? "sparse_branch" : "dense_branch"
    for Td in (Float64, Float32, Int)
        dest = SparseIterate(Td, dims...)
        op = () -> (copyto!(dest, src); nothing)
        check = function ()
            fill!(dest, 0)
            for _ in 1:2        # the first call starts from an empty destination, the second from a full one
                copyto!(dest, src)
                Array(dest) == Td.(dsrc) || error("converted copy differs from the dense reference")
            end
            nothing
        end
        addcase!(cases, "sparse", "copyto!_convert", op, check; inputs=(; src), T=Float64, shape,
                 layout="to_$(Td)_$(branch)", zero_alloc=false, work=branch == "sparse_branch" ? k : prod(dims))
    end
end

# ---- the cheap ID enumerators and the grid of the sparse group ----
#
# What `sparse_cases!` registers, spelled from cheap fields only (type, sizes, labels, the arithmetic for the
# batch count, the repetitions of `rmul!` and the branch of the conversions), as (operation name, ID) pairs: the
# builders skip a scenario or function none of whose IDs the selector matches, and the tests use the same lists.
# A missing ID would be a false exclusion, so the tests compare them with the registered IDs.

const SPARSE_POOL = 4096                      # absent coordinates kept per vector scenario
const SPARSE_ZLABELS = ("z0", "z50")
const SPARSE_CONVERT_TYPES = (Float64, Float32, Int)
const SPARSE_COMPACTION = (("zeros10", 0.1), ("zeros50", 0.5))
const SPARSE_MOMENTUM_LAYOUTS = ("accessors", "private_fields")
sparse_vector_shape(n, s) = "n$(n)_s$(s)"
sparse_matrix_shape(dims, label) = "$(dims[1])x$(dims[2])_d$label"
sparse_symmetric_shape(p, s) = "p$(p)_s$(s)"
sparse_convert_shape(dims) = "$(dims[1])x$(dims[2])_d20"
sparse_batch_count() = min(16, SPARSE_POOL ÷ 256)
sparse_rmul_repeats(T, s) = min(exponent(floatmax(T)) - 4, repeats_for(s, 16384))
sparse_batch_layouts() = [how * suffix for suffix in ("", "_x$(sparse_batch_count())")
                          for how in ("absent_distinct_batch", "activate_distinct_batch")]

# the branch of the copy kernel once source and destination hold the same entries (4(nnz + nnz) < length)
sparse_convert_branch(dims, k) = 4 * (2 * k) < prod(dims) ? "sparse_branch" : "dense_branch"
function sparse_convert_ids(dims, k, shape)
    branch = sparse_convert_branch(dims, k)
    [("copyto!_convert", case_id("sparse", "copyto!_convert", Float64, shape, "to_$(Td)_$(branch)", DEFAULT_PHASE))
     for Td in SPARSE_CONVERT_TYPES]
end
sparse_matrix_convert_ids(dims) = sparse_convert_ids(dims, round(Int, 0.2 * prod(dims)), sparse_convert_shape(dims))
sparse_construct_ids(T, n) =
    [("construct", case_id("sparse", "construct", T, "n$n", "empty", "construct"))]

# the pairs of `sparse_vector_cases!` for (T, n, s); `rows` are the rows of the dense operands of `mul!`
function sparse_vector_ids(T, n, s; rows=())
    shape = sparse_vector_shape(n, s)
    ids = Tuple{String,String}[]
    add(name, layout, phase=DEFAULT_PHASE; shape=shape) = push!(ids, (name, case_id("sparse", name, T, shape, layout, phase)))
    add("read_x[i]", "active"); add("read_x[i]", "absent")
    add("write_x[i]_active", "active")
    add("isactive", "half_active")
    for layout in sparse_batch_layouts()
        add("first_activation", layout, "first_activation")
    end
    add("sweep_full", "vector"); add("sweep_active_update", "trivial_update")
    add("sweep_activecoordinates", "vector"); add("sweep_activevalues", "vector")
    add("sweep_via_coordinates", "rank1"); add("sweep_via_x", "rank1")
    add("dot_via_coordinates", "rank1"); add("dot_x", "rank1")
    for z in SPARSE_ZLABELS
        add("dot_vx", z); add("dot_xy", z)
    end
    for m in rows, name in ("strided", "transposed"), z in SPARSE_ZLABELS
        add("mul!", "$(name)_$z"; shape=shape * "_m$m")
    end
    r = sparse_rmul_repeats(T, s)
    add("rmul!_finite", "x$r"); add("rmul!_inf", "x$r")
    add("fill!_coordinates_zero", "vector")
    add("prox_coordinate!_5arg", "NormL1_scalar_u")
    for layout in SPARSE_MOMENTUM_LAYOUTS, name in ("_values!", "_setvalues!", "_restartdot")
        add(name, layout)
    end
    add("copyto!_unrelated", "dense_same_nonzeros")
    add("broadcast_self", "abs")
    add("copyto!_activevalues", "slot_order")
    add("fill!_zero", "vector"); add("fill!_zero_dropzeros", "vector")
    add("broadcast_assign", "vector"); add("copyto!_state", "vector")
    add("copy", "vector", "allocating")
    add("copyto!_same_kind", "same_entries")
    add("copyto!_dense", "dense_other_nonzeros")
    T == Float64 && append!(ids, sparse_convert_ids((n,), s, shape))
    add("replace_cheap", "vector"); add("replace_staged", "vector")
    for (zlabel, _) in SPARSE_COMPACTION
        add("dropzeros!", zlabel)
    end
    ids
end

# the pairs of `sparse_matrix_cases!` for (T, dims, label); the 20% conversions are `sparse_matrix_convert_ids`
function sparse_matrix_ids(T, dims, label, rows)
    shape = sparse_matrix_shape(dims, label)
    ids = Tuple{String,String}[]
    add(name, layout) = push!(ids, (name, case_id("sparse", name, T, shape, layout, DEFAULT_PHASE)))
    add("sweep_via_coordinates", "rank2"); add("sweep_via_x", "rank2")
    for mrows in rows, z in SPARSE_ZLABELS
        add("mul!_matrix", "rows$(mrows)_$z")
    end
    for z in SPARSE_ZLABELS
        add("dot_aXb", z)
    end
    ids
end

function sparse_symmetric_ids(T, p, density)
    shape = sparse_symmetric_shape(p, round(Int, density * p))
    [("mul!", case_id("sparse", "mul!", T, shape, "symmetric_$z", DEFAULT_PHASE)) for z in SPARSE_ZLABELS]
end

# The grid of the sparse group, with its scenarios in building order: the constructions, the vectors (the dense
# operands of `mul!` only at (n, s) = (10⁴, 10²)), the matrices, the Symmetric operands and the 20% conversions.
function sparse_grid(quick)
    (; types=quick ? (Float64,) : (Float32, Float64),
       vectors=quick ? ((10^4, 10^2), (10^6, 10^3)) : ((10^4, 10^2), (10^6, 10^2), (10^6, 10^3), (10^6, 10^4)),
       rows=quick ? (256,) : (256, 1024),       # rows of the dense operands, and p of the Symmetric ones
       matrix_dims=((1000, 64), (256, 1000)),
       densities=quick ? SPARSE_DENSITIES[2:2] : SPARSE_DENSITIES,
       symmetric_densities=quick ? SPARSE_DENSITIES[2:2] : SPARSE_DENSITIES[2:3],
       convert_dims=quick ? ((1000, 64),) : ((1000, 64), (256, 1000)))
end
function sparse_scenarios(quick)
    grid = sparse_grid(quick)
    (; constructs=[(; T, n) for T in grid.types for n in unique(first.(grid.vectors))],
       vectors=[(; T, n, s, rows=(n, s) == (10^4, 10^2) ? grid.rows : ()) for T in grid.types for (n, s) in grid.vectors],
       matrices=[(; T, dims, density, label, rows=grid.rows) for T in grid.types for dims in grid.matrix_dims
                 for (density, label) in grid.densities],
       symmetrics=[(; T, p, density, label) for T in grid.types for p in grid.rows for (density, label) in grid.symmetric_densities],
       converts=collect(grid.convert_dims))
end

# FW-4 on the dense branch of the copy kernel: a matrix at 20%, a dedicated iterate with Float64 sources
function sparse_matrix_convert_cases!(cases, rep, dims)
    L = prod(dims)
    key = "sparse/convert/$(dims[1])x$(dims[2])/20"
    ids = randperm(scenario_rng(rep, key * "/ids"), L)[1:round(Int, 0.2 * L)]
    sparse_convert_cases!(cases, rep, dims, ids, key, sparse_convert_shape(dims))
end

function sparse_construct_case!(cases, T, n)
    addcase!(cases, "sparse", "construct", () -> (SparseIterate(T, n); nothing),
             function ()
                 x = SparseIterate(T, n)
                 # retained memory: one T array and two Int arrays of length n, plus headers
                 abs(Base.summarysize(x) - (sizeof(T) + 2 * sizeof(Int)) * n) <= 1024 || error("unexpected retained size")
                 nactive(x) == 0 || error("a new iterate has active coordinates")
             end;
             # the operation reads only scalars
             inputs=(;), T, shape="n$n", layout="empty", phase="construct", evals=1, zero_alloc=false, work=n)
end

# The operations of a vector scenario are registered by groups, in the original order. With a `selector` a group
# none of whose operation names has a matching ID is not run, through `Base.invokelatest` so that it is not
# compiled either, and the fixtures beyond `x` and its snapshot are built, once, by the accessors below, when a
# group that reads them runs. Without a selector every group runs and nothing is enumerated. Every random draw of
# this scenario comes from its own keyed `scenario_rng` stream (the reads and writes draw nothing), so a draw
# that is skipped with its fixture changes no other fixture and no global stream is used.
function sparse_vector_cases!(cases, rep, T, n, s, vcache; rows=(), selector=nothing)
    key = "sparse/vector/$T/$n/$s"
    shape = sparse_vector_shape(n, s)
    # one permutation: the active coordinates, a pool of absent ones (the batches of first activations,
    # the block trial) and the extra coordinates of the second iterate are disjoint slices of it
    perm = randperm(scenario_rng(rep, key * "/ids"), n)
    ids = perm[1:s]
    pool = perm[s+1:s+SPARSE_POOL]
    ids_y = vcat(ids[1:s÷2], perm[s+SPARSE_POOL+1:s+SPARSE_POOL+(s-s÷2)])
    vals = fixture_values(scenario_rng(rep, key * "/values"), T, s)
    wanted = selector === nothing ? nothing : operations_wanted(selector, sparse_vector_ids(T, n, s; rows))
    want(names...) = want_any(wanted, names)
    x = build_iterate(T, (n,), ids, vals)
    snap = copy(x)
    memo = Dict{Symbol,Any}()
    d_() = get!(() -> dense_array(T, (n,), ids, vals), memo, :d)::Vector{T}
    vals_y_() = get!(() -> fixture_values(scenario_rng(rep, key * "/values_y"), T, s), memo, :vals_y)::Vector{T}
    vals50_() = get!(() -> with_zeros(vals, zero_slots(scenario_rng(rep, key * "/zeros"), s, 0.5)), memo, :vals50)::Vector{T}
    vals_y50_() = get!(() -> with_zeros(vals_y_(), zero_slots(scenario_rng(rep, key * "/zeros_y"), s, 0.5)), memo, :vals_y50)::Vector{T}
    y_() = get!(() -> build_iterate(T, (n,), ids_y, vals_y_()), memo, :y)::typeof(x)
    dy_() = get!(() -> dense_array(T, (n,), ids_y, vals_y_()), memo, :dy)::Vector{T}
    x50_() = get!(() -> build_iterate(T, (n,), ids, vals50_()), memo, :x50)::typeof(x)
    y50_() = get!(() -> build_iterate(T, (n,), ids_y, vals_y50_()), memo, :y50)::typeof(x)
    d50_() = get!(() -> dense_array(T, (n,), ids, vals50_()), memo, :d50)::Vector{T}
    dy50_() = get!(() -> dense_array(T, (n,), ids_y, vals_y50_()), memo, :dy50)::Vector{T}
    v_() = get!(() -> randn(scenario_rng(rep, "sparse/v/$T/$n"), T, n), vcache, (T, n))::Vector{T}
    # the digests of the dense vectors that the copy and replace cases compare with at construction
    d_source_() = get!(() -> (; d=d_()), memo, :d_source)::NamedTuple{(:d,),Tuple{Vector{T}}}
    dy_source_() = get!(() -> (; dy=dy_()), memo, :dy_source)::NamedTuple{(:dy,),Tuple{Vector{T}}}
    d_built_() = get!(() -> input_digests(d_source_()), memo, :d_built)::Vector{UInt64}
    dy_built_() = get!(() -> input_digests(dy_source_()), memo, :dy_built)::Vector{UInt64}
    restore_state!() = (copyto!(x, snap); nothing)
    guarded(check) = restored(restore_state!, check)

    reads_writes = function ()
        d = d_()
        # scalar reads: 256 reads of active and, separately, of absent coordinates
        nread = 256
        for (kind, src) in (("active", ids), ("absent", pool))
            rids = cycled(src, nread)
            expected = sum_reads(d, rids, T)
            addcase!(cases, "sparse", "read_x[i]", () -> sum_reads(x, rids, T),
                     guarded(() -> sum_reads(x, rids, T) == expected || error("reads differ from the dense vector"));
                     inputs=(; x, rids), T, shape, layout=kind, zero_alloc=true, work=nread)
        end
        # writes to 1024 active coordinates, with the values they hold: the state is unchanged, and the
        # check first overwrites them
        wids = cycled(ids, 1024)
        wvals = d[wids]
        function check_write()
            write_values!(x, wids, -wvals)
            Array(x)[wids] == -wvals || error("writes were not stored")
            write_values!(x, wids, wvals)
            Array(x) == d || error("writes were not stored")
            nactive(x) == s || error("a write to an active coordinate changed the working set")
            nothing
        end
        addcase!(cases, "sparse", "write_x[i]_active", () -> write_values!(x, wids, wvals), guarded(check_write);
                 inputs=(; x, wids, wvals), T, shape, layout="active", zero_alloc=true, work=length(wids))
    end
    want("read_x[i]", "write_x[i]_active") && Base.invokelatest(reads_writes)

    isactive_group = function ()
        # isactive: 256 queries, half of them active
        qids = shuffle(scenario_rng(rep, key * "/queries"), vcat(cycled(ids, 128), pool[1:128]))
        idset = Set(ids)
        expected_active = count(in(idset), qids)          # the stored membership of the fixture, not the values
        addcase!(cases, "sparse", "isactive", () -> count_active(x, qids),
                 guarded(function ()
                             count_active(x, qids) == expected_active || error("isactive differs from the stored membership")
                             check_isactive_stored_zero(T)
                         end);
                 inputs=(; x, qids), T, shape, layout="half_active", zero_alloc=true, work=length(qids))
    end
    want("isactive") && Base.invokelatest(isactive_group)

    first_activation_group = function ()
        d = d_()
        # first activation: a batch of distinct absent coordinates per call, by assignment and by activate!;
        # the companion has up to 16 disjoint batches in one call, which a timer can resolve
        nbatch = sparse_batch_count()
        for (suffix, bids) in (("", pool[1:256]), ("_x$nbatch", pool[1:256*nbatch]))
            bvals = fixture_values(scenario_rng(rep, key * "/batch_values$suffix"), T, length(bids))
            for (how, op, written, inputs) in (("absent_distinct_batch", () -> write_values!(x, bids, bvals), bvals, (; bids, bvals)),
                                               ("activate_distinct_batch", () -> activate_all!(x, bids), zeros(T, length(bids)), (; bids)))
                function restore_batch!()      # outside the timed region
                    @inbounds for k in bids
                        x[k] = zero(T)
                    end
                    dropzeros!(x)
                    any(k -> isactive(x, k), bids) && error("a batch coordinate is still active: not a first activation")
                    nothing
                end
                # the activated coordinates and their values, copied when the case is built
                bids0, written0 = copy(bids), copy(written)
                function check_activation()
                    copyto!(x, snap)
                    restore_batch!()
                    length(unique(bids)) == length(bids) || error("batch coordinates are not distinct")
                    op()
                    all(k -> isactive(x, k), bids) || error("activation did not add every coordinate")
                    nactive(x) == s + length(bids) || error("unexpected working set after the activations")
                    expected = copy(d)
                    expected[bids0] = written0
                    Array(x) == expected || error("activated values differ from the dense reference")
                    restore_batch!()
                    nactive(x) == s && Array(x) == d || error("restore did not recover the iterate")
                    copyto!(x, snap)
                    nothing
                end
                addcase!(cases, "sparse", "first_activation", op, check_activation; inputs, T, shape, layout=how * suffix,
                         phase="first_activation", prepare=restore_batch!, evals=1, zero_alloc=true, work=length(bids))
            end
        end
    end
    want("first_activation") && Base.invokelatest(first_activation_group)

    sweeps_group = function ()
        d = d_()
        v = v_()
        # a full sweep and an active sweep with a trivial update: the iterate is unchanged
        unchanged() = Array(x) == d && nactive(x) == s || error("the sweep changed the iterate")
        addcase!(cases, "sparse", "sweep_full", () -> full_sweep!(x),
                 guarded(() -> (full_sweep!(x); unchanged())); inputs=(; x), T, shape, layout="vector", zero_alloc=true, work=n)
        addcase!(cases, "sparse", "sweep_active_update", () -> active_sweep!(x),
                 guarded(function ()
                             active_sweep!(x) == s || error("the sweep did not visit every active coordinate")
                             unchanged()
                         end);
                 inputs=(; x), T, shape, layout="trivial_update", zero_alloc=true, work=s)

        # FD-7: sweeps through the handles created inside the call
        sorted_ids = sort(ids)
        addcase!(cases, "sparse", "sweep_activecoordinates", () -> sum_active_ids(x),
                 guarded(function ()
                             sum_active_ids(x) == sum(ids) || error("sweep visited the wrong coordinates")
                             sort!(collect(activecoordinates(x))) == sorted_ids || error("the active coordinates are not the fixture's (a sum can agree)")
                         end);
                 inputs=(; x), T, shape, layout="vector", zero_alloc=true, work=s)
        absvals = sum(abs, vals)
        addcase!(cases, "sparse", "sweep_activevalues", () -> sum_active_values(x, T),
                 guarded(() -> close_scalar(sum_active_values(x, T), sum(vals), absvals, T));
                 inputs=(; x), T, shape, layout="vector", zero_alloc=true, work=s)

        # FD-5: a read-write sweep and a product through coordinates(x), against the same through x
        addcase!(cases, "sparse", "sweep_via_coordinates", () -> coordinate_sweep!(x),
                 guarded(() -> (coordinate_sweep!(x); unchanged())); inputs=(; x), T, shape, layout="rank1", zero_alloc=true, work=n)
        addcase!(cases, "sparse", "sweep_via_x", () -> index_sweep!(x),
                 guarded(() -> (index_sweep!(x); unchanged())); inputs=(; x), T, shape, layout="rank1", zero_alloc=true, work=n)
        # the references of the products are computed here, as the operand v is read by the operation
        absdot = dot(abs.(v), abs.(d))
        ref_dot = dot(v, d)
        addcase!(cases, "sparse", "dot_via_coordinates", () -> dot(v, coordinates(x)),
                 guarded(() -> close_scalar(dot(v, coordinates(x)), ref_dot, absdot, T));
                 inputs=(; x, v), T, shape, layout="rank1", zero_alloc=false, work=s)
        addcase!(cases, "sparse", "dot_x", () -> dot(v, x),
                 guarded(() -> close_scalar(dot(v, x), ref_dot, absdot, T));
                 inputs=(; x, v), T, shape, layout="rank1", zero_alloc=true, work=s)
    end
    want("sweep_full", "sweep_active_update", "sweep_activecoordinates", "sweep_activevalues", "sweep_via_coordinates",
         "sweep_via_x", "dot_via_coordinates", "dot_x") && Base.invokelatest(sweeps_group)

    dots_group = function ()
        v = v_()
        d = d_()
        x50 = x50_()
        vals50 = vals50_()
        # FD-9: products, with 0% and 50% of the stored values exactly zero at random slots
        for (zlabel, xz, valsz) in (("z0", x, vals), ("z50", x50, vals50))
            dz = zlabel == "z0" ? d : d50_()
            if want("dot_vx")
                ref_vx, scale_vx = dot(v, dz), dot(abs.(v), abs.(dz))
                addcase!(cases, "sparse", "dot_vx", () -> dot(v, xz),
                         guarded(() -> close_scalar(dot(v, xz), ref_vx, scale_vx, T));
                         inputs=(; x=xz, v), T, shape, layout=zlabel, zero_alloc=true, work=s)
            end
            if want("dot_xy")
                yz = zlabel == "z0" ? y_() : y50_()
                dyz = zlabel == "z0" ? dy_() : dy50_()
                ref_xy, scale_xy = dot(dz, dyz), dot(abs.(dz), abs.(dyz))
                addcase!(cases, "sparse", "dot_xy", () -> dot(xz, yz),
                         guarded(() -> close_scalar(dot(xz, yz), ref_xy, scale_xy, T));
                         inputs=(; x=xz, y=yz), T, shape, layout=zlabel, zero_alloc=false, work=s)
            end
        end
    end
    want("dot_vx", "dot_xy") && Base.invokelatest(dots_group)

    mul_group = function ()
        x50 = x50_()
        vals50 = vals50_()
        for m in rows        # dense operands, only at (n, s) = (10⁴, 10²)
            A = randn(scenario_rng(rep, "sparse/A/$T/$n/$m"), T, m, n)
            tB = transpose(randn(scenario_rng(rep, "sparse/B/$T/$n/$m"), T, n, m))
            out = zeros(T, m)
            for (name, M) in (("strided", A), ("transposed", tB)), (zlabel, xz, valsz) in (("z0", x, vals), ("z50", x50, vals50))
                ref_mul = M * dense_array(T, (n,), ids, valsz)
                function check_mul()
                    fill!(out, T(1000))        # a stale output must not pass
                    mul!(out, M, xz)
                    close_enough(out, ref_mul)
                end
                addcase!(cases, "sparse", "mul!", () -> mul!(out, M, xz), guarded(check_mul);
                         inputs=(; M, x=xz), T, shape=shape * "_m$m", layout="$(name)_$zlabel", zero_alloc=true, work=m * s)
            end
        end
    end
    want("mul!") && Base.invokelatest(mul_group)

    rmul_group = function ()
        d = d_()
        # FN-9: scaling by a number; a call repeats rmul! so that it takes a few µs at small s (the scale is a
        # power of two, and the repetitions stop before the values overflow, so the result is exact). For Inf
        # only the active entries are checked: the inactive ones stay 0 on 0.5.x and become NaN on 0.6.
        r = sparse_rmul_repeats(T, s)
        rscale = ldexp(one(T), r)
        check_finite() = (rmul_repeat!(x, T(2), r); Array(x) == d .* rscale || error("rmul! differs from the dense reference"))
        check_inf() = (rmul_repeat!(x, T(Inf), r);
                       all(x[k] == sign(d[k]) * T(Inf) for k in ids) || error("rmul!(x, Inf) differs at the active entries"))
        # the cases that change the iterate declare the snapshot that prepare restores it from
        addcase!(cases, "sparse", "rmul!_finite", () -> rmul_repeat!(x, T(2), r), guarded(check_finite);
                 inputs=(; snap), T, shape, layout="x$r", prepare=restore_state!, zero_alloc=false, work=r * s)
        addcase!(cases, "sparse", "rmul!_inf", () -> rmul_repeat!(x, T(Inf), r), guarded(check_inf);
                 inputs=(; snap), T, shape, layout="x$r", prepare=restore_state!, zero_alloc=false, work=r * s)
    end
    want("rmul!_finite", "rmul!_inf") && Base.invokelatest(rmul_group)

    fill_prox_group = function ()
        d = d_()
        # FP-7: fill!(coordinates(x), 0); only the values are checked (0.6 keeps the working set)
        addcase!(cases, "sparse", "fill!_coordinates_zero", () -> fill!(coordinates(x), 0),
                 guarded(() -> (fill!(coordinates(x), 0); iszero(Array(x)) || error("an entry is not zero after the fill")));
                 inputs=(; snap), T, shape, layout="vector", prepare=restore_state!, zero_alloc=true, work=s)

        # FD-12: the five-argument prox_coordinate! with a scalar u. One step takes about 1.5 ns on the pinned
        # source, so a call takes 1536 distinct coordinates: the active ones (at most 768), moved by a step
        # that keeps them nonzero, and absent ones that mostly end at zero and stay inactive, as in a solver.
        f, γ = NormL1(T(0.1)), T(0.5)
        nprox = 1536
        na = min(s, 768)
        kb = vcat(ids[1:na], pool[257:256+nprox-na])
        rng_u = scenario_rng(rep, key * "/steps")
        ub = T[(rand(rng_u, Bool) ? 1 : -1) * T(j <= na ? 0.1 + 0.4 * rand(rng_u) : 0.055 * rand(rng_u)) for j in 1:nprox]
        # the coordinates and their soft-thresholded values, computed when the case is built
        prox_ids = copy(kb)
        prox_vals = [sign(ub[j]) * max(abs(ub[j]) - γ * T(0.1), zero(T)) for j in eachindex(kb)]
        function check_prox()
            prox_batch!(x, f, kb, ub, γ)
            expected = copy(d)
            for j in eachindex(prox_ids)
                expected[prox_ids[j]] = prox_vals[j]
            end
            close_enough(Array(x), expected)
        end
        addcase!(cases, "sparse", "prox_coordinate!_5arg", () -> prox_batch!(x, f, kb, ub, γ), guarded(check_prox);
                 inputs=(; kb, ub, snap), T, shape, layout="NormL1_scalar_u", prepare=restore_state!, zero_alloc=false, work=length(kb))
    end
    want("fill!_coordinates_zero", "prox_coordinate!_5arg") && Base.invokelatest(fill_prox_group)

    momentum_group = function ()
        d = d_()
        # FD-8: the momentum value loops, through the accessor and through the private array
        ybuf = randn(scenario_rng(rep, key * "/momentum_y"), T, s)
        xcur = randn(scenario_rng(rep, key * "/momentum_cur"), T, s)
        xprev = randn(scenario_rng(rep, key * "/momentum_prev"), T, s)
        buf = zeros(T, s)
        # in the slot order of the snapshot, which a state copy keeps
        rd_total, rd_magnitude = restart_reference(d[collect(activecoordinates(snap))], ybuf, xcur, xprev)
        for (layout, getvals) in (("accessors", activevalues), ("private_fields", stored_values_by_field(x)))
            w = copy(vals)
            values_op, setvalues_op, dot_op = momentum_ops(getvals, x, buf, w, ybuf, xcur, xprev, s)
            function check_values()
                fill!(buf, zero(T))
                values_op()
                buf == d[collect(activecoordinates(x))] || error("_values! differs from the dense gather")
            end
            function check_setvalues()
                fill!(activevalues(x), zero(T))
                setvalues_op()
                Array(x) == d || error("_setvalues! differs from the dense reference")
            end
            check_restartdot() = close_scalar(dot_op(), rd_total, rd_magnitude, T)
            for (name, op, check, inputs) in (("_values!", values_op, check_values, (; x)),
                                              ("_setvalues!", setvalues_op, check_setvalues, (; x, w)),
                                              ("_restartdot", dot_op, check_restartdot, (; x, ybuf, xcur, xprev)))
                addcase!(cases, "sparse", name, op, guarded(check); inputs, T, shape, layout, zero_alloc=true, work=s)
            end
        end
    end
    want("_values!", "_setvalues!", "_restartdot") && Base.invokelatest(momentum_group)

    copies_group = function ()
        d = d_()
        dy = dy_()
        # FD-14: aliasing and self-reference. A copy whose source is the reference first compares the source
        # with its digest at construction.
        d_source, d_built, dy_source, dy_built = d_source_(), d_built_(), dy_source_(), dy_built_()
        y = y_()
        function check_unrelated()
            check_inputs(d_source, d_built, AFTER_BUILD)
            write_values!(x, ids, -vals)
            copyto!(x, d)
            Array(x) == d && nactive(x) == s || error("copyto! from a dense vector differs from the reference")
        end
        addcase!(cases, "sparse", "copyto!_unrelated", () -> (copyto!(x, d); nothing), guarded(check_unrelated);
                 inputs=(; x, d), T, shape, layout="dense_same_nonzeros", zero_alloc=false, work=n)
        addcase!(cases, "sparse", "broadcast_self", () -> (x .= abs.(x); nothing),
                 guarded(() -> (x .= abs.(x); Array(x) == abs.(d) || error("x .= abs.(x) differs from the dense reference")));
                 inputs=(; snap), T, shape, layout="abs", prepare=restore_state!, zero_alloc=false, work=n)
        # the aliasing copy on its own iterate whose active coordinates are 1:s in slot order, so that the
        # result is the identity with or without the aliasing defect of 0.5.x
        va = fixture_values(scenario_rng(rep, key * "/alias_values"), T, s)
        xa = build_iterate(T, (n,), 1:s, va)
        da = dense_array(T, (n,), 1:s, va)
        function check_alias()
            collect(activecoordinates(xa)) == collect(1:s) || error("the aliasing fixture is not in slot order")
            copyto!(xa, activevalues(xa))
            Array(xa) == da && nactive(xa) == s || error("copyto!(x, activevalues(x)) changed the values")
        end
        addcase!(cases, "sparse", "copyto!_activevalues", () -> (copyto!(xa, activevalues(xa)); nothing), check_alias;
                 inputs=(; xa), T, shape, layout="slot_order", zero_alloc=false, work=s)

        # FD-2: clearing and assigning, against the state copy; values only
        addcase!(cases, "sparse", "fill!_zero", () -> (fill!(x, 0); nothing),
                 guarded(() -> (fill!(x, 0); iszero(Array(x)) || error("fill! left a nonzero entry")));
                 inputs=(; snap), T, shape, layout="vector", prepare=restore_state!, zero_alloc=false, work=s)
        addcase!(cases, "sparse", "fill!_zero_dropzeros", () -> (dropzeros!(fill!(x, 0)); nothing),
                 guarded(() -> (dropzeros!(fill!(x, 0)); iszero(Array(x)) || error("fill! and dropzeros! left a nonzero entry")));
                 inputs=(; snap), T, shape, layout="vector", prepare=restore_state!, zero_alloc=false, work=s)
        addcase!(cases, "sparse", "broadcast_assign", () -> (x .= y; nothing),
                 guarded(() -> (x .= y; Array(x) == dy || error("x .= y differs from y")));
                 inputs=(; y, snap), T, shape, layout="vector", prepare=restore_state!, zero_alloc=false, work=n)
        addcase!(cases, "sparse", "copyto!_state", () -> (copyto!(x, y); nothing),
                 guarded(() -> (copyto!(x, y); Array(x) == dy || error("copyto!(x, y) differs from y")));
                 inputs=(; y, snap), T, shape, layout="vector", prepare=restore_state!, zero_alloc=false, work=2 * s)

        # copy and import
        addcase!(cases, "sparse", "copy", () -> (copy(x); nothing),
                 guarded(() -> (c = copy(x); c !== x && Array(c) == d || error("copy differs from the iterate")));
                 inputs=(; x), T, shape, layout="vector", phase="allocating", evals=1, zero_alloc=false, work=n)
        # a state copy onto an iterate that already holds the same entries (the state is unchanged); the check
        # clears the iterate first, so that a copy that does nothing cannot pass
        addcase!(cases, "sparse", "copyto!_same_kind", () -> (copyto!(x, snap); nothing),
                 guarded(() -> (fill!(x, 0); copyto!(x, snap); Array(x) == d || error("state copy differs from the source")));
                 inputs=(; x, snap), T, shape, layout="same_entries", zero_alloc=false, work=2 * s)
        addcase!(cases, "sparse", "copyto!_dense", () -> (copyto!(x, dy); nothing),
                 guarded(() -> (check_inputs(dy_source, dy_built, AFTER_BUILD); copyto!(x, dy);
                                Array(x) == dy || error("import differs from the dense array")));
                 inputs=(; dy, snap), T, shape, layout="dense_other_nonzeros", prepare=restore_state!, zero_alloc=false, work=n)
    end
    want("copyto!_unrelated", "broadcast_self", "copyto!_activevalues", "fill!_zero", "fill!_zero_dropzeros",
         "broadcast_assign", "copyto!_state", "copy", "copyto!_same_kind", "copyto!_dense") && Base.invokelatest(copies_group)

    # FW-4: conversions on dedicated iterates; the sources are Float64
    T == Float64 && want("copyto!_convert") && Base.invokelatest(sparse_convert_cases!, cases, rep, (n,), ids, key, shape)

    replace_group = function ()
        dy = dy_()
        dy_source, dy_built = dy_source_(), dy_built_()
        # FW-6: the cheap replace against the staged one
        addcase!(cases, "sparse", "replace_cheap", () -> replace_cheap!(x, dy),
                 guarded(() -> (check_inputs(dy_source, dy_built, AFTER_BUILD); replace_cheap!(x, dy);
                                Array(x) == dy || error("replace differs from the dense array")));
                 inputs=(; dy, snap), T, shape, layout="vector", prepare=restore_state!, zero_alloc=false, work=n)
        addcase!(cases, "sparse", "replace_staged", () -> replace_staged!(x, dy),
                 guarded(() -> (check_inputs(dy_source, dy_built, AFTER_BUILD); replace_staged!(x, dy);
                                Array(x) == dy || error("staged replace differs from the dense array")));
                 inputs=(; dy, snap), T, shape, layout="vector", prepare=restore_state!, zero_alloc=false, work=n)
    end
    want("replace_cheap", "replace_staged") && Base.invokelatest(replace_group)

    dropzeros_group = function ()
        # compaction: dropzeros! with 10% and 50% of the active coordinates holding an explicit zero
        for (zlabel, fraction) in SPARSE_COMPACTION
            zslots = zero_slots(scenario_rng(rep, key * "/compaction_$zlabel"), s, fraction)
            expected = with_zeros(vals, zslots)
            dz = dense_array(T, (n,), ids, expected)
            function prepare_zeros!()          # the snapshot with explicit zeros at those slots
                copyto!(x, snap)
                av = activevalues(x)
                @inbounds for p in zslots
                    av[p] = zero(T)
                end
                nothing
            end
            function check_compaction()
                prepare_zeros!()
                nactive(x) == s || error("the zeros were not stored")
                dropzeros!(x)
                nactive(x) == s - length(zslots) || error("dropzeros! left the wrong number of coordinates")
                all(!iszero, activevalues(x)) || error("dropzeros! left a stored zero")
                Array(x) == dz || error("dropzeros! changed the values")
            end
            addcase!(cases, "sparse", "dropzeros!", () -> dropzeros!(x), guarded(check_compaction);
                     inputs=(; snap, zslots), T, shape, layout=zlabel, prepare=prepare_zeros!, zero_alloc=true, work=s)
        end
    end
    want("dropzeros!") && Base.invokelatest(dropzeros_group)
    nothing
end

# rank 2: the read-write sweeps of FD-5 and the products of FD-9; like the vector scenarios, in groups. The
# dense operands `A`, `a` and `b` come from `acache`, keyed per (type, shape).
function sparse_matrix_cases!(cases, rep, T, dims, density, label, rows, acache; selector=nothing)
    m, ncol = dims
    L = m * ncol
    k = round(Int, density * L)               # exactly round(d·L) distinct entries
    key = "sparse/matrix/$T/$(m)x$(ncol)/$label"
    shape = sparse_matrix_shape(dims, label)
    ids = randperm(scenario_rng(rep, key * "/ids"), L)[1:k]
    vals = fixture_values(scenario_rng(rep, key * "/values"), T, k)
    wanted = selector === nothing ? nothing : operations_wanted(selector, sparse_matrix_ids(T, dims, label, rows))
    want(names...) = want_any(wanted, names)
    X = build_iterate(T, dims, ids, vals)
    snap = copy(X)
    memo = Dict{Symbol,Any}()
    vals50_() = get!(() -> with_zeros(vals, zero_slots(scenario_rng(rep, key * "/zeros"), k, 0.5)), memo, :vals50)::Vector{T}
    X50_() = get!(() -> build_iterate(T, dims, ids, vals50_()), memo, :X50)::typeof(X)
    restore_state!() = (copyto!(X, snap); nothing)
    guarded(check) = restored(restore_state!, check)

    sweeps_group = function ()
        D = dense_array(T, dims, ids, vals)
        unchanged() = Array(X) == D && nactive(X) == k || error("the sweep changed the iterate")
        addcase!(cases, "sparse", "sweep_via_coordinates", () -> coordinate_sweep!(X),
                 guarded(() -> (coordinate_sweep!(X); unchanged())); inputs=(; X), T, shape, layout="rank2", zero_alloc=true, work=L)
        addcase!(cases, "sparse", "sweep_via_x", () -> index_sweep!(X),
                 guarded(() -> (index_sweep!(X); unchanged())); inputs=(; X), T, shape, layout="rank2", zero_alloc=true, work=L)
    end
    want("sweep_via_coordinates", "sweep_via_x") && Base.invokelatest(sweeps_group)

    mul_group = function ()
        X50 = X50_()
        vals50 = vals50_()
        # the matrix mul! walks the stored entries when 32 nnz <= length and the columns otherwise
        function check_walk(Z)
            (32 * nnz(Z) <= length(Z)) == (density < 0.1) || error("the density $density changed the walk of mul!")
        end
        for mrows in rows
            A = get!(() -> randn(scenario_rng(rep, "sparse/A/$T/$(m)x$(ncol)/$mrows"), T, mrows, m), acache,
                     (T, dims, mrows))::Matrix{T}
            out = zeros(T, mrows, ncol)
            for (zlabel, Z, valsz) in (("z0", X, vals), ("z50", X50, vals50))
                # through a CSC copy of the dense reference: the same product, at the cost of the stored entries
                ref_mul = A * sparse(dense_array(T, dims, ids, valsz))
                function check_mul()
                    check_walk(Z)
                    fill!(out, T(1000))        # a stale output must not pass
                    mul!(out, A, Z)
                    close_enough(out, ref_mul)
                end
                addcase!(cases, "sparse", "mul!_matrix", () -> mul!(out, A, Z), guarded(check_mul);
                         inputs=(; A, Z), T, shape, layout="rows$(mrows)_$zlabel", zero_alloc=true, work=mrows * k)
            end
        end
    end
    want("mul!_matrix") && Base.invokelatest(mul_group)

    dot_group = function ()
        X50 = X50_()
        vals50 = vals50_()
        a = get!(() -> randn(scenario_rng(rep, "sparse/a/$T/$(m)x$(ncol)"), T, m), acache, (T, dims, :a))::Vector{T}
        b = get!(() -> randn(scenario_rng(rep, "sparse/b/$T/$(m)x$(ncol)"), T, ncol), acache, (T, dims, :b))::Vector{T}
        for (zlabel, Z, valsz) in (("z0", X, vals), ("z50", X50, vals50))
            Dz = dense_array(T, dims, ids, valsz)
            ref_dot, scale_dot = dot(a, Dz * b), dot(abs.(a), abs.(Dz) * abs.(b))
            check_dot() = close_scalar(dot(a, Z, b), ref_dot, scale_dot, T)
            addcase!(cases, "sparse", "dot_aXb", () -> dot(a, Z, b), guarded(check_dot);
                     inputs=(; a, Z, b), T, shape, layout=zlabel, zero_alloc=false, work=k)
        end
    end
    want("dot_aXb") && Base.invokelatest(dot_group)
    nothing
end

# mul! with a Symmetric operand and a SparseIterate vector of the same length (s = round(d·p))
function sparse_symmetric_cases!(cases, rep, T, p, density, label, scache)
    s = round(Int, density * p)
    key = "sparse/symmetric/$T/$p/$label"
    shape = sparse_symmetric_shape(p, s)
    S = get!(() -> Symmetric(randn(scenario_rng(rep, "sparse/S/$T/$p"), T, p, p)), scache, (T, p))::Symmetric{T,Matrix{T}}
    ids = randperm(scenario_rng(rep, key * "/ids"), p)[1:s]
    vals = fixture_values(scenario_rng(rep, key * "/values"), T, s)
    vals50 = with_zeros(vals, zero_slots(scenario_rng(rep, key * "/zeros"), s, 0.5))
    out = zeros(T, p)
    for (zlabel, valsz) in (("z0", vals), ("z50", vals50))
        x = build_iterate(T, (p,), ids, valsz)
        ref_mul = S * dense_array(T, (p,), ids, valsz)
        function check_mul()
            fill!(out, T(1000))        # a stale output must not pass
            mul!(out, S, x)
            close_enough(out, ref_mul)
        end
        addcase!(cases, "sparse", "mul!", () -> mul!(out, S, x), check_mul;
                 inputs=(; S, x), T, shape, layout="symmetric_$zlabel", zero_alloc=true, work=p * s)
    end
    nothing
end

# With a `selector` (see `build_cases`) a scenario or function none of whose IDs matches is not built, and the
# operation groups inside a scenario are run only when wanted. The guarded calls go through `Base.invokelatest`,
# so that what is skipped is not compiled. Without a selector nothing is enumerated and everything is built.
function sparse_cases!(cases, rep; quick=false, selector=nothing)
    scenarios = sparse_scenarios(quick)
    vcache, acache, scache = Dict{Any,Any}(), Dict{Any,Any}(), Dict{Any,Any}()
    for s in scenarios.constructs
        selector === nothing || any_wanted(selector, sparse_construct_ids(s.T, s.n)) || continue
        Base.invokelatest(sparse_construct_case!, cases, s.T, s.n)
    end
    for s in scenarios.vectors
        selector === nothing || any_wanted(selector, sparse_vector_ids(s.T, s.n, s.s; s.rows)) || continue
        Base.invokelatest(sparse_vector_cases!, cases, rep, s.T, s.n, s.s, vcache; s.rows, selector)
    end
    for s in scenarios.matrices
        selector === nothing || any_wanted(selector, sparse_matrix_ids(s.T, s.dims, s.label, s.rows)) || continue
        Base.invokelatest(sparse_matrix_cases!, cases, rep, s.T, s.dims, s.density, s.label, s.rows, acache; selector)
    end
    for s in scenarios.symmetrics
        selector === nothing || any_wanted(selector, sparse_symmetric_ids(s.T, s.p, s.density)) || continue
        Base.invokelatest(sparse_symmetric_cases!, cases, rep, s.T, s.p, s.density, s.label, scache)
    end
    # FW-4 on the dense branch of the copy kernel: matrices at 20% (dedicated, Float64 sources)
    for dims in scenarios.converts
        selector === nothing || any_wanted(selector, sparse_matrix_convert_ids(dims)) || continue
        Base.invokelatest(sparse_matrix_convert_cases!, cases, rep, dims)
    end
    nothing
end

# ---- the atom group: AtomIterate ----
#
# Fixtures: p entries in K = p/g groups, contiguous (ranges) or strided (the multi-response layout,
# group r holding r, r + K, r + 2K, ...); a fraction of the groups is active, and every entry of an
# active group is nonzero. The harness keeps its own group arrays: they are the reference of every
# check, and an iterate built from them owns a copy. Only calls that exist on 0.5.x and on 0.6 are
# used; `atomindices` and `samegroups` are reached through the build-time spelling below.

const ATOM_DENSITIES = ((0.01, "1"), (0.1, "10"))

# K = p/g contiguous ranges, or the strided groups
function atom_groups(p, g, strided)
    K = p ÷ g
    strided ? [collect(r:K:p) for r in 1:K] : [(g * (k - 1) + 1):(g * k) for k in 1:K]
end

# an iterate whose active groups `aids`, in this order, hold `vals`, group by group; an exact zero is
# stored after activate!, as a zero write does not activate its group
function fill_atom!(x, groups, aids, vals)
    pos = 0
    for k in aids
        activate!(x, k)
        for i in groups[k]
            pos += 1
            x[i] = vals[pos]
        end
    end
    x
end
build_atom(T, p, groups, aids, vals) = fill_atom!(AtomIterate(T, p, groups), groups, aids, vals)

# the dense vector the iterate must equal: the reference of every check
function dense_atom(T, p, groups, aids, vals)
    D = zeros(T, p)
    pos = 0
    for k in aids, i in groups[k]
        pos += 1
        D[i] = vals[pos]
    end
    D
end

# the entries of the groups `ks`, in this order
group_entries(groups, ks) = Int[i for k in ks for i in groups[k]]

# Version-dependent spelling, resolved when the case is built (0.5.x spelling as the fallback):
# atomindices(x, k) is x.groups[k], and samegroups(x, f) is the private ProximalBase._samelayout(f, x).
private_group(x, k) = x.groups[k]
group_indices_by() = isdefined(ProximalBase, :atomindices) ? ProximalBase.atomindices : private_group
private_samelayout(x, f) = ProximalBase._samelayout(f, x)
samegroups_by() = isdefined(ProximalBase, :samegroups) ? ProximalBase.samegroups : private_samelayout

# the loops of the cases are functions of their arguments, so that the timed closures are thin
function xor_group_indices(indices::F, x) where {F}
    acc = 0
    for k in activecoordinates(x)
        for j in indices(x, k)
            # an exclusive or, not a sum: the sum over a range has a closed form that the compiler uses,
            # and the loop would then measure nothing per index
            acc ⊻= j
        end
    end
    acc
end
function sum_group_views(indices::F, x, gvec) where {F}
    acc = zero(eltype(gvec))
    for k in activecoordinates(x)
        acc += sum(view(gvec, indices(x, k)))
    end
    acc
end
# the function barrier: the closures are built where the spelling has a concrete type
atom_index_ops(indices::F, x, gvec) where {F} =
    (() -> xor_group_indices(indices, x), () -> sum_group_views(indices, x, gvec))
# the entries of the active groups, through the write-through view `atom(x, k)`; the update adds a zero,
# which the compiler cannot drop (see full_sweep!), and no stored value is -0.0, so x is unchanged. The
# sweep returns the number of entries it touched.
function atom_update_sweep!(x)
    z = zero(eltype(x))
    touched = 0
    for k in activecoordinates(x)
        a = atom(x, k)
        @inbounds for j in eachindex(a)
            a[j] = a[j] + z
        end
        touched += length(a)
    end
    touched
end
function sum_atom_norms(x, ks, ::Type{T}) where {T}
    acc = zero(T)
    for k in ks
        acc += norm(atom(x, k))
    end
    acc
end
function sum_atom_dots(x, ks, w, ::Type{T}) where {T}
    acc = zero(T)
    for k in ks
        acc += dot(atom(x, k), w)
    end
    acc
end
function prox_group_batch!(x, f, ks, us, γ)
    @inbounds for j in eachindex(ks)
        prox_coordinate!(x, f, ks[j], us[j], γ)
    end
    nothing
end
function staged_group_batch!(stage!::F, x, f, ks, us, γ) where {F}
    @inbounds for j in eachindex(ks)
        k = ks[j]
        stage!(x, k, us[j])
        prox_coordinate!(x, f, k, γ)
    end
    nothing
end
# The staging control of FD-12 writes u into the entries of the group and takes the four-argument step.
# The private spelling writes the storage (no activation); the fallback writes through atom(x, k), which
# activates the group by the rule of x[i] = v. The two are different paths per penalty: NormL1 takes the
# separable step, a group penalty its block step.
struct StageStorage{I}
    indices::I
end
function (s::StageStorage)(x, k, u)
    st = x.storage
    G = s.indices(x, k)
    @inbounds for a in 1:length(G)
        st[G[a]] = u[a]
    end
    nothing
end
struct StageAtom end
(::StageAtom)(x, k, u) = (atom(x, k) .= u; nothing)
staging_by(x, indices) = hasfield(typeof(x), :storage) ? StageStorage(indices) : StageAtom()
# the function barrier of FD-12: the closures are built where the penalty, the iterate and the staging have concrete types
atom_prox_ops(stage, x, snap, f, ks, us, γ) =
    (() -> prox_group_batch!(x, f, ks, us, γ),
     () -> staged_group_batch!(stage, x, f, ks, us, γ),
     () -> (copyto!(x, snap); nothing))

# the momentum value loops of the audited solver over an AtomIterate (acceptance guide §3.4), through the
# accessors of 0.6 (activecoordinates and atomindices) or through the private fields that the downstream
# code reads today; a missing field falls back to the accessors
struct AtomAccessors{I}
    indices::I
end
struct AtomFields end
atom_loops_by(x, indices) =
    all(f -> hasfield(typeof(x), f), (:storage, :active, :nactive, :groups)) ? AtomFields() : AtomAccessors(indices)

function atom_nvalues(L::AtomAccessors, x)
    n = 0
    for k in activecoordinates(x)
        n += length(L.indices(x, k))
    end
    n
end
function atom_nvalues(::AtomFields, x)
    n = 0
    @inbounds for a in 1:x.nactive
        n += length(x.groups[x.active[a]])
    end
    n
end
function atom_values!(L::AtomAccessors, v, x)
    i = 0
    @inbounds for k in activecoordinates(x), j in L.indices(x, k)
        i += 1
        v[i] = x[j]
    end
    v
end
function atom_values!(::AtomFields, v, x)
    i = 0
    @inbounds for a in 1:x.nactive, j in x.groups[x.active[a]]
        i += 1
        v[i] = x.storage[j]
    end
    v
end
function atom_setvalues!(L::AtomAccessors, x, v)
    i = 0
    @inbounds for k in activecoordinates(x), j in L.indices(x, k)
        i += 1
        x[j] = v[i]
    end
    x
end
function atom_setvalues!(::AtomFields, x, v)
    i = 0
    @inbounds for a in 1:x.nactive, j in x.groups[x.active[a]]
        i += 1
        x.storage[j] = v[i]
    end
    x
end
function atom_restartdot(L::AtomAccessors, x, y, xcur, xprev)
    d, i = zero(eltype(y)), 0
    @inbounds for k in activecoordinates(x), j in L.indices(x, k)
        i += 1
        d += (x[j] - y[i]) * (xcur[i] - xprev[i])
    end
    d
end
function atom_restartdot(::AtomFields, x, y, xcur, xprev)
    d, i = zero(eltype(y)), 0
    @inbounds for a in 1:x.nactive, j in x.groups[x.active[a]]
        i += 1
        d += (x.storage[j] - y[i]) * (xcur[i] - xprev[i])
    end
    d
end
# the function barrier: the closures are built where the spelling has a concrete type
function atom_momentum_ops(L, x, buf, w, y, xcur, xprev)
    (() -> atom_nvalues(L, x),
     () -> atom_values!(L, buf, x),
     () -> atom_setvalues!(L, x, w),
     () -> atom_restartdot(L, x, y, xcur, xprev))
end

# FD-11: the timed closures are built where the spelling of samegroups has a concrete type
struct ResetLayoutCaches{X}
    xs::Vector{X}
end
function (r::ResetLayoutCaches)()
    for x in r.xs
        x.layout_cache = nothing
    end
    nothing
end
reset_layout_caches_by(xs) = hasfield(eltype(xs), :layout_cache) ? ResetLayoutCaches(xs) : noop
function first_seen(samegroups::F, xs, f) where {F}
    same = true
    for x in xs
        same &= samegroups(x, f)
    end
    same
end
function samegroups_ops(samegroups::F, x_shared, xs_first, x_alt, f, f2) where {F}
    (() -> samegroups(x_shared, f),
     () -> first_seen(samegroups, xs_first, f),
     () -> (samegroups(x_alt, f); samegroups(x_alt, f2)))
end

# the layout with its first two groups exchanged: a valid layout that differs from `groups` in one pair
function swapped_groups(groups)
    bad = copy(groups)
    bad[1], bad[2] = bad[2], bad[1]
    bad
end

# one step per group of the block trial: the call takes a batch of distinct groups, since one group step
# takes 12 ns (g = 1, NormL1) to a µs or more (g = 64), and a call must take at least about 2 µs
atom_prox_batch(g) = g == 64 ? 64 : 256

# a penalty of the block trial (FD-12), by name
# Euclidean projection of u onto {v : ‖v‖₁ ≤ τ} (sort and threshold)
function project_l1_ball(u::AbstractVector{<:AbstractFloat}, τ::AbstractFloat)
    sum(abs, u) <= τ && return copy(u)
    a = sort(abs.(u); rev=true)
    cs = cumsum(a)
    ρ = findlast(j -> a[j] > (cs[j] - τ) / j, eachindex(a))
    θ = (cs[ρ] - τ) / ρ
    return sign.(u) .* max.(abs.(u) .- θ, 0.0)
end

function atom_penalty(name, T, groups)
    name == "GroupNormL2" && return GroupNormL2(T(0.1), groups)
    name == "GroupNormLinf" && return GroupNormLinf(T(0.1), groups)
    name == "SparseGroupNormL2" && return SparseGroupNormL2(T(0.1), T(0.1), groups)
    NormL1(T(0.1))
end

# The result of the block trial on a batch of groups, by hand: soft thresholding for NormL1 (entry by
# entry), the group shrinkage max(1 - γ λ √g / ‖u‖, 0) u for GroupNormL2 (default weight √g), soft
# thresholding and then that shrinkage for SparseGroupNormL2; GroupNormLinf (default weight 1) by Moreau's
# identity: the prox of τ‖·‖∞ at u is u minus its Euclidean projection onto the ℓ1 ball of radius τ
# (`project_l1_ball`, a sort-based projection that shares no code with the package, in Float64).
function prox_batch_reference(name, f, T, p, g, groups, d, ks, ubases, γ)
    expected = copy(d)
    λ = T(0.1)
    soft(u, c) = sign(u) * max(abs(u) - c, zero(T))
    if name == "GroupNormLinf"
        τ = Float64(γ) * Float64(λ)
        for (k, ub) in zip(ks, ubases)
            u = Float64.(ub[1:2:end])
            expected[groups[k]] = T.(u .- project_l1_ball(u, τ))
        end
        return expected
    end
    for (k, ub) in zip(ks, ubases)
        u = ub[1:2:end]
        if name == "NormL1"
            expected[groups[k]] = soft.(u, γ * λ)
        elseif name == "GroupNormL2"
            nrm = norm(u)
            expected[groups[k]] = nrm == 0 ? zero.(u) : max(one(T) - γ * λ * sqrt(T(g)) / nrm, zero(T)) .* u
        else    # SparseGroupNormL2(λ1 = 0.1, λ2 = 0.1)
            s = soft.(u, γ * λ)
            nrm = norm(s)
            expected[groups[k]] = nrm == 0 ? zero.(s) : max(one(T) - γ * λ * sqrt(T(g)) / nrm, zero(T)) .* s
        end
    end
    expected
end

# Selection of atom cases before their fixtures are built. A scenario (or function) lists the IDs it registers
# from cheap fields only (type, sizes, label, `full`, the arithmetic of its counts), as (operation name, ID)
# pairs. With a selector it is skipped when none of them matches, and inside a retained scenario an operation
# group runs only when one of its names has a matching ID. A missing ID is a false exclusion (the one bug that
# matters), an extra ID only costs build time. The registrations and the enumerators spell shapes, keys,
# layouts and counts through the helpers below and `case_id`. A scenario or function whose IDs cannot be listed
# completely and cheaply is built, and the exact filter that follows `build_cases` decides.
atom_kind(strided) = strided ? "strided" : "contiguous"
atom_shape(p, g, strided, label) = "p$(p)_g$(g)_$(atom_kind(strided))_act$label"
atom_layout_shape(p, g, strided) = "p$(p)_g$(g)_$(atom_kind(strided))"
atom_symmetric_shape(g) = "p1024_g$(g)_contiguous_act10"
atom_key(T, p, g, strided, label) = "atom/$T/$p/$g/$(atom_kind(strided))/$label"
atom_label_density(label) = only(d for (d, l) in ATOM_DENSITIES if l == label)
atom_nactive_groups(density, K) = max(1, round(Int, density * K))
atom_batch_count(K, na) = min(16, (K - na) ÷ 256)
atom_rmul_repeats(T, p) = min(exponent(floatmax(T)) - 4, repeats_for(p, 16384))
atom_penalties(full) = full ? ("GroupNormL2", "GroupNormLinf", "SparseGroupNormL2", "NormL1") : ("GroupNormL2", "NormL1")
atom_prox_layout(pname) = "$(pname)_strided_u"
atom_mul_layout(m, zlabel) = "rows$(m)_$zlabel"
const ATOM_PROX_NAMES = ("prox_coordinate!_5arg", "prox_coordinate!_staged")
const ATOM_ZLABELS = ("z0", "z50")
const ATOM_MOMENTUM_LAYOUTS = ("accessors", "private_fields")
const ATOM_MOMENTUM_NAMES = ("_nvalues", "_values!", "_setvalues!", "_restartdot")
const ATOM_CONVERT_TYPES = (Float64, Float32, Int)
const ATOM_COMPACTION = (("zeros10", 0.1), ("zeros50", 0.5))
const ATOM_LAYOUTS = (; isactive="half_active", update="trivial_update", loops="active_groups", dense_same="dense_same_entries",
                      abs="abs", same_entries="same_entries", dense_other="dense_other_entries")
atom_batch_layouts(nbatch) = [how * suffix for suffix in ("", "_x$nbatch") for how in ("absent_distinct_batch", "activate_distinct_batch")]

# the (operation name, ID) pairs that `atom_scenario_cases!` registers for this scenario, `full` and `rows`
function atom_scenario_ids(T, p, g, strided, label; full=true, rows=(), density=atom_label_density(label))
    K = p ÷ g
    na = atom_nactive_groups(density, K)
    shape = atom_shape(p, g, strided, label)
    lay = "atoms"
    ids = Tuple{String,String}[]
    add(name, layout, phase=DEFAULT_PHASE; type=T) = push!(ids, (name, case_id("atom", name, type, shape, layout, phase)))
    add("read_x[i]", "active"); add("read_x[i]", "absent")
    add("write_x[i]_active", "active")
    add("isactive", ATOM_LAYOUTS.isactive)
    for layout in atom_batch_layouts(atom_batch_count(K, na))
        add("first_activation", layout, "first_activation")
    end
    add("sweep_active_update", ATOM_LAYOUTS.update)
    add("sweep_activecoordinates", lay)
    add("atom_norm", lay); add("atom_dot", lay)
    add("atomindices_loop", ATOM_LAYOUTS.loops)
    add("atomindices_view", strided ? "view_vector_group" : "view_range_group")
    for z in ATOM_ZLABELS
        add("dot_vx", z)
    end
    for m in rows, z in ATOM_ZLABELS
        add("mul!", atom_mul_layout(m, z))
    end
    r = atom_rmul_repeats(T, p)
    add("rmul!_finite", "x$r"); add("rmul!_inf", "x$r")
    for pname in atom_penalties(full), name in ATOM_PROX_NAMES
        add(name, atom_prox_layout(pname))
    end
    if full
        for z in ATOM_ZLABELS
            add("dot_xy", "atom_atom_$z"); add("dot_xy", "atom_sparse_$z")
        end
        for layout in ATOM_MOMENTUM_LAYOUTS, name in ATOM_MOMENTUM_NAMES
            add(name, layout)
        end
        add("copyto!_unrelated", ATOM_LAYOUTS.dense_same)
        add("broadcast_self", ATOM_LAYOUTS.abs)
        add("fill!_zero", lay); add("fill!_zero_dropzeros", lay)
        add("broadcast_assign", lay); add("copyto!_state", lay)
        add("copy", lay, "allocating")
        add("copyto!_same_kind", ATOM_LAYOUTS.same_entries)
        add("copyto!_dense", ATOM_LAYOUTS.dense_other)
        if T == Float64
            for Td in ATOM_CONVERT_TYPES
                add("copyto!_convert", "to_$(Td)"; type=Float64)
            end
        end
        add("replace_cheap", lay); add("replace_staged", lay)
        for (zlabel, _) in ATOM_COMPACTION
            add("dropzeros!", zlabel)
        end
    end
    ids
end

function atom_construct_ids(T, p, g, strided)
    [("construct", case_id("atom", "construct", T, atom_layout_shape(p, g, strided), "empty", "construct"))]
end
function atom_layout_ids(p, g, strided)
    [("samegroups", case_id("atom", "samegroups", Float64, atom_layout_shape(p, g, strided), layout, DEFAULT_PHASE))
     for layout in ("shared", "distinct_first_seen", "alternating")]
end
function atom_symmetric_ids(T, g)
    [("mul!", case_id("atom", "mul!", T, atom_symmetric_shape(g), "symmetric_$z", DEFAULT_PHASE)) for z in ATOM_ZLABELS]
end

any_wanted(::Nothing, pairs) = true
any_wanted(selector::Regex, pairs) = any(pair -> occursin(selector, pair[2]), pairs)
# the operation names of a scenario with a selected ID, or `nothing` (every operation) without a selector
operations_wanted(::Nothing, pairs) = nothing
operations_wanted(selector::Regex, pairs) = Set{String}(name for (name, id) in pairs if occursin(selector, id))
want_any(::Nothing, names) = true
want_any(wanted::Set{String}, names) = any(in(wanted), names)

# FD-12: the five-argument block trial against staging u and the four-argument step, with a strided u (a
# view with stride 2 of a longer vector), for the four penalties; the batch has distinct groups, half of
# them active (moved to nonzero values) and half inactive (small trial points, which mostly stay at zero)
function atom_prox_cases!(cases, rep, T, p, g, strided, groups, shape, key, x, snap, d, aids, pool, fcache, penalties; selector=nothing)
    γ = T(0.5)
    nb = atom_prox_batch(g)
    na_b = min(length(aids), nb ÷ 2)
    ks = vcat(aids[1:na_b], pool[1:nb-na_b])
    rng_u = scenario_rng(rep, key * "/trial_points")
    ubases = Vector{T}[]
    for j in 1:nb
        ub = fill(T(100), 2g)             # the odd slots are the trial point, the even ones are never read
        scale = j <= na_b ? T(0.5) : T(0.02)
        for a in 1:g
            ub[2a-1] = (rand(rng_u, Bool) ? one(T) : -one(T)) * (j <= na_b ? scale + scale * T(rand(rng_u)) : scale * T(rand(rng_u)))
        end
        push!(ubases, ub)
    end
    us = [view(ub, 1:2:2g) for ub in ubases]
    indices = group_indices_by()
    for pname in penalties
        # a penalty none of whose IDs is selected builds neither its penalty, its iterate nor its reference
        selector === nothing || any(name -> wants(selector, case_id("atom", name, T, shape, atom_prox_layout(pname), DEFAULT_PHASE)),
                                    ATOM_PROX_NAMES) || continue
        f = get!(() -> atom_penalty(pname, T, groups), fcache, (T, p, g, strided, pname))
        # a group penalty requires an iterate with its layout; NormL1 takes the iterate of the scenario
        xf, snapf = if pname == "NormL1"
            x, snap
        else
            z = AtomIterate(d, f)
            z, copy(z)
        end
        op5, opstaged, restore! = atom_prox_ops(staging_by(xf, indices), xf, snapf, f, ks, us, γ)
        # the reference on the entries of the batch, computed when the case is built (the trial points are
        # read by the operations); the other entries keep the scenario's dense vector
        ents = group_entries(groups, ks)
        ents_vals = prox_batch_reference(pname, f, T, p, g, groups, d, copy(ks), deepcopy(ubases), γ)[ents]
        # both spellings must reach the reference and each other (values only, not the working set)
        function check_pair(op_a, op_b)
            expected = copy(d)
            expected[ents] = ents_vals
            restore!()
            op_a()
            ra = Vector(xf)
            close_enough(ra, expected)
            restore!()
            op_b()
            rb = Vector(xf)
            close_enough(rb, expected)
            close_enough(ra, rb)
            restore!()
            nothing
        end
        layout = atom_prox_layout(pname)
        addcase!(cases, "atom", ATOM_PROX_NAMES[1], op5, () -> check_pair(op5, opstaged); inputs=(; ks, ubases, snapf),
                 T, shape, layout, prepare=restore!, zero_alloc=true, work=nb * g)
        addcase!(cases, "atom", ATOM_PROX_NAMES[2], opstaged, () -> check_pair(opstaged, op5); inputs=(; ks, ubases, snapf),
                 T, shape, layout, prepare=restore!, zero_alloc=false, work=nb * g)
    end
    nothing
end

# FW-4: a state copy with a conversion, on dedicated iterates. The source holds integral values and is
# built from scratch, so that its whole storage converts to Int; it is never changed, and `similar` gives
# the destinations its layout, so the same-layout kernel runs. Only the success path is measured.
function atom_convert_cases!(cases, rep, p, g, groups, x, aids, key, shape)
    nvals = length(aids) * g
    rng = scenario_rng(rep, key * "/integral")
    iv = Float64[(rand(rng, Bool) ? 1 : -1) * rand(rng, 1:7) for _ in 1:nvals]
    src = fill_atom!(similar(x), groups, aids, iv)
    dsrc = dense_atom(Float64, p, groups, aids, iv)
    for Td in (Float64, Float32, Int)
        dest = similar(src, Td)
        op = () -> (copyto!(dest, src); nothing)
        check = function ()
            fill!(dest, 0)
            for _ in 1:2        # the first call starts from an empty destination, the second from a full one
                copyto!(dest, src)
                Vector(dest) == Td.(dsrc) || error("converted copy differs from the dense reference")
            end
            nothing
        end
        addcase!(cases, "atom", "copyto!_convert", op, check; inputs=(; src), T=Float64, shape, layout="to_$(Td)", zero_alloc=false, work=nvals)
    end
end

# one scenario (T, p, g, strided, density): the working iterate, its snapshot and the references; the
# cases of a scenario run one at a time, so they share the iterate and restore it. `full` is false for
# the layouts that the quick suite builds only in part. Dense operands exist only for `rows`.
#
# With a `selector` the scenario builds what the operations with a selected ID read. The operations are
# registered by groups, each a closure run through `Base.invokelatest` when one of its operation names has a
# selected ID: a branch around inline code would still be inferred and compiled, and compiling is most of the
# cost of a cold build. Without a selector every group runs, in this order. The iterate and its snapshot are
# built for every retained operation; the other fixtures are built, once and shared by the operations that
# read them, by the accessors below (`memo`), whose results have concrete types so that the timed closures
# capture what they always captured. The draws of `rng_pick` are one stream in registration order, and the
# draws after the read cases are made even when only the writes are built; every other draw has its own
# keyed stream and goes with its fixture.
function atom_scenario_cases!(cases, rep, T, p, g, strided, density, label, groups, caches; full=true, rows=(), selector=nothing)
    vcache, acache, fcache = caches
    K = p ÷ g
    na = atom_nactive_groups(density, K)
    key = atom_key(T, p, g, strided, label)
    shape = atom_shape(p, g, strided, label)
    lay = "atoms"
    # one permutation: the active groups, in the order they join, and the other groups, in random order
    perm = randperm(scenario_rng(rep, key * "/groups"), K)
    aids = perm[1:na]
    pool = perm[na+1:end]
    nvals = na * g
    vals = fixture_values(scenario_rng(rep, key * "/values"), T, nvals)
    # the second iterate shares half of the active groups, and has the other half among inactive ones
    ny = na - na ÷ 2
    aids_y = vcat(aids[1:na÷2], pool[end-ny+1:end])
    wanted = selector === nothing ? nothing :
             operations_wanted(selector, atom_scenario_ids(T, p, g, strided, label; full, rows, density))
    want(names...) = want_any(wanted, names)
    x = build_atom(T, p, groups, aids, vals)
    snap = copy(x)
    memo = Dict{Symbol,Any}()
    aentries_() = get!(() -> group_entries(groups, aids), memo, :aentries)::Vector{Int}
    d_() = get!(() -> dense_atom(T, p, groups, aids, vals), memo, :d)::Vector{T}
    vals_y_() = get!(() -> fixture_values(scenario_rng(rep, key * "/values_y"), T, nvals), memo, :vals_y)::Vector{T}
    vals50_() = get!(() -> with_zeros(vals, zero_slots(scenario_rng(rep, key * "/zeros"), nvals, 0.5)), memo, :vals50)::Vector{T}
    vals_y50_() = get!(() -> with_zeros(vals_y_(), zero_slots(scenario_rng(rep, key * "/zeros_y"), nvals, 0.5)), memo, :vals_y50)::Vector{T}
    y_() = get!(() -> fill_atom!(similar(x), groups, aids_y, vals_y_()), memo, :y)::typeof(x)
    dy_() = get!(() -> dense_atom(T, p, groups, aids_y, vals_y_()), memo, :dy)::Vector{T}
    x50_() = get!(() -> fill_atom!(similar(x), groups, aids, vals50_()), memo, :x50)::typeof(x)
    y50_() = get!(() -> fill_atom!(similar(x), groups, aids_y, vals_y50_()), memo, :y50)::typeof(x)
    d50_() = get!(() -> dense_atom(T, p, groups, aids, vals50_()), memo, :d50)::Vector{T}
    dy50_() = get!(() -> dense_atom(T, p, groups, aids_y, vals_y50_()), memo, :dy50)::Vector{T}
    v_() = get!(() -> randn(scenario_rng(rep, "atom/v/$T/$p"), T, p), vcache, (T, p))::Vector{T}
    # the digests of the dense vectors that the copy and replace cases compare with at construction
    d_source_() = get!(() -> (; d=d_()), memo, :d_source)::NamedTuple{(:d,),Tuple{Vector{T}}}
    dy_source_() = get!(() -> (; dy=dy_()), memo, :dy_source)::NamedTuple{(:dy,),Tuple{Vector{T}}}
    d_built_() = get!(() -> input_digests(d_source_()), memo, :d_built)::Vector{UInt64}
    dy_built_() = get!(() -> input_digests(dy_source_()), memo, :dy_built)::Vector{UInt64}
    restore_state!() = (copyto!(x, snap); nothing)
    guarded(check) = restored(restore_state!, check)

    reads_writes = function ()
        d = d_()
        aentries = aentries_()
        rng_pick = scenario_rng(rep, key * "/picks")
        # scalar reads: 256 reads of entries of active and, separately, of inactive groups
        nread = 256
        active_reads = cycled(aentries[randperm(rng_pick, nvals)[1:min(nread, nvals)]], nread)
        absent_reads = Int[groups[k][rand(rng_pick, 1:g)] for k in cycled(pool, nread)]
        if want("read_x[i]")
            for (what, rids) in (("active", active_reads), ("absent", absent_reads))
                expected = sum_reads(d, rids, T)
                addcase!(cases, "atom", "read_x[i]", () -> sum_reads(x, rids, T),
                         guarded(() -> sum_reads(x, rids, T) == expected || error("reads differ from the dense vector"));
                         inputs=(; x, rids), T, shape, layout=what, zero_alloc=true, work=nread)
            end
        end
        # writes to 1024 entries of active groups, with the values they hold: the state is unchanged, and the
        # check first overwrites them
        wids = cycled(aentries[randperm(rng_pick, nvals)[1:min(1024, nvals)]], 1024)
        if want("write_x[i]_active")
            wvals = d[wids]
            function check_write()
                write_values!(x, wids, -wvals)
                Vector(x)[wids] == -wvals || error("writes were not stored")
                write_values!(x, wids, wvals)
                Vector(x) == d || error("writes were not stored")
                nactive(x) == na || error("a write to an active group changed the working set")
                nothing
            end
            addcase!(cases, "atom", "write_x[i]_active", () -> write_values!(x, wids, wvals), guarded(check_write);
                     inputs=(; x, wids, wvals), T, shape, layout="active", zero_alloc=true, work=length(wids))
        end
    end
    want("read_x[i]", "write_x[i]_active") && Base.invokelatest(reads_writes)

    isactive_group = function ()
        # isactive: 256 queries, half of them active
        qgroups = shuffle(scenario_rng(rep, key * "/queries"), vcat(cycled(aids, 128), pool[1:128]))
        aidset = Set(aids)
        expected_active = count(in(aidset), qgroups)       # the stored membership of the fixture, not the values
        addcase!(cases, "atom", "isactive", () -> count_active(x, qgroups),
                 guarded(function ()
                             count_active(x, qgroups) == expected_active || error("isactive differs from the stored membership")
                             check_isactive_stored_zero_atom(T)
                         end);
                 inputs=(; x, qgroups), T, shape, layout=ATOM_LAYOUTS.isactive, zero_alloc=true, work=length(qgroups))
    end
    want("isactive") && Base.invokelatest(isactive_group)

    first_activation_group = function ()
        d = d_()
        # first activation: a nonzero write to the first entry of each of a batch of distinct inactive groups,
        # and activate!; the companion has as many disjoint batches of 256 groups as fit (up to 16) in one call,
        # which a timer can resolve
        nbatch = atom_batch_count(K, na)
        for (suffix, bgroups) in (("", pool[1:256]), ("_x$nbatch", pool[1:256*nbatch]))
            bents = Int[first(groups[k]) for k in bgroups]
            bvals = fixture_values(scenario_rng(rep, key * "/batch_values$suffix"), T, length(bgroups))
            for (how, op, written, inputs) in (("absent_distinct_batch", () -> write_values!(x, bents, bvals), bvals, (; bents, bvals, bgroups)),
                                               ("activate_distinct_batch", () -> activate_all!(x, bgroups), zeros(T, length(bgroups)), (; bgroups, bents)))
                function restore_batch!()      # outside the timed region
                    @inbounds for i in bents
                        x[i] = zero(T)
                    end
                    dropzeros!(x)
                    any(k -> isactive(x, k), bgroups) && error("a batch group is still active: not a first activation")
                    nothing
                end
                # the activated entries and their values, copied when the case is built
                bents0, written0 = copy(bents), copy(written)
                function check_activation()
                    copyto!(x, snap)
                    restore_batch!()
                    length(unique(bgroups)) == length(bgroups) || error("batch groups are not distinct")
                    op()
                    all(k -> isactive(x, k), bgroups) || error("activation did not add every group")
                    nactive(x) == na + length(bgroups) || error("unexpected working set after the activations")
                    expected = copy(d)
                    expected[bents0] = written0
                    Vector(x) == expected || error("activated values differ from the dense reference")
                    restore_batch!()
                    nactive(x) == na && Vector(x) == d || error("restore did not recover the iterate")
                    copyto!(x, snap)
                    nothing
                end
                addcase!(cases, "atom", "first_activation", op, check_activation; inputs, T, shape, layout=how * suffix,
                         phase="first_activation", prepare=restore_batch!, evals=1, zero_alloc=true, work=length(bgroups))
            end
        end
    end
    want("first_activation") && Base.invokelatest(first_activation_group)

    sweeps_group = function ()
        d = d_()
        # an active sweep with a trivial update (the state is unchanged) and FD-7: the sweep through the handle
        # created inside the call
        addcase!(cases, "atom", "sweep_active_update", () -> atom_update_sweep!(x),
                 guarded(function ()
                             atom_update_sweep!(x) == nvals || error("the sweep did not visit every entry of the active groups")
                             Vector(x) == d && nactive(x) == na || error("the sweep changed the iterate")
                         end);
                 inputs=(; x), T, shape, layout=ATOM_LAYOUTS.update, zero_alloc=true, work=nvals)
        sorted_aids = sort(aids)
        addcase!(cases, "atom", "sweep_activecoordinates", () -> sum_active_ids(x),
                 guarded(function ()
                             sum_active_ids(x) == sum(aids) || error("sweep visited the wrong groups")
                             sort!(collect(activecoordinates(x))) == sorted_aids || error("the active groups are not the fixture's (a sum can agree)")
                         end);
                 inputs=(; x), T, shape, layout=lay, zero_alloc=true, work=na)

        # atom access: norm and dot of atom(x, k) over 64 active groups per call
        ks64 = cycled(aids, 64)
        w = randn(scenario_rng(rep, key * "/atom_w"), T, g)
        expected_norm = sum(norm(d[groups[k]]) for k in ks64)
        expected_dot = sum(dot(d[groups[k]], w) for k in ks64)
        scale_dot = sum(dot(abs.(d[groups[k]]), abs.(w)) for k in ks64)
        addcase!(cases, "atom", "atom_norm", () -> sum_atom_norms(x, ks64, T),
                 guarded(() -> close_scalar(sum_atom_norms(x, ks64, T), expected_norm, expected_norm, T));
                 inputs=(; x, ks64), T, shape, layout=lay, zero_alloc=true, work=64 * g)
        addcase!(cases, "atom", "atom_dot", () -> sum_atom_dots(x, ks64, w, T),
                 guarded(() -> close_scalar(sum_atom_dots(x, ks64, w, T), expected_dot, scale_dot, T));
                 inputs=(; x, ks64, w), T, shape, layout=lay, zero_alloc=true, work=64 * g)

        # FD-6: the indices of every active group through atomindices(x, k) (0.5.x: x.groups[k]), and the sum of
        # a dense vector over them (a range in the contiguous layout, a vector in the strided one)
        v = v_()
        loop_op, view_op = atom_index_ops(group_indices_by(), x, v)
        expected_indices = foldl(⊻, (j for k in aids for j in groups[k]); init=0)
        addcase!(cases, "atom", "atomindices_loop", loop_op,
                 guarded(() -> loop_op() == expected_indices || error("the indices of the active groups differ from the layout"));
                 inputs=(; x), T, shape, layout=ATOM_LAYOUTS.loops, zero_alloc=true, work=nvals)
        expected_view = sum(sum(v[groups[k]]) for k in aids)
        scale_view = sum(sum(abs, v[groups[k]]) for k in aids)
        addcase!(cases, "atom", "atomindices_view", view_op,
                 guarded(() -> close_scalar(view_op(), expected_view, scale_view, T));
                 inputs=(; x, v), T, shape, layout=strided ? "view_vector_group" : "view_range_group", zero_alloc=true, work=nvals)
    end
    want("sweep_active_update", "sweep_activecoordinates", "atom_norm", "atom_dot", "atomindices_loop", "atomindices_view") &&
        Base.invokelatest(sweeps_group)

    dot_vx_group = function ()
        v = v_()
        d = d_()
        # FD-9: products, with 0% and 50% of the entries of the active groups exactly zero at random places in
        # storage order (a group left all zero stays active)
        absv = abs.(v)
        for zlabel in ATOM_ZLABELS
            xz = zlabel == "z0" ? x : x50_()
            dz = zlabel == "z0" ? d : d50_()
            scale = dot(absv, abs.(dz))
            expected = dot(v, dz)
            addcase!(cases, "atom", "dot_vx", () -> dot(v, xz),
                     guarded(() -> close_scalar(dot(v, xz), expected, scale, T));
                     inputs=(; x=xz, v), T, shape, layout=zlabel, zero_alloc=true, work=nvals)
        end
    end
    want("dot_vx") && Base.invokelatest(dot_vx_group)

    mul_group = function ()
        d = d_()
        for m in rows
            A = get!(() -> randn(scenario_rng(rep, "atom/A/$T/$m/$p"), T, m, p), acache, (T, m, p))::Matrix{T}
            out = zeros(T, m)
            for zlabel in ATOM_ZLABELS
                xz = zlabel == "z0" ? x : x50_()
                dz = zlabel == "z0" ? d : d50_()
                ref_mul = A * dz
                check_mul() = (fill!(out, T(1000)); mul!(out, A, xz); close_enough(out, ref_mul))
                addcase!(cases, "atom", "mul!", () -> mul!(out, A, xz), guarded(check_mul);
                         inputs=(; A, x=xz), T, shape, layout=atom_mul_layout(m, zlabel), zero_alloc=true, work=m * nvals)
            end
        end
    end
    want("mul!") && Base.invokelatest(mul_group)

    rmul_group = function ()
        d = d_()
        # FN-9: scaling by a number. AtomIterate has no rmul! method, so the generic loop runs: O(p) for a finite
        # scale, and rmul!(x, Inf) writes NaN into every inactive entry and activates every group. The scale is a
        # power of two and the repetitions stop before the values overflow, so the finite result is exact; for Inf
        # the whole vector is compared with isequal (NaN where the entry was zero). prepare restores by a state
        # copy, which zeroes the entries of every active group and so also resets the poisoning.
        r = atom_rmul_repeats(T, p)
        rscale = ldexp(one(T), r)
        check_finite() = (rmul_repeat!(x, T(2), r); Vector(x) == d .* rscale || error("rmul! differs from the dense reference"))
        check_inf() = (rmul_repeat!(x, T(Inf), r); isequal(Vector(x), d .* T(Inf)) || error("rmul!(x, Inf) differs from the dense reference"))
        addcase!(cases, "atom", "rmul!_finite", () -> rmul_repeat!(x, T(2), r), guarded(check_finite);
                 inputs=(; snap), T, shape, layout="x$r", prepare=restore_state!, zero_alloc=false, work=r * p)
        addcase!(cases, "atom", "rmul!_inf", () -> rmul_repeat!(x, T(Inf), r), guarded(check_inf);
                 inputs=(; snap), T, shape, layout="x$r", prepare=restore_state!, zero_alloc=false, work=r * p)
    end
    want("rmul!_finite", "rmul!_inf") && Base.invokelatest(rmul_group)

    prox_group = function ()
        # FD-12: the block trial, the five-argument call against staging
        atom_prox_cases!(cases, rep, T, p, g, strided, groups, shape, key, x, snap, d_(), aids, pool, fcache, atom_penalties(full); selector)
    end
    want(ATOM_PROX_NAMES...) && Base.invokelatest(prox_group)

    full || return nothing

    dot_xy_group = function ()
        d, dy = d_(), dy_()
        # FD-9: dot(x, y) of two iterates that share the layout, and with a SparseIterate
        idsy = group_entries(groups, aids_y)
        for zlabel in ATOM_ZLABELS
            xz = zlabel == "z0" ? x : x50_()
            yz = zlabel == "z0" ? y_() : y50_()
            dz = zlabel == "z0" ? d : d50_()
            dyz = zlabel == "z0" ? dy : dy50_()
            valsyz = zlabel == "z0" ? vals_y_() : vals_y50_()
            sz = build_iterate(T, (p,), idsy, valsyz)
            scale = dot(abs.(dz), abs.(dyz))
            expected = dot(dz, dyz)
            addcase!(cases, "atom", "dot_xy", () -> dot(xz, yz),
                     guarded(() -> close_scalar(dot(xz, yz), expected, scale, T));
                     inputs=(; x=xz, y=yz), T, shape, layout="atom_atom_$zlabel", zero_alloc=false, work=nvals)
            addcase!(cases, "atom", "dot_xy", () -> dot(xz, sz),
                     guarded(() -> close_scalar(dot(xz, sz), expected, scale, T));
                     inputs=(; x=xz, y=sz), T, shape, layout="atom_sparse_$zlabel", zero_alloc=false, work=nvals)
        end
    end
    want("dot_xy") && Base.invokelatest(dot_xy_group)

    momentum_group = function ()
        d = d_()
        # FD-8: the momentum value loops, through the accessors and through the private fields
        ybuf = randn(scenario_rng(rep, key * "/momentum_y"), T, nvals)
        xcur = randn(scenario_rng(rep, key * "/momentum_cur"), T, nvals)
        xprev = randn(scenario_rng(rep, key * "/momentum_prev"), T, nvals)
        buf = zeros(T, nvals)
        gather = group_entries(groups, aids)
        rd_total, rd_magnitude = restart_reference(d[gather], ybuf, xcur, xprev)
        for (layout, loops) in ((ATOM_MOMENTUM_LAYOUTS[1], AtomAccessors(group_indices_by())),
                                (ATOM_MOMENTUM_LAYOUTS[2], atom_loops_by(x, group_indices_by())))
            wm = copy(vals)
            nvalues_op, values_op, setvalues_op, dot_op = atom_momentum_ops(loops, x, buf, wm, ybuf, xcur, xprev)
            function check_nvalues()
                nvalues_op() == nvals || error("_nvalues differs from the layout")
            end
            function check_values()
                fill!(buf, zero(T))
                values_op()
                buf == d[gather] || error("_values! differs from the dense gather")
            end
            function check_setvalues()
                write_values!(x, gather, -vals)
                setvalues_op()
                Vector(x) == d || error("_setvalues! differs from the dense reference")
            end
            check_restartdot() = close_scalar(dot_op(), rd_total, rd_magnitude, T)
            for (name, op, check, inputs) in ((ATOM_MOMENTUM_NAMES[1], nvalues_op, check_nvalues, (; x)),
                                              (ATOM_MOMENTUM_NAMES[2], values_op, check_values, (; x)),
                                              (ATOM_MOMENTUM_NAMES[3], setvalues_op, check_setvalues, (; x, w=wm)),
                                              (ATOM_MOMENTUM_NAMES[4], dot_op, check_restartdot, (; x, ybuf, xcur, xprev)))
                # _nvalues takes the length of each active group: one step per group, not per entry
                addcase!(cases, "atom", name, op, guarded(check); inputs, T, shape, layout, zero_alloc=true,
                         work=name == "_nvalues" ? na : nvals)
            end
        end
    end
    want(ATOM_MOMENTUM_NAMES...) && Base.invokelatest(momentum_group)

    copy_group = function ()
        d = d_()
        aentries = aentries_()
        # FD-14: copyto! from an unrelated dense vector with the same entries, and a self broadcast. A copy
        # whose source is the reference first compares the source with its digest at construction.
        want("copyto!_unrelated") && let d_source = d_source_(), d_built = d_built_()
            function check_unrelated()
                check_inputs(d_source, d_built, AFTER_BUILD)
                write_values!(x, aentries, -d[aentries])
                copyto!(x, d)
                Vector(x) == d || error("copyto! from a dense vector differs from the reference")
            end
            addcase!(cases, "atom", "copyto!_unrelated", () -> (copyto!(x, d); nothing), guarded(check_unrelated);
                     inputs=(; x, d), T, shape, layout=ATOM_LAYOUTS.dense_same, zero_alloc=false, work=p)
        end
        want("broadcast_self") &&
            addcase!(cases, "atom", "broadcast_self", () -> (x .= abs.(x); nothing),
                     guarded(() -> (x .= abs.(x); Vector(x) == abs.(d) || error("x .= abs.(x) differs from the dense reference")));
                     inputs=(; snap), T, shape, layout=ATOM_LAYOUTS.abs, prepare=restore_state!, zero_alloc=false, work=p)

        # FD-2: clearing and assigning, against the state copy; values only
        want("fill!_zero") &&
            addcase!(cases, "atom", "fill!_zero", () -> (fill!(x, 0); nothing),
                     guarded(() -> (fill!(x, 0); iszero(Vector(x)) || error("fill! left a nonzero entry")));
                     inputs=(; snap), T, shape, layout=lay, prepare=restore_state!, zero_alloc=false, work=nvals)
        want("fill!_zero_dropzeros") &&
            addcase!(cases, "atom", "fill!_zero_dropzeros", () -> (dropzeros!(fill!(x, 0)); nothing),
                     guarded(() -> (dropzeros!(fill!(x, 0)); iszero(Vector(x)) || error("fill! and dropzeros! left a nonzero entry")));
                     inputs=(; snap), T, shape, layout=lay, prepare=restore_state!, zero_alloc=false, work=nvals)
        want("broadcast_assign") && let y = y_(), dy = dy_()
            addcase!(cases, "atom", "broadcast_assign", () -> (x .= y; nothing),
                     guarded(() -> (x .= y; Vector(x) == dy || error("x .= y differs from y")));
                     inputs=(; y, snap), T, shape, layout=lay, prepare=restore_state!, zero_alloc=false, work=p)
        end
        want("copyto!_state") && let y = y_(), dy = dy_()
            addcase!(cases, "atom", "copyto!_state", () -> (copyto!(x, y); nothing),
                     guarded(() -> (copyto!(x, y); Vector(x) == dy || error("copyto!(x, y) differs from y")));
                     inputs=(; y, snap), T, shape, layout=lay, prepare=restore_state!, zero_alloc=false, work=2 * nvals)
        end

        # copy and import
        want("copy") &&
            addcase!(cases, "atom", "copy", () -> (copy(x); nothing),
                     guarded(() -> (c = copy(x); c !== x && Vector(c) == d && nactive(c) == na || error("copy differs from the iterate")));
                     inputs=(; x), T, shape, layout=lay, phase="allocating", evals=1, zero_alloc=false, work=p)
        # a state copy onto an iterate that already holds the same entries (the state is unchanged); the check
        # clears the iterate first, so that a copy that does nothing cannot pass
        want("copyto!_same_kind") &&
            addcase!(cases, "atom", "copyto!_same_kind", () -> (copyto!(x, snap); nothing),
                     guarded(() -> (fill!(x, 0); copyto!(x, snap); Vector(x) == d || error("state copy differs from the source")));
                     inputs=(; x, snap), T, shape, layout=ATOM_LAYOUTS.same_entries, zero_alloc=false, work=2 * nvals)
        want("copyto!_dense") && let dy = dy_(), dy_source = dy_source_(), dy_built = dy_built_()
            addcase!(cases, "atom", "copyto!_dense", () -> (copyto!(x, dy); nothing),
                     guarded(() -> (check_inputs(dy_source, dy_built, AFTER_BUILD); copyto!(x, dy);
                                    Vector(x) == dy || error("import differs from the dense array")));
                     inputs=(; dy, snap), T, shape, layout=ATOM_LAYOUTS.dense_other, prepare=restore_state!, zero_alloc=false, work=p)
        end

        # FW-4: conversions on dedicated iterates; the sources are Float64
        T == Float64 && want("copyto!_convert") &&
            Base.invokelatest(atom_convert_cases!, cases, rep, p, g, groups, x, aids, key, shape)

        # FW-6: the cheap replace against the staged one
        want("replace_cheap", "replace_staged") && let dy = dy_(), dy_source = dy_source_(), dy_built = dy_built_()
            want("replace_cheap") &&
                addcase!(cases, "atom", "replace_cheap", () -> replace_cheap!(x, dy),
                         guarded(() -> (check_inputs(dy_source, dy_built, AFTER_BUILD); replace_cheap!(x, dy);
                                        Vector(x) == dy || error("replace differs from the dense array")));
                         inputs=(; dy, snap), T, shape, layout=lay, prepare=restore_state!, zero_alloc=false, work=p)
            want("replace_staged") &&
                addcase!(cases, "atom", "replace_staged", () -> replace_staged!(x, dy),
                         guarded(() -> (check_inputs(dy_source, dy_built, AFTER_BUILD); replace_staged!(x, dy);
                                        Vector(x) == dy || error("staged replace differs from the dense array")));
                         inputs=(; dy, snap), T, shape, layout=lay, prepare=restore_state!, zero_alloc=false, work=p)
        end
    end
    want("copyto!_unrelated", "broadcast_self", "fill!_zero", "fill!_zero_dropzeros", "broadcast_assign", "copyto!_state", "copy",
         "copyto!_same_kind", "copyto!_dense", "copyto!_convert", "replace_cheap", "replace_staged") && Base.invokelatest(copy_group)

    dropzeros_group = function ()
        # compaction: dropzeros! with 10% and 50% of the active groups all zero (they stay active until the call)
        for (zlabel, fraction) in ATOM_COMPACTION
            nz = max(1, round(Int, fraction * na))
            zpos = randperm(scenario_rng(rep, key * "/compaction_$zlabel"), na)[1:nz]
            valsz = copy(vals)
            for j in zpos
                valsz[((j-1)*g+1):(j*g)] .= zero(T)
            end
            snapz = fill_atom!(similar(x), groups, aids, valsz)
            kept = sort(aids[setdiff(1:na, zpos)])
            function prepare_zeros!()          # the snapshot with all-zero active groups
                copyto!(x, snapz)
                nothing
            end
            function check_compaction()
                prepare_zeros!()
                nactive(x) == na || error("the all-zero groups were not stored as active")
                dropzeros!(x)
                nactive(x) == na - nz || error("dropzeros! left the wrong number of groups")
                sort(collect(activecoordinates(x))) == kept || error("dropzeros! kept the wrong groups")
                Vector(x) == dense_atom(T, p, groups, aids, valsz) || error("dropzeros! changed the values")
            end
            addcase!(cases, "atom", "dropzeros!", () -> dropzeros!(x), guarded(check_compaction);
                     inputs=(; snapz), T, shape, layout=zlabel, prepare=prepare_zeros!, zero_alloc=true, work=nvals)
        end
    end
    want("dropzeros!") && Base.invokelatest(dropzeros_group)
    nothing
end

# FD-11: samegroups through the build-time spelling, with a shared layout object, a distinct equal layout
# seen for the first time (prepare resets the layout cache, so the call does one full comparison), and two
# distinct equal layouts used alternately (one full comparison per call). With few groups one comparison
# is below the 2 µs that a call with evals = 1 must take, so the first-seen call compares several iterates.
function atom_layout_cases!(cases, p, g, strided, groups)
    T = Float64
    K = p ÷ g
    shape = atom_layout_shape(p, g, strided)
    f = GroupNormL2(T(0.1), groups)           # a penalty owns a copy of the groups
    f2 = GroupNormL2(T(0.1), groups)
    x_shared = AtomIterate(T, p, f)           # shares f.groups
    nfirst = cld(4096, K)
    xs_first = [AtomIterate(T, p, groups) for _ in 1:nfirst]      # equal to f.groups, in other arrays
    x_alt = AtomIterate(T, p, groups)
    samegroups = samegroups_by()
    reset_first = reset_layout_caches_by(xs_first)
    op_shared, op_first, op_alt = samegroups_ops(samegroups, x_shared, xs_first, x_alt, f, f2)
    # what makes the cases what they are: the arrays are shared in one case and distinct in the others
    function check_fixture()
        if hasfield(typeof(x_shared), :groups)
            x_shared.groups === f.groups || error("the shared case does not share the layout object")
            (any(x -> x.groups === f.groups, xs_first) || x_alt.groups === f.groups || f.groups === f2.groups) &&
                error("a distinct layout shares its array")
        end
        nothing
    end
    function check_false()          # a layout that differs in one pair of entries
        fbad = GroupNormL2(T(0.1), swapped_groups(groups))
        samegroups(x_shared, fbad) === false || error("samegroups accepted a different layout")
        samegroups(xs_first[1], fbad) === false || error("samegroups accepted a different layout")
        samegroups(x_alt, fbad) === false || error("samegroups accepted a different layout")
        nothing
    end
    function check_shared()
        check_fixture()
        samegroups(x_shared, f) === true || error("samegroups rejected the shared layout")
        check_false()
    end
    function check_first()
        check_fixture()
        reset_first()
        op_first() === true || error("samegroups rejected an equal layout")
        reset_first()
        check_false()
        reset_first()
        nothing
    end
    function check_alternating()
        check_fixture()
        for _ in 1:2
            samegroups(x_alt, f) === true && samegroups(x_alt, f2) === true || error("samegroups rejected an equal layout")
        end
        check_false()
    end
    # the group arrays of the penalties, reached through hasfield (the layout cache is not an input)
    f_groups = hasfield(typeof(f), :groups) ? (; f_groups=f.groups) : (;)
    f2_groups = hasfield(typeof(f2), :groups) ? (; f2_groups=f2.groups) : (;)
    addcase!(cases, "atom", "samegroups", op_shared, check_shared; inputs=merge((; x_shared), f_groups),
             T, shape, layout="shared", zero_alloc=false, work=1)
    addcase!(cases, "atom", "samegroups", op_first, check_first; inputs=merge((; xs_first), f_groups), T, shape, layout="distinct_first_seen",
             prepare=reset_first, evals=1, zero_alloc=false, work=nfirst * K)
    addcase!(cases, "atom", "samegroups", op_alt, check_alternating; inputs=merge((; x_alt), f_groups, f2_groups),
             T, shape, layout="alternating", zero_alloc=false, work=2 * K)
    nothing
end

# mul! with a Symmetric operand (p = 1024) and an AtomIterate on contiguous groups, 10% of them active
function atom_symmetric_cases!(cases, rep, T, g, scache)
    p = 1024
    K = p ÷ g
    na = max(1, round(Int, 0.1 * K))
    key = "atom/symmetric/$T/$p/$g"
    shape = atom_symmetric_shape(g)
    S = get!(() -> Symmetric(randn(scenario_rng(rep, "atom/S/$T/$p"), T, p, p)), scache, (T, p))::Symmetric{T,Matrix{T}}
    groups = atom_groups(p, g, false)
    aids = randperm(scenario_rng(rep, key * "/groups"), K)[1:na]
    vals = fixture_values(scenario_rng(rep, key * "/values"), T, na * g)
    vals50 = with_zeros(vals, zero_slots(scenario_rng(rep, key * "/zeros"), na * g, 0.5))
    out = zeros(T, p)
    for (zlabel, valsz) in (("z0", vals), ("z50", vals50))
        x = build_atom(T, p, groups, aids, valsz)
        ref_mul = S * dense_atom(T, p, groups, aids, valsz)
        function check_mul()
            fill!(out, T(1000))
            mul!(out, S, x)
            close_enough(out, ref_mul)
        end
        addcase!(cases, "atom", "mul!", () -> mul!(out, S, x), check_mul;
                 inputs=(; S, x), T, shape, layout="symmetric_$zlabel", zero_alloc=true, work=p * na * g)
    end
    nothing
end

function atom_construct_case!(cases, T, p, g, strided, groups)
    K = p ÷ g
    function check()
        x = AtomIterate(T, p, groups)
        retained = Base.summarysize(x)
        if strided      # each group vector has its own header: a lower bound
            retained >= (sizeof(T) + 16) * p + 24 * K || error("unexpected retained size")
        else            # entries, group_of, K ranges, active and position
            abs(retained - ((sizeof(T) + 8) * p + 32 * K)) <= 1024 || error("unexpected retained size")
        end
        length(x) == p && ncoordinates(x) == K && nactive(x) == 0 || error("a new iterate is not empty")
    end
    addcase!(cases, "atom", "construct", () -> (AtomIterate(T, p, groups); nothing), check; inputs=(; groups),
             T, shape=atom_layout_shape(p, g, strided), layout="empty", phase="construct", evals=1,
             zero_alloc=false, work=p)
end

# The scenarios of the atom group, in building order: the grid of sizes, layouts and densities of the suite, with
# the rows of the dense operands (at p = 8·10⁴ only) and whether the scenario builds every operation (`full`).
function atom_grid(quick)
    (; types=quick ? (Float64,) : (Float32, Float64),
       sizes=quick ? (8 * 10^4,) : (8 * 10^4, 8 * 10^5),
       layouts=quick ? [(1, false), (8, false), (8, true)] : [(g, s) for g in (1, 8, 64) for s in (false, true)],
       densities=quick ? ATOM_DENSITIES[2:2] : ATOM_DENSITIES,
       rows=quick ? (256,) : (256, 1024),
       symmetric_groups=quick ? (8,) : (1, 8, 64))
end
function atom_scenarios(quick)
    grid = atom_grid(quick)
    scenarios = NamedTuple[]
    for T in grid.types, p in grid.sizes, (g, strided) in grid.layouts, (density, label) in grid.densities
        full = !quick || (g == 8 && !strided)
        push!(scenarios, (; T, p, g, strided, density, label, full, rows=p == 8 * 10^4 ? grid.rows : ()))
    end
    scenarios
end

# With a `selector` (see `build_cases`) a scenario or function none of whose IDs matches is not built, and the
# group layouts are built only for what is. The guarded calls go through `Base.invokelatest`, so that a function
# that is skipped is not compiled. Without a selector nothing is enumerated and everything is built.
function atom_cases!(cases, rep; quick=false, selector=nothing)
    grid = atom_grid(quick)
    caches = (Dict{Any,Any}(), Dict{Any,Any}(), Dict{Any,Any}())     # vectors, dense operands, penalties
    scache = Dict{Any,Any}()
    gcache = Dict{Any,Any}()
    groups_of(p, g, strided) = get!(() -> atom_groups(p, g, strided), gcache, (p, g, strided))
    for T in grid.types, p in grid.sizes, (g, strided) in grid.layouts
        selector === nothing || any_wanted(selector, atom_construct_ids(T, p, g, strided)) || continue
        Base.invokelatest(atom_construct_case!, cases, T, p, g, strided, groups_of(p, g, strided))
    end
    for s in atom_scenarios(quick)
        selector === nothing ||
            any_wanted(selector, atom_scenario_ids(s.T, s.p, s.g, s.strided, s.label; s.full, s.rows, s.density)) || continue
        Base.invokelatest(atom_scenario_cases!, cases, rep, s.T, s.p, s.g, s.strided, s.density, s.label,
                          groups_of(s.p, s.g, s.strided), caches; s.full, s.rows, selector)
    end
    for p in grid.sizes, (g, strided) in grid.layouts
        selector === nothing || any_wanted(selector, atom_layout_ids(p, g, strided)) || continue
        Base.invokelatest(atom_layout_cases!, cases, p, g, strided, groups_of(p, g, strided))
    end
    for T in grid.types, g in grid.symmetric_groups
        selector === nothing || any_wanted(selector, atom_symmetric_ids(T, g)) || continue
        Base.invokelatest(atom_symmetric_cases!, cases, rep, T, g, scache)
    end
    nothing
end

const CASE_GROUPS = ("vectors", "smooth", "matrices", "constraints", "tensors", "products", "blocks", "owner", "sparse", "atom", "ridge")

# `selector` (a Regex or `nothing`) lets the builders that take it, `sparse_cases!` and `atom_cases!`, skip scenarios and
# operations none of whose case IDs it matches, before their fixtures exist. The result is a superset of the
# cases whose IDs match, in the original relative order, with the fixtures they always had; the caller applies
# the exact filter. Without a selector it is the whole catalog.
function build_cases(rep; suite="quick", groups=collect(CASE_GROUPS), selector::Union{Nothing,Regex}=nothing)
    suite in ("quick", "extended") || error("PB_SUITE must be quick or extended")
    isempty(groups) && error("Select at least one benchmark group")
    all(in(CASE_GROUPS), groups) || error("PB_GROUPS must select from $(join(CASE_GROUPS, ','))")
    length(unique(groups)) == length(groups) || error("Duplicate benchmark group")
    cases=PerfCase[]
    builders=(separable_cases!,smooth_cases!,matrix_cases!,constrained_cases!,tensor_cases!,product_cases!,block_cases!,owner_cases!,sparse_cases!,atom_cases!,ridge_cases!)
    # Select groups and sizes before allocating data or constructing function objects.
    for (group,builder) in zip(CASE_GROUPS,builders)
        group in groups || continue
        builder in (sparse_cases!, atom_cases!) ? builder(cases,rep;quick=suite=="quick",selector) : builder(cases,rep;quick=suite=="quick")
    end
    suite == "extended" && "vectors" in groups && group_layout_cases!(cases, rep)
    ids=[c.meta.id for c in cases]
    length(unique(ids))==length(ids) || error("duplicate case IDs")
    cases
end
