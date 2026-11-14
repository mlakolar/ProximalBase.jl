# SparseIterate 0.6: source-level method inventory and disposition

Status: static inventory of the pre-removal `master` source on 2026-11-08, archived as evidence of the methods **deleted or migrated in PR #835**. The 0.6 public contract is the amended `iterate-family-0.6.md` and the current `sparse_iterate.md` guide. This is an inventory of explicit definitions and required inherited functionality; it is **not** a runtime dump of Julia's effective method table, a performance measurement, or a passed test run. All line references point to `src/sparse_iterate.jl` unless noted. The code for symmetric-owner methods shares this file; it is intentionally excluded except where a common helper serves `SparseIterate`.

## Definition of "all"

1. Public data-structure operations implemented directly for `SparseIterate`.
2. Explicit support methods on the wrappers `SparseCoordinates` and `_ActiveIDs`.
3. Internal functions used by those methods, including shared `_StoredIterate = Union{SparseIterate,SymmetricSparseIterate}`.
4. Concrete `SparseIterate` specializations outside `src/sparse_iterate.jl`.
5. Ordinary `AbstractArray` behavior inherited from Julia. This group does **not** mean a separate method for each call in this package or sparse-optimal complexity.
6. Exclusions: numerical-function oracles such as `restrict`, `prox!`, `prox_coordinate!`, `kkt_violation`, `gradient!`, etc. are catalogued separately because they are **not** methods that the iterate must implement.

## A. Coordinate storage and basic array implementation: KEEP

| Exact methods / group | Where | Why / status |
|:--|:--|:--|
| `SparseIterate(n)`, `SparseIterate(T,n)`, `SparseIterate(n,m)`, `SparseIterate(T,n,m)` | 524–527 | Explicit, capacity-preallocating constructors; essential |
| `SparseIterate(::Array)`, `SparseIterate(::SparseVector)`, `SparseIterate(::SparseMatrixCSC)` | 610–642 | Explicit imports; useful construction convenience; review retention separately from core |
| `SparseIterate{T,N}(nzval,nzval2ind,ind2nzval,nnz)` and field-based outer constructor | 60–66 | Explicit constructors currently callable; require invariants/visibility review, not a solver API |
| `Base.length(x::SparseIterate)` | 563 | Explicit O(1) |
| `Base.size(x::SparseIterate)` | 564 | Explicit O(1) |
| `Base.IndexStyle(::Type{<:SparseIterate})` | 565 | Explicit `IndexLinear()` |
| `Base.getindex(x::SparseIterate,i::Int)` | 567 | Explicit O(1) logical value read |
| `Base.setindex!(x::SparseIterate,v,i::Int)` | 568 | Explicit O(1) logical value write/activation |
| `ncoordinates(x::_StoredIterate)` | 312 | Explicit via shared union, equals length; O(1) |
| `nactive(x::_StoredIterate)` | 313 | Explicit via shared union; O(1) |
| `activecoordinates(x::_StoredIterate)` | 314 | Explicit read-only `_ActiveIDs` handle; O(1) creation |
| `activevalues(x::_StoredIterate)` | 315 | Explicit writable value view; O(1) creation |
| `isactive(x::_StoredIterate,k)` | 316 | Explicit O(1) reverse-map lookup |
| `activate!(x::_StoredIterate,k)` | 364–368 | Explicit O(1), inserts stored zero |
| `dropzeros!(x::SparseIterate)` | 381–405 | Explicit O(active), already optimized; keep despite SparseArrays namespace |
| `coordinates(x::SparseIterate)` | 599 | Explicit wrapper entry point, common coordinate abstraction with symmetric owner |

**Note:** For matrix iterates, `x[i,j]` uses Julia's generic multidimensional indexing via the explicitly defined linear-index methods, rather than a bespoke two-index specialization.

## B. Coordinate wrapper and active-ID handle: KEEP if coordinates stays

| Exact methods / group | Where | Why |
|:--|:--|:--|
| `SparseCoordinates{T,N} <: AbstractVector{T}` | 571–573 | One-reference live view of all logical entries, vector shape |
| `Base.size(c::SparseCoordinates)` | 578 | `AbstractVector` interface |
| `Base.IndexStyle(::Type{<:SparseCoordinates})` | 579 | Linear indexing |
| `Base.getindex(c::SparseCoordinates,i::Int)` | 580–583 | Calls `_getstored`, O(1) |
| `Base.setindex!(c::SparseCoordinates,v,i::Int)` | 584–588 | Calls `_setstored!`, O(1) |
| `Base.dataids(c::SparseCoordinates)` | 592 | Alias detection; internal Base integration |
| `Base.unaliascopy(c::SparseCoordinates)` | 593 | Independent same-wrapper-type source for aliasing |
| `Base.fill!(c::SparseCoordinates,v)` | 597 | Forwards value fill; crucially not a working-set clear |
| `_coordinates_boundserror` | 575–576 | Internal bounds-exception helper |
| `_ActiveIDs <: AbstractVector{Int}` | 240–242 | Read-only active-slot-ID handle, shared across three iterate types |
| `Base.size(h::_ActiveIDs)` | 243 | Required handle shape |
| `Base.IndexStyle(::Type{_ActiveIDs})` | 244 | Linear indexing |
| `Base.dataids(h::_ActiveIDs)` | 245 | Alias detection for slot IDs |
| `Base.unaliascopy(h::_ActiveIDs)` | 246 | Safe source copy for aliasing |
| `Base.getindex(h::_ActiveIDs,i::Int)` | 247 | Active ID read |
| `Base.setindex!(h::_ActiveIDs,v,i::Int)` | 248–249 | Explicitly rejects writing active IDs |

The `_ActiveIDs` wrapper is separate from `SparseCoordinates`; one is a view of **active IDs**, the other is a view of **all values indexed by logical coordinate**. Both must be included in an exhaustive wrapper audit.

## C. State copying, ordinary writes, and aliasing: KEEP, but review each special source

| Exact methods / group | Where | Role |
|:--|:--|:--|
| `Base.similar(x::SparseIterate)`, `Base.similar(x::SparseIterate,::Type{T})` | 603–604 | Empty same-shaped iterate |
| `Base.copyto!(x::SparseIterate,y::SparseIterate)` | 672–677 | Same-shape **state copy**; other shape of equal length is a value write |
| `Base.copy!(x::SparseIterate,y::SparseIterate)` | 678 | State-copy alias |
| `Base.copyto!(x::SparseIterate,y::AbstractArray)` | 686–689 | Checked generic value write |
| `Base.copyto!(x::SparseIterate{2},y::_AbstractSparseMatrixCSC)` | 692–695 | SparseArrays dispatch ambiguity resolution |
| `Base.copyto!(x::SparseIterate{2},y::Adjoint/Transpose{<:_AbstractSparseMatrixCSC})` | 696–700 | Dispatch ambiguity resolution |
| `Base.copyto!(x::SparseIterate{2},y::_SparseMatrixCSCOrView)` | 704–707 | Dispatch ambiguity resolution, conditional on Julia version |
| `Base.copyto!(x::SparseIterate{2},y::_AdjOrTransSparseVectorOrView)` | 710–713 | Dispatch ambiguity resolution, conditional |
| `Base.copyto!(x::SparseIterate,y::_CHOLMODDense)` | 716–719 | CHOLMOD dispatch ambiguity resolution, conditional |
| `Base.copyto!(x::SparseIterate{T},y::_CHOLMODDense{T})` | 721–724 | CHOLMOD conditional |
| `Base.copyto!(x::SparseIterate{T,2},y::_CHOLMODDense{T})` | 725–728 | CHOLMOD conditional |
| `Base.copyto!(x::SparseIterate{2},y::UniformScaling)` | 732–735 | Matrix value-copy convenience |
| `Base.copyto!(x::SparseIterate{T,2},y::_AbstractQ)` | 740–748 | Correctness-first orthogonal-factor materialization; #797 optional optimization |
| `Base.copyto!(x::SparseIterate,bc::Broadcast.Broadcasted{Nothing})` | 751–755 | Value-write broadcasting, avoids accidental same-kind state copy |
| `Base.copyto!(x::SparseIterate,bc::Broadcast.Broadcasted{<:DefaultArrayStyle{0}})` | 771–775 | Scalar-broadcast specialization |
| `Base.fill!(x::SparseIterate,v)` | 779–793 | Value fill; preserves active zeros |
| `Base.dataids(x::SparseIterate)` | 321–322 | **Explicit required aliasing hook**, includes values, active IDs, reverse map |
| `Base.convert(::Type{<:SparseIterate{T}},::AbstractArray)` for `T <: Complex` | 608 | Explicit guard against non-real iterate element type |
| `Base.copyto!`/`Base.copy!` between `SparseIterate` and `AtomIterate` (both directions) | `src/atom_iterate.jl:416–427` | Linear-order value write across types; cross-type compatibility, not sparse state copy |

Internal implementations serving this group: `_getstored` (325), `_setstored!` (332), `_store!` (345), `_converted_values` and `_convert_may_throw` (426–439), `_copyto_stored!` (443), `_clear!` (466), `_check_iterate_lengths` (681), `_broadcast_loop!` (761), plus the type/aliasing preflight helpers called from shared code. They are **not public API**.

**Inherited, not explicitly defined here:** `copy(x)` (Base uses `similar` and `copyto!`); generic indexing through views/ranges/Cartesian indices, `eachindex`, `axes`, `iterate`, `vec`, `reshape`, `map!`, and reductions. Their existence should not be misreported as dedicated SparseIterate specializations. In particular, `Base.unaliascopy(x::SparseIterate)` is **not** explicitly defined, whereas `Base.unaliascopy(c::SparseCoordinates)` **is**.

## D. SparseArrays names: removal decision and migration

| Exact method | Where | Assessment |
|:--|:--|:--|
| `SparseArrays.issparse(x::SparseIterate)` | 529 | No identified production ProximalSolvers requirement; candidate for removal |
| `SparseArrays.nnz(x::SparseIterate)` | 217 | Alias of `nactive`; **used in production ProximalBase source**, so migrate before removal |
| `SparseArrays.nonzeros(x::SparseIterate)` | 538 | Alias of `activevalues`; no identified ProximalSolvers requirement for this type |
| `SparseArrays.findnz(x::SparseIterate{1})` | 547–550 | Sorted export; no identified ProximalSolvers production usage |
| `SparseArrays.findnz(x::SparseIterate{2})` | 552–560 | Sorted row/column/value export |
| `SparseArrays.dropzeros!(x::SparseIterate)` | 381 | **KEEP**: core active-set compaction, despite namespace |

**Amended policy (2026-11-09):** FC-11 now excludes these four methods from `SparseIterate`; `dropzeros!` remains essential. The table above records historical definitions found on `master` before the removal, rather than prescribing future methods. The other methods' presence or absence should not be inferred from `issparse` alone.

Source-site evidence: `src/sparse_iterate.jl` uses `nnz` in dense exports and matrix kernels; `src/functions/smooth.jl` uses it in a `KronQuadratic` densification path; `src/deprecated.jl` uses it in deprecated products. `nonzeros` on standard `SparseMatrixCSC` appears in several files and **must not be removed** when migrating SparseIterate-only uses.

## E. Sparse-aware numerical primitives: KEEP / BENCHMARK, BUT NOT A GENERIC ARRAY PROMISE

| Exact methods / group | Where | Scope |
|:--|:--|:--|
| `LinearAlgebra.norm(x::SparseIterate,p)` | 795 | Explicit sparse norm dispatch |
| `LinearAlgebra.rmul!`, `lmul!`, `rdiv!`, `ldiv!` | 512–515 | Explicit scalar mutating paths, O(active) when zero maps to zero, otherwise O(length) |
| `*(A::StridedMatrix,x::SparseIterate{1})` | 890 | Sparse matrix-vector product |
| `mul!(out::StridedVector,A::StridedMatrix,x::SparseIterate{1})`, with / without `α,β` | 899–920 | Sparse vector kernel |
| `mul!(out::StridedMatrix,A::StridedMatrix,X::SparseIterate{2})`, with / without `α,β` | 922–960 | Sparse matrix kernel |
| `*(A::_StridedSymmetric,x::SparseIterate{1})`, `*(A::_StridedSymmetric,X::SparseIterate{2})` | 989–996 | Symmetric left-factor specializations |
| `mul!` symmetric left factor, vector/matrix result, with / without `α,β` | 998–1028 | Symmetric left-factor kernels |
| `*(A::TransposedStridedMatrix,x::SparseIterate{1})` | 1034 | Transposed matrix kernel |
| `mul!(out, A::TransposedStridedMatrix,x::SparseIterate{1})`, with / without `α,β` | 1044–1067 | Transposed matrix kernels |
| `dot(v::StridedVector,x::SparseIterate{1})` | 1072–1082 | Sparse vector dot |
| `dot(x::SparseIterate{1},v::StridedVector)` | 1083 | Reversed dot |
| `dot(x::SparseIterate{1},y::SparseIterate{1})` | 1086–1097 | Sparse-sparse dot |
| `dot(a::StridedVector,X::SparseIterate{2},b::StridedVector)` | 1099–1119 | Bilinear matrix dot |
| `*(A::Symmetric{T,<:StridedMatrix{T}},X::SparseIterate{T,2},B::Symmetric{T,<:StridedMatrix{T}})` | 1277 | Three-factor matrix product, narrow type specialization |
| `*(A::Symmetric{T},X::SparseIterate{T,2},B::Symmetric{T})` | 1302 | Three-factor matrix product, broader wrapper specialization |
| `norm_diff(x,y,p)` | `src/utils.jl` | Generic array utility, **not a guaranteed O(active) sparse-only specialization** |

Relevant non-exported computational helpers include `_scaled_dot` (148), `_split_dot` (169), `_stored_maxabs`/`_all_finite` (193–195), `_scale!` and `_scale_all!` (480–510), `_triple` (1122), `_axpy_terms!` (1134), `_add_coefficient!` (1145), `_logical_coordinates` (1190), `_rescue_entry!` (1200), `_rescue_nonfinite!` (1232). These are kernel support, not separately promised public methods.

## F. Display and conversions: OPTIONAL API / KEEP CORRECTNESS

| Exact methods | Where | Notes |
|:--|:--|:--|
| `Vector(x::SparseIterate{1})`, `Matrix(x::SparseIterate{2})`, `Array(x::SparseIterate{1,2})` | 644–667 | Explicit dense snapshots |
| `show(io,x::SparseIterate)` | 801 | Explicit |
| `show(io,MIME"text/plain",x::SparseIterate{1})` | 803 | Explicit |
| `show(io::IOContext,x::SparseIterate{1})` | 813 | Explicit |
| `show(io,MIME"text/plain",X::SparseIterate{2})` | 842 | Explicit |
| `show(io::IOContext,X::SparseIterate{2})` | 852 | Explicit |
| `_show_entries` | 862 | Internal display helper |

## G. Deprecations and algorithm-specific specializations outside the data structure

These are explicit methods accepting `SparseIterate` or sharing dispatch, **not data-structure operations that a new iterate type must provide**:

- `src/deprecated.jl:76,84,88`: deprecated vector-to-matrix and `A_mul_X_mul_B`/`A_mul_X_mul_B_rc` shims; inspect when retiring old APIs.
- `src/interface.jl:452,471`: 4- and 5-argument `prox_coordinate!(::SparseIterate,...)`; mathematical function + iterate oracle, not storage.
- `src/functions/separable.jl:104,161,235,927`: sparse-aware separable value/prox internal kernels.
- `src/functions/sets.jl:115,337,343`: `IndBallL0` NaN count and sparse prox internals.
- `src/functions/smooth.jl:109,270,287,1070`: linear/squared-value helpers and `KronQuadratic` densification.
- `src/functions/matrixsets.jl:69`: `IndPattern(::SparseIterate)` constructor.
- `src/functions/spectral.jl:825`: sparse-matrix trace helper.
- `src/ballconstrained.jl:787,826,926`: nonfinite/non-NaN scans over stored values.
- `src/functions/sets.jl:1324–1326`: union-dispatched stored dot-normal helper.
- `src/composition.jl` and `src/calculus.jl`: references to sparse iterates as permitted output/input types in higher-level function compositions.

This list should be checked against every Julia source file when doing a behavior-changing deletion; it is an explicit-source inventory, not a guarantee that the search index includes every dynamically generated or macro-generated method.

## H. Decisions and validation before the interface is frozen

1. **No essential coordinate-storage method is currently known missing.** Continue to test inherited array behavior, rather than implement redundant overrides.
2. **#694 open:** complete `SparseCoordinates` `AbstractVector` compatibility tests (sortperm, display, indexing) on Julia 1.10 and 1.13. Existing `test/allocations.jl` asserts zero allocations for creating the coordinate wrapper, scalar reads/writes and `fill!`, for Float32/Float64 and vector/matrix shapes.
3. **#697 open:** fix incorrect conversion wording and same-size-copy docs.
4. **#644 closed:** the all-zero `dropzeros!` fast path is implemented; do not label it outstanding.
5. **#797 open, deferred:** orthogonal-factor copy performance investigation; correctness path exists.
6. **#341 open:** separate function-oracle sparse-value optimization, not a missing iterate API.
7. **Four SparseArrays methods:** the maintainer approved removal and FC-11 was amended. Migrate production/test/doc call sites; preserve standard sparse-matrix methods and `dropzeros!`. Validate downstream.
8. **Wrapper policy:** retain `coordinates(x)` for common symmetric/non-symmetric coordinate-descent code; benchmark `x[k]` against `coordinates(x)[k]`. Do not infer a performance regression from wrapper existence.
9. **Run before merge:** Julia-version method-table probes (`methodswith(SparseIterate)`, `methodswith(SparseCoordinates)`, `@which` for inherited/ambiguous dispatch), conformance tests, Linux test:thorough, downstream ProximalSolvers smoke and representative Float32/Float64 benchmark cases.

No dynamic Julia method-table probe, CI job, or benchmark was run as part of this static audit.
