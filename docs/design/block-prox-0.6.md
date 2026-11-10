# Pure block proximal interface (0.6, #844)

The implementation instruction for #844 approves `prox_block` / `prox_block!`.
The maintainer's 2026-11-10 ruling also requires the PB-only `BlockIterate`,
`blockview` and `blockindices` storage rename. ProximalSolvers migration is
separate work under PS #238; do not edit its code, pin or PR here.
This record supersedes the storage-aware proximal requirements of the iterate
family and symmetric-owner contracts; their storage and KKT contracts remain.

## Mathematical contract

| Form | Input and index | Output |
| --- | --- | --- |
| `prox_block(g,u,k,γ)` for separable `g` | Scalar trial `u`; underlying array's linear entry index `k` | Scalar proximal point |
| `prox_block(g,u,k,γ)` for grouped `g` | Compact vector trial `u`; penalty group index `k` | Allocated floating vector |
| `prox_block!(out,g,u,k,γ)` | Compact vector buffers; penalty group index `k` | The identical `out` object |

Every form computes `argmin_z g_k(z) + norm(z-u)^2/(2γ)` with the existing
0.6 endpoint, precision, NaN/Inf and tie policies. The caller constructs the
trial using its loss gradient and curvature. A penalty's index is never an
active slot or an implicit iterate coordinate ID. For a symmetric owner,
`(r,c) = tril_cartesian(p,k)` maps to entry `(c-1)p+r`. The caller validates
pair symmetry with `check_symmetric_parameters` before scalar block updates.

Scalar dispatch is the renamed scalar kernel, including weighted/unweighted
`NormL1`, `ElasticNet`, all built-in separable nonconvex functions and sets,
`Scaled`, `Tilt`, `Translate`, `Regularize`, separable `Precompose`, separable
`Conjugate`, `MoreauEnvelope`, `SqrDistL2` and supported constrained wrappers.
Custom `SeparableFunction` authors rename `prox_entry` to `prox_block` and
keep `value_entry`. Full `prox!` invokes the same scalar kernel.

Compact group dispatch covers `GroupNormL2`, `GroupNormLinf`,
`SparseGroupNormL2`, `GroupMCP`, `GroupSCAD`, their `Scaled` and `Regularize`
wrappers, including nested versions. The group kernels are shared with full
`prox!`: `_shrink_group!`, `_sparse_shrink_group!`, `_soft_group_only!` and
`_radial_shrink!`. Only the selected group's parameters are accessed; no
construction or scan of all groups is needed. `GroupNormLinf` retains its
penalty-owned reusable threshold scratch. Concurrent calls sharing such a
penalty retain the existing full-oracle scratch restrictions.

A separable penalty does not specify a solver's arbitrary group partition:
loop over that partition's global entry indices and call scalar `prox_block`.
SLOPE, cardinality constraints across groups, and other coupled functions do
not acquire an artificial block decomposition. Array-scaled group penalties,
heterogeneous diagonal metrics and zero-curvature affine minimization are
outside this migration. Existing solver branches for zero curvature stay in
ProximalSolvers; `γ = Inf` is not redefined as an affine minimizer.

## Buffers, validation and storage

Group buffers are one-based real vectors with the selected group's length,
in precisely that group's entry order. Exact aliasing `out === u` is supported.
Other overlap is rejected before writes. **Caller precondition:** the output
must be independent of iterate storage and model caches. The mathematical
oracle does not infer ownership from the buffer's representation. In
particular, `activevalues(x)` can be a view of a plain vector while still
aliasing an iterate; `coordinates(x)` and `blockview(x,k)` are also storage
views. None is a valid independent output. Use an ordinary scratch vector or
the allocating form, which returns independent storage. Passing aliased owner
storage violates this precondition and can mutate iterate values or caches;
there is no representation-specific ownership guard.
Group bounds, lengths, representation, real element types, wrapper-center
extent and radial scalar parameter validity are checked before writes. A
failed element access, conversion or custom numerical implementation can leave
a partial output, as with array writes. With independent output, no iterate or model cache has been
mutated by such a failure. The allocating form never returns
its input. The mixed-input/output model precision follows full-array prox;
solvers that need their previous storage precision convert the trial buffer
to the iterate's element type before invoking the oracle.

The solver checks group-layout equality through `samegroups`, computes the
proximal point, then writes `x[i]`, `coordinates(x)[k]`, or
`copyto!(blockview(x,k),out)`. These public writers perform O(1) activation lookup
per scalar write and do not scan other groups. A stored zero stays active;
a zero result on an absent coordinate does not activate it. Solvers retain
their existing exact movement/NaN decisions before updating residuals.
Performance of these public writes is measured with the companion migration;
this does not redesign the generalized storage API of #787.

The PB #787 follow-up specializes complete owned-block view copies for
ordinary matching floating buffers. It checks the block once, snapshots an
overlapping source before writing, and updates active storage without per-entry
activation. An inactive block activates once only for a converted nonzero;
checked/fallible sources retain scalar value-write failure behavior. This
optimization belongs to storage, not to the mathematical oracle.
A specialized validated `GroupNormL2` boundary exposes its existing checks
to small-loop optimization. The generic boundary and wrapper forwarding keep
their original route. No check, wrapper dispatch or public extension method is
removed; nested wrappers retain public validation.
The separate size-5, 24-group diagnostic uses individual entry writes, so its
result cannot be attributed to the bulk writer. PB-only timings do not replace
the separately scoped PS whole-solver integration/performance gate.

## Removal and downstream migration

`prox_entry`, both public `prox_coordinate!` signatures, and the older
`cdprox!` alias are absent. There are no runtime compatibility shims.
The test-only `take_block_step!` helper assembles public math and public writes
for existing mathematical/storage integration cases; it is not a package API.
Tests of the old implementation's private staging/failure state are replaced
with pure-buffer failure tests and explicit writer invariant tests.

The exhaustive pinned source/documentation inventory is
[block-prox-844-call-sites.tsv](../audit/block-prox-844-call-sites.tsv).
It includes wrappers, numerical helper calls, nonconvex functions, extension
hooks, tests, GPU-related tests, examples, benchmark harness and docs in both
repositories. Historical audit/baseline records retain the names they measured.
No GPU extension defined either retired oracle; scalar device code now refers
to the renamed kernels. Compact group calls retain CPU indexing semantics;
this does not add a device group oracle.

The companion PS migration updates scalar CD, Newton and stopping calls;
both generic `src/cd/groups.jl` calls; and specialized LS, quadratic and
multiresponse `GroupNormL2` calls. It deletes `_groupshrink`. The KKT zero-group
skip, lazy curvature, cache-fused coefficient writes, and stopping calculation
stay in PS. The exact tested source pair, commands, companion patch and
remaining #230 / #527 integration requirements are recorded with validation.

## Review and release gates

This is a breaking 0.6 change. Review the public API inventory, local numerical
and whole-solver results, companion patch and dependency pin together. Local
runs do not substitute for green PB/PS CI or complete the separate symmetric
owner migration. Do not close #844 or merge before those coordinated gates.
The companion is [PS #237](https://github.com/mlakolar/ProximalSolvers.jl/pull/237).
Matching its baseline failures is regression evidence and does not meet the
green required-CI gate. The writer-performance gate remains pending by the
maintainer's 2026-11-10 instruction; the measured regression is not approved.
