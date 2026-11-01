# Family files live in a module and use `using ..Conformance` after init_code
# includes this file. Frozen row interface (assignments may use caller locals):
# @row begin
#   id = "V5.value"; case = "V5"; family = :kernels; oracle = :value
#   types = "Float32 arg/Float64 par"; backend = :cpu
#   rules = "R2"; problem = "ridge after conversion"
#   expected = 0.5f0; observed = f(x)
#   compare = same  # optional function (got, want)::Bool
#   broken = (group = :G1, baseline = Returned(isinf))  # optional; add `volatile = true` for a result
#   that depends on a reduction: then meeting the promise is a pass, and `near(x0)` is a baseline
#   predicate that tolerates last-bit differences
# end
# expected is evaluated before observed, separately from its try. Throws(:Name,
# r"message") expects an exception by type name (pattern defaults to r"").
# Baselines are Returned(predicate) or Threw(:Name[, pattern]). Unexpected
# passes must be promoted; reference/comparison errors never count as broken.
# @notcovered :wrappers :value "Float64" :cpu "R1: unsupported transformation"
# bigref(f; bits=256, T=Float64), or bigref(f, Float32; bits=256), returns
# the stable rounded reference. f must construct its inputs inside the closure.
# Absolute bounds: typed_bound(a, ref, bound) is false unless a and ref have the
# same type, so use it whenever the expected value has an explicit delivery type
# (a Float64 result of a Float32 promise must fail). within_bound is untyped (it
# widens both operands): use it only for a genuine high-precision reference.
# once(f) runs f now and returns a thunk that replays its value or exception, so
# several rows can assert parts of one invocation (a fused call and its buffer).
# Reports: CONFORMANCE_REPORT/<julia>-<platform>-<bounds>-<pkgsha>/<test file name>.jsonl.
module Conformance

using Test
using ProximalBase

export pow2, bigref, combine, same, near, within_ulps, within_bound, typed_bound, once,
       Throws, Returned, Threw, @row, @notcovered

pow2(T, k) = ldexp(one(T), k)

_issubnormal(x) = x isa BigFloat ? false : issubnormal(x)
_class(x) = isnan(x) ? :nan : isinf(x) ? :inf : iszero(x) ? :zero :
            _issubnormal(x) ? :subnormal : :finite
_roundref(T, x::Tuple) = map(v -> _roundref(T, v), x)
_roundref(T, x::AbstractArray) = map(v -> _roundref(T, v), x)
_roundref(T, x) = T(x)
_classes(x::Union{Tuple, AbstractArray}) = map(_classes, x)
_classes(x) = _class(x)

function bigref(f; bits = 256, T = Float64)
  2 <= bits <= 2048 || throw(ArgumentError("bits must be between 2 and 2048"))
  previous = setprecision(BigFloat, bits) do
    _roundref(T, f())
  end
  while bits < 4096
    bits = min(2bits, 4096)
    current = setprecision(BigFloat, bits) do
      _roundref(T, f())
    end
    same(previous, current) && isequal(_classes(previous), _classes(current)) && return current
    previous = current
  end
  error("BigFloat reference did not stabilize by 4096 bits")
end
bigref(f, ::Type{T}; bits = 256) where {T} = bigref(f; bits, T)

function combine(::Type{W}, factors...) where {W}
  P = promote_type(W, map(typeof, factors)...)
  convert(W, foldl(*, map(x -> convert(P, x), factors); init = one(P)))
end

same(a, b) = typeof(a) == typeof(b) && isequal(a, b)
function same(a::AbstractArray, b::AbstractArray)
  eltype(a) == eltype(b) && axes(a) == axes(b) && all(same(x, y) for (x, y) in zip(a, b))
end
function same(a::Tuple, b::Tuple)
  length(a) == length(b) && all(same(x, y) for (x, y) in zip(a, b))
end

function within_ulps(a::T, ref::Real, n::Integer) where {T<:AbstractFloat}
  n >= 0 || throw(ArgumentError("ULP budget must be nonnegative"))
  # The budget is in ulps of the observed type; a reference in another type
  # (round it first, `bigref(f, T)`) would let a narrow result pass a wide one.
  ref isa T || return false
  _class(a) == _class(ref) || return false
  (!isfinite(a) || iszero(a)) && return same(a, T(ref))
  r = T(ref)
  # Adjacent representable values, including binade boundaries and subnormals.
  prevfloat(a, n) <= ref <= nextfloat(a, n) && isfinite(r)
end
function within_bound(a::Real, ref::Real, bound::Real)
  bound >= 0 || throw(ArgumentError("bound must be nonnegative"))
  _class(a) == _class(ref) || return false
  (!isfinite(a) || iszero(a)) && return same(a, typeof(a)(ref))
  # Widen before subtracting so that finite differences cannot overflow.
  abs(BigFloat(a) - BigFloat(ref)) <= bound
end

# The delivery type is part of the promise: no widening, no narrowing.
typed_bound(a, ref, bound::Real) = typeof(a) === typeof(ref) && within_bound(a, ref, bound)

# A baseline predicate for a result that depends on a reduction or a library
# routine, whose last bits differ between Julia versions and platforms: the same
# type and class, and every entry within `n` ulps of the observed value `x0`.
near(x0, n::Integer = 16) = v -> _near(v, x0, n)
_near(v::AbstractArray, x0::AbstractArray, n) = eltype(v) == eltype(x0) && axes(v) == axes(x0) &&
  all(_near(a, b, n) for (a, b) in zip(v, x0))
_near(v::Tuple, x0::Tuple, n) = length(v) == length(x0) && all(_near(a, b, n) for (a, b) in zip(v, x0))
_near(v::AbstractFloat, x0::AbstractFloat, n) = typeof(v) == typeof(x0) && within_ulps(v, x0, n)
_near(v, x0, n) = false

struct Throws
  name::Symbol
  pattern::Regex
end
Throws(name::Symbol) = Throws(name, r"")
Throws(name::Symbol, pattern::AbstractString) = Throws(name, Regex(pattern))
struct Threw
  name::Symbol
  pattern::Regex
end
Threw(name::Symbol) = Threw(name, r"")
Threw(name::Symbol, pattern::AbstractString) = Threw(name, Regex(pattern))
struct Returned{F}
  predicate::F
end
struct Observation
  threw::Bool
  value::Any
end
function observe(f)
  try
    Observation(false, f())
  catch e
    Observation(true, e)
  end
end
function once(f)
  o = observe(f)
  () -> o.threw ? throw(o.value) : o.value
end
_matches(d::Union{Throws, Threw}, o::Observation) =
  o.threw && nameof(typeof(o.value)) == d.name && occursin(d.pattern, sprint(showerror, o.value))
_matches(d::Returned, o::Observation) = !o.threw && d.predicate(o.value)::Bool

const _ids = Dict{String, Set{String}}()
const _reports = Set{Tuple{String, String}}()
const _package = pkgdir(ProximalBase)
_git(args...) = readchomp(Cmd(["git", "-C", _package, args...]))

# Only strings are written as JSON values; this keeps non-finite numbers exact.
function json_string(x)
  io = IOBuffer()
  print(io, '"')
  for c in string(x)
    if c == '"'
      print(io, "\\\"")
    elseif c == '\\'
      print(io, "\\\\")
    elseif c == '\n'
      print(io, "\\n")
    elseif c == '\r'
      print(io, "\\r")
    elseif c == '\t'
      print(io, "\\t")
    elseif Int(c) < 0x20
      print(io, "\\u", string(Int(c); base = 16, pad = 4))
    else
      print(io, c)
    end
  end
  print(io, '"')
  String(take!(io))
end
json_object(fields) = "{" * join((json_string(k) * ":" * json_string(v) for (k, v) in pairs(fields)), ",") * "}"
_display(x) = sprint(show, x; context = :limit => false)
_display(o::Observation) = o.threw ? sprint(showerror, o.value) : _display(o.value)

function report(source, row)
  root = get(ENV, "CONFORMANCE_REPORT", "")
  isempty(root) && return nothing
  sha = get(ENV, "CONFORMANCE_PKG_SHA", "")
  isempty(sha) && (sha = _git("rev-parse", "HEAD"))
  bounds = string(Base.JLOptions().check_bounds)
  platform = Sys.MACHINE * "-" * string(Sys.KERNEL)
  key = join((string(VERSION), platform, bounds, sha), "-")
  dir = joinpath(root, key)
  mkpath(dir)
  family = string(row.family)
  occursin(r"^[A-Za-z0-9_]+$", family) || error("invalid report family: $family")
  stem = first(splitext(basename(source)))
  occursin(r"^[A-Za-z0-9_]+$", stem) || error("invalid report source name: $stem")
  path = joinpath(dir, stem * ".jsonl")
  identity = (abspath(source), abspath(dir))
  if !(identity in _reports)
    header = (; kind = "header", julia = string(VERSION), platform, bounds,
              pkgsha = sha, harness = _git("hash-object", @__FILE__))
    open(path, "w") do io
      println(io, json_object(header))
    end
    push!(_reports, identity)
  end
  open(path, "a") do io
    println(io, json_object(row))
  end
  nothing
end

function reserve_id(source, id)
  ids = get!(() -> Set{String}(), _ids, source)
  id in ids && error("duplicate conformance row id $id in $source")
  push!(ids, id)
end

function runrow(source, meta, expected_fn, observed_fn, compare, broken)
  reserve_id(source, meta.id)
  if broken !== nothing
    broken.group in (Symbol("G", i) for i in 1:12) || error("broken group must be G1 to G12")
    broken.baseline isa Union{Returned, Threw} || error("invalid baseline descriptor")
    hasproperty(broken, :volatile) && !(broken.volatile isa Bool) && error("volatile must be true or false")
  end
  volatile = broken !== nothing && hasproperty(broken, :volatile) && broken.volatile === true
  expected = observe(expected_fn)
  # A bad reference must fail even if the observation would match a baseline.
  got = expected.threw ? Observation(true, ErrorException("reference failed")) : observe(observed_fn)
  verdict = :error
  detail = expected.threw ? "harness error: reference threw $(_display(expected))" :
    "harness error: expected $(_display(expected.value)), observed $(_display(got))"
  if !expected.threw
    comparison = observe(() -> expected.value isa Throws ?
      _matches(expected.value, got) : (!got.threw && compare(got.value, expected.value)::Bool))
    if comparison.threw
      detail = "harness error: comparison threw $(_display(comparison))"
    else
      if comparison.value
        # A volatile row's outcome depends on reduction rounding, so meeting the
        # promise on some Julia version or platform is a pass, not a promotion.
        verdict = broken === nothing || volatile ? :pass : :unexpected_pass
        detail = broken === nothing || volatile ? "pass" : "unexpected pass: promote the row"
      elseif broken !== nothing
        baseline = observe(() -> _matches(broken.baseline, got))
        if baseline.threw
          detail = "harness error: baseline predicate threw $(_display(baseline))"
        elseif baseline.value
          verdict = :broken
          detail = "named baseline"
        end
      end
    end
  end
  ok = verdict == :pass
  known = verdict == :broken
  @testset "$(meta.id): $detail" begin
    if known
      @test_broken ok
    else
      @test ok
    end
  end
  if ok || known
    report(source, merge(meta, (; group = broken === nothing ? "" : string(broken.group),
      status = ok ? "pass" : "broken", expected = _display(expected.value), observed = _display(got))))
  end
  verdict
end

macro row(block)
  block isa Expr && block.head == :block || error("@row requires an assignment block")
  fields = Dict{Symbol, Any}()
  for ex in block.args
    ex isa LineNumberNode && continue
    ex isa Expr && ex.head == :(=) && ex.args[1] isa Symbol || error("@row accepts only assignments")
    key = ex.args[1]
    haskey(fields, key) && error("duplicate @row field $key")
    fields[key] = ex.args[2]
  end
  required = (:id, :case, :family, :oracle, :types, :backend, :rules, :problem, :expected, :observed)
  all(k -> haskey(fields, k), required) || error("missing required @row field")
  all(k -> k in (required..., :compare, :broken), keys(fields)) || error("unknown @row field")
  metadata = Expr(:tuple, (Expr(:(=), k, fields[k]) for k in required[1:8])...)
  esc(:($runrow($(string(__source__.file)), $metadata,
    () -> $(fields[:expected]), () -> $(fields[:observed]),
    $(get(fields, :compare, same)), $(get(fields, :broken, nothing)))))
end

function notcovered(source, family, oracle, types, backend, reason)
  isempty(strip(reason)) && error("not-covered records need a contract reason")
  report(source, (; id = "", case = "", family, oracle, types, backend, rules = "",
    problem = reason, group = "", status = "notcovered", expected = "", observed = ""))
  nothing
end
macro notcovered(family, oracle, types, backend, reason)
  esc(:($notcovered($(string(__source__.file)), $family, $oracle, $types, $backend, $reason)))
end

end
