#!/usr/bin/env julia
# Parse frozen blobs without loading ProximalBase or evaluating package code.
using SHA

const BASELINE = "eff21b6d7d33c622313122f46c6ee71693c101ee"
const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const OUT = @__DIR__
git(args...) = read(Cmd(["git", "-C", ROOT, args...]), String)
clean(x) = replace(string(x), '\t' => ' ', '\n' => ' ', '\r' => ' ')
pretty(x) = clean(Base.remove_linenums!(deepcopy(x)))
const rows = NamedTuple[]
const directives = NamedTuple[]
const segments = NamedTuple[]

function symbols!(s, x)
  x isa Symbol && push!(s, string(x))
  x isa Expr && foreach(a -> symbols!(s, a), x.args)
  x isa QuoteNode && x.value isa Symbol && push!(s, string(x.value))
  s
end
symbols(x) = symbols!(Set{String}(), x)
function callname(x)
  x isa Symbol && return string(x)
  x isa GlobalRef && return string(x.mod, ".", x.name)
  x isa QuoteNode && return callname(x.value)
  x isa Expr || return ""
  x.head == :(::) && return "call:" * pretty(last(x.args))
  x.head in (:where, :curly, :<:) && return callname(first(x.args))
  x.head == :call && return callname(first(x.args))
  x.head == :. && return join(filter(!isempty, callname.(x.args)), ".")
  x.head == :$ && return "\$" * callname(first(x.args))
  ""
end
function calls!(s, x)
  if x isa Expr
    x.head == :call && push!(s, callname(first(x.args)))
    foreach(a -> calls!(s, a), x.args)
  end
  s
end
function annotations!(s, x)
  if x isa Expr
    if x.head in (:(::), :<:)
      union!(s, symbols(last(x.args)))
    end
    foreach(a -> annotations!(s, a), x.args)
  end
  s
end
annotations(x) = annotations!(Set{String}(), x)
function substitute(x, bindings)
  x isa Expr || return x
  x.head == :$ && first(x.args) isa Symbol && haskey(bindings, first(x.args)) && return bindings[first(x.args)]
  Expr(x.head, (substitute(a, bindings) for a in x.args)...)
end
function record(path, lo, hi, line, x, kind, sig, context)
  method_sig = sig
  while method_sig isa Expr && method_sig.head in (:where, :(::))
    method_sig = first(method_sig.args)
  end
  name = callname(method_sig)
  kind in ("abstract", "primitive") && (name = callname(x.args[1]))
  kind == "struct" && (name = callname(x.args[2]))
  kind == "binding" && (name = callname(sig))
  refs = symbols(x)
  typed = annotations(sig)
  kind == "struct" && union!(typed, annotations(x))
  id = string(path, ":", line, ":", length(rows) + 1)
  push!(rows, (;id, path, group_start=lo, group_end=hi, line, kind, name,
    signature=pretty(sig), typed=sort!(collect(typed)),
    calls=sort!(filter(!isempty, collect(calls!(Set{String}(), x)))),
    refs=sort!(collect(refs)), context))
end

literal(x) = x isa QuoteNode ? x.value : x isa Expr && x.head == :tuple ? tuple(literal.(x.args)...) : x
function loopbindings(spec, bindings)
  spec isa Expr && spec.head == :(=) || return nothing
  binder, values = spec.args
  values isa Expr && values.head == :tuple || return nothing
  out = Dict{Symbol,Any}[]
  for value in literal(values)
    b = copy(bindings)
    if binder isa Symbol
      b[binder] = value
    elseif binder isa Expr && binder.head == :tuple && value isa Tuple && length(binder.args) == length(value)
      foreach(p -> b[p[1]] = p[2], zip(binder.args, value))
    else
      return nothing
    end
    push!(out, b)
  end
  out
end

function walk(x, path, lo, hi, line=lo, context="", bindings=Dict{Symbol,Any}())
  x isa Expr || return
  if x.head == :macrocall
    macro_name = callname(first(x.args))
    if macro_name in ("Core.@doc", "@doc")
      walk(last(x.args), path, lo, hi, line, context, bindings)
    elseif macro_name == "@eval"
      walk(substitute(last(x.args), bindings), path, lo, hi, line, context * " @eval", bindings)
    elseif macro_name in ("@inline", "@noinline", "@generated", "@static", "Base.@propagate_inbounds", "Base.@assume_effects")
      walk(last(x.args), path, lo, hi, line, context * " " * macro_name, bindings)
    elseif macro_name in ("@deprecate", "Base.@deprecate")
      record(path, lo, hi, line, x, "macro-generated-method", x.args[3], context * " " * macro_name)
    elseif macro_name == "Base.@deprecate_binding"
      record(path, lo, hi, line, x, "binding", x.args[3], context * " " * macro_name)
    else
      push!(directives, (;path, start=lo, stop=hi, kind="macro", expression=pretty(x)))
      # Workloads are hooks, not methods. Do not inventory local assignments inside them.
    end
  elseif x.head == :function
    record(path, lo, hi, line, x, "function", first(x.args), context)
  elseif x.head == :(=) && first(x.args) isa Expr && first(x.args).head in (:call, :where, :(::))
    record(path, lo, hi, line, x, "method", first(x.args), context)
  elseif x.head in (:abstract, :primitive)
    record(path, lo, hi, line, x, string(x.head), first(x.args), context)
  elseif x.head == :struct
    record(path, lo, hi, line, x, "struct", x.args[2], context)
    # Inner constructors are distinct declarations with the type's shared source segment.
    walk(x.args[3], path, lo, hi, line, context * " inner-constructor", bindings)
  elseif x.head == :const
    y = first(x.args)
    record(path, lo, hi, line, x, "binding", y isa Expr && y.head == :(=) ? first(y.args) : y, context)
  elseif x.head == :for
    spec = first(x.args)
    specs = spec isa Expr && spec.head == :block ? filter(a -> a isa Expr, spec.args) : [spec]
    bs = [bindings]
    for s in specs
      expanded = [loopbindings(s, b) for b in bs]
      if any(isnothing, expanded)
        bs = Dict{Symbol,Any}[]
        break
      end
      bs = reduce(vcat, expanded)
    end
    if !isempty(bs)
      for b in bs
        ctx = join(("$k=$(pretty(b[k]))" for k in sort!(collect(keys(b)); by=string)), ",")
        walk(last(x.args), path, lo, hi, line, context * " for " * ctx, b)
      end
    else
      push!(directives, (;path, start=lo, stop=hi, kind="loop-template", expression=pretty(x)))
      walk(last(x.args), path, lo, hi, line, context * " loop-template", bindings)
    end
  elseif x.head in (:toplevel, :block)
    for a in x.args
      a isa LineNumberNode ? (line = lo + Int(a.line) - 1) : walk(a, path, lo, hi, line, context, bindings)
    end
  elseif x.head in (:if, :elseif, :let, :try, :quote, :module)
    for a in x.args
      walk(a, path, lo, hi, line, context * " " * string(x.head), bindings)
    end
  elseif x.head in (:using, :import, :export, :public) || (x.head == :call && callname(first(x.args)) == "include")
    push!(directives, (;path, start=lo, stop=hi, kind=string(x.head), expression=pretty(x)))
  elseif x.head in (:&&, :||)
    push!(directives, (;path, start=lo, stop=hi, kind="conditional-hook", expression=pretty(x)))
  elseif x.head == :(=)
    record(path, lo, hi, line, x, "binding", first(x.args), context)
  else
    push!(directives, (;path, start=lo, stop=hi, kind="top-level-" * string(x.head), expression=pretty(x)))
  end
end

const paths = split(chomp(git("ls-tree", "-r", "--name-only", BASELINE, "src")), '\n')
length(paths) == 52 || error("baseline source count changed")
for path in paths
  source = git("show", "$BASELINE:$path")
  pos = firstindex(source)
  while pos <= lastindex(source)
    ex, next = Meta.parse(source, pos; greedy=true, raise=true, filename=path)
    ex === nothing && break
    lo = count(==('\n'), SubString(source, 1, prevind(source, pos))) + 1
    hi = count(==('\n'), SubString(source, 1, prevind(source, next))) + 1
    push!(segments, (;path, lo, hi, byte_start=pos, byte_end=prevind(source, next), head=string(ex.head)))
    walk(ex, path, lo, hi)
    next > pos || error("parser did not advance")
    pos = next
  end
end

# Disposition policy is separate from AST extraction, so a critic can audit/change it.
include(joinpath(OUT, "inventory_policy.jl"))
function write_tsv(path, header, data)
  open(path, "w") do io
    println(io, "# Frozen baseline $BASELINE; Julia $(VERSION); generated by generate_inventory.jl")
    println(io, join(header, '\t'))
    foreach(r -> println(io, join(clean.(r), '\t')), data)
  end
end
policies = Dict(r.id => disposition(r) for r in rows)
write_tsv(joinpath(OUT, "source-segments.tsv"), ["path", "start_line", "end_line", "byte_start", "byte_end", "ast_head"],
  ((s.path, s.lo, s.hi, s.byte_start, s.byte_end, s.head) for s in segments))
byname = Dict{String,Vector{String}}()
for r in rows
  isempty(r.name) || push!(get!(byname, r.name, String[]), r.id)
  occursin('.', r.name) && push!(get!(byname, last(split(r.name, '.')), String[]), r.id)
end
write_tsv(joinpath(OUT, "declarations.tsv"),
  ["id", "path", "segment_start", "segment_end", "declaration_line", "kind", "name", "signature", "disposition", "reason", "typed_refs", "calls", "symbol_refs", "context"],
  ((r.id, r.path, r.group_start, r.group_end, r.line, r.kind, r.name, r.signature,
    policies[r.id]..., join(r.typed, ';'), join(r.calls, ';'), join(filter(s -> haskey(byname, s), r.refs), ';'), r.context) for r in rows))
write_tsv(joinpath(OUT, "module-hooks.tsv"), ["path", "start", "end", "kind", "expression"],
  ((r.path, r.start, r.stop, r.kind, r.expression) for r in directives))

# Resolve lexical references by declared name, not by Julia run-time dispatch. Each edge names
# all possible same-name targets; signature refs remain distinct from body refs.
write_tsv(joinpath(OUT, "symbol-targets.tsv"), ["symbol", "possible_target_ids"],
  ((name, join(byname[name], ';')) for name in sort!(collect(keys(byname)))))
edges = Tuple[]
for r in rows, ref in r.refs
  targets = get(byname, ref, String[])
  isempty(targets) && continue
  kind = ref in r.typed ? "typed" : ref in r.calls ? "call" : "lexical"
  push!(edges, (r.id, ref, kind))
end
write_tsv(joinpath(OUT, "dependency-edges.tsv"), ["source_id", "symbol", "kind"], edges)

# All external hooks indexed against declaration names, with compact line lists.
hookpaths = split(chomp(git("ls-tree", "-r", "--name-only", BASELINE, "test", "docs", "ext", "benchmark", "examples", "scripts", ".github", "Project.toml")), '\n')
hookrows = Tuple[]
names = Set(filter(n -> occursin(r"^[A-Za-z_][A-Za-z_0-9!]*$", n), collect(keys(byname))))
for path in hookpaths
  any(ext -> endswith(path, ext), (".jl", ".md", ".toml", ".yml", ".yaml")) || continue
  startswith(path, "docs/audit/") && continue
  source = git("show", "$BASELINE:$path")
  refs_by_name = Dict{String,Vector{Int}}()
  for (line, text) in enumerate(split(source, '\n'))
    refs = sort!(collect(intersect(names, Set(m.match for m in eachmatch(r"[A-Za-z_][A-Za-z_0-9!]*", text)))))
    foreach(n -> push!(get!(refs_by_name, n, Int[]), line), refs)
  end
  foreach(n -> push!(hookrows, (path, n, join(refs_by_name[n], ','))), sort!(collect(keys(refs_by_name))))
end
write_tsv(joinpath(OUT, "external-references.tsv"), ["path", "declared_symbol", "lines"], hookrows)

summaries = Tuple[]
for path in paths
  rs = filter(r -> r.path == path, rows)
  counts = [count(r -> first(policies[r.id]) == p, rs) for p in ("retain", "extract", "archive")]
  push!(summaries, (path, length(rs), counts..., bytes2hex(sha256(git("show", "$BASELINE:$path")))))
end
write_tsv(joinpath(OUT, "declaration-summary.tsv"), ["path", "declarations", "retain", "extract", "archive", "sha256"], summaries)
println("Parsed $(length(paths)) files: $(length(rows)) declarations, $(length(edges)) dependency edges, $(length(hookrows)) external-reference lines.")
