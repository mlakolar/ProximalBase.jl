# Parse Julia syntax, excluding docstrings/comments, for a repeatable source inventory.
# These are syntactic call sites and method definitions, not runtime call counts.
using TOML, SHA
length(ARGS)==3 || error("usage: inventory.jl PB_ROOT PS_ROOT OUTPUT_DIRECTORY")
pb,ps,out=abspath.(ARGS)
mkpath(out)
const surface=Symbol.(split("issmooth isproximable isseparable isconvex iszeropreserving hashessian prox prox! value_and_prox! gradient gradient! value_and_gradient value_and_gradient! value_entry prox_entry prox_coordinate! deriv deriv2 curvature value_and_deriv hessian_vec hessian_vec! quadratic_model quadratic_model! lipschitz coordinate_lipschitz coordinate_lipschitz! kkt_violation restrict gradient_mapping gradient_mapping! lla_weight lla_weights lla_weights! check_parameters entry_parameters check_symmetric_parameters haslmo lmo lmo! lmo_atom reset_cache! opnorm2 samegroups _checkparams"))
const hits=NamedTuple[]
function callname(x)
  x isa Symbol && return x
  x isa Expr || return nothing
  x.head===:. && return x.args[end] isa QuoteNode ? x.args[end].value : nothing
  x.head===:curly && return callname(x.args[1])
  x.head===:where && return callname(x.args[1])
  x.head===:(::) && return callname(x.args[1])
  x.head===:call && return callname(x.args[1])
  nothing
end
function record(expr,kind,repo,path,line)
  name=callname(expr)
  name in surface || return
  push!(hits,(;oracle=string(name),kind,repo,path,line))
end
function scan(x,repo,path,line=1)
  x isa Expr || return
  if x.head===:function
    record(x.args[1],"definition",repo,path,line)
    for a in x.args[2:end]; scan(a,repo,path,line); end
    return
  elseif x.head===:(=) && x.args[1] isa Expr &&
    (x.args[1].head===:call || x.args[1].head===:where)
    record(x.args[1],"definition",repo,path,line)
    scan(x.args[2],repo,path,line)
    return
  elseif x.head===:call
    record(x,"call",repo,path,line)
  end
  for a in x.args
    a isa LineNumberNode ? (line=a.line) : scan(a,repo,path,line)
  end
end
function inventory(root,repo)
  files=sort([joinpath(dir,f) for (dir,_,fs) in walkdir(joinpath(root,"src"))
    for f in fs if endswith(f,".jl")])
  for path in files
    scan(Meta.parseall(read(path,String)),repo,relpath(path,root))
  end
  files
end
pbfiles=inventory(pb,"PB"); psfiles=inventory(ps,"PS")
sha=Dict("PB"=>readchomp(`git -C $pb rev-parse HEAD`),"PS"=>readchomp(`git -C $ps rev-parse HEAD`))
url(h)="https://github.com/mlakolar/Proximal$(h.repo=="PB" ? "Base" : "Solvers").jl/blob/$(sha[h.repo])/$(h.path)#L$(h.line)"
open(joinpath(out,"source-sites.tsv"),"w") do io
  println(io,"oracle\tkind\trepo\tpath\tline\tpinned_url")
  for h in sort(unique(hits);by=h->(h.oracle,h.repo,h.path,h.line,h.kind))
    println(io,join((h.oracle,h.kind,h.repo,h.path,h.line,url(h)),'\t'))
  end
end
open(joinpath(out,"consumer-matrix.tsv"),"w") do io
  println(io,"oracle\tPB_definitions\tPS_calls\tPS_extensions\tPS_workload_calls\tPS_call_sites")
  for name in string.(surface)
    hs=unique(filter(h->h.oracle==name,hits))
    defs=count(h->h.repo=="PB"&&h.kind=="definition",hs)
    calls=filter(h->h.repo=="PS"&&h.kind=="call"&&!endswith(h.path,"precompile.jl"),hs)
    ext=count(h->h.repo=="PS"&&h.kind=="definition",hs)
    work=count(h->h.repo=="PS"&&h.kind=="call"&&endswith(h.path,"precompile.jl"),hs)
    println(io,join((name,defs,length(calls),ext,work,join(url.(calls)," ")),'\t'))
  end
end
open(joinpath(out,"inventory-provenance.toml"),"w") do io
  TOML.print(io,Dict("PB"=>sha["PB"],"PS"=>sha["PS"],
    "PB_files"=>relpath.(pbfiles,pb),"PS_files"=>relpath.(psfiles,ps),
    "julia"=>string(VERSION),"harness_sha256"=>bytes2hex(sha256(read(@__FILE__))));sorted=true)
end
println("Inventoried ",length(pbfiles)," PB and ",length(psfiles)," PS Julia source files; ",
  length(surface)," oracle names; ",length(unique(hits))," syntactic sites.")
