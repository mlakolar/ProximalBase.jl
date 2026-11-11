# Historical benchmark adapter ONLY: run against immutable pre-#844 PB and the
# final PB candidate with the same driver. Retired names below are historical
# lookups, not compatibility methods. No ProximalSolvers is imported or changed.
using ProximalBase, BenchmarkTools, LinearAlgebra, Statistics, SHA
const PB=ProximalBase
const modern=isdefined(PB,:prox_block!)
const renamed=isdefined(PB,:BlockIterate)
const Iterate=getfield(PB,renamed ? :BlockIterate : :AtomIterate)
const groupview=getfield(PB,renamed ? :blockview : :atom)
const oldstep=modern ? nothing : getfield(PB,:prox_coordinate!)
Base.JLOptions().check_bounds == 2 || error("use --check-bounds=no")
BLAS.set_num_threads(1)

struct HistoricalProbe{F,R}
    id::String
    operation::F
    result::R
end
const probes=Any[]
function addcases(::Type{T},n,zero_trial) where T
    gs=[collect(1:n),collect(n+1:2n)]
    u=zero_trial ? fill(T(.01),n) : T[sin(i)+2 for i in 1:n]
    for f in (GroupNormL2(T(.3),gs,ones(T,2)),GroupNormLinf(T(.3),gs,ones(T,2)),
              SparseGroupNormL2(T(.1),T(.3),gs,ones(T,2)))
        x,out=Iterate(T,2n,gs),similar(u)
        op=modern ? ()->(PB.prox_block!(out,f,u,1,T(.7));copyto!(groupview(x,1),out);nothing) :
                    ()->(oldstep(x,f,1,u,T(.7));nothing)
        id="group_write/$T/$(nameof(typeof(f)))/n$n/zero$zero_trial"
        observe=()->vcat(Vector(x),T.(collect(activecoordinates(x))))
        push!(probes,HistoricalProbe(id,op,observe))
    end
end
for T in (Float32,Float64),n in (5,8,32,256),zero_trial in (false,true)
    addcases(T,n,zero_trial)
end
case_filter=get(ENV,"PB_BLOCK_FILTER","")
isempty(case_filter) || filter!(p->occursin(Regex(case_filter),p.id),probes)
isempty(probes) && error("empty historical probe selection")
function warm!(p::HistoricalProbe)
    p.operation();p.operation();p.operation()
    string(p.id,'\t',bytes2hex(sha256(reinterpret(UInt8,p.result()))))
end
function measure(io,p::HistoricalProbe)
    op=p.operation
    trial=run(@benchmarkable($op());evals=1,
        samples=parse(Int,get(ENV,"PB_BLOCK_SAMPLES","500")),
        seconds=parse(Float64,get(ENV,"PB_BLOCK_SECONDS",".02")),gctrial=false,gcsample=false)
    t=trial.times
    println(io,p.id,'\t',median(t),'\t',minimum(t),'\t',quantile(t,.25),'\t',quantile(t,.75),
        '\t',trial.memory,'\t',trial.allocs,'\t',length(t))
end
function measure_all(path)
    GC.gc()
    open(path,"w") do io
        println(io,"case\tmedian_ns\tminimum_ns\tp25_ns\tp75_ns\tbytes\tallocations\tsamples")
        foreach(p->measure(io,p),probes)
    end
    map(warm!,probes)==metadata || error("historical fixture changed")
end
println(stderr,"Julia=",VERSION," PB=",pathof(PB)," historical=",!modern,
    " threads=",Threads.nthreads()," BLAS=",BLAS.get_num_threads()," cases=",length(probes))
metadata=map(warm!,probes)
metadata_path=get(ENV,"PB_BLOCK_METADATA","")
isempty(metadata_path) || write(metadata_path,join(metadata,'\n')*"\n")
if "--server" in ARGS
    println("READY ",length(probes));flush(stdout)
    for path in eachline(stdin)
        path=="STOP" && break
        measure_all(path)
        println("DONE ",path);flush(stdout)
    end
else
    length(ARGS)==1 || error("supply an output path or --server")
    measure_all(only(ARGS))
end
