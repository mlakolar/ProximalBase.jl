using BenchmarkTools, LinearAlgebra, Random, Statistics, SHA, Dates
BLAS.set_num_threads(1)
Threads.nthreads()==1 || error("one Julia thread required")
baseline=abspath(get(ENV,"PB_BASELINE","/tmp/pb849-baseline"))
candidate=abspath(get(ENV,"PB_CANDIDATE","/tmp/pb849-minimal"))
module Baseline end
module Candidate end
module BaselineControl end
Base.include(Baseline,joinpath(baseline,"src","ProximalBase.jl"))
Base.include(Candidate,joinpath(candidate,"src","ProximalBase.jl"))
Base.include(BaselineControl,joinpath(baseline,"src","ProximalBase.jl"))
module SuiteA
    const PB=Main.Baseline.ProximalBase
    include("probes.jl")
end
module SuiteB
    const PB=Main.Candidate.ProximalBase
    include("probes.jl")
end
module SuiteControl
    const PB=Main.BaselineControl.ProximalBase
    include("probes.jl")
end
const C=vcat(SuiteControl.makeprobes(Float32),SuiteControl.makeprobes(Float64))
const A=vcat(SuiteA.makeprobes(Float32),SuiteA.makeprobes(Float64))
const B=vcat(SuiteB.makeprobes(Float32),SuiteB.makeprobes(Float64))
case_filter=get(ENV,"PB_FILTER","")
if !isempty(case_filter)
    selection=Regex(case_filter)
    for suite in (A,B,C)
        filter!(p->occursin(selection,p.id),suite)
    end
end
@assert getfield.(A,:id)==getfield.(B,:id)==getfield.(C,:id)
function almosteq(a,b)
    if a isa Tuple
        return all(almosteq(x,y) for (x,y) in zip(a,b))
    end
    isequal(a,b) || isapprox(a,b;rtol=2e-5,atol=1e-6,nans=true)
end
for (a,b,c) in zip(A,B,C)
    a.setup();b.setup();c.setup();a.op();b.op();c.op()
    almosteq(a.check(),b.check()) || error("differential mismatch: $(a.id)")
    almosteq(a.check(),c.check()) || error("A/A differential mismatch: $(a.id)")
end
println("Cross-implementation differential: $(length(A)) probes pass")
flush(stdout)
get(ENV,"PB_CHECK_ONLY","0")=="1" && exit()
function bench(p)
    op=p.op;prep=p.setup
    b=@benchmarkable $op() setup=($prep()) evals=1 samples=180 seconds=.04
    for _ in 1:3
        prep();op()
    end
    run(b)
end
outdir=get(ENV,"PB_OUTPUT","/tmp/pb849-performance/results")
mkpath(outdir)
open(joinpath(outdir,"provenance.txt"),"w") do io
    println(io,"UTC: ",now(UTC),"\nJulia: ",VERSION,"\nThreads: ",Threads.nthreads(),"\nBLAS threads: ",BLAS.get_num_threads(),"\nBLAS: ",BLAS.get_config(),"\nBounds option: ",Base.JLOptions().check_bounds,"\nBaseline: ",baseline,"\nCandidate: ",candidate)
    println(io,"Archived full-post848 reference commit: eff21b6d7d33c622313122f46c6ee71693c101ee\nAnnotated baseline tag object: 5239e1043a370f999753710e958fb2742aa786ac")
    println(io,"CPU: ",Sys.CPU_NAME,"\nKernel: ",Sys.KERNEL,"\nMachine: ",Sys.MACHINE,"\nJulia binary SHA256: ",bytes2hex(sha256(read(joinpath(Sys.BINDIR,Base.julia_exename())))))
    for file in ("Project.toml","Manifest.toml","run.jl","probes.jl","differential.jl")
        println(io,file," SHA256: ",bytes2hex(sha256(read(joinpath(@__DIR__,file)))))
    end
    for (label,root) in (("baseline",baseline),("candidate",candidate))
        files=sort(filter(p->endswith(p,".jl"),[joinpath(r,f) for (r,ds,fs) in walkdir(joinpath(root,"src")) for f in fs]))
        println(io,label," source digest: ",bytes2hex(sha256(join([relpath(p,root)*":"*bytes2hex(sha256(read(p))) for p in files],"\n"))))
    end
end
# Compile every timed specialization before recording any samples.
foreach(bench,A);foreach(bench,B);foreach(bench,C)
get(ENV,"PB_WARM_ONLY","0")=="1" && exit()
rounds=parse(Int,get(ENV,"PB_ROUNDS","11"))
open(joinpath(outdir,"raw.csv"),"w") do io
    println(io,"case,round,order,side,median_ns,min_ns,memory,allocs")
    for round in 1:rounds
        ids=isodd(round) ? eachindex(A) : reverse(eachindex(A))
        for i in ids
            # Independently compiled baseline module is the A/A control.
            order=isodd(round) ? (("A",A[i]),("B",B[i]),("Acontrol",C[i])) : (("Acontrol",C[i]),("B",B[i]),("A",A[i]))
            for (ordinal,(label,p)) in enumerate(order)
                trial=bench(p)
                e=median(trial)
                println(io,p.id,',',round,',',ordinal,',',label,',',e.time,',',minimum(trial).time,',',e.memory,',',e.allocs)
            end
        end
        flush(io)
        println("Completed round ",round,"/",rounds);flush(stdout)
    end
end
