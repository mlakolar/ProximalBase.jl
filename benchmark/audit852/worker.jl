using BenchmarkTools, LinearAlgebra, Statistics, SHA, TOML, Dates
VERSION==v"1.13.3" || error("audit requires Julia 1.13.3")
Sys.islinux() || error("Linux required")
Threads.nthreads()==1 || error("one Julia thread required")
BLAS.set_num_threads(1)
root,label,processdir=ARGS
mkpath(processdir)
load_seconds=@elapsed Base.include(Main,joinpath(root,"src","ProximalBase.jl"))
const PB=Main.ProximalBase
include("adapters.jl")
include("cases.jl")
function digesttree(dir)
    paths=sort([joinpath(r,f) for (r,ds,fs) in walkdir(dir) for f in fs if any(ext->endswith(f,ext),(".jl",".py",".toml"))])
    bytes2hex(sha256(join([relpath(p,dir)*":"*bytes2hex(sha256(read(p))) for p in paths],"\n")))
end
function calibration()
    t=@benchmark noop() evals=1 samples=1000 seconds=.02
    Dict("median_ns"=>median(t).time,"min_ns"=>minimum(t).time,"times_ns"=>t.times)
end
meta=Dict("label"=>label,"source_root"=>root,"source_digest"=>digesttree(joinpath(root,"src")),
 "julia"=>string(VERSION),"julia_binary"=>joinpath(Sys.BINDIR,Base.julia_exename()),
 "binary_sha256"=>bytes2hex(sha256(read(joinpath(Sys.BINDIR,Base.julia_exename())))),
 "threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),"blas"=>string(BLAS.get_config()),
 "bounds"=>Base.JLOptions().check_bounds,"opt_level"=>Base.JLOptions().opt_level,
 "cpu_name"=>Sys.CPU_NAME,"kernel"=>string(Sys.KERNEL),"machine"=>Sys.MACHINE,
 "source_load_seconds"=>load_seconds,"utc"=>string(now(UTC)),
 "pid"=>getpid(),"harness_digest"=>digesttree(@__DIR__),"calibration_start"=>calibration())
open(joinpath(processdir,"provenance.toml"),"w") do io;TOML.print(io,meta);end
println("READY");flush(stdout)
function measure(p)
    op=p.op;prep=p.setup
    # Compile both operation and the measurement wrapper before sampling.
    warm=@elapsed begin
        for _ in 1:3;prep();op();end
        run(@benchmarkable $op() setup=($prep()) evals=1 samples=3 seconds=.001)
    end
    b=@benchmarkable $op() setup=($prep()) evals=1 samples=80 seconds=.008
    t=run(b);t,warm
end
function command(line)
    fields=split(line,'\t');key,fam,dt,ns,ss,bs,layout,oplist,seeds,mode,outfile=fields
    n=parse(Int,ns);s=parse(Int,ss);b=parse(Int,bs);seed=parse(Int,seeds)
    T=dt=="Float32" ? Float32 : Float64
    ops=String.(split(oplist,','));rows=String[]
    mkpath(dirname(outfile))
    # Original public API cannot represent these layouts/precision/active groups.
    incompatible=ORIGINAL && fam=="block" && (T==Float32 || layout!="contiguous")
    if incompatible
        open(outfile,"w") do io
            println(io,"case,status,median_ns,min_ns,p90_ns,bytes,allocs,samples,work,memory_bytes,warm_seconds,error")
            for name in ops;println(io,key,'/',name,",incompatible,,,,,,,,,,original block public constructor supports contiguous Float64 only");end
        end
        return
    end
    # Hold allocation-diverse padding throughout each fixture/trial. Same padding
    # schedule across source processes; actual pointers are separately recorded.
    padding=[zeros(UInt8,mod(seed*131+j*4093,65536)+1) for j in 1:mod(seed,13)]
    build_seconds=@elapsed cases,src=fam=="block" ? block_cases(T,n,s,b,layout,ops,seed) : scalar_cases(fam,T,n,s,ops,seed)
    ptrs=if fam=="block";Dict("storage"=>UInt(pointer(src.storage)))
    else;Dict("values"=>UInt(pointer(src.nzval)),"ids"=>UInt(pointer(src.nzval2ind)),"map"=>UInt(pointer(src.ind2nzval)))
    end
    fixture=Dict("seed"=>seed,"build_seconds"=>build_seconds,"pointers"=>ptrs,"held_padding_bytes"=>sum(length,padding;init=0),"memory_bytes"=>Base.summarysize(src))
    open(outfile*".fixture.toml","w") do io;TOML.print(io,fixture);end
    open(outfile,"w") do io
        println(io,"case,status,median_ns,min_ns,p90_ns,bytes,allocs,samples,work,memory_bytes,warm_seconds,error")
        for p in cases
            id=key*"/"*p.id
            Core.Box in fieldtypes(typeof(p.op)) && error("boxed benchmark callback: $id")
            if label=="historical05" && p.id=="clear"
                println(io,id,",incompatible,,,,,,,,,,historical fill zero removes active coordinates whereas original/current retain them")
                continue
            end
            if fam=="sparse" && s==n && p.id in ("activate_batch","cd_expand")
                println(io,id,",incompatible,,,,,,,,,,full active set has no inactive coordinates for expansion")
                continue
            end
            if ORIGINAL && fam=="block" && p.id in ("activate_batch","dropzeros")
                println(io,id,",incompatible,,,,,,,,,,original block has no active group metadata")
                continue
            end
            try
                # Validate after one operation and again after a restored repeat.
                p.setup();result=p.op();p.validate(result)
                p.setup();result=p.op();p.validate(result)
                if mode=="check"
                    println(io,id,",pass,,,,,,,",p.work,',',Base.summarysize(src),",,")
                else
                    trial,warm=measure(p);e=median(trial)
                    println(io,id,",pass,",e.time,',',minimum(trial).time,',',quantile(trial.times,.9),',',e.memory,',',e.allocs,',',length(trial.times),',',p.work,',',Base.summarysize(src),',',warm,',')
                    open(outfile*"."*p.id*".samples.csv","w") do raw
                        println(raw,"time_ns,gc_ns")
                        for (t,gc) in zip(trial.times,trial.gctimes);println(raw,t,',',gc);end
                    end
                end
            catch e
                msg=replace(sprint(showerror,e),'\n'=>' ','\r'=>' ',','=>';')
                println(io,id,",failure,,,,,,,,,,",msg)
                open(outfile*"."*p.id*".failure.txt","w") do err
                    showerror(err,e,catch_backtrace())
                end
            end
            flush(io)
        end
    end
    GC.@preserve padding nothing
    GC.gc()
end
for line in eachline(stdin)
    line=="QUIT" && break
    try
        command(line);println("DONE")
    catch e
        println(stderr,"fixture failure: ",line)
        showerror(stderr,e,catch_backtrace());println(stderr);flush(stderr)
        println("ERROR")
    end
    flush(stdout)
end
meta["calibration_end"]=calibration()
open(joinpath(processdir,"provenance.toml"),"w") do io;TOML.print(io,meta);end
