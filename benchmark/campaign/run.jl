using BenchmarkTools, ProximalBase, LinearAlgebra, Random, Statistics, SparseArrays
using Profile, TOML, Printf
include("cases.jl")

clean(s) = replace(string(s), '\t'=>' ', '\n'=>' ', '\r'=>' ')
line(io, xs...) = println(io,join(clean.(xs),'\t'))

function allocated_call(op::F, prep::S) where {F,S}
    prep()
    @allocated op()
end

function batch!(op::F, n) where F
    for _ in 1:n
        op()
    end
end

function profile_cpu!(io, c::PerfCase, seconds, ns)
    seconds <= 0 && return
    op=c.operation; prep=c.prepare
    batch=clamp(round(Int,1e5/max(ns,1)),1,1000)
    traceop=()->(prep();op())
    batch!(traceop,batch)
    Profile.clear()
    deadline=time_ns()+round(UInt64,seconds*1e9)
    println(io,"\nCASE\t",c.meta.id,"\nCPU trace includes prepare callback; benchmark timings exclude it.")
    Profile.@profile while time_ns()<deadline
        batch!(traceop,batch)
    end
    Profile.print(io;format=:flat,C=true,sortedby=:count,mincount=2)
    Profile.print(io;format=:tree,C=true,maxdepth=22,mincount=2)
    flush(io)
end

function profile_alloc!(io, c::PerfCase)
    op=c.operation
    c.prepare(); op()
    Profile.Allocs.clear()
    c.prepare()
    Profile.Allocs.@profile sample_rate=1.0 op()
    results=Profile.Allocs.fetch()
    for allocation in results.allocs
        frames=join((string(f.file,":",f.line,":",f.func) for f in allocation.stacktrace)," > ")
        line(io,c.meta.id,allocation.size,allocation.type,frames)
    end
    Profile.Allocs.clear()
    flush(io)
end

function measure!(c::PerfCase, metrics, raw, allocations, profiles, allocation_profiles,
                  rep, samples, seconds, profile_seconds)
    op,prep=c.operation,c.prepare
    # JIT work and correctness checks are outside measured samples.
    c.validate()
    for _ in 1:3
        prep(); op()
    end
    allocated_call(op,prep); allocated_call(op,prep)
    bytes=[allocated_call(op,prep) for _ in 1:10]
    b=@benchmarkable $op() setup=($prep()) evals=$(c.evals) samples=$samples seconds=$seconds
    trial=run(b)
    times=trial.times
    isempty(times) && error("empty benchmark trial")
    gc_share=sum(trial.gctimes)/max(sum(times),1)
    m=c.meta
    line(metrics,m.id,m.family,m.name,m.dtype,m.shape,m.layout,m.phase,m.work,m.zero_alloc,
         rep,length(times),c.evals,mean(times),median(times),std(times;corrected=false),
         minimum(times),maximum(times),quantile(times,0.95),gc_share,
         mean(bytes),minimum(bytes),maximum(bytes),trial.memory,trial.allocs)
    for (i,t) in enumerate(times)
        line(raw,m.id,rep,i,t,trial.gctimes[i])
    end
    for (i,v) in enumerate(bytes)
        line(allocations,m.id,rep,i,v)
    end
    flush(metrics); flush(raw); flush(allocations)
    if c.profile
        profile_cpu!(profiles,c,profile_seconds,median(times))
    end
    if maximum(bytes)>0 && m.zero_alloc
        profile_alloc!(allocation_profiles,c)
    end
end

function main()
    rep=parse(Int,get(ENV,"PB_REPETITION","1"))
    output=get(ENV,"PB_OUTPUT",joinpath(@__DIR__,"results"))
    mode=get(ENV,"PB_MODE","full")
    samples=parse(Int,get(ENV,"PB_SAMPLES",mode=="canary" ? "3" : "40"))
    seconds=parse(Float64,get(ENV,"PB_SECONDS",mode=="canary" ? "0.01" : "0.1"))
    profile_seconds=parse(Float64,get(ENV,"PB_PROFILE_SECONDS",mode=="canary" ? "0.015" : "0.2"))
    mkpath(output)
    prefix=joinpath(output,"julia-$(VERSION)-rep$(lpad(rep,2,'0'))")
    any(isfile(prefix*suffix) for suffix in ("-metrics.tsv","-errors.tsv")) &&
        error("Refusing to overwrite a repetition; use a fresh output directory")
    BLAS.set_num_threads(1)
    Profile.init(n=2_000_000,delay=0.001)
    started=time()
    cases=build_cases(rep)
    open(prefix*"-catalog.tsv","w") do io
        line(io,"id","family","dtype","shape","layout","phase","zero_alloc")
        for c in cases
            m=c.meta; line(io,m.id,m.family,m.dtype,m.shape,m.layout,m.phase,m.zero_alloc)
        end
    end
    shuffle!(scenario_rng(rep,"measurement_order"),cases)
    failures=0
    open(prefix*"-metrics.tsv","w") do metrics
      open(prefix*"-samples.tsv","w") do raw
       open(prefix*"-allocations.tsv","w") do allocations
        open(prefix*"-profiles.txt","w") do profiles
         open(prefix*"-allocation-stacks.tsv","w") do allocation_profiles
          open(prefix*"-errors.tsv","w") do errors
            line(metrics,"id","family","name","dtype","shape","layout","phase","work","expected_zero","rep","samples","evals","mean_ns","median_ns","sd_ns","min_ns","max_ns","p95_ns","gc_share","mean_bytes","min_bytes","max_bytes","bt_bytes","bt_allocations")
            line(raw,"id","rep","sample","time_ns","gc_ns")
            line(allocations,"id","rep","sample","bytes")
            line(allocation_profiles,"id","bytes","type","stack")
            line(errors,"id","exception","stack")
            for (k,c) in enumerate(cases)
                try
                    measure!(c,metrics,raw,allocations,profiles,allocation_profiles,rep,samples,seconds,profile_seconds)
                catch err
                    failures+=1
                    line(errors,c.meta.id,sprint(showerror,err),sprint(Base.show_backtrace,catch_backtrace()))
                    flush(errors)
                    println(stderr,"CASE_FAILED ",c.meta.id,": ",sprint(showerror,err))
                end
                k%50==0 && println("PROGRESS ",k,"/",length(cases)," failures=",failures)
            end
          end
         end
        end
       end
      end
    end
    meta=Dict("julia"=>string(VERSION),"source_sha"=>get(ENV,"PB_SOURCE_SHA","unknown"),
              "harness_sha"=>get(ENV,"PB_HARNESS_SHA","unknown"),"rep"=>rep,"mode"=>mode,
              "seed_scheme"=>"FNV1a64 keyed scenario xor repetition", "host"=>get(ENV,"SLURMD_NODENAME",get(ENV,"HOSTNAME","unknown")),
              "slurm_job_id"=>get(ENV,"SLURM_JOB_ID","none"),"julia_threads"=>Threads.nthreads(),
              "blas_threads"=>BLAS.get_num_threads(),"blas"=>string(BLAS.get_config()),
              "cpu"=>Sys.CPU_NAME,"benchmarktools"=>string(pkgversion(BenchmarkTools)),
              "cases"=>length(cases),"failures"=>failures,"samples_target"=>samples,
              "seconds_limit"=>seconds,"profile_seconds"=>profile_seconds,"elapsed_s"=>time()-started)
    open(prefix*"-metadata.toml","w") do io
        TOML.print(io,meta)
    end
    println("REPETITION_FINISHED cases=",length(cases)," failures=",failures," elapsed=",time()-started)
    failures==0 || exit(2)
end

main()
