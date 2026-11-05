using BenchmarkTools, ProximalBase, LinearAlgebra, Random, Statistics, SparseArrays
using Profile, TOML, Printf, SHA
include("cases.jl")

function fingerprint(root, files)
    io = IOBuffer()
    for file in sort(files)
        write(io, relpath(file, root), '\0', read(file), '\0')
    end
    bytes2hex(sha1(take!(io)))
end

function source_metadata()
    root = dirname(dirname(realpath(pathof(ProximalBase))))
    files = [joinpath(root, "Project.toml")]
    for directory in ("src", "ext")
        isdir(joinpath(root, directory)) || continue
        for (folder, _, names) in walkdir(joinpath(root, directory)), name in names
            endswith(name, ".jl") && push!(files, joinpath(folder, name))
        end
    end
    revision = try
        readchomp(pipeline(`git -C $root rev-parse HEAD`;stderr=devnull))
    catch
        "unavailable"
    end
    dirty = try
        !isempty(readchomp(pipeline(`git -C $root status --porcelain --untracked-files=no`;stderr=devnull)))
    catch
        true
    end
    # the harness itself (this file and the case definitions), so a baseline names what produced it
    here = @__DIR__
    harness = [joinpath(here, "run.jl"), joinpath(here, "cases.jl")]
    Dict("source_path"=>root, "source_sha"=>revision, "source_dirty"=>dirty,
         "source_hash"=>fingerprint(root, files), "harness_hash"=>fingerprint(here, harness))
end

clean(s) = replace(string(s), '\t'=>' ', '\n'=>' ', '\r'=>' ')
line(io, xs...) = println(io,join(clean.(xs),'\t'))

function calibrate!(io, phase)
    b=@benchmarkable noop() setup=(noop())
    trial=run(b;evals=1,samples=1000,seconds=1.0)
    for (i,t) in enumerate(trial.times)
        line(io,phase,i,t)
    end
    flush(io)
    timer_summary(trial.times)
end

function timer_summary(times)
    values=filter(>(0),round.(Int,times))
    step=isempty(values) ? NaN : minimum(values)
    (;samples=length(times), median=isempty(times) ? NaN : median(times),
      minimum=isempty(times) ? NaN : minimum(times), step)
end

function timer_floor(start, finish)
    if min(start.samples,finish.samples)<100 ||
       (isnan(start.step) && isnan(finish.step) && start.median<1.0 && finish.median<1.0)
        return NaN
    end
    maximum(filter(!isnan,[start.median,finish.median,start.step,finish.step]))
end

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
    b=@benchmarkable $op() setup=($prep())
    trial=if c.evals===nothing
        tune!(b;seconds=seconds)
        run(b;samples=samples,seconds=seconds)
    else
        run(b;evals=c.evals,samples=samples,seconds=seconds)
    end
    c.evals===nothing && c.validate()
    times=trial.times
    isempty(times) && error("empty benchmark trial")
    gc_share=sum(trial.gctimes)/max(sum(times),1)
    m=c.meta
    line(metrics,m.id,m.family,m.name,m.dtype,m.shape,m.layout,m.phase,m.work,m.zero_alloc,
         rep,length(times),trial.params.evals,mean(times),median(times),std(times;corrected=false),
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
    if profile_seconds>0 && maximum(bytes)>0 && m.zero_alloc
        profile_alloc!(allocation_profiles,c)
    end
end

function main()
    package_root=realpath(dirname(dirname(pathof(ProximalBase))))
    expected_root=get(ENV,"PB_EXPECTED_PACKAGE_ROOT","")
    if !isempty(expected_root)
        expected_root=realpath(expected_root)
        package_root==expected_root ||
            error("Loaded ProximalBase from $package_root; expected $expected_root")
    end
    rep=parse(Int,get(ENV,"PB_REPETITION","1"))
    1 <= rep <= 10 || error("PB_REPETITION must be between 1 and 10")
    output=get(ENV,"PB_OUTPUT",joinpath(@__DIR__,"results"))
    suite=get(ENV,"PB_SUITE","quick")
    groups=sort(split(get(ENV,"PB_GROUPS",join(CASE_GROUPS,',')),','))
    samples=parse(Int,get(ENV,"PB_SAMPLES",suite=="quick" ? "10" : "40"))
    seconds=parse(Float64,get(ENV,"PB_SECONDS",suite=="quick" ? "0.01" : "0.1"))
    profile_seconds=parse(Float64,get(ENV,"PB_PROFILE_SECONDS","0"))
    samples>0 && isfinite(seconds) && seconds>0 && isfinite(profile_seconds) && profile_seconds>=0 ||
        error("Samples and seconds must be positive; profile seconds must be nonnegative")
    BLAS.set_num_threads(1)
    Threads.nthreads()==1 || error("Run with JULIA_NUM_THREADS=1")
    started=time()
    Random.seed!(rep)
    cases=build_cases(rep;suite,groups)
    # Cases are built first, so their inputs do not depend on the selection.
    selection=get(ENV,"PB_CASES","")
    isempty(selection) || filter!(c -> occursin(Regex(selection),c.meta.id),cases)
    isempty(cases) && error("PB_CASES selects no case")
    if get(ENV,"PB_LIST","false")=="true"
        foreach(c -> println(c.meta.id),cases)
        return
    end
    mkpath(output)
    prefix=joinpath(output,"julia-$(VERSION)-rep$(lpad(rep,2,'0'))")
    any(isfile(prefix*suffix) for suffix in ("-metrics.tsv","-errors.tsv")) &&
        error("Refusing to overwrite a repetition; use a fresh output directory")
    profile_seconds>0 && Profile.init(n=2_000_000,delay=0.001)
    open(prefix*"-catalog.tsv","w") do io
        line(io,"id","family","dtype","shape","layout","phase","zero_alloc","evals_policy")
        for c in cases
            m=c.meta; line(io,m.id,m.family,m.dtype,m.shape,m.layout,m.phase,m.zero_alloc,
                           c.evals===nothing ? "auto" : "fixed")
        end
    end
    shuffle!(scenario_rng(rep,"measurement_order"),cases)
    failures=0
    calibration=Dict{String,Any}("calibration_samples"=>1000,"calibration_seconds"=>1.0,
                                 "tune_seconds"=>seconds)
    open(prefix*"-metrics.tsv","w") do metrics
      open(prefix*"-samples.tsv","w") do raw
       open(prefix*"-allocations.tsv","w") do allocations
        open(prefix*"-profiles.txt","w") do profiles
         open(prefix*"-allocation-stacks.tsv","w") do allocation_profiles
          open(prefix*"-errors.tsv","w") do errors
           open(prefix*"-calibration.tsv","w") do calibration_io
            line(metrics,"id","family","name","dtype","shape","layout","phase","work","expected_zero","rep","samples","evals","mean_ns","median_ns","sd_ns","min_ns","max_ns","p95_ns","gc_share","mean_bytes","min_bytes","max_bytes","bt_bytes","bt_allocations")
            line(raw,"id","rep","sample","time_ns","gc_ns")
            line(allocations,"id","rep","sample","bytes")
            line(allocation_profiles,"id","bytes","type","stack")
            line(errors,"id","exception","stack")
            line(calibration_io,"phase","sample","time_ns")
            start=calibrate!(calibration_io,"start")
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
            finish=calibrate!(calibration_io,"end")
            for (phase,summary) in (("start",start),("end",finish))
                calibration["timer_$(phase)_median_ns"]=summary.median
                calibration["timer_$(phase)_min_ns"]=summary.minimum
                calibration["timer_$(phase)_step_ns"]=summary.step
            end
            calibration["timer_floor_ns"]=timer_floor(start,finish)
           end
          end
         end
        end
       end
      end
    end
    meta=Dict("julia"=>string(VERSION),"rep"=>rep,"suite"=>suite,"groups"=>join(groups,','),
              "expected_package_root"=>expected_root,
              "harness_hash"=>fingerprint(@__DIR__,[joinpath(@__DIR__,name) for name in ("cases.jl","run.jl")]),
              "seed_scheme"=>"FNV1a64 keyed scenario xor repetition", "host"=>get(ENV,"SLURMD_NODENAME",get(ENV,"HOSTNAME","unknown")),
              "slurm_job_id"=>get(ENV,"SLURM_JOB_ID","none"),"julia_threads"=>Threads.nthreads(),
              "blas_threads"=>BLAS.get_num_threads(),"blas"=>string(BLAS.get_config()),
              "cpu"=>Sys.CPU_NAME,"cpu_model"=>Sys.cpu_info()[1].model,"benchmarktools"=>string(pkgversion(BenchmarkTools)),
              "check_bounds"=>("default","yes","no")[Base.JLOptions().check_bounds+1],
              "os"=>string(Sys.KERNEL," ",Sys.MACHINE),
              "cases"=>length(cases),"failures"=>failures,"samples_target"=>samples,
              "seconds_limit"=>seconds,"profile_seconds"=>profile_seconds,"elapsed_s"=>time()-started)
    merge!(meta,source_metadata())
    merge!(meta,calibration)
    open(prefix*"-metadata.toml","w") do io
        TOML.print(io,meta)
    end
    println("REPETITION_FINISHED cases=",length(cases)," failures=",failures," elapsed=",time()-started)
    failures==0 || exit(2)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
