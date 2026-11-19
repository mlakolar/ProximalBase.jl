include("common.jl")

function timing_cases(emit, sizes)
    for T in (Float32,Float64), n in sizes
        # Identical dyadics under both Julia versions and compiler settings.
        # x and v are read-only during every paired comparison.
        v = T[(mod(53*i+19,257)-128)/64 for i in 1:n]
        z = zeros(T,n)
        inner = T(3n)
        outer = T(max(1,n÷5))/T(2)
        emit("zero_inside",T,n,v,z,inner,())
        emit("zero_outside",T,n,v,z,outer,())
        for (id,k) in (("single_face",1), ("small_face",min(8,max(1,n÷8))),
                       ("large_face",max(1,n÷2)))
            x = T[i <= k ? (-1)^i : 0.25 for i in 1:n]
            emit(id,T,n,v,x,T(0.625),())
        end
        x = ones(T,n)
        emit("all_active",T,n,v,x,T(5n),())
        emit("many_leave",T,n,v,x,T(0.125),())
        emit("scaled",T,n,v,x,T(0.375),(T(2),))
        # Head-last tie family. The analytic head-only level is 1 - c.
        c = T(0.1)
        tie = T(0.9/(1+0.06*Float64(n-1)^2*eps(T)/2))
        w = fill(-tie,n)
        w[end] = -one(T)
        emit("adversarial_face",T,n,w,x,c,())
        emit("adversarial_zero",T,n,w,z,c,())
    end
end

@noinline function batch(call::F, f, v, x, count) where F
    total = zero(float(eltype(v)))
    for _ in 1:count
        total += call(f,v,x)
    end
    return total
end

function sample(call::F, f, v, x, count) where F
    start = time_ns()
    value = batch(call, f, v, x, count)
    return (time_ns()-start)/count, value
end

function allocation(call::F, f, v, x) where F
    return @allocated call(f,v,x)
end

function med(v)
    s = sort(v)
    n = length(s)
    return (s[(n+1)÷2] + s[(n+2)÷2])/2
end

function time_pair(io, labels, calls, fs, id, T, n, v, x, reps, caseidx, target)
    sideorder = isodd(caseidx) ? (1,2) : (2,1)
    p = zeros(2)
    for side in sideorder
        call, f = calls[side], fs[side]
        sample(call, f, v, x, 1)
        # Pilot with multiple calls for a steadier estimate on short inputs.
        k = max(1,min(256,8192÷n))
        p[side], value = sample(call, f, v, x, k)
        tsv(io,VERSION,labels[side],side,T,id,n,"pilot",0,findfirst(==(side),sideorder),k,p[side],"",value)
    end
    count = clamp(ceil(Int,target/minimum(p)),1,100_000)
    warmtimes = [Float64[] for _ in 1:2]
    bytes = [allocation(calls[s],fs[s],v,x) for s in 1:2]
    for rep in 1:reps
        order = isodd(rep) ? sideorder : reverse(sideorder)
        for (pos,side) in enumerate(order)
            t, value = sample(calls[side],fs[side],v,x,count)
            push!(warmtimes[side],t)
            tsv(io,VERSION,labels[side],side,T,id,n,"warm",rep,pos,count,t,bytes[side],value)
        end
    end
    for side in 1:2
        tsv(io,VERSION,labels[side],side,T,id,n,"median",reps,"",count,med(warmtimes[side]),bytes[side],"")
    end
    flush(io)
end

# Cold means first observed public call for T in this checkout in this process.
# Dispatch and JIT of the public call are included. Source load and constructing
# arguments occur before the timer; stdlib compilation is shared by both sides.
function measure_cold(io, label, side, call, f, T, v, x)
    r = @timed Base.invokelatest(call,f,v,x)
    tsv(io,VERSION,label,side,T,"first_public_call",length(v),"first",0,side,1,
        r.time*1e9,r.bytes,r.value)
end

function timing(labels, checkouts, filename, reps, sizes, target)
    reps >= 5 || error("at least 5 paired repetitions")
    calls = (checkouts[1].kkt_violation, checkouts[2].kkt_violation)
    open(filename,"w") do io
        tsv(io,"julia","label","side","type","case","n","phase","rep","position","batch","ns_per_call","bytes_per_call","checksum")
        for T in (Float32,Float64)
            v, x = T[0.25,0.5], T[1,-1]
            for s in 1:2
                measure_cold(io,labels[s],s,calls[s],checkouts[s].NormLinf(T(0.25)),T,v,x)
            end
        end
        idx = Ref(0)
        timing_cases(sizes) do id,T,n,v,x,level,factors
            idx[] += 1
            fs = map(pb->penalty(pb,level,factors),checkouts)
            time_pair(io,labels,calls,fs,id,T,n,v,x,reps,idx[],target)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    5 <= length(ARGS) <= 6 || error("usage: timing.jl LABEL_A CHECKOUT_A LABEL_B CHECKOUT_B OUT.tsv [REPS]")
    sizes = parse.(Int, split(get(ENV,"N981_SIZES","2,8,32,256,4096,100000,1000000"),','))
    target = parse(Float64,get(ENV,"N981_TARGET_NS","250000"))
    a = load_checkout(ARGS[2],:TimingCheckoutA)
    b = load_checkout(ARGS[4],:TimingCheckoutB)
    Base.invokelatest(timing, (ARGS[1],ARGS[3]),(a,b),ARGS[5],length(ARGS)==6 ? parse(Int,ARGS[6]) : 6,sizes,target)
end
