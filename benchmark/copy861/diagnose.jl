using BenchmarkTools, Profile, InteractiveUtils, Statistics
include(joinpath(get(ENV, "PB_SOURCE", pwd()), "src", "ProximalBase.jl"))
const PB = ProximalBase

function rebuild!(d, s)
    for a in 1:s.nactive
        k = s.active[a]
        PB.activate!(d, k)
        copyto!(PB._block_storage(d, k), PB._block_storage(s, k))
    end
    d
end
function direct!(d, s)
    @inbounds for a in 1:d.nactive
        k = d.active[a]
        if s.position[k] == 0
            for i in d.groups[k]; d.storage[i] = zero(eltype(d)); end
        end
        d.position[k] = 0
    end
    @inbounds for a in 1:s.nactive
        k = s.active[a]
        for i in s.groups[k]; d.storage[i] = s.storage[i]; end
        d.active[a] = k
        d.position[k] = a
    end
    d.nactive = s.nactive
    d
end
function activate_all!(d,s)
    for a in 1:s.nactive; PB.activate!(d,s.active[a]); end
    d
end
function copyblocks!(d,s)
    for a in 1:s.nactive
        k=s.active[a]
        copyto!(PB._block_storage(d,k),PB._block_storage(s,k))
    end
    d
end

function main()
    for T in (Float32, Float64), b in (1, 8, 64), layout in ("contiguous", "strided", "arbitrary")
        n = 8192; K = n ÷ b; s = max(1, K ÷ 10)
        gs = layout == "contiguous" ? [(k-1)*b+1:k*b for k in 1:K] :
             layout == "strided" ? [k:K:n for k in 1:K] : [collect(k:K:n) for k in 1:K]
        src = PB.BlockIterate(T, n, gs)
        for k in 1:s, i in gs[k]; src[i] = T(0.5); end
        for shared in (false, true)
            d = shared ? copy(src) : PB.BlockIterate(T, n, gs)
            copyto!(d, src)
            cases = ("layout" => (() -> PB._samelayout(d, src)),
                     "zero" => (() -> PB._zero_active!(d)),
                     "activate" => (() -> activate_all!(d, src)),
                     "blocks" => (() -> copyblocks!(d, src)),
                     "rebuild" => (() -> rebuild!(d, src)),
                     "copyto" => (() -> copyto!(d, src)),
                     "direct" => (() -> direct!(d, src)))
            for (name, op) in cases
                prep = name in ("activate","rebuild") ?
                    (() -> (copyto!(d, src); PB._zero_active!(d))) : (() -> copyto!(d, src))
                t = @benchmark $op() setup=($prep()) evals=1 samples=500 seconds=.05
                println(join((T,b,layout,shared,name,median(t).time,median(t).memory,median(t).allocs), ','))
            end
        end
    end
    src = PB.BlockIterate(Float32,8192,[k:8192:8192 for k in 1:8192])
    for k in 1:819; src[k]=0.5f0; end
    d=copy(src)
    mkpath(get(ENV,"PB_DIAGNOSTICS","/tmp/pb861-diagnostics"))
    out=get(ENV,"PB_DIAGNOSTICS","/tmp/pb861-diagnostics")
    open(joinpath(out,"typed.txt"),"w") do io
        code_warntype(io,copyto!,Tuple{typeof(d),typeof(src)})
        code_llvm(io,copyto!,Tuple{typeof(d),typeof(src)};debuginfo=:none)
    end
    copyto!(d,src)
    Profile.clear()
    @profile for _ in 1:100000; copyto!(d,src); end
    open(joinpath(out,"profile.txt"),"w") do io; Profile.print(io;format=:flat,sortedby=:count); end
end
main()
