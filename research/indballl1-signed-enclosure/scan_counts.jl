using LinearAlgebra, Random
include("load_source.jl")
include("micro_fixtures.jl")
BLAS.set_num_threads(1)

function call(arm, mode, y, x::Vector{T}, r::T) where {T}
    PB = arm.ProximalBase
    f = PB.IndBallL1(r)
    if mode == "prox"
        PB.prox!(y,f,x,one(T))
        return (nothing,nothing)
    elseif mode == "fused"
        return (PB.value_and_prox!(y,f,x,one(T)),nothing)
    elseif mode == "prox+A"
        PB.prox!(y,f,x,one(T))
        return (nothing,arm.wide_screen(y,r))
    elseif mode == "fused+A"
        v = PB.value_and_prox!(y,f,x,one(T))
        return (v,arm.wide_screen(y,r))
    end
    arm.value_with_native_screen!(y,f,x,one(T))
end

function profile(original, measured, fields, path)
    PB = measured.ProximalBase
    open(path,"w") do io
        categories = ("calls_" .* fields, "full_" .* fields, "entries_" .* fields)
        println(io, join(("type","n","case","mode","alias","matched","bounds_matched","full_total",
                          Iterators.flatten(categories)...), '\t'))
        micro_fixtures(original.ProximalBase,false) do label, x, r
            y, counted = similar(x), similar(x)
            PB._trace_n[] = length(x)
            for alias in (false,true), mode in ("prox","fused","prox+A","fused+A","shared")
                if alias
                    copyto!(y,x)
                    copyto!(counted,x)
                end
                value, b = call(original,mode,y,alias ? y : x,r)
                fill!(PB._trace_scans,0)
                other, measured_b = call(measured,mode,counted,alias ? counted : x,r)
                totals = PB._trace_scans
                wrote_match = isequal(y,counted) && isequal(value,other)
                row = (eltype(x),length(x),label,mode,alias,wrote_match,isequal(b,measured_b),sum(totals[2,:]),
                       totals[1,:]...,totals[2,:]...,totals[3,:]...)
                println(io, join(row, '\t'))
            end
        end
    end
end

@assert Threads.nthreads() == BLAS.get_num_threads() == 1
original, traced, out = ARGS
arms = (load_arm(:Original,original), load_arm(:Traced,traced))
for arm in arms
    Base.include(arm, joinpath(@__DIR__,"fusion.jl"))
end
Base.invokelatest(profile, arms...,readlines(joinpath(traced,"scan_fields.txt")),out)
