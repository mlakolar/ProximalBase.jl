using LinearAlgebra, Random
include("load_source.jl")
include("historical_cases.jl")
BLAS.set_num_threads(1)

bitwise(x::Vector{Float32}, y::Vector{Float32}) = reinterpret(UInt32, x) == reinterpret(UInt32, y)
bitwise(x::Vector{Float64}, y::Vector{Float64}) = reinterpret(UInt64, x) == reinterpret(UInt64, y)

function fusion_fixtures(emit)
    cases() do group, label, x, r
        emit(group, label, ndims(x), vec(copy(x)), r)
    end
    for T in (Float32, Float64)
        for r in T[843.4569612191655, 1765.9906913347882, 1762.9198115568126]
            emit("extended", "compiler-sensitive-tie", 1, ones(T, 2048), r)
        end
        for n in (1, 2, 4, 16, 256, 2048)
            s = T(n)
            tied = fill(one(T), n)
            for r in (s, prevfloat(s), nextfloat(s), s * eps(T))
                emit("boundary", "tie", 1, tied, r)
            end
        end
        d, M = nextfloat(zero(T)), floatmax(T)
        for (name, x, r) in (("empty", T[], one(T)), ("signed-copy", T[-0.0, 1, 0], one(T)),
                ("overflow", T[M, -M], M), ("NaN", T[NaN, 1], one(T)),
                ("one_inf", T[Inf, -0.0], one(T)), ("many_inf", T[Inf, -Inf], one(T)),
                ("tiny", T[M, -M], d), ("zero", T[1, -0.0], zero(T)),
                ("infinite_radius", T[1, -0.0], T(Inf)))
            emit("exceptional", name, 1, x, r)
        end
    end
end

function check_all(arm, path)
    PB, fuse = arm.ProximalBase, arm.value_with_native_screen!
    open(path, "w") do io
        println(io, "group\tcase\toriginal_dims\ttype\tn\tradius\talias\tstep\tbitwise_output\tvalue_match\tstandalone_match\tnative_bounds\tnative_state\twidth\tbaseline_copy")
        setprecision(BigFloat,2304) do
            fusion_fixtures() do group, label, ndims, x, r
                T = eltype(x)
                f = PB.IndBallL1(r)
                base, opt = similar(x), similar(x)
                original = copy(x)
                for alias in (false,true), step in (zero(T), one(T), T(Inf))
                    if alias
                        copyto!(base,x)
                        copyto!(opt,x)
                    end
                    value = PB.value_and_prox!(base,f,alias ? base : x,step)
                    v, b = fuse(opt,f,alias ? opt : x,step)
                    @assert bitwise(x, original)
                    sat = isfinite(r) && all(isfinite,opt)
                    d = sat ? sum(v -> abs(BigFloat(v)),opt; init=BigFloat(0))-BigFloat(r) : nothing
                    @assert b === nothing || (sat && BigFloat(b[1]) <= d <= BigFloat(b[2]))
                    status = b === nothing ? "unavailable" : b[2] <= 0 ? "feasible" : b[1] > 0 ? "infeasible" : "cross"
                    width = b === nothing ? missing : Float64(BigFloat(b[2])-BigFloat(b[1]))
                    row = (group,label,ndims,T,length(x),Float64(r),alias,step,bitwise(base,opt),
                           isequal(v,value),isequal(v,f(opt)),b !== nothing,status,width,bitwise(base,x))
                    println(io, join(row, '\t'))
                end
            end
        end
    end
    for T in (Float32, Float64)
        f, y = PB.IndBallL1(one(T)), T[-0.0, 9]
        for (x, step, errtype) in ((T[1,2], T(-1), ArgumentError), (T[1,2], T(NaN), ArgumentError),
                                   (T[1,2,3], one(T), DimensionMismatch))
            err = try fuse(y,f,x,step); nothing catch e; e end
            @assert err isa errtype && bitwise(y,T[-0.0, 9])
        end
    end
    println("fused output, value, containment, alias and invalid-call probes written")
end

@assert Threads.nthreads() == BLAS.get_num_threads() == 1
arm = load_arm(:Fusion, ARGS[1])
Base.include(arm, joinpath(@__DIR__,"fusion.jl"))
Base.invokelatest(check_all, arm, ARGS[2])
