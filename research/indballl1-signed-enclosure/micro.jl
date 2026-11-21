using LinearAlgebra, Random
include("load_source.jl")
include("timing_support.jl")
include("micro_fixtures.jl")
BLAS.set_num_threads(1)

function kernels(arm)
    PB = arm.ProximalBase
    (p=PB.prox!, v=PB.value_and_prox!, allocp=PB.prox, screen=arm.wide_screen,
     refine=arm.screen_then_signed, residual=arm.signed_residual, accurate=arm.compensated,
     sharing=arm.value_with_native_screen!)
end

struct Call{Mode,K,F,T}
    api::K
    ball::F
    input::Vector{T}
    output::Vector{T}
    stored::Vector{T}
    cap::Float64
end

make_call(::Val{M}, api::K, f::F, input::Vector{T}, stored, cap) where {M,K,F,T} =
    Call{M,K,F,T}(api, f, input, similar(input), stored, cap)

@inline score(b::Nothing) = -1.0
@inline score(b::Tuple{Float64,Float64}) = b[1] + b[2]

@noinline function one_call(op::Call{Mode}) where {Mode}
    a, f, input, y, stored = op.api, op.ball, op.input, op.output, op.stored
    r = f.r
    unit = one(r)
    if Mode === :A
        return score(a.screen(stored,r))
    elseif Mode === :B_sign
        return score(a.refine(stored,r,0.0))
    elseif Mode === :B_wide
        return score(a.refine(stored,r,op.cap))
    elseif Mode === :B_tight
        return score(a.refine(stored,r,Float64(r)*2.0^-40))
    elseif Mode === :B_residual
        return score(a.residual(stored,r))
    elseif Mode === :C
        return score(a.accurate(stored,r))
    elseif Mode === :indicator
        return Float64(iszero(f(input)))
    elseif Mode === :allocating_prox
        return Float64(first(a.allocp(f,input,unit)))
    elseif Mode === :prox
        return Float64(first(a.p(y,f,input,unit)))
    elseif Mode === :fused
        return Float64(iszero(a.v(y,f,input,unit))) + Float64(first(y))
    elseif Mode === :prox_A
        a.p(y,f,input,unit)
        return score(a.screen(y,r)) + Float64(first(y))
    elseif Mode === :fused_A
        v = a.v(y,f,input,unit)
        return Float64(iszero(v)) + score(a.screen(y,r)) + Float64(first(y))
    elseif Mode === :fused_C
        v = a.v(y,f,input,unit)
        return Float64(iszero(v)) + score(a.accurate(y,r)) + Float64(first(y))
    elseif Mode === :shared
        v, b = a.sharing(y,f,input,unit)
        return Float64(iszero(v)) + score(b) + Float64(first(y))
    else
        copyto!(y,input)
        if Mode === :fused_alias
            return Float64(iszero(a.v(y,f,y,unit))) + Float64(first(y))
        elseif Mode === :fused_alias_A
            v = a.v(y,f,y,unit)
            return Float64(iszero(v)) + score(a.screen(y,r)) + Float64(first(y))
        end
        v, b = a.sharing(y,f,y,unit)
        return Float64(iszero(v)) + score(b) + Float64(first(y))
    end
end

function (op::Call)(n)
    sink = 0.0
    for _ in 1:n
        sink += one_call(op)
    end
    sink
end

function sample(io, base, control, candidate, pin, label, x::Vector{T}, r::T, rounds, quick) where {T}
    profiles = map(kernels, (base,control,candidate,pin))
    f0 = base.ProximalBase.IndBallL1(r)
    stored = similar(x)
    value = profiles[1].v(stored, f0, x, one(T))
    diagnostic = profiles[1].screen(stored, r)
    cap = 2.0^-20*Float64(r)
    function scenario(arm, api, mode)
        f = arm.ProximalBase.IndBallL1(r)
        make_call(Val(mode),api,f,x,stored,cap)
    end
    experimental = similar(x)
    v, _ = profiles[3].sharing(experimental, candidate.ProximalBase.IndBallL1(r), x, one(T))
    parity = isequal(experimental,stored) && isequal(v,value)
    dref = setprecision(BigFloat,2304) do
        sum(v -> abs(BigFloat(v)),stored)-BigFloat(r)
    end
    sign = diagnostic === nothing || base.straddles(diagnostic,0.0)
    cap_cross = diagnostic === nothing || base.straddles(diagnostic,cap)
    all_pairs = (
        ("diag-B-sign", :A, :B_sign), ("diag-B-wide", :A, :B_wide),
        ("diag-B-tight", :A, :B_tight), ("diag-B-residual", :A, :B_residual),
        ("diag-C", :A, :C), ("proj-fused", :prox, :fused),
        ("proj-plus-A", :prox, :prox_A), ("fused-plus-A", :fused, :fused_A),
        ("fused-plus-C", :fused, :fused_C), ("fused-plus-shared", :fused, :shared),
        ("shared-versus-separate", :fused_A, :shared),
        ("alias-shared-versus-separate", :fused_alias_A, :shared_alias),
        ("gate-prox", :prox, :prox), ("gate-fused", :fused, :fused),
        ("gate-allocating-prox", :allocating_prox, :allocating_prox),
        ("gate-indicator", :indicator, :indicator), ("master-versus-pin", :fused, :fused),
    )
    config = quick ? (all_pairs[1],all_pairs[5],all_pairs[10],all_pairs[16]) : all_pairs
    for (labelpair, call_a, call_b) in config
        usepin = labelpair == "master-versus-pin"
        b = usepin ? pin : candidate
        meta = (T,length(x),label,labelpair,!isequal(x,stored),iszero(value),Float64(dref),sign,cap_cross,parity)
        paired(io, meta, scenario(base,profiles[1],call_a), scenario(b,profiles[usepin ? 4 : 3],call_b),
               scenario(control,profiles[2],call_a), rounds)
    end
end

function study(arms, path, rounds, quick)
    base, control, candidate, pin = arms
    open(path,"w") do io
        println(io, "type\tn\tcase\tpair\tactive\tpb_feasible\tmargin\tsign_refine\tcap_refine\tparity\t", TIMING_COLUMNS)
        micro_fixtures(base.ProximalBase,quick) do label, x, r
            sample(io, base,control,candidate,pin,label,x,r,rounds,quick)
            flush(io)
        end
    end
end

@assert Threads.nthreads() == BLAS.get_num_threads() == 1
@assert !Base.get_zero_subnormals()
master, pin, path, rounds, flavor = ARGS
arms = (load_arm(:BaseArm,master), load_arm(:ControlArm,master), load_arm(:CandidateArm,master), load_arm(:OldPinArm,pin))
for arm in arms
    Base.include(arm, joinpath(@__DIR__,"fusion.jl"))
end
println("Julia=", VERSION, " CPU=", Sys.CPU_NAME, " BLAS=", BLAS.get_num_threads())
Base.invokelatest(study, arms, path, parse(Int,rounds), flavor == "canary")
