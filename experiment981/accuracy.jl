include("common.jl")
include("oracle.jl")
include("cases.jl")

function record_score(io, label, pb, name, path, v, x, level, factors)
    W = float(promote_type(eltype(v), eltype(x)))
    c = isinf(level) ? Inf : dyadic(W(level)) * prod(dyadic.(factors); init=one(Q))
    d2, v2 = exact_score(v, x, c)
    reference = sqrt(BigFloat(d2))
    scale = sqrt(BigFloat(v2)) + BigFloat(c)
    expected = W(reference)
    observed = try
        pb.kkt_violation(penalty(pb, level, factors), v, x)
    catch ex
        nameof(typeof(ex))
    end
    typed = observed isa W
    ae, ne, ratio, status = if typed && isfinite(observed) && isfinite(expected)
        e = abs(BigFloat(observed) - reference)
        normalized = iszero(e) ? zero(BigFloat) : e/scale
        # The strict screen is an experimental accuracy target. A separate
        # ratio compares with the rounded first-order formula of frozen #981.
        floor = BigFloat(nextfloat(zero(W)))
        strict = (32*BigFloat(eps(W)))*scale + 2floor
        n = length(v)
        bound = ((2*BigFloat(n)+sqrt(BigFloat(n))+5)*BigFloat(eps(W))/2)*scale + 4n*floor
        boundratio = iszero(e) ? zero(BigFloat) : e/bound
        verdict = isinf(c) ? (isequal(observed, expected) ? "pass" : "fail") :
                  iszero(reference) && !iszero(observed) ? (e <= strict ? "roundoff_at_zero" : "fail") :
                  e <= strict ? "pass" : normalized >= big"0.001" ? "material" : "accuracy"
        e, normalized, boundratio, verdict
    elseif path == "scaled_documented_range" && typed && isinf(observed)
        # docs/src/numerics.md covers unrepresentable entries of the lazy v/α.
        Inf, Inf, Inf, "documented_range"
    elseif typed && observed == expected && isinf(expected)
        0, 0, 0, "pass"
    else
        Inf, Inf, Inf, "fail"
    end
    tsv(io, label, VERSION, W, name, path, length(v), level, join(factors,','), "score", d2,
        reference, scale, observed, ae, ne, ratio, status)
end

function record_contract(io, label, pb, T)
    v, x = T[1,-2], T[-1,1]
    f = pb.NormLinf(T(1))
    tiny = ldexp(one(T), T === Float32 ? -100 : -600)
    rules = (
        ("entry", ArgumentError, ()->pb.kkt_violation(f,v,1)),
        ("point_entry", ArgumentError, ()->pb.kkt_violation(f,v,x,1)),
        ("complex", ArgumentError, ()->pb.kkt_violation(f,complex.(v),x)),
        ("shape", DimensionMismatch, ()->pb.kkt_violation(f,v,x[1:1])),
        ("infinite_factor", ArgumentError, ()->pb.kkt_violation(pb.Scaled(f,T(Inf)),v,x)),
        ("unrepresentable_nested", ArgumentError, ()->pb.kkt_violation(pb.Scaled(pb.Scaled(f,tiny),tiny),v,x)),
        ("symmetric", ArgumentError, ()->pb.kkt_violation(f,T[1 2;2 1],pb.SymmetricSparseIterate(T[1 2;2 1]))),
        ("nan_precedes_inf", T(NaN), ()->pb.kkt_violation(f,T[Inf,2],T[0,NaN])),
        ("infinite_input", T(Inf), ()->pb.kkt_violation(f,T[Inf,2],x)),
        ("zero_scaled_returns_zero_function", T(2), ()->pb.kkt_violation(pb.Scaled(f,zero(T)),v,x)),
    )
    for (name, expected, call) in rules
        got = try call() catch ex; typeof(ex) end
        success = (expected isa Type ? got === expected : got isa T && isequal(got,expected))
        tsv(io,label,VERSION,T,name,"contract",length(v),1,"",expected,"","","",got,"","","",success ? "pass" : "fail")
    end
end

function accuracy(label, pb, file)
    setprecision(BigFloat, 256) do
        open(file, "w") do io
            tsv(io, "label","julia","type","case","path","n","base_level","factors","expectation","exact_squared","reference","normalizer","observed","abs_error","normalized_error","frozen_bound_ratio","status")
            score_cases((args...) -> record_score(io,label,pb,args...))
            for T in (Float32,Float64)
                record_contract(io,label,pb,T)
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 3 || error("usage: accuracy.jl LABEL CHECKOUT OUT.tsv")
    pb = load_checkout(ARGS[2], :AccuracyCheckout)
    accuracy(ARGS[1], pb, ARGS[3])
end
