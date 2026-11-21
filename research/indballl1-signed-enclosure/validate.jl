using LinearAlgebra, Random
include("load_source.jl")
include("historical_cases.jl")
BLAS.set_num_threads(1)

# Every dyadic sum in this matrix needs at most 2098 + ceil(log2(4096)) bits.
function exact_margin(x, r)
    isfinite(r) && all(isfinite, x) || return nothing
    s = -BigFloat(r)
    for v in x
        s += abs(BigFloat(v))
    end
    s
end

function study_cases(emit)
    cases() do group, label, x, r
        emit(ndims(x) == 1 ? group : group * "-vectorized", label, vec(copy(x)), r, "unit")
    end
    for r in (843.4569612191655, 1765.9906913347882, 1762.9198115568126)
        emit("extended", "compiler-sensitive-tie", ones(Float64, 2048), r, "unit")
    end
    rng = Xoshiro(202611211138)
    for T in (Float32, Float64)
        step = T === Float32 ? 70 : 400
        for (power, scale) in ((-step, "small"), (0, "unit"), (step, "large"))
            r = ldexp(one(T), power)
            for n in (1, 2, 4, 16, 256, 2048)
                x = [copysign(r / n, isodd(i) ? -one(T) : one(T)) for i in 1:n]
                emit("ordinary", "exact-tie", x, r, scale)
                emit("ordinary", "radius-prev", x, prevfloat(r), scale)
                emit("ordinary", "radius-next", x, nextfloat(r), scale)
                for (name, stepfloat) in (("coefficient-prev", prevfloat), ("coefficient-next", nextfloat))
                    neighbor = copy(x)
                    neighbor[1] = -stepfloat(abs(x[1]))
                    emit("ordinary", name, neighbor, r, scale)
                end
                emit("ordinary", "far-interior", x ./ 4, r, scale)
                emit("ordinary", "tiny-relative-radius", x, r * eps(T), scale)
                for sample in 1:5
                    mags = r .* T.(exp2.(rand(rng, -18:8, n))) .* (rand(rng, T, n) .+ T(0.5))
                    z = mags .* rand(rng, T[-1, +1], n)
                    exact = T(exact_margin(z, zero(T)))
                    emit("ordinary", "random-$sample-projection", z, exact * T(0.23), scale)
                    emit("ordinary", "random-$sample-prev", z, prevfloat(exact), scale)
                    emit("ordinary", "random-$sample-next", z, nextfloat(exact), scale)
                end
                if n > 1
                    xbad = zeros(T, n)
                    xbad[1] = -r
                    xbad[end] = eps(T) * r / 4
                    xbad[2] = n > 2 ? -zero(T) : xbad[2]
                    emit("ordinary", "rounded-copy", xbad, r, scale)
                end
            end
        end
        emit("contract", "empty", T[], one(T), "unit")
        delta, tiny, big = nextfloat(zero(T)), floatmin(T), floatmax(T)
        for (label, x, r) in (
            ("large-exact", T[big], big),
            ("large-tail", T[big, delta], big),
            ("large-overflow", T[big, -big], big),
            ("large-overflow-small-radius", T[big, -big], one(T)),
            ("extreme-cancellation", T[big, -big / 2], big),
            ("tiny-tail", T[tiny, -delta], tiny),
            ("tiny-distance", T[prevfloat(tiny), -0.0], tiny),
            ("tiny-internal", T[delta, -delta, -0.0], tiny),
            ("tiny-exact", T[prevfloat(tiny), -delta], tiny),
            ("unsupported-subnormal-radius", T[delta, -delta], delta),
            ("unsupported-zero-radius", T[one(T)], zero(T)),
            ("unsupported-infinite-radius", T[one(T)], T(Inf)),
            ("nonfinite-infinity", T[one(T), T(Inf)], one(T)),
            ("nonfinite-nan", T[one(T), T(NaN)], one(T)),
        )
            emit("range", label, x, r, "exceptional")
        end
    end
    emit("known-rejection", "two-pass-1119-stored-coefficients", fill(0.000999999999999999, 2048), 2.048, "unit")
end

@inline compare(b, q) = b === nothing ? "unavailable" : b[2] <= q ? "le" : b[1] > q ? "gt" : "cross"

function row(io, meta, tag, bounds, d, cap, scans)
    sign = compare(bounds, 0.0)
    cap_state = compare(bounds, cap)
    width = missing
    relative_width = missing
    lo, hi = missing, missing
    if bounds !== nothing
        lo, hi = bounds
        d !== nothing && BigFloat(lo) <= d <= BigFloat(hi) || error("containment failed $meta $tag $bounds $d")
        @assert (sign != "le" || d <= 0) && (sign != "gt" || d > 0)
        @assert (cap_state != "le" || d <= BigFloat(cap)) && (cap_state != "gt" || d > BigFloat(cap))
        width = Float64(BigFloat(hi) - BigFloat(lo))
        r = meta[6]
        relative_width = Float64((BigFloat(hi) - BigFloat(lo)) / BigFloat(r))
    end
    println(io, join((meta..., tag, cap, sign, cap_state, lo, hi, width, relative_width, scans), '\t'))
end

function probe(io, a, id, group, label, scale, route, f, x, unchanged)
    T = eltype(x)
    r = f.r
    given = copy(x)
    d = exact_margin(x, r)
    v = f(x)
    exact = d === nothing ? "nonfinite" : d < 0 ? "negative" : d > 0 ? "positive" : "zero"
    score = d === nothing ? missing : Float64(d)
    meta = (id, group, label, T, length(x), Float64(r), scale, route, iszero(v), unchanged, exact, score)
    cap = (4 * eps(T)) * Float64(r)
    A = a.wide_screen(x, r)
    C = a.compensated(x, r)
    signed = a.signed_residual(x, r)
    refine_sign = A !== nothing && a.straddles(A, 0.0)
    refine_cap = A !== nothing && a.straddles(A, cap)
    scanned = a.supported_radius(r) ? 1 : 0
    row(io, meta, "A", A, d, cap, scanned)
    row(io, meta, "B-sign", refine_sign ? signed : A, d, cap, scanned + refine_sign)
    row(io, meta, "B-cap", refine_cap ? signed : A, d, cap, scanned + refine_cap)
    row(io, meta, "B-residual-only", signed, d, cap, scanned)
    row(io, meta, "C", C, d, cap, scanned)
    @assert isequal(x, given)
end

function check_routes(io, a)
    PB = a.ProximalBase
    id = 0
    study_cases() do group, label, x, r, scale
        id += 1
        T = eltype(x)
        f = PB.IndBallL1(r)
        original = copy(x)
        probe(io, a, id, group, label, scale, "input", f, x, true)
        p = PB.prox(f, x)
        b = similar(x)
        @assert PB.prox!(b, f, x, one(T)) === b
        c = copy(x)
        @assert PB.prox!(c, f, c, one(T)) === c
        d = similar(x)
        e = copy(x)
        @assert isequal(PB.value_and_prox!(d, f, x, one(T)), f(d))
        @assert isequal(PB.value_and_prox!(e, f, e, one(T)), f(e))
        @assert isequal(x, original)
        for (route, y) in (("prox",p), ("distinct",b), ("alias",c), ("fused",d), ("fused_alias",e))
            probe(io, a, id, group, label, scale, route, f, y, isequal(y, x))
        end
    end
    # Use the mode guard only here. Do not call PB projection under flush-to-zero.
    before = Base.get_zero_subnormals()
    try
        Base.set_zero_subnormals(true)
        if Base.get_zero_subnormals()
            for T in (Float32, Float64), method in (a.wide_screen, a.screen_then_signed, a.compensated)
                @assert method(T[one(T), nextfloat(zero(T))], one(T)) === nothing
            end
        end
    finally
        Base.set_zero_subnormals(before)
    end
    println("validated fixtures=", id, " and six routes each")
end

function main(arm, path)
    @assert Threads.nthreads() == BLAS.get_num_threads() == 1
    @assert !Base.get_zero_subnormals()
    println("Julia=", VERSION, " check_bounds=", Base.JLOptions().check_bounds,
            " CPU=", Sys.CPU_NAME, " BLAS=", BLAS.get_num_threads())
    open(path, "w") do io
        println(io, "id\tgroup\tcase\ttype\tn\tr\tscale\troute\tpb_feasible\tunchanged\texact_sign\tmargin\tdesign\tcap\tsign_state\tcap_state\tlo\thi\twidth\trelative_width\tscans")
        setprecision(BigFloat, 2304) do
            check_routes(io, arm)
        end
    end
end

Base.invokelatest(main, load_arm(:Validation, ARGS[1]), ARGS[2])
