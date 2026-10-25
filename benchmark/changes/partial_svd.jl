# Run this identical harness against version-specific before/after projects.
include(joinpath(@__DIR__, "..", "campaign", "run.jl"))

function svd_input(rep, T, m, n, rank, spectrum)
    rng = scenario_rng(rep, "partial_svd/$T/$m/$n/$rank/$spectrum")
    if spectrum == "gapped"
        U = Matrix(qr(randn(rng, T, m, rank)).Q)
        V = Matrix(qr(randn(rng, T, n, rank)).Q)
        s = collect(range(T(20), T(10); length=rank))
        X = U * Diagonal(s) * V' + T(0.02)/sqrt(T(m)) .* randn(rng, T, m, n)
    else
        p = min(m, n)
        U = Matrix(qr(randn(rng, T, m, p)).Q)
        V = Matrix(qr(randn(rng, T, n, p)).Q)
        s = vcat(collect(range(T(1.2), T(1.01); length=rank)),
                 collect(range(T(0.99), T(0.11); length=p-rank)))
        X = U * Diagonal(s) * V'
    end
    X, X + T(1e-5) .* randn(rng, T, m, n)
end

function svd_case!(cases, kind, X, previous, rank, spectrum, state)
    T = eltype(X)
    step = one(T)
    make(partial) = kind == "nuclear" ? NuclearNorm(step; partial) : IndBallRank(rank; partial)
    f, reference = make(true), make(false)
    Y, scratch = similar(X), similar(X)
    prepare = state == "random_restart" ?
        () -> (f.pws === nothing || (f.pws.warm = 0); nothing) :
        () -> prox!(scratch, f, previous, step)
    operation = () -> prox!(Y, f, X, step)
    name = "$kind/rank$rank/$spectrum/$state"
    function check!()
        ref = prox(reference, X, step)
        for _ in 1:3
            prepare(); operation()
        end
        # Observe the chosen path outside all measurements, after learning rank.
        f.ws = nothing
        prepare(); operation()
        close_enough(Y, ref)
        block = f.pws === nothing ? 0 : f.pws.ℓ
        println("SVD_PATH ", name, " ", T, " ", size(X),
                " full=", f.ws !== nothing, " block=", block)
    end
    addcase!(cases, "partial_svd", name, operation, check!; T,
             shape=join(size(X), "x"), prepare, work=length(X),
             profile=T==Float64 && rank==4 && spectrum=="gapped" && state=="nearby")
end

function partial_svd_cases(rep)
    cases = PerfCase[]
    shapes = ((64,64), (128,64), (64,128), (96,64), (160,80), (128,96), (192,128), (512,256))
    canary = get(ENV, "PB_MODE", "full") == "canary"
    for T in (Float32, Float64), (m,n) in shapes, rank in (2,4,6,8,12)
        canary && !((m,n) in ((128,64),(64,128),(512,256)) && rank==4) && continue
        X, previous = svd_input(rep, T, m, n, rank, "gapped")
        for kind in ("nuclear", "rank"), state in ("nearby", "random_restart")
            svd_case!(cases, kind, X, previous, rank, "gapped", state)
        end
    end
    for T in (Float32,Float64), (m,n,rank) in ((128,64,4),(128,64,6),(128,64,8),
                                          (128,64,28),(160,80,32),(192,128,20),(512,256,20))
        X, previous = svd_input(rep, T, m, n, rank, "clustered")
        unrelated, _ = svd_input(rep+10_000, T, m, n, rank, "clustered")
        for state in ("nearby", "random_restart", "unrelated")
            seed_input = state == "unrelated" ? unrelated : previous
            svd_case!(cases, "nuclear", X, seed_input, rank, "clustered", state)
        end
    end
    for T in (Float32,Float64), (m,n) in ((128,64),(512,256))
        X, _ = svd_input(rep, T, m, n, 4, "gapped")
        step = one(T)
        addcold!(cases, "partial_svd", "nuclear/rank4/first_prox",
                 () -> NuclearNorm(step; partial=true), X; γ=step)
        addprox!(cases, "partial_svd_control", "nuclear_full", NuclearNorm(step), X;
                 γ=step, value=false)
    end
    cases
end

main(partial_svd_cases)
