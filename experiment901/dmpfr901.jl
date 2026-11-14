# BigFloat directed-rounding check of the GroupNormL2 screen (decision D-MPFR, #901), orchestrator-owned.
#
#   PB_TREES="base=/path;head=/path;Y1=/path" PB_OUTPUT=<dir> julia --threads=1 --startup-file=no --project=<env> dmpfr901.jl
#
# The screen kkt_violation(GroupNormL2(λ, [1:m], [w]), g, x, 1) at a zero group (x = 0) is ‖g‖ - λw. For BigFloat at
# precision 256 and 512, under RoundNearest, RoundUp, RoundDown and RoundToZero, on a grid of λ, w and g whose exponents
# include MPFR's extremes, the script reports for every tree:
#   (1) differences from the first tree (base), split as in the D-MPFR grid of #901: the product λw has an exponent
#       inside or outside MPFR's range;
#   (2) for cases whose operands, every g[i]², λw and the result are inside MPFR's range (with a margin of a few binades), the error against a reference computed
#       at 8x the precision under RoundNearest and rounded once in the active mode, in units of the bound
#       (m+8) eps (‖g‖ + λw) (the bound of accuracy901.jl), and how many results are not the correctly rounded value;
#   (3) Codex's reproducer of D-MPFR (precision 256, RoundToZero, m = floatmin(BigFloat)).
using Printf, Random, LinearAlgebra, SparseArrays
const TREES = [let (l, p) = split(s, "="; limit = 2); (String(strip(l)), abspath(strip(p))) end
               for s in split(ENV["PB_TREES"], ";") if !isempty(strip(s))]
const OUT = get(ENV, "PB_OUTPUT", "dmpfr-out"); mkpath(OUT)
const PBS = Module[]
for (i, (label, root)) in enumerate(TREES)
    m = Module(Symbol("Dm", i))
    Base.include(m, joinpath(root, "src", "ProximalBase.jl"))
    push!(PBS, Base.invokelatest(getfield, m, :ProximalBase))
end

emin_mpfr() = exponent(floatmin(BigFloat))
emax_mpfr() = exponent(floatmax(BigFloat))
inrange(e) = emin_mpfr() <= e <= emax_mpfr()
same(a::BigFloat, b::BigFloat) = (isnan(a) && isnan(b)) || (isequal(a, b) && signbit(a) == signbit(b))

screen(PB, λ, w, g) = PB.kkt_violation(PB.GroupNormL2(λ, [1:length(g)], [w]), g, zeros(BigFloat, length(g)), 1)

function run_grid()
    modes = (RoundNearest, RoundUp, RoundDown, RoundToZero)
    exps(p) = (emin_mpfr(), emin_mpfr() + 1, -(2^61), -1000, 0, 1000, 2^61, emax_mpfr() - 1)
    sigs = (1.0, 1.6180339887)
    rows = Dict{Tuple{String, Int, String, String}, Vector{Float64}}()   # tree, prec, mode, class -> [n, diff_vs_base, maxunits, notcr]
    for prec in (256, 512), mode in modes
        setprecision(BigFloat, prec) do
            setrounding(BigFloat, mode) do
                for eλ in exps(prec), sλ in sigs, ew in exps(prec), sw in sigs, eg in exps(prec), sg in sigs, m in (1, 3)
                    λ = ldexp(BigFloat(sλ), eλ); w = ldexp(BigFloat(sw), ew)
                    g = [ldexp(BigFloat(sg), eg) * BigFloat(k == 1 ? 1 : 0.75^k) for k in 1:m]
                    (isfinite(λ) && isfinite(w) && all(isfinite, g) && !iszero(λ) && !iszero(w)) || continue
                    pe = exponent(λ) + exponent(w)
                    cls = inrange(pe) ? "lw_in" : "lw_out"
                    vals = BigFloat[]
                    for PB in PBS
                        v = try screen(PB, λ, w, g) catch err; BigFloat(NaN) end
                        push!(vals, v)
                    end
                    # reference: operands, g², λw and the result inside the range
                    ref = nothing
                    if all(!iszero, g) && inrange(pe + 2) && inrange(pe - 2) && inrange(2 * minimum(exponent, g) - 4) && inrange(2 * exponent(g[1]) + 4)
                        r = setprecision(BigFloat, 8prec) do
                            setrounding(BigFloat, RoundNearest) do
                                sqrt(sum(big.(g) .^ 2)) - big(λ) * big(w)
                            end
                        end
                        if isfinite(r) && !iszero(r) && inrange(exponent(r))
                            ref = (BigFloat(r, mode; precision = prec), (m + 8) * eps(BigFloat) * (abs(BigFloat(sqrt(sum(g .^ 2)))) + λ * w))
                        end
                    end
                    for (ti, (label, _)) in enumerate(TREES)
                        key = (label, prec, string(mode), cls)
                        st = get!(rows, key, [0.0, 0.0, 0.0, 0.0])
                        st[1] += 1
                        same(vals[ti], vals[1]) || (st[2] += 1)
                        if ref !== nothing
                            e, b = ref
                            u = isfinite(vals[ti]) ? Float64(abs(vals[ti] - e) / b) : Inf
                            st[3] = max(st[3], u)
                            same(vals[ti], e) || (st[4] += 1)
                        end
                    end
                end
            end
        end
    end
    rows
end

rows = run_grid()
open(joinpath(OUT, "dmpfr.csv"), "w") do io
    println(io, "tree,precision,mode,lw_class,cases,differs_from_base,max_bound_units_vs_reference,not_correctly_rounded")
    for k in sort(collect(keys(rows)))
        v = rows[k]; println(io, join(k, ","), ",", Int(v[1]), ",", Int(v[2]), ",", @sprintf("%.4g", v[3]), ",", Int(v[4]))
    end
end
println("D-MPFR grid (differences from $(TREES[1][1]); max error vs 8x-precision reference in bound units):")
@printf("  %-6s %4s %-13s %-7s %7s %9s %10s %8s\n", "tree", "prec", "mode", "λw", "cases", "≠base", "max units", "not CR")
for k in sort(collect(keys(rows)))
    v = rows[k]
    @printf("  %-6s %4d %-13s %-7s %7d %9d %10.4g %8d\n", k[1], k[2], k[3], k[4], Int(v[1]), Int(v[2]), v[3], Int(v[4]))
end
# Codex's reproducer of D-MPFR
println("\nCodex's D-MPFR reproducer: precision 256, RoundToZero, m = floatmin(BigFloat), kkt_violation(GroupNormL2(m, [1:1], [m]), [m], [0], 1)")
setprecision(BigFloat, 256) do
    setrounding(BigFloat, RoundToZero) do
        m = floatmin(BigFloat)
        for (ti, (label, _)) in enumerate(TREES)
            PB = PBS[ti]
            v = PB.kkt_violation(PB.GroupNormL2(m, [1:1], [m]), [m], [BigFloat(0)], 1)
            println("  ", rpad(label, 6), " ", v == 0 ? "0" : v == m ? "m (= floatmin)" : string(v))
        end
    end
end
