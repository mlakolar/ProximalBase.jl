# Issue #747 probe: BigFloat value path of NormL1Fused(λ1, Inf) near the MPFR exponent limit.
using ProximalBase, Random, InteractiveUtils
const PB = ProximalBase
const emax = Base.MPFR.get_emax()
const emin = Base.MPFR.get_emin()

println("== 1. platform")
@show sizeof(Clong) sizeof(Int) VERSION Sys.MACHINE
@show Base.MPFR.version()
@show emax emin

println("== 2. exponent types")
let M = floatmax(BigFloat)
  println("typeof(exponent(M)) = ", typeof(exponent(M)))
  println("typeof(frexp(M)[2]) = ", typeof(frexp(M)[2]))
  println("typeof(PB._frexp(M)) = ", typeof(PB._frexp(M)), "  exponent = ", PB._frexp(M)[2])
  println("exponent(M) = ", exponent(M))
end

println("== 3. ldexp dispatch")
println("which(ldexp, Tuple{BigFloat, Int64})  = ", which(ldexp, Tuple{BigFloat, Int64}))
println("which(ldexp, Tuple{BigFloat, Clong}) = ", which(ldexp, Tuple{BigFloat, Clong}))
let k = emax + 1
  println("dispatch demo k = emax+1 = ", k)
  for (lab, g) in (("ldexp(big(0.75), Int64(k))", () -> ldexp(big(0.75), Int64(k))),
                   ("ldexp(big(0.75), Clong(k))", () -> ldexp(big(0.75), Clong(k))),
                   ("big(0.75)*exp2(BigFloat(Int64(k)))", () -> big(0.75) * exp2(BigFloat(Int64(k)))))
    r = try g() catch err; "EXC " * sprint(showerror, err) end
    println("  ", lab, " = ", r)
  end
end

function cls(x)
  x isa AbstractFloat || return string(x)
  isnan(x) ? "NaN" : isinf(x) ? "Inf($(sign(x)))" : iszero(x) ? "zero" : "finite($(x))"
end
tryc(g) = try g() catch err; "EXC " * sprint(showerror, err) end

println("== 4. helpers near the limit")
for p in (256, 2, 3, 5, 8, 16, 53)
  setprecision(BigFloat, p) do
    M = floatmax(BigFloat)
    one_ = one(BigFloat)
    println("-- precision ", p, " M=", M)
    for k in (emax - 1, emax, emax + 1, emax + 5, -emax - 5)
      r = tryc(() -> PB._power_factors(BigFloat, k))
      println("  _power_factors(BigFloat, ", k, ") = ", r isa Tuple ? (cls(r[1]), cls(r[2])) : r)
    end
    for (lab, g) in (
        ("_scaled_product(0.75, M, 0)", () -> PB._scaled_product(big(0.75), M, 0)),
        ("_scaled_product(0.25, M, 0)", () -> PB._scaled_product(big(0.25), M, 0)),
        ("_scaled_product(0.75, M, -emax)", () -> PB._scaled_product(big(0.75), M, -emax)),
        ("_scaled_product(M, M, -emax)", () -> PB._scaled_product(M, M, -emax)),
        ("_scaled_product(1, M, 0)", () -> PB._scaled_product(one_, M, 0)),
        ("_scaled_product(1, ldexp(M,-emax), emax)", () -> PB._scaled_product(one_, ldexp(M, Clong(-emax)), emax)),
        ("_scaled_product(0.25, 3, M, 0) [3-factor]", () -> PB._scaled_product(big(0.25), big(3), M, 0)),
        ("_scaled_product(0.75, M, 1, -emax) [3-factor]", () -> PB._scaled_product(big(0.75), M, one_, -emax)),
        ("_scaled_product(1, ldexp(M,-emax), 1, emax) [3-factor]", () -> PB._scaled_product(one_, ldexp(M, Clong(-emax)), one_, emax)),
        ("_scaled_product(0.25, M, 1, 1) [3-factor]", () -> PB._scaled_product(big(0.25), M, one_, 1)))
      println("  ", lab, " = ", cls(tryc(g)))
    end
    x = fill(M, 3)
    f = NormL1Fused(big(0.25), big(Inf))
    println("  _fused_equal_term(0.25, BigFloat, 3, M) = ", cls(tryc(() -> PB._fused_equal_term(big(0.25), BigFloat, 3, M))))
    println("  f(x)  x=fill(M,3) = ", cls(tryc(() -> f(x))), "   expected 0.75M = ", cls(M * big(0.75)))
    out = similar(x)
    println("  value_and_prox!(out,f,x,1) = ", cls(tryc(() -> value_and_prox!(out, f, x, big(1)))))
    x2 = copy(x)
    println("  value_and_prox!(x2,f,x2,1) aliased = ", cls(tryc(() -> value_and_prox!(x2, f, x2, big(1)))))
  end
end

println("== 5. sweep against an independent reference")
# r = exact sum scaled by 2^-Ex (Ex = exponent of max|x|); result value = r * 2^Ex, never formed when it overflows.
struct Ref; r::BigFloat; Ex::Int; end
function refpair(λ1w, x, γw, kind)    # kind :f  -> λ1 Σ|x| ; :vp -> λ1 n |softthreshold(mean, γ λ1)| rounded to p bits
  p = precision(BigFloat)
  setprecision(BigFloat, 4000) do
    xs = filter(!iszero, x)
    isempty(xs) && return Ref(zero(BigFloat), 0)
    Ex = Int(maximum(exponent(BigFloat(a)) for a in xs))
    sc(a) = ldexp(BigFloat(a), Clong(-Ex))
    l1 = BigFloat(λ1w)
    n = length(x)
    if kind === :f
      return Ref(l1 * sum(abs(sc(a)) for a in x), Ex)
    else
      m = sum(sc(a) for a in x) / n
      thr = try ldexp(BigFloat(γw) * l1, Clong(-Ex)) catch; BigFloat(Inf) end
      y = sign(m) * max(abs(m) - thr, zero(m))
      yr = BigFloat(y, RoundNearest; precision = p)       # the stored output
      return Ref(l1 * n * abs(yr), Ex)
    end
  end
end
# classify expected: (:inf | :under | :finite), and the signed relative error
function judge(val, rf::Ref)
  p = precision(BigFloat)
  mM, eM = frexp(floatmax(BigFloat))          # at precision p
  eMin = Int128(exponent(floatmin(BigFloat)))
  setprecision(BigFloat, 4000) do
    iszero(rf.r) && return iszero(val) ? (:zero, 0.0, true) : (:zero, NaN, false)
    rr = BigFloat(rf.r, RoundNearest; precision = p)
    m, e = frexp(rr)
    te = Int128(e) + Int128(rf.Ex)
    if te > eM || (te == eM && BigFloat(m) > BigFloat(mM))
      return (:inf, NaN, val == Inf)
    elseif te < eMin + 2
      return (:under, NaN, true)     # near/below the range: no verdict
    end
    isfinite(val) || return (:finite, NaN, false)
    d = abs(ldexp(BigFloat(val), Clong(-rf.Ex)) - rr) / abs(rr)
    (:finite, Float64(d), d <= 4 * BigFloat(2)^(1 - p))
  end
end

mkvec(kind, n, s, M) = begin
  E = exponent(M)
  tiny = nextfloat(floatmin(BigFloat))
  base = kind === :constM ? fill(M, n) :
         kind === :constHalf ? fill(M / 2, n) :
         kind === :pow2 ? fill(ldexp(one(BigFloat), Clong(E)), n) :
         kind === :small ? fill(tiny, n) :
         kind === :nearconst ? [isodd(i) ? M : prevfloat(M) for i in 1:n] :
         kind === :mixed ? [(M, one(BigFloat), -one(BigFloat), tiny)[mod1(i, 4)] for i in 1:n] :
         kind === :nearemax ? [i == 1 ? M : ldexp(M, Clong(-i)) for i in 1:n] : error()
  s .* base
end

let nfail = 0, ncase = 0, nunder = 0, ninf = 0, fails = String[]
  for p in (2, 3, 5, 8, 16, 53, 256)
    setprecision(BigFloat, p) do
      M = floatmax(BigFloat)
      for λ in (0.25, 1.0, 3.0), n in (1, 2, 3, 5), s in (1, -1),
          kind in (:constM, :constHalf, :pow2, :small, :nearconst, :mixed, :nearemax)
        x = mkvec(kind, n, s, M)
        λ1w = BigFloat(λ)
        f = NormL1Fused(λ1w, BigFloat(Inf))
        allsame = all(==(x[1]), x)
        fv = tryc(() -> f(x))
        out = similar(x)
        vv = tryc(() -> value_and_prox!(out, f, x, big(1)))
        x2 = copy(x)
        va = tryc(() -> value_and_prox!(x2, f, x2, big(1)))
        # reference: f(x) is Inf unless all entries are equal
        rf = refpair(λ1w, x, big(1), :f)
        rv = refpair(λ1w, x, big(1), :vp)
        tag = "p=$p λ1=$λ n=$n s=$s $kind"
        for (what, val, rfx, expinf) in (("f", fv, rf, !allsame), ("vp", vv, rv, false), ("vp-alias", va, rv, false))
          ncase += 1
          if val isa String
            nfail += 1; push!(fails, "FAIL $tag $what EXC $val"); continue
          end
          verdict, err, ok = if expinf
            (:inf, NaN, val == Inf)
          else
            judge(val, rfx)
          end
          verdict === :under && (nunder += 1)
          verdict === :inf && (ninf += 1)
          if ok === false
            nfail += 1
            push!(fails, "FAIL $tag $what val=$(cls(val)) expected=$(verdict) ref=r$(Float64(rfx.r))*2^$(rfx.Ex) relerr=$err")
          end
          if p == 256 || (what == "f" && n == 3 && kind === :constM)
            println("  ", ok === false ? "FAIL" : "PASS", " ", tag, " ", what, " val=", cls(val), " expected=", verdict, " relerr=", err)
          end
        end
      end
    end
  end
  println("cases=", ncase, " expected-Inf/overflow=", ninf, " underflow-skipped=", nunder)
  println("FAIL count = ", nfail)
  foreach(println, fails)
end
println("== done")
