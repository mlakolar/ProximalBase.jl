# Gate for the module-local ldexp of ProximalBase (#747, option C). Usage: julia --project diag747/gate.jl <before.txt>
# <before.txt> was written by diag747/gate_before.jl in an earlier process, before ProximalBase was loaded.
# Every line prints PASS or FAIL; nothing stops at the first failure.
using InteractiveUtils, Printf, Random

const NPASS = Ref(0)
const NFAIL = Ref(0)
function check(name::AbstractString, f::Function)
  ok = try
    f() === true
  catch err
    println("  exception: ", sprint(showerror, err))
    false
  end
  println(ok ? "PASS" : "FAIL", ": ", name)
  ok ? (NPASS[] += 1) : (NFAIL[] += 1)
  ok
end

println("== 0. platform")
println("julia ", VERSION, "  ", Sys.MACHINE, "  sizeof(Clong) = ", sizeof(Clong), "  sizeof(Int) = ", sizeof(Int))
println("MPFR ", Base.MPFR.version(), "  emax = ", Base.MPFR.get_emax(), "  emin = ", Base.MPFR.get_emin())

before = readlines(ARGS[1])
using ProximalBase
const PB = ProximalBase
const emax = Base.MPFR.get_emax()

####################################
println("== 1. the name and Base's method table")
####################################
sigs(f) = sort!(String[string(m.sig) for m in methods(f)])
after_sigs = sigs(Base.ldexp)
before_sigs = [l[5:end] for l in before if startswith(l, "sig=")]
before_n = parse(Int, only(l[3:end] for l in before if startswith(l, "n=")))
before_which = only(l[7:end] for l in before if startswith(l, "which="))
println("Base.ldexp methods before loading ProximalBase: ", before_n, "   after: ", length(methods(Base.ldexp)))
check("PB.ldexp !== Base.ldexp", () -> PB.ldexp !== Base.ldexp)
check("parentmodule(PB.ldexp) === PB", () -> parentmodule(PB.ldexp) === PB)
check("Core.eval(PB, :ldexp) === PB.ldexp (the name resolves to the module's function inside the module)",
      () -> Core.eval(PB, :ldexp) === PB.ldexp)
check("Base.ldexp: number of methods unchanged by loading ProximalBase", () -> length(methods(Base.ldexp)) == before_n)
check("Base.ldexp: the set of method signatures unchanged by loading ProximalBase", () -> after_sigs == before_sigs)
check("which(Base.ldexp, Tuple{BigFloat,Int64}) is the same method as before", () -> string(which(Base.ldexp, Tuple{BigFloat, Int64}).sig) == before_which)
println("which(Base.ldexp, Tuple{BigFloat,Int64}) = ", which(Base.ldexp, Tuple{BigFloat, Int64}))
check(sizeof(Clong) < sizeof(Int) ? "Windows: Base's Int64 method is the Integer fallback" : "Linux/macOS: Base's Int64 method is the exact one",
      () -> (which(Base.ldexp, Tuple{BigFloat, Int64}).sig.parameters[3] === Integer) == (sizeof(Clong) < sizeof(Int)))
println("PB.ldexp methods: ", [string(m.sig) for m in methods(PB.ldexp)])
# a reflection site of the package that now sees the module's function: norms.jl asks hasmethod(ldexp, Tuple{W, Int})
println("hasmethod(PB.ldexp, Tuple{Rational{Int}, Int}) = ", hasmethod(PB.ldexp, Tuple{Rational{Int}, Int}),
        "   hasmethod(Base.ldexp, Tuple{Rational{Int}, Int}) = ", hasmethod(Base.ldexp, Tuple{Rational{Int}, Int}))

####################################
println("== 2. the shadow reaches the calls inside the module (values that are Inf or wrong at master on Windows)")
####################################
check("_scaled_product(0.75, floatmax, 0) is finite at 256 bits", () -> setprecision(BigFloat, 256) do
  isfinite(PB._scaled_product(big(0.75), floatmax(BigFloat), 0))
end)
check("_scaled_product(1, floatmax, 0) is floatmax at 256 bits", () -> setprecision(BigFloat, 256) do
  M = floatmax(BigFloat)
  PB._scaled_product(one(BigFloat), M, 0) == M
end)
check("_power_factors(BigFloat, emax-1): finite first factor at p = 2", () -> setprecision(BigFloat, 2) do
  isfinite(PB._power_factors(BigFloat, emax - 1)[1])
end)
function factors_ok(p, k)
  setprecision(BigFloat, p) do
    s = PB._power_factors(BigFloat, k)
    all(isfinite, s) && isequal(PB._scale_value(one(BigFloat), s), Base.ldexp(one(BigFloat), Clong(k)))
  end
end
check("_power_factors(BigFloat, k): both factors finite, product 2^k, p in (2, 5, 16, 256), k near emax",
      () -> all(factors_ok(p, k) for p in (2, 5, 16, 256), k in (emax - 1, emax, emax + 5, -emax - 5, -emax + 1)))
check("NormL1Fused(0.25, Inf)(fill(floatmax, 3)) == 0.75 floatmax at 256 bits", () -> setprecision(BigFloat, 256) do
  M = floatmax(BigFloat)
  f = NormL1Fused(big(0.25), big(Inf))
  y = f(fill(M, 3))
  isfinite(y) && y == BigFloat(0.75) * M
end)
check("value_and_prox! of the same, separate and aliased", () -> setprecision(BigFloat, 256) do
  M = floatmax(BigFloat)
  f = NormL1Fused(big(0.25), big(Inf))
  x = fill(M, 3)
  out = similar(x)
  v = value_and_prox!(out, f, x, one(BigFloat))
  a = copy(x)
  w = value_and_prox!(a, f, a, one(BigFloat))
  v == BigFloat(0.75) * M && w == v
end)
check("cone: IndRotatedSOC()(fill(2^35, 3)) == 0 at p = 5", () -> setprecision(BigFloat, 5) do
  iszero(IndRotatedSOC()(fill(Base.ldexp(one(BigFloat), Clong(35)), 3)))
end)
check("cone: IndSOC()(fill(floatmax, 2)) == 0 at p = 5", () -> setprecision(BigFloat, 5) do
  iszero(IndSOC()(fill(floatmax(BigFloat), 2)))
end)
check("cone: prox(IndSOC(), fill(floatmax, 2)) == the argument at p = 5", () -> setprecision(BigFloat, 5) do
  z = fill(floatmax(BigFloat), 2)
  prox(IndSOC(), z) == z
end)
check("parts: Scaled(NormL1Fused(1, 0), 2^-16777217)(fill(floatmax, 2)) finite, = floatmax 2^-16777216, p = 5", () -> setprecision(BigFloat, 5) do
  M = floatmax(BigFloat)
  f = Scaled(NormL1Fused(one(BigFloat), zero(BigFloat)), Base.ldexp(one(BigFloat), Clong(-16_777_217)))
  y = f(fill(M, 2))
  ref = Base.ldexp(M, Clong(-16_777_216))
  isfinite(ref) && isfinite(y) && abs(y - ref) <= Base.ldexp(abs(ref), Clong(2 - 5))
end)
check("nested chain: Scaled(Scaled(NormL1Fused(1, Inf), 2^545259518), 2^545259519)(fill(2^-33554432, 2)) == 2^1056964606, p = 5",
      () -> setprecision(BigFloat, 5) do
  c = Base.ldexp(one(BigFloat), Clong(-33_554_432))
  a = Base.ldexp(one(BigFloat), Clong(545_259_518))
  b = Base.ldexp(one(BigFloat), Clong(545_259_519))
  g = NormL1Fused(one(BigFloat), BigFloat(Inf))
  h = Scaled(Scaled(g, a), b)
  y = h(fill(c, 2))
  ref = Base.ldexp(one(BigFloat), Clong(1_056_964_606))
  isfinite(ref) && isfinite(y) && abs(y - ref) <= Base.ldexp(abs(ref), Clong(2 - 5))
end)

####################################
println("== 3. delegation for the other types")
####################################
function outcome(g)
  try
    (:ok, g())
  catch err
    (:err, typeof(err))
  end
end
same(a, b) = a[1] === b[1] && (a[1] === :err ? a[2] === b[2] :
  (typeof(a[2]) === typeof(b[2]) && isequal(a[2], b[2]) && (!(a[2] isa AbstractFloat) || signbit(a[2]) == signbit(b[2]))))
function grid(T)
  xs = T[zero(T), -zero(T), one(T), -one(T), T(1.5), T(0.375), T(3), floatmin(T), nextfloat(zero(T)), prevfloat(floatmin(T)),
         nextfloat(floatmin(T)), floatmax(T), -floatmax(T), prevfloat(floatmax(T)), T(NaN), T(Inf), -T(Inf)]
  xs
end
ns = Any[0, 1, -1, 2, 5, -5, 10, 127, -127, 128, 1023, 1024, 1100, -1100, 2000, -2000, 10^6, -10^6, 2^40, -2^40]
for I in (Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64)
  push!(ns, typemax(I), typemin(I))
end
push!(ns, Int8(7), Int16(-300), Int32(5), UInt8(3), BigInt(5), BigInt(-1100), BigInt(2)^70, -BigInt(2)^70, Int128(2)^80, -Int128(2)^80)
for T in (Float16, Float32, Float64)
  bad = String[]
  for x in grid(T), n in ns
    a, b = outcome(() -> PB.ldexp(x, n)), outcome(() -> Base.ldexp(x, n))
    same(a, b) || push!(bad, "$T $x $(typeof(n)) $n: PB $a, Base $b")
  end
  isempty(bad) || foreach(println, first(bad, 5))
  check("PB.ldexp equals Base.ldexp for $T over $(length(grid(T))) values and $(length(ns)) shifts (results, signed zeros, NaN, thrown exception types)", () -> isempty(bad))
end
f64(x, n) = PB.ldexp(x, n)
g64(x, n) = Base.ldexp(x, n)
f32(x, n) = PB.ldexp(x, n)
g32(x, n) = Base.ldexp(x, n)
f16(x, n) = PB.ldexp(x, n)
g16(x, n) = Base.ldexp(x, n)
alloc(h, x, n) = @allocated h(x, n)
normir(s) = replace(replace(s, r"julia_[fg](64|32|16)_[0-9]+" => "julia_F"), r"L[0-9]+" => "L")
for (T, f, g) in ((Float64, f64, g64), (Float32, f32, g32), (Float16, f16, g16))
  fi = sprint(io -> code_llvm(io, f, Tuple{T, Int}; debuginfo = :none))
  gi = sprint(io -> code_llvm(io, g, Tuple{T, Int}; debuginfo = :none))
  ok = normir(fi) == normir(gi)
  ok || println("---- PB.ldexp\n", fi, "---- Base.ldexp\n", gi)
  check("LLVM IR of PB.ldexp equals Base.ldexp for ($T, Int) (names and labels normalized; $(count(==('\n'), fi)) lines)", () -> ok)
  x, n = T(1.5), 3
  f(x, n); g(x, n)
  check("zero allocations of PB.ldexp for ($T, Int), warm", () -> alloc(f, x, n) == 0 && alloc(g, x, n) == 0)
end
println("code_warntype-style: ", Base.return_types(f64, Tuple{Float64, Int}), " ", Base.return_types(f32, Tuple{Float32, Int}))

####################################
println("== 4. BigFloat exactness")
####################################
exp_clong(n) = Clong(clamp(n, typemin(Clong), typemax(Clong)))
for p in (2, 3, 5, 8, 16, 53, 256)
  setprecision(BigFloat, p) do
    xs = BigFloat[big(0.75), -big(0.75), one(BigFloat), -one(BigFloat), big(0.375), floatmax(BigFloat), -floatmax(BigFloat),
                  nextfloat(zero(BigFloat)), floatmin(BigFloat), big(NaN), big(Inf), -big(Inf), zero(BigFloat), -zero(BigFloat)]
    shifts = Any[0, 1, -1, 3, 1000, -1000, emax - 1, emax, emax + 1, -emax, -emax - 1, 2^40, -2^40, typemax(Int), typemin(Int),
                 Int8(7), Int128(emax) + 5, -Int128(emax) - 5, BigInt(2)^80, -BigInt(2)^80, UInt64(emax) + 1, typemax(UInt64)]
    bad = String[]
    for x in xs, n in shifts
      a = outcome(() -> PB.ldexp(x, n))
      b = outcome(() -> Base.ldexp(x, exp_clong(n)))
      same(a, b) || push!(bad, "p=$p x=$x $(typeof(n)) $n: PB $a, Base-with-Clong $b")
    end
    isempty(bad) || foreach(println, first(bad, 5))
    check("p=$p: PB.ldexp(x, n) equals Base.ldexp(x, Clong(clamp(n))) for $(length(xs)) values and $(length(shifts)) shifts", () -> isempty(bad))
    check("p=$p: shift beyond the range overflows to Inf and underflows to a signed zero", () -> begin
      o = one(BigFloat)
      PB.ldexp(o, typemax(Int)) == Inf && PB.ldexp(-o, typemax(Int)) == -Inf &&
        iszero(PB.ldexp(o, typemin(Int))) && !signbit(PB.ldexp(o, typemin(Int))) &&
        iszero(PB.ldexp(-o, typemin(Int))) && signbit(PB.ldexp(-o, typemin(Int)))
    end)
    check("p=$p: 2^(emax-1) exact, 2^emax overflows, a shift of an Int past a 32-bit Clong is exact", () -> begin
      o = one(BigFloat)
      PB.ldexp(o, emax - 1) == Base.ldexp(o, Clong(emax - 1)) && PB.ldexp(o, emax) == Inf &&
        isfinite(PB.ldexp(big(0.375), Int(emax + 1)))
    end)
  end
end
check("the argument's own precision differs from the ambient one", () -> begin
  x = BigFloat(0.7; precision = 60)
  r = setprecision(BigFloat, 4) do
    a = PB.ldexp(x, 3)
    b = Base.ldexp(x, Clong(3))
    isequal(a, b) && precision(a) == precision(b)
  end
  r2 = setprecision(BigFloat, 100) do
    y = BigFloat(0.7; precision = 10)
    a = PB.ldexp(y, Int64(5))
    isequal(a, Base.ldexp(y, Clong(5))) && precision(a) == precision(Base.ldexp(y, Clong(5)))
  end
  r && r2
end)

####################################
println("== 6. the ext/ modules")
####################################
println("ldexp calls in ext/ (the package-module ldexp does not cover another module):")
for f in sort(readdir("ext"))
  for (i, l) in enumerate(eachline(joinpath("ext", f)))
    occursin("ldexp", l) && println("  ext/$f:$i: ", strip(l))
  end
end
println("fact: both calls are in _group_levels(::Type{T}, γ, λ) with R = promote_type(T, float(typeof(γ)), float(typeof(λ))), so R is BigFloat for BigFloat")
println("      levels; the shifts are fld(e, 2) and e - fld(e, 2) of (m, e) = frexp(c), whose exponent for a BigFloat is a Clong (Int32 on Windows):")
println("      the shift is then Base's exact method, so the calls are exact without the shadow.")
println("      frexp(BigFloat) exponent type here: ", typeof(frexp(big(1.0))[2]), "   Clong = ", Clong)
check("frexp(::BigFloat) returns a Clong exponent, so ext/ shifts of BigFloat are exact Base calls", () -> typeof(frexp(big(1.0))[2]) === Clong)
check("their shift fld(e, 2) of a Clong stays a Clong", () -> typeof(fld(frexp(big(1.0))[2], 2)) === Clong)

println("== summary")
println("julia ", VERSION, " ", Sys.MACHINE, ": PASS ", NPASS[], "  FAIL ", NFAIL[])
println("== done")
