# Issue #747: the mechanisms behind the differences that remain between Windows and Linux (with the exponent range of Windows) at
# the head. Usage: julia --project diag747/probe_mismatch.jl   (EXPRANGE=win as for diff2.jl)
using ProximalBase
const PB = ProximalBase
if get(ENV, "EXPRANGE", "") == "win"
  Base.MPFR.set_emax!(2^30 - 1)
  Base.MPFR.set_emin!(-(2^30 - 1))
end
println("julia ", VERSION, "  ", Sys.MACHINE, "  Clong = ", Clong, "  emax = ", Base.MPFR.get_emax())
emax = Base.MPFR.get_emax()

println("== 1. raw frexp exponents of a BigFloat are Clongs: their sum wraps where a C long is 32 bits")
setprecision(BigFloat, 256) do
  a = Base.ldexp(one(BigFloat), Clong(-1000))
  b = Base.ldexp(one(BigFloat), Clong(-(emax - 10)))
  c = b
  p = a * b * c                              # underflows to 0 (the exact product is 2^(-1000 - 2 emax + 20))
  ea, eb, ec = frexp(a)[2], frexp(b)[2], frexp(c)[2]
  println("typeof(frexp(a)[2]) = ", typeof(ea), "   ea + eb + ec = ", ea + eb + ec, " :: ", typeof(ea + eb + ec),
          "   (the exact sum ", Int128(ea) + Int128(eb) + Int128(ec), ")")
  println("_exceptional_product(a, b, c, a*b*c) = ", PB._exceptional_product(a, b, c, p), "   (the product underflows: 0 is right)")
  println("_positive_product(a, b, c)            = ", PB._positive_product(a, b, c))
  a2 = Base.ldexp(one(BigFloat), Clong(emax - 2)); b2 = a2; c2 = Base.ldexp(one(BigFloat), Clong(-(2 * emax - 6)))
  println("overflow side: a2 b2 c2 = 2^(emax-2), 2^(emax-2), 2^(-2emax+6): the product is 2^(emax-... ) ",
          PB._exceptional_product(a2, b2, c2, a2 * b2 * c2), "   exact exponent ", 2 * (emax - 2) - (2 * emax - 6))
end

println("== 2. an Int64 operand of a BigFloat operation where a C long is 32 bits is converted at the working precision")
setprecision(BigFloat, 2) do
  t = big(1.5)
  println("p = 2: big(1.5) / 5 = ", t / 5, "   big(1.5) / big(5) = ", t / big(5), "   (5 is rounded to 4 at two bits)   Int32(5): ", t / Int32(5))
  println("       big(1.5) * 5 = ", t * 5, "   big(1.5) + 5 = ", t + 5)
end

println("== 3. the cone projection at [-floatmax, 2^-35, 1] (IndRotatedSOC prox), 256 bits")
setprecision(BigFloat, 256) do
  M = floatmax(BigFloat)
  x = [-M, Base.ldexp(one(BigFloat), Clong(-35)), one(BigFloat)]
  println("prox = ", prox(IndRotatedSOC(), x, one(BigFloat)))
  a, b, s, k = PB._rotatedsoc_frame(x, 1:3)
  println("frame a, b, s, k = ", a, ", ", b, ", ", s, ", ", k, "   typeof(k) = ", typeof(k))
  d = hypot(a - b, sqrt(BigFloat(2)) * s); r = s / (d + abs(a - b))
  println("d = ", d, "   r = ", r, "   s/d = ", s / d, "   λ r (s/d) as _positive_product(λ, r, s/d) with λ = 2^-35: ",
          PB._positive_product(Base.ldexp(one(BigFloat), Clong(-35)), r, s / d))
end

println("== 4. QuadraticFunction at [5 x floatmin-ish, ...] value, 2 bits")
setprecision(BigFloat, 2) do
  M = floatmax(BigFloat)
  A = BigFloat[2 1 0; 1 2 1; 0 1 2]
  f = QuadraticFunction(A, BigFloat[1, 2, 3], one(BigFloat))
  x = [Base.ldexp(big(1.5), Clong(emax ÷ 4)), -M / 2, one(BigFloat)]
  println("f(x) = ", f(x), "   f at [1, -M, 1] = ", f([one(BigFloat), -M, one(BigFloat)]))
end
println("== done")
