# Pilot 1: NormL1.  Independent formulas: value = λ Σ w|x|, prox = soft threshold sign(x)·max(|x|−γλw,0).
using ProximalBase, LinearAlgebra
const PB = ProximalBase
chk(label, ok) = println("  check ", rpad(label, 58), ok ? "OK" : "FAIL")
tryshow(label, f) = try; r = f(); println("  ", rpad(label, 52), " -> ", repr(r)); r; catch e; println("  ", rpad(label, 52), " -> THROWS ", typeof(e), ": ", first(split(sprint(showerror, e), '\n'))); nothing; end
println("== constructors / validation")
tryshow("NormL1()", () -> NormL1())
tryshow("NormL1(0.5)", () -> NormL1(0.5))
tryshow("NormL1(0.5, [1.0,2.0])", () -> NormL1(0.5, [1.0,2.0]))
tryshow("NormL1(1)  (Int level)", () -> NormL1(1))
tryshow("NormL1(0.5, [1,2]) (Int weights)", () -> NormL1(0.5, [1,2]))
tryshow("NormL1(-0.1)", () -> NormL1(-0.1))
tryshow("NormL1(NaN)", () -> NormL1(NaN))
tryshow("NormL1(Inf)", () -> NormL1(Inf))
tryshow("NormL1(0.5, [1.0,-1.0])", () -> NormL1(0.5, [1.0,-1.0]))
tryshow("NormL1(0.5, [1.0,NaN])", () -> NormL1(0.5, [1.0,NaN]))
tryshow("NormL1(0.5, [1.0,2.0,3.0])([1.0,2.0]) (weights too long)", () -> NormL1(0.5,[1.0,2.0,3.0])([1.0,2.0]))
tryshow("NormL1(0.5, ones(2,2)) on 2x2 matrix", () -> NormL1(0.5, ones(2,2))(ones(2,2)))
tryshow("NormL1(1+1im)", () -> NormL1(1+1im))
tryshow("NormL1(0.5)([1.0+1im])", () -> NormL1(0.5)([1.0+1im]))
println("== traits")
g = NormL1(0.5)
println("  issmooth=", issmooth(g), " isproximable=", isproximable(g), " isseparable=", isseparable(g), " isconvex=", isconvex(g), " iszeropreserving=", iszeropreserving(g))
println("== value  f(x)=λ Σ w|x|")
x = [2.0, -1.0, 0.0, 0.25]; w = [1.0, 2.0, 3.0, 0.0]; λ = 0.5
gw = NormL1(λ, w)
println("  g(x) = ", g(x), "   independent 0.5*sum(abs,x) = ", 0.5*sum(abs, x)); chk("unweighted value", g(x) == 0.5*sum(abs, x))
println("  gw(x) = ", gw(x), "   independent λ*Σ w|x| = ", λ*sum(w .* abs.(x))); chk("weighted value", gw(x) ≈ λ*sum(w .* abs.(x)))
println("== prox  soft-threshold; step γ")
soft(x, t) = sign(x) * max(abs(x) - t, 0)
for γ in (0.0, 0.1, 1.0, 4.0)
  p = prox(gw, x, γ); ref = soft.(x, γ*λ .* w)
  println("  γ=", γ, ": prox = ", p, "  independent = ", ref); chk("prox γ=$γ", p == ref)
end
println("  prox(g,x) default γ=1: ", prox(g, x))
out = similar(x)
r1 = prox!(out, gw, x, 1.0)
println("  prox!(out,gw,x,1.0) returns out? ", r1 === out, "; out = ", out)
r2 = value_and_prox!(out, gw, x, 1.0)
println("  value_and_prox!(out,gw,x,1.0) = ", r2, " (a scalar, ", typeof(r2), "); f(out) = ", gw(out)); chk("value_and_prox! returns f at stored output", r2 == gw(out) && r2 isa Real)
println("  prox!(x,gw,x,1.0) in place on a copy: ", (c = copy(x); prox!(c, gw, c, 1.0); c)); 
tryshow("prox with array step [0.1,0.2,0.3,0.4]", () -> prox(g, x, [0.1,0.2,0.3,0.4]))
chk("array step formula", prox(g, x, [0.1,0.2,0.3,0.4]) == soft.(x, [0.1,0.2,0.3,0.4] .* λ))
tryshow("prox with negative step", () -> prox(g, x, -1.0))
tryshow("prox with NaN step", () -> prox(g, x, NaN))
tryshow("prox with step Inf (finite x, positive level)", () -> prox(g, x, Inf))
tryshow("prox of NaN entry", () -> prox(g, [NaN, 1.0], 1.0))
tryshow("prox! with wrong output length", () -> prox!(zeros(3), g, x, 1.0))
tryshow("prox with weight length mismatch", () -> prox(NormL1(0.5,[1.0,2.0]), x, 1.0))
tryshow("gradient(g, x) (not smooth)", () -> gradient(g, x))
println("== scalar/coordinate hooks")
for (v, i, γ) in ((2.0, 1, 1.0), (-1.0, 2, 1.0), (0.4, 3, 0.1), (1.0, 4, 3.0))
  p = prox_block(gw, v, i, γ); ref = soft(v, γ*λ*w[i])
  println("  prox_block(gw, $v, $i, $γ) = ", p, "  independent ", ref); chk("prox_block i=$i", p == ref)
end
println("  value_entry(gw, -1.0, 2) = ", value_entry(gw, -1.0, 2), "  independent ", λ*w[2]*1.0)
tryshow("prox_block(g, [0.5,1.0], 1, 1.0) (vector/group form)", () -> prox_block(g, [0.5,1.0], 1, 1.0))
println("== kkt_violation  (v = gradient of the smooth part, supplied)")
v = [0.9, -0.3, 0.8, 0.1]; cs = λ .* w
println("  weights λw = ", cs, "  x = ", x, "  v = ", v)
for i in 1:4
  s = kkt_violation(gw, v, i); ref = abs(v[i]) - cs[i]
  println("  screen kkt_violation(gw,v,$i) = ", s, "  independent |v|-λw = ", ref); chk("screen $i", s ≈ ref)
end
for k in 1:4
  s = kkt_violation(gw, v, x, k)
  ref = x[k] == 0 ? abs(v[k]) - cs[k] : abs(v[k] + cs[k]*sign(x[k]))
  println("  point kkt_violation(gw,v,x,$k) = ", s, "  independent ", ref); chk("point form $k", s ≈ ref)
end
full = kkt_violation(gw, v, x); reff = max(0, maximum(k -> x[k]==0 ? abs(v[k])-cs[k] : abs(v[k]+cs[k]*sign(x[k])), 1:4))
println("  full-point kkt_violation(gw,v,x) = ", full, "  independent ", reff); chk("full-point", full ≈ reff)
println("== representations (prox at γ=1 against the dense vector result)")
xd = [2.0, -1.0, 0.0, 0.25]; refp = prox(gw, xd, 1.0)
reps = Pair{String,Any}[
 "dense Vector{Float64}" => xd,
 "strided view" => (b = zeros(8); b[1:2:7] .= xd; view(b, 1:2:7)),
 "SparseIterate (vector)" => (s = SparseIterate(4); for i in 1:4; iszero(xd[i]) || (s[i] = xd[i]); end; s),
 "BlockIterate" => BlockIterate(copy(xd), [[1,2],[3,4]]),
 "Float32 vector" => Float32.(xd),
 "BigFloat vector" => BigFloat.(xd),
 "Vector{Int}" => [2, -1, 0, 3],
]
for (nm, xr) in reps
  try
    p = prox(gw, xr, 1.0)
    println("  ", rpad(nm, 26), " prox ok; result type ", typeof(p), "; values ", collect(p))
  catch e
    println("  ", rpad(nm, 26), " THROWS ", typeof(e))
  end
end
println("  Float32 data, Float64 NormL1: eltype(prox(g, Float32[2,-1], 1.0)) = ", eltype(prox(g, Float32[2,-1], 1.0)))
println("  Float32 data, Float32 level: eltype(prox(NormL1(0.5f0), Float32[2,-1], 1.0)) = ", eltype(prox(NormL1(0.5f0), Float32[2,-1], 1.0)))
M = [2.0 -1.0; 0.0 0.25]
println("  dense 2x2 matrix, entrywise: prox(NormL1(0.5), M, 1.0) = ", prox(NormL1(0.5), M, 1.0))
S = Symmetric([2.0 -1.0; -1.0 0.25])
println("  Symmetric: prox -> ", typeof(prox(NormL1(0.5), S, 1.0)), " ", prox(NormL1(0.5), S, 1.0))
tryshow("Symmetric same-kind output prox!(similar(S), g, S, 1.0)", () -> prox!(similar(S), g, S, 1.0))
sp = SparseIterate(2,2); sp[1,1]=2.0; sp[2,2]=0.25; sp[1,2] = -1.0
println("  SparseIterate matrix: prox -> ", typeof(prox(NormL1(0.5), sp, 1.0)), " nactive(out)=", nactive(prox(NormL1(0.5), sp, 1.0)))
ss = SymmetricSparseIterate([2.0 -1.0; -1.0 0.25])
ps = prox(NormL1(0.5), ss, 1.0)
println("  SymmetricSparseIterate: prox -> ", typeof(ps), " matrixview = ", Matrix(matrixview(ps)))
tryshow("SymmetricSparseIterate with array step", () -> prox(NormL1(0.5), ss, [1.0 1.0; 1.0 1.0]))
tryshow("SymmetricSparseIterate kkt_violation(g,v,x) full-point", () -> kkt_violation(NormL1(0.5), ones(2,2), ss))
println("  SymmetricSparseIterate NormL1(0.5)(ss) = ", NormL1(0.5)(ss), "  (off-diagonals counted twice: 0.5*(2+0.25+2*1) = ", 0.5*(2+0.25+2), ")")
println("== aliasing")
c = copy(x); prox!(c, g, c, 1.0); println("  in-place prox!(c,g,c,1.0) supported: ", c)
tryshow("gradient aliasing n/a (no gradient)", () -> nothing)
println("== types: 2.0*g, g*2.0, α*g")
println("  typeof(2.0*g) = ", typeof(2.0*g), "; (2.0*g)(x) = ", (2.0*g)(x), " expected ", 2*g(x))
println("== show / repr")
println("  repr(NormL1(0.5)) = ", repr(NormL1(0.5)))
println("  repr(NormL1(0.5,[1.0,2.0])) = ", repr(NormL1(0.5,[1.0,2.0])))
println("  display:"); show(stdout, MIME("text/plain"), NormL1(0.5)); println()
