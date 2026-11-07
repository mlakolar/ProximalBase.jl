# Issue #747 differential: one line per case, for bit-for-bit diffing between base and head.
using ProximalBase, Random
const PB = ProximalBase
out = open(ARGS[1], "w")
fmt(v::BigFloat) = string(v)
fmt(v::AbstractFloat) = repr(v)
fmt(v::Union{Tuple,AbstractArray}) = "(" * join(map(fmt, v), ", ") * ")"
fmt(v) = repr(v)
rec(tag, g) = println(out, tag, " => ", try fmt(g()) catch err; "ERR " * string(typeof(err)) end)

function specials(T)
  [T(NaN), T(Inf), -T(Inf), zero(T), -zero(T), T(1), T(-1), T(0.5), T(3), nextfloat(zero(T)), floatmin(T),
   nextfloat(floatmin(T)), floatmax(T), -floatmax(T), floatmax(T) / 2, prevfloat(floatmax(T))]
end
function vectors(T, rng)
  sp = specials(T)
  vs = Vector{Vector{T}}()
  for v in sp
    for n in (1, 2, 3, 5)
      push!(vs, fill(v, n))
    end
  end
  for i in eachindex(sp), j in eachindex(sp)
    push!(vs, [sp[i], sp[j]]); push!(vs, [sp[i], sp[j], sp[i]])
  end
  for n in (2, 3, 5, 9), r in 1:4
    push!(vs, T.(randn(rng, n)))
    push!(vs, T.(randn(rng, n)) .* floatmax(T) ./ 4)
    push!(vs, T.(rand(rng, 1:3, n)))
  end
  vs
end

function run_type(T, label)
  rng = MersenneTwister(747)
  grid = ((0.25, Inf), (0.0, Inf), (1.0, Inf), (3.0, Inf), (0.25, 0.5), (1.0, 1.0), (Inf, Inf), (0.5, 0.0))
  vs = vectors(T, rng)
  for (l1, l2) in grid, γ in (1.0, 0.5), (i, x) in enumerate(vs)
    tag = "$label λ=($l1,$l2) γ=$γ x$i"
    f = NormL1Fused(T(l1), T(l2))
    g = T(γ)
    rec("$tag f", () -> f(x))
    rec("$tag prox", () -> prox(f, x, g))
    rec("$tag vprox", () -> (o = similar(x); (value_and_prox!(o, f, x, g), o)))
    rec("$tag alias", () -> (a = copy(x); (value_and_prox!(a, f, a, g), a)))
    if length(x) == 2
      rec("$tag pair", () -> f((x[1], x[2])))
      rec("$tag arrays", () -> f(([x[1]], [x[2]])))
      rec("$tag arrays-prox", () -> prox(f, ([x[1]], [x[2]]), g))
      rec("$tag arrays-vprox", () -> (o1 = [zero(T)]; o2 = [zero(T)]; (value_and_prox!((o1, o2), f, ([x[1]], [x[2]]), g), o1, o2)))
    end
  end
  sp = specials(T)
  emx = Base.MPFR.get_emax()
  ks = T === BigFloat ? (0, 1, -1, 5, -5, emx - 1, emx, emx + 1, emx + 5, -emx - 5, -emx, 1000, -1000) :
       (0, 1, -1, 5, -5, 127, 128, 1023, 1024, 1100, -1100, 2000, -2000)
  for k in ks
    rec("$label _power_factors k=$k", () -> PB._power_factors(T, k))
    rec("$label _power_factors_exact k=$k", () -> PB._power_factors_exact(T, k))
    for (i, w) in enumerate(sp), (j, v) in enumerate(sp)
      rec("$label _scaled_product w$i v$j k=$k", () -> PB._scaled_product(w, v, k))
      rec("$label _scaled_product_exact w$i v$j k=$k", () -> PB._scaled_product_exact(w, v, k))
    end
    for (i, a) in enumerate(sp), (j, b) in enumerate((T(0.25), T(1), T(3), floatmax(T))), (m, c) in enumerate((T(1), floatmax(T), T(0.75)))
      rec("$label _scaled_product3 a$i b$j c$m k=$k", () -> PB._scaled_product(a, b, c, k))
    end
  end
  for n in (1, 2, 3, 5), c in sp, λ in (0.25, 1.0, 3.0)
    rec("$label _fused_equal_term λ=$λ n=$n c=$c", () -> PB._fused_equal_term(T(λ), T, n, c))
  end
end


# ---- other families that call _power_factors / _scaled_product (and raw ldexp companions) with BigFloat: they must not change
function pool(T)
  M = floatmax(T)
  T[one(T), T(2)^35, T(1.5) * T(2)^35, M, M / 2, T(2)^-35]
end
function run_families(T, label)
  P = pool(T)
  M = floatmax(T)
  g = one(T)
  # the entries are named by their index in the pool (and a sign), so that the lines stay short
  # cones: heads and tails mixed, scales near 1, near 2^35 and near floatmax
  for (it, t) in enumerate(P), (ia, a) in enumerate(P), (ib, b) in enumerate(P), st in (1, -1)
    x = T[st * t, a, b]
    f = IndSOC()
    id = "$(st * it),$ia,$ib"
    rec("fam=cone-soc $label x=$id f", () -> f(x))
    rec("fam=cone-soc $label x=$id prox", () -> prox(f, x, g))
  end
  for (iu, u) in enumerate(P), (iv, v) in enumerate(P), (iw, w) in enumerate(P), su in (1, -1), sv in (1, -1)
    x = T[su * u, sv * v, w]
    f = IndRotatedSOC()
    id = "$(su * iu),$(sv * iv),$iw"
    rec("fam=cone-rsoc $label x=$id f", () -> f(x))
    rec("fam=cone-rsoc $label x=$id prox", () -> prox(f, x, g))
  end
  # sets: balls, sphere, simplex; norms
  fs = Any[]
  for (ir, r) in enumerate((one(T), T(2)^35, M / 2))
    push!(fs, ("IndBallL1 r$ir", IndBallL1(r)), ("IndBallL2 r$ir", IndBallL2(r)), ("IndSphereL2 r$ir", IndSphereL2(r)),
           ("IndSimplex r$ir", IndSimplex(r)), ("IndSimplexLE r$ir", IndSimplex(r; equality = false)))
  end
  push!(fs, ("NormL2 0.5", NormL2(T(0.5))), ("NormL1 0.5", NormL1(T(0.5))), ("NormL1 3", NormL1(T(3))))
  for (nm, f) in fs, (ia, a) in enumerate(P), (ib, b) in enumerate(P), (ic, c) in enumerate(P)
    x = T[a, b, c]
    id = "$ia,$ib,$ic"
    rec("fam=sets $label $nm x=$id f", () -> f(x))
    rec("fam=sets $label $nm x=$id prox", () -> prox(f, x, g))
  end
  # ordered functions share _fused_scaled_value / _fused_scaled_parts with NormL1Fused
  ords = Any[("TotalVariation1D", TotalVariation1D(T(0.5))), ("FusedLasso 0.25,0.5", FusedLasso(T(0.25), T(0.5))),
             ("FusedLasso 1,3", FusedLasso(T(1), T(3))), ("SLOPE", SLOPE(T[3, 2, 1] ./ 4))]
  for (nm, f) in ords, (ia, a) in enumerate(P), (ib, b) in enumerate(P), (ic, c) in enumerate(P), st in (1, -1)
    x = T[st * a, b, c]
    id = "$(st * ia),$ib,$ic"
    rec("fam=ordered $label $nm x=$id f", () -> f(x))
    rec("fam=ordered $label $nm x=$id prox", () -> prox(f, x, g))
    nm == "SLOPE" || rec("fam=ordered $label $nm x=$id scaled", () -> (T(0.5) * f)(x))
  end
  # FiberSum of NormL1Fused (the fibers call the shared recovery with their own scaling), K = 2 equal entries and K = 3
  rng = MersenneTwister(7470)
  for (nm, K, f) in (("fib-inf", 2, NormL1Fused(T(0.25), T(Inf))), ("fib-fin", 3, NormL1Fused(T(0.25), T(0.5)))), r in 1:60
    A = zeros(T, 2, 2, K)
    for i in 1:2, j in 1:2
      c = rand(rng, P) * rand(rng, (1, -1))
      for k in 1:K
        A[i, j, k] = (K == 2 && i != j) || rand(rng) < 0.5 ? c : rand(rng, P)
      end
    end
    fb = FiberSum(f; diagonal = nothing)
    rec("fam=fibers $label $nm r=$r f", () -> fb(A))
    rec("fam=fibers $label $nm r=$r vprox", () -> (Y = similar(A); (value_and_prox!(Y, fb, A, g), Y)))
  end
  # Scaled(NormL1Fused) takes the value as parts (the NormL1Fused family)
  for (iα, α) in enumerate((T(0.5), T(1.5))), (ia, a) in enumerate(P), (ib, b) in enumerate(P)
    f = NormL1Fused(T(0.25), T(Inf))
    rec("fam=scaledNF $label α$iα x=$ia,$ia,$ib f", () -> (α * f)(T[a, a, b]))
    rec("fam=scaledNF $label α$iα y=$ia,$ia f", () -> (α * f)(T[a, a]))
  end
end

run_type(Float32, "Float32")
run_type(Float64, "Float64")
for p in (2, 53, 256)
  setprecision(BigFloat, p) do
    run_type(BigFloat, "BigFloat$p")
  end
end
run_families(Float32, "Float32")
run_families(Float64, "Float64")
for p in (2, 5, 53, 256)
  setprecision(BigFloat, p) do
    run_families(BigFloat, "BigFloat$p")
  end
end
close(out)
println("wrote ", ARGS[1])
