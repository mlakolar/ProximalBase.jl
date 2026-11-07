# Issue #747 differential: one line per case, for bit-for-bit comparison of base, head and the oracle platform.
# Usage: julia --project diag747/diff2.jl <out.txt>   (EXPRANGE=win sets MPFR's exponent range to that of Windows first)
if get(ENV, "EXPRANGE", "") == "win"
  Base.MPFR.set_emax!(2^30 - 1)
  Base.MPFR.set_emin!(-(2^30 - 1))
end
using ProximalBase, Random, SparseArrays
const PB = ProximalBase
out = open(ARGS[1], "w")
atexit(() -> isopen(out) && close(out))
fmt(v::BigFloat) = string(v)
fmt(v::AbstractFloat) = repr(v)
fmt(v::Union{Tuple,AbstractArray}) = "(" * join(map(fmt, v), ", ") * ")"
fmt(v) = repr(v)
rec(tag, g) = println(out, tag, " => ", try fmt(g()) catch err; "ERR " * string(typeof(err)) end)
# a group of rows that fails as a whole (an error outside rec) prints one line and the run goes on with the next group
function guard(name, f)
  try
    f()
  catch err
    println(out, "harness ", name, " => ERR ", typeof(err))
    println(stderr, "harness group ", name, " failed: ", sprint(showerror, err))
  end
  flush(out)
end

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
    for (i, w) in enumerate(sp), (j, v) in enumerate(sp)
      rec("$label _scaled_product w$i v$j k=$k", () -> PB._scaled_product(w, v, k))
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
  # Scaled, nested Scaled and Translate of NormL1Fused with a FINITE fusion level (zero included): the value of a vector near
  # floatmax overflows, so the wrappers take the parts of the value and restore them with the original _scaled_product. The
  # parts path is unchanged: these lines must not change on any platform.
  emx = T === BigFloat ? Base.MPFR.get_emax() : exponent(floatmax(T)) + 1
  q = emx ÷ 64
  exps = unique((-16_777_217, -q, -q - 1, q, q + 1, -1, 0, 1, -emx ÷ 2, -emx + 5, emx ÷ 2, 5 - emx ÷ 3))
  exps = T === BigFloat ? exps : [e for e in exps if abs(e) < emx]
  vecs = Vector{Tuple{String, Vector{T}}}()
  for (i, a) in enumerate(P), n in (2, 3)
    push!(vecs, ("f$i^$n", fill(a, n)))
  end
  for (i, a) in enumerate(P), (j, b) in enumerate(P)
    push!(vecs, ("m$i,$j", T[a, b]))
  end
  for (l1, l2) in ((1.0, 0.0), (1.0, 0.5), (0.25, 0.5), (0.0, 0.5)), (ie, e) in enumerate(exps), (vid, x) in vecs
    g = NormL1Fused(T(l1), T(l2))
    α = ldexp(one(T), Clong(e))
    α2 = ldexp(one(T), Clong(e ÷ 2))
    α3 = ldexp(one(T), Clong(e - e ÷ 2))
    id = "$label λ=($l1,$l2) e=$e x=$vid"
    z = zeros(T, length(x))
    rec("fam=scaledNF-fin $id scaled", () -> Scaled(g, α)(x))
    rec("fam=scaledNF-fin $id nested", () -> Scaled(Scaled(g, α2), α3)(x))
    rec("fam=scaledNF-fin $id scaled-translate", () -> Scaled(Translate(g, z), α)(x))
    rec("fam=scaledNF-fin $id translate-scaled", () -> Translate(Scaled(g, α), z)(x))
  end
  # Scaled(NormL1Fused) takes the value as parts (the NormL1Fused family)
  for (iα, α) in enumerate((T(0.5), T(1.5))), (ia, a) in enumerate(P), (ib, b) in enumerate(P)
    f = NormL1Fused(T(0.25), T(Inf))
    rec("fam=scaledNF $label α$iα x=$ia,$ia,$ib f", () -> (α * f)(T[a, a, b]))
    rec("fam=scaledNF $label α$iα y=$ia,$ia f", () -> (α * f)(T[a, a]))
  end
end


####################################
#  the broad table: a function of every source file, value / prox / gradient at magnitudes from tiny to floatmax
####################################
# the magnitudes: 1, 3/2, 2^h, 2^-h, floatmax, floatmax/2, a tiny number (h a quarter of the exponent range)
function magnitudes(T)
  M = floatmax(T)
  h = T === BigFloat ? Base.MPFR.get_emax() ÷ 4 : exponent(floatmax(T)) ÷ 2
  tiny = T === BigFloat ? ldexp(one(T), Clong(Base.MPFR.get_emin() + 8)) : nextfloat(zero(T)) * 16
  T[one(T), T(1.5), ldexp(one(T), Clong(h)), ldexp(one(T), Clong(-h)), M, M / 2, tiny]
end
# the vectors: every 7th ordered triple of magnitudes (times the stride of the family), signs from the indices; named by the indices
function triples(T, stride)
  P = magnitudes(T)
  out = Tuple{String, Vector{T}}[]
  c = 0
  for i in eachindex(P), j in eachindex(P), k in eachindex(P)
    c += 1
    (c - 1) % stride == 0 || continue
    s1 = iseven(i + j) ? 1 : -1
    s2 = iseven(j + k) ? -1 : 1
    push!(out, ("$(s1 * i),$(s2 * j),$k", T[s1 * P[i], s2 * P[j], P[k]]))
  end
  out
end
struct Fam
  name::String
  make::Any            # T -> function
  kind::Symbol         # :vec, :tuple (two blocks), :slice (3 x 2 array)
  stride::Int
end
fam(name, make; kind = :vec, stride = 1) = Fam(name, make, kind, stride)
input(::Val{:vec}, x) = x
input(::Val{:tuple}, x) = (x[1:2], x[3:3])
input(::Val{:slice}, x) = hcat(x, reverse(x))
input(::Val{:mat}, x) = [x[1] x[2]; x[2] x[3]]            # a symmetric 2 x 2 matrix
input(::Val{:graph}, x) = (x, x[1:2])
input(::Val{:sparse}, x) = SparseIterate(sparsevec(x))
input(::Val{:symsparse}, x) = SymmetricSparseIterate([x[1] x[2]; x[2] x[3]])
function groups3() Vector{Int}[[1, 2], [3]] end
function families()
  F = Fam[]
  # --- penalties and separable functions (functions/separable.jl, norms.jl, nonconvex.jl, losses.jl, entropy.jl, ordered.jl, sparsegroup.jl)
  push!(F, fam("NormL1", T -> NormL1(T(0.5))), fam("NormL1w", T -> NormL1(T(0.5), T[1, 2, 3])), fam("NormL1Nonneg", T -> NormL1Nonneg(T(0.5))),
         fam("ElasticNet", T -> ElasticNet(T(0.5), T(0.25))), fam("NormL2", T -> NormL2(T(0.5))), fam("CubeNormL2", T -> CubeNormL2(T(0.5))),
         fam("NormLinf", T -> NormLinf(T(0.5))), fam("GroupNormL2", T -> GroupNormL2(T(0.5), groups3())),
         fam("GroupNormLinf", T -> GroupNormLinf(T(0.5), groups3())), fam("SparseGroupNormL2", T -> SparseGroupNormL2(T(0.5), T(0.25), groups3())),
         fam("NormL1Fused-0.25-Inf", T -> NormL1Fused(T(0.25), T(Inf))), fam("NormL1Fused-1-0.5", T -> NormL1Fused(T(1), T(0.5))),
         fam("NormL1Fused-0-0.5", T -> NormL1Fused(T(0), T(0.5))), fam("NormL0", T -> NormL0(T(0.5))), fam("MCP", T -> MCP(T(0.5), T(3))),
         fam("SCAD", T -> SCAD(T(0.5), T(3.7))), fam("TotalVariation1D", T -> TotalVariation1D(T(0.5))), fam("FusedLasso", T -> FusedLasso(T(0.25), T(0.5))),
         fam("SLOPE", T -> SLOPE(T[3, 2, 1] ./ 4)), fam("NegEntropy", T -> NegEntropy(T(0.5))), fam("KLDivergence", T -> KLDivergence(T[1, 2, 3]; λ = T(0.5))),
         fam("NegLog", T -> NegLog(T(0.5))), fam("Berhu", T -> Berhu(T(0.5), T(2))), fam("CheckLoss", T -> CheckLoss(T(0.3))),
         fam("HingeLoss", T -> HingeLoss()), fam("SqrHingeLoss", T -> SqrHingeLoss()), fam("EpsInsensitiveLoss", T -> EpsInsensitiveLoss(T(0.5))),
         fam("CappedL1", T -> CappedL1(T(0.5), T(2))), fam("LogPenalty", T -> LogPenalty(T(0.5), T(0.5))), fam("NormLq", T -> NormLq(T(0.5), T(0.5))),
         fam("GroupMCP", T -> GroupMCP(T(0.5), groups3())), fam("GroupSCAD", T -> GroupSCAD(T(0.5), groups3())))
  # --- further families found by the audit of every exported type: SumLargest, BernoulliLoss, KronQuadratic, matrix sets,
  # IndGraph, the sparse iterates (value, prox, gradient where they exist; a constructor or method that rejects its argument
  # prints its error, which the comparison treats like any other result)
  push!(F, fam("SumLargest", T -> SumLargest(2, T(0.5))), fam("BernoulliLoss", T -> BernoulliLoss(T[0.5, 0.25, 0.75])),
         fam("KronQuadratic", T -> KronQuadratic(T[2 1; 1 2], T[1 0; 0 1]); kind = :mat),
         fam("IndBanded", T -> IndBanded(0, (2, 2)); kind = :mat), fam("IndToeplitz", T -> IndToeplitz(symmetric = true); kind = :mat),
         fam("IndPattern", T -> IndPattern(Bool[1 0; 0 1]); kind = :mat), fam("IndPSD", T -> IndPSD(); kind = :mat),
         fam("IndBallRank", T -> IndBallRank(1); kind = :mat), fam("IndStiefel", T -> IndStiefel(); kind = :mat),
         fam("IndGraph", T -> IndGraph(T[1 0 1; 0 1 1]); kind = :graph),
         fam("Sparse-NormL1", T -> NormL1(T(0.5)); kind = :sparse), fam("Sparse-NormL2", T -> NormL2(T(0.5)); kind = :sparse),
         fam("SymSparse-NormL1", T -> NormL1(T(0.5)); kind = :symsparse))
  # --- smooth functions (functions/smooth.jl, leastsquares.jl, glm.jl)
  push!(F, fam("SqrNormL2", T -> SqrNormL2(T(0.5))), fam("LinearFunction", T -> LinearFunction(T[1, -2, 3], T(1))),
         fam("QuadraticFunction", T -> QuadraticFunction(T[2 1 0; 1 2 1; 0 1 2], T[1, 2, 3], T(1))),
         fam("LeastSquaresLoss", T -> LeastSquaresLoss(T[1, 2, 3], T[1 0 1; 0 1 1; 1 1 0])),
         fam("LogisticLoss", T -> LogisticLoss(T[1, 0, 1], T[1 0 1; 0 1 1; 1 1 0])), fam("PoissonLoss", T -> PoissonLoss(T[1, 0, 2], T[1 0 1; 0 1 1; 1 1 0])),
         fam("HuberLoss", T -> HuberLoss(T[1, 2, 3], T[1 0 1; 0 1 1; 1 1 0], T(1))))
  # --- indicators (functions/sets.jl, cones.jl, polyhedron.jl, affine.jl, intersection.jl, ordered.jl)
  push!(F, fam("IndBox", T -> IndBox(T(-1), T(1))), fam("IndNonnegative", T -> IndNonnegative()), fam("IndBinary", T -> IndBinary(T(0), T(1))),
         fam("IndBallLinf", T -> IndBallLinf(T(1))), fam("IndBallL0", T -> IndBallL0(2)), fam("IndBallL1", T -> IndBallL1(T(1))),
         fam("IndBallL2", T -> IndBallL2(T(1))), fam("IndBallL2-big", T -> IndBallL2(ldexp(one(T), Clong(T === BigFloat ? Base.MPFR.get_emax() ÷ 4 : 60)))),
         fam("IndSimplex", T -> IndSimplex(T(1))), fam("IndSimplexLE", T -> IndSimplex(T(1); equality = false)), fam("IndSphereL2", T -> IndSphereL2(T(1))),
         fam("IndHyperplane", T -> IndHyperplane(T[1, 2, 3], T(1))), fam("IndHalfspace", T -> IndHalfspace(T[1, 2, 3], T(1))),
         fam("IndHyperslab", T -> IndHyperslab(T[1, 2, 3], T(-1), T(1))), fam("IndSOC", T -> IndSOC()), fam("IndRotatedSOC", T -> IndRotatedSOC()),
         fam("IndSOCg", T -> IndSOC(groups3())), fam("IndExpPrimal", T -> IndExpPrimal()), fam("IndExpDual", T -> IndExpDual()),
         fam("IndIsotonic", T -> IndIsotonic()), fam("IndBallGroupL0", T -> IndBallGroupL0(1, groups3())),
         fam("IndAffine", T -> IndAffine(T[1 2 3; 0 1 1], T[1, 2]); stride = 4), fam("IndPolyhedron", T -> IndPolyhedron(T[1 2 3; 0 1 1], T[-1, -1], T[1, 1]); stride = 8),
         fam("IndIntersection", T -> IndIntersection(IndBallL2(T(1)), IndBox(T(-1), T(1))); stride = 8))
  # --- wrappers (calculus.jl, composition.jl, distance.jl, metric.jl, envelope.jl, minimum.jl, constrained.jl, ballconstrained.jl, image.jl, second_order.jl)
  alphas(T) = (T(0.5), ldexp(one(T), Clong(T === BigFloat ? Base.MPFR.get_emax() ÷ 8 : 30)), ldexp(one(T), Clong(T === BigFloat ? -Base.MPFR.get_emax() ÷ 8 : -30)),
               ldexp(one(T), Clong(T === BigFloat ? -16_777_217 : -60)))
  for (ia, name) in enumerate(("half", "up", "down", "far"))
    push!(F, fam("Scaled-L2-$name", T -> Scaled(NormL2(T(0.5)), alphas(T)[ia])), fam("Scaled-NF-$name", T -> Scaled(NormL1Fused(T(1), T(0.5)), alphas(T)[ia])),
           fam("Scaled-NF0-$name", T -> Scaled(NormL1Fused(T(1), T(0)), alphas(T)[ia])), fam("Scaled-SOC-$name", T -> Scaled(IndSOC(), alphas(T)[ia])),
           fam("Scaled-L1-$name", T -> Scaled(NormL1(T(0.5)), alphas(T)[ia])), fam("Scaled-TV-$name", T -> Scaled(TotalVariation1D(T(0.5)), alphas(T)[ia])),
           fam("NestedScaled-NFinf-$name", T -> Scaled(Scaled(NormL1Fused(T(1), T(Inf)), alphas(T)[ia]), alphas(T)[ia])),
           fam("TranslateScaled-$name", T -> Translate(Scaled(NormL2(T(0.5)), alphas(T)[ia]), zeros(T, 3))))
  end
  push!(F, fam("Translate-L1", T -> Translate(NormL1(T(0.5)), T[1, 2, 3])), fam("Translate-L2", T -> Translate(NormL2(T(0.5)), zeros(T, 3))),
         fam("Tilt", T -> Tilt(NormL1(T(0.5)), T[1, 2, 3])), fam("Precompose", T -> Precompose(NormL1(T(0.5)), T(2))),
         # src/composition.jl:79-83: A*A' must equal mu*I (rows orthogonal, norm^2 mu); here mu = 9
         fam("Precompose-A", T -> Precompose(NormL1(T(0.5)), T[1 2 2; 2 1 -2], T(9)); stride = 4),
         fam("Regularize", T -> Regularize(NormL1(T(0.5)), T(0.25))), fam("MoreauEnvelope-L1", T -> MoreauEnvelope(NormL1(T(0.5)), T(1))),
         fam("MoreauEnvelope-L2", T -> MoreauEnvelope(NormL2(T(0.5)), T(1))), fam("Conjugate-L2", T -> Conjugate(NormL2(T(0.5)))),
         fam("Conjugate-L1", T -> Conjugate(NormL1(T(0.5)))), fam("DistL2", T -> DistL2(IndBox(T(-1), T(1)))), fam("SqrDistL2", T -> SqrDistL2(IndBox(T(-1), T(1)))),
         fam("PointwiseMinimum", T -> PointwiseMinimum(NormL1(T(0.5)), NormL2(T(0.5)))), fam("Constrained-L1-ball", T -> Constrained(NormL1(T(0.5)), IndBallL2(T(1)))),
         # src/ballconstrained.jl:90-92: the sphere takes NormL1, GroupNormL2 or IndBallL0 only
         fam("Constrained-L2-sphere", T -> Constrained(NormL1(T(0.5)), IndSphereL2(T(1)))), fam("Sum-L1-L2", T -> Sum(NormL1(T(0.5)), NormL2(T(0.5)))),
         # src/image.jl:110-115: L'*L must equal mu*I; src/image.jl:170-171 (_checkparams): the argument has length size(L, 1), so L is square 3x3
         # (the harness feeds length-3 vectors); mu = 9
         fam("LinearImage", T -> LinearImage(NormL2(T(0.5)), T[1 2 2; 2 1 -2; 2 -2 1], T(9))),
         fam("SeparableSum", T -> SeparableSum(NormL1(T(0.5)), NormL2(T(0.5))); kind = :tuple),
         fam("SliceSum", T -> SliceSum(NormL1(T(0.5)), NormL2(T(0.5))); kind = :slice))
  F
end
function run_broad(T, label)
  g = one(T)
  for F in families()
    f = try F.make(T) catch err; (rec("broad $(F.name) $label make", () -> throw(err)); continue) end
    for (id, x0) in triples(T, 7 * F.stride)
      tag = "broad $(F.name) $label x=$id"
      x = try input(Val(F.kind), x0) catch err; (rec("$tag input", () -> throw(err)); continue) end
      rec("$tag value", () -> f(x))
      prox_ok = try isproximable(f) catch; false end
      if prox_ok
        rec("$tag prox", () -> prox(f, x, g))
        rec("$tag vprox", () -> (y = similar(x); (value_and_prox!(y, f, x, g), y)))
      end
      smooth_ok = try issmooth(f) catch; false end
      smooth_ok && rec("$tag vgrad", () -> (gr = similar(x); (value_and_gradient!(gr, f, x), gr)))
      if F.kind === :vec
        # the scalar entry points of the separable functions, the coordinate curvature and the Hessian-vector product of smooth ones
        (try isseparable(f) catch; false end) && rec("$tag entry", () -> (value_entry(f, x[1], 1), prox_entry(f, x[1], 1, g)))
        smooth_ok && rec("$tag coordlip", () -> coordinate_lipschitz(f, 1))
        (smooth_ok && (try hashessian(f) catch; false end)) && rec("$tag hvec", () -> hessian_vec(f, x, x))
      end
    end
  end
end


####################################
#  spectral families: SingularValue and Spectral of functions with BigFloat levels, scales and weights, on tiny exact matrices
####################################
# The matrices are Float32/Float64 and tiny (diagonal, exactly representable) so that LAPACK returns the same spectrum on every
# platform; the permuted diagonal and the constant matrix (σ = (2 floatmax, 0), an overflowing spectrum) are named ":lapack"
# and are informational. The levels, weights and scales are BigFloat at precision p, from the normal range of T to the
# exponent range limits of BigFloat.
function spec_mats(T)
  M = floatmax(T); h = M / 2; fm = floatmin(T)
  Any[("d1", T[M 0; 0 h], false), ("d2", T[M 0; 0 M], false), ("d3", T[M 0; 0 fm], false), ("d4", T[3 0; 0 1], false),
      ("d5", T[-M 0; 0 h], false), ("p6", T[0 M; h 0], true), ("f7", fill(M, 2, 2), true)]
end
function spec_symm(T)
  M = floatmax(T); h = M / 2
  Any[("s1", T[M 0; 0 -h], false), ("s2", T[M 0; 0 M], false), ("s3", T[3 0; 0 1], false), ("s4", T[M 0; 0 0], false),
      ("s5", fill(M, 2, 2), true)]
end
function spec_levels(T)
  ex = exponent(floatmax(T))
  W = Base.MPFR.get_emax() - 8
  Any[("a", big(0.5)), ("b", ldexp(big(1.5), Clong(-ex))), ("c", ldexp(big(1.0), Clong(-W))), ("d", ldexp(big(1.0), Clong(W))),
      ("e", ldexp(big(1.5), Clong(-ex - 20)))]
end
# the scales of the nested wrappers: (2^h, 2^h) overflows as a product, (2^h, 2^-h) is one; h = W/2 + 5
function spec_scales()
  W = Base.MPFR.get_emax() - 8
  h = W ÷ 2 + 5
  Any[("n1", ldexp(big(1.0), Clong(h)), ldexp(big(1.0), Clong(h))), ("n2", ldexp(big(1.0), Clong(h)), ldexp(big(1.0), Clong(-h)))]
end
# (name, constructor from a level, entries)  entries: :value, :prox (value, prox, value_and_prox!)
function spec_fams(T)
  F = Any[]
  push!(F, ("SV-NormL1", λ -> SingularValue(NormL1(λ)), :prox), ("SV-NormL2", λ -> SingularValue(NormL2(λ)), :prox),
         ("SV-NormLinf", λ -> SingularValue(NormLinf(λ)), :prox), ("SV-SqrNormL2", λ -> SingularValue(SqrNormL2(λ)), :prox),
         ("SV-ElasticNet", λ -> SingularValue(ElasticNet(λ, λ)), :prox), ("SV-SumLargest", λ -> SingularValue(SumLargest(1, λ)), :prox),
         ("SV-SLOPE", λ -> SingularValue(SLOPE([λ, λ / 2])), :prox), ("SV-NormL1w", λ -> SingularValue(NormL1(λ, [big(1.0), big(2.0)])), :prox),
         ("SV-NormL1Nonneg", λ -> SingularValue(NormL1Nonneg(λ)), :prox), ("SV-CubeNormL2", λ -> SingularValue(CubeNormL2(λ)), :value),
         ("SV-Conj-IndBallL1", λ -> SingularValue(Conjugate(IndBallL1(λ))), :value), ("SV-Conj-IndBallL2", λ -> SingularValue(Conjugate(IndBallL2(λ))), :value),
         ("SV-Conj-IndSimplex", λ -> SingularValue(Conjugate(IndSimplex(λ))), :value), ("SV-Conj-SqrNormL2", λ -> SingularValue(Conjugate(SqrNormL2(λ))), :value),
         ("SV-Conj-Scaled-NormL1", λ -> SingularValue(Conjugate(Scaled(NormL1(λ), big(4.0)))), :value),
         ("SV-Constrained-Nonneg", λ -> SingularValue(Constrained(NormL1(λ), IndBox(0, Inf))), :value),
         ("NuclearNorm", λ -> NuclearNorm(λ), :prox), ("TracePSD", λ -> TracePSD(λ), :prox))
  F
end
function spec_scaled_fams()
  Any[("SV-Scaled-NormL1", (λ, a, b) -> SingularValue(Scaled(NormL1(λ), a))),
      ("SV-Nested-NormL1", (λ, a, b) -> SingularValue(Scaled(Scaled(NormL1(λ), a), b))),
      ("SV-Nested-NormL2", (λ, a, b) -> SingularValue(Scaled(Scaled(NormL2(λ), a), b))),
      ("Spectral-Nested-NormL1", (λ, a, b) -> Spectral(Scaled(Scaled(NormL1(λ), a), b)))]
end
function spec_symm_fams()
  Any[("Sp-NormL1", λ -> Spectral(NormL1(λ)), :prox), ("Sp-NormL2", λ -> Spectral(NormL2(λ)), :prox),
      ("Sp-SumLargest", λ -> Spectral(SumLargest(1, λ)), :prox), ("Sp-Nonneg", λ -> Spectral(Constrained(NormL1(λ), IndBox(0, Inf))), :value),
      ("Sp-NormL1Nonneg", λ -> Spectral(NormL1Nonneg(λ)), :prox), ("Sp-SqrNormL2", λ -> Spectral(SqrNormL2(λ)), :prox),
      ("Sp-Conj-IndBallL1", λ -> Spectral(Conjugate(IndBallL1(λ))), :value)]
end
function spec_record(fam, T, p, mid, lid, lapack, f, Z, entries)
  fn = lapack ? "spectral:lapack:$fam" : "spectral:$fam"
  tag = "$fn BigFloat$p mat=$T M=$mid lvl=$lid"
  rec("$tag value", () -> f(Z))
  if entries === :prox
    g = one(T)
    rec("$tag prox", () -> prox(f, Z, g))
    rec("$tag vprox", () -> (Y = similar(Z); (value_and_prox!(Y, f, Z, g), Y)))
  end
end
function run_spectral(T)
  p = precision(BigFloat)
  lv = spec_levels(T)
  for (name, make, entries) in spec_fams(T), (mid, Z, lap) in spec_mats(T), (lid, λ) in lv
    f = try make(λ) catch err; rec("spectral:$name BigFloat$p mat=$T M=$mid lvl=$lid make", () -> throw(err)); continue end
    spec_record(name, T, p, mid, lid, lap, f, Z, entries)
  end
  for (name, make, entries) in spec_symm_fams(), (mid, Z, lap) in spec_symm(T), (lid, λ) in lv
    f = try make(λ) catch err; rec("spectral:$name BigFloat$p mat=$T M=$mid lvl=$lid make", () -> throw(err)); continue end
    spec_record(name, T, p, mid, lid, lap, f, Z, entries)
  end
  for (name, make) in spec_scaled_fams(), (mid, Z, lap) in (name == "Spectral-Nested-NormL1" ? spec_symm(T) : spec_mats(T)), (lid, λ) in lv, (sid, a, b) in spec_scaled_fams_scales(name)
    f = try make(λ, a, b) catch err; rec("spectral:$name BigFloat$p mat=$T M=$mid lvl=$lid$sid make", () -> throw(err)); continue end
    fn = lap ? "spectral:lapack:$name" : "spectral:$name"
    rec("$fn BigFloat$p mat=$T M=$mid lvl=$lid$sid value", () -> f(Z))
  end
end
# the scale pairs of a scaled family: the plain Scaled takes one factor (the first of each pair)
spec_scaled_fams_scales(name) = name == "SV-Scaled-NormL1" ? [("/" * s[1], s[2], s[3]) for s in spec_scales()] : [("/" * s[1], s[2], s[3]) for s in spec_scales()]


####################################
#  LogDetLoss at a BigFloat X: S and w are stored in Float32/Float64 (the constructor rejects BigFloat), X is copied into the
#  Float workspace for the Cholesky factor, but the dot product, the value and its scaled recovery keep the BigFloat X
####################################
function run_logdet()
  p = precision(BigFloat)
  emin = Base.MPFR.get_emin()
  sh(k) = ldexp(one(BigFloat), Clong(k))
  ts = Any[("fm", floatmin(BigFloat)), ("m0", sh(emin - 1)), ("m8", sh(emin + 8)), ("m400", sh(emin + 400)), ("n35", sh(-35)), ("h", big(0.5))]
  Ss = Any[("s1", [0.0 0.5; 0.5 0.0]), ("s2", [1.0 0.0; 0.0 1.0]), ("s3", [0.0 0.25; 0.25 0.0])]
  ws = Any[("w0", 1.0), ("w30", 2.0^30), ("w100", 2.0^100), ("w1000", 2.0^1000)]
  for (sid, S) in Ss, (tid, t) in ts, (wid, w) in ws
    tag = "logdet:LogDetLoss BigFloat$p S=$sid t=$tid e=$(exponent(t)) w=$wid"
    f = try LogDetLoss(S, w) catch err; (rec("$tag make", () -> throw(err)); continue) end
    X = BigFloat[1 t; t 1]
    rec("$tag value", () -> f(X))
    rec("$tag vgrad", () -> (G = similar(X); (value_and_gradient!(G, f, X), G)))
    rec("$tag prox", () -> prox(f, X, big(1)))
    rec("$tag vprox", () -> (Y = similar(X); (value_and_prox!(Y, f, X, big(1)), Y)))
    rec("$tag hvec", () -> hessian_vec(f, X, X))
  end
end

guard("run_type Float32", () -> run_type(Float32, "Float32"))
guard("run_type Float64", () -> run_type(Float64, "Float64"))
for p in (2, 53, 256)
  guard("run_type BigFloat$p", () -> setprecision(() -> run_type(BigFloat, "BigFloat$p"), BigFloat, p))
end
guard("run_families Float32", () -> run_families(Float32, "Float32"))
guard("run_families Float64", () -> run_families(Float64, "Float64"))
for p in (2, 5, 53, 256)
  guard("run_families BigFloat$p", () -> setprecision(() -> run_families(BigFloat, "BigFloat$p"), BigFloat, p))
end
# LogDetLoss at a BigFloat X
for p in (2, 3, 5, 53, 256)
  guard("run_logdet BigFloat$p", () -> setprecision(() -> run_logdet(), BigFloat, p))
end
# the spectral families
for T in (Float32, Float64), p in (2, 3, 5, 53, 256)
  guard("run_spectral $T BigFloat$p", () -> setprecision(() -> run_spectral(T), BigFloat, p))
end
# the broad table
guard("run_broad Float32", () -> run_broad(Float32, "Float32"))
guard("run_broad Float64", () -> run_broad(Float64, "Float64"))
for p in (2, 3, 5, 16, 53, 256)
  guard("run_broad BigFloat$p", () -> setprecision(() -> run_broad(BigFloat, "BigFloat$p"), BigFloat, p))
end
close(out)
println("wrote ", ARGS[1])
