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

run_type(Float32, "Float32")
run_type(Float64, "Float64")
for p in (2, 53, 256)
  setprecision(BigFloat, p) do
    run_type(BigFloat, "BigFloat$p")
  end
end
close(out)
println("wrote ", ARGS[1])
