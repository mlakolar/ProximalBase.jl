# PB #911: does the disposable PB refactor (pkgs/pb-H: LS f(β), value_and_gradient!, gradient! and the
# new oracle sharing `_square_value_level`) leave PB's existing least-squares methods unchanged?
# Original PB master (the loaded package) and the refactored copy (loaded as a separate module from
# source) run in one process on the same inputs: bitwise comparison of returned values and gradients,
# then paired, alternating timing of the existing methods. Disposable.
include(joinpath(@__DIR__, "envsetup.jl"))
using Random, Printf
print_provenance()
const PBO = ProximalBase
const W = Module(:PBRefactor)
Base.include(W, joinpath(RUNDIR, "pkgs", "pb-H", "src", "ProximalBase.jl"))
const PBR = Base.invokelatest(getfield, W, :ProximalBase)
println("refactored copy loaded from ", joinpath(RUNDIR, "pkgs", "pb-H", "src"))
bits(v::AbstractFloat) = reinterpret(Unsigned, v)
bits(v::AbstractArray) = map(bits, v)
same(a, b) = isequal(a, b) && bits(a) == bits(b)
function inputs(::Type{T}) where {T}
  rng = Xoshiro(911)
  L = Any[]
  for (n, p, k) in ((64, 6, 1), (240, 160, 1), (30, 5, 3))
    X = randn(rng, T, n, p); Y = k == 1 ? randn(rng, T, n) : randn(rng, T, n, k)
    β = k == 1 ? randn(rng, T, p) : randn(rng, T, p, k)
    push!(L, ("ordinary n$n p$p k$k", X, Y, β))
    push!(L, ("exactfit n$n k$k", X, X * β, β))
    push!(L, ("zero beta n$n k$k", X, Y, zero(β)))
    t = T === Float64 ? 1e-160 : 1f-22
    push!(L, ("underflow n$n k$k", X .* T(t), Y .* T(t), β))
    t = T === Float64 ? 1e154 : 1f19
    push!(L, ("overflow squares n$n k$k", X, Y .* T(t), β .* T(t)))
    push!(L, ("nan beta n$n k$k", X, Y, (b = copy(β); b[1] = T(NaN); b)))
    push!(L, ("inf Y n$n k$k", X, (y = copy(Y); y[2] = T(Inf); y), β))
  end
  # image overflow with cancellation (PS #277 R1) and a difference that overflows
  push!(L, ("image overflow R1", fill(floatmax(T) / 2, 1, 2), T[1], T[2, -2]))
  push!(L, ("difference overflow", fill(floatmax(T) * T(0.75), 1, 1), T[-floatmax(T) * T(0.75)], T[1]))
  L
end
const SCALES = (true, 0.25, 1e-300, 1e200)
ok = Dict{String, Int}(); tot = Dict{String, Int}()
tally!(k, b) = (tot[k] = get(tot, k, 0) + 1; ok[k] = get(ok, k, 0) + b)
for T in (Float64, Float32), (lab, X, Y, β) in inputs(T), α in SCALES
  (α isa Float64 && T === Float32 && abs(α) > 1e30) && continue
  fo = PBO.LeastSquaresLoss(copy(Y), copy(X)); fr = PBR.LeastSquaresLoss(copy(Y), copy(X))
  if α !== true
    fo = PBO.Scaled(fo, α); fr = PBR.Scaled(fr, α)
  end
  vo = fo(β); vr = fr(β)
  go = similar(β); gr = similar(β)
  wo = PBO.value_and_gradient!(go, fo, β); wr = PBR.value_and_gradient!(gr, fr, β)
  ho = similar(β); hr = similar(β)
  PBO.gradient!(ho, fo, β); PBR.gradient!(hr, fr, β)
  s1, s2, s3, s4 = same(vo, vr), same(wo, wr), same(go, gr), same(ho, hr)
  tally!("value", s1); tally!("value_and_gradient value", s2); tally!("value_and_gradient gradient", s3); tally!("gradient", s4)
  (s1 & s2 & s3 & s4) || @printf "DIFF %-7s %-24s α=%-6s value %s/%s vg %s/%s grad_eq=%s gradient_eq=%s\n" T lab string(α) repr(vo) repr(vr) repr(wo) repr(wr) s3 s4
end
for k in sort(collect(keys(tot)))
  @printf "bitwise identical, %-28s %d / %d\n" k ok[k] tot[k]
end
# the new oracle against the refactored f(β) at the same image bits (same-type scales)
let n = 0, m = 0
  for T in (Float64, Float32), (lab, X, Y, β) in inputs(T)
    ndims(Y) == 1 || continue
    for α in (true, T(0.25))
      f = PBR.LeastSquaresLoss(copy(Y), copy(X)); fs = α === true ? f : PBR.Scaled(f, α)
      z = X * β
      all(isfinite, z) || continue
      a = fs(β); b = PBR.value_at_image(fs, z)
      g1 = similar(β); g2 = similar(β); d = similar(Y)
      c = PBR.value_and_gradient!(g1, fs, β); e = PBR.value_and_gradient_at_image!(g2, d, fs, z)
      n += 1; m += same(a, b) & same(c, e) & same(g1, g2)
      same(a, b) & same(c, e) & same(g1, g2) || @printf "oracle vs f(β) differ: %-7s %-24s α=%s  %s/%s  %s/%s\n" T lab string(α) repr(a) repr(b) repr(c) repr(e)
    end
  end
  @printf "value_at_image / value_and_gradient_at_image! equal to f(β) / value_and_gradient! (finite images): %d / %d\n" m n
end
# paired timing of the existing methods, original vs refactored
med(v) = (s = sort(v); k = length(s); isodd(k) ? s[(k + 1) ÷ 2] : (s[k ÷ 2] + s[k ÷ 2 + 1]) / 2)
for T in (Float64, Float32), (n, p) in ((2000, 500), (240, 160), (100, 32)), α in (true, T(1 / n))
  rng = Xoshiro(5); X = randn(rng, T, n, p); Y = randn(rng, T, n); β = randn(rng, T, p) ./ T(10)
  fo = PBO.LeastSquaresLoss(Y, X); fr = PBR.LeastSquaresLoss(copy(Y), copy(X))
  α === true || (fo = PBO.Scaled(fo, α); fr = PBR.Scaled(fr, α))
  g = similar(β)
  ops = (("f(β)", () -> fo(β), () -> fr(β)), ("value_and_gradient!", () -> PBO.value_and_gradient!(g, fo, β), () -> PBR.value_and_gradient!(g, fr, β)),
         ("gradient!", () -> PBO.gradient!(g, fo, β), () -> PBR.gradient!(g, fr, β)))
  for (name, a, b) in ops
    a(); b(); ab = @allocated a(); bb = @allocated b()
    t0 = time_ns(); a(); t1 = time_ns()
    inner = clamp(ceil(Int, 2e6 / max(t1 - t0, 1)), 1, 100000)
    ra = Float64[]; rb = Float64[]
    for r in 1:41
      for which in (isodd(r) ? (1, 2) : (2, 1))
        fn = which == 1 ? a : b
        t = time_ns(); for _ in 1:inner; fn(); end
        push!(which == 1 ? ra : rb, (time_ns() - t) / inner)
      end
    end
    @printf "timing %-7s n=%-5d p=%-4d %-6s %-20s original %9.1f ns  refactored %9.1f ns  paired-median ratio %.4f  bytes %d/%d\n" T n p (α === true ? "raw" : "scaled") name med(ra) med(rb) med(rb ./ ra) ab bb
  end
end
