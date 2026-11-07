# Issue #747: LLVM IR, return types, allocations and timings of the scaled-product helpers at Float32 and Float64.
# Lines that start with TIMING are measurements; everything else is expected to be identical at base and head.
using ProximalBase, InteractiveUtils
const PB = ProximalBase

println("== platform")
@show VERSION Sys.MACHINE Sys.CPU_NAME

# the targets: (label, function, argument types)
targets(T) = (
  ("_scaled_product(::$T, ::$T, ::Int)", PB._scaled_product, Tuple{T, T, Int}),
  ("_scaled_product(::$T, ::$T, ::$T, ::Int)", PB._scaled_product, Tuple{T, T, T, Int}),
  ("_power_factors(::Type{$T}, ::Int)", PB._power_factors, Tuple{Type{T}, Int}),
)

println("== return types")
for T in (Float32, Float64), (lab, f, at) in targets(T)
  rts = Base.return_types(f, at)
  println(lab, " -> ", rts, "  concrete = ", all(isconcretetype, rts))
end

println("== LLVM IR")
for T in (Float32, Float64), (lab, f, at) in targets(T)
  println("---- ", lab)
  println(sprint(io -> code_llvm(io, f, at; debuginfo = :none)))
end

# allocations of a warmed call, in function barriers
sp2(w, v, k) = @allocated PB._scaled_product(w, v, k)
sp3(a, b, c, k) = @allocated PB._scaled_product(a, b, c, k)
pf(::Type{T}, k) where {T} = @allocated PB._power_factors(T, k)
println("== allocations (bytes, after one warm call)")
for T in (Float32, Float64)
  w, v = T(0.75), T(3e10)
  sp2(w, v, 3); sp3(w, v, T(2), 3); pf(T, 5)
  println("ALLOC _scaled_product(::$T, ::$T, ::Int) = ", sp2(w, v, 3))
  println("ALLOC _scaled_product(::$T, ::$T, ::$T, ::Int) = ", sp3(w, v, T(2), 3))
  println("ALLOC _power_factors(::Type{$T}, ::Int) = ", pf(T, 5))
end

# timings: 10^7 calls per repetition, the minimum of 5 repetitions
function loop2(w::T, v::T, n) where {T}
  s = zero(T)
  for i in 1:n
    s += PB._scaled_product(w, v, (i & 63) - 32)
  end
  s
end
function loop3(w::T, v::T, n) where {T}
  s = zero(T)
  c = T(1.5)
  for i in 1:n
    s += PB._scaled_product(w, v, c, (i & 63) - 32)
  end
  s
end
function loopf(::Type{T}, n) where {T}
  s = zero(T)
  for i in 1:n
    a, b = PB._power_factors(T, (i % 4001) - 2000)
    s += a + b
  end
  s
end
# the result of every repetition is accumulated into a sink that is printed, so that no call can be dropped
const SINK = Ref(0.0)
mintime(g) = minimum(begin t0 = time_ns(); SINK[] += Float64(g()); (time_ns() - t0) / 1e7 end for _ in 1:5)
println("== timings (ns per call)")
for T in (Float32, Float64)
  w, v = T(0.75), T(3.5)
  loop2(w, v, 10); loop3(w, v, 10); loopf(T, 10)
  println("TIMING _scaled_product(::$T, ::$T, ::Int) = ", mintime(() -> loop2(w, v, 10^7)), " ns")
  println("TIMING _scaled_product(::$T, ::$T, ::$T, ::Int) = ", mintime(() -> loop3(w, v, 10^7)), " ns")
  println("TIMING _power_factors(::Type{$T}, ::Int) = ", mintime(() -> loopf(T, 10^7)), " ns")
end
println("sink = ", SINK[])
println("== done")
