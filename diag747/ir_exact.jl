# Issue #747: the exact siblings _scaled_product_exact and _power_factors_exact against the originals, Float32 and Float64:
# the LLVM IR (function names and block labels normalized), return types, allocations and timings.
# At the base the siblings do not exist and the script says so.
using ProximalBase, InteractiveUtils
const PB = ProximalBase

norm(txt) = replace(replace(replace(txt, "_exact" => ""), r"julia_[A-Za-z0-9_]*?_[0-9]+" => "julia_F"), r"L[0-9]+" => "L")
siblings(T) = (
  ("_scaled_product(::$T, ::$T, ::Int)", () -> PB._scaled_product, () -> PB._scaled_product_exact, Tuple{T, T, Int}),
  ("_power_factors(::Type{$T}, ::Int)", () -> PB._power_factors, () -> PB._power_factors_exact, Tuple{Type{T}, Int}),
)
sp2(f, w, v, k) = @allocated f(w, v, k)
pf(f, ::Type{T}, k) where {T} = @allocated f(T, k)
function loop2(sp, w::T, v::T, n) where {T}
  s = zero(T)
  for i in 1:n
    s += sp(w, v, (i & 63) - 32)
  end
  s
end
function loopf(pfun, ::Type{T}, n) where {T}
  s = zero(T)
  for i in 1:n
    a, b = pfun(T, (i % 4001) - 2000)
    s += a + b
  end
  s
end
const SINK = Ref(0.0)
mintime(g) = minimum(begin t0 = time_ns(); SINK[] += Float64(g()); (time_ns() - t0) / 1e7 end for _ in 1:5)

function report()
  have = isdefined(PB, :_scaled_product_exact) && isdefined(PB, :_power_factors_exact)
  println("exact siblings defined = ", have)
  have || return
  for T in (Float32, Float64), (lab, orig, exact, at) in siblings(T)
    orig, exact = orig(), exact()
    println("== ", lab)
    println("return types: original ", Base.return_types(orig, at), ", exact ", Base.return_types(exact, at))
    io1 = sprint(io -> code_llvm(io, orig, at; debuginfo = :none))
    io2 = sprint(io -> code_llvm(io, exact, at; debuginfo = :none))
    println("IR exact == original (names and labels normalized): ", norm(io1) == norm(io2),
            "  (lines ", count(==('\n'), io1), " and ", count(==('\n'), io2), ")")
    if norm(io1) != norm(io2)
      println("---- original\n", io1, "---- exact\n", io2)
    end
  end
  println("== allocations (bytes, after one warm call)")
  for T in (Float32, Float64)
    w, v = T(0.75), T(3e10)
    sp2(PB._scaled_product, w, v, 3); sp2(PB._scaled_product_exact, w, v, 3); pf(PB._power_factors, T, 5); pf(PB._power_factors_exact, T, 5)
    println("ALLOC exact _scaled_product_exact(::$T, ::$T, ::Int) = ", sp2(PB._scaled_product_exact, w, v, 3),
            "   original = ", sp2(PB._scaled_product, w, v, 3))
    println("ALLOC exact _power_factors_exact(::Type{$T}, ::Int) = ", pf(PB._power_factors_exact, T, 5),
            "   original = ", pf(PB._power_factors, T, 5))
  end
  println("== timings (ns per call; the original and the exact sibling interleaved)")
  for T in (Float32, Float64)
    w, v = T(0.75), T(3.5)
    loop2(PB._scaled_product, w, v, 10); loop2(PB._scaled_product_exact, w, v, 10)
    loopf(PB._power_factors, T, 10); loopf(PB._power_factors_exact, T, 10)
    for rep in 1:2
      println("TIMING _scaled_product(::$T) original = ", mintime(() -> loop2(PB._scaled_product, w, v, 10^7)),
              " exact = ", mintime(() -> loop2(PB._scaled_product_exact, w, v, 10^7)), " ns")
      println("TIMING _power_factors(::$T) original = ", mintime(() -> loopf(PB._power_factors, T, 10^7)),
              " exact = ", mintime(() -> loopf(PB._power_factors_exact, T, 10^7)), " ns")
    end
  end
  println("sink = ", SINK[])
end
report()
println("== done")
