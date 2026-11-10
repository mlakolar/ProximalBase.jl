module BlockStepGridsTest
using Test, ProximalBase

@testset "block oracle numerical endpoints" begin
  for T in (Float32, Float64), level in (zero(T), T(.3), T(Inf)),
      weight in (zero(T), one(T), T(Inf)), step in (zero(T), T(.7), T(Inf))
    gs = [[3,1], [2,4]]
    fs = (GroupNormL2(level, gs, T[weight,1]), GroupNormLinf(level, gs, T[weight,1]),
          SparseGroupNormL2(T(.2), level, gs, T[weight,1]),
          GroupMCP(level, gs, T[weight,1]), GroupSCAD(level, gs, T[weight,1]))
    for f in fs, u in (T[3,4], T[0,0], T[NaN,2], T[Inf,0], T[floatmax(T)/2,floatmax(T)/2])
      x = zeros(T, 4); x[gs[1]] .= u
      reference = prox(f, x, step)[gs[1]]
      out = fill(T(-19), 2)
      prox_block!(out, f, u, 1, step)
      @test all(eachindex(out)) do j
        isequal(out[j], reference[j]) || isapprox(out[j], reference[j]; rtol=8eps(T), atol=eps(T))
      end
      alias = copy(u)
      prox_block!(alias, f, alias, 1, step)
      @test isequal(alias, out)
    end
  end
end
end
