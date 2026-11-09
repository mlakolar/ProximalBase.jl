module BlockStepGridsTest

# The Float32 combinations of "block steps / references" that block_step.jl leaves to its Float64 pass: of the 83
# (function, iterate) combinations of that testset at Float32, all but seven (the seven are listed in
# FLOAT32_REFERENCES of block_step.jl, which runs them). block_step.jl keeps the Float64 pass, the seven Float32
# combinations and the other testsets of "block steps". `consistent`, `nested_scaled` and `check_block` are copied
# from it, and the loops have the same functions and iterates in the same order; here they run the Float32
# combinations that are not in FLOAT32_REFERENCES. The fixtures are the same as in block_step.jl. No case here
# draws random numbers.

using Test
using ProximalBase

function consistent(x::AtomIterate)
  act = collect(activecoordinates(x))
  length(unique(act)) == nactive(x) &&
    count(!iszero, x.position) == nactive(x) &&
    all(a -> x.position[act[a]] == a, eachindex(act)) &&
    all(k -> isactive(x, k) || all(iszero, x.storage[x.groups[k]]), eachindex(x.groups))
end
consistent(x) = all(j -> x.ind2nzval[x.nzval2ind[j]] == j, 1:x.nnz)
nested_scaled(f, α) = (s = Scaled(f, α); Scaled{typeof(s), typeof(α)}(s, α))

# the Float32 combinations that block_step.jl runs, in the keys of its loops
const FLOAT32_REFERENCES = ((:sparse, 5, 1), (:sparse, 6, 2), (:symmetric, 7, 1), (:symmetric, 3, 2),
                            (:group, 1, 4), (:group, 2, 7), (:wrapped, 3, 2, 4))
moved(key) = !(key in FLOAT32_REFERENCES)

function check_block(x::AtomIterate, f, k, u, γ)
  x[x.groups[2][1]] = 2
  before, trial = Vector(x), copy(u)
  G = x.groups[k]
  dense = copy(before)
  dense[G] .= eltype(x).(u)
  expected = prox(f, dense, γ)[G]
  written = copy(x)
  atom(written, k) .= u
  prox_coordinate!(written, f, k, γ)
  result = prox_coordinate!(x, f, k, u, γ)
  @test result ≈ expected
  @test result == atom(written, k)
  @test parent(result) === x
  @test Vector(x)[setdiff(1:length(x), G)] == before[setdiff(1:length(x), G)]
  @test u == trial
  @test consistent(x) && consistent(written)
  result[1] = one(eltype(x))
  @test isactive(x, k) && consistent(x)
end

function check_block(x, f, k, u, γ)
  x isa SymmetricSparseIterate ? (x[2, 2] = 2) : (x[4] = 2)
  before = Array(x)
  written = copy(x)
  if x isa SymmetricSparseIterate
    r, c = tril_cartesian(x.p, k)
    i = (c - 1) * x.p + r
    coordinates(written)[k] = u
  else
    i = k
    written[k] = u
  end
  expected = prox_entry(f, convert(eltype(x), u), i, γ)
  reference = prox_coordinate!(written, f, k, γ)
  result = prox_coordinate!(x, f, k, u, γ)
  @test result == expected == reference
  dense = copy(before)
  dense[i] = convert(eltype(x), expected)
  if x isa SymmetricSparseIterate
    dense[c, r] = convert(eltype(x), expected)
  end
  @test Array(x) == dense
  @test consistent(x) && consistent(written)
end

@testset "block steps" begin
  @testset "references" begin
    for T in (Float32,)
      fs = (NormL1(T(0.3)), ElasticNet(T(0.3), T(0.2)), IndBox(T(1), T(2)),
            Translate(NormL1(T(0.3)), T[1, 2, 2, 4]))
      scalarfs = (fs..., Scaled(fs[1], T(2)), nested_scaled(fs[1], T(2)),
                  Regularize(fs[1], T(0.2)), Regularize(fs[1], T(0.2), T[1, 2, 2, 4]))
      for (i, f) in enumerate(scalarfs)
        for (j, x) in enumerate((SparseIterate(zeros(T, 4)), SparseIterate(zeros(T, 2, 2))))
          moved((:sparse, i, j)) && check_block(x, f, 2, T(3), T(0.7))
        end
        for k in (1, 2)
          moved((:symmetric, i, k)) && check_block(SymmetricSparseIterate(zeros(T, 2, 2)), f, k, T(3), T(0.7))
        end
      end
      for (g, gs) in enumerate(([1:2, 3:4], [[1, 3], [2, 4]], [[4, 2], [3, 1]]))
        groupfs = (GroupNormL2(T(0.3), gs), GroupNormLinf(T(0.3), gs),
                   SparseGroupNormL2(T(0.1), T(0.3), gs), GroupMCP(T(0.3), gs), GroupSCAD(T(0.3), gs))
        for (m, f) in enumerate((fs..., groupfs...))
          moved((:group, g, m)) && check_block(AtomIterate(zeros(T, 4), gs), f, 1, T[3, -4], T(0.7))
        end
        for (n, f) in enumerate((fs[1], groupfs[1]))
          for (q, h) in enumerate((Scaled(f, T(2)), nested_scaled(f, T(2)),
                                   Regularize(f, T(0.2)), Regularize(f, T(0.2), T[1, 2, 3, 4])))
            moved((:wrapped, g, n, q)) && check_block(AtomIterate(zeros(T, 4), gs), h, 1, T[3, -4], T(0.7))
          end
        end
      end
    end
  end
end

end
