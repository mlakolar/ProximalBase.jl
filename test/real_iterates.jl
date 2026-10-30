module RealIteratesTest

using Test, ProximalBase, LinearAlgebra, SparseArrays

function rejects(f, name, outputs...)
  before = map(copy, outputs)
  @test_throws ArgumentError("$name requires real entries") f()
  for (out, saved) in zip(outputs, before)
    @test isequal(out, saved)
    if out isa Union{SparseIterate, SymmetricSparseIterate}
      @test out.nnz == saved.nnz
      @test out.ind2nzval == saved.ind2nzval
      @test out.nzval2ind == saved.nzval2ind
    elseif out isa AtomIterate
      @test out.nactive == saved.nactive
      @test out.active == saved.active
      @test out.position == saved.position
      @test out.layout_cache === saved.layout_cache
    end
  end
end

@testset "real iterate constructors" begin
  for T in (ComplexF32, ComplexF64)
    for make in (() -> SparseIterate(T, 2), () -> SparseIterate(T, 2, 2),
                 () -> SparseIterate(zeros(T, 2)), () -> SparseIterate(zeros(T, 2, 2)),
                 () -> SparseIterate(spzeros(T, 2)), () -> SparseIterate(spzeros(T, 2, 2)),
                 () -> SparseIterate{T, 1}(zeros(T, 2), zeros(Int, 2), zeros(Int, 2), 0),
                 () -> SparseIterate(zeros(T, 2), zeros(Int, 2), zeros(Int, 2), 0),
                 () -> similar(SparseIterate(2), T),
                 () -> convert(SparseIterate{T, 1}, SparseIterate(2)),
                 () -> convert(SparseIterate{T}, SparseIterate(2)))
      rejects(make, :SparseIterate)
    end
    for make in (() -> SymmetricSparseIterate(T, 2),
                 () -> SymmetricSparseIterate(zeros(T, 2, 2)),
                 () -> SymmetricSparseIterate(spzeros(T, 2, 2)),
                 () -> SymmetricSparseIterate{T}(zeros(T, 3), zeros(Int, 3), zeros(Int, 3), 0, 2),
                 () -> similar(SymmetricSparseIterate(2), T),
                 () -> convert(SymmetricSparseIterate{T}, SymmetricSparseIterate(2)))
      rejects(make, :SymmetricSparseIterate)
    end
    G = [1:1, 2:2]
    for make in (() -> AtomIterate(T, 2, G), () -> AtomIterate(T, 2, 2),
                 () -> AtomIterate(zeros(T, 2), G), () -> AtomIterate(zeros(T, 2), 2),
                 () -> AtomIterate{T, typeof(G)}(zeros(T, 2), G, [1, 2], nothing, zeros(Int, 2), zeros(Int, 2), 0),
                 () -> similar(AtomIterate(2, G), T),
                 () -> convert(AtomIterate{T, typeof(G)}, AtomIterate(2, G)),
                 () -> convert(AtomIterate{T}, AtomIterate(2, G)))
      rejects(make, :AtomIterate)
    end
  end
  for T in (Int, Bool, Float32, Float64)
    @test eltype(SparseIterate(T, 2)) == T
    @test eltype(SymmetricSparseIterate(T, 2)) == T
    @test eltype(AtomIterate(T, 2, 2)) == T
  end
end

@testset "real sparse and atom products" begin
  for x in (SparseIterate([1., 0.]), SparseIterate([1. 0.; 0. 2.]), AtomIterate([1., 0.], 2),
            SparseIterate(2), AtomIterate(2, 2))
    name = nameof(typeof(x))
    out = fill(7., size(x))
    for A in (ones(2, 2), Symmetric(ones(2, 2)), transpose(ones(2, 2)))
      # The transposed specialization is for sparse vectors only.
      A isa Transpose && !(x isa SparseIterate{<:Any, 1}) && continue
      Z = A isa Transpose ? transpose(complex.(parent(A))) : complex.(A)
      rejects(() -> Z * x, name)
      rejects(() -> mul!(out, Z, x), name, out, x)
      rejects(() -> mul!(out, Z, x, 0, 2), name, out, x)
      cout = complex.(out)
      rejects(() -> mul!(cout, A, x), name, cout, x)
      rejects(() -> mul!(cout, Z, x), name, cout, x)
      for (α, β) in ((1 + 0im, 0), (0, 1 + 0im), (1 + im, 2), (0, 2 + im))
        rejects(() -> mul!(out, A, x, α, β), name, out, x)
      end
    end
    if ndims(x) == 1
      for v in (ComplexF64[1, 2], ComplexF64[1 + im, 2])
        rejects(() -> dot(v, x), name, x)
        rejects(() -> dot(x, v), name, x)
      end
    else
      for (a, b) in ((ComplexF64[1, 2], [1., 2.]), ([1., 2.], ComplexF64[1, 2]))
        rejects(() -> dot(a, x, b), name, x)
      end
      A = Symmetric(ones(2, 2)); Z = Symmetric(complex.(ones(2, 2)))
      rejects(() -> *(Z, x, A), name, x)
      rejects(() -> *(A, x, Z), name, x)
    end
  end
end

@testset "real symmetric iterate products and writers" begin
  x = SymmetricSparseIterate([1. 0.; 0. 2.])
  name = :SymmetricSparseIterate
  A = ones(2, 2); Z = complex.(A)
  for B in ([1., 2.], A, UpperTriangular(A)), C in (fill(7., size(B)), fill(7 + 0im, size(B)))
    CZ = complex.(B)
    rejects(() -> mul!(C, x, CZ), name, C, x)
    rejects(() -> mul!(C, x, CZ, 0, 2), name, C, x)
    if eltype(C) <: Complex
      rejects(() -> mul!(C, x, B), name, C, x)
    end
    for (α, β) in ((1 + 0im, 0), (0, 1 + 0im))
      rejects(() -> mul!(C, x, B, α, β), name, C, x)
    end
  end
  rejects(() -> x * ComplexF64[1, 2], name, x)
  for (L, R) in ((Z, A), (A, Z), (UpperTriangular(Z), A), (A, UpperTriangular(Z)), (im * I, A), (A, im * I))
    rejects(() -> mul!(x, L, R), name, x)
    rejects(() -> mul!(x, L, R, 0, 2), name, x)
  end
  for (α, β) in ((1 + 0im, 0), (0, 1 + 0im))
    rejects(() -> mul!(x, A, A, α, β), name, x)
  end
  for (a, b) in ((ComplexF64[1, 2], [1., 2.]), ([1., 2.], ComplexF64[1, 2]))
    rejects(() -> dot(a, x, b), name, x)
  end
  rejects(() -> mul!(x, x, Z), name, x)
  rejects(() -> mul!(x, x, Z, 0, 2), name, x)
  rejects(() -> dot(Z, x), name, x)
  rejects(() -> dot(x, Z), name, x)
  rejects(() -> *(Symmetric(Z), x, Symmetric(A)), name, x)
  rejects(() -> *(Symmetric(A), x, Symmetric(Z)), name, x)
  for source in (Z, sparse(Z), transpose(sparse(Z)), Symmetric(Z), Hermitian(Z))
    rejects(() -> copyto!(x, source), name, x)
  end
  for scale! in (s -> rmul!(x, s), s -> lmul!(s, x), s -> rdiv!(x, s), s -> ldiv!(s, x))
    rejects(() -> scale!(1 + 0im), name, x)
  end
  rejects(() -> (x .= real.(Z)), name, x)
  rejects(() -> map!(real, x, Z), name, x)
  rejects(() -> map!(+, x, A, Z), name, x)
  rejects(() -> map!((a, b, c, d) -> a + b + c + real(d), x, A, A, A, Z), name, x)
  rejects(() -> axpy!(0, Z, x), name, x)
  rejects(() -> axpby!(0, A, 1 + 0im, x), name, x)
  for B in ([1., 2.], A, UpperTriangular(A))
    out = zeros(size(B))
    @test mul!(out, x, B, 2., 0.) ≈ 2Matrix(x) * B
  end
  out = zeros(Float32, 2, 2)
  @test mul!(out, x, Float32.(A)) ≈ Matrix(x) * A
end

@testset "broadcast data and control arguments" begin
  A = [1.2 2.8; 2.8 3.1]
  for x in (SparseIterate(A), SymmetricSparseIterate(A), AtomIterate(vec(A), 2))
    data = reshape(A, size(x))
    @test (x .= round.(x, RoundDown)) == floor.(data)
    @test (x .= ((v, f) -> f(v)).(data, identity)) == data
    @test (x .= ((v, label) -> label === :keep ? v : -v).(data, Ref(:keep))) == data
    @test (x .= ((v, T) -> T(v)).(data, Float64)) == data
    @test (x .= ((v, pair) -> v + sum(pair)).(data, Ref((1, 2)))) == data .+ 3
    for scale in (true, 2, 1//2), value in (scale, Ref(scale))
      @test (x .= data .* value) == data .* scale
      @test (x .= value) == fill(scale, size(x))
    end
    for value in (1 + 2im, Ref(1 + 2im))
      rejects(() -> (x .= value), nameof(typeof(x)), x)
      rejects(() -> (x .= data .+ value), nameof(typeof(x)), x)
      rejects(() -> (x .= real.(data .+ value)), nameof(typeof(x)), x)
    end
  end
  x = SymmetricSparseIterate(2)
  modes = fill(RoundDown, 2, 2)
  labels = fill(:keep, 2, 2)
  @test map!((v, mode) -> round(v, mode), x, A, modes) == floor.(A)
  @test map!(mode -> round(1.2, mode), x, modes) == ones(2, 2)
  @test map!((a, b, mode) -> round(a + b, mode), x, A, A, modes) == floor.(2A)
  @test map!((a, b, mode, label) -> label === :keep ? round(a + b, mode) : 0,
             x, A, A, modes, labels) == floor.(2A)
end

@testset "real scalar and structured operands" begin
  A = [1. 2.; 2. 3.]
  for x in (SparseIterate([1., 2.]), AtomIterate([1., 2.], 2))
    for B in (A, Diagonal([1., 2.]), transpose(A), adjoint(A))
      for (α, β) in ((true, false), (2, 1), (1//2, 1//3))
        out = fill(3., 2)
        @test mul!(out, B, x, α, β) ≈ α .* (B * Vector(x)) .+ β .* 3
      end
    end
  end
  x = SymmetricSparseIterate(A)
  for B in (Diagonal([1., 2.]), transpose(A), adjoint(A))
    @test mul!(zeros(2, 2), x, B) == A * B
  end
  for scale in (true, 2, 1//2)
    @test Matrix(rmul!(copy(x), scale)) == A .* scale
    @test Matrix(lmul!(scale, copy(x))) == scale .* A
    @test Matrix(rdiv!(copy(x), scale)) == A ./ scale
    @test Matrix(ldiv!(scale, copy(x))) == A ./ scale
    for (L, R) in ((A, scale * I), (scale * I, A))
      y = copy(x)
      @test mul!(y, L, R, true, false) == A .* scale
    end
  end
end

@testset "iterate dispatch" begin
  ambiguities = Test.detect_ambiguities(ProximalBase, LinearAlgebra; recursive = true)
  owned = filter(ambiguities) do pair
    any(m -> basename(string(m.file)) in ("sparse_iterate.jl", "atom_iterate.jl"), pair)
  end
  @test isempty(owned)
end

end
