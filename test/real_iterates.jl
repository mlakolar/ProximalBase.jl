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
    elseif out isa BlockIterate
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
    for make in (() -> BlockIterate(T, 2, G), () -> BlockIterate(T, 2, 2),
                 () -> BlockIterate(zeros(T, 2), G), () -> BlockIterate(zeros(T, 2), 2),
                 () -> BlockIterate{T, typeof(G)}(zeros(T, 2), G, [1, 2], nothing, zeros(Int, 2), zeros(Int, 2), 0),
                 () -> similar(BlockIterate(2, G), T),
                 () -> convert(BlockIterate{T, typeof(G)}, BlockIterate(2, G)),
                 () -> convert(BlockIterate{T}, BlockIterate(2, G)))
      rejects(make, :BlockIterate)
    end
  end
  for T in (Int, Bool, Float32, Float64)
    @test eltype(SparseIterate(T, 2)) == T
    @test eltype(SymmetricSparseIterate(T, 2)) == T
    @test eltype(BlockIterate(T, 2, 2)) == T
  end
end

@testset "real sparse and block products" begin
  for x in (SparseIterate([1., 0.]), SparseIterate([1. 0.; 0. 2.]), BlockIterate([1., 0.], 2),
            SparseIterate(2), BlockIterate(2, 2))
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
  for (a, b) in ((ComplexF64[1, 2], [1., 2.]), ([1., 2.], ComplexF64[1, 2]))
    rejects(() -> dot(a, x, b), name, x)
  end
  rejects(() -> dot(Z, x), name, x)
  rejects(() -> dot(x, Z), name, x)
  rejects(() -> *(Symmetric(Z), x, Symmetric(A)), name, x)
  rejects(() -> *(Symmetric(A), x, Symmetric(Z)), name, x)
  for source in (Z, sparse(Z), transpose(sparse(Z)), Symmetric(Z), Hermitian(Z))
    rejects(() -> copyto!(x, source), name, x)
  end
  # the vector of coordinates is an ordinary vector: it converts one value at a time
  @test_throws InexactError (coordinates(x)[2] = 2im)
  @test_throws InexactError fill!(coordinates(x), 1 + 1im)
  for B in ([1., 2.], A, UpperTriangular(A))
    out = zeros(size(B))
    @test mul!(out, matrixview(x), B, 2., 0.) ≈ 2Matrix(x) * B
  end
  out = zeros(Float32, 2, 2)
  @test mul!(out, matrixview(x), Float32.(A)) ≈ Matrix(x) * A
end

@testset "broadcast data and control arguments" begin
  A = [1.2 2.8; 2.8 3.1]
  for x in (SparseIterate(A), BlockIterate(vec(A), 2))
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
end

@testset "real scalar and structured operands" begin
  A = [1. 2.; 2. 3.]
  for x in (SparseIterate([1., 2.]), BlockIterate([1., 2.], 2))
    for B in (A, Diagonal([1., 2.]), transpose(A), adjoint(A))
      for (α, β) in ((true, false), (2, 1), (1//2, 1//3))
        out = fill(3., 2)
        @test mul!(out, B, x, α, β) ≈ α .* (B * Vector(x)) .+ β .* 3
      end
    end
  end
  x = SymmetricSparseIterate(A)
  for B in (Diagonal([1., 2.]), transpose(A), adjoint(A))
    @test mul!(zeros(2, 2), matrixview(x), B) == A * B
  end
  for scale in (true, 2, 1//2)
    y = copy(x); rmul!(coordinates(y), scale)
    @test Matrix(y) == A .* scale
    y = copy(x); lmul!(scale, coordinates(y))
    @test Matrix(y) == scale .* A
    y = copy(x); rdiv!(coordinates(y), scale)
    @test Matrix(y) == A ./ scale
    y = copy(x); ldiv!(scale, coordinates(y))
    @test Matrix(y) == A ./ scale
  end
end

@testset "real-only bulk writes of SparseIterate and BlockIterate" begin
  c = ComplexF64[1, 2im, 0, 0]
  C = reshape(c, 2, 2)
  hasdense = isdefined(SparseArrays, :CHOLMOD) && isdefined(SparseArrays.CHOLMOD, :Dense)
  vector_iterates = ((() -> (x = SparseIterate(4); x[4] = 9.0; x), :SparseIterate),
                     (() -> (x = BlockIterate(4, 2); x[4] = 9.0; x), :BlockIterate))
  for (make, name) in vector_iterates
    for write! in (x -> copyto!(x, c), x -> copy!(x, c), x -> (x .= c), x -> (x .= 2im),
                   x -> fill!(x, 2im), x -> fill!(x, 1 + 0im), x -> fill!(x, 0im),
                   x -> copyto!(x, ComplexF64[1, 0, 0, 0]), x -> copyto!(x, sparsevec(c)),
                   x -> copyto!(x, ComplexF64[]))
      x = make()
      rejects(() -> write!(x), name, x)
    end
    if hasdense
      x = make()
      rejects(() -> copyto!(x, SparseArrays.CHOLMOD.Dense(reshape(c, 4, 1))), name, x)
    end
    # copy! from another length fails at resize! before the check
    x = make(); saved = copy(x)
    @test_throws MethodError copy!(x, ComplexF64[1, 2im])
    @test isequal(x, saved) && collect(activecoordinates(x)) == collect(activecoordinates(saved))
    # writes through wrappers follow Base's conversion
    @test_throws InexactError (view(make(), 1:2) .= ComplexF64[1, 2im])
    @test_throws InexactError (make()[:] = c)
    @test_throws InexactError copyto!(make(), Number[1.0, 2im, 0, 0])
  end
  @test_throws InexactError (blockview(BlockIterate(4, 2), 1) .= ComplexF64[1, 2im])
  @test_throws Exception (coordinates(SparseIterate(4)) .= c)
  # the coordinate view has no check of its own: it writes as a plain vector of the element type
  let fixture() = (x = SparseIterate(4); x[3] = 1.0; x)
    x = fixture()
    @test_throws InexactError (coordinates(x) .= c)
    @test Vector(x) == [1.0, 0, 1, 0]
    x = fixture()
    @test_throws InexactError copyto!(coordinates(x), c)
    @test Vector(x) == [1.0, 0, 1, 0]
    x = fixture()
    @test_throws InexactError (coordinates(x)[2] = 2im)
    @test Vector(x) == [0.0, 0, 1, 0]
  end

  makem() = (x = SparseIterate(2, 2); x[4] = 9.0; x)
  for write! in (x -> copyto!(x, c), x -> copyto!(x, C), x -> copy!(x, C), x -> (x .= C), x -> fill!(x, 2im),
                 x -> copyto!(x, sparse(C)), x -> copyto!(x, view(sparse(C), :, :)),
                 x -> copyto!(x, sparse(C)'), x -> copyto!(x, transpose(sparse(C))),
                 x -> copyto!(x, 2im * I), x -> copyto!(x, (1 + 0im) * I),
                 x -> copyto!(x, qr(ComplexF64[1 2; 3 4]).Q))
    x = makem()
    rejects(() -> write!(x), :SparseIterate, x)
  end
  if hasdense
    x = makem()
    rejects(() -> copyto!(x, SparseArrays.CHOLMOD.Dense(C)), :SparseIterate, x)
  end
  # transposed sparse vectors into a 1×4 matrix
  for src in (sparsevec(c)', transpose(sparsevec(c)), view(sparsevec(c), :)')
    x = SparseIterate(1, 4); x[1, 4] = 9.0
    rejects(() -> copyto!(x, src), :SparseIterate, x)
  end
  @test_throws InexactError (vec(makem()) .= c)

  # vec and the full-length reshape of a vector are the iterate itself
  s0 = SparseIterate([0., 1.])
  b0 = BlockIterate([0., 1.], [1:1, 2:2])
  rejects(() -> (vec(s0) .= ComplexF64[1, 2im]), :SparseIterate, s0)
  rejects(() -> (reshape(s0, 2) .= ComplexF64[1, 2im]), :SparseIterate, s0)
  rejects(() -> (vec(b0) .= ComplexF64[1, 2im]), :BlockIterate, b0)
  rejects(() -> SparseIterate(ComplexF64, 2), :SparseIterate)
  rejects(() -> BlockIterate(ComplexF64, 4, 2), :BlockIterate)

  # real sources keep their results
  S = sparse([1.0 0; 0 2])
  for src in (S, view(S, :, :), view(S, :, 1:2), S', transpose(S))
    y = SparseIterate(2, 2); y[2, 1] = 5.0
    @test copyto!(y, src) === y
    @test Matrix(y) == [1 0; 0 2] && nactive(y) == 3
  end
  v = sparsevec([1.0, 0, 0, 2])
  for src in (v', transpose(v), view(v, :)')
    y = SparseIterate(1, 4); y[1, 2] = 5.0
    @test copyto!(y, src) === y
    @test Matrix(y) == [1 0 0 2]
  end
  @test Vector(copyto!(SparseIterate(4), S)) == [1, 0, 0, 2]
  @test Vector(copyto!(BlockIterate(4, 2), S)) == [1, 0, 0, 2]
  if hasdense
    D = SparseArrays.CHOLMOD.Dense
    v = reshape([1.0, 0, 0, 2], 4, 1)
    @test Vector(copyto!(SparseIterate(4), D(v))) == [1, 0, 0, 2]
    @test Matrix(copyto!(SparseIterate(2, 2), D([1.0 0; 0 2]))) == [1 0; 0 2]
    @test Matrix(copyto!(SparseIterate(Float32, 2, 2), D([1.0 0; 0 2]))) == Float32[1 0; 0 2]
    @test Vector(copyto!(BlockIterate(4, 2), D(v))) == [1, 0, 0, 2]
    @test Vector(copyto!(BlockIterate(Float32, 4, 2), D(v))) == Float32[1, 0, 0, 2]
  end
  @test Matrix(copyto!(makem(), 2I)) == [2 0; 0 2]
  Q = qr([2.0 1; 1 3], ColumnNorm()).Q
  @test Matrix(copyto!(makem(), Q)) ≈ Matrix(Q)
  for (make, name) in vector_iterates
    @test Vector(copyto!(make(), Any[1.0, 2.0, 0, 0])) == [1, 2, 0, 0]
    @test Vector(copyto!(make(), [7.0])) == [7, 0, 0, 9]
    x = make()
    @test_throws BoundsError copyto!(x, zeros(5))
    @test Vector(x) == [0, 0, 0, 9]
    for src in ([1, 2, 0, 0], [true, false, false, true], Float32[1, 2, 0, 0])
      @test Vector(copyto!(make(), src)) == src
    end
    @test Vector(copyto!(make(), Float64[])) == [0, 0, 0, 9]
    @test Vector(copy!(make(), [1.0, 2, 3, 4])) == [1, 2, 3, 4]
    @test Vector(fill!(make(), 2)) == fill(2, 4)
    x = fill!(make(), 0)                       # a value write keeps the working set
    @test Vector(x) == zeros(4) && nactive(x) == 1 && nactive(dropzeros!(x)) == 0
  end
end

@testset "broadcast results" begin
  makers = ((() -> (x = SparseIterate(4); x[4] = 7.0; x), :SparseIterate, (4,)),
            (() -> (x = BlockIterate(4, 2); x[4] = 7.0; x), :BlockIterate, (4,)),
            (() -> (x = SparseIterate(2, 2); x[2, 2] = 7.0; x), :SparseIterate, (2, 2)))
  for (make, name, dims) in makers
    shaped(v) = reshape(v, dims)
    vals(x) = ndims(x) == 2 ? Matrix(x) : Vector(x)
    z1 = shaped([1.0, 0, 0, 0]); z2 = shaped([0.0, 1, 0, 0])
    opaque = Ref{Any}(complex)
    # a complex result is rejected before any write, even with a zero imaginary part
    for write! in (x -> (x .= complex.(z1, z2)), x -> (x .= complex.(z1)),
                   x -> (x .= Complex.(z1, 0.0)), x -> (x .= (v -> v + 0im).(z1)),
                   x -> (x .= sqrt.(Complex.(z1))), x -> (x .= complex.(x, 0.0)),
                   x -> (x .= (v -> opaque[](v)).(z1)), x -> (x .= ((u, v) -> opaque[](u, v)).(z1, z2)),
                   x -> (x .= (v -> v > 0.5 ? 1im : 1.0).(z1)),
                   x -> (x .= (v -> v > 0.5 ? 1.0 : 1im).(shaped([0.0, 0, 0, 1.0]))),
                   x -> (x .= shaped(Any[1.0, 2im, 0, 0])),
                   x -> (x .= (v -> v).(shaped(Any[1.0, 2im, 0, 0]))),
                   x -> (x .= (v -> v).(shaped(Any[1.0, 0, 0, 0im]))),
                   x -> (x .= (v -> v).(shaped(Number[1.0, 2, 3, 4im]))),
                   x -> (x .= (v -> v > 5 ? 1im : 2v).(x)),
                   # complex leaves and scalars, rejected as before
                   x -> (x .= z1 .+ 0im), x -> (x .= fill(0im, dims)), x -> (x .= 1.0 + 0im),
                   x -> (x .= Ref(1.0) .+ 0im), x -> (x .= sqrt.(complex.(1.0))),
                   x -> (x .= exp.(im .* x)), x -> (x .+= 1im),
                   x -> (x .= real.(z1 .+ (1 + 2im))),
                   # opaque[] is read before the broadcast, so this is the known-complex path
                   x -> (x .= opaque[].(z1)))
      x = make()
      rejects(() -> write!(x), name, x)
    end
    # the complex element of a Union result at the last position leaves no partial write
    x = make(); saved = vals(x)
    rejects(() -> (x .= (v -> v > 0.5 ? 1im : v).(shaped([0.0, 0.25, 0.0, 1.0]))), name, x)
    @test vals(x) == saved
    # an exception of f while the result is evaluated happens before any write
    late = Ref{Any}(v -> v == 4 ? error("late") : v)
    late_f(v) = late[](v)
    x = make(); saved = vals(x); nsaved = nactive(x); w = shaped([1.0, 2, 3, 4])
    @test_throws ErrorException (x .= late_f.(w))
    @test vals(x) == saved && nactive(x) == nsaved
    x = make(); saved = vals(x); nsaved = nactive(x)
    @test_throws DomainError (x .= sqrt.(-z1 .- 1))
    @test vals(x) == saved && nactive(x) == nsaved
    # real values of a Union, Any or Number result are written
    x = make()
    @test vals((x .= (v -> v > 5 ? 1im : 1.0).(z1))) == fill(1.0, dims)
    x = make()
    @test vals((x .= shaped(Number[1.0, 2, 3, 4]))) == shaped([1.0, 2, 3, 4])
    x = make()
    @test vals((x .= (v -> v).(shaped(Number[1.0, 2, 3, 4])))) == shaped([1.0, 2, 3, 4])
    x = make()
    @test vals((x .= shaped(Any[1.0, 2, 0, 0]))) == shaped([1.0, 2, 0, 0])
    # a staged result is computed from the unmodified x, then written
    g = Ref{Any}(v -> v > 5 ? 10.0 : 2v)
    gf(v) = g[](v)
    x = make(); expected = gf.(vals(x))
    @test vals((x .= gf.(x))) == expected
    x = make(); w = shaped([1.0, 2, 3, 4])
    @test vals((x .= gf.(x .+ w))) == gf.(w .+ shaped([0.0, 0, 0, 7]))
    # the staged path writes what the direct path writes: values, working set, nactive
    same(x, y) = vals(x) == vals(y) && nactive(x) == nactive(y) &&
                 collect(activecoordinates(x)) == collect(activecoordinates(y))
    gid = Ref{Any}(identity)
    opq(v) = gid[](v)
    for (staged, direct) in (
        (x -> (x .= shaped(Number[1.0, 2, 3, 4])), x -> (x .= shaped([1.0, 2, 3, 4]))),
        (x -> (x .= (v -> opq(v)).(shaped([1.0, 2, 3, 4]))), x -> (x .= shaped([1.0, 2, 3, 4]))),
        (x -> (x .= (v -> v > 5 ? 1im : v).(z2)), x -> (x .= z2)),
        (x -> (x .= (v -> v > 0.5 ? 1.0 : v).(shaped([0.0, 0, 0, 1.0]))), x -> (x .= shaped([0.0, 0, 0, 1.0]))))
      x = make(); x2 = make()
      staged(x); direct(x2)
      @test same(x, x2)
    end
    # singleton expansion and an empty destination keep the axes of the broadcast
    if dims == (4,)
      for staged in (x -> (x .= Number[2.0]), x -> (x .= (v -> v).(Number[2.0])),
                     x -> (x .= (v -> opq(v)).([2.0])), x -> (x .= (v -> opq(v)).(Ref(2.0))),
                     x -> (x .= (v -> opq(v)).((2.0,))), x -> (x .= (v -> opq(v)).(Any[2.0])))
        x = make(); x2 = make()
        staged(x); x2 .= [2.0]
        @test vals(x) == fill(2.0, 4) && same(x, x2)
      end
    else
      x = make(); x2 = make()
      x .= Number[1.0, 2.0]; x2 .= [1.0, 2.0]
      @test vals(x) == [1.0 1.0; 2.0 2.0] && same(x, x2)
      x = make(); x2 = make()
      x .= (v -> opq(v)).(reshape(Number[1.0, 2.0], 1, 2)); x2 .= reshape([1.0, 2.0], 1, 2)
      @test vals(x) == [1.0 2.0; 1.0 2.0] && same(x, x2)
    end
    # the staged function reads the destination while the result is evaluated: it sees the old x
    x = make(); idx = ndims(x) == 2 ? (1, 1) : (1,); x[idx...] = 10.0
    old = vals(x); h = Ref{Any}(+)
    x .= (v -> h[](v, x[idx...])).(shaped([1.0, 2, 3, 4]))
    @test vals(x) == old[idx...] .+ shaped([1.0, 2, 3, 4])
    # accepted as before
    x = make(); @test vals((x .= real.(complex.(z1, z2)))) == z1
    x = make(); @test vals((x .= abs.(complex.(z1, z2)))) == z1 .+ z2
    x = make(); @test vals((x .= Float32.(shaped([1, 2, 3, 4])))) == shaped([1.0, 2, 3, 4])
    x = make(); @test vals((x .= (v -> BigFloat(v)).(z1))) == z1
    x = make(); @test vals((x .+= 1)) == shaped([1.0, 1, 1, 8])
    x = make(); @test vals((x .= 2 .* x)) == shaped([0.0, 0, 0, 14])
  end
  # an empty destination: a broadcast that is valid for it writes nothing
  for x in (SparseIterate(0), SparseIterate(0, 2), BlockIterate(Float64, 0, Vector{Int}[]))
    @test (x .= Number[]) === x
    @test (x .= Number[2.0]) === x
    @test nactive(x) == 0 && isempty(x)
  end
  y = SparseIterate(0, 2); y .= Number[]
  @test size(Matrix(y)) == (0, 2)
  # the 0-dimensional methods reject a complex value through fill!
  for (make, name, dims) in makers
    for write! in (x -> (x .= 1im), x -> (x .= Ref(1im)), x -> (x .= complex.(1.0)))
      x = make()
      rejects(() -> write!(x), name, x)
    end
  end
end

@testset "iterate dispatch" begin
  ambiguities = Test.detect_ambiguities(ProximalBase, LinearAlgebra; recursive = true)
  owned = filter(ambiguities) do pair
    any(m -> basename(string(m.file)) in ("iterate_storage.jl", "sparse_iterate.jl", "symmetric_sparse_iterate.jl", "block_iterate.jl"), pair)
  end
  @test isempty(owned)
end

# Every source that SparseArrays or LinearAlgebra copy with a method of their own reaches a
# ProximalBase method, which checks the element type before that writer runs; a source of the
# element type of the iterate included, where CHOLMOD has typed methods of its own.
@testset "copyto! sources reach the real-only check" begin
  S = sparse([1.0 0; 0 2]); v = sparsevec([1.0, 0, 0, 2])
  hasdense = isdefined(SparseArrays, :CHOLMOD) && isdefined(SparseArrays.CHOLMOD, :Dense)
  for T in (Float32, Float64)
    matrix_sources = Any[T[1 0; 0 2], sparse(T[1 0; 0 2]), S, view(S, :, :), view(S, :, 1:2), S', transpose(S),
                         2I, qr(T[2 1; 1 3]).Q, qr(T[2 1; 1 3], ColumnNorm()).Q, lq(T[2 1; 1 3]).Q, hessenberg(T[2 1; 1 3]).Q]
    row_sources = Any[v', transpose(v), view(v, :)']
    vector_sources = Any[T[1, 0, 0, 2], v, S]
    if hasdense
      D = SparseArrays.CHOLMOD.Dense
      push!(matrix_sources, D(T[1 0; 0 2]), D([1.0 0; 0 2]))
      push!(vector_sources, D(reshape(T[1, 0, 0, 2], 4, 1)), D(reshape([1.0, 0, 0, 2], 4, 1)))
    end
    for y in vcat(matrix_sources, row_sources)
      @test parentmodule(which(copyto!, Tuple{SparseIterate{T, 2}, typeof(y)})) === ProximalBase
    end
    for y in vector_sources, X in (SparseIterate{T, 1}, typeof(BlockIterate(T, 4, 2)))
      @test parentmodule(which(copyto!, Tuple{X, typeof(y)})) === ProximalBase
    end
    # these sources write the values and the activity of the dense source of the same values
    state(x) = (Array(x), collect(activecoordinates(x)), nactive(x))
    fresh2() = (x = SparseIterate(T, 2, 2); x[2, 1] = 5; x)
    fresh1() = (x = SparseIterate(T, 4); x[3] = 5; x)
    freshb() = (x = BlockIterate(T, 4, 2); x[3] = 5; x)
    for y in matrix_sources
      y isa UniformScaling && continue
      dense = y isa LinearAlgebra.AbstractQ ? T.(Matrix(y)) : T.(Array(y))
      @test state(copyto!(fresh2(), y)) == state(copyto!(fresh2(), dense))
    end
    for y in vector_sources, fresh in (fresh1, freshb)
      @test state(copyto!(fresh(), y)) == state(copyto!(fresh(), T.(vec(Array(y)))))
    end
  end
end

end
