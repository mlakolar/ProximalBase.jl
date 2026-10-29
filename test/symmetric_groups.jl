module SymmetricGroupsTest

using Test, ProximalBase, SparseArrays

# compare entries in bounded units, even when the full matrix norm overflows
function entries_match(A, B; rtol)
  @test size(A) == size(B)
  scale = maximum(abs, filter(isfinite, vec(B)); init = zero(eltype(B)))
  scale = iszero(scale) ? one(scale) : scale
  for i in eachindex(B)
    if isfinite(B[i])
      @test isfinite(A[i])
      @test isapprox(A[i] / scale, B[i] / scale; rtol, atol = rtol)
    else
      @test isequal(A[i], B[i])
    end
  end
end

function value_matches(v, ref, rtol)
  if isfinite(ref)
    @test isfinite(v)
    @test isapprox(v, ref; rtol, atol = rtol)
  else
    @test isequal(v, ref)
  end
end

function check_case(f, A, γ; output_type = eltype(A), finite = true)
  tol = output_type === Float32 ? 3e-6 : 3e-14
  for op in (prox!, value_and_prox!)
    reference = zeros(output_type, size(A))
    refvalue = op(reference, f, copy(A), γ)
    finite && @test all(isfinite, reference)
    for source in (Matrix, SparseIterate, SymmetricSparseIterate)
      x = source(copy(A))
      before = Matrix(x)
      y = SymmetricSparseIterate(output_type, size(A, 1))
      fill!(y, -1)
      v = op(y, f, x, γ)
      entries_match(Matrix(y), reference; rtol = tol)
      @test isequal(Matrix(x), before)
      if op === prox!
        @test v === y
      else
        value_matches(v, refvalue, tol)
      end
    end
    if output_type === eltype(A)
      x = SymmetricSparseIterate(copy(A))
      v = op(x, f, x, γ)
      entries_match(Matrix(x), reference; rtol = tol)
      op === prox! ? (@test v === x) : value_matches(v, refvalue, tol)
    end
  end
end

function check_vector_steps(f, A, γ)
  tol = eltype(A) === Float32 ? 3e-6 : 3e-14
  for op in (prox!, value_and_prox!)
    reference = similar(A)
    refvalue = op(reference, f, copy(A), γ)
    @test all(isfinite, reference)
    for inplace in (false, true)
      x = SymmetricSparseIterate(A)
      y = inplace ? x : SymmetricSparseIterate(eltype(A), size(A, 1))
      v = op(y, f, x, γ)
      entries_match(Matrix(y), reference; rtol = tol)
      inplace || @test isequal(Matrix(x), A)
      op === prox! ? (@test v === y) : value_matches(v, refvalue, tol)
    end
  end
end

function rejects_group_steps(f, A, γ)
  for op in (prox!, value_and_prox!), inplace in (false, true)
    x = SymmetricSparseIterate(A)
    T = eltype(A)
    y = inplace ? x : SymmetricSparseIterate(T[7 0; 0 0])
    activate!(y, tril_index(2, 2, 1))
    before = (Matrix(y), nnz(y), copy(activecoordinates(y)), copy(y.nzval))
    err = try
      op(y, f, x, γ)
    catch e
      e
    end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("the groups of $(typeof(f)) must be symmetric for a SymmetricSparseIterate output", msg)
    @test occursin("position (2, 1)", msg) && occursin("position (1, 2)", msg)
    @test occursin(length(f.groups) == 2 ? "in group 2" : "in no group", msg)
    @test isequal((Matrix(y), nnz(y), copy(activecoordinates(y)), copy(y.nzval)), before)
  end
end

@testset "symmetric group writes" begin
  @testset "$T" for T in (Float64, Float32)
    M = floatmax(T)
    A = T[1 2; 2 3]
    huge = T[0 M; M 0]
    @testset "NormL2" begin
      f = NormL2(T(1)/4)
      for γ in (one(T), M)
        check_case(f, huge, γ)
      end
      check_case(f, T[M/2 M/2 M/2; M/2 M/2 M/2; M/2 M/2 M/2], M)
      check_case(f, A, one(T))
      check_case(f, A, M)
      check_case(NormL2(zero(T)), huge, M)
      # exercise the rescaled zero branch with a finite threshold
      small = fill(M/4, 2, 2)
      for source in (Matrix, SparseIterate, SymmetricSparseIterate), inplace in (false, true)
        x = source(small)
        y = inplace ? SymmetricSparseIterate(small) : SymmetricSparseIterate(T, 2)
        inplace && (x = y)
        @test iszero(ProximalBase._norm2_shrink!(f, y, x, M))
        @test all(iszero, y)
        x = source(small)
        y = inplace ? SymmetricSparseIterate(small) : SymmetricSparseIterate(T, 2)
        inplace && (x = y)
        G = 1:4
        W = ProximalBase._group_writes(y, G)
        @test iszero(ProximalBase._rescaled_group_prox!(f, y, x, G, M, M/4, W))
        @test all(iszero, y)
      end
      for B in (T[0 Inf; Inf 0], T[0 NaN; NaN 0], T[Inf NaN; NaN 0])
        check_case(f, B, one(T); finite = false)
        x = SymmetricSparseIterate(B)
        v = value_and_prox!(x, f, x, one(T))
        if any(isnan, B)
          @test all(iszero, x)
          @test iszero(v)
        else
          @test isinf(x[1, 2]) && isinf(x[2, 1]) && isinf(v)
        end
      end
    end
    @testset "group penalties" begin
      for groups in ([1:4], [[2, 3], [1], [4]])
        for f in (GroupNormL2(T(1)/4, groups), GroupNormLinf(T(1)/4, groups),
                  SparseGroupNormL2(T(1)/16, T(1)/4, groups))
          check_case(f, A, one(T))
          check_vector_steps(f, A, T.(1:length(groups)) ./ T(4))
          check_case(f, A, T(100))
          for γ in (one(T), M)
            check_case(f, huge, γ)
          end
        end
        for f in (GroupNormL2(zero(T), groups), GroupNormLinf(zero(T), groups),
                  SparseGroupNormL2(T(1)/8, zero(T), groups),
                  SparseGroupNormL2(T(1)/8, T(1)/4, groups, zeros(T, length(groups))),
                  GroupNormL2(T(1)/4, groups, zeros(T, length(groups))),
                  GroupNormLinf(T(1)/4, groups, zeros(T, length(groups))))
          check_case(f, A, one(T))
        end
      end
      # positions 3, 7 and 9 are ungrouped, including beyond group_of
      groups = [[2, 4], [6, 8], [1, 5]]
      for B in (T[1 2 3; 2 4 5; 3 5 6], fill(M/2, 3, 3))
        for f in (GroupNormL2(T(1)/16, groups), GroupNormLinf(T(1)/16, groups),
                  SparseGroupNormL2(T(1)/32, T(1)/16, groups))
          check_case(f, B, one(T))
        end
      end
      for groups in ([[2], [1, 3, 4]], [[2]], [[3]])
        for f in (GroupNormL2(T(1)/4, groups), GroupNormLinf(T(1)/4, groups),
                  SparseGroupNormL2(T(1)/8, T(1)/4, groups))
          rejects_group_steps(f, A, one(T))
          rejects_group_steps(f, A, fill(one(T), length(groups)))
        end
      end
    end
  end
  @testset "mixed precision preserves the intermediate store" begin
    A = [floatmax(Float64) 1e20; 1e20 floatmax(Float64)]
    for f in (NormL2(1/4), GroupNormL2(1/4, [1:4]), GroupNormLinf(1/4, [1:4]),
              SparseGroupNormL2(1/16, 1/4, [1:4]))
      check_case(f, A, 1.0; output_type = Float32, finite = false)
    end
    y = SymmetricSparseIterate(Float32, 2)
    prox!(y, NormL2(1/4), A, 1.0)
    @test isequal(y[2, 1], 0.0f0)
  end
end

end
