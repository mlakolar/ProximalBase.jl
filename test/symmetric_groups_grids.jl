module SymmetricGroupsGridsTest

# The Float32 pass of the symmetric group writes for GroupNormLinf (its levels, weights and zero-level
# variants in group penalties, and its rejects_group_steps cases) and for GroupSCAD (the radial groups).
# symmetric_groups.jl runs Float32 for NormL2, GroupNormL2, SparseGroupNormL2 and GroupMCP, and Float64
# for all functions. The helpers are copied from it.

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
      fill!(coordinates(y), -1)
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

function check_symmetric_steps(f, A, γ; finite = true)
  tol = eltype(A) === Float32 ? 3e-6 : 3e-14
  for op in (prox!, value_and_prox!)
    reference = similar(A)
    refvalue = op(reference, f, copy(A), γ)
    finite && @test all(isfinite, reference)
    for inplace in (false, true)
      x = SymmetricSparseIterate(A)
      y = inplace ? x : SymmetricSparseIterate(eltype(A), size(A, 1))
      if γ isa AbstractArray
        # a step per group is not defined for an owner: refused before anything is written
        before = (Matrix(y), nactive(y), copy(activecoordinates(y)), copy(y.nzval))
        err = try
          op(y, f, x, γ)
        catch e
          e
        end
        @test err isa ArgumentError
        @test occursin("is not supported with a SymmetricSparseIterate as the output", sprint(showerror, err))
        @test isequal((Matrix(y), nactive(y), copy(activecoordinates(y)), copy(y.nzval)), before)
        continue
      end
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
    before = (Matrix(y), nactive(y), copy(activecoordinates(y)), copy(y.nzval))
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
    @test isequal((Matrix(y), nactive(y), copy(activecoordinates(y)), copy(y.nzval)), before)
  end
end

@testset "symmetric group writes" begin
  @testset "$T" for T in (Float32,)
    M = floatmax(T)
    A = T[1 2; 2 3]
    huge = T[0 M; M 0]
    @testset "group penalties" begin
      for groups in ([1:4], [[2, 3], [1], [4]])
        for f in (GroupNormLinf(T(1)/4, groups),)
          check_case(f, A, one(T))
          check_symmetric_steps(f, A, T.(1:length(groups)) ./ T(4))
          check_case(f, A, T(100))
          for γ in (one(T), M)
            check_case(f, huge, γ)
          end
        end
        for f in (GroupNormLinf(zero(T), groups),
                  GroupNormLinf(T(1)/4, groups, zeros(T, length(groups))))
          check_case(f, A, one(T))
        end
      end
      # positions 3, 7 and 9 are ungrouped, including beyond group_of
      groups = [[2, 4], [6, 8], [1, 5]]
      for B in (T[1 2 3; 2 4 5; 3 5 6], fill(M/2, 3, 3))
        for f in (GroupNormLinf(T(1)/16, groups),)
          check_case(f, B, one(T))
        end
      end
      for groups in ([[2], [1, 3, 4]], [[2]], [[3]])
        for f in (GroupNormLinf(T(1)/4, groups),)
          rejects_group_steps(f, A, one(T))
        end
      end
    end
  end
  @testset "radial groups, $T" for T in (Float32,)
    A = T[1 2; 2 3]
    M = floatmax(T)
    for groups in ([1:4], [[2, 3], [1], [4]])
      for f in (GroupSCAD(one(T), groups),)
        for B in (A, fill(M/4, 2, 2))
          check_symmetric_steps(f, B, one(T))
          check_symmetric_steps(f, B, T.(1:length(groups)) ./ T(4))
        end
      end
    end
    for f in (GroupSCAD(one(T), [1:4]),)
      # an unrepresentable norm retains the dense kernel's nonfinite result
      check_symmetric_steps(f, fill(M, 2, 2), one(T); finite = false)
      check_symmetric_steps(f, fill(M, 2, 2), T[1]; finite = false)
    end
    for f in (GroupSCAD(one(T), [[2], [1, 3, 4]]),)
      rejects_group_steps(f, A, one(T))
    end
  end
end

end
