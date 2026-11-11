module TestUtils
using Test, ProximalBase
export allocs, allocations, test_iterate
# Explicit arities preserve specialization for allocation regression checks.
allocs(f::F, a::A) where {F,A} = (@allocated f(a); @allocated f(a))
allocs(f::F, a::A, b::B) where {F,A,B} = (@allocated f(a,b); @allocated f(a,b))
allocs(f::F, a::A, b::B, c::C) where {F,A,B,C} = (@allocated f(a,b,c); @allocated f(a,b,c))
allocs(f::F, a::A, b::B, c::C, d::D) where {F,A,B,C,D} = (@allocated f(a,b,c,d); @allocated f(a,b,c,d))
allocs(f::F, a::A, b::B, c::C, d::D, e::E) where {F,A,B,C,D,E} = (@allocated f(a,b,c,d,e); @allocated f(a,b,c,d,e))
const allocations=allocs

function test_iterate(x::BlockIterate)
  ok = length(x.group_of) == length(x.storage)
  @test length(x.group_of) == length(x.storage)
  ok || return x
  ok = length(x.position) == length(x.groups) == length(x.active)
  @test length(x.position) == length(x.groups) == length(x.active)
  ok || return x
  ok = 0 <= x.nactive <= length(x.groups)
  @test 0 <= x.nactive <= length(x.groups)
  ok || return x
  bad = [(k, i) for (k, G) in enumerate(x.groups) for i in G
         if !(1 <= i <= length(x.storage))]
  @test isempty(bad)
  isempty(bad) || return x
  memberships = zeros(Int, length(x.storage))
  for G in x.groups, i in G
    memberships[i] += 1
  end
  bad = (missing = findall(iszero, memberships), repeated = findall(>(1), memberships))
  @test bad == (missing = Int[], repeated = Int[])
  isempty(bad.missing) && isempty(bad.repeated) || return x
  bad = [(k, i) for (k, G) in enumerate(x.groups) for i in G if x.group_of[i] != k]
  @test isempty(bad)
  active = falses(length(x.groups))
  duplicates, out_of_range = Int[], Int[]
  for a in 1:x.nactive
    k = x.active[a]
    if 1 <= k <= length(x.groups)
      active[k] && push!(duplicates, a)
      active[k] = true
    else
      push!(out_of_range, a)
    end
  end
  @test isempty(duplicates)
  @test isempty(out_of_range)
  isempty(out_of_range) || return x
  bad = [a for a in 1:x.nactive if x.position[x.active[a]] != a]
  @test isempty(bad)
  bad = [k for k in eachindex(x.groups) if !active[k] && x.position[k] != 0]
  @test isempty(bad)
  bad = [i for k in eachindex(x.groups) if !active[k] for i in x.groups[k]
         if !iszero(x.storage[i])]
  @test isempty(bad)
  x
end

function test_iterate(x::Union{SparseIterate, SymmetricSparseIterate})
  ok = length(x.nzval) == length(x.nzval2ind) == length(x.ind2nzval)
  @test length(x.nzval) == length(x.nzval2ind) == length(x.ind2nzval)
  ok || return x
  if x isa SymmetricSparseIterate
    ok = length(x.ind2nzval) == x.p * (x.p + 1) ÷ 2
    @test length(x.ind2nzval) == x.p * (x.p + 1) ÷ 2
    ok || return x
  end
  ok = 0 <= x.nnz <= length(x.ind2nzval)
  @test 0 <= x.nnz <= length(x.ind2nzval)
  ok || return x
  stored = falses(length(x.ind2nzval))
  duplicates, out_of_range = Int[], Int[]
  for j in 1:x.nnz
    i = x.nzval2ind[j]
    if 1 <= i <= length(x.ind2nzval)
      stored[i] && push!(duplicates, j)
      stored[i] = true
    else
      push!(out_of_range, j)
    end
  end
  @test isempty(duplicates)
  @test isempty(out_of_range)
  isempty(out_of_range) || return x
  bad = [j for j in 1:x.nnz if x.ind2nzval[x.nzval2ind[j]] != j]
  @test isempty(bad)
  bad = [i for i in eachindex(x.ind2nzval) if !(0 <= x.ind2nzval[i] <= x.nnz)]
  @test isempty(bad)
  bad = [i for i in eachindex(x.ind2nzval) if !stored[i] && x.ind2nzval[i] != 0]
  @test isempty(bad)
  x
end

end
