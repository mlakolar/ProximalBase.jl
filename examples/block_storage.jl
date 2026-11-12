# Ordered grouped storage
#
# *Assumes: vector indexing and grouped variables · You will learn: ordered block storage and compact block updates.*
using ProximalBase, LinearAlgebra
input_groups = [[3, 1], [4, 2]]
x = BlockIterate(Float32, 4, input_groups)
input_groups[1][1] = 2  # constructor owns a copy of vector groups
@assert collect(blockindices(x, 1)) == [3, 1]
copyto!(blockview(x, 1), Float32[5, 0])
@assert Vector(x) == Float32[0, 0, 5, 0]
@assert ncoordinates(x) == 2 && nactive(x) == 1
activate!(x, 2)
copyto!(blockview(x, 1), zeros(Float32, 2))
@assert nactive(x) == 2  # all-zero active groups are deliberate state
z = copy(x)
dropzeros!(x)
@assert nactive(x) == 0 && nactive(z) == 2
@assert ProximalBase.samegroups(x, z)
@assert !ProximalBase.samegroups(x, BlockIterate(4, [[1, 3], [4, 2]]))
# NormL1 is entry-separable; its index is a global entry, not a group ID.
g = NormL1(0.5, Float32[1, 2, 3, 4])
u = Float32[3, -2]
ids = blockindices(x, 1)
out = [prox_block(g, u[a], ids[a], 1f0) for a in eachindex(u)]
@assert out ≈ Float32[1.5, -1.5]
copyto!(blockview(x, 1), out)
@assert collect(activecoordinates(x)) == [1]
@assert kkt_violation(g, Float32[1, 2, 3, 4], x, 1) ≈ 1.5f0
A = Float32[1 2 3 4; 4 3 2 1]
@assert A * x ≈ A * Vector(x)
Vector(x)
