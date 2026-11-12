# Sparse working sets
#
# *Assumes: arrays and coordinate descent · You will learn: active stored zeros, screening, restriction and compaction.*
# Here the term [active coordinates](../glossary.md#Active-coordinate).
using ProximalBase, LinearAlgebra
x = SparseIterate(5)
activate!(x, 4)
x[2] = 3
@assert collect(activecoordinates(x)) == [4, 2]
@assert collect(activevalues(x)) == [0, 3]
x[2] = prox_block(NormL1(4.0), x[2], 2, 1.0)
@assert nactive(x) == 2 && all(iszero, x)
# State copy retains explicit zeros; value writes keep the destination working set.
saved = copy(x)
y = SparseIterate(5)
y[1] = 2
copyto!(y, saved)
@assert collect(activecoordinates(y)) == [4, 2]
y .= [0, 0, 1, 0, 0]
@assert nactive(y) == 3
@assert collect(activecoordinates(saved)) == [4, 2]
dropzeros!(y)
@assert collect(activecoordinates(y)) == [3]
# ZeroFunction is a copying exception: matching array-iterate prox replaces state.
identity_src = SparseIterate(3)
activate!(identity_src, 2)
identity_dest = SparseIterate(3)
activate!(identity_dest, 1)
prox!(identity_dest, ZeroFunction(), identity_src)
@assert collect(activecoordinates(identity_dest)) == [2]
# A full-gradient screen selects violating zero coordinates for a working set.
penalty = NormL1(0.5)
grad = [0.2, -0.9, 0.0, 0.8, 0.1]
for j in 1:ncoordinates(x)
    kkt_violation(penalty, grad, x, j) > 0 && activate!(x, j)
end
@assert isactive(x, 2) && isactive(x, 4)
# Restriction slices entry parameters; the caller maps local indices back to S.
S = [4, 2]
w = [1.0, 2, 3, 4, 5]
h = restrict(NormL1(0.5, w), S)
@assert h([2.0, -1]) ≈ 0.5 * (4 * 2 + 2 * 1)
# A centered ridge restriction drops an additive constant from omitted zeros.
ridge = SqrNormL2(2.0, [1.0, 3])
restricted_ridge = restrict(ridge, [1])
@assert ridge([2.0, 0]) == restricted_ridge([2.0]) + 9
# Active IDs are unsorted handles. Take snapshots if they must outlive a change.
(ids = collect(activecoordinates(x)), values = collect(activevalues(x)))
