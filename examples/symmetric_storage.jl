# Independent symmetric coordinates
#
# *Assumes: symmetric matrices and norms · You will learn: independent matrix coordinates, views and symmetric operations.*
# See [iterates](../glossary.md#Iterate) for the meaning of the stored optimization variable.
using ProximalBase, LinearAlgebra, SparseArrays
x = SymmetricSparseIterate(3)
x[3, 1] = 2
x[2, 2] = 7
activate!(x, 2)
@assert ncoordinates(x) == 6 && nactive(x) == 3
@assert tril_index(3, 2, 2) == 4
@assert tril_cartesian(3, 3) == (3, 1)
@assert coordinates(x)[4] == 7 && matrixview(x)[4] == 0
@assert x[1, 3] == 2
# Off-diagonal values appear twice in the matrix and once in coordinates.
@assert norm(x)^2 ≈ 7^2 + 2 * 2^2
@assert norm(coordinates(x))^2 ≈ 7^2 + 2^2
@assert NormL1()(x) == 11
A = matrixview(x)
x[1, 1] = 1
@assert A[1, 1] == 1
@assert x * ones(3) ≈ Matrix(x) * ones(3)
@assert Matrix(sparse(A)) == Matrix(x)
# Proximal output reads a nonsymmetric argument through its symmetric part.
out = SymmetricSparseIterate(2)
prox!(out, NormL1(0.5), [0.0 1; 3 0], 1.0)
@assert out[2, 1] == 1.5
W = [1.0 2; 2 1]
@assert ProximalBase.check_symmetric_parameters(NormL1(1.0, W), out)
@assert Matrix(prox(NormL1(1.0, W), out)) == zeros(2, 2)
# A matrix import replaces the working set; a state copy preserves active zeros.
saved = copy(x)
copyto!(x, Matrix(x))
@assert nactive(x) == 3 && nactive(saved) == 4
fill!(coordinates(x), 0)
@assert nactive(x) == 3
dropzeros!(x)
@assert nactive(x) == 0
Matrix(saved)
