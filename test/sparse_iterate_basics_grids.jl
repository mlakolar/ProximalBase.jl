module SparseIterateBasicsGridsTest

# Cases of sparse_iterate_basics.jl that repeat one that stays there for another element type: the
# Float16 pass and the left operations at Float32 of the scaling by a number (FN-9), and the Float32
# pass of the sparse triple products. The rest of each testset is in sparse_iterate_basics.jl; the
# helpers are copied from it.

using Test
using ProximalBase
using Random
using LinearAlgebra
using SparseArrays

# Keep each operation and its broadcast together so the reference preserves operand order.
scale_right!(x, s) = rmul!(x, s)
scale_left!(x, s) = lmul!(s, x)
divide_right!(x, s) = rdiv!(x, s)
divide_left!(x, s) = ldiv!(s, x)
broadcast_scale_right!(x, s) = (x .*= s)
broadcast_scale_left!(x, s) = (x .= s .* x)
broadcast_divide_right!(x, s) = (x ./= s)
broadcast_divide_left!(x, s) = (x .= s .\ x)

function check_scaling(method!, reference!, fixture, s)
    x, expected = copy(fixture), copy(fixture)
    reference!(expected, s)
    @test method!(x, s) === x
    @test isequal(collect(activevalues(x)), collect(activevalues(expected)))
    @test isequal(Array(x), Array(expected))
    @test collect(activecoordinates(x)) == collect(activecoordinates(expected))
end

function scaling_fixtures(T)
    unsorted = SparseIterate(T, 6)
    unsorted[3] = 1; unsorted[1] = -2; activate!(unsorted, 4)
    special = SparseIterate(T, 6)
    special[3] = NaN; special[1] = Inf; special[5] = -Inf
    activate!(special, 2); special[2] = -0.0
    full = SparseIterate(T, 5)
    for i in 1:5
        activate!(full, i)
        full[i] = i - 3
    end
    matrix = SparseIterate(T, 2, 3)
    matrix[2, 2] = 1; matrix[1, 1] = -2; activate!(matrix, 3)
    (SparseIterate(T, 0), SparseIterate(T, 5), unsorted, special, full, matrix)
end

@testset "sparseIterate" begin

    @testset "scaling by a number follows the broadcast (FN-9)" begin
        multipliers = (2, -3.0, 0, -0.0, 0.5, 1e300, 1e-300, Inf, -Inf, NaN, 1//2)
        divisors = (2, -0.0, 0.0, Inf, -Inf, NaN)
        operations = ((scale_right!, broadcast_scale_right!, multipliers),
                      (scale_left!, broadcast_scale_left!, multipliers),
                      (divide_right!, broadcast_divide_right!, divisors),
                      (divide_left!, broadcast_divide_left!, divisors))
        # the right operations at Float32 are in sparse_iterate_basics.jl
        for T in (Float16, Float32)
            left_operations = (operations[2], operations[4])
            for fixture in scaling_fixtures(T), (method!, reference!, scalars) in (T === Float16 ? operations : left_operations)
                for s in scalars
                    check_scaling(method!, reference!, fixture, s)
                end
            end
        end
    end

end

@testset "sparse triple products agree with dense products" begin
    rng = Xoshiro(20261025)
    for T in (Float32,)
        m, n = 8, 10
        A = Symmetric(randn(rng, T, m, m))
        B = Symmetric(randn(rng, T, n, n))
        # Concentrated support exercises each multiplication order, and no
        # support verifies the empty-product dimensions. Stored zeros must
        # contribute neither a row nor a column to the restricted product.
        for (rows, columns) in (([2], [3, 7]), ([2, 6], [3]), (Int[], Int[]))
            dense = zeros(T, m, n)
            for i in rows, j in columns
                dense[i, j] = T(i - j)
            end
            sparse = SparseIterate(dense)
            activate!(sparse, 1)
            product = A * sparse * B
            @test product ≈ A * dense * B
            @test size(product) == (m, n)
            @test eltype(product) == T
            @test Matrix(sparse) == dense
        end
    end
end

end
