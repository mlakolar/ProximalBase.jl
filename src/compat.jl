# Keep dependencies on non-public stdlib bindings in this file.

isdefined(LinearAlgebra, :BlasInt) || error("LinearAlgebra.BlasInt is required for the BLAS integer ABI")
const _BlasInt = LinearAlgebra.BlasInt
const _libblastrampoline = LinearAlgebra.libblastrampoline
_blasfunc(name::Symbol)::Symbol = _BlasInt === Int64 ? Symbol(name, "64_") : name
const _potrf! = LAPACK.potrf!
const _potri! = LAPACK.potri!

@static if isdefined(SparseArrays, :AbstractSparseMatrixCSC)
  const _AbstractSparseMatrixCSC = SparseArrays.AbstractSparseMatrixCSC
end
@static if isdefined(SparseArrays, :MatrixWrappersOrView)
  const _SparseDotMatrix = Union{SparseArrays.DenseMatrixUnion,
    SparseArrays.MatrixWrappersOrView{<:Any, <:Union{SparseArrays.DenseMatrixUnion, AbstractSparseMatrix}}}
elseif isdefined(SparseArrays, :WrapperMatrixTypes)
  const _SparseDotMatrix = Union{SparseArrays.DenseMatrixUnion,
    SparseArrays.WrapperMatrixTypes{<:Any, <:Union{SparseArrays.DenseMatrixUnion, AbstractSparseMatrix}}}
else
  const _SparseDotMatrix = StridedMatrix
end
@static if isdefined(SparseArrays, :SparseMatrixCSCOrColumnSubset)
  const _SparseDotCSC = SparseArrays.SparseMatrixCSCOrColumnSubset
elseif isdefined(SparseArrays, :AbstractSparseMatrixCSC)
  const _SparseDotCSC = SparseArrays.AbstractSparseMatrixCSC
else
  const _SparseDotCSC = SparseMatrixCSC
end
# Mixed sparse wrappers check shapes only when SparseArrays specializes dot.
@static if isdefined(SparseArrays, :SparseMatrixCSCOrColumnSubset)
  const _SparseMixedDotChecksSize =
    which(dot, Tuple{Adjoint{Float64, SparseMatrixCSC{Float64, Int}},
                     Transpose{Float64, SparseMatrixCSC{Float64, Int}}}).module === SparseArrays &&
    which(dot, Tuple{Transpose{Float64, SparseMatrixCSC{Float64, Int}},
                     Adjoint{Float64, SparseMatrixCSC{Float64, Int}}}).module === SparseArrays
else
  const _SparseMixedDotChecksSize = false
end
const _SparseDotWrappedCSC = Union{Adjoint{<:Any, <:_SparseDotCSC}, Transpose{<:Any, <:_SparseDotCSC}}
# Column-subset parents gained the transposed sparse dot family later than CSC.
@static if isdefined(SparseArrays, :SparseMatrixCSCColumnSubset)
  const _SparseWrappedColumnDotChecksSize =
    which(dot, Tuple{Adjoint{Float64, SparseArrays.SparseMatrixCSCColumnSubset{Float64, Int}},
                     SparseMatrixCSC{Float64, Int}}).module === SparseArrays &&
    which(dot, Tuple{SparseMatrixCSC{Float64, Int},
                     Adjoint{Float64, SparseArrays.SparseMatrixCSCColumnSubset{Float64, Int}}}).module === SparseArrays
else
  const _SparseWrappedColumnDotChecksSize = false
end
@static if isdefined(SparseArrays, :SparseMatrixCSCOrView)
  const _SparseMatrixCSCOrView = SparseArrays.SparseMatrixCSCOrView
end
@static if isdefined(SparseArrays, :AdjOrTransSparseVectorOrView)
  const _AdjOrTransSparseVectorOrView = SparseArrays.AdjOrTransSparseVectorOrView
end
@static if isdefined(SparseArrays, :CHOLMOD)
  @static if isdefined(SparseArrays.CHOLMOD, :Dense)
    const _CHOLMODDense = SparseArrays.CHOLMOD.Dense
  end
  @static if isdefined(SparseArrays.CHOLMOD, :VTypes)
    const _CHOLMODVTypes = SparseArrays.CHOLMOD.VTypes
  end
end

# Orthogonal factors and rotations have specialized in-place matrix writers.
@static if isdefined(LinearAlgebra, :AbstractQ)
  const _AbstractQ = LinearAlgebra.AbstractQ
end
@static if isdefined(LinearAlgebra, :AbstractRotation)
  const _AbstractRotation = LinearAlgebra.AbstractRotation
end
@static if isdefined(Base, :AbstractBroadcasted)
  const _AbstractBroadcasted = Base.AbstractBroadcasted
end
@static if isdefined(Base, :RangeVecIntList)
  const _RangeVecIntList = Base.RangeVecIntList
end
@static if isdefined(LinearAlgebra, :QRPackedQ)
  const _QRPackedQ = LinearAlgebra.QRPackedQ
end
@static if isdefined(LinearAlgebra, :AdjointQ)
  const _AdjointQ = LinearAlgebra.AdjointQ
end
@static if isdefined(LinearAlgebra, :Rotation)
  const _Rotation = LinearAlgebra.Rotation
end
@static if isdefined(LinearAlgebra, :AdjointRotation)
  const _AdjointRotation = LinearAlgebra.AdjointRotation
end
@static if isdefined(LinearAlgebra, :QRCompactWY)
  const _QRCompactWY = LinearAlgebra.QRCompactWY
end
@static if isdefined(LinearAlgebra, :AdjointFactorization)
  const _AdjointFactorization = LinearAlgebra.AdjointFactorization
end
@static if isdefined(LinearAlgebra, :TransposeFactorization)
  const _TransposeFactorization = LinearAlgebra.TransposeFactorization
end

# Range copies and band fills are not public on every supported Julia version.
@static if isdefined(LinearAlgebra, :copy_transpose!)
  const _copy_transpose! = LinearAlgebra.copy_transpose!
end
@static if isdefined(LinearAlgebra, :copy_adjoint!)
  const _copy_adjoint! = LinearAlgebra.copy_adjoint!
end
@static if isdefined(LinearAlgebra, :fillband!)
  const _fillband! = LinearAlgebra.fillband!
end
