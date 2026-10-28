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
const _SparseDotMatrix = Union{SparseArrays.DenseMatrixUnion,
  SparseArrays.WrapperMatrixTypes{<:Any, <:Union{SparseArrays.DenseMatrixUnion, AbstractSparseMatrix}}}
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
