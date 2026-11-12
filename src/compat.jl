
@static if isdefined(SparseArrays, :AbstractSparseMatrixCSC)
  const _AbstractSparseMatrixCSC = SparseArrays.AbstractSparseMatrixCSC
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

# Orthogonal factors have a specialized in-place matrix writer.
@static if isdefined(LinearAlgebra, :AbstractQ)
  const _AbstractQ = LinearAlgebra.AbstractQ
end
