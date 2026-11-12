
# Names that only some SparseArrays versions or builds define; the copyto! methods that
# need them are defined only where they exist.

# SparseArrays 1.14 adds copyto!(dest::AbstractMatrix, src) for these sources, which is
# ambiguous with copyto!(x::SparseIterate, y::AbstractArray); Julia 1.10 to 1.13 lack them.
@static if isdefined(SparseArrays, :SparseMatrixCSCOrView)
  const _SparseMatrixCSCOrView = SparseArrays.SparseMatrixCSCOrView
end
@static if isdefined(SparseArrays, :AdjOrTransSparseVectorOrView)
  const _AdjOrTransSparseVectorOrView = SparseArrays.AdjOrTransSparseVectorOrView
end
# CHOLMOD is part of SparseArrays only in a Julia built with its GPL libraries
# (Base.USE_GPL_LIBS); its copyto!(dest::AbstractArray, D::Dense) is ambiguous with the
# copyto! of SparseIterate and BlockIterate from an AbstractArray.
@static if isdefined(SparseArrays, :CHOLMOD) && isdefined(SparseArrays.CHOLMOD, :Dense)
  const _CHOLMODDense = SparseArrays.CHOLMOD.Dense
end
