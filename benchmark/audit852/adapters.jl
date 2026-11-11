# Benchmark-only API adapters. No source patching, fallback kernels or added package methods.
const ORIGINAL = !isdefined(PB, :ncoordinates)
const OWNER = isdefined(PB, :coordinates) && !(PB.SymmetricSparseIterate <: AbstractArray)
const BT = isdefined(PB,:BlockIterate) ? PB.BlockIterate : PB.AtomIterate
const IS_SYM = PB.SymmetricSparseIterate
@inline coord(x::PB.SparseIterate, k, rows, cols) = x[k]
@inline coordset!(x::PB.SparseIterate,v,k,rows,cols) = (x[k]=v)
@static if ORIGINAL
    @inline coord(x::IS_SYM,k,rows,cols)=x[k]
    @inline coordset!(x::IS_SYM,v,k,rows,cols)=(x[k]=v)
else
    @inline coord(x::IS_SYM,k,rows,cols)=PB.coordinates(x)[k]
    @inline coordset!(x::IS_SYM,v,k,rows,cols)=(PB.coordinates(x)[k]=v)
end
@static if ORIGINAL
    block(x,k)=x.atoms[k]
    ids(x)=view(x.nzval2ind,1:x.nnz)
    clear!(x::PB.SparseIterate)=fill!(x,zero(eltype(x)))
    function clear!(x::IS_SYM)
        # Original linear indices denote independent variables; never use its
        # inconsistent generic AbstractMatrix iteration for a whole-state write.
        for k in eachindex(x.ind2nzval);x[k]=zero(eltype(x));end
        x
    end
    clear!(x::BT)=fill!(x,zero(eltype(x)))
    compact!(x)=SparseArrays.dropzeros!(x)
else
    block(x,k)=isdefined(PB,:blockview) ? PB.blockview(x,k) : PB.atom(x,k)
    ids(x)=PB.activecoordinates(x)
    clear!(x::PB.SparseIterate)=fill!(x,zero(eltype(x)))
    clear!(x::IS_SYM)=OWNER ? fill!(PB.coordinates(x),zero(eltype(x))) : fill!(x,zero(eltype(x)))
    clear!(x::BT)=fill!(x,zero(eltype(x)))
    compact!(x)=PB.dropzeros!(x)
end
newblock(T,n,b,groups)=ORIGINAL ? BT(n,n÷b) : BT(T,n,groups)
newscalar(fam,T,n)=fam=="sparse" ? PB.SparseIterate(T,n) : IS_SYM(T,n)
