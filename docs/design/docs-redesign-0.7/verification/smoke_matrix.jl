# Automated smoke matrix: every exported function type x representation x operation.
# Each cell: ok / MISMATCH vs dense Float64 path / ERR:<ExceptionType>.
# Usage, from the repository root: julia --project=docs docs/design/docs-redesign-0.7/verification/smoke_matrix.jl > smoke.tsv
using ProximalBase, LinearAlgebra, SparseArrays
const PB = ProximalBase

const M4 = [2.0 0.5 0.0 0.3; 0.5 1.5 -0.4 0.0; 0.0 -0.4 1.8 0.2; 0.3 0.0 0.2 1.2]   # symmetric, SPD, has zeros
const V4 = [0.9, -0.4, 0.0, 1.3]
const X6 = [1.0 2.0 0.0 1.0; 3.0 1.0 1.0 0.0; 0.0 2.0 1.0 1.0; 1.0 0.0 2.0 1.0; 0.5 1.0 0.0 2.0; 1.0 1.0 1.0 0.5]
const Y6 = [1.0, 2.0, 3.0, 0.5, 1.5, 2.5]
const Y64 = hcat(Y6, 2Y6, Y6 .- 1, Y6 .+ 0.5)    # 6x4 response for matrix coefficient
const YB6 = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0]       # binary
const YP6 = [0.0, 1.0, 2.0, 0.0, 3.0, 1.0]       # counts
const GV = [[1,2],[3,4]]
const GM = let idx = [(c-1)*4+r for c in 1:4 for r in 1:4]
  g1 = [(c-1)*4+r for c in 1:4 for r in 1:4 if (r<=2)==(c<=2)]
  g2 = [(c-1)*4+r for c in 1:4 for r in 1:4 if (r<=2)!=(c<=2)]
  [g1, g2]
end

cv(T, a) = T.(a)
groups(shape) = shape === :vec ? GV : GM

# factories: name => (T, shape) -> function
const FACT = Pair{String,Function}[
 "ZeroFunction" => (T,s)->ZeroFunction(),
 "LinearFunction" => (T,s)->LinearFunction(s===:vec ? cv(T,[1.0,-2.0,0.5,1.0]) : cv(T,M4)),
 "SqrNormL2" => (T,s)->SqrNormL2(T(0.7)),
 "SqrNormL2(center)" => (T,s)->SqrNormL2(T(0.7), s===:vec ? cv(T,[0.1,0.2,0.3,0.4]) : cv(T,M4)),
 "QuadraticFunction" => (T,s)->QuadraticFunction(cv(T,M4), cv(T,V4)),
 "KronQuadratic" => (T,s)->KronQuadratic(cv(T,M4), cv(T,M4 + 0.5I)),
 "LeastSquaresLoss" => (T,s)->(s===:vec ? LeastSquaresLoss(cv(T,Y6), cv(T,X6)) : LeastSquaresLoss(cv(T,Y64), cv(T,X6))),
 "SqrtLeastSquaresLoss" => (T,s)->(s===:vec ? SqrtLeastSquaresLoss(cv(T,Y6), cv(T,X6)) : SqrtLeastSquaresLoss(cv(T,Y64), cv(T,X6))),
 "LogDetLoss" => (T,s)->LogDetLoss(cv(T,M4)),
 "LogisticLoss" => (T,s)->LogisticLoss(cv(T,YB6), cv(T,X6)),
 "PoissonLoss" => (T,s)->PoissonLoss(cv(T,YP6), cv(T,X6)),
 "NormL0" => (T,s)->NormL0(T(0.3)),
 "NormL0(diagonal=false)" => (T,s)->NormL0(T(0.3); diagonal=false),
 "NormL1" => (T,s)->NormL1(T(0.3)),
 "NormL1(weights)" => (T,s)->NormL1(T(0.3), s===:vec ? cv(T,[1.0,2.0,0.5,1.5]) : cv(T,[1.0 2.0 0.5 1.5; 2.0 1.0 0.7 0.9; 0.5 0.7 1.0 0.3; 1.5 0.9 0.3 1.0])),
 "NormL1Nonneg" => (T,s)->NormL1Nonneg(T(0.3)),
 "ElasticNet" => (T,s)->ElasticNet(T(0.3), T(0.2)),
 "SCAD" => (T,s)->SCAD(T(0.3), T(3.7)),
 "MCP" => (T,s)->MCP(T(0.3), T(3.0)),
 "IndNonnegative" => (T,s)->IndNonnegative(),
 "IndBox" => (T,s)->IndBox(T(-1), T(1)),
 "IndBallLinf" => (T,s)->IndBallLinf(T(1)),
 "IndBallL0" => (T,s)->IndBallL0(2),
 "IndBallL1" => (T,s)->IndBallL1(T(1)),
 "IndBallL2" => (T,s)->IndBallL2(T(1)),
 "NormL2" => (T,s)->NormL2(T(0.3)),
 "NormLinf" => (T,s)->NormLinf(T(0.3)),
 "GroupNormL2" => (T,s)->GroupNormL2(T(0.3), groups(s)),
 "GroupNormLinf" => (T,s)->GroupNormLinf(T(0.3), groups(s)),
 "SparseGroupNormL2" => (T,s)->SparseGroupNormL2(T(0.2), T(0.3), groups(s)),
 "GroupMCP" => (T,s)->GroupMCP(T(0.3), groups(s); a=T(3)),
 "GroupSCAD" => (T,s)->GroupSCAD(T(0.3), groups(s); a=T(3.7)),
 "Scaled(NormL1)" => (T,s)->Scaled(NormL1(T(0.3)), T(2)),
 "Scaled(LeastSquaresLoss)" => (T,s)->Scaled(s===:vec ? LeastSquaresLoss(cv(T,Y6), cv(T,X6)) : LeastSquaresLoss(cv(T,Y64), cv(T,X6)), T(0.5)),
 "Scaled(GroupNormL2)" => (T,s)->Scaled(GroupNormL2(T(0.3), groups(s)), T(2)),
]

# representations: name => (shape, T of x, T of function, builder -> (x, dense))
function sparse_from(T, d)
  x = d isa AbstractVector ? SparseIterate(T, length(d)) : SparseIterate(T, size(d)...)
  for i in eachindex(d); iszero(d[i]) || (x[i] = d[i]); end
  x
end
const REPS = Pair{String,NamedTuple}[
 "dense vector"          => (shape=:vec, TF=Float64, mk=()->(let d=copy(V4); (d, d) end)),
 "dense matrix"          => (shape=:mat, TF=Float64, mk=()->(let d=copy(M4); (d, d) end)),
 "Symmetric dense"       => (shape=:mat, TF=Float64, mk=()->(Symmetric(copy(M4)), copy(M4))),
 "SparseIterate vector"  => (shape=:vec, TF=Float64, mk=()->(sparse_from(Float64, V4), copy(V4))),
 "SparseIterate matrix"  => (shape=:mat, TF=Float64, mk=()->(sparse_from(Float64, M4), copy(M4))),
 "SymmetricSparseIterate"=> (shape=:mat, TF=Float64, mk=()->(SymmetricSparseIterate(copy(M4)), copy(M4))),
 "BlockIterate"          => (shape=:vec, TF=Float64, mk=()->(BlockIterate(copy(V4), GV), copy(V4))),
 "strided view vector"   => (shape=:vec, TF=Float64, mk=()->(let b=zeros(8); b[1:2:7].=V4; (view(b,1:2:7), copy(V4)) end)),
 "strided view matrix"   => (shape=:mat, TF=Float64, mk=()->(let b=zeros(6,6); b[2:5,2:5].=M4; (view(b,2:5,2:5), copy(M4)) end)),
 "Float32 vector (Float32 fn)" => (shape=:vec, TF=Float32, mk=()->(Float32.(V4), copy(V4))),
 "Float32 vector (Float64 fn)" => (shape=:vec, TF=Float64, mk=()->(Float32.(V4), copy(V4))),
 "Float32 matrix (Float32 fn)" => (shape=:mat, TF=Float32, mk=()->(Float32.(M4), copy(M4))),
 "BigFloat vector (BigFloat fn)" => (shape=:vec, TF=BigFloat, mk=()->(BigFloat.(V4), copy(V4))),
 "BigFloat vector (Float64 fn)"  => (shape=:vec, TF=Float64, mk=()->(BigFloat.(V4), copy(V4))),
 "Int vector (Float64 fn)" => (shape=:vec, TF=Float64, mk=()->([1,-2,0,3], Float64[1,-2,0,3])),
]

todense(x::SymmetricSparseIterate) = Matrix(matrixview(x))
todense(x::SparseIterate) = Array(x)
todense(x::BlockIterate) = collect(x)
todense(x::Symmetric) = Matrix(x)
todense(x::AbstractArray) = Array(x)
todense(x::Number) = x

function cmp(a, b)
  if a isa Number && b isa Number
    (isnan(a) && isnan(b)) && return "ok"
    (a == b) && return "ok"
    err = abs(Float64(a) - Float64(b)) / max(abs(Float64(b)), 1e-12)
    return err <= 1e-4 ? "ok" : "MISMATCH($(round(err; sigdigits=2)))"
  end
  try
    A = todense(a); B = todense(b)
    size(A) == size(B) || return "MISMATCH(size $(size(A)) vs $(size(B)))"
    any(isnan, A) && any(isnan, B) && return "ok"
    err = maximum(abs.(Float64.(A) .- Float64.(B)); init=0.0) / max(maximum(abs.(Float64.(B)); init=0.0), 1e-12)
    err <= 1e-4 ? "ok" : "MISMATCH($(round(err; sigdigits=2)))"
  catch e
    "cmp-error:$(typeof(e))"
  end
end

errname(e) = begin
  s = e isa MethodError ? "MethodError" : string(typeof(e))
  msg = e isa MethodError ? "" : replace(first(split(sprint(showerror, e), '\n')), '\t'=>' ')
  "ERR:" * s * (isempty(msg) ? "" : ":" * first(msg, 90))
end

function run(thunk, ref=nothing)
  try
    r = thunk()
    return ref === nothing ? "ok" : cmp(r, ref())
  catch e
    return errname(e)
  end
end

rows = Tuple{String,String,String,String}[]
emit(f, r, op, st) = push!(rows, (f, r, op, st))

γ = 0.3
for (fname, mkf) in FACT
  (!isempty(ARGS) && !(fname in ARGS)) && continue
  for (rname, rep) in REPS
    # reference function/data in Float64 dense
    fref = try mkf(Float64, rep.shape) catch e; nothing end
    f = try mkf(rep.TF, rep.shape) catch e
      emit(fname, rname, "construct", errname(e)); continue
    end
    x, xd = rep.mk()
    # the dense reference uses a copy of the dense data
    ref(op) = () -> op(fref, copy(xd))
    ismat = rep.shape === :mat
    # value
    emit(fname, rname, "value f(x)", run(() -> f(x), fref === nothing ? nothing : () -> fref(copy(xd))))
    # gradient (allocating)
    emit(fname, rname, "gradient", run(() -> gradient(f, x), fref === nothing ? nothing : () -> gradient(fref, copy(xd))))
    emit(fname, rname, "gradient!", run(() -> (g = PB._gradient_buffer(x); gradient!(g, f, x)), fref === nothing ? nothing : () -> gradient(fref, copy(xd))))
    emit(fname, rname, "value_and_gradient", run(() -> value_and_gradient(f, x)[2], fref === nothing ? nothing : () -> gradient(fref, copy(xd))))
    emit(fname, rname, "value_and_gradient!", run(() -> (g = PB._gradient_buffer(x); value_and_gradient!(g, f, x)), fref === nothing ? nothing : () -> value_and_gradient!(PB._gradient_buffer(copy(xd)), fref, copy(xd))))
    emit(fname, rname, "prox", run(() -> prox(f, x, γ), fref === nothing ? nothing : () -> prox(fref, copy(xd), γ)))
    # prox! with a dense output
    dout() = (x isa SymmetricSparseIterate ? zeros(size(xd)) : zeros(float(eltype(x)), size(x)))
    emit(fname, rname, "prox! (dense out)", run(() -> prox!(dout(), f, x, γ), fref === nothing ? nothing : () -> prox(fref, copy(xd), γ)))
    # prox! with an output of the iterate's own kind
    sout() = (x isa Union{SparseIterate, SymmetricSparseIterate, BlockIterate} ? similar(x) : similar(x, float(eltype(x))))
    emit(fname, rname, "prox! (same-kind out)", run(() -> prox!(sout(), f, x, γ), fref === nothing ? nothing : () -> prox(fref, copy(xd), γ)))
    # value_and_prox!: return is f(out) scalar
    emit(fname, rname, "value_and_prox! (dense out)", run(() -> (o = dout(); v = value_and_prox!(o, f, x, γ); v), fref === nothing ? nothing : () -> (p = prox(fref, copy(xd), γ); fref(p))))
    emit(fname, rname, "value_and_prox! (same-kind out)", run(() -> (o = sout(); v = value_and_prox!(o, f, x, γ); v), fref === nothing ? nothing : () -> (p = prox(fref, copy(xd), γ); fref(p))))
    # in-place prox!(x, f, x, γ) on a copy
    emit(fname, rname, "prox! in place (y===x)", run(() -> (c = x isa Symmetric ? Matrix(x) : (x isa Union{SparseIterate,SymmetricSparseIterate,BlockIterate} ? copy(x) : copy(x)); eltype(c) <: Integer && error("int"); prox!(c, f, c, γ)), fref === nothing ? nothing : () -> prox(fref, copy(xd), γ)))
    # kkt
    v = rep.shape === :vec ? [0.1,-0.2,0.3,0.05] : (0.1 .* reshape(1:16, 4, 4)) ./ 4 |> a -> (a .+ a')./2
    vv = rep.TF === Float32 ? Float32.(v) : v
    emit(fname, rname, "kkt_violation(f,v,x)", run(() -> kkt_violation(f, vv, x), fref === nothing ? nothing : () -> kkt_violation(fref, v, copy(xd))))
    emit(fname, rname, "kkt_violation(f,v,x,k)", run(() -> kkt_violation(f, vv, x, 2), fref === nothing ? nothing : () -> kkt_violation(fref, v, copy(xd), 2)))
    emit(fname, rname, "kkt_violation(f,v,i)", run(() -> kkt_violation(f, vv, 2)))
  end
end
isempty(ARGS) && println("function\trepresentation\toperation\tstatus")
for r in rows; println(join(r, '\t')); end
