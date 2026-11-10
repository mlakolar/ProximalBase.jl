# Run against the pinned pre-844 and migrated PB/PS environments. Single Julia
# and BLAS thread; TSV output records medians and allocations after warm-up.
using ProximalBase, ProximalSolvers, BenchmarkTools, LinearAlgebra, Random, Statistics, SparseArrays
const PB = ProximalBase
const modern = isdefined(PB, :prox_block!)
const scalar_prox = getfield(PB, modern ? :prox_block : :prox_entry)
const retired_step! = modern ? nothing : getfield(PB, :prox_coordinate!)
BLAS.set_num_threads(1)

function measure(label, op; work=1, samples=300, seconds=.2)
    op(); op()
    b = @benchmarkable $op()
    trial = run(b; evals=1, samples, seconds)
    println(label, '\t', median(trial).time/work, '\t', trial.memory, '\t', trial.allocs)
    flush(stdout)
end
function scalar_sweep(f, ::Type{T}) where T
    acc = zero(T)
    for i in 1:10_000
        acc += scalar_prox(f,T(mod(i,7)-3),1,T(.7))
    end
    acc
end
function scalar_case(::Type{T}) where T
    f = NormL1(T(.3))
    measure("scalar/$T",()->scalar_sweep(f,T);work=10_000)
end
function group_case(::Type{T}, n, zero_trial) where T
    gs = [collect(1:n),collect(n+1:2n)]
    u = zero_trial ? fill(T(.01),n) : T[sin(i)+2 for i in 1:n]
    for f in (GroupNormL2(T(.3),gs,ones(T,2)), GroupNormLinf(T(.3),gs,ones(T,2)),
              SparseGroupNormL2(T(.1),T(.3),gs,ones(T,2)))
        x, out = AtomIterate(T,2n,gs), similar(u)
        combined = if modern
            () -> (PB.prox_block!(out,f,u,1,T(.7)); copyto!(atom(x,1),out); nothing)
        else
            () -> (retired_step!(x,f,1,u,T(.7)); nothing)
        end
        name = "$T/$(nameof(typeof(f)))/n$n/zero$zero_trial"
        measure("group_write/$name",combined)
        if modern
            measure("pure_group/$name", () -> (PB.prox_block!(out,f,u,1,T(.7)); nothing))
            measure("public_write/$name", () -> (copyto!(atom(x,1),out); nothing))
        end
    end
end
function whole_solves(::Type{T}) where T
    rng = Xoshiro(844)
    n,p = 160,120
    X = randn(rng,T,n,p)
    β = zeros(T,p); β[1:10] .= T(2)
    y = X*β + T(.1)*randn(rng,T,n)
    gs = [i:i+4 for i in 1:5:p]
    f = (one(T)/T(n))*LeastSquaresLoss(y,X)
    q = QuadraticFunction(X'X/T(n),-X'y/T(n))
    for (name,loss,g,x) in (("scalar",f,NormL1(T(.1)),SparseIterate(T,p)),
                           ("group_ls",f,GroupNormL2(T(.1),gs),AtomIterate(T,p,gs)),
                           ("group_quad",q,GroupNormL2(T(.1),gs),AtomIterate(T,p,gs)),
                           ("group_ls_csc",(one(T)/T(n))*LeastSquaresLoss(y,sparse(X)),GroupNormL2(T(.1),gs),AtomIterate(T,p,gs)),
                           ("generic_linf",f,GroupNormLinf(T(.1),gs),AtomIterate(T,p,gs)),
                           ("generic_sparsegroup",f,SparseGroupNormL2(T(.03),T(.1),gs),AtomIterate(T,p,gs)))
        solver = CDSolver(;tol=T===Float32 ? 1e-5 : 1e-10)
        op = () -> (fill!(x,zero(T)); dropzeros!(x); solve!(x,loss,g,solver))
        result=op()
        result.converged || error("nonconverged $name $T")
        measure("solve/$T/$name",op;samples=30,seconds=.4)
    end
    qresponses=4
    B=zeros(T,p,qresponses); B[1:5,:] .= T(2)
    Y=X*B+T(.1)*randn(rng,T,n,qresponses)
    rows=[collect(k:p:p*qresponses) for k in 1:p]
    loss=(one(T)/T(n))*LeastSquaresLoss(Y,X)
    g=GroupNormL2(T(.1),rows,ones(T,p))
    x=AtomIterate(T,p*qresponses,rows)
    solver=CDSolver(;tol=T===Float32 ? 1e-5 : 1e-10)
    op=()->(fill!(x,zero(T));dropzeros!(x);solve!(x,loss,g,solver))
    op().converged || error("nonconverged multiresponse $T")
    measure("solve/$T/multiresponse",op;samples=30,seconds=.4)
end
println("case\tmedian_ns\tbytes\tallocations")
for T in (Float32,Float64)
    scalar_case(T)
    for n in (8,32,256), zero_trial in (false,true)
        group_case(T,n,zero_trial)
    end
    whole_solves(T)
end
