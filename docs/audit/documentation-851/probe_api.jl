# Record live binding/method evidence without treating broad fallback signatures
# as proof that every concrete operand combination is supported.
using ProximalBase, LinearAlgebra, InteractiveUtils
versioninfo()
println("\nPublic bindings and defining methods (PB methods only):")
for name in names(ProximalBase)
    value = getproperty(ProximalBase, name)
    exported = Base.isexported(ProximalBase, name)
    println("\n", name, exported ? " [exported]" : " [qualified public/module]")
    value isa Module && continue
    for method in methods(value)
        method.module === ProximalBase || continue
        println(method)
    end
end
println("\nConcrete capability boundaries:")
x = ones(2)
for f in (ZeroFunction(), LinearFunction(x), SqrNormL2(), NormL1(),
          QuadraticFunction(Matrix{Float64}(I,2,2)), LeastSquaresLoss(x,Matrix{Float64}(I,2,2)))
    println(nameof(typeof(f)), ": scalar block=", applicable(prox_block,f,1.0,1,1.0),
            "; value_entry=", applicable(value_entry,f,1.0,1),
            "; traits=", (issmooth(f),isproximable(f),isseparable(f),isconvex(f),iszeropreserving(f)))
end
println("\nReported separately as #853:")
for level in (0.0,1.0,Inf), trial in (2.0,Inf,NaN)
    println("level=",level," weight=Inf trial=",trial," prox=",
            prox(NormL1(level,[Inf]),[trial],1.0))
end
