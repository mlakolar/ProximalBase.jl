# Compiler audit, not a timing harness. Run with the recorded immutable project.
using ProximalBase, InteractiveUtils, LinearAlgebra
writeprobe!(x,u) = copyto!(blockview(x,1),u)
mathprobe!(out,f,u) = prox_block!(out,f,u,1,eltype(u)(.7))
inplaceprobe!(u,f) = prox_block!(u,f,u,1,eltype(u)(.7))
normprobe(x) = norm(blockview(x,1))
println("Julia=",VERSION," PB=",pathof(ProximalBase))
for T in (Float32,Float64), groups in ([1:32,33:64], [collect(1:2:64),collect(2:2:64)])
    x=BlockIterate(ones(T,64),groups);u=ones(T,32);v=blockview(x,1)
    println("\nCOPY ",T," ",typeof(v)," METHOD ",which(copyto!,Tuple{typeof(v),typeof(u)}))
    show(stdout,MIME"text/plain"(),code_typed(writeprobe!,Tuple{typeof(x),typeof(u)};optimize=true));println()
    code_llvm(stdout,writeprobe!,Tuple{typeof(x),typeof(u)};debuginfo=:none)
    code_llvm(stdout,normprobe,Tuple{typeof(x)};debuginfo=:none)
end
for T in (Float32,Float64), n in (5,256)
    f=GroupNormL2(T(.3),[1:n]);u=view(ones(T,n+2),2:n+1);out=similar(u)
    println("\nMATH ",T," n",n," METHOD ",which(prox_block!,Tuple{typeof(out),typeof(f),typeof(u),Int,T}))
    code_llvm(stdout,mathprobe!,Tuple{typeof(out),typeof(f),typeof(u)};debuginfo=:none)
    code_llvm(stdout,inplaceprobe!,Tuple{typeof(u),typeof(f)};debuginfo=:none)
end
