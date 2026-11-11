# Complete lasso-style residual-maintaining synthetic workloads. Only the
# supplemental harness includes these functions. Scratch and operands are owned
# by fixtures; no solver dependency or production package methods are added.
@inline residual_soft(u,t)=copysign(max(abs(u)-t,zero(u)),u)
function residual_cd!(x,idx,rows,cols,A,r,L,lambda,repeats)
    T=eltype(x)
    for _ in 1:repeats,k in idx
        old=coord(x,k,rows,cols);g=zero(T)
        for j in axes(A,1);g+=A[j,k]*r[j];end
        new=residual_soft(old+g/L[k],lambda/L[k]);delta=new-old
        coordset!(x,new,k,rows,cols)
        for j in axes(A,1);r[j]-=delta*A[j,k];end
    end
    x
end
function residual_bcd!(x,ks,groups,A,r,steps,lambda,newvalues,deltas,repeats)
    T=eltype(x)
    for _ in 1:repeats,k in ks
        v=block(x,k);gamma=steps[k]
        # Every trial gradient reads the residual before this whole block changes.
        for (j,i) in enumerate(groups[k])
            g=zero(T)
            for row in axes(A,1);g+=A[row,i]*r[row];end
            old=v[j];newvalues[j]=residual_soft(old+gamma*g,gamma*lambda)
            deltas[j]=newvalues[j]-old
        end
        for (j,i) in enumerate(groups[k])
            v[j]=newvalues[j]
            for row in axes(A,1);r[row]-=deltas[j]*A[row,i];end
        end
    end
    x
end
