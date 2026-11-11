using ProximalBase, LinearAlgebra
X=[1.0 2;3 1;0 2;2 -1]
y=[1.0,2,3,-.5]
loss=(1/size(X,1))*LeastSquaresLoss(y,X)
penalty=NormL1(.2)
β=[.3,-.4]
objective=loss(β)+penalty(β)
g=gradient(loss,β)
z=prox(penalty,β,.1)
@assert objective≈sum(abs2,X*β-y)/(2size(X,1))+.2sum(abs,β)
@assert g≈X'*(X*β-y)/size(X,1)
@assert z≈sign.(β).*max.(abs.(β).-.02,0)
