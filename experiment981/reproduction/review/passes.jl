using ProximalBase
function test_passes(::Type{T}, n) where {T}
    v = ones(T,n)
    for j in 3:n
        v[j] = -T(big(factorial(big(j)))//factorial(big(n)))
    end
    x=-ones(T,n)
    reads=Ref(0)
    function countread(a)
        reads[]+=1
        a
    end
    tau=ProximalBase._linf_face_level(v,x,one(T),countread,T(2))
    println(join((T,n,reads[]÷n-1,tau), '\t'))
end
for T in (Float32,Float64), n in (8,12,16)
   test_passes(T,n)
end
setprecision(2048) do
   for n in (16,48,96)
       test_passes(BigFloat,n)
   end
end
