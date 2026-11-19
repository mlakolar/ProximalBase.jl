using ProximalBase
for n in (4096,8192,16000,32000)
    v=ones(Float16,n)
    for x in (zero(v),-v)
        c=Float16(n/2)
        score=kkt_violation(NormLinf(c),v,x)
        println(join((n,all(iszero,x) ? "half-ball" : "half-face",score,(n-Float64(c))/sqrt(n)), '\t'))
    end
    h=fill(ldexp(Float16(1.5),-14),n)
    z=kkt_violation(NormLinf(zero(Float16)),h,zero(v))
    println(join((n,"half-remainder",z,Float64(h[1])*sqrt(n)), '\t'))
end
