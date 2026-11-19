using ProximalBase
for T in (Float32,Float64)
    n=4096
    δ=nextfloat(zero(T))
    for H in (floatmin(T),2*floatmin(T))
        # Use a mass increment representable at this binade.
        increment=H==floatmin(T) ? 2047 : 2046
        v=zeros(T,n)
        v[1]=H
        c=H+increment*δ
        y=kkt_violation(NormLinf(c),v,-ones(T,n))
        ref=T(increment*big(δ)/64)
        println(join((T,H,increment,y/δ,ref/δ), '\t'))
    end
end
