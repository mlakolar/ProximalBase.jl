using ProximalBase
x=setprecision(256) do
    [BigFloat(1), nextfloat(BigFloat(1))]
end
setprecision(128) do
    v=BigFloat[-1,-2]
    try
        println("BigFloat\t",kkt_violation(NormLinf(BigFloat(2)),v,x),"\texpected 1")
    catch err
        println("BigFloat\t",typeof(err),"\t",sprint(showerror,err),"\texpected 1")
    end
end
v=fill(Float16(200),30000)
println("zero_radius_Float16\t", kkt_violation(NormLinf(Float16(0)),v,zero(v)), "\t",Float16(200sqrt(30000)))
