using SHA
function stripdocs(x)
    x isa LineNumberNode && return nothing
    if x isa Expr
        if x.head==:macrocall && length(x.args)>=4 && x.args[1]==GlobalRef(Core,Symbol("@doc"))
            return stripdocs(x.args[end])
        end
        # Parsed doc macro is usually a dotted Core.@doc expression.
        if x.head==:macrocall && occursin("@doc",string(x.args[1]))
            return stripdocs(x.args[end])
        end
        return Expr(x.head,filter(!isnothing,stripdocs.(x.args))...)
    end
    x
end
for file in ("sparse_iterate.jl","block_iterate.jl")
 a=stripdocs(Meta.parseall(read(joinpath("/tmp/pb852-frozen/post848/src",file),String)))
 b=stripdocs(Meta.parseall(read(joinpath("/tmp/pb852-frozen/post850/src",file),String)))
 println(file," executable AST equal: ",isequal(a,b)," digest848=",bytes2hex(sha256(string(a)))," digest850=",bytes2hex(sha256(string(b))))
end
