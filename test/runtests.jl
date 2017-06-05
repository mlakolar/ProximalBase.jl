using FactCheck

using ProximalBase
using Distributions


function try_import(name::Symbol)
    try
        @eval import $name
        return true
    catch e
        return false
    end
end

grb = try_import(:Gurobi)
jmp = try_import(:JuMP)
ipopt = try_import(:Ipopt)


tests = [
	"utils",
  "sparseIterate",
  "proximal_functions",
	"differentiable_functions"
]

srand(1)

for t in tests
	f = "$t.jl"
	println("* running $f ...")
	include(f)
end

FactCheck.exitstatus()
