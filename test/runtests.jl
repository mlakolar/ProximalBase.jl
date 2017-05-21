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
scs = try_import(:SCS)


tests = [
	"test_utils",
  "sparseIterate",
  "test_proximal_functions",
	"test_differentiable_functions"
]

for t in tests
	f = "$t.jl"
	println("* running $f ...")
	include(f)
end

FactCheck.exitstatus()
