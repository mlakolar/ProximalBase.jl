using Test

tests = [
  "utils",
  "sparseIterate",
  "proximal_functions",
  "differentiable_functions"
]

for t in tests
	f = "$t.jl"
	println("* running $f ...")
	t = @elapsed include(f)
    println("done (took $t seconds).")
end
