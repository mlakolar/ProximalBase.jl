using FactCheck

using ProximalBase


tests = [
	#"test_utils",
  "test_proximal_functions"
]

for t in tests
	f = "$t.jl"
	println("* running $f ...")
	include(f)
end

FactCheck.exitstatus()
