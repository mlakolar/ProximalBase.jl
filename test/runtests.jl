using Test

include("testutils.jl")

tests = [
  "utils",
  "lapack",
  "sparseIterate",
  "atomIterate",
  "interface",
  "smooth",
  "separable",
  "norms",
  "spectral",
  "sets",
  "calculus",
  "allocations",
  "deprecated",
  "aqua"
]

for t in tests
  f = "$t.jl"
  println("* running $f ...")
  t = @elapsed include(f)
  println("done (took $t seconds).")
end
