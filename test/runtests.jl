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
  "sparsegroup",
  "fibers",
  "spectral",
  "sets",
  "calculus",
  "allocations",
  "deprecated",
  "examples",
  "aqua"
]

for t in tests
  f = "$t.jl"
  println("* running $f ...")
  t = @elapsed include(f)
  println("done (took $t seconds).")
end
