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
  "losses",
  "entropy",
  "norms",
  "sparsegroup",
  "fibers",
  "spectral",
  "singular",
  "intersection",
  "sets",
  "ordered",
  "nonconvex",
  "matrixsets",
  "affine",
  "calculus",
  "constrained",
  "spectralfunction",
  "ballconstrained",
  "constrained_nonconvex",
  "testing",
  "second_order",
  "envelope",
  "composition",
  "metric",
  "allocations",
  "gpu",
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
