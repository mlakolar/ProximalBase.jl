using Test

include("testutils.jl")

tests = [
  "utils",
  "lapack",
  "partial",
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
  "lla",
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
  "caches",
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
