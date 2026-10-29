using Test

include("testutils.jl")

tests = [
  "utils",
  "lapack",
  "compat",
  "partial",
  "sparse_iterate",
  "iterate_queries",
  "atom_iterate",
  "symmetric_products",
  "interface",
  "precision",
  "smooth",
  "separable",
  "binary",
  "losses",
  "sqrhinge",
  "entropy",
  "bernoulli",
  "norms",
  "cubenorm",
  "sparsegroup",
  "fibers",
  "spectral",
  "singular",
  "intersection",
  "sets",
  "soc",
  "extreme_normals",
  "large_inputs",
  "spectral_overflow",
  "ordered",
  "nonconvex",
  "lla",
  "matrixsets",
  "affine",
  "calculus",
  "blockchecks",
  "constrained",
  "spectralfunction",
  "ballconstrained",
  "constrained_nonconvex",
  "testing",
  "second_order",
  "envelope",
  "distance",
  "composition",
  "precompose_array",
  "caches",
  "metric",
  "allocations",
  "gpu",
  "deprecated",
  "examples",
  "readme",
  "docstrings",
  "aqua"
]

for t in tests
  f = "$t.jl"
  println("* running $f ...")
  t = @elapsed include(f)
  println("done (took $t seconds).")
end
