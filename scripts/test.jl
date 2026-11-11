# Use the declared test environment, with the same runner/inventory as CI.
using Pkg
Pkg.activate(dirname(@__DIR__))
Pkg.test(; test_args=ARGS, julia_args=["--check-bounds=yes"])
