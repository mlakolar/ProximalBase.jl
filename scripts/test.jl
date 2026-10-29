# Run the standard package tests with the options used by CI.
using Pkg

depwarn = get(ENV, "JULIA_TEST_DEPWARN", "yes")
depwarn in ("yes", "no", "error") || error("invalid JULIA_TEST_DEPWARN")
coverage = get(ENV, "JULIA_TEST_COVERAGE", "false") == "true"

Pkg.activate(dirname(@__DIR__))
Pkg.test(; coverage, julia_args=`--startup-file=no --check-bounds=yes --depwarn=$depwarn`, test_args=ARGS)
