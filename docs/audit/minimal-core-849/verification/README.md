# Independent verification scripts

The separate architecture critic compared 10,080 exceptional scalar-step
configurations and 3,360 array-step configurations against the frozen post-848
source with zero mismatches in exact canonical outputs or exception kinds.
The captured logs accompany the scripts. Path selection was parameterized on
integration; numerical bodies are unchanged. A custom Scaled separable function
also passed 80 assertions over dense/sparse/block/symmetric storage.

Reproduce with the baseline snapshot prepared as in benchmark/README.md:

```sh
PB_BASELINE=/tmp/pb849-baseline PB_CANDIDATE="$PWD" julia --project=benchmark docs/audit/minimal-core-849/verification/exceptional.jl
PB_BASELINE=/tmp/pb849-baseline PB_CANDIDATE="$PWD" julia --project=benchmark docs/audit/minimal-core-849/verification/arraysteps.jl
julia --project=. docs/audit/minimal-core-849/verification/custom.jl
```

These complement the independent dense/BigFloat references in the supported
test inventory and the cross-implementation benchmark smoke. They do not imply
that the historical full package suite passed.

`baseline-display.jl` reproduces the inherited display-allocation expectation
against the frozen source and frozen test, under the current Julia runtime.
Its captured log contains two failures of the old 64-byte growth allowance;
this is intentional historical evidence, not a passing test or part of the
supported runner. The migrated test keeps a constant 512-byte allowance and
the original strict total-allocation cap. Display/storage code is unchanged.

The fresh final reviewer passed 78 further assertions on source e272df3b/d11,
including empty, underdetermined and multiresponse least squares; exact aliases
and cache resets; ordered weighted Scaled block KKT; a live read-only symmetric
matrix view; and custom Scaled symmetric parameter validation. The reviewer
executed the body once through stdin, saved the script afterward and provided
an explicitly transcribed stdout summary/provenance. The .log labels this
accurately; it is not represented as a captured full raw process log.
