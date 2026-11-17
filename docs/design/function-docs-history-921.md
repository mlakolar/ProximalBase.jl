# Function documentation implementation history

This file preserves implementation rationale and measurements moved out of
user-facing help in the documentation refactor for
[issue #921](https://github.com/mlakolar/ProximalBase.jl/issues/921).
The source snapshot is `18b2714955c224300f8924aeb3a205b84c62be57`. These notes describe that implementation
and its reported measurements; they do not replace the live numerical or
representation contracts or establish new guarantees.

## Elastic-net step traversal

The public rule remains in the rendered penalty guide: scalar proximal steps
are checked, array-step entries are caller-validated, and the specialized and
general loops store the same results bit for bit. The following implementation
conditions explain the fast-path selection and why validation was not added.

This is an exception to the requirement of
[#908](https://github.com/mlakolar/ProximalBase.jl/issues/908) that
negative and `NaN` steps fail before writes: a check reads every step,
while such a call reads the steps before its loop only to choose the loop,
up to the first block of 256 steps that holds a step outside the ordinary
branches (below), and a call from a `SparseIterate` into a `SparseIterate`
reads none; the check would add a pass over the steps, or the rest of one
([#874](https://github.com/mlakolar/ProximalBase.jl/pull/874)).

Let `T` be `float` of the element type of the entries. A step ``\gamma``
of entry ``i`` takes the ordinary branches when both conditions hold:

- `λ1` is zero, or the product ``\gamma\lambda_1`` is a positive normal
  number of the type `R` in which it is formed. `R` is the promotion of `T`
  with one type for each of ``\gamma``, `λ1` and the weight of entry ``i``:
  `float` of its type, or `Float16` for a `Bool` (an `ElasticNet` without
  weights has the weight `true`).
- For `T` = `Float32` or `Float64`: one of ``\gamma``, `λ2` is zero or not
  finite, or ``\gamma\lambda_2``, formed in the promotion of `T` with `float`
  of the types of ``\gamma`` and `λ2`, lies in ``[2^{-126}, 2^{125}]``
  (`Float32`) or ``[2^{-1022}, 2^{1021}]`` (`Float64`). Other types of `T`
  have no condition on ``\gamma\lambda_2``.

When the argument and the array of steps have concrete element
types, a call other than from a `SparseIterate` into a `SparseIterate` runs
a loop specialized to these branches if every step takes them; one step
outside them, anywhere, sends the whole call to the general loop. Both loops
store the same results, bit for bit.

## LinearFunction sparse-value measurements

The following timings were recorded in the source comment of the pinned
snapshot. They are historical measurements, not a new benchmark campaign.
The source retains the coefficient-access and summation invariants.

In benchmark/linear_sparse_value.jl at this change (AMD Zen 3, Julia
1.13.3, Float32 and Float64, m = 128 and 1024), fully stored iterates in sequential or shuffled
storage order with 0, 50% and 90% stored zeros took 0.45 to 0.94 times the time of the base's
generic dot(c, x) over the matrix SparseIterate. In quick paired probes, a traversal in column
order through ind2nzval was slower than this loop in every cell measured (iterates of 2% to 100%
stored entries).
