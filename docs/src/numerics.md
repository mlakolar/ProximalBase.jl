# Numerical contract

The supported validation targets are Float32 and Float64. Functions accept
real data and inputs. Retained kernels preserve their established model
precision: input/output float precision determines delivered values and
parameters are rounded according to each function's documented role; design
and quadratic data participate in promotion. `value_and_gradient!` returns the
standalone value at the input and stores the standalone gradient.
`value_and_prox!` returns the value at the stored output.

Ordinary finite calls preserve the existing arithmetic order. Exceptional
coefficients and sums use bounded significands, compensated accumulation and
powers of two to recover representable results where intermediate products
would overflow or underflow. NaN/Inf and zero levels/weights retain the
function-specific priorities described by the API docstrings. Infinite
weights of NormL1 fix an entry at zero, while zero weights leave it unpenalized.
Do not infer general cancellation rules from ordinary real algebra at Inf/NaN.

Exact input/output aliasing is supported by proximal operators. Other shared
memory, including shifted/reversed views, is unsupported; copy the source
first. Parameter and data aliases follow the retained kernels' checks.
Predictable shape, type and parameter failures precede output writes. Failures
that occur while calculating may leave output changed.

Symmetric owner output takes a scalar step. Per-entry step arrays are rejected
before writes. Caller-supplied per-entry parameter arrays must be exactly
symmetric at the pairs read, including through Scaled. Stored-only kernels may
check only stored pairs. Off-diagonal matrix values have multiplicity two;
independent coordinates have one stored value.
