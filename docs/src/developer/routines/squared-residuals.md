# [Squared residuals in power-of-two units](@id routine-squared-residuals)

The squared-residual kernel evaluates weighted sums of squares for
`SqrNormL2` and least-squares values. Scaling the residuals can keep the
computation in range when an ordinary subtraction, square, or sum overflows
or when the squares underflow. The
[SqrNormL2 range rules](../../spec/sqrnorml2.md#spec-sqrnorml2-range) and
[least-squares range rules](../../spec/leastsquares.md#spec-leastsquares-range)
give the conditions for the public operations.

## Ordinary path

The shared value helper forms ``l s/2`` from the chosen level ``l`` and
the ordinary square sum ``s``. It uses that result when the product
is finite, the level passes its conversion check, and the sum clears
the range screen
``s\ge n\,{\operatorname{floatmin}(T)}/{\operatorname{eps}(T)}``.
Here ``n`` counts the terms reduced and ``T`` is the input floating type.
Below that screen, rounding the individual squares can lose accuracy.
The same-type strided vector paths in `LeastSquaresLoss` use a
specialized reduction and range screen before calling the shared
helpers for recovery.

## Range recovery

For homogeneous finite inputs, recovery locates the exponent ``k`` of
the largest nonzero residual magnitude. The scan forms ``x_i-y_i``
before rescaling, so cancellation occurs in the original units. If the difference of two finite entries overflows, it
instead forms ``x_i/2-y_i/2`` and includes the factor of two in the scale.
The kernel then sums the squares in units of ``2^{2k}`` with a
compensated reduction. If ``s`` is this scaled sum, the value is
reassembled as ``\lambda s\,2^{2k-1}`` using the
[exponent-scaled product](@ref routine-scaled-products).

When the second operand is absent, the kernel uses the entries themselves.
In that case sparse storage supplies its stored values, and each stored
off-diagonal square of a symmetric iterate contributes once for each
matrix position it represents. With two operands, the kernel visits
their paired indices. An implicit zero can then have a nonzero residual
against the other operand.

An exactly zero residual sum is represented as zero with exponent zero.
If the operands supplied to this kernel contain non-finite entries, the
scaled helper returns no sum and its value wrapper keeps the ordinary
result. See the
[shared range rules](../../numerics.md#numerical-value-range) for
uncertainty near the overflow boundary.

## Source and tests

- [Ordinary and exceptional value paths](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/numerical_kernels.jl#L510-L547),
  [residual formation](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/numerical_kernels.jl#L588-L638), and
  [scaled sum and reassembly](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/numerical_kernels.jl#L676-L719).
- [The `SqrNormL2` value and symmetric multiplicity](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/sqrnorml2.jl#L79-L107)
  and [least-squares exceptional value dispatch](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/leastsquares.jl#L167-L212)
  show which operands those routes pass to the kernel.
- [Kernel tests](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/numerical_kernels.jl#L113-L169) compare
  the residual and scaled sum against higher-precision references,
  including large finite differences, cancellation, and sparse and symmetric
  storage. [Weighted value tests](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/numerical_kernels.jl#L513-L530)
  check reassembly when the ordinary square leaves the working range.
