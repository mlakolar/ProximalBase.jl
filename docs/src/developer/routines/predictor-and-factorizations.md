# [Predictor oracles and cached least-squares solves](@id routine-predictor-and-factorizations)

The value and gradient of `LeastSquaresLoss` use the residual ``X\beta-Y``.
A caller that has the predictor can supply it directly. The proximal
operation uses a cached factorization of the design matrix.
The [least-squares contract](../../spec/leastsquares.md#spec-leastsquares-range)
and the [predictor and residual guide](../../smooth.md#Least-squares-at-a-predictor-or-residual-that-the-caller-holds)
own the supported arguments and numerical behavior.

## Forming the residual

The ordinary value and fused value/gradient calls form the predictor
``X\beta`` in the function's scratch buffer. The fused call overwrites
that buffer with the residual, multiplies the residual by ``X^\mathsf{T}``,
and passes the residual to the value reduction.
[Ordinary value and fused paths](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/leastsquares.jl#L87-L114).

With a strided vector response, the caller can supply a predictor or
a residual for a value evaluation. The fused predictor call writes
``d=z-Y`` in the caller's buffer and uses one product ``X^\mathsf{T}d``
for the gradient. The matrix-response specialization has a same-type
dense design and response and supports the two predictor calls with
strided buffers. Its value-only call reduces the differences
``z_i-Y_i`` directly, while its fused call reduces the entries of
`d` after the transpose product. The guide specifies the float types
and supported `Scaled` wrapper.
[Supported specializations](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/leastsquares_predictor.jl#L7-L17),
[vector calls](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/leastsquares_predictor.jl#L241-L262),
[matrix calls](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/leastsquares_predictor.jl#L264-L296),
and [vector residual call](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/leastsquares_predictor.jl#L299-L303).

## Solving the proximal problem

The nonempty proximal solve requests a cache built from a dense copy of
the design in the computation type. From the thin SVD ``X=U\operatorname{diag}(s)V^\mathsf{T}``,
it stores ``V``, the squared singular values, and
``b=\operatorname{diag}(s)U^\mathsf{T}Y``. The solve projects the trial
``v`` to ``q=V^\mathsf{T}v``. With a square ``V``, the result is
``V`` times the componentwise quotient ``(b+\rho q)/(s^2+\rho)``,
where ``\rho=1/\gamma``. For steps at most one, the implementation
uses the equivalent quotient ``(\gamma b+q)/(\gamma s^2+1)``.
With a rectangular ``V``, it adds ``V((b-s^2q)/(s^2+\rho))`` to
the trial. The [least-squares guide](../../smooth.md#Least-squares-loss)
describes when to reset this cache.
[Cache construction and solve](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/leastsquares.jl#L308-L365)
and [cache invalidation](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/caches.jl#L1-L18).

## Tests

- [Product-count tests](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/leastsquares_predictor.jl#L638-L678)
  check the vector calls, and [matrix-view tests](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/leastsquares_predictor.jl#L1235-L1277)
  check strided predictor and output buffers.
- [Matrix-response proximal test](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/core_functions.jl#L56-L61)
  checks against a direct linear solve. [Cache lifecycle tests](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/cache_and_blocks.jl#L3-L12)
  compare a reset solve with a fresh loss after changing the data.
