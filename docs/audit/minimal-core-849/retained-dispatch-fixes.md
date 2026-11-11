# Retained dispatch corrections

Independent validation restored the exact frozen public/private BlockIterate
KKT reducers for scalar penalties. The block number denotes an ordered group;
the violation is the maximum over its indices, including for Scaled(NormL1).
The AbstractArray fallback would otherwise silently change that meaning.

Four exact reset_cache! methods remain necessary for retained data stored by
reference: AbstractFunction fallback, LeastSquaresLoss, QuadraticFunction and
Scaled forwarding. Tests construct the decomposition, mutate referenced X/Y/A,
reset and compare a fresh function. No constrained/group function is restored.

The generic owner write preflights and tuple parameter checks recovered during
the first load/reference pass remain included. The final declaration policy
selects 407 unchanged storage declarations and 912 extracted declarations.
The unused _ridge_weights helper's consumers were exclusively deferred and it
is now archived. PrecisionError remains archived because only deferred Newton
oracles raised it.

Coherent source names are functions/scalar_loss.jl, scaled.jl and array_steps.jl.
Documentation-only edits remove retired links/examples; storage executable ASTs
still match the frozen baseline exactly. There are 17 source files,333106 bytes
(79.8% fewer than the frozen full tree). Broader test/docs/CI packaging and the
final independent review are separate commits/gates.
