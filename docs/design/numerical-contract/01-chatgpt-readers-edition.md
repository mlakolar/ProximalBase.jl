<a id="start"></a>

# The ProximalBase.jl numerical contract

*READER’S EDITION · DRAFT 2*

**A guide to what a value, gradient, or proximal step means when floating-point arithmetic reaches its limits.**

Suppose you fit a model with a penalty on one coefficient. You store the coefficient in `Float32` to save memory, but you keep the penalty weight in `Float64`. You ask the package for a new coefficient. The correct answer is one half. Does your choice of storage give the package permission to return zero instead?

It does not. One half fits exactly in either type. The risk is that an implementation can turn the weight into infinity *before* using it. At that point it is no longer solving the problem you gave it.

> **Our promise.** For the operations covered here, ProximalBase works with the numbers you actually supplied. Choosing a smaller output type permits rounding the answer. It does not permit turning a finite penalty into a hard constraint, reporting a feasible point as infeasible, or dropping a coefficient that should remain nonzero because an intermediate calculation used the wrong range.

### How to read this edition

**For the main argument**, start with [one penalty, three answers](#ridge). Then read [who chooses the precision](#types), [the half that must not disappear](#mixed), and [which point a combined call describes](#joint). Those four examples explain most of the architecture.

**For a particular behavior**, use the [case finder](#finder). The chapters distinguish mathematical zero, an ignored penalty, a fixed coordinate, invalid data, and genuine overflow. They also identify exceptions by name.

**For implementation planning**, read the contract first. The [separate planning brief](#handoff) explains what evidence to collect and when to leave working code alone.

**Status · 31 October 2026.** This is a proposed contract, not a statement that a released version passes every example. It expands Draft 1 without replacing its named exceptions or case identifiers. Statements about existing behavior refer to the pinned repository baseline `3f90a29`, not to an uninspected checkout. [[R1](#ref-R1)]

<a id="ridge"></a>

## One penalty, three different answers

*I · THE PUBLIC PROMISE*

The running example is ridge regularization, called `SqrNormL2` in ProximalBase. For one real coordinate with no center, its penalty is **f(x) = λx²/2**. The weight λ says how costly it is for the coordinate to be away from zero. No familiarity with the package is needed to see the distinction below. [[R2](#ref-R2)]

Take an ordinary case: **x = 6, λ = 4**. Calling the package for the *value* asks, “What is the penalty at six?” It is 4 × 36 / 2 = **72**. Calling for the *gradient* asks, “How fast does that penalty change at six?” The derivative is λx, so the answer is **24**. Both questions refer to the original point. In the centered version, replace x by its difference from the specified center when calculating the penalty and the gradient.

A *proximal step* asks a different question: “Where should I move if I want to reduce the penalty, but moving away from six also costs something?” A positive step size γ sets the price of moving. The new point z minimizes **f(z) + (z − x)²/(2γ)**. Setting the derivative to zero gives **z = x/(1 + γλ)**. With γ = ¼ the new point is **3**.

```julia
f = SqrNormL2(4.0)
x = [6.0]
g, z = similar(x), similar(x)
value_and_gradient!(g, f, x)     # returns 72.0; writes [24.0]
value_and_prox!(z, f, x, 0.25)   # returns 18.0; writes [3.0]
```

Why **18** in the last line? It is the penalty at the new point: 4 × 9 / 2. It is neither the old penalty (72) nor the minimized expression, which includes another 18 for moving. This distinction becomes consequential when the new point is rounded for storage.

> **Reading convention.** An *oracle* is simply a routine that answers one of these mathematical questions. A *combined* or *fused* call answers two at once. Sharing arithmetic is welcome; changing either question is not. All examples in this guide describe expected behavior, not execution logs.

<a id="scope"></a>

## What is guaranteed, and for which answers?

*I · THE PUBLIC PROMISE*

A scalar penalty computed by adding squares and a matrix projection obtained from an eigendecomposition should not be held to the same test. But putting the two in different categories should not let a matrix routine return an arbitrary objective value. We therefore attach the promise to **each result**, not to an entire function name. [[R1](#ref-R1), [R5](#ref-R5)]

### A · Scalar function values

For `f(x)` when it exists, and for the scalar returned by a combined call, protect a normal finite answer against avoidable loss of range in intermediate arithmetic. Return the specified type and respect the function’s domain. For a value calculated from an approximate point or spectrum, state how accurately that quantity is known. A documented gap is an exception to track, not a reason to give up the value guarantee for the whole family.

### B · Named closed-form outputs

Initially this stronger component-by-component promise applies to the gradient and prox of `SqrNormL2`, and to the prox of `NormL1`, `NormL2`, and `GroupNormL2` on supported inputs. Ridge values are also covered by A. Protect resolvable normal components and their zero/nonzero decisions. A method with a simple formula does not automatically enter B; naming the supported families makes the commitment testable.

### C · Spectral, iterative, and other composite outputs

Use a stable method appropriate to the problem. Specify the tolerance, where rounding occurs, and how failure is reported. A result may be reliable as a matrix even when one tiny entry is not relatively accurate. Tier C is not permission to ignore an obvious avoidable overflow.

**Where this applies.** The reference is supported real `Float32` and `Float64` CPU calls, including mixtures, with round-to-nearest and gradual underflow. Gradual underflow means the type retains some nonzero numbers smaller than its smallest normal number. Supported device and structured methods have the qualifications later in this guide. No new support for complex numbers, AD dual numbers, half precision, integer destinations, or arbitrary-precision backends is implied; existing support is not removed.

**How to read a rule.** “Must” describes the target; “should” allows a justified departure; “may” leaves the choice open. None implies universal correct rounding, identical bits on every platform, or exact feasibility from an approximate projection.

<a id="types"></a>

## Who chooses the precision?

*I · THE PUBLIC PROMISE*

There are three decisions, not one. **Input precision** determines what information the caller supplied. **Working precision** is what the implementation uses along the way. **Output precision** determines what the caller receives. Only the first and third belong to the public interface. [[R1](#ref-R1)]

Imagine a `Float32` vector `x`, a `Float64` weight, and a `Float64` gradient buffer `g`. The joint value-and-gradient call still returns a `Float32` *value*, because the value is at `x`. It writes a `Float64` *gradient*, because `g` belongs to the caller. A wider weight or a wider scratch buffer does not change either rule.

| Call on one real array | What the caller receives |
|---|---|
| `f(x)` | A scalar of type `float(eltype(x))`, at `x`. |
| `gradient!(g,f,x)` | The gradient is written in `eltype(g)`; there is no scalar objective result. |
| `value_and_gradient!(g,f,x)` | The same gradient, plus the scalar value at `x` in `float(eltype(x))`. |
| `prox!(z,f,x,γ)` | The point is written in `eltype(z)`; the call returns `z` itself. |
| `value_and_prox!(z,f,x,γ)` | The point is written in `eltype(z)`; the scalar normally describes that stored point and has its floating type. |

Allocating `prox`, `gradient`, `value_and_gradient`, and `hessian_vec` use the input’s floating type for an ordinary single-array output. In-place Hessian actions retain the destination type. For tuple methods, a scalar uses the documented promotion across the relevant blocks; each written block retains its own type. Ordinary integer inputs follow the existing floating-result convention.

**What widening cannot do.** A `Float32` coordinate already stored as zero does not contain a small decimal value waiting to be restored. Likewise a factorization built from `Float32` data does not acquire `Float64` accuracy merely because its answer is copied into a wider array. Conversely, a finite, wide parameter still exists when the caller asks for a narrow output. A constructor that must store parameters more narrowly must retain the effect of a positive weight or reject an unsupported representation—not silently turn it into a zero mask or an infinite constraint.

<a id="range"></a>

## When the square does not fit, but the value does

*I · THE PUBLIC PROMISE*

Consider `SqrNormL2(1e-300)([1e200])`. Squaring the coordinate first gives a number near 10⁴⁰⁰. That exceeds the `Float64` range. Yet the *weighted* square is only about 10¹⁰⁰, and the requested penalty is about **5 × 10⁹⁹**. An answer of `Inf` would not describe the model or the output type. It would describe the order of operations. [[R1](#ref-R1), [R2](#ref-R2)]

**Case V1 · A check with exact binary numbers.** Let x = 2⁶⁰⁰ and λ = 2⁻¹⁰⁰⁰ in `Float64`. The unweighted square is 2¹²⁰⁰, but the exponents in the final value add to 1200 − 1000 − 1 = 199. The required answer is **2¹⁹⁹**. This explains why calling a reliable norm routine and then squaring its answer is not by itself a solution. The weight and the factor of one half still matter.

### A finite answer can also be wrong

**Case V2 · Small terms can matter together.** Use one coordinate of size 2⁻⁵¹¹, another 32 of size 2⁻⁵³⁸, and λ = 2¹⁰⁰⁰, all in `Float64`. The first square is 2⁻¹⁰²², which fits. Each of the other squares is 2⁻¹⁰⁷⁶, which rounds to zero when formed on its own. A calculation that drops them all returns 2⁻²³. That looks reasonable, and it passes an `isfinite` test.

But the missing terms add up. After applying λ/2, their combined contribution is 2⁻⁷². The required reference is **2⁻²³ + 2⁻⁷²**. At that magnitude, one step between adjacent `Float64` numbers is 2⁻⁷⁵. The lost contribution is eight such steps. The acceptance test permits at most two. Ordinary approximate equality is too forgiving here.

> **Rule.** For finite, in-domain data, a covered normal result must survive an intermediate square, sum, difference, product, ratio, or constant factor going out of range. A cheap ordinary formula plus selective rescaling is one way to honor this rule. The contract requires the result, not a particular algorithm. [[R1](#ref-R1), [X1](#ref-X1)]

<a id="mixed"></a>

## The half that must not disappear

*I · THE PUBLIC PROMISE*

Return to the question on the first page. In an uncentered ridge step, only the product γλ matters to the denominator. A large weight and a small step can exactly cancel one another’s scale. This is an example where mixed precision is not about gaining a few extra digits. It is about preserving the problem itself.

**Case V3 · Exactly one half.** Choose λ = 2¹⁶⁰ and γ = 2⁻¹⁶⁰, both stored as `Float64`. Put one in a `Float32` vector and ask for a `Float32` result. In exact arithmetic γλ = 1. The denominator is two, and **the stored point must be `0.5f0`**. In Julia, the suffix `f0` on `0.5f0` denotes a `Float32` literal.

```julia
λ = ldexp(1.0, 160)       # 2^160 in Float64
γ = ldexp(1.0, -160)      # 2^-160 in Float64
x = Float32[1]
z = similar(x)
prox!(z, SqrNormL2(λ), x, γ)  # target: z == Float32[0.5]
```

The wrong reasoning is “the output is `Float32`, so cast both scalars now.” The first cast produces infinity, the second zero. Multiplying them produces `NaN`. Handling that NaN as though the *caller* had passed invalid data would misdiagnose the error. Both stored inputs were finite. [[R1](#ref-R1), [R2](#ref-R2)]

**Case V5 · The same mistake affects gradients.** Keep λ = 2¹⁶⁰. Set the `Float32` coordinate to 2⁻⁸⁰. The gradient is λx = 2⁸⁰, an exactly representable `Float32`. Replacing λ by `Inf32` before multiplying would turn that finite answer into infinity.

> **Rule.** An implementation may narrow a quantity early only when doing so preserves the promised value, class, accuracy, and branch decision. It may combine scalar parameters in a wider type without widening a whole vector. It must still retain that combined quantity’s effect until it has been applied.

<a id="coefficient"></a>

## A shrinkage factor can be too small to store

*I · THE PUBLIC PROMISE*

A tempting fix for the previous example is to compute γλ in `Float64`, then convert the final shrinkage factor to `Float32` and apply it to the vector. That fixes the half. It does **not** fix the general problem. A caller asks for a new coordinate, not for a separately rounded multiplier.

**Case V4 · Multiply before losing the factor.** Let x = 2¹⁰⁰ in `Float32`, λ = 2¹⁶⁰ in `Float64`, and γ = 1. The shrinkage factor is 1/(1 + 2¹⁶⁰), slightly smaller than 2⁻¹⁶⁰. Converting the factor to `Float32` gives zero. Multiplying that zero by x gives zero too.

Now evaluate the requested coordinate as a whole: 2¹⁰⁰/(1 + 2¹⁶⁰). It is very close to 2⁻⁶⁰, and it **rounds to 2⁻⁶⁰ in `Float32`**. That is a normal nonzero number. The factor was too small; the answer was not.

**Case V6 · There may be no wider native type.** Use only `Float64`, with x = 2⁹⁰⁰ and γ = λ = 2⁸⁰⁰. Their product, 2¹⁶⁰⁰, overflows. Its reciprocal is too small to store on its own. But the requested coordinate is near 2⁻⁷⁰⁰ and **rounds to 2⁻⁷⁰⁰ in `Float64`**. Promoting mixed calls would not help: nothing was mixed.

A scale and an exponent can keep this information until the final operation. For instance, it is possible to carry the magnitude of a factor without materializing that factor as a native floating number. The contract does not require this representation on ordinary inputs. It requires the implementation not to discard a factor that still has a representable effect. [[R2](#ref-R2), [X1](#ref-X1)]

### What counts as underflow at the end?

A *normal* number lies between the type’s smallest positive normal magnitude, `floatmin(T)`, and `floatmax(T)`. A smaller nonzero magnitude may be *subnormal*, with fewer significant digits. For a genuinely subnormal final scalar, the general promise is absolute error on the scale of `floatmin(T)`, not uniform relative accuracy; zero may be permitted. This exception does not apply to V4 or V6. For true final overflow use the correctly signed infinity when final rounding overflows, not automatic saturation. An exact value just above `floatmax(T)` need not yet round to infinity. Named matrix reconstruction exceptions remain separate. [[R1](#ref-R1), [R6](#ref-R6)]

<a id="threshold"></a>

## Zero can change the model a solver fits

*I · THE PUBLIC PROMISE*

An L1 proximal step subtracts a threshold from the magnitude of each coordinate. When the magnitude is at or below the threshold, the new coordinate is zero. When it is above the threshold, the remaining difference is kept with the original sign. An optimizer that tracks only nonzero coefficients may treat those two outcomes very differently.

**Case V8 · A decision lost by early rounding.** Set x = `1f0`, γ = 1, and λ = `1.0 - 2.0^-30` in `Float64`. The coordinate is just *above* the threshold. Its new value is exactly **2⁻³⁰ in `Float32`**. The number one and this small difference can both be stored in `Float32`. What cannot be distinguished in that type is the threshold from one: narrowing λ first rounds it to `1f0`. The code now sees a tie and sets a genuinely active coordinate to zero.

Do not repair this by adding a small tolerance to the comparison. That enlarges the region classified as zero and changes the penalty’s behavior. Test just below the boundary, exactly at it, and just above it using the neighboring representable inputs. At equality the L1 prox is exactly zero. At a nonconvex threshold with more than one minimizer, the family’s documented tie choice applies instead. **Case S8** repeats these tests with and without in-place writes. [[R1](#ref-R1), [R2](#ref-R2)]

### Range and accuracy are different promises

Here every number after the bad conversion is finite. The error is a lost distinction, not an overflow. Similarly, an expression can subtract nearly equal finite numbers and lose a meaningful small difference. Use relative error or spacing in units of the last place (ULPs) for well-conditioned nonzero results; use an absolute or input-scaled error measure where relative error is meaningless. First check the correct branch and value class. Getting numerically close is not enough when the contract specifies exact zero.

Positive and negative floating-point zero mean the same thing for an objective value and for active-set membership unless the method promises otherwise. A genuine identity copy should preserve the sign of zero when the destination permits it. Ordinary bitwise fast-path behavior already promised for a fixed homogeneous computation stays in force; this is not a new guarantee of identical bits across devices or mixed computations. [[R1](#ref-R1)]

<a id="infinity"></a>

## Does infinity mean that something broke?

*II · EXCEPTIONS WITH MATHEMATICAL MEANING*

**Case S1 · A legitimate infinity.** Take the indicator of the interval [0, 1], `IndBox(0,1)`. An indicator costs zero for belonging to the set and positive infinity for violating it. At x = 2, `f(x)` must return **`+Inf`**. The corresponding proximal step with an ordinary positive step size projects 2 onto the interval. It writes **1**, and the combined call returns **zero** because one belongs to the set. All three answers are correct. [[R1](#ref-R1)]

Contrast this with the weighted square on the earlier page. There, the real answer was a finite number near `5e99`; getting `Inf` because the square overflowed would be a bug. There is also a third possibility: the real answer is finite but so large that it genuinely rounds to infinity in the requested output type. The scalar API need not tag these infinities differently. It **must** give the right result for each reason, and its documentation must not teach a solver that every `+Inf` proves infeasibility.

### The order of decisions

**First, is this a valid call?** Check the promised structural conditions: shapes, supported types, controls, and required disjoint buffers. A negative step is not an infeasible trial *point*. Under the target rule it is an invalid instruction.

**Second, does this operation explicitly ignore or constrain an operand?** A constant function has no active dependence on its input. An infinite hard weight, where allowed, is a constraint rather than a numeric multiplier. A method may define an input-based sentinel; that is a named exception, not a default.

**Third, which remaining inputs are non-finite?** An active NaN ordinarily propagates. A genuine positive or negative infinity uses the documented domain or a well-defined extension. A finite point outside an extended-real domain ordinarily returns a typed `+Inf` for the value; the derivative follows its own domain rule. Use finite-input recovery only for finite operands. This ordering describes meaning, not a requirement to scan the entire array three times.

<a id="nan"></a>

## How far should one NaN travel?

*II · EXCEPTIONS WITH MATHEMATICAL MEANING*

Consider `x = [NaN, 4.0]` and a unit step. The key question is not “does this array contain a NaN?” It is **“does the requested answer depend on the unknown coordinate?”** Two penalties with almost the same name give different answers.

**Case S3 · Separate coordinates versus a shared scale.** For `NormL1(1)`, the prox applies a threshold of one to each coordinate independently. The first output cannot be determined and is NaN. The second is 4 − 1 = 3. The target is **`[NaN, 3.0]`**. Invalidating the second coordinate would discard a result the package knows.

For `NormL2(1)`, both coordinates use a shrinkage factor determined by the norm of the entire vector. The norm is unknown, so neither output can be inferred. The target is **`[NaN, NaN]`**. An implementation might write “if the norm exceeds the threshold, shrink; otherwise fill with zeros.” With a NaN norm that comparison is false, but false does not mean the norm is small. Returning a whole block of zeros would tell the solver, incorrectly, that it should drop both coordinates.

For disjoint group norms, draw the boundary around the group: an active NaN invalidates the dependent group, not a separate group. For an ordinary scalar sum of active contributions, a NaN takes precedence over `+Inf` in another term. That scalar has no unaffected coordinates to preserve. A documented whole-gradient invalidation or an explicitly named sentinel may have different scope. [[R1](#ref-R1), [R3](#ref-R3)]

**A genuine infinity is not a NaN.** In uncentered ridge with finite positive λ and γ, a single input `-Inf` has value `+Inf`, gradient `-Inf`, and prox `-Inf`; positive infinity has the corresponding positive signs. These are extensions of that specific formula, not a general prescription for all penalties at infinity. When an extension depends on how one approaches a non-finite point, do not select a convenient limit and call it the mathematical answer.

> **Rule.** Keep exceptional behavior as local as the mathematical dependence allows. Do not hide an active NaN behind a false comparison, and do not contaminate independent results merely because they share an array.

<a id="weights"></a>

## Removing a penalty is not removing a constraint

*II · EXCEPTIONS WITH MATHEMATICAL MEANING*

The expression `0 * Inf` is not a policy for optimization software. A zero can disable a term, identify a feasible coordinate, or be a point at which an otherwise valid formula is singular. Decide which of those roles it plays before choosing the arithmetic. [[R2](#ref-R2), [R8](#ref-R8)]

**Case S2 · A genuinely disabled penalty.** For `SqrNormL2(0.0)` with a valid finite center, the penalty is identically zero. At input `[NaN]`, the target value is **zero**, the gradient is **`[0.0]`**, and the prox is an identity copy: **`[NaN]`**. The point was copied, not sanitized. Change the weight to one and the same input now gives a NaN value, gradient component, and prox component. With a finite positive weight, the empty or all-zero input has zero value for uncentered ridge. None of these shortcuts excuses an invalid shape or step.

**Case S4 · One free coordinate and one fixed coordinate.** For a weighted L1 family that supports hard weights, take weights `[0.0, Inf]`, x = `[3.0, 4.0]`, and a positive finite step. The first coordinate is free: keeping 3 costs nothing. The second must be zero: keeping 4 violates the constraint. The value at the input is **`+Inf`**, the prox is **`[3.0, 0.0]`**, and the value at that point is **zero**. The same free/fixed distinction holds when the overall penalty level is zero. A true infinite coordinate weight retains its hard-constraint meaning.

The same idea explains `0 * IndBox(0,1)`: the box remains a box. A nonnegative-domain penalty may simplify to the indicator of that domain, not to a function that accepts all points. When no supported domain-preserving zero scaling exists, retain the documented rejection. An infinite bound can mean “no bound on this side”; do not replace it with the largest finite float.

**Boundary of the rule.** An infinite overall level is not automatically an infinite coordinate weight. Only allow infinity in roles the family defines. Reject NaN model controls such as levels, weights, and bounds. For an ordinary zero weight and an infinite coordinate, honor a documented absent term; for a NaN coordinate, use that family’s explicit masking rule. In no case may a finite weight *acquire* hard-weight semantics through conversion.

<a id="loss"></a>

## Domain errors and missing data are not interchangeable

*II · EXCEPTIONS WITH MATHEMATICAL MEANING*

A line search trying a negative argument to an entropy penalty needs to know that the point is outside the domain. A NaN originating upstream does not establish that the point was negative. A comparison such as `t > 0` is false in both cases, so an implementation that only tests that condition cannot tell them apart.

**Case S5 · Negative entropy.** For positive unit weight, `NegEntropy` computes x log x on the nonnegative real line. At **zero**, its value is **zero** by the one-sided limit. At **−1**, its value is **`+Inf`**, because −1 is outside the domain. At **NaN**, the proposed value is **NaN**. The baseline scalar formula takes the `Inf` branch at NaN. The last outcome is therefore a proposed observable change that needs a compatibility note, not a claim about a released version. [[R3](#ref-R3)]

**A different policy for a masked observation.** `BernoulliLoss` deliberately tests a NaN probability before it tests a zero observation weight. Preserve that ordering. At a non-NaN input, a zero weight contributes zero value and gradient and leaves the point unchanged in the prox. This is not the same as the entirely absent penalty in S2. A reviewer must be able to tell which rule applies without guessing what `0 * NaN` did. [[R3](#ref-R3)]

### A finite tail is not an endpoint

For binary logistic loss with label one and a positive finite predictor η, the loss is `log1p(exp(-η))`. At η = 40 in `Float64` the loss is about `4.25e-18` and the derivative is about `-4.25e-18`. Both are nonzero. Computing the value as a large softplus minus 40, or the gradient as a rounded sigmoid minus one, can erase them. Keep the small terms explicit in the joint formula. [[R4](#ref-R4)]

Genuine infinite predictors are a separate case in S5. The proposed endpoint limits `(η,y)=(+Inf,1)` and `(-Inf,0)` have **zero loss and zero derivative**; `(+Inf,0)` has **`+Inf` loss and derivative +1**. Implementing those endpoint conventions does not justify cutting off the finite tail. Conversely, a stable finite formula does not establish what `0 * Inf` should mean at the limiting endpoint.

<a id="steps"></a>

## When is a zero step meaningful?

*II · EXCEPTIONS WITH MATHEMATICAL MEANING*

The ordinary proximal step has a finite positive step size. For finite positive ridge weight and a finite center, its two scalar limits have an intuitive meaning. As γ goes to zero, moving becomes too expensive and the point stays at the input. As γ grows without bound, the penalty dominates and the point goes to the center (zero for uncentered ridge). For finite inputs, those are the ridge endpoint results proposed here. They do not define the endpoints for every other function. [[R1](#ref-R1), [R2](#ref-R2)]

For an indicator, changing a positive scalar step does not change the projection: it multiplies the squared-distance objective by a positive constant. Where the family admits a scalar endpoint, the projection convention remains in force. It is **not** correct to install a package-wide rule that γ = 0 always means “copy the input.” Negative and NaN steps are invalid under the target policy; raise `ArgumentError` before mutation. A zero or infinite step requires an explicitly admitted extension. Do not quietly enlarge the existing backend’s support.

### Why an array of zero steps is different

**Case S7 · No unique limiting projection.** Start at `(0,0)` and project onto the two-coordinate simplex: choose nonnegative z₁ and z₂ that add to one. With positive coordinate steps a and b, the objective is `z₁²/(2a) + z₂²/(2b)`. The minimizer is `(a/(a+b), b/(a+b))`.

Along the path `(a,b)=(t,t)` both steps approach zero, and the projection stays **(½, ½)**. Along `(t,2t)` both still approach zero, but the projection stays **(⅓, ⅔)**. Specifying the endpoint `(0,0)` does not tell the package which answer the user intended. This is a mathematical ambiguity, not a missing numerical trick.

> **Rule.** For coupled diagonal metrics that require positive finite entries, reject a zero, NaN, or infinite entry **before writing any part of the destination**. Preload a buffer with recognizable values and test that it is unchanged. Preserve stricter method-specific conditions, such as a representable ratio for `IndIsotonic`. A scalar endpoint or a per-group rule is not evidence that the same endpoint is supported in a coupled array metric. [[R1](#ref-R1)]

<a id="joint"></a>

## Which point does a combined call describe?

*III · AGREEMENT BETWEEN CALLS*

A fused call saves work by computing two results together. It does not give the implementation a choice of which point the results describe. `value_and_gradient!` evaluates *at the input*. `value_and_prox!` ordinarily reports the penalty *at the point written into the destination*, after conversion to the destination type. [[R1](#ref-R1)]

**Case V7 · A genuinely infinite scalar and a finite gradient.** For centered ridge, let x = 2¹⁰²³, c = −2¹⁰²³, and λ = 2⁻¹⁰²², all stored in `Float64`. The real difference x − c is 2¹⁰²⁴, too large to store. The gradient, however, is λ(x − c) = 2² = **4**. The penalty is (λ/2)(x − c)² = 2¹⁰²⁵, which really does overflow. The joint call must therefore **return `+Inf` and write `[4.0]`**. Inferring that the derivative is invalid because the scalar overflowed would throw away correct information. [[R2](#ref-R2)]

**Case V9 · A stored point is not always the minimizer.** Choose a `Float64` center c = 1 + 2⁻²⁸, set x equal to c, and use unit weight and step. The mathematical proximal point is c itself, so its penalty is zero. But a `Float32` destination stores **1**, not c. The point the solver now holds is 2⁻²⁸ below the center. Its penalty is half the square of that difference: **2⁻⁵⁷**, a normal `Float32` number. Returning zero would attach a correct value to the wrong point.

> **Rule.** A fused calculation may reuse intermediate results. It need not always call `f(z)` again. Last-bit differences consistent with the promised accuracy are acceptable; a false zero, false feasibility, or a changed finite/NaN class is not. When a standalone value is unavailable, test the combined scalar against an independent mathematical characterization. The next section names the two exceptions to the stored-point rule.

The same logic applies to a gradient written to a narrower buffer. Do not reconstruct `f(x)` from the rounded gradient if doing so discards information. Evaluate both outputs for the same *input*, with the type rules for each result.

<a id="exceptions"></a>

## Two exceptions to the stored-point rule

*III · AGREEMENT BETWEEN CALLS*

One exception reports something about the *input*. The other reports something about the *ideal projection*. Neither should be inferred merely because evaluating the stored point would cost more. [[R1](#ref-R1), [R6](#ref-R6)]

### 1. An input-based warning for spectral methods

**Case S6 · An identity can still return a warning.** `NuclearNorm(0)` can copy an input matrix as its proximal point. Give it a matrix with a non-finite entry. The documented joint call preserves that identity output **and returns `+Inf`**. Here `+Inf` is an input-based *sentinel*: the API uses a particular scalar to flag a non-finite matrix. This exception takes precedence over the ordinary rule for a disabled ridge penalty and over recomputing the value at the stored point.

The baseline names `NuclearNorm`, `SingularValue`, `Spectral`, `TracePSD`, `IndPSD`, `IndBallRank`, `IndStiefel`, and `LogDetLoss` for this non-finite-input joint-scalar convention. It includes the full-rank `IndBallRank` identity case. For a direct `Conjugate(SingularValue(f))`, preserve `+Inf` **when the existing prox path returns**; paths documented to throw still throw. The convention is triggered by the *input*. It is not a rule for all non-finite outputs arising from finite inputs.

Do not pass a known non-finite matrix into an eigensolver or singular-value decomposition that cannot process it safely. Follow the documented behavior of the method: an allowed closed form can use its stated arithmetic, while an undefined coupled result may require NaNs in the dependent output entries. Do not automatically reject valid identities.

### 2. An approximate projection is not a feasibility certificate

An indicator is exactly zero at an exact projection. A method such as `IndIntersection` may approximate that projection iteratively and return zero as the ideal indicator value. This does **not** certify that its stored point satisfies every constraint. Test the violation of the constraints separately, using the method’s feasibility tolerance. A solver needing certified feasibility cannot obtain it just by checking whether the returned scalar equals zero.

**Review boundary.** Keep these exceptions visible in both the method documentation and conformance tests. A new exception needs a name, scope, rationale, and maintainer review. Do not extend either one to an ordinary closed-form method such as V9.

<a id="storage"></a>

## The layout can change; the mathematical vector cannot

*III · AGREEMENT BETWEEN CALLS*

A sparse vector can represent `[0, 3]` by storing only the three. The zero still belongs to the vector. May the implementation skip it? That depends on the operation. [[R7](#ref-R7)]

**A missing entry can move.** With unit weight and step, uncentered ridge maps zero to zero. But if the center is two, the new coordinate is halfway from zero to two: **one**. Skipping the implicit zero misses that output. For `NegLog`, a zero coordinate makes the value `+Inf`. Summing only over stored nonzeros even gives the wrong domain decision.

**A stored entry can count twice.** Take a symmetric 2 × 2 matrix with zero diagonal and off-diagonal entries equal to three. Storing only the lower triangle saves one copy of the three. The squared Frobenius norm still counts 9 + 9, not 9. The unit-weight ridge value is **9**, not 4.5. Read the correct multiplicity for a reduction; write the single stored off-diagonal coordinate only once. [[R2](#ref-R2), [R7](#ref-R7)]

For group or atom storage, check a dense expansion with unequal groups and implicit and explicit zeros. Preserve promised sparse complexity. A zero can matter for bookkeeping without being a mathematically active coefficient.

### When the output is also the input

**Case S8 · Aliasing and ties.** The supported call `prox!(x,f,x,γ)` must work even when a rare numerical fallback is triggered. A retry that needs the original x cannot read it from a buffer whose first half has already been overwritten. Compare the in-place answer with a disjoint-buffer answer using V4 and with the threshold neighbors of V8. For primitive separable operators, scalar-entry, coordinate-update, and full-array calls at comparable destination precision must take the same branch. A coupled group still needs the whole group.

Partial overlap such as shifted views is not promised. Gradient buffers must be disjoint unless an operator expressly supports overlap. Structural errors that can be validated in advance must leave caller-owned outputs unchanged, including tuple blocks and coordinate storage. A prescribed NaN result may be written: it is not an exception. There is no universal rollback promise for a failed external factorization or iterative solve; document which outputs remain valid. [[R1](#ref-R1)]

<a id="wrappers"></a>

## What happens when one function wraps another?

*III · AGREEMENT BETWEEN CALLS*

Scaling a function looks harmless: `Scaled(f, α)` asks for α times its value. Suppose the inner call returns `Inf`. Did it do so because the point was forbidden, or because the inner value was too large for its type? With **α positive and small**, the first answer is still infinity. In the second case the outer result may well be finite. Once the distinction has been lost inside a black-box call, multiplying its scalar output cannot reconstruct it. [[R5](#ref-R5), [R8](#ref-R8)]

For a simple known penalty, it may be possible to combine α with the weight before evaluating the value. The corresponding prox can sometimes be expressed with step αγ. This does not authorize rounding αγ to zero or infinity when the normalized result would fit. For a general wrapper with no way to recover the inner information, register the precise uncovered combination. Do not claim the guarantee or guess a finite answer.

### Some intermediate rounding is public behavior

There is an important contrast with the primitive ridge example. Some mixed `Translate` and `Tilt` array methods explicitly store an intermediate in the destination precision before the next stage. That intermediate rounding is a documented part of their behavior. A scalar `prox_entry` that carries the intermediate in a wider type need not match the staged array result. Test the staged result against a reference with the same storage boundary. Removing the boundary requires review; introducing an unnecessary cast inside an ordinary primitive is not the same thing. [[R1](#ref-R1)]

Other equivalences also have conditions. A restriction may drop an additive constant: gradients can agree even when the raw function values differ by that constant. A conjugate may provide a joint proximal value without a standalone evaluator. Neither difference is, by itself, evidence of a numerical bug. [[R1](#ref-R1), [R8](#ref-R8)]

**Cached state belongs to a calculation, not to mathematics.** A working buffer or cached factorization has a type, a size, a backend, and data from which it was built. Respect its documented reset and ownership rules. For example, after mutating retained model data where `reset_cache!` is prescribed, reset the outer object before using the affected prox. Reconstruct objects for which a reset is insufficient. A change of cache history alone must not silently change a valid result. Do not share mutable workspaces across callers contrary to their ownership contract, or introduce global mutable precision settings. [[R1](#ref-R1)]

<a id="spectral"></a>

## What can we promise about a matrix decomposition?

*III · AGREEMENT BETWEEN CALLS*

Suppose a method reconstructs a matrix from its eigenvalues. It may reconstruct the matrix accurately relative to its overall size, yet be unable to give many correct digits for an individual eigenvalue close to zero. A test that demands a small relative error for *every* eigenvalue is asking a different question from “did the method solve the stated problem accurately?” [[R5](#ref-R5), [R6](#ref-R6)]

For a factorization, a useful question is how closely the factors reconstruct the input. For an iterative projection, it is how closely the point satisfies the constraints and its optimality condition. For a linear system, a scaled residual compares the discrepancy in the equation with the size of the data. These are ways of testing **backward error** or a related residual: how much would the data or equation need to change for this computed answer to be exact? A small residual need not imply a small error in the solution of a nearly singular problem.

### Why not just round all tiny eigenvalues to zero?

Two callers may ask different questions about the *same* computed spectrum. One asks for a numerical rank. The other asks for the sum of the singular values. A relative zero threshold can be appropriate for the first: counting nonzeros is discontinuous at zero. Applying the same threshold to the second silently discards small contributions from a continuous penalty. Those contributions can add up. [[R5](#ref-R5)]

Preserve the recorded decision in issue #283: apply the relative zero mask to rank/nonzero-count functions for which it is intended. A continuous penalty sees the **unmasked computed spectrum**. A test of membership should use a tolerance appropriate to the source of the tested quantity. A directly supplied entry and an eigenvalue estimated from a large matrix need not carry the same uncertainty. Do not use the norm of a large unrelated entry to excuse violation of a directly supplied bound. [[R5](#ref-R5)]

> **Rule.** Tier-C methods must state what tolerance is used, what scale it refers to, how iteration is bounded if applicable, and what happens on failure. Test the appropriate residual or feasibility measure; use componentwise accuracy only where justified. This is not permission to ignore a preventable range failure in the separately returned scalar value. Known limitations, such as an overflowing spectrum in a particular standalone value, must be identified by method and result. [[R1](#ref-R1), [R5](#ref-R5), [R6](#ref-R6)]

<a id="devices"></a>

## What should stay the same on a GPU?

*III · AGREEMENT BETWEEN CALLS*

Take the L1 example in which the correct `Float32` result is 2⁻³⁰. A user can reasonably expect that moving a supported operator to a GPU will not remove that coordinate from the model. It is much less reasonable to demand that every last bit of a long parallel sum match a particular CPU’s order of addition. The contract distinguishes these two expectations. [[R1](#ref-R1), [X4](#ref-X4)]

**The semantic target** is the same output type, domain, zero/threshold decision, and scope of NaN or infinity for a supported operation in its advertised arithmetic mode. **The accuracy target** is appropriate to the result and mode. There is no cross-device bitwise promise for reductions or fused multiply-add choices. Likewise a backend that does not implement a method must not be described as one that implements it inaccurately: “not supported” and “failed a test” are different findings.

A device mode that *flushes subnormals to zero* may treat a stored tiny input as zero when it enters arithmetic. Do not promise to recover information discarded by that mode. Describe the limitation and test it on the device. This does **not** excuse forming an avoidably subnormal intermediate from normal operands and losing a normal final answer. The default CPU contract is not weakened to match a device setting. Avoid fast-math assumptions that permit a compiler to ignore NaNs or infinities in a kernel that explicitly promises to distinguish them. [[X4](#ref-X4)]

**A fallback must not change where the computation runs without saying so.** A risky scalar weight is not, by itself, a reason to copy the whole vector to the CPU, allocate a wide device array, or synchronize after each group. Keep ordinary native-precision calls cheap and preserve existing allocation and complexity guarantees on their documented paths. Report any necessary exception. Emulated GPU arrays can exercise dispatch and layout; they cannot certify the subnormal behavior, throughput, or synchronization cost of physical hardware. [[R1](#ref-R1), [R9](#ref-R9)]

<a id="solver"></a>

## What the solver must decide for itself

*III · AGREEMENT BETWEEN CALLS*

Imagine a line search starting from a valid point. It proposes x = 2 for a problem constrained to the box [0, 1]. ProximalBase is doing its job when the indicator returns `+Inf`. The line search is doing *its* job when it rejects the trial and tries a shorter step. There is no reason for the indicator to throw or to run its own backtracking algorithm. Nor is one rejected trial evidence that the model has no feasible solution. [[R1](#ref-R1), [X3](#ref-X3)]

Now change the example: the trial objective is NaN. The test “trial objective is greater than the acceptance bound” is false. That does not make the trial acceptable. A solver should check the conditions that establish acceptance, not treat failure of a rejection comparison as proof of success. It can bound its retries and report a numerical failure rather than looping indefinitely. [[X3](#ref-X3), [X5](#ref-X5)]

**Preserve the last accepted iterate.** To recover from an invalid trial, a solver should evaluate the trial in candidate storage and commit exactly the stored point it checked. Recomputing the point with slightly different rounding can mean that the gradient or objective belongs to a neighbor rather than the accepted point. Optim’s L-BFGS implementation and the behavior of LineSearches illustrate this distinction. They are precedents for a *caller* of the oracles, not a reason to forbid all infinite values inside the oracles. [[X3](#ref-X3)]

**Distinguish an unmeasurably small step from convergence.** With finite-precision storage, a positive step can leave every coordinate unchanged. That may indicate stagnation. A NaN residual is likewise not a certificate of stationarity. A solver should report what its stopping test actually establishes. ProximalBase should supply usable values and points and describe exceptional outcomes; it should not decide a whole solver’s stopping policy.

> **Where Part III ends.** A maintainer reviewing an oracle should now be able to answer four questions: Which point was evaluated? In which type is each answer delivered? Which operands does each answer depend on? Is an exception part of the mathematics, part of the documented API, or a preventable numerical error? The planning brief follows; it does not add new public behavior.

<a id="handoff"></a>

## Prepare a plan, not a patch

*IV · BRIEF FOR THE PLANNING AGENT*

**Assignment.** Identify the smallest set of changes that would make the public promises above true and reviewable. Some paths already meet them. Some promises strengthen existing behavior. Some methods are deliberately unsupported. The plan must distinguish those three situations before it recommends changing code. Do not change production code, add dependencies, or announce conformance as part of this assignment.

Start by recording the checkout SHA. The baseline for this guide is `3f90a29`. At that revision the documented normal-range rule already covers scalar values; it is not a new universal guarantee for gradients. The ridge source converts the step and level to the destination type before computing its proximal weights; shared routines for scaled products and weighted squares already exist. Those are places to investigate, not proof that every ridge example fails. Issue #283 and its later comments are useful history; reproduce a case on the checkout you actually plan against. [[R1](#ref-R1), [R2](#ref-R2), [R5](#ref-R5)]

### Trace one case all the way through

Start with V3. Write down the caller’s input types and why the answer must be `0.5f0`. Follow the public call to the scalar arithmetic and then to the destination write. Identify the first place where the stored finite problem changes. Compare the primitive, the entrywise route, any supported structured route, and the combined return. Which code paths inherit a fix and which perform their own conversion? Report the distinction; do not manufacture a new route to make the inventory symmetrical.

Then try V4 and V6. They determine whether a coefficient-only fix actually retains the coefficient’s *action*. V7 and V9 test the two outputs of a combined call independently. Extend the inventory to L1 thresholding, L2 and grouped shrinkage only after identifying shared failures. Keep source inspection separate from runtime evidence.

> **Deliver five things.** A coverage map by family, oracle, type, layout, and backend; a gap register with a minimal example and independent expected answer; a compatibility decision list; small proposed change groups with tests and measurements; and the documentation or release notes justified by each group. Include deliberate non-changes. A maintainer should be able to decline an expensive extension without losing sight of the user guarantee it would leave uncovered.

<a id="simple"></a>

## Choose the least complicated correct design

*IV · BRIEF FOR THE PLANNING AGENT*

**Start by asking whether anything is broken.** When the existing path gives the right class, branch, type, and accuracy, keep it. A regression test can be the complete recommendation. Making all operator implementations look alike is not a numerical objective.

**When the risk is shared, decide once.** For a scalar ridge level and step shared by a vector, ask whether the exceptional arithmetic can be chosen before the coordinate loop. Compare that with running a conversion check on every element. A factor that fits may use an ordinary vector loop; a factor that does not fit may need a different way to act on the original coordinates. Do not store a rounded zero and then try to diagnose it downstream.

**Prefer a stable formula that answers the exact question.** Preserve the existing paired logistic value and derivative when it retains a small tail with one exponential. For KL divergence near equality, evaluate the small divergence against a stable independent reference rather than assuming that subtracting large, individually accurate terms produces an accurate small result. For a scaled square, a reliable `norm(x)` is useful but `norm(x)^2` is not a general fix. [[R2](#ref-R2), [R3](#ref-R3), [R4](#ref-R4), [X1](#ref-X1)]

**Use exceptional recovery where the trigger is reliable.** An overflowing sum of squares is easy to detect. The finite but inaccurate sum in V2 is not. Propose a justified bound or selective extra pass when needed. Preserve documented homogeneous fast-path results on a fixed backend. Leave the special meaning of masks and hard weights with their caller; a general finite product helper should not invent those semantics.

**A small abstraction has to earn its keep.** Reuse an existing private helper first. Introduce one for a familiar operation such as a scaled product or threshold comparison when two real uses need the same semantics. Give it explicit input assumptions and tests. Do not start with a custom numeric type, a policy hierarchy, a public `robust=true` switch, unconditional full-array widening, or `BigFloat` in every hot loop. A new runtime `BigFloat` fallback needs a named failure, bounded cost, and an explicit backend policy; existing justified specialized fallbacks need not be removed.

<a id="borrow"></a>

## What to borrow, and what to leave upstream

*IV · BRIEF FOR THE PLANNING AGENT*

**From LAPACK and BLAS, borrow the scaled representation.** `xLASSQ` is a useful way to think about a sum of squares: retain a scale together with scaled contributions, instead of materializing each original square. `xNRM2` answers a norm question. For ProximalBase, the weight and the factor of one half may still change whether the *final* answer fits. Adopt the idea without assuming that the output of a norm routine contains all the information needed later. [[X1](#ref-X1)]

**From LogExpFunctions.jl, borrow explicit endpoint tests.** `xlogy` distinguishes a zero times a singular logarithm from a NaN argument, and `logaddexp` handles equal infinities deliberately. Test different argument types and the returned type. Do not import the entire mathematical domain of a helper: a real logarithm can throw for a negative argument, whereas an optimization objective may owe the caller `+Inf` there. Natural promotion of a scalar return is also not the package’s destination-precision rule. [[X2](#ref-X2)]

**From LineSearches.jl and Optim.jl, borrow the distinction between a trial and an accepted point.** A trial can be rejected without destroying good state. Borrow that practice for solver-facing examples or internal iterative routines where it applies; do not put a blanket finiteness scan in every penalty. Related packages such as ProximalOperators.jl and ProximalAlgorithms.jl are good ordinary-range cross-checks, not independent references for extreme arithmetic. [[X3](#ref-X3), [X5](#ref-X5)]

**Decide explicitly between using, adapting, and not borrowing.** Prefer an existing in-package routine or an existing dependency when its behavior matches. For one small mathematical identity, an independent implementation with attribution to the method and original tests may be easiest to maintain. A maintained dependency is more attractive when it genuinely replaces several kernels. Before copying code or tests, record the exact upstream commit and file, check its applicable license, retain required notices, and document local changes. Do not infer a file’s license from a repository label alone. [[X2](#ref-X2), [X3](#ref-X3)]

> **A useful review sentence.** “This helper solves the finite product in V4, but the caller still decides what a genuine infinite weight means.” A proposed reuse should make that boundary at least as clear as the existing code does.

<a id="verify"></a>

## How to tell whether an answer is correct

*IV · BRIEF FOR THE PLANNING AGENT*

Choose the expected answer *before* comparing implementations. For finite algebraic cases, exact powers of two or rational arithmetic often suffice. For harder cases, compute an independent expression from the actual stored binary operands using `BigFloat` with a locally scoped precision. Start at 256 bits; increase precision until the relevant rounding and classification are stable. An unstable reference or cancellation at high precision is still a bad reference. Use an algebraically different expression, exact or interval bounds, or an independent solver when necessary. [[X6](#ref-X6)]

```julia
# A target test, not a report of a passing package run.
using Test, ProximalBase
λ, γ = ldexp(1.0, 160), ldexp(1.0, -160)
x, z = Float32[1], zeros(Float32, 1)
ref = setprecision(BigFloat, 256) do
    Float32(BigFloat(x[1]) / (1 + BigFloat(γ) * BigFloat(λ)))
end
@test ref === 0.5f0
@test prox!(z, SqrNormL2(λ), x, γ) === z
@test z[1] === ref
```

This test asks three independent questions: is the expected number well understood, does the in-place routine return the caller’s buffer, and was the correct number stored? For a combined call, also check the scalar’s type, class, and value. Do not use a single approximate comparison for all four properties. An absolute tolerance of `1e-8`, for example, would bless zero in V8 and V9, destroying the very distinction being tested.

Use exact checks for values known to be exact and for specified zero/tie decisions. Use ULP or justified relative bounds for smooth, well-conditioned scalar results. For small answers use an absolute scale that distinguishes the failure from the reference; for matrices use an appropriate feasibility or stationarity residual. V2’s two-ULP bound is specific to V2, not a package-wide tolerance. For NaNs, masks, sentinel infinities, and errors before mutation, write the expected behavior explicitly. A high-precision number cannot tell you what policy the package should adopt.

<a id="evidence"></a>

## Coverage, cost, and the decision to release

*IV · BRIEF FOR THE PLANNING AGENT*

**Choose informative tests rather than every permutation.** Cover homogeneous `Float32` and `Float64`, then vary input, model data, step, and destination independently in a small pairwise set. Add targeted interactions like V3 and V8. Test empty arrays, signed zeros, both infinities, active and masked NaNs, exact/adjacent thresholds, and neighbors of `floatmin` and `floatmax`. Compare disjoint and supported aliased buffers, sparse and dense expansions, scalar and supported array/group steps, and ordinary and fallback paths. Use actual hardware for each claimed device-mode guarantee.

**Cross-check identities only where they hold.** For positive finite α compare the prox of αf at step γ with that of f at step αγ; form the reference step without losing it to overflow. For proper closed convex functions, check Moreau’s decomposition, fixed points at minimizers, or nonexpansiveness. For a nonunique solution, check optimality and the stated tie choice. Do not demand invariance across a documented rounding stage or compare two methods sharing the same broken helper as the only evidence.

**Measure the real cost.** In the existing harness compare baseline and proposal on the same hardware, Julia/BLAS versions, thread settings, seeds, and cache state. Separate compilation from warmed runs; ordinary homogeneous, ordinary mixed, and forced fallback from one another. Include coordinate updates, short and large vectors, sparse active sets, and one solver workload. Report nanoseconds, percentages, and allocations. For devices, synchronize timed work and measure kernels, transfers, and device allocations where relevant. [[R9](#ref-R9)]

**Use 5% as a review trigger, not a latency guarantee.** Investigate repeatable slowdowns above that level on representative ordinary workloads. A single inconclusive measurement does not establish “no cost.” Request maintainer approval for a justified regression, new allocation, extra pass or synchronization, changed classification/tie, or change to promised ordinary bits. State the failing case, simpler options tried, and measured cost. Earlier timings for value checks are not measurements of a new gradient or GPU path. [[R1](#ref-R1), [R9](#ref-R9)]

**Release gate.** Label evidence *verified*, *observed failure*, *documented exception*, *not supported*, or *not run*. Record a source-level suspicion as unverified, not as a failure. Every advertised guarantee needs passing evidence in its named scope or a public, bounded exception with rationale, owner/approval or tracking issue, and a condition for revisiting it. Separate migration decisions from regressions and new features. Do not publish the target as already delivered.

<a id="finder"></a>

## Find a case by the question it answers

*REFERENCE · ACCEPTANCE EXAMPLES*

The original case labels are retained so a test, issue, or code review can refer to one specific example. Each link leads to its assumptions and derivation. In a test, construct the power 2ᵏ as `ldexp(one(T), k)` in the intended floating type; do not let integer exponentiation or an early cast alter the fixture.

### Finite arithmetic

| Case | Question | Expected result |
|---|---|---|
| [V1](#range) | The square overflows; does the weighted value? | 2¹⁹⁹, finite. |
| [V2](#range) | Can a finite sum omit significant terms? | 2⁻²³ + 2⁻⁷², within two ULPs. |
| [V3](#mixed) | Can two wide parameters act on a narrow vector? | Exactly `0.5f0`. |
| [V4](#coefficient) | Can a factor vanish but its product fit? | `Float32` result rounds to 2⁻⁶⁰. |
| [V5](#mixed) | Can a wide weight give a finite narrow gradient? | Exactly 2⁸⁰ in `Float32`. |
| [V6](#coefficient) | Is using `Float64` everywhere sufficient? | Not in this case; output rounds to 2⁻⁷⁰⁰. |
| [V7](#joint) | Does an infinite value imply an infinite gradient? | No: `+Inf` and `[4.0]`. |
| [V8](#threshold) | Which side of a threshold is the coordinate on? | Active; exactly 2⁻³⁰. |
| [V9](#joint) | Is the value at the ideal or stored point? | Stored point `1f0`; value 2⁻⁵⁷. |

### Special values and API behavior

| Case | What is tested | Distinction that must survive |
|---|---|---|
| [S1](#infinity) | Infeasible input, valid projection | `+Inf` at 2; zero at its projection, 1. |
| [S2](#weights) | Disabled versus active ridge at NaN | Constant value and identity copy are different. |
| [S3](#nan) | One NaN in L1 versus L2 | `[NaN,3]` versus `[NaN,NaN]`. |
| [S4](#weights) | Zero and hard weights | Free first coordinate, fixed second. |
| [S5](#loss) | Entropy and logistic endpoints | Domain violation, NaN, and a limit differ. |
| [S6](#exceptions) | `NuclearNorm(0)` on non-finite input | Identity output and `+Inf` joint scalar. |
| [S7](#steps) | Invalid coupled metric | Error before any output changes. |
| [S8](#storage) | Fallback with aliasing and threshold ties | In-place agrees; ties are not enlarged. |

<a id="policies"></a>

## Find a rule from Draft 1

*REFERENCE · POLICY MAP*

The letter codes below are for traceability; the guide does not require readers to memorize them. A “target” is a proposed obligation, not a certification of the package. A named baseline exception stays in effect until deliberately changed.

**N1–N3 · Scope and strength.** [Which answers are covered](#scope) sets the supported configurations (N1), assigns a tier to each output rather than a whole function (N2), and defines the strength of the language and the non-goals (N3).

**P1–P4 · Precision.** [Who chooses the precision?](#types) gives public result types (P1), stored-input accuracy (P2), and constructor responsibilities (P4). [The mixed ridge example](#mixed) explains why working precision can be private without arbitrary early narrowing (P3).

**F1–F5 · Finite arithmetic.** [Weighted squares](#range) covers normal-range protection (F1). [The small coefficient](#coefficient) distinguishes final underflow and overflow from an intermediate failure (F2). [L1 thresholding](#threshold) covers exact zero and ties (F3), appropriate error measures (F4), and existing bitwise promises (F5).

**E1–E3 and W1–W4 · Exceptional values and controls.** [Three decisions in order](#infinity) gives precedence (E1); [NaN dependence](#nan) gives its scope (E2); [the named exceptions](#exceptions) specify sentinel behavior (E3). [Weights](#weights) distinguishes the roles of zero and infinity (W1). [Steps](#steps) distinguishes scalar and coupled diagonal endpoints (W2–W3); [masked observations](#loss) explains family-specific mask precedence (W4).

**O1–O5 · Oracle agreement and mutation.** [Combined calls](#joint) distinguishes the gradient’s input (O1) from the stored proximal point (O2). [Two exceptions](#exceptions) specifies O3. [Layout and buffers](#storage) describes agreement across entry, vector, and alias paths (O4) and what must remain unchanged after validation fails (O5).

**A1–A5 · Data representation.** [Sparse storage](#storage) covers implicit zeros (A1) and multiplicity (A2). [Wrappers](#wrappers) covers domains, staged rounding, and cached data (A3–A5).

**B1–B5 · Execution and responsibility.** [Devices](#devices) covers supported backends and execution modes (B1–B2). [Matrix methods](#spectral) explains conditioning (B3); [the solver boundary](#solver) explains trial acceptance (B4). [The review gate](#evidence) explains costs and what evidence must precede a release (B5).

<a id="primary"></a>

## Where the package rules come from

*REFERENCE · PACKAGE SOURCES*

The links below identify the repository snapshot used in Draft 1. They are evidence for descriptions of baseline behavior, not proof that proposed extensions are implemented. In particular, read the interface together with the maintainer decisions in R5 before changing a spectral tolerance.


<a id="ref-R1"></a>

**R1 · Public interface.** [Precision, range, special inputs, aliasing and GPU support](https://github.com/mlakolar/ProximalBase.jl/blob/3f90a29f21bea5f442d1b1f25f67a8e972037e0f/docs/src/interface.md).

<a id="ref-R2"></a>

**R2 · Ridge and shared arithmetic.** [SqrNormL2 methods](https://github.com/mlakolar/ProximalBase.jl/blob/3f90a29f21bea5f442d1b1f25f67a8e972037e0f/src/functions/smooth.jl) · [scaled products and sums of squares](https://github.com/mlakolar/ProximalBase.jl/blob/3f90a29f21bea5f442d1b1f25f67a8e972037e0f/src/utils.jl) · [norm and group operators](https://github.com/mlakolar/ProximalBase.jl/blob/3f90a29f21bea5f442d1b1f25f67a8e972037e0f/src/functions/norms.jl).

<a id="ref-R3"></a>

**R3 · Entropy and binary likelihood.** [NegEntropy, KL and Bernoulli routines](https://github.com/mlakolar/ProximalBase.jl/blob/3f90a29f21bea5f442d1b1f25f67a8e972037e0f/src/functions/entropy.jl).

<a id="ref-R4"></a>

**R4 · GLM scalar losses.** [Paired logistic value and derivative](https://github.com/mlakolar/ProximalBase.jl/blob/3f90a29f21bea5f442d1b1f25f67a8e972037e0f/src/functions/glm.jl).

<a id="ref-R5"></a>

**R5 · Finite-value audit and maintainer decisions.** [Issue #283](https://github.com/mlakolar/ProximalBase.jl/issues/283) · [spectral mask and membership decisions](https://github.com/mlakolar/ProximalBase.jl/issues/283#issuecomment-6516442181).

<a id="ref-R6"></a>

**R6 · Spectral and log-determinant methods.** [Spectral wrappers](https://github.com/mlakolar/ProximalBase.jl/blob/3f90a29f21bea5f442d1b1f25f67a8e972037e0f/src/spectralfunction.jl) · [LogDetLoss](https://github.com/mlakolar/ProximalBase.jl/blob/3f90a29f21bea5f442d1b1f25f67a8e972037e0f/src/functions/logdet.jl).

<a id="ref-R7"></a>

**R7 · Structured and separable data.** [Separable kernels and sparse multiplicities](https://github.com/mlakolar/ProximalBase.jl/blob/3f90a29f21bea5f442d1b1f25f67a8e972037e0f/src/functions/separable.jl) · [iterate interface](https://github.com/mlakolar/ProximalBase.jl/blob/3f90a29f21bea5f442d1b1f25f67a8e972037e0f/docs/src/iterates.md).

<a id="ref-R8"></a>

**R8 · Wrapper semantics.** [Scaling and zero scaling](https://github.com/mlakolar/ProximalBase.jl/blob/3f90a29f21bea5f442d1b1f25f67a8e972037e0f/src/calculus.jl) · [compositions and regularization](https://github.com/mlakolar/ProximalBase.jl/blob/3f90a29f21bea5f442d1b1f25f67a8e972037e0f/src/composition.jl).

<a id="ref-R9"></a>

**R9 · Benchmarking and test infrastructure.** [Benchmark protocol](https://github.com/mlakolar/ProximalBase.jl/blob/3f90a29f21bea5f442d1b1f25f67a8e972037e0f/benchmark/README.md) · [oracle test utilities](https://github.com/mlakolar/ProximalBase.jl/blob/3f90a29f21bea5f442d1b1f25f67a8e972037e0f/src/testing.jl).

<a id="reading"></a>

## Where to study the numerical techniques

*REFERENCE · FURTHER READING*

These references explain techniques, tradeoffs, and comparison implementations. They do not set ProximalBase’s public policy. For range protection start with X1; for special-value tests start with X2; for the boundary between an oracle and a solver start with X3.


<a id="ref-X1"></a>

**X1 · Safe scaling.** [LAPACK DLASSQ](https://netlib.org/lapack/explore-html/d8/d76/group__lassq_gae8f40b0a34771b4f2d9c863de3af7be5.html) · [BLAS DNRM2](https://netlib.org/lapack/explore-html/d6/de0/dnrm2_8f90_source.html).

<a id="ref-X2"></a>

**X2 · LogExpFunctions.jl.** [Scalar kernels](https://github.com/JuliaStats/LogExpFunctions.jl/blob/8f755913f29036984d25402fa97e200ef57a26a8/src/basicfuns.jl) · [tests](https://github.com/JuliaStats/LogExpFunctions.jl/blob/8f755913f29036984d25402fa97e200ef57a26a8/test/basicfuns.jl) · [upstream license](https://github.com/JuliaStats/LogExpFunctions.jl/blob/8f755913f29036984d25402fa97e200ef57a26a8/LICENSE.md).

<a id="ref-X3"></a>

**X3 · Solver-side recovery.** [LineSearches backtracking](https://github.com/JuliaNLSolvers/LineSearches.jl/blob/823cc4f8cc24b7c40834f6f4456468f209af24a3/src/backtracking.jl) · [LineSearches license](https://github.com/JuliaNLSolvers/LineSearches.jl/blob/823cc4f8cc24b7c40834f6f4456468f209af24a3/LICENSE.md) · [Optim L-BFGS acceptance](https://github.com/JuliaNLSolvers/Optim.jl/blob/347f05731596955831693a728f87ee034a3aae05/src/multivariate/solvers/first_order/l_bfgs.jl) · [trial handoff](https://github.com/JuliaNLSolvers/Optim.jl/blob/347f05731596955831693a728f87ee034a3aae05/src/utilities/perform_linesearch.jl) · [Optim license](https://github.com/JuliaNLSolvers/Optim.jl/blob/347f05731596955831693a728f87ee034a3aae05/LICENSE.md).

<a id="ref-X4"></a>

**X4 · Execution modes.** [Julia performance tips](https://docs.julialang.org/en/v1/manual/performance-tips/) · [CUDA.jl fast math and subnormals](https://cuda.juliagpu.org/stable/tutorials/performance/).

<a id="ref-X5"></a>

**X5 · Related proximal packages.** [ProximalOperators squared norm](https://github.com/JuliaFirstOrder/ProximalOperators.jl/blob/master/src/functions/sqrNormL2.jl) · [ProximalAlgorithms backtracking](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl/blob/master/src/utilities/fb_tools.jl).

<a id="ref-X6"></a>

**X6 · High-precision references.** [Julia floating-point and arbitrary-precision arithmetic](https://docs.julialang.org/en/v1/manual/integers-and-floating-point-numbers/).
