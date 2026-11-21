# Selected verbatim research functions from PS #508, jobs 14431772 and 14431773.
# Source tar SHA-256 b70fea8a217e1dfd7a7511dd283916912ebd9bea53a1d6cb1dc1216000e40e65.
# The generators, cap policy, and legacy enclosure are used for replay only.

function synthetic(T, n, p, geometry)
    if geometry === :interior
        rng = Xoshiro(508031 + n + p)
        A = randn(rng, T, n, p) / sqrt(T(n))
        y = randn(rng, T, n)
        return A, y, T(16 * p)
    end
    A, y = zeros(T, n, p), zeros(T, n)
    A[1, 1] = one(T)
    y[1] = T(2)
    A, y, T(0.5)
end

function fixtures(T, visit)
    I2 = Matrix{T}(I, 2, 2)
    for power in (-20, 0, 20)
        y = fill(ldexp(one(T), power), 2)
        radius = ldexp(T(0.1), power)
        visit("tie_scale_$power", I2, y, radius, zeros(T, 2), one(T), true)
    end
    for tail in (eps(T) / 4, 2 * eps(T))
        y = T[1, tail]
        visit("stationary_tail_$(tail / eps(T))_eps", I2, y, one(T), copy(y), one(T), true)
    end
    y = T[0.25, -0.375]
    visit("interior", I2, y, one(T), zeros(T, 2), one(T), true)
    visit("feasible_boundary_start", I2, T[0.75, -0.5], one(T), T[0.5, -0.5], one(T), true)
    visit("materially_infeasible_start", I2, T[1, 1], one(T), T[0.75, 0.75], one(T), true)
    A = T[1 0.375 0.125 -0.125; 0.25 1 -0.25 0.125;
          -0.125 0.125 1 0.375; 0.125 -0.25 0.125 1]
    y = T[1, -0.75, 0.375, 0.5]
    visit("correlated_active", A, y, T(0.6), zeros(T, 4), one(T), false)
    visit("correlated_scaled", A, y, T(0.6), zeros(T, 4), T(0.25), false)
    visit("correlated_interior", A, y, T(4), zeros(T, 4), one(T), false)
    visit("correlated_change", A, y, T(1.5), T[-0.75, 0, 0.5, 0], one(T), false)
    x = T[0.75, 0.75]
    # Exact represented boundary with zero smooth gradient.
    visit("boxlike_boundary", I2, x, T(1.5), copy(x), one(T), true)
end

function loss_and_budgets(A, y, r, alpha, coefficient_relative, objective_relative)
    T = eltype(y)
    f = PB.LeastSquaresLoss(y, A)
    loss = alpha == one(T) ? f : PB.Scaled(f, alpha)
    M2 = Float64(alpha) * maximum(c -> sum(v -> abs2(Float64(v)), c), eachcol(A))
    response = sqrt(Float64(alpha)) * norm(Float64.(y))
    radius = Float64(r)
    scale = max(response, sqrt(M2) * radius)^2 / 2
    budget = objective_relative * scale / 2
    cost = sqrt(M2) * response + 1.5 * M2 * radius
    cap = min(radius, coefficient_relative * radius, iszero(cost) ? Inf : budget / cost)
    loss, cap, M2, budget
end

function opt_for_study(maxiter, callback=nothing)
    PS.ProximalOptions(; maxiter, callback, allow_f_increases=true,
                       ftol=1e-6, gradtol=1e-5, xtol=1e-8)
end

function trial_only(f, x, guard, method, options)
    state = PS.initial_state(method, options, f, guard, x)
    PS._solve_gradient!(state, f, guard, method, options)
end

function fixed_solve_options()
    PS.ProximalOptions(maxiter=12, ftol=0, gradtol=0, xtol=1e-300, allow_f_increases=true)
end

@inline function split_sum(a::Float64, b::Float64)
    s = a + b
    e = abs(a) >= abs(b) ? (a - s) + b : (b - s) + a
    s, e
end

@inline function add_lower(a, b)
    s, e = split_sum(a, b)
    e < 0.0 ? prevfloat(s) : s
end

@inline function add_upper(a, b)
    s, e = split_sum(a, b)
    e > 0.0 ? nextfloat(s) : s
end

# An enclosure of the exact stored mass minus the represented radius.
# The high word absorbs each positive term with an error-free transform.
# Each directed low word bounds the exact sum of transform errors.
function residual_interval(y, r)
    high, low, up = -Float64(r), 0.0, 0.0
    @inbounds for v in y
        high, error = split_sum(high, abs(Float64(v)))
        low = add_lower(low, error)
        up = add_upper(up, error)
    end
    add_lower(high, low), add_upper(high, up)
end

@inline function mass_enclosure(x, r)
    s = sum(v -> abs(Float64(v)), x)
    radius = Float64(r)
    s == 0 && return (-radius, -radius)
    nu2 = length(x) * eps(Float64)
    if !isfinite(s) || s > floatmax(Float64) / 2 || nu2 > 0.5
        return (-Inf, Inf)
    end
    err = nextfloat(nu2 * s)
    (add_lower(add_lower(s, -err), -radius), add_upper(add_upper(s, err), -radius))
end

@inline function screened_enclosure(x, r, cap)
    lo, hi = mass_enclosure(x, r)
    if hi <= 0 || lo > cap || (lo > 0 && hi <= cap)
        return lo, hi, false
    end
    # The directed residual enclosure is the read-only PB experiment D.
    if isfinite(lo) && isfinite(hi) && !Base.get_zero_subnormals()
        lo, hi = residual_interval(x, r)
    end
    lo, hi, true
end
