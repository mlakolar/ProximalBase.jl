# Optional suite: deliberately not included by test/runtests.jl.
module ConicOracles

using Test, LinearAlgebra, Random
using ProximalBase, JuMP, Hypatia
import MathOptInterface as MOI

const SOLVER_TOL = 1e-9
const RETRY_TOL = 1e-8
const FEAS_TOL = 2e-7
const OBJECTIVE_TOL = 2e-7
const SOLUTION_TOL = 5e-5
const KKT_TOL = 5e-5
const SEED = 1042026

conic_optimizer(tol) = Hypatia.Optimizer(; verbose=false, iter_limit=200,
    tol_rel_opt=tol, tol_abs_opt=tol, tol_feas=tol)

function oracle_model(A)
    n = size(A, 1)
    model = Model(() -> conic_optimizer(SOLVER_TOL))
    @variable(model, X[1:n, 1:n], Symmetric)
    @variable(model, q)
    # 2*q*1 >= ||X-A||_F^2; include BOTH triangles in the Frobenius norm.
    @constraint(model, [q; 1.0; vec(X - A)] in MOI.RotatedSecondOrderCone(n^2 + 2))
    model, X, q
end

function solve_or_fail!(model)
    optimize!(model)
    status = (termination_status(model), primal_status(model), dual_status(model))
    # One explicit retry for a nearly optimal solve; independent checks below
    # retain the same thresholds. Never compare values from the rejected solve.
    if termination_status(model) == MOI.ALMOST_OPTIMAL
        println("ORACLE_RETRY rejected_status=$status retry_tol=$RETRY_TOL")
        # Hypatia cannot restart a completed solver; reload the model into a
        # fresh optimizer rather than reusing its terminated state.
        set_optimizer(model, () -> conic_optimizer(RETRY_TOL))
        optimize!(model)
        status = (termination_status(model), primal_status(model), dual_status(model))
    end
    # Do not read values after a failed, nearly optimal, or inaccurate solve.
    status == (MOI.OPTIMAL, MOI.FEASIBLE_POINT, MOI.FEASIBLE_POINT) ||
        error("Unacceptable oracle status: $status; $(raw_status(model))")
    status
end

function logdet_value(X, S, w)
    w * (dot(S, X) - logdet(Symmetric(X)))
end

function logdet_case(A, S, w, γ)
    n = size(A, 1)
    model, X, q = oracle_model(A)
    @variable(model, t)
    # Scaled upper-triangle packing: off-diagonal coefficients carry sqrt(2).
    packed = [i == j ? 1.0 * X[i, j] : sqrt(2.0) * X[i, j]
              for j in 1:n for i in 1:j]
    @constraint(model, [t; 1.0; packed] in MOI.Scaled(MOI.LogDetConeTriangle(n)))
    @objective(model, Min, q / γ + w * (sum(S .* X) - t))
    status = solve_or_fail!(model)
    H = Matrix(JuMP.value.(X))
    Y = similar(A)
    returned = value_and_prox!(Y, LogDetLoss(S, w), A, γ)
    @test isposdef(Symmetric(H))
    @test isposdef(Symmetric(Y))
    @test returned ≈ logdet_value(Y, S, w) atol=1e-10 rtol=1e-10
    objective(Z) = logdet_value(Z, S, w) + sum(abs2, Z - A) / (2γ)
    @test objective(H) ≈ objective_value(model) atol=OBJECTIVE_TOL rtol=OBJECTIVE_TOL
    @test objective(Y) ≈ objective(H) atol=OBJECTIVE_TOL rtol=OBJECTIVE_TOL
    @test norm(Y - H) <= SOLUTION_TOL * max(1, norm(H))
    residual(Z) = norm((Z - A) / γ + w * (S - inv(Symmetric(Z)))) /
                  max(1, norm((Z - A) / γ), w * norm(S))
    @test residual(H) <= KKT_TOL
    @test residual(Y) <= 1e-10
    println("CASE logdet n=$n w=$w gamma=$γ status=$status",
        " objective_error=", abs(objective(Y) - objective(H)),
        " solution_error=", norm(Y - H), " oracle_kkt=", residual(H),
        " package_kkt=", residual(Y))
end

# All constraints here are convex. No rank constraint is certified by this suite.
function spectral_case(A, kind, γ; mass=1.0)
    n = size(A, 1)
    lo, hi = kind == :box ? (-0.3, 0.8) : (0.0, kind == :fantope ? 1.0 : Inf)
    λ = kind == :trace_penalty ? 0.4 : 0.0
    f = if kind == :psd
        IndPSD()
    elseif kind == :box
        Spectral(IndBox(lo, hi))
    elseif kind == :trace_penalty
        TracePSD(λ)
    elseif kind == :simplex
        Spectral(IndSimplex(mass))
    elseif kind == :trace_ball
        Spectral(IndSimplex(mass; equality=false))
    elseif kind == :fantope
        Spectral(Constrained(IndBox(0, 1), IndHyperplane(1, mass)))
    else
        error("Unknown spectral case: $kind")
    end
    model, X, q = oracle_model(A)
    lower = @constraint(model, X - lo * I in PSDCone())
    upper = isfinite(hi) ? @constraint(model, hi * I - X in PSDCone()) : nothing
    trace_constraint = if kind in (:simplex, :fantope)
        @constraint(model, tr(X) == mass)
    elseif kind == :trace_ball
        @constraint(model, tr(X) <= mass)
    else
        nothing
    end
    @objective(model, Min, q / γ + λ * tr(X))
    status = solve_or_fail!(model)
    H = Matrix(JuMP.value.(X))
    Y = similar(A)
    returned = value_and_prox!(Y, f, A, γ)
    @test returned ≈ λ * tr(Y) atol=1e-10 rtol=1e-10
    objective(Z) = λ * tr(Z) + sum(abs2, Z - A) / (2γ)
    @test objective(H) ≈ objective_value(model) atol=OBJECTIVE_TOL rtol=OBJECTIVE_TOL
    @test objective(Y) ≈ objective(H) atol=OBJECTIVE_TOL rtol=OBJECTIVE_TOL
    @test norm(Y - H) <= SOLUTION_TOL * max(1, norm(H))

    L = Matrix(dual(lower))
    U = upper === nothing ? zeros(n, n) : Matrix(dual(upper))
    ν = trace_constraint === nothing ? 0.0 : dual(trace_constraint)
    @test eigmin(Symmetric(L)) >= -FEAS_TOL
    @test eigmin(Symmetric(U)) >= -FEAS_TOL
    kind == :trace_ball && (@test ν <= FEAS_TOL)
    residuals = Float64[]
    for Z in (H, Y)
        @test norm(Z - Z') <= FEAS_TOL
        @test eigmin(Symmetric(Z)) >= lo - FEAS_TOL
        isfinite(hi) && (@test eigmax(Symmetric(Z)) <= hi + FEAS_TOL)
        if kind in (:simplex, :fantope)
            @test abs(tr(Z) - mass) <= FEAS_TOL
        elseif kind == :trace_ball
            @test tr(Z) <= mass + FEAS_TOL
            @test abs(ν * (tr(Z) - mass)) <= KKT_TOL * max(1, norm(Z))
        end
        # MOI dual convention: gradient - L + U - nu*I = 0.
        residual = norm((Z - A) / γ + (λ - ν) * I - L + U) /
                   max(1, norm((Z - A) / γ), norm(L), norm(U), abs(ν))
        push!(residuals, residual)
        @test residual <= KKT_TOL
        @test abs(dot(L, Z - lo * I)) <= KKT_TOL * max(1, norm(L) * norm(Z))
        isfinite(hi) && (@test abs(dot(U, hi * I - Z)) <=
                                  KKT_TOL * max(1, norm(U) * norm(Z)))
    end
    println("CASE $kind n=$n gamma=$γ mass=$mass status=$status",
        " objective_error=", abs(objective(Y) - objective(H)),
        " solution_error=", norm(Y - H), " kkt=", residuals)
end

# These inequalities are independent of the package's membership kernels.
function expcone_feasible(x, dual, tol)
  r, s, t = dual ? (-x[1], -x[2], -x[3]) : x
  if dual
    return (abs(r) <= tol && s <= tol && t <= tol) ||
      (r > 0 && t <= tol && tol - t > 0 && s <= r * (1 + log((tol - t) / r)) + tol)
  end
  (abs(s) <= tol && r <= tol && t >= -tol) ||
    (s > 0 && t >= -tol && t + tol > 0 && r <= s * log((t + tol) / s) + tol)
end

function expcone_case(x, dual)
  model = Model(() -> conic_optimizer(SOLVER_TOL))
  @variable(model, z[1:3])
  @variable(model, q)
  @constraint(model, [q; 1.0; z - x] in MOI.RotatedSecondOrderCone(5))
  # The linear map represents -u exp(v/u) <= e*w directly, with no
  # projection or Moreau formula from the implementation under test.
  cone = if dual
    @constraint(model, [-z[2], -z[1], exp(1.0) * z[3]] in MOI.ExponentialCone())
  else
    @constraint(model, z in MOI.ExponentialCone())
  end
  @objective(model, Min, q)
  status = solve_or_fail!(model)
  h = JuMP.value.(z)
  multiplier = JuMP.dual(cone)
  normal = dual ? [-multiplier[2], -multiplier[1], exp(1.0) * multiplier[3]] : multiplier
  f = dual ? IndExpDual() : IndExpPrimal()
  y = similar(x)
  @test value_and_prox!(y, f, x, 0.7) === 0.0
  objective(v) = sum(abs2, v - x) / 2
  @test objective(h) ≈ objective_value(model) atol=OBJECTIVE_TOL rtol=OBJECTIVE_TOL
  @test objective(y) ≈ objective(h) atol=OBJECTIVE_TOL rtol=OBJECTIVE_TOL
  @test norm(y - h) <= SOLUTION_TOL * max(1, norm(h))
  magnitude = max(1, norm(x))
  for v in (h, y)
    @test expcone_feasible(v / magnitude, dual, FEAS_TOL)
    @test expcone_feasible((v - x) / magnitude, !dual, FEAS_TOL)
    @test abs(dot(v, v - x)) / magnitude^2 <= KKT_TOL
  end
  @test expcone_feasible(normal / magnitude, !dual, FEAS_TOL)
  @test norm(h - x - normal) / magnitude <= KKT_TOL
  @test abs(dot(h, normal)) / magnitude^2 <= KKT_TOL
  println("CASE expcone dual=$dual x=$x status=$status",
    " objective_error=", abs(objective(y) - objective(h)),
    " solution_error=", norm(y - h),
    " oracle_kkt=", norm(h - x - normal) / magnitude)
end

function run_expcone_oracles()
  rng = Xoshiro(1712026)
  println("EXP_ORACLE seed=1712026 feasibility_tol=$FEAS_TOL",
    " solution_tol=$SOLUTION_TOL kkt_tol=$KKT_TOL")
  cases = [[0.0, 1, 2], [-1.0, 0, 1], [0.0, 0, 0],
           [1.0, 1, 1], [0.01, -8, 8], [-8.0, 0.01, -8]]
  for delta in (-1e-6, 0.0, 1e-6)
    append!(cases, [[-1.0, delta, 1], [delta, 1.0, 1], [1.0, 1, delta]])
  end
  append!(cases, [randn(rng, 3) for _ in 1:20])
  @testset "Independent Hypatia exponential cone projections" begin
    for dual in (false, true), (i, v) in enumerate(cases)
      scale = (0.1, 1.0, 10.0)[mod1(i, 3)]
      @testset "dual=$dual case=$i scale=$scale" begin
        expcone_case(scale .* v, dual)
      end
    end
  end
end

function run_oracles()
    LinearAlgebra.BLAS.set_num_threads(1)
    root = normpath(joinpath(@__DIR__, "../.."))
    commit = readchomp(`git -C $root rev-parse HEAD`)
    dirty = !isempty(readchomp(`git -C $root status --porcelain --untracked-files=normal`))
    println("ORACLE commit=$commit dirty=$dirty julia=$VERSION",
        " Hypatia=", Base.pkgversion(Hypatia), " JuMP=", Base.pkgversion(JuMP),
        " MOI=", Base.pkgversion(MOI), " seed=$SEED solver_tol=$SOLVER_TOL")
    rng = Xoshiro(SEED)
    @testset "Independent Hypatia primitive oracles" begin
        for n in (2, 3, 5)
            Q = Matrix(qr(randn(rng, n, n)).Q)
            B = randn(rng, n + 3, n)
            S = Matrix(Symmetric(B' * B / (n + 3) + I))
            # Well-conditioned diagonal, seeded rotated, and feasible near-boundary inputs.
            inputs = (Matrix(Diagonal(range(-0.5, 1.5; length=n))),
                      Matrix(Symmetric(Q * Diagonal(range(-1, 2; length=n)) * Q')),
                      Matrix(Symmetric(Q * Diagonal([1e-6; fill(0.5, n - 1)]) * Q')))
            for (j, A) in enumerate(inputs)
                @testset "n=$n case=$j" begin
                    γ = (0.3, 1.0, 1.7)[j]
                    logdet_case(A, S, (0.5, 1.0, 2.0)[j], γ)
                    for kind in (:psd, :box, :trace_penalty, :simplex, :trace_ball, :fantope)
                        spectral_case(A, kind, γ)
                    end
                end
            end
            # Already feasible points close to the simplex/Fantope boundary.
            D = Matrix(Symmetric(Q * Diagonal([1 - 1e-6; fill(1e-6 / (n - 1), n - 1)]) * Q'))
            for kind in (:simplex, :trace_ball, :fantope)
                spectral_case(D, kind, 0.7)
            end
            # Nonintegral Fantope trace; this is not a rank-constrained set.
            spectral_case(inputs[2], :fantope, 0.7; mass=1.5)
        end
    end
    run_expcone_oracles()
end

run_oracles()

end
