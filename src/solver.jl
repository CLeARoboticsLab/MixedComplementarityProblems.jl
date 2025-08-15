abstract type SolverType end
struct InteriorPoint <: SolverType end

""" Basic interior point solver, based on Nocedal & Wright, ch. 19.
Computes step directions `δz` by solving the relaxed primal-dual system, i.e.
                         ∇F(z; ϵ) δz = -F(z; ϵ).

Given a step direction `δz`, performs a "fraction to the boundary" linesearch,
i.e., for `(x, s)` it chooses step size `α_s` such that
              α_s = max(α ∈ [0, 1] : s + α δs ≥ (1 - τ) s)
and for `y` it chooses step size `α_s` such that
              α_y = max(α ∈ [0, 1] : y + α δy ≥ (1 - τ) y).

A typical value of τ is 0.995. Once we converge to ||F(z; \epsilon)|| ≤ ϵ,
we typically decrease ϵ by a factor of 0.1 or 0.2, with smaller values chosen
when the previous subproblem is solved in fewer iterations.

Positional arguments:
    - `mcp::PrimalDualMCP`: the mixed complementarity problem to solve.
    - `θ::AbstractVector{<:Real}`: the parameter vector.

Keyword arguments:
    - `x₀::AbstractVector{<:Real}`: the initial primal variable.
    - `y₀::AbstractVector{<:Real}`: the initial dual variable.
    - `s₀::AbstractVector{<:Real}`: the initial slack variable.
    - `ϵ₀::Real`: the initial relaxation scale.
    - `tol::Real = 1e-4`: the tolerance for the KKT error.
    - `max_inner_iters::Int = 20`: the maximum number of inner iterations.
    - `max_outer_iters::Int = 50`: the maximum number of outer iterations.
    - `tightening_rate::Real = 0.1`: rate for tightening tolerance and regularization.
    - `loosening_rate::Real = 0.5`: rate for loosening tolerance and regularization.
    - `min_stepsize::Real = 1e-2`: the minimum step size for the linesearch.
    - `verbose::Bool = false`: whether to print debug information.
    - `linear_solve_algorithm::LinearSolve.SciMLLinearSolveAlgorithm`: the linear solve algorithm to use. Any solver from `LinearSolve.jl` can be used.
    - `regularize_linear_solve::Symbol = :none`: scheme for regularizing the linear system matrix ∇F. Options are {:none, :identity, :internal}.
"""
function solve(
    ::InteriorPoint,
    mcp::PrimalDualMCP,
    θ::AbstractVector{<:Real};
    x₀ = nothing,
    y₀ = nothing,
    s₀ = nothing,
    tol = 1e-4,
    ϵ₀ = :auto,
    max_inner_iters = 20,
    max_outer_iters = 50,
    tightening_rate = 0.1,
    loosening_rate = 0.5,
    min_stepsize = 1e-4,
    verbose = false,
    linear_solve_algorithm = UMFPACKFactorization(),
    regularize_linear_solve = :identity,
)
    # Set up common memory.
    ∇F = mcp.∇F_z!.result_buffer
    F = zeros(mcp.unconstrained_dimension + 2mcp.constrained_dimension)
    δz = zeros(mcp.unconstrained_dimension + 2mcp.constrained_dimension)
    δx = @view δz[1:(mcp.unconstrained_dimension)]
    δy =
        @view δz[(mcp.unconstrained_dimension + 1):(mcp.unconstrained_dimension + mcp.constrained_dimension)]
    δs = @view δz[(mcp.unconstrained_dimension + mcp.constrained_dimension + 1):end]

    linsolve = init(LinearProblem(∇F, δz), linear_solve_algorithm)

    # Initialize primal, dual, and slack variables.
    x = @something(x₀, zeros(mcp.unconstrained_dimension))
    y = @something(y₀, ones(mcp.constrained_dimension))
    s = @something(s₀, ones(mcp.constrained_dimension))

    # Initialize IP relaxation parameter.
    if ϵ₀ === :auto
        is_warmstarted = !isnothing(x₀) && !isnothing(y₀) && !isnothing(s₀)
        if is_warmstarted
            ϵ = tol
        else
            ϵ = one(tol)
        end
    else
        ϵ = ϵ₀
    end

    # Initialize regularization parameter.
    η = tol

    # Main solver loop.
    status = :solved
    total_iters = 0
    inner_iters = 1
    outer_iters = 1
    kkt_error = Inf
    while outer_iters < max_outer_iters || iszero(total_iters)
        inner_iters = 1
        status = :solved

        while kkt_error > ϵ && inner_iters < max_inner_iters
            total_iters += 1

            # Compute the (regularized) Newton step.
            # TODO: use a linear operator with a lazy gradient computation here.
            if regularize_linear_solve === :internal
                mcp.F!(F, x, y, s; θ, ϵ, η = 0.0)
                mcp.∇F_z!(∇F, x, y, s; θ, ϵ, η)
            else
                mcp.F!(F, x, y, s; θ, ϵ)
                mcp.∇F_z!(∇F, x, y, s; θ, ϵ)
            end

            if regularize_linear_solve === :identity
                if size(∇F, 1) == size(∇F, 2)
                    linsolve.A = ∇F + η * I
                else
                    @warn "Cannot use identity regularization on a nonsquare problem."
                end
            else
                linsolve.A = ∇F
            end

            linsolve.b = -F
            solution = solve!(linsolve)

            if !SciMLBase.successful_retcode(solution) &&
               (solution.retcode !== SciMLBase.ReturnCode.Default)
                verbose &&
                    @warn "Linear solve failed. Exiting prematurely. Return code: $(solution.retcode)"
                status = :failed
                break
            end

            δz .= solution.u

            # Fraction to the boundary linesearch.
            α_s = fraction_to_the_boundary_linesearch(s, δs; tol = min_stepsize)
            α_y = fraction_to_the_boundary_linesearch(y, δy; tol = min_stepsize)

            if isnan(α_s) || isnan(α_y)
                verbose && @warn "Linesearch failed. Exiting prematurely."
                status = :failed
                break
            end

            # Update variables accordingly.
            @. x += α_s * δx
            @. s += α_s * δs
            @. y += α_y * δy

            kkt_error = norm(F, Inf)
            inner_iters += 1
        end

        if kkt_error <= ϵ <= tol
            break
        end

        if status === :solved
            ϵ *= 1 - exp(-tightening_rate * inner_iters)
            η *= 1 - exp(-tightening_rate * inner_iters)
        else
            ϵ *= 1 + exp(-loosening_rate * inner_iters)
            η *= 1 + exp(-loosening_rate * inner_iters)
        end
        ϵ = min(ϵ, one(ϵ))
        outer_iters += 1
    end

    if outer_iters == max_outer_iters
        status = :failed
    end

    (; status, x, y, s, kkt_error, ϵ, outer_iters, total_iters)
end

"""Helper function to compute the step size `α` which solves:
                   α* = max(α ∈ [0, 1] : v + α δ ≥ (1 - τ) v).
"""
function fraction_to_the_boundary_linesearch(v, δ; τ = 0.995, decay = 0.5, tol = 1e-4)
    α = 1.0
    while any(@. v + α * δ < (1 - τ) * v)
        if α < tol
            return NaN
        end

        α *= decay
    end

    α
end
