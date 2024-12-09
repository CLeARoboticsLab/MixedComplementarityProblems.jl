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
"""
function solve(
    ::InteriorPoint,
    mcp::PrimalDualMCP,
    θ::AbstractVector{<:Real};
    x₀ = zeros(mcp.unconstrained_dimension),
    y₀ = ones(mcp.constrained_dimension),
    s₀ = ones(mcp.constrained_dimension),
    tol = 1e-4,
    max_inner_iters = 20,
    max_outer_iters = 50,
)
    # Set up common memory.
    ∇F = get_result_buffer(mcp.∇F_z!)
    F = zeros(mcp.unconstrained_dimension + 2mcp.constrained_dimension)
    δz = zeros(mcp.unconstrained_dimension + 2mcp.constrained_dimension)
    δx = @view δz[1:(mcp.unconstrained_dimension)]
    δy =
        @view δz[(mcp.unconstrained_dimension + 1):(mcp.unconstrained_dimension + mcp.constrained_dimension)]
    δs = @view δz[(mcp.unconstrained_dimension + mcp.constrained_dimension + 1):end]

    # Main solver loop.
    x = x₀
    y = y₀
    s = s₀
    ϵ = 1.0
    kkt_error = Inf
    status = :solved
    outer_iters = 1
    while kkt_error > tol && ϵ > tol && outer_iters < max_outer_iters
        inner_iters = 1
        status = :solved

        while kkt_error > ϵ && inner_iters < max_inner_iters
            # Compute the Newton step.
            # TODO! Can add some adaptive regularization.
            mcp.F!(F, x, y, s; θ, ϵ)
            mcp.∇F_z!(∇F, x, y, s; θ, ϵ)
            LinearAlgebra.ldiv!(
                δz,
                LinearAlgebra.qr(-collect(∇F), LinearAlgebra.ColumnNorm()),
                collect(F),
            )

            # Fraction to the boundary linesearch.
            α_s = fraction_to_the_boundary_linesearch(s, δs; tol)
            α_y = fraction_to_the_boundary_linesearch(y, δy; tol)

            if isnan(α_s) || isnan(α_y)
                @warn "Linesearch failed. Exiting prematurely."
                status = :failed
                break
            end

            # Update variables accordingly.
            x += α_s * δx
            s += α_s * δs
            y += α_y * δy

            kkt_error = maximum(abs.(F))
            inner_iters += 1
        end

        ϵ *= (status == :solved) ? 1 - exp(-inner_iters) : 1 + exp(-inner_iters)
        outer_iters += 1
    end

    if outer_iters == max_outer_iters
        status = :failed
    end

    (; status, x, y, s, kkt_error, ϵ, outer_iters)
end

"""Helper function to compute the step size `α` which solves:
                   α* = max(α ∈ [0, 1] : v + α δ ≥ (1 - τ) v).
"""
function fraction_to_the_boundary_linesearch(v, δ; τ = 0.995, decay = 0.5, tol = 1e-4)
    α = 1.0
    while any(v + α * δ .< (1 - τ) * v)
        if α < tol
            return NaN
        end

        α *= decay
    end

    α
end
