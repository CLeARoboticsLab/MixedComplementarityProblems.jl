""" Support for automatic differentiation of an MCP's solution (x, y) with respect
 to its parameters θ. Since a solution satisfies
                            F(z; θ, ϵ) = 0
for the primal-dual system, the derivative we are looking for is given by
                            ∂z∂θ = -(∇F_z)⁺ ∇F_θ.

Modifed from https://github.com/JuliaGameTheoreticPlanning/ParametricMCPs.jl/blob/main/src/AutoDiff.jl.
"""

module AutoDiff

using ..MixedComplementarityProblems: MixedComplementarityProblems
using ChainRulesCore: ChainRulesCore
using ForwardDiff: ForwardDiff
using LinearAlgebra: LinearAlgebra

function _solve_jacobian_θ(mcp::MixedComplementarityProblems.PrimalDualMCP, solution, θ)
    !isnothing(mcp.∇F_θ!) || throw(
        ArgumentError(
            "Missing sensitivities. Set `compute_sensitivities = true` when constructing the PrimalDualMCP.",
        ),
    )

    (; x, y, s, ϵ) = solution

    ∇F_z = let
        ∇F = MixedComplementarityProblems.get_result_buffer(mcp.∇F_z!)
        mcp.∇F_z!(∇F, x, y, s; θ, ϵ)
        ∇F
    end

    ∇F_θ = let
        ∇F = MixedComplementarityProblems.get_result_buffer(mcp.∇F_θ!)
        mcp.∇F_θ!(∇F, x, y, s; θ, ϵ)
        ∇F
    end

    LinearAlgebra.qr(-collect(∇F_z), LinearAlgebra.ColumnNorm()) \ collect(∇F_θ)
end

function ChainRulesCore.rrule(
    ::typeof(MixedComplementarityProblems.solve),
    solver_type::MixedComplementarityProblems.SolverType,
    mcp::MixedComplementarityProblems.PrimalDualMCP,
    θ;
    kwargs...,
)
    solution = MixedComplementarityProblems.solve(solver_type, mcp, θ; kwargs...)
    project_to_θ = ChainRulesCore.ProjectTo(θ)

    function solve_pullback(∂solution)
        no_grad_args = (;
            ∂self = ChainRulesCore.NoTangent(),
            ∂solver_type = ChainRulesCore.NoTangent(),
            ∂mcp = ChainRulesCore.NoTangent(),
        )

        ∂θ = ChainRulesCore.@thunk let
            ∂z∂θ = _solve_jacobian_θ(mcp, solution, θ)
            ∂l∂x = ∂solution.x
            ∂l∂y = ∂solution.y
            ∂l∂s = ∂solution.s

            @views project_to_θ(
                ∂z∂θ[1:(mcp.unconstrained_dimension), :]' * ∂l∂x +
                ∂z∂θ[
                    (mcp.unconstrained_dimension + 1):(mcp.unconstrained_dimension + mcp.constrained_dimension),
                    :,
                ]' * ∂l∂y +
                ∂z∂θ[
                    (mcp.unconstrained_dimension + mcp.constrained_dimension + 1):end,
                    :,
                ]' * ∂l∂s,
            )
        end

        no_grad_args..., ∂θ
    end

    solution, solve_pullback
end

function MixedComplementarityProblems.solve(
    solver_type::MixedComplementarityProblems.InteriorPoint,
    mcp::MixedComplementarityProblems.PrimalDualMCP,
    θ::AbstractVector{<:ForwardDiff.Dual{T}};
    kwargs...,
) where {T}
    # strip off the duals
    θ_v = ForwardDiff.value.(θ)
    θ_p = ForwardDiff.partials.(θ)
    # forward pass
    solution = MixedComplementarityProblems.solve(solver_type, mcp, θ_v; kwargs...)
    # backward pass
    ∂z∂θ = _solve_jacobian_θ(mcp, solution, θ_v)
    # downstream gradient
    z_p = ∂z∂θ * θ_p
    # glue forward and backward pass together into dual number types
    x_d = ForwardDiff.Dual{T}.(solution.x, @view z_p[1:(mcp.unconstrained_dimension)])
    y_d =
        ForwardDiff.Dual{
            T,
        }.(
            solution.y,
            @view z_p[(mcp.unconstrained_dimension + 1):(mcp.unconstrained_dimension + mcp.constrained_dimension)]
        )
    s_d =
        ForwardDiff.Dual{
            T,
        }.(
            solution.y,
            @view z_p[(mcp.unconstrained_dimension + mcp.constrained_dimension + 1):end]
        )

    (; solution.status, solution.kkt_error, solution.ϵ, x = x_d, y = y_d, s = s_d)
end

end
