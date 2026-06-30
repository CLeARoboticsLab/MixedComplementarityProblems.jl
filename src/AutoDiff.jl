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
using KernelAbstractions: KernelAbstractions
using LinearAlgebra: LinearAlgebra
using SymbolicTracingUtils: SymbolicTracingUtils

function _solve_jacobian_θ(mcp::MixedComplementarityProblems.PrimalDualMCP, solution, θ)
    !isnothing(mcp.∇F_θ!) || throw(
        ArgumentError(
            "Missing sensitivities. Set `compute_sensitivities = true` when constructing the PrimalDualMCP.",
        ),
    )

    (; x, y, s, ϵ) = solution

    ∇F_z = let
        ∇F = mcp.∇F_z!.result_buffer
        mcp.∇F_z!(∇F, x, y, s; θ, ϵ)
        ∇F
    end

    ∇F_θ = let
        ∇F = mcp.∇F_θ!.result_buffer
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

# ---------------------------------------------------------------------------
# Batched solver (BatchedInteriorPoint): the same sensitivity machinery, but
# applied per instance over the batch axis. `Θ` is `(nθ × B)`; the solution
# fields `x, y, s` are `(· × B)` and `∂z∂θ` is `(d × nθ × B)`.
# ---------------------------------------------------------------------------

"Per-instance loss cotangent ∂Θ from a batched solution sensitivity `∂z∂θ`."
function _batched_∂Θ(mcp, ∂z∂θ, ∂solution, solution)
    nx = mcp.unconstrained_dimension
    ny = mcp.constrained_dimension
    B = size(∂z∂θ, 3)
    nθ = size(∂z∂θ, 2)

    # Downstream cotangents may be structurally zero for unused fields; treat
    # those as numeric zeros so the per-instance contractions stay well-typed.
    aszero(g, ref) = g isa ChainRulesCore.AbstractZero ? zero(ref) : g
    ∂l∂x = aszero(∂solution.x, solution.x)
    ∂l∂y = aszero(∂solution.y, solution.y)
    ∂l∂s = aszero(∂solution.s, solution.s)

    ∂Θ = similar(∂z∂θ, nθ, B)
    for b in 1:B
        @views ∂Θ[:, b] =
            ∂z∂θ[1:nx, :, b]' * ∂l∂x[:, b] +
            ∂z∂θ[(nx + 1):(nx + ny), :, b]' * ∂l∂y[:, b] +
            ∂z∂θ[(nx + ny + 1):end, :, b]' * ∂l∂s[:, b]
    end
    ∂Θ
end

function ChainRulesCore.rrule(
    ::typeof(MixedComplementarityProblems.solve),
    solver_type::MixedComplementarityProblems.BatchedInteriorPoint,
    mcp::MixedComplementarityProblems.PrimalDualMCP,
    Θ::AbstractMatrix;
    strategy = MixedComplementarityProblems.BatchedSparse(),
    device = KernelAbstractions.CPU(),
    kwargs...,
)
    solution =
        MixedComplementarityProblems.solve(solver_type, mcp, Θ; strategy, device, kwargs...)
    project_to_Θ = ChainRulesCore.ProjectTo(Θ)

    function solve_pullback(∂solution)
        no_grad_args = (;
            ∂self = ChainRulesCore.NoTangent(),
            ∂solver_type = ChainRulesCore.NoTangent(),
            ∂mcp = ChainRulesCore.NoTangent(),
        )

        ∂Θ = ChainRulesCore.@thunk let
            ∂z∂θ = MixedComplementarityProblems.solve_jacobian_θ(
                mcp,
                solution.x,
                solution.y,
                solution.s,
                Θ,
                solution.ϵ;
                strategy,
                device,
            )
            project_to_Θ(_batched_∂Θ(mcp, ∂z∂θ, ∂solution, solution))
        end

        no_grad_args..., ∂Θ
    end

    solution, solve_pullback
end

function MixedComplementarityProblems.solve(
    solver_type::MixedComplementarityProblems.BatchedInteriorPoint,
    mcp::MixedComplementarityProblems.PrimalDualMCP,
    Θ::AbstractMatrix{<:ForwardDiff.Dual{T}};
    strategy = MixedComplementarityProblems.BatchedSparse(),
    device = KernelAbstractions.CPU(),
    kwargs...,
) where {T}
    # strip off the duals
    Θ_v = ForwardDiff.value.(Θ)
    Θ_p = ForwardDiff.partials.(Θ)

    # forward pass
    solution =
        MixedComplementarityProblems.solve(solver_type, mcp, Θ_v; strategy, device, kwargs...)

    # backward pass
    ∂z∂θ = MixedComplementarityProblems.solve_jacobian_θ(
        mcp,
        solution.x,
        solution.y,
        solution.s,
        Θ_v,
        solution.ϵ;
        strategy,
        device,
    )
    nx = mcp.unconstrained_dimension
    ny = mcp.constrained_dimension
    B = size(Θ_v, 2)
    # Propagate parameter partials through each instance: z_p[:, b] = ∂z∂θ[:,:,b] Θ_p[:, b].
    z_p = reduce(hcat, (∂z∂θ[:, :, b] * Θ_p[:, b] for b in 1:B))

    # Glue forward and backward pass together into dual number types.
    x_d = ForwardDiff.Dual{T}.(solution.x, z_p[1:nx, :])
    y_d = ForwardDiff.Dual{T}.(solution.y, z_p[(nx + 1):(nx + ny), :])
    s_d = ForwardDiff.Dual{T}.(solution.s, z_p[(nx + ny + 1):end, :])

    (; solution.status, solution.kkt_error, solution.ϵ, x = x_d, y = y_d, s = s_d)
end

end
