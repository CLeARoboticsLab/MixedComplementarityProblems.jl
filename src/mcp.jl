""" Store key elements of the primal-dual KKT system for a MCP composed of
functions G(.) and H(.) such that
                             0 = G(x, y; θ)
                             0 ≤ H(x, y; θ) ⟂ y ≥ 0.

The primal-dual system arises when we introduce slack variable `s` and set
                             G(x, y; θ)     = 0
                             H(x, y; θ) - s = 0
                             s ⦿ y - ϵ      = 0
for some ϵ > 0. Define the function `F(x, y, s; θ, ϵ, [η])` to return the left
hand side of this system of equations. Here, `η` is an optional nonnegative
regularization parameter defined by "internally-regularized" problems.
"""
struct PrimalDualMCP{T1,T2,T3}
    "A callable `F!(result, x, y, s; θ, ϵ, [η])` to the KKT error in-place."
    F!::T1
    "A callable `∇F_z!(result, x, y, s; θ, ϵ, [η])` to compute ∇F wrt z in-place."
    ∇F_z!::T2
    "A callable `∇F_θ!(result, x, y, s; θ, ϵ, [η])` to compute ∇F wrt θ in-place."
    ∇F_θ!::T3
    "Dimension of unconstrained variable."
    unconstrained_dimension::Int
    "Dimension of constrained variable."
    constrained_dimension::Int
end

"Helper to construct a PrimalDualMCP from callable functions `G(.)` and `H(.)`."
function PrimalDualMCP(
    G,
    H;
    unconstrained_dimension,
    constrained_dimension,
    parameter_dimension,
    compute_sensitivities = false,
    backend = SymbolicTracingUtils.SymbolicsBackend(),
    backend_options = (;),
)
    x_symbolic = SymbolicTracingUtils.make_variables(backend, :x, unconstrained_dimension)
    y_symbolic = SymbolicTracingUtils.make_variables(backend, :y, constrained_dimension)
    θ_symbolic = SymbolicTracingUtils.make_variables(backend, :θ, parameter_dimension)
    G_symbolic = G(x_symbolic, y_symbolic; θ = θ_symbolic)
    H_symbolic = H(x_symbolic, y_symbolic; θ = θ_symbolic)

    PrimalDualMCP(
        G_symbolic,
        H_symbolic,
        x_symbolic,
        y_symbolic,
        θ_symbolic;
        compute_sensitivities,
        backend_options,
    )
end

"Construct a PrimalDualMCP from symbolic expressions of G(.) and H(.)."
function PrimalDualMCP(
    G_symbolic::Vector{T},
    H_symbolic::Vector{T},
    x_symbolic::Vector{T},
    y_symbolic::Vector{T},
    θ_symbolic::Vector{T},
    η_symbolic::Union{Nothing,T} = nothing;
    compute_sensitivities = false,
    backend_options = (;),
) where {T<:Union{SymbolicTracingUtils.FD.Node,SymbolicTracingUtils.Symbolics.Num}}
    # Create symbolic slack variable `s` and parameter `ϵ`.
    if T == SymbolicTracingUtils.FD.Node
        backend = SymbolicTracingUtils.FastDifferentiationBackend()
    else
        @assert T === SymbolicTracingUtils.Symbolics.Num
        backend = SymbolicTracingUtils.SymbolicsBackend()
    end

    s_symbolic = SymbolicTracingUtils.make_variables(backend, :s, length(y_symbolic))
    ϵ_symbolic = only(SymbolicTracingUtils.make_variables(backend, :ϵ, 1))
    z_symbolic = [x_symbolic; y_symbolic; s_symbolic]

    F_symbolic = [
        G_symbolic
        H_symbolic - s_symbolic
        s_symbolic .* y_symbolic .- ϵ_symbolic
    ]

    F! = if isnothing(η_symbolic)
        _F! = SymbolicTracingUtils.build_function(
            F_symbolic,
            x_symbolic,
            y_symbolic,
            s_symbolic,
            θ_symbolic,
            ϵ_symbolic;
            in_place = true,
            backend_options,
        )

        (result, x, y, s; θ, ϵ) -> _F!(result, x, y, s, θ, ϵ)
    else
        _F! = SymbolicTracingUtils.build_function(
            F_symbolic,
            x_symbolic,
            y_symbolic,
            s_symbolic,
            θ_symbolic,
            ϵ_symbolic,
            η_symbolic;
            in_place = true,
            backend_options,
        )

        (result, x, y, s; θ, ϵ, η = 0.0) -> _F!(result, x, y, s, θ, ϵ, η)
    end

    function process_∇F(F, var)
        ∇F_symbolic = SymbolicTracingUtils.sparse_jacobian(F, var)
        rows, cols, _ = SparseArrays.findnz(∇F_symbolic)
        constant_entries = SymbolicTracingUtils.get_constant_entries(∇F_symbolic, var)

        if isnothing(η_symbolic)
            _∇F! = SymbolicTracingUtils.build_function(
                ∇F_symbolic,
                x_symbolic,
                y_symbolic,
                s_symbolic,
                θ_symbolic,
                ϵ_symbolic;
                in_place = true,
                backend_options,
            )

            return SymbolicTracingUtils.SparseFunction(
                (result, x, y, s; θ, ϵ) -> _∇F!(result, x, y, s, θ, ϵ),
                rows,
                cols,
                size(∇F_symbolic),
                constant_entries,
            )
        else
            _∇F! = SymbolicTracingUtils.build_function(
                ∇F_symbolic,
                x_symbolic,
                y_symbolic,
                s_symbolic,
                θ_symbolic,
                ϵ_symbolic,
                η_symbolic;
                in_place = true,
                backend_options,
            )

            return SymbolicTracingUtils.SparseFunction(
                (result, x, y, s; θ, ϵ, η = 0.0) -> _∇F!(result, x, y, s, θ, ϵ, η),
                rows,
                cols,
                size(∇F_symbolic),
                constant_entries,
            )
        end
    end

    ∇F_z! = process_∇F(F_symbolic, z_symbolic)
    ∇F_θ! = !compute_sensitivities ? nothing : process_∇F(F_symbolic, θ_symbolic)

    PrimalDualMCP(F!, ∇F_z!, ∇F_θ!, length(x_symbolic), length(y_symbolic))
end

""" Construct a PrimalDualMCP from `K(z; θ) ⟂ z̲ ≤ z ≤ z̅`, where `K` is callable.
NOTE: Assumes that all upper bounds are Inf, and lower bounds are either -Inf or 0.
"""
function PrimalDualMCP(
    K,
    lower_bounds::Vector,
    upper_bounds::Vector;
    parameter_dimension,
    internally_regularized = false,
    compute_sensitivities = false,
    backend = SymbolicTracingUtils.SymbolicsBackend(),
    backend_options = (;),
)
    z_symbolic = SymbolicTracingUtils.make_variables(backend, :z, length(lower_bounds))
    θ_symbolic = SymbolicTracingUtils.make_variables(backend, :θ, parameter_dimension)
    K_symbolic = K(z_symbolic; θ = θ_symbolic)

    if internally_regularized
        η_symbolic = only(SymbolicTracingUtils.make_variables(backend, :η, 1))

        return PrimalDualMCP(
            K_symbolic,
            z_symbolic,
            θ_symbolic,
            lower_bounds,
            upper_bounds;
            η_symbolic,
            compute_sensitivities,
            backend_options,
        )
    end

    PrimalDualMCP(
        K_symbolic,
        z_symbolic,
        θ_symbolic,
        lower_bounds,
        upper_bounds;
        compute_sensitivities,
        backend_options,
    )
end

"""Construct a PrimalDualMCP from symbolic `K(z; θ) ⟂ z̲ ≤ z ≤ z̅`.
NOTE: Assumes that all upper bounds are Inf, and lower bounds are either -Inf or 0.
"""
function PrimalDualMCP(
    K_symbolic::Vector{T},
    z_symbolic::Vector{T},
    θ_symbolic::Vector{T},
    lower_bounds::Vector,
    upper_bounds::Vector;
    η_symbolic::Union{Nothing,T} = nothing,
    compute_sensitivities = false,
    backend_options = (;),
) where {T<:Union{SymbolicTracingUtils.FD.Node,SymbolicTracingUtils.Symbolics.Num}}
    @assert all(isinf.(upper_bounds)) && all(isinf.(lower_bounds) .|| lower_bounds .== 0)

    unconstrained_indices = findall(isinf, lower_bounds)
    constrained_indices = findall(!isinf, lower_bounds)

    G_symbolic = K_symbolic[unconstrained_indices]
    H_symbolic = K_symbolic[constrained_indices]
    x_symbolic = z_symbolic[unconstrained_indices]
    y_symbolic = z_symbolic[constrained_indices]

    PrimalDualMCP(
        G_symbolic,
        H_symbolic,
        x_symbolic,
        y_symbolic,
        θ_symbolic,
        η_symbolic;
        compute_sensitivities,
        backend_options,
    )
end
