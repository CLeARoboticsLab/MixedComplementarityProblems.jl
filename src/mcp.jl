""" Store key elements of the primal-dual KKT system for a MCP composed of
functions G(.) and H(.) such that
                             0 = G(x, y; θ)
                             0 ≤ H(x, y; θ) ⟂ y ≥ 0.

The primal-dual system arises when we introduce slack variable `s` and set
                             G(x, y; θ)     = 0
                             H(x, y; θ) - s = 0
                             s ⦿ y - ϵ      = 0
for some ϵ > 0. Define the function `F(x, y, s; θ, ϵ)` to return the left
hand side of this system of equations.
"""
struct PrimalDualMCP{T1,T2,T3}
    "A callable `F(x, y, s; θ, ϵ)` which computes the KKT error in the primal-dual system."
    F::T1
    "A callable `∇F_z(x, y, s; θ, ϵ)` which stores the Jacobian of the KKT error wrt z."
    ∇F_z::T2
    "A callable `∇F_θ(x, y, s; θ, ϵ)` which stores the Jacobian of the KKT error wrt θ."
    ∇F_θ::T3
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
    backend = SymbolicUtils.SymbolicsBackend(),
    backend_options = (;),
)
    x_symbolic = SymbolicUtils.make_variables(backend, :x, unconstrained_dimension)
    y_symbolic = SymbolicUtils.make_variables(backend, :y, constrained_dimension)
    θ_symbolic = SymbolicUtils.make_variables(backend, :θ, parameter_dimension)
    G_symbolic = G(x_symbolic, y_symbolic; θ = θ_symbolic)
    H_symbolic = H(x_symbolic, y_symbolic; θ = θ_symbolic)

    PrimalDualMCP(
        G_symbolic,
        H_symbolic,
        x_symbolic,
        y_symbolic,
        θ_symbolic;
        compute_sensitivities,
        backend,
        backend_options,
    )
end

"Construct a PrimalDualMCP from symbolic expressions of G(.) and H(.)."
function PrimalDualMCP(
    G_symbolic::Vector{T},
    H_symbolic::Vector{T},
    x_symbolic::Vector{T},
    y_symbolic::Vector{T},
    θ_symbolic::Vector{T};
    compute_sensitivities = false,
    backend_options = (;),
) where {T<:Union{FD.Node,Symbolics.Num}}
    # Create symbolic slack variable `s` and parameter `ϵ`.
    if T == FD.Node
        backend = SymbolicUtils.FastDifferentiationBackend()
    else
        @assert T === Symbolics.Num
        backend = SymbolicUtils.SymbolicsBackend()
    end

    s_symbolic = SymbolicUtils.make_variables(backend, :s, length(y_symbolic))
    ϵ_symbolic = only(SymbolicUtils.make_variables(backend, :ϵ, 1))
    z_symbolic = [x_symbolic; y_symbolic; s_symbolic]

    F_symbolic = [
        G_symbolic
        H_symbolic - s_symbolic
        s_symbolic .* y_symbolic .- ϵ_symbolic
    ]

    F = let
        _F = SymbolicUtils.build_function(
            F_symbolic,
            [z_symbolic; θ_symbolic; ϵ_symbolic];
            in_place = false,
            backend_options,
        )

        (x, y, s; θ, ϵ) -> _F([x; y; s; θ; ϵ])
    end

    ∇F_z = let
        ∇F_symbolic = SymbolicUtils.sparse_jacobian(F_symbolic, z_symbolic)
        _∇F = SymbolicUtils.build_function(
            ∇F_symbolic,
            [z_symbolic; θ_symbolic; ϵ_symbolic];
            in_place = false,
            backend_options,
        )

        (x, y, s; θ, ϵ) -> _∇F([x; y; s; θ; ϵ])
    end

    ∇F_θ =
        !compute_sensitivities ? nothing :
        let
            ∇F_symbolic = SymbolicUtils.sparse_jacobian(F_symbolic, θ_symbolic)
            _∇F = SymbolicUtils.build_function(
                ∇F_symbolic,
                [z_symbolic; θ_symbolic; ϵ_symbolic];
                in_place = false,
                backend_options,
            )

            (x, y, s; θ, ϵ) -> _∇F([x; y; s; θ; ϵ])
        end

    PrimalDualMCP(F, ∇F_z, ∇F_θ, length(x_symbolic), length(y_symbolic))
end

""" Construct a PrimalDualMCP from `K(z; θ) ⟂ z̲ ≤ z ≤ z̅`, where `K` is callable.
NOTE: Assumes that all upper bounds are Inf, and lower bounds are either -Inf or 0.
"""
function PrimalDualMCP(
    K,
    lower_bounds::Vector,
    upper_bounds::Vector;
    parameter_dimension,
    compute_sensitivities = false,
    backend = SymbolicUtils.SymbolicsBackend(),
    backend_options = (;),
)
    z_symbolic = SymbolicUtils.make_variables(backend, :z, length(lower_bounds))
    θ_symbolic = SymbolicUtils.make_variables(backend, :θ, parameter_dimension)
    K_symbolic = K(z_symbolic; θ = θ_symbolic)

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
    compute_sensitivities = false,
    backend_options = (;),
) where {T<:Union{FD.Node,Symbolics.Num}}
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
        θ_symbolic;
        compute_sensitivities,
        backend_options,
    )
end
