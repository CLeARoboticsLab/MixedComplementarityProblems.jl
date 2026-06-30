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
struct PrimalDualMCP{T1,T2,T3,T4,T5,T6}
    "A callable `F!(result, x, y, s; θ, ϵ, [η])` to compute the KKT error in-place."
    F!::T1
    "A callable `∇F_z!(result, x, y, s; θ, ϵ, [η])` to compute ∇F wrt z in-place."
    ∇F_z!::T2
    "A callable `∇F_θ!(result, x, y, s; θ, ϵ, [η])` to compute ∇F wrt θ in-place."
    ∇F_θ!::T3
    "Dimension of unconstrained variable."
    unconstrained_dimension::Int
    "Dimension of constrained variable."
    constrained_dimension::Int
    "Kernel-safe per-instance residual evaluator `F_kernel(out, x, y, s, θ, ϵ)`
     (SerialForm + cse, no sharding) for the batched/GPU path. `nothing` unless the
     MCP was built with `compute_kernel_evaluators = true`."
    F_kernel::T4
    "Kernel-safe per-instance ∂F/∂z evaluator `∇F_z_kernel(out, x, y, s, θ, ϵ)` that
     fills the shared pattern's `nnz` nonzero values, in the same column-major order
     as `∇F_z!.rows`/`.cols`. `nothing` unless built."
    ∇F_z_kernel::T5
    "Kernel-safe per-instance ∂F/∂θ evaluator `∇F_θ_kernel(out, x, y, s, θ, ϵ)` that
     fills the DENSE `(d × nθ)` parameter Jacobian (for batched sensitivities).
     `nothing` unless built with `compute_kernel_evaluators` AND `compute_sensitivities`."
    ∇F_θ_kernel::T6
end

"Helper to construct a PrimalDualMCP from callable functions `G(.)` and `H(.)`."
function PrimalDualMCP(
    G,
    H;
    unconstrained_dimension,
    constrained_dimension,
    parameter_dimension,
    compute_sensitivities = false,
    compute_kernel_evaluators = false,
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
        compute_kernel_evaluators,
        backend_options,
    )
end

""" Augment a CSC sparsity pattern `(rows, cols)` of a `d × d` matrix with any missing
diagonal entries, returned together in CSC (column-major) order. Used only for the
kernelized ∂F/∂z (the batched path): adding the full diagonal lets the batched assembler
regularize every row additively (`∇F + η·I`), the way the unbatched solver does. The
unbatched `∇F_z!` is left untouched.
"""
function _augment_full_diagonal(rows, cols, d)
    present = Set{Int}(rows[k] for k in eachindex(rows) if rows[k] == cols[k])
    missing_diagonal = [i for i in 1:d if i ∉ present]
    aug_rows = vcat(rows, missing_diagonal)
    aug_cols = vcat(cols, missing_diagonal)
    order = sortperm(collect(zip(aug_cols, aug_rows)))   # CSC order: by column, then row
    (aug_rows[order], aug_cols[order])
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
    compute_kernel_evaluators = false,
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

    # Kernel-safe evaluators for the batched/GPU path. Built with SerialForm + cse
    # (no sharding) so the generated body is a single straight-line scalar function
    # callable inside a KernelAbstractions kernel. Opt-in: SerialForm can be slow to
    # compile on large problems (see D2 in docs/gpu_kkt_design.md), so CPU-only users
    # pay nothing by default.
    F_kernel, ∇F_z_kernel, ∇F_θ_kernel = if compute_kernel_evaluators
        T <: SymbolicTracingUtils.Symbolics.Num || error(
            "Kernel evaluators are currently only supported with the Symbolics " *
            "backend (got symbolic element type $T).",
        )

        # Only ∂F/∂z carries an η argument. With internal regularization (η embedded in
        # `F`, as for games) it is evaluated at the schedule's η to regularize the Newton
        # system; otherwise η here is a dummy variable the generated code ignores and the
        # batched assembler adds η additively (the `:identity` scheme). The residual is
        # the TRUE KKT error and ∂F/∂θ is η-independent, so both are taken at η = 0 and
        # need no η argument.
        η_kernel =
            isnothing(η_symbolic) ?
            only(SymbolicTracingUtils.make_variables(backend, :η, 1)) : η_symbolic
        F_at_η0 =
            isnothing(η_symbolic) ? F_symbolic :
            SymbolicTracingUtils.Symbolics.substitute.(F_symbolic, Ref(Dict(η_symbolic => 0.0)))

        _build = (expr, extra_args...) -> SymbolicTracingUtils.Symbolics.build_function(
            expr,
            x_symbolic,
            y_symbolic,
            s_symbolic,
            θ_symbolic,
            ϵ_symbolic,
            extra_args...;
            expression = Val{false},
            parallel = SymbolicTracingUtils.Symbolics.SerialForm(),
            cse = true,
        )[2]   # in-place form

        # ∂F/∂z values. Augment the symbolic pattern with the FULL diagonal — missing
        # diagonal entries become structural zeros — so the batched `:identity` scheme can
        # add η to EVERY row (matching the unbatched `∇F + η·I`); without this, rows whose
        # diagonal is structurally absent (e.g. the `H − s` block) stay unregularized and
        # the system is singular. `materialize` reproduces this augmented order, so the
        # kernel's output lines up with `cache.nzval`. (With internal η the pattern already
        # carries the η-regularized primal diagonal; the extra zeros are harmless there.)
        ∇Fz_symbolic = SymbolicTracingUtils.sparse_jacobian(F_symbolic, z_symbolic)
        fz_rows, fz_cols, _ = SparseArrays.findnz(∇Fz_symbolic)
        aug_rows, aug_cols = _augment_full_diagonal(fz_rows, fz_cols, length(z_symbolic))
        ∇Fz_values = [∇Fz_symbolic[aug_rows[k], aug_cols[k]] for k in eachindex(aug_rows)]

        # Dense (d × nθ) parameter Jacobian for batched sensitivities (nθ is typically
        # small; the sensitivity solve's rhs is dense regardless).
        θ_kernel =
            compute_sensitivities ?
            _build(SymbolicTracingUtils.Symbolics.jacobian(F_at_η0, θ_symbolic)) : nothing

        (_build(F_at_η0), _build(∇Fz_values, η_kernel), θ_kernel)
    else
        (nothing, nothing, nothing)
    end

    PrimalDualMCP(
        F!,
        ∇F_z!,
        ∇F_θ!,
        length(x_symbolic),
        length(y_symbolic),
        F_kernel,
        ∇F_z_kernel,
        ∇F_θ_kernel,
    )
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
    compute_kernel_evaluators = false,
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
            compute_kernel_evaluators,
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
        compute_kernel_evaluators,
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
    compute_kernel_evaluators = false,
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
        compute_kernel_evaluators,
        backend_options,
    )
end
