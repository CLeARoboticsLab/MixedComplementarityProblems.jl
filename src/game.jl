"Utility to represent a parameterized optimization problem."
Base.@kwdef struct OptimizationProblem{T1,T2,T3}
    objective::T1
    private_equality::T2 = nothing
    private_inequality::T3 = nothing
end

"""A structure to represent a game with objectives and constraints parameterized by
a vector θ ∈ Rⁿ.

We will assume that the players' primal variables are stacked into a BlockVector
x with the i-th block corresponding to the i-th player's decision variable.
Similarly, we will assume that the parameter θ has a block structure so that the
i-th player's problem depends only upon the i-th block of θ.
"""
struct ParametricGame{T1,T2,T3}
    problems::Vector{<:OptimizationProblem}
    shared_equality::T1
    shared_inequality::T2
    dims::T3

    "PrimalDualMCP representation."
    mcp::PrimalDualMCP
end

function ParametricGame(;
    test_point,
    test_parameter,
    problems,
    shared_equality = nothing,
    shared_inequality = nothing,
)
    (; K_symbolic, z_symbolic, θ_symbolic, lower_bounds, upper_bounds, dims) =
        game_to_mcp(;
            test_point,
            test_parameter,
            problems,
            shared_equality,
            shared_inequality,
        )

    mcp = PrimalDualMCP(K_symbolic, z_symbolic, θ_symbolic, lower_bounds, upper_bounds)
    ParametricGame(problems, shared_equality, shared_inequality, dims, mcp)
end

"Helper for converting game components to MCP components."
function game_to_mcp(;
    test_point,
    test_parameter,
    problems,
    shared_equality = nothing,
    shared_inequality = nothing,
)
    N = length(problems)
    @assert N == length(blocks(test_point))

    dims = dimensions(
        test_point,
        test_parameter,
        problems,
        shared_equality,
        shared_inequality,
    )

    # Define primal and dual variables for the game, and game parameters..
    # Note that BlockArrays can handle blocks of zero size.
    backend = SymbolicTracingUtils.SymbolicsBackend()
    x =
        SymbolicTracingUtils.make_variables(backend, :x, sum(dims.x)) |>
        to_blockvector(dims.x)
    λ =
        SymbolicTracingUtils.make_variables(backend, :λ, sum(dims.λ)) |>
        to_blockvector(dims.λ)
    μ =
        SymbolicTracingUtils.make_variables(backend, :μ, sum(dims.μ)) |>
        to_blockvector(dims.μ)
    λ̃ = SymbolicTracingUtils.make_variables(backend, :λ̃, dims.λ̃)
    μ̃ = SymbolicTracingUtils.make_variables(backend, :μ̃, dims.μ̃)
    θ =
        SymbolicTracingUtils.make_variables(backend, :θ, sum(dims.θ)) |>
        to_blockvector(dims.θ)

    # Build symbolic expressions for objectives and constraints.
    fs = map(problems, blocks(θ)) do p, θi
        p.objective(x, θi)
    end
    gs = map(problems, blocks(θ)) do p, θi
        isnothing(p.private_equality) ? nothing : p.private_equality(x, θi)
    end
    hs = map(problems, blocks(θ)) do p, θi
        isnothing(p.private_inequality) ? nothing : p.private_inequality(x, θi)
    end

    g̃ = isnothing(shared_equality) ? nothing : shared_equality(x, θ)
    h̃ = isnothing(shared_inequality) ? nothing : shared_inequality(x, θ)

    # Build gradient of each player's Lagrangian.
    ∇Ls = map(fs, gs, hs, blocks(x), blocks(λ), blocks(μ)) do f, g, h, xi, λi, μi
        L =
            f - (isnothing(g) ? 0 : sum(λi .* g)) - (isnothing(h) ? 0 : sum(μi .* h)) - (isnothing(g̃) ? 0 : sum(λ̃ .* g̃)) -
            (isnothing(h̃) ? 0 : sum(μ̃ .* h̃))
        SymbolicTracingUtils.gradient(L, xi)
    end

    # Build MCP representation.
    symbolic_type = eltype(x)
    K = Vector{symbolic_type}(
        filter!(
            !isnothing,
            [
                reduce(vcat, ∇Ls)
                reduce(vcat, gs)
                g̃
                reduce(vcat, hs)
                h̃
            ],
        ),
    )

    z = Vector{symbolic_type}(
        filter!(
            !isnothing,
            [
                x
                mapreduce(b -> length(b) == 0 ? nothing : b, vcat, blocks(λ))
                length(λ̃) == 0 ? nothing : λ̃
                mapreduce(b -> length(b) == 0 ? nothing : b, vcat, blocks(μ))
                length(μ̃) == 0 ? nothing : μ̃
            ],
        ),
    )

    lower_bounds = [
        fill(-Inf, length(x))
        fill(-Inf, length(λ))
        fill(-Inf, length(λ̃))
        fill(0, length(μ))
        fill(0, length(μ̃))
    ]

    upper_bounds = [
        fill(Inf, length(x))
        fill(Inf, length(λ))
        fill(Inf, length(λ̃))
        fill(Inf, length(μ))
        fill(Inf, length(μ̃))
    ]

    (;
        K_symbolic = collect(K),
        z_symbolic = collect(z),
        θ_symbolic = collect(θ),
        lower_bounds,
        upper_bounds,
        dims,
    )
end

function dimensions(
    test_point,
    test_parameter,
    problems,
    shared_equality,
    shared_inequality,
)
    x = only(blocksizes(test_point))
    θ = only(blocksizes(test_parameter))
    λ = map(problems, blocks(test_parameter)) do p, θi
        isnothing(p.private_equality) ? 0 : length(p.private_equality(test_point, θi))
    end
    μ = map(problems, blocks(test_parameter)) do p, θi
        isnothing(p.private_inequality) ? 0 : length(p.private_inequality(test_point, θi))
    end

    λ̃ =
        isnothing(shared_equality) ? 0 :
        length(shared_equality(test_point, test_parameter))
    μ̃ =
        isnothing(shared_inequality) ? 0 :
        length(shared_inequality(test_point, test_parameter))

    (; x, θ, λ, μ, λ̃, μ̃)
end

"Solve a parametric game."
function solve(
    game::ParametricGame,
    θ;
    solver_type = InteriorPoint(),
    x₀ = zeros(sum(game.dims.x) + sum(game.dims.λ) + game.dims.λ̃),
    y₀ = ones(sum(game.dims.μ) + game.dims.μ̃),
    tol = 1e-4,
)
    (; x, y, s, kkt_error, status) = solve(solver_type, game.mcp, θ; x₀, y₀, tol)

    # Unpack primals per-player for ease of access later.
    end_dims = cumsum(game.dims.x)
    primals = map(1:num_players(game)) do ii
        (ii == 1) ? x[1:end_dims[ii]] : x[(end_dims[ii - 1] + 1):end_dims[ii]]
    end

    (; primals, variables = (; x, y, s), kkt_error, status)
end

"Return the number of players in this game."
function num_players(game::ParametricGame)
    length(game.problems)
end
