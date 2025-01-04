"Utility for unpacking trajectory."
function unpack_trajectory(flat_trajectories; dynamics::ProductDynamics)
    horizon = Int(
        length(flat_trajectories[Block(1)]) /
        (state_dim(dynamics, 1) + control_dim(dynamics, 1)),
    )
    trajs = map(1:num_players(dynamics), blocks(flat_trajectories)) do ii, traj
        num_states = state_dim(dynamics, ii) * horizon
        X = reshape(traj[1:num_states], (state_dim(dynamics, ii), horizon))
        U = reshape(traj[(num_states + 1):end], (control_dim(dynamics, ii), horizon))

        (; xs = eachcol(X) |> collect, us = eachcol(U) |> collect)
    end

    stack_trajectories(trajs)
end

"Utility for packing trajectory."
function pack_trajectory(traj)
    trajs = unstack_trajectory(traj)
    mapreduce(vcat, trajs) do τ
        [reduce(vcat, τ.xs); reduce(vcat, τ.us)]
    end
end

"Pack an initial state and set of other params into a single parameter vector."
function pack_parameters(initial_state, other_params_per_player)
    mortar(map((x, θ) -> [x; θ], blocks(initial_state), blocks(other_params_per_player)))
end

"Unpack parameters into initial state and other parameters."
function unpack_parameters(params; dynamics::ProductDynamics)
    initial_state = mortar(map(1:num_players(dynamics), blocks(params)) do ii, θi
        θi[1:state_dim(dynamics, ii)]
    end)
    other_params = mortar(map(1:num_players(dynamics), blocks(params)) do ii, θi
        θi[((state_dim(dynamics, ii) + 1):end)]
    end)

    (; initial_state, other_params)
end

"Generate a BitVector mask with 0 entries corresponding to the initial states."
function parameter_mask(; dynamics::ProductDynamics, params_per_player)
    x = mortar([ones(state_dim(dynamics, i)) for i in 1:num_players(dynamics)])
    x̃ = 2mortar([ones(state_dim(dynamics, i)) for i in 1:num_players(dynamics)])
    θ = 3mortar([ones(kk) for kk in params_per_player])

    pack_parameters(x, θ) .== pack_parameters(x̃, θ)
end

"Convert a TrajectoryGame to a PrimalDualMCP."
function build_parametric_game(;
    game::TrajectoryGame,
    horizon = 10,
    params_per_player = 0, # not including initial state, which is always a param
)
    (;
        K_symbolic,
        z_symbolic,
        θ_symbolic,
        lower_bounds,
        upper_bounds,
        dims,
        problems,
        shared_equality,
        shared_inequality,
    ) = build_mcp_components(; game, horizon, params_per_player)

    mcp = MixedComplementarityProblems.PrimalDualMCP(
        K_symbolic,
        z_symbolic,
        θ_symbolic,
        lower_bounds,
        upper_bounds,
    )
    MixedComplementarityProblems.ParametricGame(
        problems,
        shared_equality,
        shared_inequality,
        dims,
        mcp,
    )
end

"Construct MCP components from game components."
function build_mcp_components(;
    game::TrajectoryGame,
    horizon = 10,
    params_per_player = 0, # not including initial state, which is always a param
)
    N = 2
    N == num_players(game) || error("Should have only two players.")

    # Construct costs.
    function player_cost(τ, θi, ii)
        (; xs, us) = unpack_trajectory(τ; game.dynamics)
        ts = Iterators.eachindex(xs)
        map(xs, us, ts) do x, u, t
            game.cost.discount_factor^(t - 1) * game.cost.stage_cost[ii](x, u, t, θi)
        end |> game.cost.reducer
    end

    objectives = map(1:N) do ii
        (τ, θi) -> player_cost(τ, θi, ii)
    end

    # Shared equality constraints.
    shared_equality = (τ, θ) -> let
        (; xs, us) = unpack_trajectory(τ; game.dynamics)
        (; initial_state) = unpack_parameters(θ; game.dynamics)

        # Initial state constraint.
        g̃1 = xs[1] - initial_state

        # Dynamics constraints.
        ts = Iterators.eachindex(xs)
        g̃2 = mapreduce(vcat, ts[2:end]) do t
            xs[t] - game.dynamics(xs[t - 1], us[t - 1])
        end

        vcat(g̃1, g̃2)
    end

    # Shared inequality constraints.
    shared_inequality =
        (τ, θ) -> let
            (; xs, us) = unpack_trajectory(τ; game.dynamics)

            # Collision-avoidance constriant.
            h̃1 = game.coupling_constraints(xs, us, θ)

            # Environment boundaries.
            env_constraints = TrajectoryGamesBase.get_constraints(game.env)
            h̃2 = mapreduce(vcat, xs) do x
                env_constraints(x)
            end

            # Actuator/state limits.
            actuator_constraint = TrajectoryGamesBase.get_constraints_from_box_bounds(
                control_bounds(game.dynamics),
            )
            h̃3 = mapreduce(vcat, us) do u
                actuator_constraint(u)
            end

            state_constraint = TrajectoryGamesBase.get_constraints_from_box_bounds(
                state_bounds(game.dynamics),
            )
            h̃4 = mapreduce(vcat, xs) do x
                state_constraint(x)
            end

            vcat(h̃1, h̃2, h̃3, h̃4)
        end

    primal_dims = [
        horizon * (state_dim(game.dynamics, ii) + control_dim(game.dynamics, ii)) for
        ii in 1:N
    ]

    problems = map(
        f -> MixedComplementarityProblems.OptimizationProblem(; objective = f),
        objectives,
    )

    components = MixedComplementarityProblems.game_to_mcp(;
        test_point = BlockArray(zeros(sum(primal_dims)), primal_dims),
        test_parameter = mortar([
            zeros(state_dim(game.dynamics, ii) + params_per_player) for ii in 1:N
        ]),
        problems,
        shared_equality,
        shared_inequality,
    )

    (; problems, shared_equality, shared_inequality, components...)
end

"Generate an initial guess for primal variables following a zero input sequence."
function zero_input_trajectory(;
    game::TrajectoryGame{<:ProductDynamics},
    horizon,
    initial_state,
)
    rollout_strategy =
        map(1:num_players(game)) do ii
            (x, t) -> zeros(control_dim(game.dynamics, ii))
        end |> TrajectoryGamesBase.JointStrategy

    TrajectoryGamesBase.rollout(game.dynamics, rollout_strategy, initial_state, horizon)
end

"Solve a parametric trajectory game, where the parameter is just the initial state."
function TrajectoryGamesBase.solve_trajectory_game!(
    game::TrajectoryGame{<:ProductDynamics},
    horizon,
    parameter_value,
    strategy;
    parametric_game = build_parametric_game(;
        game,
        horizon,
        params_per_player = Int(
            (length(parameter_value) - state_dim(game)) / num_players(game),
        ),
    ),
)
    # Solve, maybe with warm starting.
    if !isnothing(strategy.last_solution) && strategy.last_solution.status == :solved
        solution = MixedComplementarityProblems.solve(
            parametric_game,
            parameter_value;
            solver_type = MixedComplementarityProblems.InteriorPoint(),
            x₀ = strategy.last_solution.variables.x,
            y₀ = strategy.last_solution.variables.y,
        )
    else
        (; initial_state) = unpack_parameters(parameter_value; game.dynamics)
        solution = MixedComplementarityProblems.solve(
            parametric_game,
            parameter_value;
            solver_type = MixedComplementarityProblems.InteriorPoint(),
            x₀ = [
                pack_trajectory(zero_input_trajectory(; game, horizon, initial_state))
                zeros(sum(parametric_game.dims.λ) + parametric_game.dims.λ̃)
            ],
        )
    end

    # Update warm starting info.
    if solution.status == :solved
        strategy.last_solution = solution
    end
    strategy.solution_status = solution.status

    # Pack solution into OpenLoopStrategy.
    trajs = unstack_trajectory(unpack_trajectory(mortar(solution.primals); game.dynamics))
    JointStrategy(map(traj -> OpenLoopStrategy(traj.xs, traj.us), trajs))
end

"Receding horizon strategy that supports warm starting."
Base.@kwdef mutable struct WarmStartRecedingHorizonStrategy
    game::TrajectoryGame
    parametric_game::MixedComplementarityProblems.ParametricGame
    receding_horizon_strategy::Any = nothing
    time_last_updated::Int = 0
    turn_length::Int
    horizon::Int
    last_solution::Any = nothing
    parameters::Any = nothing
    solution_status::Any = nothing
end

function (strategy::WarmStartRecedingHorizonStrategy)(state, time)
    plan_exists = !isnothing(strategy.receding_horizon_strategy)
    time_along_plan = time - strategy.time_last_updated + 1
    plan_is_still_valid = 1 <= time_along_plan <= strategy.turn_length

    update_plan = !plan_exists || !plan_is_still_valid
    if update_plan
        strategy.receding_horizon_strategy = TrajectoryGamesBase.solve_trajectory_game!(
            strategy.game,
            strategy.horizon,
            pack_parameters(state, strategy.parameters),
            strategy;
            strategy.parametric_game,
        )
        strategy.time_last_updated = time
        time_along_plan = 1
    end

    strategy.receding_horizon_strategy(state, time_along_plan)
end
