module TrajectoryGameBenchmarkUtils

using MixedComplementarityProblems: MixedComplementarityProblems

using LazySets: LazySets
using TrajectoryGamesBase:
    TrajectoryGamesBase,
    PolygonEnvironment,
    ProductDynamics,
    TimeSeparableTrajectoryGameCost,
    TrajectoryGame,
    GeneralSumCostStructure,
    num_players,
    time_invariant_linear_dynamics,
    unstack_trajectory,
    stack_trajectories,
    state_dim,
    control_dim,
    state_bounds,
    control_bounds,
    OpenLoopStrategy,
    JointStrategy,
    RecedingHorizonStrategy,
    rollout
using TrajectoryGamesExamples: planar_double_integrator, animate_sim_steps
using BlockArrays: mortar, blocks, BlockArray, Block
using LinearAlgebra: norm_sqr, norm
using ProgressMeter: ProgressMeter

include("../examples/utils.jl")
include("../examples/lane_change.jl")

end # module TrajectoryGameBenchmarkUtils

"Generate a random trajectory game, based on the `LaneChange` problem in `examples/`."
function generate_test_problem(
    ::TrajectoryGameBenchmark;
    horizon = 10,
    height = 50,
    num_lanes = 2,
    lane_width = 2,
)
    (; environment) = TrajectoryGameBenchmarkUtils.setup_road_environment(;
        num_lanes,
        lane_width,
        height,
    )
    game = TrajectoryGameBenchmarkUtils.setup_trajectory_game(; environment)

    # Build a game. Each player has a parameter for lane preference. P1 wants to stay
    # in the left lane, and P2 wants to move from the right to the left lane.
    TrajectoryGameBenchmarkUtils.build_mcp_components(;
        game,
        horizon,
        params_per_player = 1,
    )
end

""" Generate a random parameter vector Θ corresponding to an initial state and
horizontal tracking reference per player.
"""
function generate_random_parameter(
    ::TrajectoryGameBenchmark;
    rng,
    num_lanes = 2,
    lane_width = 2,
    height = 50,
)
    (; environment, lane_centers) = TrajectoryGameBenchmarkUtils.setup_road_environment(;
        num_lanes,
        lane_width,
        height,
    )

    initial_states = mortar([
        [LazySets.sample(environment.set; rng); zeros(2)],
        [LazySets.sample(environment.set; rng); zeros(2)],
    ])
    horizontal_references = mortar([[rand(rng, lane_centers)], [rand(rng, lane_centers)]])

    collect(
        TrajectoryGameBenchmarkUtils.pack_parameters(
            initial_states,
            horizontal_references,
        ),
    )
end
