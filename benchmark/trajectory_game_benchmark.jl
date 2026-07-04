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

"""Batched zero-input-rollout initial guess: one `Θ_host` column per instance, unpacked
into that instance's `initial_state` (via `unpack_parameters`), rolled forward under zero
control (via `zero_input_trajectory`, dynamics-aware — exact for this problem's
double-integrator drift), then packed back into primal-vector layout (via
`pack_trajectory`). Reuses the same rollout `examples/lane_change.jl`'s own
(non-batched) receding-horizon strategy warm-starts with, instead of re-deriving the
drift dynamics by hand.
"""
function batched_zero_input_trajectory(; game, horizon, Θ_host, params_per_player = 1)
    dynamics = game.dynamics
    n = num_players(dynamics)
    theta_block_sizes = [state_dim(dynamics, ii) + params_per_player for ii in 1:n]
    offsets = [0; cumsum(theta_block_sizes)]

    reduce(
        hcat,
        map(1:size(Θ_host, 2)) do b
            params = mortar([Θ_host[(offsets[i] + 1):offsets[i + 1], b] for i in 1:n])
            (; initial_state) = unpack_parameters(params; dynamics)
            traj = zero_input_trajectory(; game, horizon, initial_state)
            pack_trajectory(traj)
        end,
    )
end

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

Initial positions are jittered in a small rectangle around a fixed nominal
lane-change scenario (P1 nominally in the leftmost lane, P2 nominally in the
rightmost lane and `lead_offset` further up the road, rather than sampled
uniformly over the whole road polygon. Sampling uniformly over the full
polygon (the old behavior) mixes near-trivial instances (players spawned far
apart, never interacting within the horizon) with genuinely hard ones (close
encounters), and can even draw already-infeasible pairs (no check that
players start ≥2m apart) — both make `solved` fraction noisy and hard to
interpret, especially at longer horizons. The tight rectangle plus rejection
sampling below keeps every instance in the same, realistic difficulty regime
this benchmark is meant to exercise.

Both players share the same lane preference (see `horizontal_references`
below), so the only feasible equilibria are "P1 leads" or "P2 leads" (one
player in front of the other, ≥2m apart, in the same lane) — a symmetric,
essentially discrete choice. Sampling both players' y-position around the
SAME `nominal_y` leaves that choice ambiguous at the initial guess, which is
exactly the kind of degenerate starting point that makes Newton-based
complementarity solvers oscillate between the two candidate equilibria
instead of converging to either (empirically, this is why `solved` fraction
collapses at longer horizons — see PR discussion). Giving P2 a head start via
`lead_offset` breaks that symmetry in P2's favor before the solve even
starts, making "P2 merges in ahead of P1" the unambiguous likely outcome.
"""
function generate_random_parameter(
    ::TrajectoryGameBenchmark;
    rng,
    num_lanes = 2,
    lane_width = 2,
    height = 50,
    x_jitter = 0.3,
    y_jitter = 3.0,
    nominal_y = 5.0,
    lead_offset = 5.0,
    min_separation = 2.2,
    initial_vy = 1.0,
    lead_vy_boost = 0.75,

    # Tolerate problem-construction kwargs (e.g. `horizon`) that `benchmark_throughput`
    # also splats here; the parameter depends only on the road environment, not them.
    kwargs...,
)
    (; lane_centers) = TrajectoryGameBenchmarkUtils.setup_road_environment(;
        num_lanes,
        lane_width,
        height,
    )

    p1 = p2 = nothing
    while true
        p1 = [
            first(lane_centers) + x_jitter * (2rand(rng) - 1),
            nominal_y + y_jitter * (2rand(rng) - 1),
        ]
        p2 = [
            last(lane_centers) + x_jitter * (2rand(rng) - 1),
            nominal_y + lead_offset + y_jitter * (2rand(rng) - 1),
        ]
        norm(p1 - p2) >= min_separation && break
    end

    # Nonzero initial vy (rather than starting fully at rest) gives the zero-input
    # rollout used as the interior-point warm start (see `benchmark/gpu` diagnostics)
    # actual forward progress to propagate, instead of a degenerate stationary guess.
    # `lead_vy_boost` additionally makes P2 (already ahead via `lead_offset`) move
    # faster than P1 in that guess, so the rollout shows the gap GROWING over the
    # horizon instead of staying fixed — reinforcing "P2 leads" as the unambiguous
    # equilibrium rather than just placing P2 ahead at a constant offset.
    initial_states =
        mortar([[p1; 0.0; initial_vy], [p2; 0.0; initial_vy + lead_vy_boost]])

    # Fixed canonical scenario (matches `examples/lane_change.jl`): P1 stays in the
    # leftmost lane, P2 merges into it from the rightmost lane. Also no longer
    # randomized, for the same homogeneity reason as the position jitter above.
    horizontal_references = mortar([[lane_centers[1]], [lane_centers[1]]])

    collect(
        TrajectoryGameBenchmarkUtils.pack_parameters(
            initial_states,
            horizontal_references,
        ),
    )
end

"""Zero-input-rollout initial guess for `BatchedInteriorPoint`'s `X₀`. Delegates the
actual per-instance rollout to `TrajectoryGameBenchmarkUtils.batched_zero_input_trajectory`
(built on the same `zero_input_trajectory`/`pack_trajectory` helpers `examples/utils.jl`
already provides for the non-batched receding-horizon strategy), then zero-pads the
remaining `λ̃` (shared-equality-multiplier) block of `mcp.unconstrained_dimension`.

The all-zero cold start (`X₀ = nothing`, the batched-solver default) is dynamically
self-consistent but ignores each instance's actual initial position/velocity entirely,
and gets progressively worse as horizon grows — see the `generate_random_parameter`
docstring above and the PR discussion for the full investigation. This rollout, combined
with the nonzero `initial_vy`/`lead_offset`/`lead_vy_boost` sampling above, is what
brings `solved` fraction back up at long horizons.
"""
function SolverBenchmarks.generate_initial_guess(
    ::TrajectoryGameBenchmark,
    mcp,
    Θ,
    device;
    horizon,
    num_lanes = 2,
    lane_width = 2,
    height = 50,
    kwargs...,
)
    (; environment) =
        TrajectoryGameBenchmarkUtils.setup_road_environment(; num_lanes, lane_width, height)
    game = TrajectoryGameBenchmarkUtils.setup_trajectory_game(; environment)

    Θ_host = Array(Θ)
    trajectory =
        TrajectoryGameBenchmarkUtils.batched_zero_input_trajectory(; game, horizon, Θ_host)

    nx = mcp.unconstrained_dimension
    X0_host = vcat(trajectory, zeros(nx - size(trajectory, 1), size(Θ_host, 2)))
    Adapt.adapt(device, X0_host)
end
