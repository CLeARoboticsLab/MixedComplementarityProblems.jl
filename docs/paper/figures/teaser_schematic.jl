""" Generates Fig. 1(a): a static schematic of the lane-change trajectory game.

Solves one "hero" instance plus a fan of ~40 other random instances from the exact same
distribution benchmarked in the Experiments section (T=10, via
`benchmark/trajectory_game_benchmark.jl`'s `generate_random_parameter`), and plots every
solved trajectory: the fan faintly in the background, the hero instance bold in front.

Run with: julia --project=benchmark docs/paper/figures/teaser_schematic.jl
Output:   docs/paper/figures/teaser_schematic.pdf
"""

const REPO = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(REPO, "benchmark", "SolverBenchmarks.jl"))
using .SolverBenchmarks
const B = SolverBenchmarks
const Utils = B.TrajectoryGameBenchmarkUtils

using MixedComplementarityProblems: MixedComplementarityProblems
const MCP = MixedComplementarityProblems
using BlockArrays: mortar
using TrajectoryGamesBase: unstack_trajectory
using CairoMakie
using FileIO: FileIO
import Random

# Car icons cropped from the user-provided reference sheet (see assets/prepare_car_icons.jl
# for how these were made and oriented specifically for `image!`'s axis convention).
const CAR_IMAGES = (
    steelblue = FileIO.load(joinpath(@__DIR__, "assets", "car_blue.png")),
    firebrick = FileIO.load(joinpath(@__DIR__, "assets", "car_red.png")),
)

const HORIZON = 20  # longer than the T=10 benchmarked in Experiments — purely illustrative,
                     # chosen to show more of the merge interaction; see the warm-start note
                     # below for why a longer horizon needs one to stay reliable.
const N_FAN = 40
const MAX_ATTEMPTS = 300
const SEED = 7

(; environment, lane_centers) = Utils.setup_road_environment()
game = Utils.setup_trajectory_game(; environment)
parametric_game = Utils.build_parametric_game(; game, horizon = HORIZON, params_per_player = 1)

rng = Random.MersenneTwister(SEED)

const END_DIMS = cumsum(parametric_game.dims.x)
const NX = parametric_game.mcp.unconstrained_dimension

# Zero-input-rollout warm start (matches `SolverBenchmarks.generate_initial_guess` and
# `examples/utils.jl`'s `solve_trajectory_game!`): a cold (all-zero) start becomes
# unreliable as the horizon grows, since it ignores each instance's actual initial
# position/velocity entirely. Reuses the same batched helper with a single column.
function warm_start(θ)
    traj = vec(Utils.batched_zero_input_trajectory(; game, horizon = HORIZON, Θ_host = reshape(θ, :, 1)))
    vcat(traj, zeros(NX - length(traj)))
end

# NOTE: `MixedComplementarityProblems.solve(::ParametricGame, θ)` hardcodes
# `regularize_linear_solve = :internal`, which the benchmark's own comment
# (benchmark/batched_benchmark.jl) warns is "insufficient and diverges" for trajectory
# games carrying an internal η. The Experiments section's near-0-failure numbers
# (Table I: 1024/1024 solved) use `:identity` against the raw mcp instead, so we
# replicate the ParametricGame-level per-player unpacking (game.jl) by hand here to
# match that, rather than calling through the wrapper with its worse default.
function solve_positions(θ)
    sol = MCP.solve(
        MCP.InteriorPoint(), parametric_game.mcp, θ;
        tol = 1e-4, regularize_linear_solve = :identity, x₀ = warm_start(θ),
    )
    sol.status != :solved && return nothing
    primals = map(1:length(END_DIMS)) do ii
        (ii == 1) ? sol.x[1:END_DIMS[ii]] : sol.x[(END_DIMS[ii - 1] + 1):END_DIMS[ii]]
    end
    trajs = unstack_trajectory(Utils.unpack_trajectory(mortar(primals); dynamics = game.dynamics))
    map(trajs) do τ
        pts = [(x[1], x[2]) for x in τ.xs]
        pts
    end
end

solved_positions = []
n_attempts = 0
while length(solved_positions) < N_FAN + 1 && n_attempts < MAX_ATTEMPTS
    global n_attempts += 1
    θ = B.generate_random_parameter(B.TrajectoryGameBenchmark(); rng, horizon = HORIZON)
    positions = solve_positions(θ)
    isnothing(positions) || push!(solved_positions, positions)
end
length(solved_positions) < N_FAN + 1 &&
    @warn "only $(length(solved_positions))/$(N_FAN + 1) instances solved within $MAX_ATTEMPTS attempts"
@info "$(length(solved_positions))/$n_attempts attempts solved"

hero = first(solved_positions)
fan = solved_positions[2:end]

lane_width = 2.0
left_edge = first(lane_centers) - 0.5lane_width
right_edge = last(lane_centers) + 0.5lane_width

# Rotate the whole scene 90° clockwise so cars drive left-to-right: plot (distance
# along road, lateral position) instead of (lateral position, distance along road).
# Mirroring the lateral coordinate about the road's centerline — rather than negating
# it — is algebraically equivalent to the rotation for a symmetric road, but keeps
# axis ticks in the original positive range instead of going negative; it also leaves
# the road-boundary/divider values unchanged (they're symmetric about that centerline).
mirror_lateral(px) = left_edge + right_edge - px
rotate(pt) = (pt[2], mirror_lateral(pt[1]))

all_pts = [pt for traj in solved_positions for pts in traj for pt in pts]
along_road = [p[2] for p in all_pts]
margin = 1.0
road_lo, road_hi = minimum(along_road) - margin, maximum(along_road) + margin

"""Place a car icon (already oriented to point in +x by assets/prepare_car_icons.jl) at
`pos`, `car_length` long (m), preserving that image's own aspect ratio for its width."""
function car_image!(ax, pos; img, car_length = 1.6)
    cx, cy = pos
    aspect_ratio = size(img, 1) / size(img, 2)  # (length px) / (width px)
    hl = car_length / 2
    hw = car_length / aspect_ratio / 2
    image!(ax, (cx - hl) .. (cx + hl), (cy - hw) .. (cy + hw), img)
end

fig = Figure(; size = (620, 350))
ax = Axis(
    fig[1, 1];
    xlabel = "distance along road (m)",
    ylabel = "lateral position (m)",
    aspect = (road_hi - road_lo) / (right_edge - left_edge),
)
xlims!(ax, road_lo, road_hi)
ylims!(ax, left_edge - 0.5, right_edge + 0.5)

# Road: solid outer boundaries, dashed internal lane divider(s) (now horizontal).
hlines!(ax, [left_edge, right_edge]; color = :black, linewidth = 2)
for k in 1:(length(lane_centers) - 1)
    divider = (lane_centers[k] + lane_centers[k + 1]) / 2
    hlines!(ax, [divider]; color = :gray, linestyle = :dash, linewidth = 1)
end

player_colors = (:steelblue, :firebrick)

# Faint fan first (background), hero on top.
for traj in fan, (ii, pts) in enumerate(traj)
    lines!(ax, rotate.(pts); color = (player_colors[ii], 0.12), linewidth = 1.5)
end
for (ii, pts) in enumerate(hero)
    rotated = rotate.(pts)
    lines!(ax, rotated; color = player_colors[ii], linewidth = 3)
    car_image!(ax, rotated[1]; img = CAR_IMAGES[player_colors[ii]])
end

outpath = joinpath(@__DIR__, "teaser_schematic.pdf")
save(outpath, fig)
@info "Wrote $outpath ($(length(solved_positions))/$(N_FAN + 1) instances solved and plotted)"
