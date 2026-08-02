""" Verifies the differentiability claim in docs/paper/main.tex's "Differentiable
Solving" section: AD gradients (via ForwardDiff, exercising the implicit-function-
theorem sensitivity machinery in src/AutoDiff.jl) match central finite differences,
on the running-example lane-change trajectory game (same T=10 family as Table I/teaser).

Run with: julia --project=benchmark benchmark/results/scripts/ad_gradient_check.jl
"""

const REPO = get(ENV, "REPO_ROOT", normpath(joinpath(@__DIR__, "..", "..", "..")))
include(joinpath(REPO, "benchmark", "SolverBenchmarks.jl"))
using .SolverBenchmarks
const B = SolverBenchmarks
const Utils = B.TrajectoryGameBenchmarkUtils

using MixedComplementarityProblems: MixedComplementarityProblems
const MCP = MixedComplementarityProblems
import ForwardDiff
import Random

const HORIZON = 10

(; environment) = Utils.setup_road_environment()
game = Utils.setup_trajectory_game(; environment)
components = Utils.build_mcp_components(; game, horizon = HORIZON, params_per_player = 1)
mcp = MCP.PrimalDualMCP(
    components.K_symbolic, components.z_symbolic, components.θ_symbolic,
    components.lower_bounds, components.upper_bounds;
    η_symbolic = components.η_symbolic, compute_sensitivities = true,
)

solve_at(θ) = MCP.solve(MCP.InteriorPoint(), mcp, θ; tol = 1e-4, regularize_linear_solve = :identity)
f(θ) = sum(solve_at(θ).x .^ 2) + sum(solve_at(θ).y .^ 2)

function central_fd(f, θ; h = 1e-6)
    map(eachindex(θ)) do i
        θp = copy(θ)
        θp[i] += h
        θm = copy(θ)
        θm[i] -= h
        (f(θp) - f(θm)) / 2h
    end
end

rng = Random.MersenneTwister(1)
n_ok = 0
max_relerrs = Float64[]
norm_relerrs = Float64[]
attempt = 0
while n_ok < 20 && attempt < 100
    global attempt += 1
    θ0 = B.generate_random_parameter(B.TrajectoryGameBenchmark(); rng, horizon = HORIZON)
    sol0 = solve_at(θ0)
    sol0.status == :solved || continue
    global n_ok += 1

    grad_ad = ForwardDiff.gradient(f, θ0)
    grad_fd = central_fd(f, θ0)

    push!(max_relerrs, maximum(abs.(grad_ad .- grad_fd) ./ max.(abs.(grad_fd), 1e-8)))
    push!(norm_relerrs, sqrt(sum((grad_ad .- grad_fd) .^ 2)) / sqrt(sum(grad_fd .^ 2)))
end

println("checked $n_ok converged instances (of $attempt attempts)")
println("componentwise max relative error: worst=$(maximum(max_relerrs))  median=$(sort(max_relerrs)[n_ok÷2+1])")
println("gradient-norm relative error:     worst=$(maximum(norm_relerrs))  median=$(sort(norm_relerrs)[n_ok÷2+1])")
