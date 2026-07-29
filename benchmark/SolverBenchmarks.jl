"Module for benchmarking different solvers against one another."
module SolverBenchmarks

using MixedComplementarityProblems: MixedComplementarityProblems
using ParametricMCPs: ParametricMCPs
using BlockArrays: BlockArrays, mortar
using LinearAlgebra: norm
using Random: Random
using Statistics: Statistics
using Distributions: Distributions
using LazySets: LazySets
using PATHSolver: PATHSolver
using ProgressMeter: @showprogress
using Symbolics: Symbolics
using KernelAbstractions: KernelAbstractions
using Adapt: Adapt

abstract type BenchmarkType end
struct QuadraticProgramBenchmark <: BenchmarkType end
struct TrajectoryGameBenchmark <: BenchmarkType end

"""Optional per-`benchmark_type` initial guess for `BatchedInteriorPoint`'s `X₀`. Default
is `nothing` (cold zero start); `TrajectoryGameBenchmark` overrides this (see
`trajectory_game_benchmark.jl`) since the trivial cold start scales badly with horizon.
"""
generate_initial_guess(::BenchmarkType, mcp, Θ, device; kwargs...) = nothing

include("quadratic_program_benchmark.jl")
include("trajectory_game_benchmark.jl")
include("path.jl")
include("batched_benchmark.jl")

end # module SolverBenchmarks
