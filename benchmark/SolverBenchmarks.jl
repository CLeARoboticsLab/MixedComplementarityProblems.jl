"Module for benchmarking different solvers against one another."
module SolverBenchmarks

using MixedComplementarityProblems: MixedComplementarityProblems
using ParametricMCPs: ParametricMCPs
using BlockArrays: BlockArrays, mortar
using Random: Random
using Statistics: Statistics
using Distributions: Distributions
using LazySets: LazySets
using PATHSolver: PATHSolver
using ProgressMeter: @showprogress

abstract type BenchmarkType end
struct QuadraticProgramBenchmark <: BenchmarkType end
struct TrajectoryGameBenchmark <: BenchmarkType end

include("quadratic_program_benchmark.jl")
include("trajectory_game_benchmark.jl")
include("path.jl")

end # module SolverBenchmarks
