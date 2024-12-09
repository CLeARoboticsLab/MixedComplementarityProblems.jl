module MixedComplementarityProblems

using SparseArrays: SparseArrays
using FastDifferentiation: FastDifferentiation as FD
using Symbolics: Symbolics
using LinearAlgebra: LinearAlgebra, I, norm, eigvals
using BlockArrays: blocks, blocksizes
using TrajectoryGamesBase: to_blockvector

include("SymbolicUtils.jl")
include("sparse_utils.jl")
include("mcp.jl")
include("solver.jl")
include("game.jl")
include("AutoDiff.jl")

export PrimalDualMCP, solve, ParametricGame, OptimizationProblem

end # module MixedComplementarityProblems
