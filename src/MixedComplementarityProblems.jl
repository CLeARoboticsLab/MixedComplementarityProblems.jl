module MixedComplementarityProblems

using SparseArrays: SparseArrays
using LinearAlgebra: LinearAlgebra, I, norm, eigvals
using BlockArrays: blocks, blocksizes
using TrajectoryGamesBase: to_blockvector
using SymbolicTracingUtils: SymbolicTracingUtils as SymbolicTracingUtils
using LinearSolve: LinearProblem, init, solve!, KrylovJL_GMRES, UMFPACKFactorization
using SciMLBase: SciMLBase
using KernelAbstractions: KernelAbstractions, @kernel, @index

include("mcp.jl")
include("solver.jl")
include("kkt_system.jl")
include("game.jl")
include("AutoDiff.jl")

export PrimalDualMCP, solve, ParametricGame, OptimizationProblem

end # module MixedComplementarityProblems
