# Validate the full BatchedSparse CPU linear solve: jacobian! -> factorize! -> ldiv!,
# checking the batched Newton step against per-instance dense solves, and confirming
# the symbolic-reuse (lu!) path works across repeated factorize! calls.
using MixedComplementarityProblems
const MCP = MixedComplementarityProblems
using KernelAbstractions: CPU
using LinearAlgebra
using SparseArrays
using Random

M = [2.0 1 0; 1 2 1; 0 1 2]
A = [1.0 0 0; 0 1 0; 0 0 1]
b = [1.0, 1, 1]
G(x, y; θ) = M * x - θ - transpose(A) * y
H(x, y; θ) = A * x - b
mcp = MCP.PrimalDualMCP(G, H;
    unconstrained_dimension = 3, constrained_dimension = 3, parameter_dimension = 3,
    compute_kernel_evaluators = true)

dev = CPU()
n, m = 3, 3
d = n + 2m
B = 400
cache = MCP.materialize(mcp, MCP.BatchedSparse(), dev; batch_size = B)

Random.seed!(7)
X = randn(n, B); Y = rand(m, B) .+ 0.2; S = rand(m, B) .+ 0.2
Θ = randn(n, B); ϵ = fill(0.1, B); η = zeros(B)

F = zeros(d, B)
δz = zeros(d, B)

function newton_step!()
    MCP.residual!(F, mcp, X, Y, S, Θ, ϵ; device = dev)
    MCP.jacobian!(cache, mcp, X, Y, S, Θ, ϵ, η; device = dev)
    MCP.factorize!(cache)
    MCP.ldiv!(δz, cache, -F)
end

# First Newton step.
newton_step!()
let linres = 0.0, densediff = 0.0
    for bb in 1:B
        Asp = sparse(cache.rows, cache.cols, cache.nzval[:, bb], d, d)
        linres = max(linres, norm(Asp * δz[:, bb] + F[:, bb], Inf))      # ∇F δz = -F ?
        densediff = max(densediff, norm(δz[:, bb] - (Matrix(Asp) \ (-F[:, bb])), Inf))
    end
    println("max ||∇F·δz + F||                : ", linres)
    println("max diff vs dense per-instance \\ : ", densediff)
end

# Second step at a perturbed iterate: exercises lu! refactorization (symbolic reuse)
# on the already-built factor objects.
X .+= 0.3 .* δz[1:n, :]
newton_step!()
let linres = 0.0
    for bb in 1:B
        Asp = sparse(cache.rows, cache.cols, cache.nzval[:, bb], d, d)
        linres = max(linres, norm(Asp * δz[:, bb] + F[:, bb], Inf))
    end
    println("max ||∇F·δz + F|| (2nd step)     : ", linres)
end
