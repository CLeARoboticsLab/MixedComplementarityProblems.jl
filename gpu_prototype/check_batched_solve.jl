# Validate the batched interior-point solver: (1) at B=1 it matches the unbatched
# solve(::InteriorPoint, ...); (2) a genuine batch with varied θ all converges.
using MixedComplementarityProblems
const MCP = MixedComplementarityProblems
using KernelAbstractions: CPU
using LinearAlgebra
using Random

# QP from the test suite: min 0.5 xᵀMx - θᵀx s.t. Ax - b ≥ 0.
M = [2.0 1; 1 2]
A = [1.0 0; 0 1]
b = [1.0, 1]
G(x, y; θ) = M * x - θ - transpose(A) * y
H(x, y; θ) = A * x - b
mcp = MCP.PrimalDualMCP(G, H;
    unconstrained_dimension = 2, constrained_dimension = 2, parameter_dimension = 2,
    compute_kernel_evaluators = true)

θ = [-0.5, 0.5]

# --- (1) B = 1 vs unbatched solver -------------------------------------------
unbatched = MCP.solve(MCP.InteriorPoint(), mcp, θ)
batched = MCP.solve(MCP.BatchedInteriorPoint(), mcp, reshape(θ, :, 1); device = CPU())

println("unbatched: x=", round.(unbatched.x; digits = 5), " status=", unbatched.status,
    " total_iters=", unbatched.total_iters)
println("batched  : x=", round.(batched.x[:, 1]; digits = 5), " status=", batched.status[1],
    " total_iters=", batched.total_iters)
# Both target tol=1e-4 on the KKT residual, so their solutions agree to ~tol; compare
# at a few× the tolerance (comparing two approximate solutions, errors add).
println("x match (atol 1e-3): ", isapprox(batched.x[:, 1], unbatched.x; atol = 1e-3))
println("y match (atol 1e-3): ", isapprox(batched.y[:, 1], unbatched.y; atol = 1e-3))
println("s match (atol 1e-3): ", isapprox(batched.s[:, 1], unbatched.s; atol = 1e-3))

# --- (2) genuine batch, varied θ ---------------------------------------------
Random.seed!(3)
Bn = 300
Θ = randn(2, Bn)
sol = MCP.solve(MCP.BatchedInteriorPoint(), mcp, Θ; device = CPU())

function kkt_ok(x, y, θb)
    Gres = M * x - θb - transpose(A) * y
    Hval = A * x - b
    all(abs.(Gres) .< 5e-3) && all(Hval .> -5e-3) && all(y .> -5e-3) &&
        abs(sum(y .* Hval)) < 5e-3
end
all_ok = all(bb -> kkt_ok(sol.x[:, bb], sol.y[:, bb], Θ[:, bb]), 1:Bn)
println("\nbatch B=$Bn : all status solved = ", all(sol.status .== :solved),
    " ; all satisfy KKT = ", all_ok,
    " ; max kkt_error = ", maximum(sol.kkt_error))

# --- timing (warmed) ----------------------------------------------------------
MCP.solve(MCP.InteriorPoint(), mcp, θ)
MCP.solve(MCP.BatchedInteriorPoint(), mcp, reshape(θ, :, 1); device = CPU())
MCP.solve(MCP.BatchedInteriorPoint(), mcp, Θ; device = CPU())
t_unbatched = @elapsed MCP.solve(MCP.InteriorPoint(), mcp, θ)
t_b1 = @elapsed MCP.solve(MCP.BatchedInteriorPoint(), mcp, reshape(θ, :, 1); device = CPU())
t_bN = @elapsed MCP.solve(MCP.BatchedInteriorPoint(), mcp, Θ; device = CPU())
println("\ntiming: unbatched B=1 ", round(t_unbatched * 1e3, digits = 3), " ms | ",
    "batched B=1 ", round(t_b1 * 1e3, digits = 3), " ms | ",
    "batched B=$Bn ", round(t_bN * 1e3, digits = 3), " ms (",
    round(t_bN * 1e6 / Bn, digits = 2), " µs/instance)")
