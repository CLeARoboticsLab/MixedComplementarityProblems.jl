# Validate residual!/jacobian! (KA CPU backend) against the existing single-instance
# CPU evaluators (mcp.F!, mcp.∇F_z!) and a finite-difference check.
using MixedComplementarityProblems
const MCP = MixedComplementarityProblems
using KernelAbstractions: CPU
using SparseArrays
using LinearAlgebra
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
B = 500
cache = MCP.materialize(mcp, MCP.BatchedSparse(), dev; batch_size = B)

Random.seed!(2)
X = randn(n, B); Y = rand(m, B) .+ 0.1; S = rand(m, B) .+ 0.1
Θ = randn(n, B); ϵ = fill(0.1, B)

# --- residual! vs mcp.F! -------------------------------------------------------
F = zeros(d, B)
MCP.residual!(F, mcp, X, Y, S, Θ, ϵ; device = dev)
refF = zeros(d, B)
for bb in 1:B
    buf = zeros(d)
    mcp.F!(buf, X[:, bb], Y[:, bb], S[:, bb]; θ = Θ[:, bb], ϵ = ϵ[bb])
    refF[:, bb] .= buf
end
println("residual! matches mcp.F!         : ", F ≈ refF)

# --- jacobian! (η = 0) vs mcp.∇F_z! nonzeros ----------------------------------
η0 = zeros(B)
MCP.jacobian!(cache, mcp, X, Y, S, Θ, ϵ, η0; device = dev)
let maxerr = 0.0
    for bb in 1:B
        Jbuf = mcp.∇F_z!.result_buffer
        mcp.∇F_z!(Jbuf, X[:, bb], Y[:, bb], S[:, bb]; θ = Θ[:, bb], ϵ = ϵ[bb])
        maxerr = max(maxerr, maximum(abs, nonzeros(Jbuf) .- cache.nzval[:, bb]))
    end
    println("jacobian! (η=0) matches ∇F_z!     : maxerr = ", maxerr)
end

# --- finite-difference sanity on instance 1 -----------------------------------
let bb = 1, h = 1e-6
    J1 = sparse(cache.rows, cache.cols, cache.nzval[:, bb], d, d)
    function Feval(z)
        out = zeros(d)
        mcp.F!(out, z[1:n], z[n+1:n+m], z[n+m+1:end]; θ = Θ[:, bb], ϵ = ϵ[bb])
        out
    end
    z1 = [X[:, bb]; Y[:, bb]; S[:, bb]]
    e = 0.0
    for _ in 1:5
        v = randn(d)
        fd = (Feval(z1 .+ h .* v) .- Feval(z1 .- h .* v)) ./ (2h)
        e = max(e, norm(J1 * v - fd, Inf))
    end
    println("FD check ∇F·v vs (F(z+hv)-F(z-hv))/2h: ", e)
end

# --- fused diagonal regularization (η = 0.7) ----------------------------------
ηc = fill(0.7, B)
MCP.jacobian!(cache, mcp, X, Y, S, Θ, ϵ, ηc; device = dev)
Jbuf = mcp.∇F_z!.result_buffer
mcp.∇F_z!(Jbuf, X[:, 1], Y[:, 1], S[:, 1]; θ = Θ[:, 1], ϵ = ϵ[1])
diff = cache.nzval[:, 1] .- nonzeros(Jbuf)
offdiag = setdiff(1:cache.nnz, cache.diag_nz)
println("η added to diag_nz entries (≈0.7) : ",
    all(isapprox.(diff[cache.diag_nz], 0.7; atol = 1e-12)))
println("off-diagonal entries unchanged    : ",
    all(abs.(diff[offdiag]) .< 1e-12))
