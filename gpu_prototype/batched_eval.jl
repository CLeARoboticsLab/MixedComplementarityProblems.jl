# Micro-prototype for batched MCP residual + Jacobian evaluation.
#
# Goal: validate that Symbolics-generated (in-place) functions run inside a
# KernelAbstractions kernel, one MCP instance per thread, with:
#   - residual F written into an (dim x B) buffer, and
#   - Jacobian stored as ONE shared sparsity pattern (rows, cols) + an (nnz x B)
#     value matrix (each thread fills its own column).
# Backend: KA CPU() — exercises the real @kernel code path (GPU register pressure
# and the batched-sparse *solve* are explicitly out of scope here).

using SymbolicTracingUtils
const STU = SymbolicTracingUtils
const Sym = STU.Symbolics
using SparseArrays
using LinearAlgebra
using KernelAbstractions
using Random

# ---------------------------------------------------------------------------
# 1. Build a small-but-nontrivial MCP symbolically:  F = [G; H - s; s.*y - ϵ]
#    with G = M x - θ - Aᵀy,  H = A x - b.
# ---------------------------------------------------------------------------
backend = STU.SymbolicsBackend()
n = 3          # unconstrained dim
m = 3          # constrained dim
x = STU.make_variables(backend, :x, n)
y = STU.make_variables(backend, :y, m)
s = STU.make_variables(backend, :s, m)
θ = STU.make_variables(backend, :θ, n)
ϵ = only(STU.make_variables(backend, :ϵ, 1))

const M = [2.0 1 0; 1 2 1; 0 1 2]
const A = Matrix(1.0I, m, n)
const bvec = ones(m)

G = M * x - θ - A' * y
H = A * x - bvec
F_sym = [G; H .- s; s .* y .- ϵ]
z = [x; y; s]
d = length(F_sym)                       # residual dimension

# Shared sparsity pattern + symbolic nonzero values of ∇F/∂z.
J_sym = Sym.sparsejacobian(F_sym, z)
ROWS, COLS, Vsym = findnz(J_sym)
const NNZ = length(Vsym)
println("dim(F) = $d, nnz(∇F_z) = $NNZ")

# ---------------------------------------------------------------------------
# 2. Compile in-place, kernel-safe functions (SerialForm + cse, NO sharding).
# ---------------------------------------------------------------------------
const F_INPLACE! = Sym.build_function(
    F_sym, x, y, s, θ, ϵ;
    expression = Val{false}, parallel = Sym.SerialForm(), cse = true,
)[2]
const J_INPLACE! = Sym.build_function(
    Vsym, x, y, s, θ, ϵ;
    expression = Val{false}, parallel = Sym.SerialForm(), cse = true,
)[2]

# ---------------------------------------------------------------------------
# 3. KernelAbstractions kernel: one instance per thread.
# ---------------------------------------------------------------------------
@kernel function assemble_batch!(Fout, Jout, X, Y, S, Θ, E)
    b = @index(Global)
    @views F_INPLACE!(Fout[:, b], X[:, b], Y[:, b], S[:, b], Θ[:, b], E[b])
    @views J_INPLACE!(Jout[:, b], X[:, b], Y[:, b], S[:, b], Θ[:, b], E[b])
end

# ---------------------------------------------------------------------------
# 4. Run a batch on the CPU backend and validate.
# ---------------------------------------------------------------------------
Random.seed!(1)
B = 1000
X = randn(n, B); Y = rand(m, B) .+ 0.1; S = rand(m, B) .+ 0.1
Θ = randn(n, B); E = fill(0.1, B)

Fout = zeros(d, B); Jout = zeros(NNZ, B)

ka = CPU()
kern = assemble_batch!(ka)
kern(Fout, Jout, X, Y, S, Θ, E; ndrange = B)
KernelAbstractions.synchronize(ka)

# (a) kernel == serial reference loop using the SAME compiled functions.
refF = zeros(d, B); refJ = zeros(NNZ, B)
for b in 1:B
    @views F_INPLACE!(refF[:, b], X[:, b], Y[:, b], S[:, b], Θ[:, b], E[b])
    @views J_INPLACE!(refJ[:, b], X[:, b], Y[:, b], S[:, b], Θ[:, b], E[b])
end
println("kernel F == serial F : ", Fout ≈ refF)
println("kernel J == serial J : ", Jout ≈ refJ)

# (b) Jacobian sanity: reconstruct instance 1's sparse ∇F and check against a
#     central finite-difference directional derivative  ∇F·v ≈ (F(z+hv)-F(z-hv))/2h.
J1 = sparse(ROWS, COLS, Jout[:, 1], d, length(z))
let b = 1, h = 1e-6
    z1 = [X[:, b]; Y[:, b]; S[:, b]]
    function Feval(zz)
        out = zeros(d)
        xp, yp, sp = zz[1:n], zz[n+1:n+m], zz[n+m+1:end]
        F_INPLACE!(out, xp, yp, sp, Θ[:, b], E[b])
        out
    end
    maxerr = 0.0
    for _ in 1:5
        v = randn(length(z))
        fd = (Feval(z1 .+ h .* v) .- Feval(z1 .- h .* v)) ./ (2h)
        maxerr = max(maxerr, norm(J1 * v - fd, Inf))
    end
    println("max |∇F·v - finite_diff| over 5 dirs (instance 1): ", maxerr)
end

# (c) quick throughput sanity (compile-warmed).
kern(Fout, Jout, X, Y, S, Θ, E; ndrange = B); KernelAbstractions.synchronize(ka)
t = @elapsed (kern(Fout, Jout, X, Y, S, Θ, E; ndrange = B); KernelAbstractions.synchronize(ka))
println("batched assemble of B=$B instances on CPU backend: ", round(t * 1e3, digits = 3), " ms")
