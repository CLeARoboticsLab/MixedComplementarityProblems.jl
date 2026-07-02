using MixedComplementarityProblems
const MCP = MixedComplementarityProblems
using KernelAbstractions: CPU

# small QP-style MCP (same as the tests/spike)
M = [2.0 1 0; 1 2 1; 0 1 2]
A = [1.0 0 0; 0 1 0; 0 0 1]
b = [1.0, 1, 1]
G(x, y; θ) = M * x - θ - transpose(A) * y
H(x, y; θ) = A * x - b
mcp = MCP.PrimalDualMCP(G, H;
    unconstrained_dimension = 3, constrained_dimension = 3, parameter_dimension = 3)

B = 1000
cache = MCP.materialize(mcp, MCP.BatchedSparse(), CPU(); batch_size = B)
println("d            = ", cache.d)
println("nnz          = ", cache.nnz)
println("batch_size   = ", cache.batch_size)
println("size(nzval)  = ", size(cache.nzval), "  (", typeof(cache.nzval), ")")
println("# diag_nz    = ", length(cache.diag_nz), " of d=", cache.d, " diagonal slots present")
println("pattern/factor (deferred): ", cache.pattern, " / ", cache.factor)

# fallback error path for an unimplemented strategy
try
    MCP.materialize(mcp, MCP.BatchedDense(), CPU(); batch_size = B)
catch e
    println("BatchedDense fallback: ", sprint(showerror, e))
end
