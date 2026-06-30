"""
GPU batched-sparse linear solver for the `BatchedSparse` strategy, via NVIDIA cuDSS
(CUDSS.jl). This is option D1(a) in `docs/gpu_kkt_design.md`: one symbolic
factorization shared across the batch, general (any MCP), least code.

This extension loads only when BOTH `CUDA` and `CUDSS` are present. It adds GPU methods
to the same verbs the CPU backend implements — `_materialize_linsolve`, `factorize!`,
`ldiv!` — so the §5 interior-point loop and the residual/Jacobian assembly kernels
(already device-portable via KernelAbstractions) are reused unchanged.

────────────────────────────────────────────────────────────────────────────────────
DESIGN (to implement + verify on a CUDA device — bodies below currently error):

Phase mapping (mirrors the CPU analyze/factorize/solve split):
  • _materialize_linsolve  → cuDSS "analysis"  (symbolic, ONCE, shared across batch)
  • factorize!             → cuDSS "factorization" (numeric, per Newton iteration)
  • ldiv!                  → cuDSS "solve"

Data layout — the one real subtlety:
  `jacobian!` fills `cache.nzval` (nnz × B) in the CSC order of the symbolic ∇F (the
  order of `findnz(sparse_jacobian(...))`, column-major). cuDSS consumes CSR. The
  pattern is shared across the batch, so compute ONCE on the host a permutation `perm`
  mapping each CSR value slot to its CSC source index, plus the CSR structure
  (rowPtr, colVal); move them to the device in `_materialize_linsolve`. Then each
  `factorize!` gathers `csr_vals[:, b] = nzval[perm, b]` on-device (a single kernel /
  broadcast over the batch) before the cuDSS numeric factorization — no host transfer,
  no structural rebuild. Build `perm`/`rowPtr`/`colVal` from
      B = sparse(cols, rows, 1:nnz, d, d)   # CSC of the transpose == CSR of ∇F
      rowPtr = B.colptr;  colVal = B.rowval;  perm = nonzeros(B)
  (a tested host helper for this belongs in the base package once the approach is
  confirmed against the installed CUDSS.jl).

Batched cuDSS:
  Use cuDSS uniform-batched mode (all instances share the pattern). Confirm the exact
  CUDSS.jl batched API for the installed version (≈0.6–0.8): how a batch of CSR value
  arrays + RHS/solution matrices are passed to `CudssSolver` / `cudss(...)`, and
  whether the (d × B) device arrays the solver hands us (`F`, `δz`) can be used as the
  batched RHS/solution directly or need reshaping. The (d × K × B) `ldiv!` (parameter
  sensitivities) is the K-RHS generalization of the same solve.

Other notes:
  • Hoist `KernelAbstractions.synchronize` out of residual!/jacobian! for GPU (it
    currently stalls every iteration); sync only where the solver reads back to host
    (the convergence check). See the NOTE in src/batched_solver.jl.
  • The cache's `pattern`/`factor` fields are untyped (P, F), so the GPU cache can hold
    a CudssSolver + CSR buffers + `perm` here without changing the struct.
────────────────────────────────────────────────────────────────────────────────────
"""
module MixedComplementarityProblemsCUDSSExt

using MixedComplementarityProblems: MixedComplementarityProblems, BatchedSparseCache
using CUDA: CUDA
using CUDSS: CUDSS
using KernelAbstractions: KernelAbstractions

const MCP = MixedComplementarityProblems

# Shared error until the cuDSS bodies are implemented and validated on a CUDA device.
_todo(verb) = error(
    "$verb for BatchedSparse on the CUDA backend is scaffolded but not yet " *
    "implemented — see the DESIGN block in MixedComplementarityProblemsCUDSSExt and " *
    "D1(a) in docs/gpu_kkt_design.md. Implement against CUDSS.jl on a CUDA device.",
)

# Analyze phase: build the device CSR pattern + CSC→CSR permutation, create the cuDSS
# solver, run cuDSS "analysis" (shared symbolic), and return (pattern, factor) to store
# in the cache. `factor` should carry the CudssSolver + the device CSR value buffer(s).
function MCP._materialize_linsolve(
    ::CUDA.CUDABackend,
    rows,
    cols,
    nnz,
    d,
    batch_size,
)
    _todo("_materialize_linsolve")
end

# Numeric factorization: gather cache.nzval (CSC order) into the CSR value buffer via
# the stored permutation, then run cuDSS "factorization" over the batch. (`active` is the
# CPU active-set optimization; the batched cuDSS factorization processes the whole batch,
# so a GPU implementation can ignore it.)
function MCP.factorize!(cache::BatchedSparseCache{<:CUDA.CUDABackend}; active = nothing)
    _todo("factorize!")
end

# Solve ∇F · out = rhs over the batch via cuDSS "solve". (d × B) RHS/solution.
function MCP.ldiv!(out, cache::BatchedSparseCache{<:CUDA.CUDABackend}, rhs; active = nothing)
    _todo("ldiv!")
end

# Multi-RHS solve (d × K × B), used by parameter sensitivities (solve_jacobian_θ).
function MCP.ldiv!(
    out::AbstractArray{<:Any,3},
    cache::BatchedSparseCache{<:CUDA.CUDABackend},
    rhs::AbstractArray{<:Any,3},
)
    _todo("ldiv! (multi-rhs)")
end

end # module
