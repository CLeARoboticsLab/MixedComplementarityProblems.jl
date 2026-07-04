"""
GPU batched-sparse linear solver for the `BatchedSparse` strategy, via NVIDIA cuDSS
(CUDSS.jl). This is option D1(a) in `docs/gpu_kkt_design.md`: one symbolic
factorization shared across the batch, general (any MCP), least code.

This extension loads only when BOTH `CUDA` and `CUDSS` are present. It adds GPU methods
to the same verbs the CPU backend implements — `_materialize_linsolve`, `factorize!`,
`ldiv!` — so the §5 interior-point loop and the residual/Jacobian assembly kernels
(already device-portable via KernelAbstractions) are reused unchanged.

────────────────────────────────────────────────────────────────────────────────────
Confirmed against the installed CUDSS.jl 0.6.5 (see `ext.jl` in that package): it
already exposes a "uniform batch" mode through the ordinary `LinearAlgebra.lu`/`lu!`/
`ldiv!` interface, keyed entirely off array shapes — no manual `CudssMatrix`/`cudss(...)`
phase juggling needed:

  • `LinearAlgebra.lu(A::CuSparseMatrixCSR)` detects a batch via
    `nbatch = length(A.nzVal) ÷ length(A.colVal)` (`A.rowPtr`/`A.colVal` are the ONE
    shared pattern; `A.nzVal` is `nbatch` per-instance blocks concatenated) and runs
    cuDSS "analysis" + "factorization" (`CudssSolver(A, "G", 'F')`, general/full since
    ∇F is not symmetric).
  • `LinearAlgebra.lu!(solver, A)` reuses the symbolic factorization from `solver`
    and does a numeric "factorization"/"refactorization" from `A`'s current values.
  • `LinearAlgebra.ldiv!(out, solver, rhs)` accepts `rhs`/`out` as plain `(d × B)` or
    `(d × K × B)` `CuArray`s directly — this is EXACTLY the layout our verb set already
    uses, so no reshaping is needed at the call site.

Data layout — the one real subtlety:
  `jacobian!` fills `cache.nzval` (nnz × B) in the CSC order of the symbolic ∇F (the
  order of `findnz(sparse_jacobian(...))`, column-major). cuDSS wants CSR. The pattern
  is shared across the batch, so we compute ONCE on the host (in `_materialize_linsolve`)
  a permutation `perm` mapping each CSR value slot to its CSC source index, plus the CSR
  structure (`rowPtr`, `colVal`):
      csc_of_transpose = sparse(cols, rows, 1:nnz, d, d)   # CSC of Aᵀ == CSR of A
      rowPtr = csc_of_transpose.colptr;  colVal = csc_of_transpose.rowval
      perm   = nonzeros(csc_of_transpose)
  `perm[k]` is the `cache.nzval` row index whose value belongs at CSR slot `k`. Moved to
  the device once; every `factorize!` gathers `nzval_csr .= view(cache.nzval, perm, :)` —
  a single fused broadcast/gather over the whole batch, no host transfer, no structural
  rebuild. `nzval_csr` is the batched `CuSparseMatrixCSR`'s `.nzVal` buffer (built once
  in `_materialize_linsolve` via `vec`, so writing into `nzval_csr` updates it in place —
  no per-iteration `CuSparseMatrixCSR` reallocation).

Other notes:
  • `KernelAbstractions.synchronize` is hoisted out of `residual!`/`jacobian!` already at
    the call-site level (see the NOTE in src/batched_solver.jl); this extension doesn't
    add any extra sync beyond what `LinearAlgebra.lu!`/`ldiv!` themselves require.
  • The cache's `pattern`/`factor` fields are untyped (P, F): `pattern` holds the shared
    CSR buffers + `perm` + the persistent `CuSparseMatrixCSR` view; `factor` is a
    `Ref{Union{Nothing,CUDSS.CudssSolver}}`, lazily built on the first `factorize!` (from
    real Jacobian values, mirroring the CPU KLU backend) and refactored in place
    thereafter via `lu!`.
  • `active` (the CPU active-set optimization) is accepted but ignored: cuDSS's batched
    factorization always processes the whole batch, so there is nothing to skip.
  • The FIRST `factorize!` manually replicates `LinearAlgebra.lu`'s internals (build
    `CudssSolver` + `CudssMatrix`s, call the "analysis"/"factorization" phases) instead of
    calling `lu()` directly, solely to set `"factorization_alg" = "algo1"` beforehand —
    `lu()` doesn't expose a way to pass config through. Verified empirically (both the QP
    and trajectory-game benchmarks, `benchmark/gpu/`): `algo1` gives identical
    `outer_iters`/`total_iters`/solved-counts (same numerics) as the default algorithm,
    while being 10-17% faster end-to-end. `algo2`-`algo5` are either unsupported for this
    matrix structure (`CUDSS_STATUS_NOT_SUPPORTED`) or slower; `reordering_alg`/
    `use_superpanels` were also swept and found not to help (see PR discussion).
────────────────────────────────────────────────────────────────────────────────────
"""
module MixedComplementarityProblemsCUDSSExt

using MixedComplementarityProblems: MixedComplementarityProblems, BatchedSparseCache
using CUDA: CUDA
using CUDA.CUSPARSE: CUSPARSE
using CUDSS: CUDSS
using KernelAbstractions: KernelAbstractions
using LinearAlgebra: LinearAlgebra
using SparseArrays: SparseArrays

const MCP = MixedComplementarityProblems

# Analyze phase (deferred): build the shared device CSR structure + CSC→CSR permutation
# and a persistent `CuSparseMatrixCSR` view over a fresh `(nnz × batch_size)` value
# buffer. The actual cuDSS "analysis" doesn't run here — like the CPU KLU backend, it's
# deferred to the first `factorize!`, once real Jacobian values are available.
function MCP._materialize_linsolve(
    ::CUDA.CUDABackend,
    rows,
    cols,
    nnz,
    d,
    batch_size,
)
    # CSC of the transpose == CSR of ∇F. `nonzeros(csc_of_transpose)[k]` is the CSC
    # (`cache.nzval`) row index whose value belongs at CSR slot `k`.
    csc_of_transpose = SparseArrays.sparse(cols, rows, 1:nnz, d, d)
    rowPtr = CUDA.CuVector{Int32}(SparseArrays.getcolptr(csc_of_transpose))
    colVal = CUDA.CuVector{Int32}(SparseArrays.rowvals(csc_of_transpose))
    perm = CUDA.CuVector{Int}(SparseArrays.nonzeros(csc_of_transpose))

    nzval_csr = CUDA.zeros(Float64, nnz, batch_size)
    A = CUSPARSE.CuSparseMatrixCSR(rowPtr, colVal, vec(nzval_csr), (d, d))

    pattern = (; A, perm, nzval_csr)
    factor = Ref{Union{Nothing,CUDSS.CudssSolver}}(nothing)
    (pattern, factor)
end

# Numeric factorization: gather cache.nzval (CSC order) into the shared CSR value buffer
# via the stored permutation, then run cuDSS factorization/refactorization over the whole
# batch (`active` is the CPU active-set optimization; cuDSS's batched factorization
# always processes the whole batch, so a GPU implementation ignores it).
function MCP.factorize!(cache::BatchedSparseCache{<:CUDA.CUDABackend}; active = nothing)
    pattern = cache.pattern
    pattern.nzval_csr .= view(cache.nzval, pattern.perm, :)
    if cache.factor[] === nothing
        # Manually replicates `LinearAlgebra.lu(pattern.A)` (see the module docstring)
        # solely to set `factorization_alg = "algo1"` first, which isn't reachable through
        # the `lu()` convenience wrapper.
        d = size(pattern.A, 1)
        nbatch = length(pattern.A.nzVal) ÷ length(pattern.A.colVal)
        solver = CUDSS.CudssSolver(pattern.A, "G", 'F')
        (nbatch > 1) && CUDSS.cudss_set(solver, "ubatch_size", nbatch)
        CUDSS.cudss_set(solver, "factorization_alg", "algo1")
        x = CUDSS.CudssMatrix(Float64, d; nbatch)
        b = CUDSS.CudssMatrix(Float64, d; nbatch)
        CUDSS.cudss("analysis", solver, x, b; asynchronous = true)
        CUDSS.cudss("factorization", solver, x, b; asynchronous = false)
        cache.factor[] = solver
    else
        LinearAlgebra.lu!(cache.factor[], pattern.A)
    end
    cache
end

# Solve ∇F · out = rhs over the batch via cuDSS. `out`/`rhs` are plain (d × B) CuArrays —
# CUDSS.jl's `ldiv!` detects the batch from the solver's own `nbatch` and accepts this
# shape directly, no reshaping needed.
function MCP.ldiv!(out, cache::BatchedSparseCache{<:CUDA.CUDABackend}, rhs; active = nothing)
    LinearAlgebra.ldiv!(out, cache.factor[], rhs)
    out
end

# Multi-RHS solve (d × K × B), used by parameter sensitivities (solve_jacobian_θ). Same
# story: CUDSS.jl's `ldiv!` accepts a 3D (d × K × B) CuArray directly.
function MCP.ldiv!(
    out::AbstractArray{<:Any,3},
    cache::BatchedSparseCache{<:CUDA.CUDABackend},
    rhs::AbstractArray{<:Any,3},
)
    LinearAlgebra.ldiv!(out, cache.factor[], rhs)
    out
end

end # module
