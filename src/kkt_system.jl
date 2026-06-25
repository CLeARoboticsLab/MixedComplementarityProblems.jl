""" KKTSystem layer (GPU/batched solver).

This file defines the abstraction boundary described in `docs/gpu_kkt_design.md`.
The interior-point solver is written ONCE against a small verb set
`{residual!, jacobian!, factorize!, ldiv!}` plus generic batched array ops.
Everything regime-specific — the numeric representation of `∇F`, the device, and
the linear-solve algorithm — lives behind those verbs in a `KKTStrategy` and its
associated cache.

Two orthogonal axes:
  - `device`  : a KernelAbstractions backend (`CPU()`, `CUDABackend()`, …) that
                determines array type and where kernels run.
  - `strategy`: a `KKTStrategy` that determines the `∇F` representation and the
                linear solver used for it.

Invariant: `F` and the Newton step `δz` are plain `(d × B)` device arrays owned by
the SOLVER; the assembled `∇F` and its factorization never leave the cache. That
single invariant is what makes the strategies interchangeable.
"""

# ---------------------------------------------------------------------------
# Strategies: numeric representation of ∇F + the linear solver used for it.
# ---------------------------------------------------------------------------

"Numeric representation of `∇F` + the linear solver used for it. See `docs/gpu_kkt_design.md`."
abstract type KKTStrategy end

"`∇F` as a dense `(d × d × B)` array, solved by batched dense LU. Viable ONLY for
small per-instance dimension `d` (≲ a few hundred) — beyond that, register
pressure and `O(d³)`/memory make it infeasible. [not yet implemented]"
struct BatchedDense <: KKTStrategy end

"`∇F` as ONE shared sparsity pattern `(rows, cols)` plus an `(nnz × B)` value
matrix (each instance fills its own column), solved by a batched-sparse / structured
solver. This is the target for the medium-and-sparse batched regime (≈1–2k vars)."
struct BatchedSparse <: KKTStrategy end

"`∇F` as a single `(d × d)` sparse matrix (`B = 1`), solved by a sparse direct or
preconditioned iterative method. The route to single large problems: it implements
the SAME verbs, so the solver loop is unchanged. [not yet implemented]"
struct SparseSingle <: KKTStrategy end

# ---------------------------------------------------------------------------
# Verb set (the entire interface the solver depends on).
# Methods are added per strategy in later steps; signatures documented here.
# ---------------------------------------------------------------------------

"""
    materialize(mcp::PrimalDualMCP, strategy::KKTStrategy, device; batch_size) -> cache

Build a preallocated workspace to solve `batch_size` instances of `mcp` with
`strategy` on `device`. Computes the shared sparsity pattern once and allocates all
device buffers. Returns a strategy-specific cache consumed by the verbs below.
"""
function materialize end

"""
    residual!(F, mcp, x, y, s, θ, ϵ; device)

Fill the residual `F` (a dense `(d × B)` device array) for the whole batch. The
residual is always dense `(d × B)`, so this is representation-agnostic — the SAME
for every strategy. `ϵ` is a length-`B` relaxation vector. [not yet implemented]
"""
function residual! end

"""
    jacobian!(cache, mcp, x, y, s, θ, ϵ, η; device)

Assemble `∇F` (with per-instance regularization `η`) into the cache's internal
representation. Strategy-specific: `BatchedSparse` fills `nzval` `(nnz × B)` and
applies `η` at the diagonal positions `cache.diag_nz`; `BatchedDense` scatters the
same `nnz` values into a zeroed `(d × d × B)`; `SparseSingle` fills a single
`(d × d)`. Whether `η` is applied internally (if the evaluator carries an `η` slot)
or additively on the diagonal is a per-strategy detail. [not yet implemented]
"""
function jacobian! end

"""
    factorize!(cache)

Factorize the batched system currently held in `cache`, leaving a factorization that
`ldiv!` reuses. `BatchedDense`: batched LU. `BatchedSparse`: numeric factorization
over the shared symbolic structure (e.g. cuDSS batched) or a structured solve.
`SparseSingle`: sparse direct / preconditioned iterative. [not yet implemented]
"""
function factorize! end

"""
    ldiv!(out, cache, rhs)

Solve `∇F · out = rhs` for all instances, reusing the factorization from
`factorize!`. Both `out` and `rhs` are plain `(d × B)` device arrays.

The multi-rhs form used by parameter sensitivities is `(d × K × B)`, where
`K = nθ` is the number of parameters: per instance the system `∇F_z · (∂z/∂θ) =
-∇F_θ` has a `(d × K)` right-hand side `-∇F_θ` and a `(d × K)` solution `∂z/∂θ`.
The equation/variable axis `d` must lead (it is the row dimension of the `(d × d)`
solve), then the `K` right-hand sides, then the batch axis last. This keeps each
instance's `(d × K)` block contiguous in column-major memory and makes the
single-rhs `(d × B)` case the `K = 1` special case. [not yet implemented]
"""
function ldiv! end

# ---------------------------------------------------------------------------
# BatchedSparse cache + materialize.
# ---------------------------------------------------------------------------

""" Preallocated workspace for the `BatchedSparse` strategy.

Holds the shared sparsity pattern and the `(nnz × B)` batched value matrix. The
primal-dual iterate (`x, y, s`), residual `F`, and step `δz` are NOT stored here —
they belong to the solver. Only what is tied to the `∇F` representation and its
factorization lives in the cache.
"""
struct BatchedSparseCache{Dev,M,P,F}
    "KernelAbstractions backend the buffers live on."
    device::Dev
    "Residual / KKT-system dimension (`∇F` is `d × d`)."
    d::Int
    "Number of structural nonzeros in the shared pattern."
    nnz::Int
    "Number of problem instances solved together."
    batch_size::Int
    "Row indices of the shared sparsity pattern (column-major / CSC order)."
    rows::Vector{Int}
    "Column indices of the shared sparsity pattern (column-major / CSC order)."
    cols::Vector{Int}
    "Indices into `nzval`'s first axis for structurally-present diagonal entries
     (used by additive `η` regularization)."
    diag_nz::Vector{Int}
    "Batched Jacobian values: `(nnz × B)`, column `b` is instance `b`'s nonzeros."
    nzval::M
    "Device sparse structure / symbolic-factorization handle; `nothing` until
     `factorize!` is implemented and populates it."
    pattern::P
    "Numeric factorization workspace; `nothing` until `factorize!` populates it."
    factor::F
end

function materialize(mcp::PrimalDualMCP, ::BatchedSparse, device; batch_size)
    d = mcp.unconstrained_dimension + 2mcp.constrained_dimension
    mcp.∇F_z!.size == (d, d) ||
        error("Expected a square ($d × $d) ∇F_z pattern, got $(mcp.∇F_z!.size).")

    # The shared sparsity pattern is already computed on the (symbolic) MCP; the
    # batch differs only in numeric values, so we store ONE pattern.
    rows = copy(mcp.∇F_z!.rows)
    cols = copy(mcp.∇F_z!.cols)
    nnz = length(rows)

    # Structurally-present diagonal entries (for additive η regularization). Note:
    # a strategy that relies on additive η must ensure the diagonal is fully present
    # in the pattern; internal regularization (η baked into the evaluator) does not.
    diag_nz = findall(k -> rows[k] == cols[k], eachindex(rows))

    nzval = KernelAbstractions.zeros(device, Float64, nnz, batch_size)

    BatchedSparseCache(
        device,
        d,
        nnz,
        batch_size,
        rows,
        cols,
        diag_nz,
        nzval,
        nothing,  # pattern — populated by `factorize!` (batched-sparse solver, D1)
        nothing,  # factor  — populated by `factorize!`
    )
end

# Fallback so unimplemented strategies fail with a clear message rather than a
# confusing MethodError.
function materialize(::PrimalDualMCP, strategy::KKTStrategy, device; batch_size)
    error("`materialize` is not yet implemented for strategy $(typeof(strategy)).")
end

# ---------------------------------------------------------------------------
# residual! and jacobian! — KernelAbstractions implementation, one MCP instance
# per thread. Validated on the CPU() backend.
#
# Naming convention: UPPERCASE Latin denotes a batched `(· × B)` device array
# (`X, Y, S, Θ, F`, and the cache's value matrix `V`); lowercase Greek `ϵ`, `η` are
# the per-instance relaxation/regularization, stored as length-`B` vectors. Inside a
# kernel, column `b` (`X[:, b]`, …) is the single instance handed to the per-instance
# evaluator.
#
# NOTE (GPU): these slice per-instance columns with `@views`, which is correct on the
# CPU backend but not generally allowed inside a GPU kernel; a GPU backend will need a
# view-free variant (manual column offsets). Likewise `cache.diag_nz` is a host
# `Vector` here — for a GPU backend it must be device-resident. The trailing
# `synchronize` is kept for correctness; for GPU it should be hoisted to where the
# solver actually reads values back to the host (the convergence check), to avoid a
# stall every Newton iteration. The per-instance evaluators are already kernel-safe.
# ---------------------------------------------------------------------------

@kernel function _residual_kernel!(F, X, Y, S, Θ, ϵ, f!)
    b = @index(Global)
    @views f!(F[:, b], X[:, b], Y[:, b], S[:, b], Θ[:, b], ϵ[b])
end

"""
    residual!(F, mcp, X, Y, S, Θ, ϵ; device)

Fill the residual `F` (a dense `(d × B)` device array) for the whole batch, one MCP
instance per thread. `X, Y, S, Θ` are `(· × B)` batched device arrays and `ϵ` is a
length-`B` relaxation vector. Requires `mcp` to carry a kernel residual evaluator
(build it with `compute_kernel_evaluators = true`).
"""
function residual!(F, mcp::PrimalDualMCP, X, Y, S, Θ, ϵ; device)
    isnothing(mcp.F_kernel) && error(
        "This MCP has no kernel residual evaluator. Construct it with " *
        "`compute_kernel_evaluators = true`.",
    )
    _residual_kernel!(device)(F, X, Y, S, Θ, ϵ, mcp.F_kernel; ndrange = size(F, 2))
    KernelAbstractions.synchronize(device)
    F
end

@kernel function _jacobian_kernel!(V, X, Y, S, Θ, ϵ, ∇f!, diag_nz, η)
    b = @index(Global)
    @views ∇f!(V[:, b], X[:, b], Y[:, b], S[:, b], Θ[:, b], ϵ[b])
    # Fused additive regularization: each thread adds its own η[b] to the
    # structurally-present diagonal entries of its column, in the same per-instance
    # parallel pass (before any synchronization).
    @inbounds for k in diag_nz
        V[k, b] += η[b]
    end
end

"""
    jacobian!(cache, mcp, X, Y, S, Θ, ϵ, η; device)

Assemble the batched `∇F` into `cache.nzval` (`(nnz × B)`, each column an instance's
nonzero values in the shared pattern's order) one instance per thread, adding the
per-instance regularization `η` (a length-`B` vector) to the structurally-present
diagonal entries within the same kernel.

NOTE: only the diagonal slots in `cache.diag_nz` are reachable additively; patterns
with missing diagonals need augmentation for full identity regularization (internal-η
regularization, baked into the evaluator, is the planned alternative — not yet
supported for kernel evaluators).
"""
function jacobian!(cache::BatchedSparseCache, mcp::PrimalDualMCP, X, Y, S, Θ, ϵ, η; device)
    isnothing(mcp.∇F_z_kernel) && error(
        "This MCP has no kernel Jacobian evaluator. Construct it with " *
        "`compute_kernel_evaluators = true`.",
    )
    _jacobian_kernel!(device)(
        cache.nzval, X, Y, S, Θ, ϵ, mcp.∇F_z_kernel, cache.diag_nz, η;
        ndrange = cache.batch_size,
    )
    KernelAbstractions.synchronize(device)
    cache
end
