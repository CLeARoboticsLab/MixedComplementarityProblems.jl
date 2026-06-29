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
struct BatchedSparseCache{Dev,M,P,F,Dnz}
    "KernelAbstractions backend the buffers live on."
    device::Dev
    "Residual / KKT-system dimension (`∇F` is `d × d`)."
    d::Int
    "Number of structural nonzeros in the shared pattern."
    nnz::Int
    "Number of problem instances solved together."
    batch_size::Int
    "Row indices of the shared sparsity pattern (column-major / CSC order). Host-side:
     used to (re)build the CPU pattern and for introspection, never inside a kernel."
    rows::Vector{Int}
    "Column indices of the shared sparsity pattern (column-major / CSC order). Host-side."
    cols::Vector{Int}
    "Indices into `nzval`'s first axis for structurally-present diagonal entries (used by
     additive `η` regularization). Device-resident (read inside `_jacobian_kernel!`)."
    diag_nz::Dnz
    "Batched Jacobian values: `(nnz × B)`, column `b` is instance `b`'s nonzeros."
    nzval::M
    "CPU: shared-pattern CSC scratch matrix whose nonzeros `factorize!` overwrites per
     instance. Other backends: their own sparse structure / symbolic handle. `nothing`
     on backends without a CPU-style solver."
    pattern::P
    "CPU: vector of per-instance KLU factorizations (independent, so `factorize!`/`ldiv!`
     parallelize across the batch), filled lazily on the first `factorize!` and refreshed
     in place by `klu!` refactor thereafter. `nothing` on backends without a CPU-style
     solver."
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
    # Moved onto `device`: `_jacobian_kernel!` iterates this inside the kernel, so it
    # must be device-resident (a plain Array on CPU, a device array on GPU).
    diag_nz_host = findall(k -> rows[k] == cols[k], eachindex(rows))
    diag_nz = KernelAbstractions.allocate(device, Int, length(diag_nz_host))
    copyto!(diag_nz, diag_nz_host)

    nzval = KernelAbstractions.zeros(device, Float64, nnz, batch_size)

    # Per-device linear-solve workspace (the "analyze" phase). Dispatched on `device`
    # so a GPU backend (cuDSS, D1 in docs/gpu_kkt_design.md) can supply its own via a
    # package extension without touching the device-generic assembly above.
    pattern, factor = _materialize_linsolve(device, rows, cols, nnz, d, batch_size)

    BatchedSparseCache(
        device,
        d,
        nnz,
        batch_size,
        rows,
        cols,
        diag_nz,
        nzval,
        pattern,
        factor,
    )
end

# Fallback so unimplemented strategies fail with a clear message rather than a
# confusing MethodError.
function materialize(::PrimalDualMCP, strategy::KKTStrategy, device; batch_size)
    error("`materialize` is not yet implemented for strategy $(typeof(strategy)).")
end

"""
    _materialize_linsolve(device, rows, cols, nnz, d, batch_size) -> (pattern, factor)

Build the `BatchedSparse` linear-solve workspace for `device` (the "analyze" phase).
Returns the device's sparse `pattern` and `factor` workspace, stored verbatim in the
cache and consumed by `factorize!`/`ldiv!`. Dispatched on `device` so GPU backends can
supply their own (cuDSS, D1 in docs/gpu_kkt_design.md) via a package extension.

CPU backend: a shared-pattern CSC scratch matrix (carrying the `colptr`/`rowval` the
initial factorization needs) plus a vector of per-instance KLU factorizations, filled
LAZILY on the first `factorize!`. We use KLU rather than UMFPACK because KLU's
`klu_refactor` reuses the numeric storage AND pivot ordering in place, so the
per-Newton-iteration `factorize!` is allocation-free and ~5× faster — removing the
memory-bandwidth/allocator wall that capped UMFPACK's batched throughput (UMFPACK
allocates a fresh numeric, ~9 MiB/batch, every iteration).

The factor vector starts `undef`: each instance's first `factorize!` does a full `klu`
(analyze + factor) using its REAL values to choose good pivots, and every later call does
`klu!` refactor reusing those pivots. (Independent per-instance factorizations also keep
`factorize!`/`ldiv!` thread-safe across the batch.) The initial factor is lazy — and not
eager in `materialize` — because good pivots need representative values, which only the
solver has, not the symbolic pattern. The GPU/cuDSS backend is unaffected.
"""
function _materialize_linsolve(
    ::KernelAbstractions.CPU,
    rows,
    cols,
    nnz,
    d,
    batch_size,
)
    template = SparseArrays.sparse(rows, cols, ones(nnz), d, d)
    Ti = eltype(SparseArrays.getcolptr(template))
    factor = Vector{KLU.KLUFactorization{Float64,Ti}}(undef, batch_size)
    (template, factor)
end

function _materialize_linsolve(device, rows, cols, nnz, d, batch_size)
    error(
        "BatchedSparse linear-solve workspace is not implemented for device " *
        "$(typeof(device)). The GPU (cuDSS) backend loads via a package extension — " *
        "ensure both CUDA and CUDSS are loaded (see D1 in docs/gpu_kkt_design.md).",
    )
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
# NOTE (GPU): these slice per-instance columns with `@views`. A column slice of a 2D
# device array is an isbits, allocation-free `SubArray`, so passing it to the
# per-instance evaluator is expected to be GPU-safe — but this should be confirmed on
# an actual CUDA device (along with whether the SerialForm evaluator itself compiles
# for the device — D2). `cache.diag_nz` is now device-resident (built via
# `KernelAbstractions.allocate` in `materialize`), so the in-kernel loop is valid on
# GPU. The trailing `synchronize` is kept for correctness; for GPU it should be hoisted
# to where the solver actually reads values back to the host (the convergence check),
# to avoid a stall every Newton iteration.
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

# ---------------------------------------------------------------------------
# factorize! and ldiv! — BatchedSparse, CPU backend (per-instance sparse LU).
#
# This is the correctness baseline for an end-to-end batched solve, NOT the
# performance path. The GPU batched-sparse solver (D1 in docs/gpu_kkt_design.md) —
# e.g. cuDSS batched mode reusing one symbolic factorization across the batch — slots
# in behind these same two verbs. Each instance owns an independent KLU factorization;
# the per-instance work parallelizes across the batch with `Threads.@threads` (distinct
# factorizations, no shared state). The first `factorize!` does a full `klu`
# (analyze + factor) from each instance's real values; every later call does an
# allocation-free `klu!` refactor that reuses the symbolic AND the pivot ordering — KLU
# refactor stays accurate as the KKT system tightens toward the boundary (verified to
# match fresh re-pivoting through s⊙y ≈ 1e-4, well past `tol`). `ldiv!` is likewise
# threaded (independent factors, per-instance solve workspaces).
# ---------------------------------------------------------------------------

"""
    factorize!(cache::BatchedSparseCache)

Factorize each instance's `∇F` (currently held in `cache.nzval`) into a per-instance
LU stored in `cache.factor`, reused by `ldiv!`. The CPU method is below; GPU backends
provide their own (cuDSS, D1) via a package extension.
"""
function factorize!(cache::BatchedSparseCache{<:KernelAbstractions.CPU})
    template = cache.pattern
    Threads.@threads for b in 1:cache.batch_size
        if isassigned(cache.factor, b)
            # Refresh this instance's numeric factorization in place from its column of
            # `nzval` (same CSC order as the factor's pattern), reusing the symbolic and
            # pivots — allocation-free, no re-analysis. Independent factors → race-free.
            @views copyto!(cache.factor[b].nzval, cache.nzval[:, b])
            KLU.klu!(cache.factor[b], cache.factor[b].nzval; check = false)
        else
            # First factorization for this instance: full analyze + factor from its real
            # values, establishing the pivot ordering reused by later refactors.
            instance = SparseArrays.SparseMatrixCSC(
                cache.d,
                cache.d,
                copy(SparseArrays.getcolptr(template)),
                copy(SparseArrays.rowvals(template)),
                cache.nzval[:, b],
            )
            cache.factor[b] = KLU.klu(instance; check = false)
        end
    end
    cache
end

function factorize!(cache::BatchedSparseCache)
    error(
        "factorize! for BatchedSparse is not implemented for device " *
        "$(typeof(cache.device)). The GPU (cuDSS) backend loads via a package " *
        "extension (D1 in docs/gpu_kkt_design.md).",
    )
end

"""
    ldiv!(out, cache::BatchedSparseCache, rhs)

Solve `∇F · out = rhs` per instance, reusing the factorizations from `factorize!`.
`out` and `rhs` are `(d × B)` arrays. The CPU method is below; GPU backends provide
their own via a package extension.
"""
function ldiv!(out, cache::BatchedSparseCache{<:KernelAbstractions.CPU}, rhs)
    Threads.@threads for b in 1:cache.batch_size
        @views LinearAlgebra.ldiv!(out[:, b], cache.factor[b], rhs[:, b])
    end
    out
end

function ldiv!(out, cache::BatchedSparseCache, rhs)
    error(
        "ldiv! for BatchedSparse is not implemented for device " *
        "$(typeof(cache.device)). The GPU (cuDSS) backend loads via a package " *
        "extension (D1 in docs/gpu_kkt_design.md).",
    )
end

# ---------------------------------------------------------------------------
# Batched interior-point solver (the §5 loop in docs/gpu_kkt_design.md), written
# entirely against the verb set so the strategy and device are pluggable. This is a
# NEW dispatch alongside the existing unbatched `solve(::InteriorPoint, ...)`; the two
# can be collapsed once the B=1 path matches/exceeds the unbatched solver.
#
# Batch-axis invariant: every per-instance scalar (ϵ, η, kkt_error, the step sizes,
# the convergence mask) is a length-B vector; reductions are over the coordinate axis
# (dims = 1) and never collapse the batch axis.
# ---------------------------------------------------------------------------

"Interior-point solver that solves a whole batch of MCP instances simultaneously."
struct BatchedInteriorPoint <: SolverType end

"""
    max_step_to_boundary(V, Δ; τ = 0.995)

Closed-form fraction-to-the-boundary step, one stepsize per instance. `V, Δ` are
`(k × B)`; returns a length-`B` vector with
`α_i = min(1, min_{j : Δ[j,i] < 0} -τ·V[j,i]/Δ[j,i])`. Exact (no backtracking) and
fully parallel: the reduction is over the coordinate axis only.
"""
function max_step_to_boundary(V, Δ; τ = 0.995)
    ratio = @. ifelse(Δ < 0, -τ * V / Δ, Inf)
    α = vec(minimum(ratio; dims = 1))
    clamp!(α, zero(eltype(α)), one(eltype(α)))
    α
end

"""
    solve(::BatchedInteriorPoint, mcp, Θ; strategy, device, kwargs...)

Solve `B = size(Θ, 2)` MCP instances simultaneously, where column `b` of the
`(nθ × B)` matrix `Θ` is instance `b`'s parameter vector. Mirrors the unbatched
`InteriorPoint` schedule (outer tightening of ϵ/η, inner Newton with fraction-to-the-
boundary line search) with all state batched and per-instance. Requires `mcp` built
with `compute_kernel_evaluators = true`.

Returns `(; status, x, y, s, kkt_error, ϵ, outer_iters, total_iters)` where `status`
is a length-`B` vector of `:solved`/`:failed`, `x, y, s` are `(· × B)`, and
`kkt_error, ϵ` are length-`B`.
"""
function solve(
    ::BatchedInteriorPoint,
    mcp::PrimalDualMCP,
    Θ::AbstractMatrix;
    strategy::KKTStrategy = BatchedSparse(),
    device = KernelAbstractions.CPU(),
    X₀ = nothing,
    Y₀ = nothing,
    S₀ = nothing,
    tol = 1e-4,
    ϵ₀ = :auto,
    max_inner_iters = 20,
    max_outer_iters = 50,
    tightening_rate = 0.1,
    loosening_rate = 0.5,
)
    nx = mcp.unconstrained_dimension
    ny = mcp.constrained_dimension
    d = nx + 2ny
    B = size(Θ, 2)

    cache = materialize(mcp, strategy, device; batch_size = B)

    # Batched primal-dual state (solver-owned). Residual F and step δz are plain
    # (d × B) arrays; the Jacobian/factorization live inside the cache.
    X = isnothing(X₀) ? KernelAbstractions.zeros(device, Float64, nx, B) : copy(X₀)
    Y = isnothing(Y₀) ? KernelAbstractions.ones(device, Float64, ny, B) : copy(Y₀)
    S = isnothing(S₀) ? KernelAbstractions.ones(device, Float64, ny, B) : copy(S₀)
    F = KernelAbstractions.zeros(device, Float64, d, B)
    δz = KernelAbstractions.zeros(device, Float64, d, B)
    δx = view(δz, 1:nx, :)
    δy = view(δz, (nx + 1):(nx + ny), :)
    δs = view(δz, (nx + ny + 1):d, :)

    warm = !isnothing(X₀) && !isnothing(Y₀) && !isnothing(S₀)
    ϵ_init = ϵ₀ === :auto ? (warm ? tol : one(tol)) : float(ϵ₀)
    ϵ = KernelAbstractions.zeros(device, Float64, B)
    ϵ .= ϵ_init
    η = KernelAbstractions.zeros(device, Float64, B)
    η .= tol
    done = KernelAbstractions.zeros(device, Bool, B)

    residual!(F, mcp, X, Y, S, Θ, ϵ; device)
    kkt = vec(maximum(abs, F; dims = 1))

    outer = 0
    total = 0
    while !all(done) && outer < max_outer_iters
        # `inner` starts at 1 (matching the unbatched solver): the tightening factor
        # `1 - exp(-rate·inner)` must never be evaluated at inner = 0, which would
        # zero out ϵ when a subproblem is already converged at entry.
        inner = 1
        while any((kkt .> ϵ) .& .!done) && inner < max_inner_iters
            jacobian!(cache, mcp, X, Y, S, Θ, ϵ, η; device)
            factorize!(cache)
            ldiv!(δz, cache, -F)

            # Per-instance step sizes; freeze instances that are done or whose current
            # subproblem has already converged (kkt ≤ ϵ) by zeroing their step.
            α_s = max_step_to_boundary(S, δs)
            α_y = max_step_to_boundary(Y, δy)
            stepping = (kkt .> ϵ) .& .!done
            α_s .*= stepping
            α_y .*= stepping
            X .+= reshape(α_s, 1, :) .* δx
            S .+= reshape(α_s, 1, :) .* δs
            Y .+= reshape(α_y, 1, :) .* δy

            total += 1
            inner += 1
            residual!(F, mcp, X, Y, S, Θ, ϵ; device)
            kkt = vec(maximum(abs, F; dims = 1))
        end

        # Outer update, per instance: tighten ϵ/η where the subproblem converged,
        # loosen where it didn't; mark instances done once converged at ϵ ≤ tol.
        subconverged = kkt .≤ ϵ
        done .= done .| (subconverged .& (ϵ .≤ tol))
        tighten = 1 - exp(-tightening_rate * inner)
        loosen = 1 + exp(-loosening_rate * inner)
        working = .!done
        @. ϵ = ifelse(working, ifelse(subconverged, ϵ * tighten, ϵ * loosen), ϵ)
        @. η = ifelse(working, ifelse(subconverged, η * tighten, η * loosen), η)
        @. ϵ = min(ϵ, one(eltype(ϵ)))
        outer += 1
    end

    status = map(b -> b ? :solved : :failed, collect(done))
    (;
        status,
        x = X,
        y = Y,
        s = S,
        kkt_error = kkt,
        ϵ,
        outer_iters = outer,
        total_iters = total,
    )
end

# ---------------------------------------------------------------------------
# Batched parameter sensitivities  ∂z/∂θ = -(∇F_z)⁻¹ ∇F_θ, per instance.
#
# Reuses the per-instance factorizations: ∇F_z is assembled (unregularized) at the
# converged iterate and factorized; the dense (d × nθ) parameter Jacobian -∇F_θ is
# the multi-rhs right-hand side, solved via the (d × K × B) `ldiv!`.
# ---------------------------------------------------------------------------

"Multi-rhs solve: `out[:,:,b] = ∇F_z[b]⁻¹ rhs[:,:,b]` per instance, reusing the
factorizations from `factorize!`. `out`/`rhs` are `(d × K × B)` (K right-hand sides;
batch axis last). The CPU method is below; GPU backends provide their own via a
package extension."
function ldiv!(
    out::AbstractArray{<:Any,3},
    cache::BatchedSparseCache{<:KernelAbstractions.CPU},
    rhs::AbstractArray{<:Any,3},
)
    Threads.@threads for b in 1:cache.batch_size
        @views LinearAlgebra.ldiv!(out[:, :, b], cache.factor[b], rhs[:, :, b])
    end
    out
end

function ldiv!(
    out::AbstractArray{<:Any,3},
    cache::BatchedSparseCache,
    rhs::AbstractArray{<:Any,3},
)
    error(
        "ldiv! for BatchedSparse is not implemented for device " *
        "$(typeof(cache.device)). The GPU (cuDSS) backend loads via a package " *
        "extension (D1 in docs/gpu_kkt_design.md).",
    )
end

@kernel function _jacobian_θ_kernel!(Jθ, X, Y, S, Θ, ϵ, ∇F_θ!)
    b = @index(Global)
    @views ∇F_θ!(Jθ[:, :, b], X[:, b], Y[:, b], S[:, b], Θ[:, b], ϵ[b])
end

"""
    solve_jacobian_θ(mcp, X, Y, S, Θ, ϵ; strategy, device) -> ∂z∂θ

Batched parameter sensitivities of the MCP solution: returns `∂z∂θ` of shape
`(d × nθ × B)`, where `∂z∂θ[:, :, b]` is `∂z/∂θ` for instance `b` at the point
`(X[:,b], Y[:,b], S[:,b])` with parameters `Θ[:,b]`. Rows `1:nx` are `∂x/∂θ`, the
next `ny` are `∂y/∂θ`, the last `ny` are `∂s/∂θ`.

For this to be the true solution sensitivity, `(X, Y, S)` must be a converged
solution (`F = 0`). Requires `mcp` built with `compute_kernel_evaluators` AND
`compute_sensitivities`.
"""
function solve_jacobian_θ(
    mcp::PrimalDualMCP,
    X,
    Y,
    S,
    Θ,
    ϵ;
    strategy::KKTStrategy = BatchedSparse(),
    device = KernelAbstractions.CPU(),
)
    isnothing(mcp.∇F_θ_kernel) && error(
        "This MCP has no kernel θ-Jacobian. Construct it with " *
        "`compute_kernel_evaluators = true` AND `compute_sensitivities = true`.",
    )
    nx = mcp.unconstrained_dimension
    ny = mcp.constrained_dimension
    d = nx + 2ny
    nθ = size(Θ, 1)
    B = size(Θ, 2)

    cache = materialize(mcp, strategy, device; batch_size = B)

    # ∇F_z at the (converged) point, unregularized (η = 0), then factorize.
    jacobian!(cache, mcp, X, Y, S, Θ, ϵ, KernelAbstractions.zeros(device, Float64, B); device)
    factorize!(cache)

    # Right-hand side -∇F_θ as a dense (d × nθ × B) multi-rhs.
    rhs = KernelAbstractions.zeros(device, Float64, d, nθ, B)
    _jacobian_θ_kernel!(device)(rhs, X, Y, S, Θ, ϵ, mcp.∇F_θ_kernel; ndrange = B)
    KernelAbstractions.synchronize(device)
    rhs .= .-rhs

    ∂z∂θ = KernelAbstractions.zeros(device, Float64, d, nθ, B)
    ldiv!(∂z∂θ, cache, rhs)
    ∂z∂θ
end
