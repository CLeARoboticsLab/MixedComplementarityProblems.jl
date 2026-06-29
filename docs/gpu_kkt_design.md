# GPU MCP solver — batched-solve interface & abstraction boundary (DRAFT)

Status: **draft for review**. Goal of this doc: pin down *one* abstraction boundary
so that the interior-point solver is written **once**, and "batched-many-small",
"batched-medium-sparse" (your 1–2k case), and "single-large" become *pluggable
backends* rather than separate code paths.

> **Central claim.** The IP solver loop depends only on a small verb set
> `{residual!, jacobian!, factorize!, ldiv!}` plus generic batched array ops
> (norms, broadcasts, reductions). Everything regime-specific — Jacobian
> representation, device, linear-solve algorithm — lives behind those verbs.
> If this holds, the "keep the large-problem door open" requirement costs us
> nothing: the large case is just a third backend with batch size `B = 1`.

---

## 1. Layering

```
 ┌─────────────────────────────────────────────────────────────┐
 │ PrimalDualMCP  (symbolic spec: G,H,z,θ[,η] + compiled evaluators) │  ← source of truth,
 │   - residual evaluator   (fills F : d×B)                       │     device/strategy
 │   - ∂F/∂z value evaluator (fills nnz_z values)                 │     agnostic
 │   - ∂F/∂θ value evaluator (fills nnz_θ values, sensitivities)  │
 └───────────────────────────────┬─────────────────────────────┘
                                  │ materialize(mcp, strategy, device; B)
                                  ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ cache     (owns the numeric ∇F representation + factorization) │  ← the ONLY thing
 │   strategy ∈ {BatchedDense, BatchedSparse, SparseSingle}       │     that differs
 │   device   ∈ KA backend {CPU(), CUDABackend(), MetalBackend()} │     across regimes
 └───────────────────────────────┬─────────────────────────────┘
                                  │ verbs: residual! jacobian! factorize! ldiv!
                                  ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ solve(::InteriorPoint, mcp, Θ; strategy, device, ...)          │  ← written ONCE,
 │   IP control flow, per-instance ϵ/η schedule, linesearch,      │     backend-agnostic
 │   convergence masking. Operates on (k×B) device arrays.        │
 └─────────────────────────────────────────────────────────────┘
```

Two **orthogonal** axes (don't conflate them):

| Axis | What it controls | Values |
|------|------------------|--------|
| `device` (a KernelAbstractions backend) | array type, where kernels run | `CPU()`, `CUDABackend()`, `MetalBackend()`, … |
| `strategy` (`BatchedSolveStrategy`) | Jacobian representation + linear solver | `BatchedDense`, `BatchedSparse`, `SparseSingle` |

---

## 2. The abstraction boundary (the key cut)

What crosses the boundary between solver and backend, every Newton iteration:

```
 solver  ──(x, y, s, θ, ϵ, η)──▶  backend         # current iterate + relaxation/reg
 solver  ◀──── F : (d × B) ─────  backend         # residual, a plain device array
 solver  ◀──── δz : (d × B) ────  backend         # Newton step, a plain device array
```

**The Jacobian never crosses.** `F` and `δz` are plain `(d × B)` device arrays the
*solver* owns; the assembled `∇F` and its factorization stay *inside* the cache in
whatever representation the strategy wants (dense `d×d×B`, shared-pattern
`nnz×B`, single sparse `d×d`). The solver cannot see the difference. That single
invariant is what makes the three regimes interchangeable.

---

## 3. Types

```julia
# --- strategy: numeric representation of ∇F + its linear solver -------------
abstract type BatchedSolveStrategy end

struct BatchedDense  <: BatchedSolveStrategy end   # ∇F as (d×d×B) dense; batched LU.
                                          #   Good ONLY for small d (≲ few hundred).
struct BatchedSparse <: BatchedSolveStrategy end   # shared (rows,cols) + (nnz×B) values;
                                          #   batched sparse / structured solve.
                                          #   ← the 1–2k regime.
struct SparseSingle  <: BatchedSolveStrategy end   # B=1, single (d×d) sparse; direct/iterative.
                                          #   ← FUTURE large-problem door. Same verbs.

# --- cache: per-(mcp, strategy, device, B) preallocated workspace -----------
# Concrete fields are strategy-specific; sketch for BatchedSparse:
struct BatchedSparseCache{Dev,TV,TP,TF}
    device::Dev                  # KA backend
    rows::Vector{Int}            # shared sparsity pattern (host-computed once)
    cols::Vector{Int}
    diag_nz::Vector{Int}         # indices into nzval of diagonal entries (for η add)
    nzval::TV                    # (nnz × B) device matrix — the batched Jacobian
    pattern::TP                  # device CSR/CSC structure (symbolic factorization input)
    factor::TF                   # numeric factorization workspace (per instance)
end
```

---

## 4. The verb set (what the solver calls)

```julia
"""Build a workspace to solve `B` instances of `mcp` with `strategy` on `device`.
Computes the shared sparsity pattern once, allocates all device buffers."""
materialize(mcp::PrimalDualMCP, strategy::BatchedSolveStrategy, device; batch_size) -> cache  # strategy-specific (e.g. BatchedSparseCache)

# Naming convention: UPPERCASE Latin = batched (·×B) arrays (X, Y, S, Θ, F);
# lowercase Greek ϵ, η = per-instance length-B vectors.

"""Fill residual `F` (d×B) for the whole batch. Representation-agnostic
(F is always dense d×B), so this is the SAME for every strategy — it's a method
on the mcp's residual evaluator + device, not on the strategy."""
residual!(F, mcp, X, Y, S, Θ, ϵ; device)

"""Assemble ∇F (+ regularization η) into the cache's internal representation.
Strategy-specific: BatchedSparse fills nzval (nnz×B) then adds η at `diag_nz`;
BatchedDense scatters the same nnz values into a zeroed (d×d×B); SparseSingle
fills a single (d×d). η may be applied internally (if the evaluator has an η slot)
or additively on the diagonal — a per-strategy detail the solver doesn't see."""
jacobian!(cache, mcp, X, Y, S, Θ, ϵ, η; device)

"""Factorize the batched system currently in `cache` (reused by ldiv!).
BatchedDense: batched LU. BatchedSparse: numeric factorization on the shared
symbolic structure (e.g. cuDSS batched) or a structured block-banded solve.
SparseSingle: sparse direct / preconditioned iterative."""
factorize!(cache)

"""Solve ∇F · out = rhs for all instances, reusing the factorization.
`out` and `rhs` are plain (d×B) device arrays. Multiple-rhs form is used for
sensitivities (rhs = -∂F/∂θ, K columns)."""
ldiv!(out, cache, rhs)
```

Generic batched ops the solver uses directly (must work on `Array` *and* `CuArray`):

```julia
# per-instance KKT error — reduce over VARIABLES (dim 1), keep batch axis → (B,)
kkt_error(F) = vec(maximum(abs, F; dims = 1))                      # (B,)

# fraction-to-boundary, EXACT closed form, one stepsize PER PROBLEM.
# Largest α∈[0,1] with v + αδ ≥ (1-τ)v: since v>0, only δ<0 coords bind, giving
#   α_i = min(1, min_{k : δ[k,i] < 0}  -τ · v[k,i] / δ[k,i]).
# No backtracking loop, no per-instance host↔device sync.
function max_step_to_boundary(v, δ; τ = 0.995)   # v, δ :: (k × B)
    ratio = @. ifelse(δ < 0, -τ * v / δ, Inf)     # (k × B); only δ<0 can bind
    α = vec(minimum(ratio; dims = 1))             # reduce over VARIABLES → (B,)
    clamp!(α, 0, 1)                               # one stepsize per problem
end
```

> **Batch-axis invariant.** Every per-instance scalar (`ϵ`, `η`, `kkt_error`,
> `α_s`, `α_y`, the convergence mask) is a length-`B` vector. Reductions are
> **always** over the coordinate axis (`dims = 1`) and **never** collapse the
> batch axis — the batch survives until the single `all(done)` that ends the loop.
> Writing `minimum(ratio)` instead of `minimum(ratio; dims = 1)` would silently
> couple all `B` problems into one shared stepsize: the bug to guard against.

---

## 5. Solver loop, rewritten against the verbs (pseudocode)

```julia
function solve(::InteriorPoint, mcp, Θ;                 # Θ : (nθ × B)
               strategy = BatchedSparse(), device = CPU(),
               batch_size = size(Θ, 2), tol = 1e-4, ϵ₀ = :auto, ...)
    B = batch_size
    cache = materialize(mcp, strategy, device; batch_size = B)

    X, Y, S = alloc(device, nx, B), ones(device, ny, B), ones(device, ny, B)
    F, δz   = alloc(device, d, B), alloc(device, d, B)
    ϵ = fill(device, init_ϵ, B)          # PER-INSTANCE
    η = fill(device, tol,    B)          # PER-INSTANCE
    done = falses(device, B)

    while !all(done) && outer < max_outer
        for inner in 1:max_inner
            residual!(F, mcp, X, Y, S, Θ, ϵ; device)
            err  = kkt_error(F)                       # (B,)
            done = err .≤ ϵ
            all(done .| (err .≤ ϵ)) && break          # inner-converged instances rest

            jacobian!(cache, mcp, X, Y, S, Θ, ϵ, η; device)
            factorize!(cache)
            ldiv!(δz, cache, -F)                      # δz : (d×B), strategy-agnostic

            δx, δy, δs = @views δz[1:nx,:], δz[nx+1:nx+ny,:], δz[nx+ny+1:end,:]
            αs = max_step_to_boundary(S, δs)          # (B,)  for (x, s)
            αy = max_step_to_boundary(Y, δy)          # (B,)  for y
            αs .*= .!done; αy .*= .!done              # frozen instances don't move
            X .+= αs' .* δx;  S .+= αs' .* δs;  Y .+= αy' .* δy
        end
        ϵ, η = update_schedule!(ϵ, η, inner_converged) # elementwise tighten/loosen
    end
    (; status, x = X, y = Y, s = S, kkt_error = err, ϵ)
end
```

Note: **no `CuArray`, no `lu`, no sparsity** appears in this loop. That's the goal.

---

## 6. Resolved decisions (carried from earlier discussion)

- **R1 — Always batched.** Every state array carries a `B` dimension; `ϵ`, `η`,
  `kkt_error` are length-`B`. `B=1` is a degenerate column, never special-cased.
  *(This is the single decision that keeps the large door open.)*
- **R2 — Closed-form linesearch.** Replace the backtracking `while any(...)`
  (host↔device sync per step) with the closed-form per-instance reduction.
- **R3 — One codegen path: in-place fill of `nnz` values.** Drop `SVector`/`force_SA`
  (only viable for tiny `d`; dies at 1–2k via register spill). The Jacobian evaluator
  always fills the `nnz` value vector; the **strategy** decides the container (sparse
  keeps it; dense scatters into `d×d×B`). The "SVector vs in-place" fork dissolves.
  These kernel-safe evaluators (`mcp.F_kernel`, `mcp.∇F_z_kernel`, SerialForm + cse)
  are built opt-in at construction via `PrimalDualMCP(...; compute_kernel_evaluators =
  true)`, so CPU-only users pay no extra compile cost.
  *(Validated end-to-end in `gpu_prototype/batched_eval.jl` and
  `gpu_prototype/check_residual_jacobian.jl`.)*
- **R4 — Regularization via `η` arg, not a separate verb.** `η` flows into
  `jacobian!`; internal-vs-additive is a per-strategy implementation detail.

---

## 7. Open decision points — for us to discuss

> Each has my recommendation first. These are the ones I think are genuinely yours
> to call.

### D1 — The batched-sparse linear solver (the big one)
The 1–2k regime needs `factorize!`/`ldiv!` for a batch of medium sparse systems
that *share a sparsity pattern*. Options:

| Option | Pros | Cons |
|--------|------|------|
| **(a) Generic batched-sparse (cuDSS batched mode)** *(rec. to start)* | one symbolic factorization reused across batch; general (any MCP, not just games); least code | NVIDIA-only; perf depends on cuDSS batched maturity |
| (b) Structure-exploiting block-banded / Riccati | highest performance for trajectory games; embarrassingly parallel across batch | requires the spec to expose block structure; game-specific; most code |
| (c) Per-instance sparse LU in a kernel (KLU-style) | portable via KA | reimplementing sparse LU on GPU is a project in itself |

My lean: **(a) to get correct end-to-end throughput**, with **(b) as an opt-in
fast path** for trajectory games behind the *same* `factorize!`/`ldiv!` verbs.

### D2 — Codegen compile-time at full scale
A single `SerialForm` body with thousands of outputs (R3) may compile slowly; the
*current CPU solver* uses `ShardedForm` precisely to manage this, and `ShardedForm`
is **GPU-incompatible**. Options:

| Option | Notes |
|--------|-------|
| **(a) SerialForm + cse, measure first** *(rec.)* | simplest; we don't yet know it's a problem — needs a real 1–2k compile-time measurement |
| (b) Block-structured codegen | generate small per-block (per-timestep/player) evaluators, loop over blocks×instances in the kernel; compile-friendly AND more parallel, but needs the spec to expose repeated structure |

My lean: **(a) now**, treat **(b)** as the fallback *and* the bridge to D1(b) — both
need the spec to expose block structure, so they share a prerequisite.

### D3 — Does the symbolic spec expose block structure?
D1(b) and D2(b) both need it. Decision: do we add an optional
`structure` descriptor to `PrimalDualMCP` (e.g. "block-tridiagonal over horizon
`H` with block size `nb`") now, or defer? My lean: **leave a documented hook now,
implement later** — cheap to reserve, expensive to retrofit.

### D4 — Where do device arrays get allocated?
Recommendation: the cache carries the KA `device`; the solver allocates via
`KernelAbstractions.zeros(device, T, dims...)` / `allocate`, never touching
`CuArray` directly. Keeps the loop in §5 literally device-free. *(Low stakes,
but worth confirming the dependency direction.)*

---

## 8. How the large-problem door stays open (concrete)

`SparseSingle` is a `BatchedSolveStrategy` with `B = 1` that implements the *same four
verbs*: `materialize` builds one `d×d` sparse system; `jacobian!` fills its
`nnz` (the existing evaluator, B=1); `factorize!` calls cuDSS / GMRES+precond;
`ldiv!` does one solve. **The §5 loop is unchanged.** The only large-specific work
is inside `SparseSingle`'s four methods — no fork in the solver, no fork in codegen
(R3 is already in-place/sparse). The thing that *would* have closed this door —
committing the solver to dense batched arrays — is exactly what R1+R3 avoid.

---

## 9. Proposed build order

1. `BatchedSolveStrategy` types + verb signatures + `BatchedSparse` cache struct (this doc → stubs).
2. `materialize` + `jacobian!`/`residual!` on KA CPU backend (extends the prototype).
3. `factorize!`/`ldiv!` for `BatchedSparse` — **CPU first** (per-instance sparse LU
   loop) to get a correct end-to-end batched solve, before any GPU solver choice (D1).
4. Rewrite `solve(::InteriorPoint, …)` against the verbs (§5); validate vs current
   solver on the QP + parametric-game tests with `B=1`.
5. Swap in a GPU `device` and the D1(a) solver; benchmark.
6. (Later) `SparseSingle` (§8) and/or structured fast path (D1b/D2b/D3).
```

---

## 10. CPU linear-solver + multithreading notes (empirical)

The CPU `BatchedSparse` backend is **KLU** (`klu_refactor`), threaded across the batch
(`Threads.@threads`); each instance is an independent sparse solve. The history is worth
keeping because it shaped the design:

**Why KLU, not UMFPACK.** UMFPACK's `lu!` allocates a fresh numeric factorization every
call (~9.25 MiB per `factorize!` at `B = 64`), making the per-Newton-iteration hot loop
memory-bound — concurrent allocation + streaming contended on bandwidth/allocator, so
threaded scaling stalled at ~2.9× on 4 threads and the threaded fraction (~95%) couldn't
be cashed in. KLU's `klu_refactor` reuses numeric storage **and** the pivot ordering in
place → allocation-free and ~5× faster per refactor. Net measured win (full solve) over
the old UMFPACK path: **4.6–8.2× single-threaded**, **~9–14× at 4 threads** vs UMFPACK
serial. Each instance keeps an independent KLU factorization (thread-safe); the first
`factorize!` does a full `klu` from real values (for good pivots), later calls refactor.
Refactor reuses iteration-1 pivots — verified accurate vs fresh re-pivoting through
s⊙y ≈ 1e-4, past `tol`, so it holds as the KKT system tightens to the boundary.

**Thread count on heterogeneous (Apple-silicon-style) machines.** Still prefer
`-t <#performance-cores>` (e.g. `-t 4` on the M2's 4P+4E). The efficiency cores add no
useful throughput for this bandwidth-bound work and make the `@threads :static` barrier
wait on the slowest chunk; `:dynamic` does not fix it. With UMFPACK, `-t 8` actively
*regressed* (allocation contention, 0.9× erratic); with KLU's allocation-free refactor
that regression is gone — `-t 8` simply plateaus at ~`-t 4` throughput rather than
helping. So 4 threads remains the sweet spot here, ~2× full-solve over single-threaded.

**Outlook.** A homogeneous many-core server should thread better and more predictably.
KLU is sparse-direct: the QP benchmark above has a *dense* `M` block (worst case for KLU,
edge narrows by `n = 100`); genuinely sparse 1–2k targets (block-tridiagonal trajectory
games) should favor it more — worth confirming at target size/structure. The real batched
throughput win remains the GPU (cuDSS batched, D1a).
