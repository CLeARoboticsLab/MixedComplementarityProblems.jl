# Solver Benchmarks

Benchmarking `MixedComplementarityProblems` solver(s) against PATH.

## Instructions

This directory provides code to benchmark the `InteriorPoint` solver against `PATH`, accessed via `ParametricMCPs` and `PATHSolver`. Currently, we provide two different benchmark problems: (i) a set of randomly-generated sparse quadratic programs with user-specified numbers of primal variables and inequality constraints, and (ii) the lane changing trajectory game from `examples/`, with initial conditions randomized around a fixed canonical merge scenario (see "Trajectory game sampling" below). To run (with the REPL activated within this directory):

```julia
julia> include("SolverBenchmarks.jl")
julia> data = SolverBenchmarks.benchmark(SolverBenchmarks.TrajectoryGameBenchmark(); num_samples = 25);
julia> SolverBenchmarks.summary_statistics(data)
```

If you want to re-run with different kwargs, you may be able to reuse the MCPs and avoid waiting for them to compile:

```julia
julia> data = SolverBenchmarks.benchmark(SolverBenchmarks.TrajectoryGameBenchmark(); num_samples = 250, data.ip_mcp, data.path_mcp);
julia> SolverBenchmarks.summary_statistics(data)
```

## Batched throughput (CPU multithreading)

`benchmark_throughput` showcases the batched `BatchedInteriorPoint` solver, which solves a
whole *batch* of problems (sharing one MCP structure, differing in their parameters) in a
single multithreaded call. This is where CPU multithreading pays off: the batched solver
factorizes/solves all instances in parallel across threads, whereas PATH (and the unbatched
`InteriorPoint`) process them one at a time on a single thread. It reports total wall-clock
to clear `num_samples` problems for PATH, the sequential `InteriorPoint`, and the batched
solver. **Start Julia with several threads** (e.g. `julia -t 4`); on heterogeneous CPUs
(Apple silicon) prefer `-t <#performance-cores>` (see §10 of `docs/gpu_kkt_design.md`).

It works for both benchmark types — the quadratic program and the trajectory game (the
latter is internally η-regularized, so it is solved with the `:internal` scheme):

```julia
julia> include("SolverBenchmarks.jl")
julia> data = SolverBenchmarks.benchmark_throughput(; num_samples = 256);                              # QP
julia> data = SolverBenchmarks.benchmark_throughput(SolverBenchmarks.TrajectoryGameBenchmark();
                                                     num_samples = 64, problem_kwargs = (; horizon = 3));  # game
julia> SolverBenchmarks.throughput_summary(data)
```

Caveat: `BatchedInteriorPoint` requires kernel evaluators built with `SerialForm` codegen,
whose **compile time grows with the symbolic KKT size** (D2 in the design doc). Keep the
QP's `num_primals` modest (dense symbolic Hessian) and the game's `horizon` modest — large
problems can make the one-time kernel-evaluator build slow. `generate_test_problem`'s own
default (`num_primals = 100`) is well past this cliff (minutes, not seconds) — always pass
`problem_kwargs` explicitly if you're not using `benchmark_throughput`'s own default of
`(; num_primals = 32, num_inequalities = 16)`.

## CPU thread scaling

`thread_scaling_benchmark.jl` compares `BatchedInteriorPoint` throughput across several
thread counts. Julia's thread count is fixed at process startup, so this spawns one
short-lived `Distributed` worker per thread count (each with its own `-t N`) rather than
requiring you to relaunch Julia manually:

```julia
julia> include("thread_scaling_benchmark.jl")
julia> data = thread_scaling_benchmark(:qp; num_samples = 1024);
julia> thread_scaling_summary(data)

julia> data = thread_scaling_benchmark(:trajectory_game; num_samples = 128, problem_kwargs = (; horizon = 3));
julia> thread_scaling_summary(data)
```

Default thread counts are capped at `Sys.CPU_THREADS` — CPU oversubscription plateaus or
regresses throughput rather than continuing to scale, unlike a GPU. Pass `problem_kwargs`
explicitly (see the caveat above); the default is deliberately `nothing` (omitted) rather
than `(;)`, so `benchmark_throughput`'s own fast default still applies.

## Trajectory game sampling & warm start

`TrajectoryGameBenchmark`'s `generate_random_parameter` samples initial positions from a
small rectangle around a fixed canonical merge scenario (P1 stays in the leftmost lane,
P2 merges in from the rightmost lane), rather than uniformly over the whole road — see
its docstring in `trajectory_game_benchmark.jl` for the full rationale. Two things worth
knowing if you touch this benchmark:
- **`lead_offset`/`lead_vy_boost`/`initial_vy` are load-bearing, not cosmetic.** Both
  players share the same lane preference, so the only feasible equilibria are "P1 leads"
  or "P2 leads" — a symmetric, essentially discrete choice. Sampling both players
  symmetrically leaves that choice ambiguous at the initial guess, which makes the
  Newton-based complementarity solver oscillate between the two candidate equilibria
  instead of converging — this is *why* `solved` fraction used to collapse at longer
  horizons (down to 0% at `horizon = 100`), independent of `height`/road length (that was
  the first, disproven hypothesis). Giving P2 a head start and higher rollout velocity
  breaks the symmetry and recovers most of the solved fraction (see numbers below).
- **`generate_initial_guess` supplies `BatchedInteriorPoint`'s `X₀`** with a zero-input
  rollout (dynamics-aware, via the same `zero_input_trajectory`/`pack_trajectory` helpers
  `examples/lane_change.jl`'s receding-horizon strategy uses) instead of the solver's
  default all-zero cold start. This is dispatched per `benchmark_type` and wired into
  `benchmark_throughput` automatically — the sampling fix alone (without this) recovers
  only a fraction of the solved-fraction gain.

## GPU vs CPU (`benchmark/gpu/`)

`benchmark/gpu/` is a **separate environment** (its own `Project.toml`, with `CUDA`/`CUDSS`
as hard deps) for comparing `BatchedInteriorPoint` on `CUDABackend()` against CPU. It's
kept out of the main `benchmark/Project.toml` because CUDA.jl/CUDSS.jl only ship artifacts
for NVIDIA-capable platforms — this keeps the rest of `benchmark/` portable (e.g. to Apple
Silicon). Three scripts, all run from an NVIDIA machine (`julia -t N --project=.`):

```julia
julia> include("gpu_batched_benchmark.jl")
julia> data = benchmark_gpu_vs_cpu(; num_samples = 4096);          # single CPU-vs-GPU point
julia> gpu_speedup_summary(data)

julia> include("gpu_scaling_benchmark.jl")                          # sweep num_samples, fixed problem size
julia> data = gpu_scaling_benchmark();
julia> gpu_scaling_summary(data)

julia> include("problem_size_scaling.jl")                           # sweep problem size, fixed num_samples
julia> data = problem_size_scaling_benchmark();
julia> problem_size_scaling_summary(data)
```

**Current status (RTX 4090, as of 2026-07-04): GPU beats 32 CPU threads by a consistent
2.5-2.8x on the trajectory game once the per-instance problem is large enough (`horizon
≳ 30`, KKT dimension `d ≳ 2000`); below that, and for small dense QPs at small batch
sizes, CPU is still faster or roughly at parity.** The headline number depends heavily on
*which* regime you're in — see the breakdown below rather than quoting a single ratio.

- **cuDSS tuning:** `factorization_alg = "algo1"` (set manually in
  `ext/MixedComplementarityProblemsCUDSSExt.jl`, since `LinearAlgebra.lu()`'s convenience
  wrapper doesn't expose it) gives a verified 10-17% speedup with byte-for-byte identical
  solve behavior (same `outer_iters`/`total_iters`/solved-counts) on both benchmarks.
  `algo2`-`algo5` are unsupported or slower; `reordering_alg`/`use_superpanels` don't help.
- **Stall detection** (`max_stall_rounds` in `src/batched_solver.jl`) cut wall-clock
  2.7-7x on both devices by ending instances that neither converge nor diverge instead of
  dragging every solve to `max_outer_iters` — necessary groundwork, but on its own it
  *shrank* GPU's apparent advantage rather than growing it (GPU was previously winning
  partly by handling wasted iterations better, not via a genuine linear-algebra edge).
- **Why GPU wins at larger `horizon`:** raw per-instance `jacobian!+factorize!` cost
  crosses over in GPU's favor right around `d ≈ 3500` (`horizon ≈ 50`) — confirmed by
  isolated, apples-to-apples timing at fixed batch size, independent of solved fraction
  or iteration count. `outer_iters` stays flat (9-16) across `horizon = 10..100`, so this
  is **not** explained by `BatchedInteriorPoint`'s CPU-only active-set skip (which would,
  if anything, favor CPU *more* as harder instances drop out early) — it's cuDSS's batched
  factorization getting relatively cheaper per instance as the matrix grows, amortizing
  its kernel-launch/occupancy overhead better than CPU's per-thread KLU factorization.
- **Trajectory game, with the sampling/warm-start fix above** (`N = 1024`, `height = 50`):
  GPU/CPU wall-clock ratio is 2.68x (`horizon=30`), 2.50x (`horizon=50`), 2.48x
  (`horizon=70`), 2.81x (`horizon=100`) — consistently GPU-favorable. Solved fraction is
  92% (`horizon=30`), 57% (`horizon=50`), 34-36% (`horizon=70`), 23-25% (`horizon=100`) —
  a large improvement over the pre-fix collapse (as low as 0% at `horizon=100`), but still
  short of a "realistic, high-confidence" benchmark at `horizon ≳ 50`; further tuning
  (larger `lead_offset`/`lead_vy_boost`, or a genuinely better warm start) is still open.
- At `horizon ≤ 20` (small `d`), CPU remains faster (GPU 2-2.5x slower) — GPU only pulls
  ahead once there's enough per-instance work to amortize its overhead.
- QP (small, dense, `num_primals = 32, num_inequalities = 16`): GPU/CPU ratio close to
  parity (0.5-1.4x) across batch sizes; problem-size sweep (32/64/128 primals) is
  non-monotonic (GPU wins at 128 primals, up to 2.9x, but loses at 64, 0.6-0.8x) —
  confounded by the random QP generator's solved-fraction changing sharply with
  `num_primals` (41% → 96% → 100%), not yet a clean isolated comparison.
- `num_samples = 16384`+ currently OOMs for the trajectory game with an unexplained low
  reported memory usage (~10%) at failure — not yet root-caused; dropped from the sweep
  for now.
