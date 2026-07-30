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

**Current status (RTX 4090, 32 threads, updated 2026-07-30): the CPU is faster than the GPU
end-to-end on the trajectory game at every tested horizon; the GPU wins end-to-end only on
large *dense* per-instance systems (≳128-primal QPs, 2–3×) or large batches of them.** Both
batched backends clear a batch far faster than sequential `PATH` (game: CPU ~60–90×, GPU
~20–55×). Raw data and analysis scripts are in [`benchmark/results/`](../results/).

> **Correction.** An earlier version of this section (and PRs #54/#55) claimed the GPU *beats*
> the CPU 2.5–2.8× on the game. That number was the GPU/CPU wall-clock *ratio* with the GPU in
> the numerator — i.e. the GPU is ~2.5× **slower** — mislabeled as GPU-favorable. Re-measured
> here (median-of-N), the raw ratios reproduce, but the direction is the opposite of the old
> headline.

- **cuDSS tuning:** `factorization_alg = "algo1"` (set manually in
  `ext/MixedComplementarityProblemsCUDSSExt.jl`, since `LinearAlgebra.lu()`'s convenience
  wrapper doesn't expose it) gives a verified 10-17% speedup with byte-for-byte identical
  solve behavior (same `outer_iters`/`total_iters`/solved-counts) on both benchmarks.
  `algo2`-`algo5` are unsupported or slower; `reordering_alg`/`use_superpanels` don't help.
- **Stall detection** (`max_stall_rounds` in `src/batched_solver.jl`) cut wall-clock
  2.7-7x on both devices by ending instances that neither converge nor diverge instead of
  dragging every solve to `max_outer_iters`.
- **The GPU factorization kernel *does* cross over — but end-to-end the CPU still wins.**
  Isolated per-call timing (`percall_timing.csv`, all instances active) shows the GPU's
  `jacobian!+factorize!` crossing over CPU's around `d ≈ 2500` (jac+fac GPU/CPU 1.92× at
  `d=700` → 0.72× at `d=4900`) and `ldiv!` crossing even earlier (~`d≈1800`). But that
  per-call number assumes the *whole* batch is factorized every step, which only holds on the
  first Newton iteration. The real driver of end-to-end cost is `BatchedInteriorPoint`'s
  **active-set skip**: on CPU each Newton step factorizes only the still-active instances
  (cost ∝ active count), while cuDSS always factorizes the whole batch (flat cost — `active`
  is a no-op on GPU). At `d = 3500` (`active_fraction_T50.csv`) the `jac+fac` GPU/CPU ratio
  goes from 0.77× with all 1024 active to 8.1× with only 32 active. Since a real solve's active
  set collapses fast as instances sub-converge, the CPU spends most of the solve in the regime
  where it dominates. (This reverses the earlier claim here that the active-set skip was *not*
  the explanation.)
- **Trajectory game, end-to-end** (`N = 1024`, warm, median-of-5): `horizon=30` CPU 4.54s vs
  GPU 7.90s (GPU 1.74× slower); `horizon=50` CPU 12.6s vs GPU 17.9s. Solved fraction (warm) is
  92% (`horizon=30`), 57% (`horizon=50`) — matching earlier runs; cold-start collapses it
  (≈17% at `horizon=30`), so cold high-horizon GPU/CPU ratios compare two mostly-failing
  backends and aren't meaningful. GPU timings are also higher-variance at large horizon.
- **QP** (`num_primals=32, num_inequalities=16`): the GPU pulls ahead as the batch grows
  (GPU/CPU ~0.6× at `N=4096`, i.e. GPU ~1.7× faster), while the CPU is faster at small batch
  sizes. Problem-size sweep (32/64/128 primals) is non-monotonic — GPU wins clearly at 128
  primals (2–3×) but loses at 64 — confounded by the QP generator's solved fraction changing
  sharply with `num_primals` (41% → 96% → 100%); not a clean isolated comparison.
