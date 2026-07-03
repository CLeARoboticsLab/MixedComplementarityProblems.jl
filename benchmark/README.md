# Solver Benchmarks

Benchmarking `MixedComplementarityProblems` solver(s) against PATH.

## Instructions

This directory provides code to benchmark the `InteriorPoint` solver against `PATH`, accessed via `ParametricMCPs` and `PATHSolver`. Currently, we provide two different benchmark problems: (i) a set of randomly-generated sparse quadratic programs with user-specified numbers of primal variables and inequality constraints, and (ii) the lane changing trajectory game from `examples/`, with initial conditions randomized. To run (with the REPL activated within this directory):

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

## GPU vs CPU (`benchmark/gpu/`)

`benchmark/gpu/` is a **separate environment** (its own `Project.toml`, with `CUDA`/`CUDSS`
as hard deps) for comparing `BatchedInteriorPoint` on `CUDABackend()` against CPU. It's
kept out of the main `benchmark/Project.toml` because CUDA.jl/CUDSS.jl only ship artifacts
for NVIDIA-capable platforms — this keeps the rest of `benchmark/` portable (e.g. to Apple
Silicon). Run from an NVIDIA machine:

```julia
julia> include("gpu_batched_benchmark.jl")
julia> data = benchmark_gpu_vs_cpu(; num_samples = 4096);          # QP
julia> gpu_speedup_summary(data)
```
