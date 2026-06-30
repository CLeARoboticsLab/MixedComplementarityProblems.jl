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
problems can make the one-time kernel-evaluator build slow.
