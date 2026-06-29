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

```julia
julia> include("SolverBenchmarks.jl")
julia> data = SolverBenchmarks.benchmark_throughput(; num_samples = 256);
julia> SolverBenchmarks.throughput_summary(data)
```

Limitations:

- Currently supports the quadratic-program benchmark only. The trajectory game cannot use
  the batched solver yet: its MCP is built with internal η-regularization, which the
  kernel evaluators `BatchedInteriorPoint` requires do not support, and its large KKT
  system would stress `SerialForm` codegen compile time (D2 in the design doc). Batched
  trajectory games are a follow-up that rides along with the GPU work.
- Keep `num_primals` modest. The QP's symbolic Hessian block is dense, so large
  `num_primals` makes kernel-evaluator compilation slow.
