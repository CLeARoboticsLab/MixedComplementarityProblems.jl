""" Sweep per-instance PROBLEM SIZE (not batch size) for `BatchedInteriorPoint` GPU vs
CPU throughput, to test whether a bigger per-system KKT dimension widens the GPU/CPU
gap. Complements `gpu_scaling_benchmark.jl` (which fixes problem size and sweeps
`num_samples`) — here `num_samples` is fixed at a few representative values and
`problem_kwargs` (`num_primals`, `num_inequalities`) varies instead.

Motivation: the default QP (`num_primals = 32, num_inequalities = 16`) showed GPU/CPU
roughly at parity once the straggler problem was fixed (see the stall-detection change
in `src/batched_solver.jl`) — cuDSS's batched sparse LU doesn't have much of an edge over
32-way threaded KLU for such a tiny per-instance system. Larger systems give the GPU more
arithmetic work per instance to amortize kernel-launch/memory-latency overhead against,
so the hypothesis is that GPU/CPU should improve as `num_primals` grows.

Run from `benchmark/gpu/` (`julia -t 32 --project=.`):

```julia
julia> include("problem_size_scaling.jl")
julia> data = problem_size_scaling_benchmark();
julia> problem_size_scaling_summary(data)
```

WARNING: kernel-evaluator compile time grows with the symbolic KKT size (dense Hessian
block for a QP) — see the caveat in `benchmark/README.md`. Each problem size in the sweep
pays its own one-time compile cost; expect this to take substantially longer than
`gpu_scaling_benchmark.jl` per size as `num_primals` grows.
"""

using CUDA: CUDA
using CUDSS: CUDSS   # loads MixedComplementarityProblemsCUDSSExt once both are present

include(joinpath(@__DIR__, "..", "SolverBenchmarks.jl"))

using .SolverBenchmarks
using KernelAbstractions: KernelAbstractions

"Print one result row immediately (not deferred to a final summary), so a later crash
(e.g. GPU OOM on a bigger problem size) doesn't erase already-computed results."
function _log_point(num_primals, num_inequalities, p)
    @info string(
        rpad("primals=$num_primals ineq=$num_inequalities", 24),
        rpad("N=$(p.num_samples)", 12),
        rpad(string("GPU ", round(p.gpu.total_time; digits = 3), " s"), 16),
        rpad(string("CPU ", round(p.cpu.total_time; digits = 3), " s"), 16),
        "GPU/CPU ", round(p.cpu.total_time / p.gpu.total_time; digits = 2), "×  ",
        "solved(gpu/cpu) ", p.gpu.num_solved, "/", p.cpu.num_solved, " of ", p.num_samples,
    )
end

"""
    problem_size_scaling_benchmark(; kwargs...)

Returns a `Vector` of `(; num_primals, num_inequalities, points)`, where `points` is a
`Vector` of `(; num_samples, gpu, cpu)` (`benchmark_throughput`'s `batched` result at
each device) for that problem size. Each point is ALSO logged immediately as it
completes (`_log_point`), since the whole run isn't returned until every problem size
finishes — a later crash (e.g. GPU OOM on a bigger size) would otherwise erase already-
computed results.
"""
function problem_size_scaling_benchmark(;
    problem_sizes = [(; num_primals = 32, num_inequalities = 16),
                      (; num_primals = 64, num_inequalities = 32),
                      (; num_primals = 128, num_inequalities = 64)],
    num_samples_sweep = [1024, 4096],
    tol = 1e-4,
)
    CUDA.functional() || error(
        "CUDA.functional() is false on this machine — no NVIDIA GPU visible. Run " *
        "`using CUDA; CUDA.versioninfo()` to diagnose.",
    )

    map(problem_sizes) do problem_kwargs
        @info "problem_kwargs = $problem_kwargs: building batched/PATH MCPs once..."
        warmup = SolverBenchmarks.benchmark_throughput(
            SolverBenchmarks.QuadraticProgramBenchmark();
            num_samples = 1, tol, run_sequential_ip = false, run_path = false, problem_kwargs,
        )
        batched_mcp = warmup.batched_mcp

        points = map(num_samples_sweep) do n
            @info "problem_kwargs = $problem_kwargs, num_samples = $n: running GPU..."
            data_gpu = SolverBenchmarks.benchmark_throughput(
                SolverBenchmarks.QuadraticProgramBenchmark();
                num_samples = n, tol, device = CUDA.CUDABackend(),
                batched_mcp, run_sequential_ip = false, run_path = false, problem_kwargs,
            )
            @info "problem_kwargs = $problem_kwargs, num_samples = $n: running CPU ($(Threads.nthreads()) threads)..."
            data_cpu = SolverBenchmarks.benchmark_throughput(
                SolverBenchmarks.QuadraticProgramBenchmark();
                num_samples = n, tol, device = KernelAbstractions.CPU(),
                batched_mcp, run_sequential_ip = false, run_path = false, problem_kwargs,
            )
            p = (; num_samples = n, gpu = data_gpu.batched, cpu = data_cpu.batched)
            _log_point(problem_kwargs.num_primals, problem_kwargs.num_inequalities, p)
            p
        end

        # Drop this size's now-dead MCP/cache/batched-state CuArrays and force a real
        # reclaim before moving to a (likely bigger) size — Julia's GC triggers on HOST
        # allocation pressure, not GPU memory pressure, so without this, dead device
        # memory from THIS size can sit unreclaimed while the NEXT size's buffers are
        # allocated on top of it, needlessly tightening the ceiling for larger sizes.
        batched_mcp = nothing
        GC.gc()
        CUDA.functional() && CUDA.reclaim()

        (; problem_kwargs..., points)
    end
end

"Print a `problem_size_scaling_benchmark` summary table (results already logged
incrementally by `_log_point` as they were computed; this just re-prints them together)."
function problem_size_scaling_summary(data)
    @info "BatchedInteriorPoint GPU vs CPU scaling over problem size:"
    for size_data in data
        (; num_primals, num_inequalities, points) = size_data
        for p in points
            _log_point(num_primals, num_inequalities, p)
        end
    end
    data
end
