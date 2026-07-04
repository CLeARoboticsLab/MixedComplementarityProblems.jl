""" Sweep `num_samples` (batch size) for `BatchedInteriorPoint` GPU vs CPU throughput, at
a FIXED CPU thread count (set via `julia -t N` at process startup — Julia's thread count
can't change at runtime, unlike `thread_scaling_benchmark.jl`'s multi-process sweep).

Unlike CPU multithreading, the GPU device has no separate "thread count" knob: our own
KernelAbstractions kernels (`_residual_kernel!`, `_jacobian_kernel!`,
`_jacobian_θ_kernel!` in `src/batched_solver.jl`) launch with `ndrange = num_samples` —
one GPU thread per batch element — and cuDSS's batched sparse LU decides internally how
to parallelize across the batch. So GPU utilization is driven entirely by `num_samples`,
not by anything analogous to `Threads.nthreads()`. This sweep is how you see the
GPU/CPU throughput gap change as a function of batch size, at one fixed CPU thread count.

Run from `benchmark/gpu/` (`julia -t 32 --project=.`):

```julia
julia> include("gpu_scaling_benchmark.jl")
julia> data = gpu_scaling_benchmark();
julia> gpu_scaling_summary(data)
```

The batched/PATH MCPs (kernel evaluators) are built ONCE (against `num_samples = 1`) and
reused for every sweep point — their symbolic structure doesn't depend on batch size. At
large `num_samples`, sequential PATH (always single-threaded, one instance at a time) and
eventually CPU-batched become so slow relative to the GPU that running them stops being
informative; `path_max_samples` / `cpu_max_samples` skip them above a threshold rather
than burning wall-clock on a foregone conclusion — the sweep still reports GPU-alone
throughput at those larger sizes.

Verified on an RTX 4090 (24 GB) with the default QP problem size (`num_primals = 32,
num_inequalities = 16`): `num_samples = 262144` throws a real `cudssExecute` "Out of GPU
memory" error from the batched sparse LU factorization (not a hang) — cuDSS's batched
factorization workspace for that many simultaneous systems exceeds 24 GB. The default
sweep stops at `65536` to stay well clear of this; pass a larger `num_samples_sweep`
explicitly (and expect an `ERROR: Out of GPU memory` near this range) if you want to find
the exact ceiling on a different GPU/problem size.
"""

using CUDA: CUDA
using CUDSS: CUDSS   # loads MixedComplementarityProblemsCUDSSExt once both are present

include(joinpath(@__DIR__, "..", "SolverBenchmarks.jl"))

using .SolverBenchmarks
using KernelAbstractions: KernelAbstractions

"Print one result row immediately (not deferred to a final summary), so a later crash
(e.g. GPU OOM on a bigger `num_samples`) doesn't erase already-computed results."
function _log_point(p)
    pieces = [
        rpad("N=$(p.num_samples)", 12),
        rpad(string("GPU ", round(p.gpu.total_time; digits = 3), " s"), 16),
        rpad(string(round(p.num_samples / p.gpu.total_time; digits = 1), " prob/s"), 16),
    ]
    if !isnothing(p.cpu)
        push!(
            pieces,
            rpad(string("CPU ", round(p.cpu.total_time; digits = 3), " s"), 16),
            string("GPU/CPU ", round(p.cpu.total_time / p.gpu.total_time; digits = 2), "×  "),
        )
    else
        push!(pieces, "CPU skipped  ")
    end
    if !isnothing(p.path)
        push!(pieces, string("GPU/PATH ", round(p.path.total_time / p.gpu.total_time; digits = 2), "×"))
    end
    @info string(pieces...)
end

"""
    gpu_scaling_benchmark(benchmark_type = SolverBenchmarks.QuadraticProgramBenchmark(); kwargs...)

Returns `(; nthreads, points)`, where `points` is a `Vector` of `(; num_samples, gpu, cpu,
path)` — `gpu`/`cpu` are `benchmark_throughput`'s `batched` result at that device (`cpu`
is `nothing` above `cpu_max_samples`), `path` is `nothing` above `path_max_samples`. Each
point is ALSO logged immediately as it completes (`_log_point`) — the whole run isn't
returned until every `num_samples` finishes, so a later crash (e.g. GPU OOM on a bigger
batch size) would otherwise erase already-computed results.
"""
function gpu_scaling_benchmark(
    benchmark_type = SolverBenchmarks.QuadraticProgramBenchmark();
    num_samples_sweep = [64, 256, 1024, 4096, 16384, 65536],
    problem_kwargs = nothing,
    tol = 1e-4,
    cpu_max_samples = 16384,
    path_max_samples = 16384,
)
    CUDA.functional() || error(
        "CUDA.functional() is false on this machine — no NVIDIA GPU visible. Run " *
        "`using CUDA; CUDA.versioninfo()` to diagnose.",
    )

    extra = isnothing(problem_kwargs) ? (;) : (; problem_kwargs)

    @info "Building batched/PATH MCPs once (kernel evaluators don't depend on batch size)..."
    warmup = SolverBenchmarks.benchmark_throughput(
        benchmark_type;
        num_samples = 1,
        tol,
        run_sequential_ip = false,
        extra...,
    )
    batched_mcp = warmup.batched_mcp
    path_mcp = warmup.path_mcp

    points = map(num_samples_sweep) do n
        run_path_here = n <= path_max_samples
        @info "num_samples = $n: running GPU$(run_path_here ? " + PATH" : "")..."

        # `extra...` (i.e. `problem_kwargs`) must be forwarded here too, even though
        # `batched_mcp`/`path_mcp` are already built: `benchmark_throughput` calls
        # `generate_test_problem`/`generate_random_parameter` UNCONDITIONALLY with
        # `problem_kwargs` before checking whether an MCP was supplied, to regenerate the
        # parameter vectors — omitting it here silently falls back to
        # `benchmark_throughput`'s own QP-shaped default, which is wrong for other
        # benchmark types (crashes loudly for `TrajectoryGameBenchmark`, since its
        # `generate_test_problem` doesn't accept `num_primals`/`num_inequalities` at all).
        data_gpu = SolverBenchmarks.benchmark_throughput(
            benchmark_type;
            num_samples = n,
            tol,
            device = CUDA.CUDABackend(),
            batched_mcp,
            path_mcp,
            run_sequential_ip = false,
            run_path = run_path_here,
            extra...,
        )

        data_cpu = if n <= cpu_max_samples
            @info "num_samples = $n: running CPU ($(Threads.nthreads()) threads)..."
            SolverBenchmarks.benchmark_throughput(
                benchmark_type;
                num_samples = n,
                tol,
                device = KernelAbstractions.CPU(),
                batched_mcp,
                path_mcp,
                run_sequential_ip = false,
                run_path = false,
                extra...,
            )
        end

        p = (;
            num_samples = n,
            gpu = data_gpu.batched,
            cpu = isnothing(data_cpu) ? nothing : data_cpu.batched,
            path = data_gpu.path,
        )
        _log_point(p)

        # Drop this point's now-dead batched-state CuArrays and force a real reclaim
        # before moving to the NEXT (likely bigger) `num_samples` — Julia's GC triggers on
        # HOST allocation pressure, not GPU memory pressure, so without this, dead device
        # memory from THIS point can sit unreclaimed while the next point's (bigger)
        # buffers are allocated on top of it, needlessly tightening the ceiling.
        GC.gc()
        CUDA.reclaim()

        p
    end

    (; nthreads = Threads.nthreads(), points)
end

"Print a `gpu_scaling_benchmark` summary table (results already logged incrementally by
`_log_point` as they were computed; this just re-prints them together)."
function gpu_scaling_summary(data)
    (; nthreads, points) = data
    @info "BatchedInteriorPoint GPU vs CPU ($nthreads threads) scaling over num_samples:"
    foreach(_log_point, points)
    data
end
