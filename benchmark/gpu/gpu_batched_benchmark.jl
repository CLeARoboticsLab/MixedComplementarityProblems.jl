""" GPU vs CPU throughput comparison for `BatchedInteriorPoint`, via CUDSS
(`MixedComplementarityProblemsCUDSSExt`). Lives in its own environment
(`benchmark/gpu/Project.toml`) because CUDA.jl/CUDSS.jl only have artifacts for
NVIDIA-capable platforms — keeping them out of `benchmark/Project.toml` keeps the rest
of `benchmark/` portable (e.g. to Apple Silicon, per its README).

Run from within this directory (`julia --project=.`):

```julia
julia> include("gpu_batched_benchmark.jl")
julia> data = benchmark_gpu_vs_cpu(; num_samples = 4096);
julia> gpu_speedup_summary(data)
```

Reuses `../SolverBenchmarks.jl`'s problem generators and `benchmark_throughput` (device
is just a KernelAbstractions backend value threaded through — the CPU-only file never
needs to know CUDA exists). The batched MCP (kernel evaluators) is built ONCE against
the CPU run and reused for the GPU run — its symbolic structure doesn't depend on
device, so this skips a redundant `eval`/compile pass.
"""

using CUDA: CUDA
using CUDSS: CUDSS   # loads MixedComplementarityProblemsCUDSSExt once both are present

include(joinpath(@__DIR__, "..", "SolverBenchmarks.jl"))

using .SolverBenchmarks
using KernelAbstractions: KernelAbstractions

"""
    benchmark_gpu_vs_cpu(benchmark_type = SolverBenchmarks.QuadraticProgramBenchmark(); kwargs...)

Runs `SolverBenchmarks.benchmark_throughput` twice against the SAME batch of problems —
once with `device = KernelAbstractions.CPU()`, once with `device = CUDA.CUDABackend()`
— and returns both results. `kwargs` are forwarded to `benchmark_throughput` (e.g.
`num_samples`, `problem_kwargs`, `tol`).
"""
function benchmark_gpu_vs_cpu(
    benchmark_type = SolverBenchmarks.QuadraticProgramBenchmark();
    kwargs...,
)
    CUDA.functional() || error(
        "CUDA.functional() is false on this machine — no NVIDIA GPU visible. Run " *
        "`using CUDA; CUDA.versioninfo()` to diagnose.",
    )

    @info "Running CPU batched throughput ($(Threads.nthreads()) threads)..."
    data_cpu = SolverBenchmarks.benchmark_throughput(
        benchmark_type;
        device = KernelAbstractions.CPU(),
        kwargs...,
    )

    @info "Running GPU batched throughput (reusing the CPU run's MCPs)..."
    data_gpu = SolverBenchmarks.benchmark_throughput(
        benchmark_type;
        device = CUDA.CUDABackend(),
        batched_mcp = data_cpu.batched_mcp,
        path_mcp = data_cpu.path_mcp,
        num_samples = data_cpu.num_samples,
        tol = data_cpu.tol,
    )

    (; cpu = data_cpu, gpu = data_gpu)
end

"Print a CPU-vs-GPU throughput summary from `benchmark_gpu_vs_cpu` data."
function gpu_speedup_summary(data)
    (; cpu, gpu) = data
    SolverBenchmarks.throughput_summary(cpu)
    SolverBenchmarks.throughput_summary(gpu)

    speedup_vs_cpu = cpu.batched.total_time / gpu.batched.total_time
    speedup_vs_path = cpu.path.total_time / gpu.batched.total_time
    @info string(
        "GPU BatchedInteriorPoint speedup: ",
        round(speedup_vs_cpu; digits = 2),
        "× vs CPU BatchedInteriorPoint ($(cpu.nthreads) threads),  ",
        round(speedup_vs_path; digits = 2),
        "× vs PATH  ",
        "(CPU batched: ", round(cpu.batched.total_time; digits = 3), " s,  ",
        "GPU batched: ", round(gpu.batched.total_time; digits = 3), " s,  ",
        "PATH: ", round(cpu.path.total_time; digits = 3), " s)",
    )

    (; gpu_vs_cpu_batched = speedup_vs_cpu, gpu_vs_path = speedup_vs_path)
end
