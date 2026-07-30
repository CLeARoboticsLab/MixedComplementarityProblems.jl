const REPO = get(ENV, "REPO_ROOT", normpath(joinpath(@__DIR__, "..", "..", "..")))
using CUDA, CUDSS
include(joinpath(REPO, "benchmark", "SolverBenchmarks.jl"))
using .SolverBenchmarks
const MCP = SolverBenchmarks
using KernelAbstractions: KernelAbstractions
using Statistics: median
using MixedComplementarityProblems: MixedComplementarityProblems
using Adapt: Adapt
import Random

btype = MCP.TrajectoryGameBenchmark(); pkw = (; horizon=30); N=1024; tol=1e-4
w = MCP.benchmark_throughput(btype; num_samples=1, tol, problem_kwargs=pkw,
        run_sequential_ip=false, run_path=false, device=KernelAbstractions.CPU(), use_initial_guess=true)
mcp = w.batched_mcp
rng = Random.MersenneTwister(1)
θs = [MCP.generate_random_parameter(btype; rng, pkw...) for _ in 1:N]
Θ_host = reduce(hcat, θs)

function timeit(device, warm; reps=5)
    Θ = Adapt.adapt(device, Θ_host)
    X₀ = warm ? MCP.generate_initial_guess(btype, mcp, Θ, device; pkw...) : nothing
    solve() = MixedComplementarityProblems.solve(MixedComplementarityProblems.BatchedInteriorPoint(),
                  mcp, Θ; tol, regularize_linear_solve=:identity, device, X₀)
    sol = solve(); device isa CUDA.CUDABackend && CUDA.synchronize()   # warmup
    nsolved = count(==(:solved), sol.status)
    ts = Float64[]
    for _ in 1:reps
        t = @elapsed begin; s = solve(); device isa CUDA.CUDABackend && CUDA.synchronize(); end
        push!(ts, t)
    end
    (; med=median(ts), all=round.(ts;digits=3), nsolved)
end

for warm in (false, true)
    cpu = timeit(KernelAbstractions.CPU(), warm)
    gpu = timeit(CUDA.CUDABackend(), warm)
    println("=== game T=30 N=1024  warm=$warm  (median-of-5) ===")
    println("  CPU: med=$(round(cpu.med;digits=3))s all=$(cpu.all) solved=$(cpu.nsolved)")
    println("  GPU: med=$(round(gpu.med;digits=3))s all=$(gpu.all) solved=$(gpu.nsolved)")
    println("  GPU/CPU=$(round(gpu.med/cpu.med;digits=2)) (>1 => GPU slower)")
end
