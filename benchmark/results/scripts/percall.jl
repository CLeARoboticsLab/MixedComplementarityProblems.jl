const REPO = get(ENV, "REPO_ROOT", normpath(joinpath(@__DIR__, "..", "..", "..")))
using CUDA, CUDSS
include(joinpath(REPO, "benchmark", "SolverBenchmarks.jl"))
using .SolverBenchmarks; const MCP = SolverBenchmarks
using MixedComplementarityProblems: MixedComplementarityProblems; const M = MixedComplementarityProblems
using KernelAbstractions: KernelAbstractions
using Adapt: Adapt
using Statistics: median
import Random

sync(device) = device isa CUDA.CUDABackend && CUDA.synchronize()
const HORIZONS = (10, 20, 30, 50, 70)
const N = 1024
const TOL = 1e-4

# Build all MCPs at TOP LEVEL first, so the runtime-eval'd kernel closures are in-world
# before the timing function is ever called (else: world-age MethodError on the evaluator).
btype = MCP.TrajectoryGameBenchmark()
mcps = Dict{Int,Any}()
for T in HORIZONS
    w = MCP.benchmark_throughput(btype; num_samples=1, tol=TOL, problem_kwargs=(; horizon=T),
            run_sequential_ip=false, run_path=false, device=KernelAbstractions.CPU())
    mcps[T] = w.batched_mcp
end

function percall(mcp, T; device, reps=7, warmup=3)
    pkw = (; horizon=T)
    rng = Random.MersenneTwister(1)
    Θ = Adapt.adapt(device, reduce(hcat, [MCP.generate_random_parameter(btype; rng, pkw...) for _ in 1:N]))
    nx=mcp.unconstrained_dimension; ny=mcp.constrained_dimension; d=nx+2ny; B=N
    cache = M.materialize(mcp, M.BatchedSparse(), device; batch_size=B)
    X=KernelAbstractions.zeros(device,Float64,nx,B); Y=KernelAbstractions.ones(device,Float64,ny,B); S=KernelAbstractions.ones(device,Float64,ny,B)
    F=KernelAbstractions.zeros(device,Float64,d,B); δz=KernelAbstractions.zeros(device,Float64,d,B)
    ϵ=KernelAbstractions.zeros(device,Float64,B); ϵ.=1.0; η=KernelAbstractions.zeros(device,Float64,B); η.=TOL
    M.residual!(F,mcp,X,Y,S,Θ,ϵ; device)
    jac()  = (M.jacobian!(cache,mcp,X,Y,S,Θ,ϵ,η; device, regularize_linear_solve=:identity); nothing)
    fac()  = (M.factorize!(cache); sync(device); nothing)
    solv() = (M.ldiv!(δz,cache,-F); sync(device); nothing)
    for _ in 1:warmup; jac(); fac(); solv(); end
    tj = median([@elapsed jac()  for _ in 1:reps])
    tf = median([@elapsed fac()  for _ in 1:reps])
    ts = median([@elapsed solv() for _ in 1:reps])
    (; d, jac=tj, fac=tf, ldiv=ts, jacfac=tj+tf)
end

println("T,d,device,jac_ms,fac_ms,ldiv_ms,jacfac_ms")
for T in HORIZONS
    cpu = percall(mcps[T], T; device=KernelAbstractions.CPU())
    gpu = percall(mcps[T], T; device=CUDA.CUDABackend())
    ms(x)=round(1000x;digits=2)
    println("$T,$(cpu.d),CPU,$(ms(cpu.jac)),$(ms(cpu.fac)),$(ms(cpu.ldiv)),$(ms(cpu.jacfac))")
    println("$T,$(gpu.d),GPU,$(ms(gpu.jac)),$(ms(gpu.fac)),$(ms(gpu.ldiv)),$(ms(gpu.jacfac))")
    println("  >> T=$T d=$(cpu.d): fac GPU/CPU=$(round(gpu.fac/cpu.fac;digits=2))  jacfac GPU/CPU=$(round(gpu.jacfac/cpu.jacfac;digits=2))  ldiv GPU/CPU=$(round(gpu.ldiv/cpu.ldiv;digits=2))")
    flush(stdout)
end
println("=== PERCALL DONE ===")
