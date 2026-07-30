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
const N = 1024; const TOL = 1e-4; const T = 50    # d = 3500, GPU wins all-active

btype = MCP.TrajectoryGameBenchmark()
w = MCP.benchmark_throughput(btype; num_samples=1, tol=TOL, problem_kwargs=(; horizon=T),
        run_sequential_ip=false, run_path=false, device=KernelAbstractions.CPU())
mcp = w.batched_mcp

function setup(device)
    rng = Random.MersenneTwister(1)
    Θ = Adapt.adapt(device, reduce(hcat, [MCP.generate_random_parameter(btype; rng, horizon=T) for _ in 1:N]))
    nx=mcp.unconstrained_dimension; ny=mcp.constrained_dimension; d=nx+2ny
    cache = M.materialize(mcp, M.BatchedSparse(), device; batch_size=N)
    X=KernelAbstractions.zeros(device,Float64,nx,N); Y=KernelAbstractions.ones(device,Float64,ny,N); S=KernelAbstractions.ones(device,Float64,ny,N)
    F=KernelAbstractions.zeros(device,Float64,d,N); δz=KernelAbstractions.zeros(device,Float64,d,N)
    ϵ=KernelAbstractions.zeros(device,Float64,N); ϵ.=1.0; η=KernelAbstractions.zeros(device,Float64,N); η.=TOL
    M.residual!(F,mcp,X,Y,S,Θ,ϵ; device)
    # Warm up with ALL active so every instance's factor is materialized once (as in a real solve).
    full = KernelAbstractions.ones(device, Bool, N)
    for _ in 1:3
        M.jacobian!(cache,mcp,X,Y,S,Θ,ϵ,η; device, regularize_linear_solve=:identity, active=full)
        M.factorize!(cache; active=full); M.ldiv!(δz,cache,-F; active=full); sync(device)
    end
    (; cache, mcp, X, Y, S, Θ, F, δz, ϵ, η, device)
end

function timek(st, k; reps=7)
    (; cache, mcp, X, Y, S, Θ, F, δz, ϵ, η, device) = st
    amask = falses(N); amask[1:k] .= true
    active = Adapt.adapt(device, amask)
    jf() = (M.jacobian!(cache,mcp,X,Y,S,Θ,ϵ,η; device, regularize_linear_solve=:identity, active);
            M.factorize!(cache; active); sync(device); nothing)
    sv() = (M.ldiv!(δz,cache,-F; active); sync(device); nothing)
    jf(); sv()   # warm this active set
    (; jacfac = median([@elapsed jf() for _ in 1:reps]), ldiv = median([@elapsed sv() for _ in 1:reps]))
end

cpu = setup(KernelAbstractions.CPU())
gpu = setup(CUDA.CUDABackend())
println("active_k,cpu_jacfac_ms,gpu_jacfac_ms,jacfac_GPU/CPU,cpu_ldiv_ms,gpu_ldiv_ms,ldiv_GPU/CPU")
for k in (1024, 512, 256, 128, 64, 32)
    c = timek(cpu, k); g = timek(gpu, k)
    ms(x)=round(1000x;digits=2)
    println("$k,$(ms(c.jacfac)),$(ms(g.jacfac)),$(round(g.jacfac/c.jacfac;digits=2)),$(ms(c.ldiv)),$(ms(g.ldiv)),$(round(g.ldiv/c.ldiv;digits=2))")
    flush(stdout)
end
println("=== ACTIVE_FRAC DONE (T=$T, d=3500, N=$N) ===")
