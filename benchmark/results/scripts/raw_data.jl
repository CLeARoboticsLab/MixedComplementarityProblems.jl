const REPO = get(ENV, "REPO_ROOT", normpath(joinpath(@__DIR__, "..", "..", "..")))
# Raw per-sample benchmark data for the headline throughput + GPU-scaling results.
#   • per_instance.csv : per-instance solve time+status for the SEQUENTIAL baselines
#                        (PATH, unbatched InteriorPoint) — reveals feasible/infeasible bimodality.
#   • per_rep.csv      : per-repetition full-batch wall-clock for the BATCHED solver (CPU/GPU)
#                        and (few reps) sequential PATH totals — for mean±stddev / violins.
# STAGE=throughput_raw | scaling_raw   WARMSTART unused (we sweep warm explicitly for the game).
using CUDA, CUDSS
include(joinpath(REPO, "benchmark", "SolverBenchmarks.jl"))
using .SolverBenchmarks; const MCP = SolverBenchmarks
using MixedComplementarityProblems: MixedComplementarityProblems; const M = MixedComplementarityProblems
using ParametricMCPs: ParametricMCPs
using PATHSolver: PATHSolver
using KernelAbstractions: KernelAbstractions
using Adapt: Adapt
import Random

const RESULTS = joinpath(REPO, "benchmark", "results")
const TOL = 1e-4
const CPU = KernelAbstractions.CPU()
sync(dev) = dev isa CUDA.CUDABackend && CUDA.synchronize()
const QP = MCP.QuadraticProgramBenchmark; const GAME = MCP.TrajectoryGameBenchmark

const PI_CSV = joinpath(RESULTS, "per_instance.csv")
const PR_CSV = joinpath(RESULTS, "per_rep.csv")
pih() = isfile(PI_CSV) || open(io->println(io,"problem,size,solver,instance,time_s,solved,tol"), PI_CSV, "w")
prh() = isfile(PR_CSV) || open(io->println(io,"problem,size,solver,device,warm_start,rep,total_time_s,num_solved,num_samples,tol"), PR_CSV, "w")
pirow(a...) = open(io->println(io, join(a, ",")), PI_CSV, "a")
prrow(a...) = open(io->println(io, join(a, ",")), PR_CSV, "a")

# Build both MCPs (+ deterministic θ's) at top level → runtime-eval'd kernels are in-world.
function build(btype, pkw, N)
    w = MCP.benchmark_throughput(btype; num_samples=1, tol=TOL, problem_kwargs=pkw, device=CPU)
    rng = Random.MersenneTwister(1)
    θs = [MCP.generate_random_parameter(btype; rng, pkw...) for _ in 1:N]
    (; batched_mcp=w.batched_mcp, path_mcp=w.path_mcp, θs, Θ_host=reduce(hcat, θs))
end

# per-instance sequential timings (PATH + unbatched IP), cold.
function per_instance!(problem, sz, b)
    ParametricMCPs.solve(b.path_mcp, b.θs[1]; warn_on_convergence_failure=false)          # warmup
    for (i,θ) in enumerate(b.θs)
        t = @elapsed s = ParametricMCPs.solve(b.path_mcp, θ; warn_on_convergence_failure=false)
        pirow(problem, sz, "path", i, round(t;digits=6), s.status==PATHSolver.MCP_Solved, TOL)
    end
    M.solve(M.InteriorPoint(), b.batched_mcp, b.θs[1]; tol=TOL, regularize_linear_solve=:identity) # warmup
    for (i,θ) in enumerate(b.θs)
        t = @elapsed s = M.solve(M.InteriorPoint(), b.batched_mcp, θ; tol=TOL, regularize_linear_solve=:identity)
        pirow(problem, sz, "ip_seq", i, round(t;digits=6), s.status==:solved, TOL)
    end
end

# per-rep batched full-batch wall-clock on a device, given warm on/off.
function per_rep_batched!(problem, sz, btype, pkw, b, N, device, warm; reps=20)
    Θ = Adapt.adapt(device, b.Θ_host)
    X₀ = warm ? MCP.generate_initial_guess(btype, b.batched_mcp, Θ, device; pkw...) : nothing
    run() = M.solve(M.BatchedInteriorPoint(), b.batched_mcp, Θ; tol=TOL, regularize_linear_solve=:identity, device, X₀)
    s = run(); sync(device)                                   # warmup
    ns = count(==(:solved), s.status)
    dev = device isa CUDA.CUDABackend ? "gpu" : "cpu"
    for rep in 1:reps
        t = @elapsed (run(); sync(device))
        prrow(problem, sz, "batched", dev, warm, rep, round(t;digits=6), ns, N, TOL)
    end
end

# few-rep sequential PATH totals (the serial baseline), cold.
function per_rep_path!(problem, sz, b, N; reps=3)
    ParametricMCPs.solve(b.path_mcp, b.θs[1]; warn_on_convergence_failure=false)
    for rep in 1:reps
        t = @elapsed n = count(θ->ParametricMCPs.solve(b.path_mcp, θ; warn_on_convergence_failure=false).status==PATHSolver.MCP_Solved, b.θs)
        prrow(problem, sz, "path", "na", false, rep, round(t;digits=6), n, N, TOL)
    end
end

pih(); prh()
stage = get(ENV, "STAGE", "throughput_raw")
@info "=== raw_data stage=$stage, threads=$(Threads.nthreads()) ==="

if stage == "throughput_raw"
    N = 1024
    for (problem, btype, pkw, sz) in (("qp", QP(), (;num_primals=32,num_inequalities=16), "p32i16"),
                                      ("game", GAME(), (;horizon=10), "T10"))
        @info "throughput_raw: $problem N=$N building..."
        b = build(btype, pkw, N)
        per_instance!(problem, sz, b)
        per_rep_batched!(problem, sz, btype, pkw, b, N, CPU, false)
        per_rep_batched!(problem, sz, btype, pkw, b, N, CUDA.CUDABackend(), false)
        if problem == "game"   # also warm (realistic MPC regime)
            per_rep_batched!(problem, sz, btype, pkw, b, N, CPU, true)
            per_rep_batched!(problem, sz, btype, pkw, b, N, CUDA.CUDABackend(), true)
        end
        GC.gc(); CUDA.reclaim()
    end
elseif stage == "scaling_raw"
    for (problem, btype, sweep, pkw, sz) in (
            ("qp", QP(), [64,256,1024,4096,16384], (;num_primals=32,num_inequalities=16), "p32i16"),
            ("game", GAME(), [64,256,1024,4096], (;horizon=10), "T10"))
        @info "scaling_raw: $problem building (max N=$(maximum(sweep)))..."
        for N in sweep
            b = build(btype, pkw, N)
            per_rep_batched!(problem, sz, btype, pkw, b, N, CPU, false; reps=10)
            per_rep_batched!(problem, sz, btype, pkw, b, N, CUDA.CUDABackend(), false; reps=10)
            problem == "game" && per_rep_batched!(problem, sz, btype, pkw, b, N, CUDA.CUDABackend(), true; reps=10)
            N <= 4096 && per_rep_path!(problem, sz, b, N; reps=(N<=256 ? 3 : 1))
            GC.gc(); CUDA.reclaim()
        end
    end
end
@info "=== raw_data stage=$stage DONE ==="
