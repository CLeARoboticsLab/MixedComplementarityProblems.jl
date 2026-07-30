#!/usr/bin/env julia
# Benchmark driver for the technical-report Experiments section.
#
# Runs the batched-throughput and GPU-vs-CPU experiments and appends results to a single
# long-format CSV (one row per solver/device/sweep-point) so all stages concatenate and a
# mid-sweep crash keeps partial results. Select the stage with ENV["STAGE"].
#
# Run from repo root:
#   STAGE=throughput julia -t 32 --project=benchmark/gpu <this file>
#
# Stages: throughput | gpu_scaling | problem_size | horizon | all

using CUDA: CUDA
using CUDSS: CUDSS                      # loads MixedComplementarityProblemsCUDSSExt
# Repo root: from ENV["REPO_ROOT"] if set, else the known checkout path. (This driver
# lives in a scratchpad outside the repo, so a relative walk-up won't find it.)
const REPO_ROOT = get(ENV, "REPO_ROOT",
    normpath(joinpath(@__DIR__, "..", "..", "..")))
isfile(joinpath(REPO_ROOT, "benchmark", "SolverBenchmarks.jl")) ||
    error("benchmark/SolverBenchmarks.jl not found under REPO_ROOT=$REPO_ROOT")
include(joinpath(REPO_ROOT, "benchmark", "SolverBenchmarks.jl"))
using .SolverBenchmarks
using KernelAbstractions: KernelAbstractions

const QP = SolverBenchmarks.QuadraticProgramBenchmark
const GAME = SolverBenchmarks.TrajectoryGameBenchmark
const TOL = 1e-4
const RESULTS = joinpath(REPO_ROOT, "benchmark", "results")
const CSV = joinpath(RESULTS, "experiments.csv")
mkpath(RESULTS)

# Warm-start knob: cold-start everywhere by default (WARMSTART=0) for clean, apples-to-
# apples comparisons; set WARMSTART=1 to capture a warm-started reference. Only affects
# the batched solve, and only for benchmark types with a `generate_initial_guess` override
# (the trajectory game). The QP has no initial guess, so it is cold regardless.
const USE_IG = get(ENV, "WARMSTART", "0") == "1"

const HEADER = "experiment,problem,num_primals,num_inequalities,horizon,num_samples,nthreads,solver,device,warm_start,total_time_s,num_solved,tol"

function ensure_header()
    if !isfile(CSV) || filesize(CSV) == 0
        open(CSV, "w") do io
            println(io, HEADER)
        end
    end
end

# Append one row. `r` is a benchmark_throughput sub-result (; total_time, num_solved) or
# `nothing` (skipped solver → no row written). `warm` records whether THIS solver/device
# actually used the initial guess (only the batched solve can; PATH/seq-IP are always cold).
function append_row!(; experiment, problem, num_primals="", num_inequalities="", horizon="",
                     num_samples, nthreads, solver, device, warm, r)
    isnothing(r) && return
    open(CSV, "a") do io
        println(io, join((experiment, problem, num_primals, num_inequalities, horizon,
                          num_samples, nthreads, solver, device, warm,
                          round(r.total_time; digits=4), r.num_solved, TOL), ","))
    end
    @info "  logged: $experiment/$problem/$solver/$device N=$num_samples warm=$warm  t=$(round(r.total_time;digits=3))s solved=$(r.num_solved)/$num_samples"
end

# The batched solve is warm only when USE_IG is on AND the type supplies a guess (game).
batched_warm(problem) = USE_IG && problem == "game"

const NT = Threads.nthreads()

# ---------------------------------------------------------------------------------------
# Stage: batched throughput (Table II) — PATH vs sequential IP vs batched-CPU, one B.
# ---------------------------------------------------------------------------------------
function stage_throughput(; qp_B=1024, game_B=1024, qp_kwargs=(; num_primals=32, num_inequalities=16),
                          game_kwargs=(; horizon=10))
    @info "STAGE throughput (QP B=$qp_B, game B=$game_B, $NT threads)"
    for (problem, btype, B, pkw) in (("qp", QP(), qp_B, qp_kwargs), ("game", GAME(), game_B, game_kwargs))
        @info "throughput: $problem (B=$B) ..."
        d = SolverBenchmarks.benchmark_throughput(btype; num_samples=B, problem_kwargs=pkw,
            tol=TOL, device=KernelAbstractions.CPU(),
            run_batched=true, run_sequential_ip=true, run_path=true, use_initial_guess=USE_IG)
        np = get(pkw, :num_primals, ""); ni = get(pkw, :num_inequalities, ""); hz = get(pkw, :horizon, "")
        common = (; experiment="throughput", problem, num_primals=np, num_inequalities=ni,
                  horizon=hz, num_samples=B, nthreads=NT)
        append_row!(; common..., solver="path",    device="na",  warm=false, r=d.path)
        append_row!(; common..., solver="ip_seq",  device="cpu", warm=false, r=d.ip)
        append_row!(; common..., solver="batched", device="cpu", warm=batched_warm(problem), r=d.batched)
    end
end

# ---------------------------------------------------------------------------------------
# Helper: build MCPs once (num_samples=1) then run a device point, reusing them.
# ---------------------------------------------------------------------------------------
function build_mcps(btype, pkw; want_path)
    w = SolverBenchmarks.benchmark_throughput(btype; num_samples=1, tol=TOL,
        problem_kwargs=pkw, run_sequential_ip=false, run_path=want_path,
        device=KernelAbstractions.CPU(), use_initial_guess=USE_IG)
    (; batched_mcp=w.batched_mcp, path_mcp=w.path_mcp)
end

function device_point(btype, pkw, n, device, mcps; run_path, warm=USE_IG)
    SolverBenchmarks.benchmark_throughput(btype; num_samples=n, tol=TOL, problem_kwargs=pkw,
        device, batched_mcp=mcps.batched_mcp, path_mcp=mcps.path_mcp,
        run_sequential_ip=false, run_path, use_initial_guess=warm)
end

# ---------------------------------------------------------------------------------------
# Stage: GPU vs CPU, sweep batch size at fixed problem size (both families).
# ---------------------------------------------------------------------------------------
function stage_gpu_scaling(; qp_sweep=[64,256,1024,4096,16384], game_sweep=[64,256,1024,4096],
                           qp_kwargs=(; num_primals=32, num_inequalities=16),
                           game_kwargs=(; horizon=10),
                           path_max=4096, cpu_max=16384)
    for (problem, btype, sweep, pkw) in (("qp", QP(), qp_sweep, qp_kwargs),
                                         ("game", GAME(), game_sweep, game_kwargs))
        @info "STAGE gpu_scaling: $problem (building MCPs once)..."
        mcps = build_mcps(btype, pkw; want_path=true)
        np = get(pkw, :num_primals, ""); ni = get(pkw, :num_inequalities, ""); hz = get(pkw, :horizon, "")
        for n in sweep
            common = (; experiment="gpu_scaling", problem, num_primals=np, num_inequalities=ni,
                      horizon=hz, num_samples=n, nthreads=NT)
            @info "gpu_scaling: $problem N=$n GPU ..."
            dg = device_point(btype, pkw, n, CUDA.CUDABackend(), mcps; run_path=(n<=path_max))
            append_row!(; common..., solver="batched", device="gpu", warm=batched_warm(problem), r=dg.batched)
            append_row!(; common..., solver="path",    device="na",  warm=false, r=dg.path)
            if n <= cpu_max
                @info "gpu_scaling: $problem N=$n CPU ($NT threads) ..."
                dc = device_point(btype, pkw, n, KernelAbstractions.CPU(), mcps; run_path=false)
                append_row!(; common..., solver="batched", device="cpu", warm=batched_warm(problem), r=dc.batched)
            end
            GC.gc(); CUDA.reclaim()
        end
        mcps = nothing; GC.gc(); CUDA.reclaim()
    end
end

# ---------------------------------------------------------------------------------------
# Stage: GPU vs CPU, sweep QP problem size at fixed batch size(s).
# ---------------------------------------------------------------------------------------
function stage_problem_size(; sizes=[(;num_primals=32,num_inequalities=16),
                                     (;num_primals=64,num_inequalities=32),
                                     (;num_primals=128,num_inequalities=64)],
                            samples=[1024,4096])
    for pkw in sizes
        @info "STAGE problem_size: $pkw (building MCP once)..."
        mcps = build_mcps(QP(), pkw; want_path=false)
        for n in samples
            common = (; experiment="problem_size", problem="qp",
                      num_primals=pkw.num_primals, num_inequalities=pkw.num_inequalities,
                      horizon="", num_samples=n, nthreads=NT)
            @info "problem_size: $pkw N=$n GPU ..."
            dg = device_point(QP(), pkw, n, CUDA.CUDABackend(), mcps; run_path=false)
            append_row!(; common..., solver="batched", device="gpu", warm=false, r=dg.batched)
            @info "problem_size: $pkw N=$n CPU ..."
            dc = device_point(QP(), pkw, n, KernelAbstractions.CPU(), mcps; run_path=false)
            append_row!(; common..., solver="batched", device="cpu", warm=false, r=dc.batched)
            GC.gc(); CUDA.reclaim()
        end
        mcps = nothing; GC.gc(); CUDA.reclaim()
    end
end

# ---------------------------------------------------------------------------------------
# Stage: GPU vs CPU, sweep trajectory-game horizon at fixed batch size (the crossover).
# ---------------------------------------------------------------------------------------
# Horizon sweep: for each horizon, build the (compile-heavy) MCP ONCE and solve it under
# both cold and warm start on both devices, so we get the cold (fair) and warm (realistic
# receding-horizon) crossover from a single compile per horizon.
function stage_horizon(; horizons=[10,20,30,40,50], N=1024)
    for hz in horizons
        pkw = (; horizon=hz)
        @info "STAGE horizon: T=$hz (building MCP once — compile grows with horizon)..."
        mcps = build_mcps(GAME(), pkw; want_path=false)
        common = (; experiment="horizon", problem="game", num_primals="", num_inequalities="",
                  horizon=hz, num_samples=N, nthreads=NT)
        for warm in (false, true)
            @info "horizon: T=$hz N=$N warm=$warm GPU ..."
            dg = device_point(GAME(), pkw, N, CUDA.CUDABackend(), mcps; run_path=false, warm)
            append_row!(; common..., solver="batched", device="gpu", warm, r=dg.batched)
            @info "horizon: T=$hz N=$N warm=$warm CPU ..."
            dc = device_point(GAME(), pkw, N, KernelAbstractions.CPU(), mcps; run_path=false, warm)
            append_row!(; common..., solver="batched", device="cpu", warm, r=dc.batched)
            GC.gc(); CUDA.reclaim()
        end
        mcps = nothing; GC.gc(); CUDA.reclaim()
    end
end

# ---------------------------------------------------------------------------------------
ensure_header()
CUDA.functional() || error("CUDA not functional")
stage = get(ENV, "STAGE", "throughput")
@info "=== running stage=$stage, threads=$NT, CSV=$CSV ==="
stage == "throughput"   && stage_throughput()
stage == "gpu_scaling"  && stage_gpu_scaling()
stage == "problem_size" && stage_problem_size()
stage == "horizon"      && stage_horizon()
if stage == "all"
    stage_throughput(); stage_gpu_scaling(); stage_problem_size(); stage_horizon()
end
@info "=== STAGE $stage DONE ==="
