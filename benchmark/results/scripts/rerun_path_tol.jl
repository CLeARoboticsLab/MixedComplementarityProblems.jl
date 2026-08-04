const REPO = get(ENV, "REPO_ROOT", normpath(joinpath(@__DIR__, "..", "..", "..")))
# Re-time ALL PATH evaluations at convergence_tolerance = 1e-4 (matching the IP/batched
# solver's `tol`) and surgically patch the PATH rows in every results CSV, leaving the
# batched / sequential-IP / GPU rows untouched. Previously PATH ran at its built-in default
# convergence_tolerance = 1e-6 while our solver used 1e-4 — an unfair 100x-tighter bar for
# PATH. One PATH measurement per (config, batch size, rep) feeds all CSVs for consistency.
# PATH appears only for the two base configs (qp p32i16, game T10); the problem-size /
# horizon studies are CPU-vs-GPU only. See results/README.md "PATH tolerance" note.
include(joinpath(REPO, "benchmark", "SolverBenchmarks.jl"))
using .SolverBenchmarks; const MCP = SolverBenchmarks
using ParametricMCPs: ParametricMCPs
using PATHSolver: PATHSolver
using KernelAbstractions: KernelAbstractions
import Random

const RESULTS = joinpath(REPO, "benchmark", "results")
const CTOL = 1e-4
const DEV = KernelAbstractions.CPU()
const QP = MCP.QuadraticProgramBenchmark; const GAME = MCP.TrajectoryGameBenchmark
psolved(s) = s.status == PATHSolver.MCP_Solved
psolve(mcp, θ) = ParametricMCPs.solve(mcp, θ; convergence_tolerance=CTOL, warn_on_convergence_failure=false)

const Bset = [64, 256, 1024, 4096]
const REPS = Dict(64=>3, 256=>3, 1024=>1, 4096=>1)   # match existing per_rep.csv rep counts
const Nmax = 4096
const NPI  = 1024                                     # per-instance sample count

# (key, size-label, btype, problem_kwargs, num_primals, num_inequalities, horizon) — the last
# three as the exact strings used in experiments.csv (empty where not applicable).
configs = (
    (:qp,   "p32i16", QP(),   (; num_primals=32, num_inequalities=16), "32", "16", ""),
    (:game, "T10",    GAME(), (; horizon=10),                          "",   "",   "10"),
)

results = Dict{Symbol,Any}()
for (key, sz, btype, pkw, np, ni, T) in configs
    @info "rerun_path: building $key $sz ..."
    w = MCP.benchmark_throughput(btype; num_samples=1, tol=CTOL, problem_kwargs=pkw, device=DEV,
            run_batched=false, run_sequential_ip=false, run_path=true)
    path_mcp = w.path_mcp
    rng = Random.MersenneTwister(1)
    θs = [MCP.generate_random_parameter(btype; rng, pkw...) for _ in 1:Nmax]
    psolve(path_mcp, θs[1])   # warmup

    # Per-instance timings (N = 1024) → per_instance.csv, per_instance_table1.csv.
    pinst = Tuple{Float64,Bool}[]
    for i in 1:NPI
        t = @elapsed s = psolve(path_mcp, θs[i])
        push!(pinst, (t, psolved(s)))
    end
    @info "  $key per-instance: solved=$(count(last, pinst))/$NPI"

    # Full-batch wall-clock totals per batch size → per_rep / throughput_klu / experiments.
    totals = Dict{Int,Vector{Tuple{Float64,Int}}}()
    for B in Bset
        totals[B] = Tuple{Float64,Int}[]
        for _ in 1:REPS[B]
            t = @elapsed n = count(θ -> psolved(psolve(path_mcp, θ)), @view θs[1:B])
            push!(totals[B], (t, n))
        end
        @info "  $key B=$B: total=$(round(totals[B][1][1];digits=3))s solved=$(totals[B][1][2])"
    end
    results[key] = (; prob=String(key), sz, np, ni, T, pinst, totals)
end

# ---- CSV surgery: drop old PATH rows, append freshly-timed ones. ----
r4(x) = round(x; digits=4); r6(x) = round(x; digits=6)
function patch!(fname, is_path_row, newrows)
    path = joinpath(RESULTS, fname)
    lines = readlines(path)
    kept = filter(l -> !is_path_row(split(l, ',')), @view lines[2:end])
    open(path, "w") do io
        println(io, lines[1])
        foreach(l -> println(io, l), kept)
        foreach(l -> println(io, l), newrows)
    end
    @info "patched $fname: dropped $(length(lines)-1-length(kept)) PATH rows, added $(length(newrows))"
end

# per_instance.csv / per_instance_table1.csv  (problem,size,solver,instance,time_s,solved,tol)
pi_rows = String[]
for (_, r) in results, (i, (t, sol)) in enumerate(r.pinst)
    push!(pi_rows, join((r.prob, r.sz, "path", i, r6(t), sol, CTOL), ","))
end
patch!("per_instance.csv",        f -> f[3] == "path", pi_rows)
patch!("per_instance_table1.csv", f -> f[3] == "path", pi_rows)

# per_rep.csv  (problem,size,solver,device,warm_start,rep,total_time_s,num_solved,num_samples,tol)
pr_rows = String[]
for (_, r) in results, B in Bset, (rep, (t, n)) in enumerate(r.totals[B])
    push!(pr_rows, join((r.prob, r.sz, "path", "na", false, rep, r6(t), n, B, CTOL), ","))
end
patch!("per_rep.csv", f -> f[3] == "path", pr_rows)

# throughput_klu.csv  (problem,solver,warm_start,total_time_s,num_solved,num_samples,tol) — B=1024
tk_rows = String[]
for (_, r) in results
    t, n = r.totals[1024][1]
    push!(tk_rows, join((r.prob, "path", false, r4(t), n, 1024, CTOL), ","))
end
patch!("throughput_klu.csv", f -> f[2] == "path", tk_rows)

# experiments.csv  (experiment,problem,num_primals,num_inequalities,horizon,num_samples,
#                   nthreads,solver,device,warm_start,total_time_s,num_solved,tol)
ex_rows = String[]
for (_, r) in results
    for B in Bset
        t, n = r.totals[B][1]
        push!(ex_rows, join(("gpu_scaling", r.prob, r.np, r.ni, r.T, B, 32, "path", "na", false, r4(t), n, CTOL), ","))
    end
    t, n = r.totals[1024][1]
    push!(ex_rows, join(("throughput", r.prob, r.np, r.ni, r.T, 1024, 32, "path", "na", false, r4(t), n, CTOL), ","))
end
patch!("experiments.csv", f -> f[8] == "path", ex_rows)

println("=== RERUN_PATH_TOL DONE ===")
