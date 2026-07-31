const REPO = get(ENV, "REPO_ROOT", normpath(joinpath(@__DIR__, "..", "..", "..")))
# Table I (single-instance): reliability + per-solve time for PATH vs unbatched IP with
# UMFPACK (current default) vs KLU (candidate). Per-instance times → per_instance_table1.csv.
include(joinpath(REPO, "benchmark", "SolverBenchmarks.jl"))
using .SolverBenchmarks; const MCP = SolverBenchmarks
using MixedComplementarityProblems: MixedComplementarityProblems; const M = MixedComplementarityProblems
using ParametricMCPs: ParametricMCPs
using PATHSolver: PATHSolver
using KernelAbstractions: KernelAbstractions
using Statistics: mean, median, std
import Random

const RESULTS = joinpath(REPO, "benchmark", "results")
const CSV = joinpath(RESULTS, "per_instance_table1.csv")
const TOL = 1e-4
const UMF = M.UMFPACKFactorization()
const KLU = M.KLUFactorization()
const QP = MCP.QuadraticProgramBenchmark; const GAME = MCP.TrajectoryGameBenchmark
open(io->println(io,"problem,size,solver,instance,time_s,solved,tol"), CSV, "w")
row(a...) = open(io->println(io, join(a, ",")), CSV, "a")

# Build MCPs + θ's at top level (world-age safety for runtime-eval'd evaluators).
N = 1024
configs = (("qp", QP(), (;num_primals=32,num_inequalities=16), "p32i16"),
           ("game", GAME(), (;horizon=10), "T10"))
built = map(configs) do (problem, btype, pkw, sz)
    w = MCP.benchmark_throughput(btype; num_samples=1, tol=TOL, problem_kwargs=pkw, device=KernelAbstractions.CPU())
    rng = Random.MersenneTwister(1)
    θs = [MCP.generate_random_parameter(btype; rng, pkw...) for _ in 1:N]
    (; problem, sz, mcp=w.batched_mcp, path_mcp=w.path_mcp, θs)
end

ip(mcp, θ, alg) = M.solve(M.InteriorPoint(), mcp, θ; tol=TOL, regularize_linear_solve=:identity, linear_solve_algorithm=alg)

for b in built
    @info "table1: $(b.problem) ..."
    # warmups
    ParametricMCPs.solve(b.path_mcp, b.θs[1]; convergence_tolerance=TOL, warn_on_convergence_failure=false)
    ip(b.mcp, b.θs[1], UMF); ip(b.mcp, b.θs[1], KLU)
    for (i,θ) in enumerate(b.θs)
        tp = @elapsed sp = ParametricMCPs.solve(b.path_mcp, θ; convergence_tolerance=TOL, warn_on_convergence_failure=false)
        row(b.problem, b.sz, "path", i, round(tp;digits=6), sp.status==PATHSolver.MCP_Solved, TOL)
        tu = @elapsed su = ip(b.mcp, θ, UMF)
        row(b.problem, b.sz, "ip_umfpack", i, round(tu;digits=6), su.status==:solved, TOL)
        tk = @elapsed sk = ip(b.mcp, θ, KLU)
        row(b.problem, b.sz, "ip_klu", i, round(tk;digits=6), sk.status==:solved, TOL)
    end
end

# Summary
using DelimitedFiles
println("\n=== Table I summary (N=$N per problem) ===")
data = readdlm(CSV, ','; skipstart=1)
for problem in ("qp","game"), solver in ("path","ip_umfpack","ip_klu")
    mask = (data[:,1].==problem) .& (data[:,3].==solver)
    ts = Float64.(data[mask,5]); solved = sum(data[mask,6].==true)
    println("  $(rpad(problem,5)) $(rpad(solver,11)): solved=$solved/$(length(ts))  mean=$(round(1000mean(ts);digits=3))ms  median=$(round(1000median(ts);digits=3))ms  std=$(round(1000std(ts);digits=3))ms")
end
println("=== TABLE1 DONE ===")
