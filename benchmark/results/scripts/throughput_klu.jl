const REPO = get(ENV, "REPO_ROOT", normpath(joinpath(@__DIR__, "..", "..", "..")))
# Clean Table 2 with the KLU default now active: whole-loop timing of PATH, sequential IP
# (KLU), and batched CPU, for QP + game at B=1024 (cold; game also warm).
include(joinpath(REPO, "benchmark", "SolverBenchmarks.jl"))
using .SolverBenchmarks; const MCP = SolverBenchmarks
const OUT = joinpath(REPO, "benchmark", "results", "throughput_klu.csv")
open(io->println(io,"problem,solver,warm_start,total_time_s,num_solved,num_samples,tol"), OUT, "w")
row(a...) = open(io->println(io, join(a, ",")), OUT, "a")
B=1024
for (problem, btype, pkw) in (("qp", MCP.QuadraticProgramBenchmark(), (;num_primals=32,num_inequalities=16)),
                              ("game", MCP.TrajectoryGameBenchmark(), (;horizon=10)))
    # cold: PATH + sequential IP (now KLU default) + batched CPU
    d = MCP.benchmark_throughput(btype; num_samples=B, problem_kwargs=pkw, tol=1e-4,
            run_batched=true, run_sequential_ip=true, run_path=true, use_initial_guess=false)
    row(problem,"path",false,round(d.path.total_time;digits=4),d.path.num_solved,B,1e-4)
    row(problem,"ip_klu",false,round(d.ip.total_time;digits=4),d.ip.num_solved,B,1e-4)
    row(problem,"batched_cpu",false,round(d.batched.total_time;digits=4),d.batched.num_solved,B,1e-4)
    @info "$problem cold: PATH=$(d.path.total_time)s IP-KLU=$(d.ip.total_time)s batched=$(d.batched.total_time)s"
    if problem == "game"   # warm batched (realistic)
        dw = MCP.benchmark_throughput(btype; num_samples=B, problem_kwargs=pkw, tol=1e-4,
                run_batched=true, run_sequential_ip=false, run_path=false, use_initial_guess=true,
                batched_mcp=d.batched_mcp)
        row(problem,"batched_cpu",true,round(dw.batched.total_time;digits=4),dw.batched.num_solved,B,1e-4)
        @info "$problem warm batched=$(dw.batched.total_time)s"
    end
end
println("=== THROUGHPUT_KLU DONE ===")
