const REPO = get(ENV, "REPO_ROOT", normpath(joinpath(@__DIR__, "..", "..", "..")))
include(joinpath(REPO, "benchmark", "thread_scaling_benchmark.jl"))
const OUT = joinpath(REPO, "benchmark", "results", "thread_scaling.csv")
open(io->println(io,"problem,nthreads,total_time_s,num_solved,num_samples,throughput_per_s,tol"), OUT, "w")
row(a...) = open(io->println(io, join(a, ",")), OUT, "a")

for (problem, btype, ns, pk) in (("qp", :qp, 1024, nothing),
                                 ("game", :trajectory_game, 1024, (; horizon=10)))
    @info "thread scaling: $problem ..."
    data = isnothing(pk) ? thread_scaling_benchmark(btype; num_samples=ns) :
                           thread_scaling_benchmark(btype; num_samples=ns, problem_kwargs=pk)
    for e in data.batched_by_threads
        thr = e.num_samples / e.batched.total_time
        row(problem, e.nthreads, round(e.batched.total_time;digits=4), e.batched.num_solved, e.num_samples, round(thr;digits=1), e.tol)
        @info "  $problem t=$(e.nthreads): $(round(e.batched.total_time;digits=3))s  $(round(thr;digits=1)) prob/s  solved $(e.batched.num_solved)/$(e.num_samples)"
    end
end
println("=== THREAD_SCALING DONE ===")
