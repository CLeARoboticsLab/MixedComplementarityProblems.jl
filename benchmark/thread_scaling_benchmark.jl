""" CPU thread-scaling study for `BatchedInteriorPoint`.

Julia's thread count is fixed at process startup (`julia -t N`), so measuring throughput
across several thread counts means running several separate Julia processes. Rather than
shelling out, this uses `Distributed.jl` from a single master process: one lightweight
worker per thread count, each started with its own `-t N` via `exeflags`, each running
ONLY the batched solve (`run_sequential_ip = run_path = false` — PATH and the sequential
`InteriorPoint` are thread-count-invariant, so timing them again on every worker would
just be wasted work). PATH/`InteriorPoint` are measured once, separately, as a reference.

Every worker solves the IDENTICAL batch of problems (`benchmark_throughput` reseeds
`Random.MersenneTwister(1)` internally), so batched results are directly comparable
across thread counts.

Usage (from `benchmark/`, `julia --project=.`):

```julia
julia> include("thread_scaling_benchmark.jl")
julia> data = thread_scaling_benchmark(:qp; num_samples = 1024);
julia> thread_scaling_summary(data)

julia> data = thread_scaling_benchmark(:trajectory_game; num_samples = 128, problem_kwargs = (; horizon = 3));
julia> thread_scaling_summary(data)
```

Default `thread_counts` are capped to `Sys.CPU_THREADS` (unlike a GPU, oversubscribing
CPU threads past the physical/logical core count doesn't make sense — throughput
plateaus or regresses rather than continuing to scale). Passing `thread_counts`
explicitly with values above `Sys.CPU_THREADS` is allowed but triggers a warning, since
that's almost always a mistake rather than an intentional oversubscription test.
"""

using Distributed: Distributed

"Default thread counts to compare, capped at what this machine actually has."
function default_thread_counts()
    filter(<=(Sys.CPU_THREADS), [1, 2, 4, 8, 16, 32])
end

"Start one worker process with `n` threads, in the same project environment as the caller."
function _spawn_worker(n, project)
    only(Distributed.addprocs(1; exeflags = ["-t", string(n), "--project=$project"]))
end

# NOTE (world age): `remotecall_fetch(pid) do ... end` ships the closure's CODE to the
# worker (unlike a named top-level function, which the worker wouldn't have — verified
# empirically). But `include(path)` inside that closure defines `SolverBenchmarks` (and
# its `QuadraticProgramBenchmark`/`TrajectoryGameBenchmark` structs) via a runtime `eval`,
# so constructing those types in the SAME closure call needs `Base.invokelatest` —
# otherwise Julia's world-age check rejects the construction as "too new". Same class of
# bug as the package's own eval'd kernel evaluators; see the note on
# `solve(::BatchedInteriorPoint, ...)` in `src/batched_solver.jl`.

"""
    thread_scaling_benchmark(benchmark_type = :qp; thread_counts = default_thread_counts(), kwargs...)

`benchmark_type` is `:qp` (`QuadraticProgramBenchmark`) or `:trajectory_game`
(`TrajectoryGameBenchmark`). `kwargs` (`num_samples`, `problem_kwargs`, `tol`) are
forwarded to `SolverBenchmarks.benchmark_throughput`.

`problem_kwargs` defaults to `nothing`, meaning "omit it" — NOT `(;)`. The two are NOT
equivalent: `benchmark_throughput`'s own default is the QP-sized `(; num_primals = 32,
num_inequalities = 16)`, but explicitly passing `(;)` overrides that with nothing,
falling through to `generate_test_problem`'s much larger internal default
(`num_primals = 100`). That matters because building kernel evaluators for a DENSE
100×100 Jacobian (the QP's `M` block) is already known to be slow with `SerialForm`
codegen — see the module docstring's NOTE and D2 in `docs/gpu_kkt_design.md` — this is
unrelated to eval-vs-RuntimeGeneratedFunction (verified: reverting to the old
RuntimeGeneratedFunction codegen is equally slow at `num_primals = 100`). Pass
`problem_kwargs` explicitly for `:trajectory_game` too, since its own default
(`horizon = 10`) isn't tuned for kernel-evaluator compile time either — its sparser,
block-structured Jacobian is comparatively cheap, but a smaller `horizon` (e.g. `3`)
keeps it fast regardless.

Returns `(; reference, batched_by_threads)`: `reference` is a `benchmark_throughput`
result (PATH + sequential `InteriorPoint`, measured once, `run_batched = false`);
`batched_by_threads` is a `Vector` of `(; nthreads, num_samples, tol, batched)`, one per
thread count, in the order of `thread_counts`.
"""
function thread_scaling_benchmark(
    benchmark_type::Symbol = :qp;
    thread_counts = default_thread_counts(),
    num_samples = 1024,
    problem_kwargs = nothing,
    tol = 1e-4,
)
    benchmark_type in (:qp, :trajectory_game) || error(
        "benchmark_type must be :qp or :trajectory_game, got $benchmark_type",
    )
    if any(>(Sys.CPU_THREADS), thread_counts)
        @warn "Requested thread_counts include values above Sys.CPU_THREADS = $(Sys.CPU_THREADS) — CPU oversubscription typically plateaus or regresses throughput rather than scaling it."
    end

    project = Base.active_project()
    solver_benchmarks_path = joinpath(@__DIR__, "SolverBenchmarks.jl")

    @info "Measuring PATH + sequential InteriorPoint once (thread-count-invariant reference)..."
    reference_pid = _spawn_worker(1, project)
    reference = try
        Distributed.remotecall_fetch(
            reference_pid,
            benchmark_type,
            solver_benchmarks_path,
            num_samples,
            problem_kwargs,
            tol,
        ) do bt, path, ns, pk, t
            SolverBenchmarks = include(path)
            Base.invokelatest() do
                problem_type = bt === :qp ? SolverBenchmarks.QuadraticProgramBenchmark() :
                    SolverBenchmarks.TrajectoryGameBenchmark()
                extra = isnothing(pk) ? (;) : (; problem_kwargs = pk)
                data = SolverBenchmarks.benchmark_throughput(
                    problem_type;
                    num_samples = ns,
                    tol = t,
                    run_batched = false,
                    extra...,
                )
                # Only the plain-data timing fields cross back to the master — `data`
                # also carries `batched_mcp`/`path_mcp` (MixedComplementarityProblems
                # structs), whose TYPE only exists in the worker's loaded modules; the
                # master never `using`s that package, so deserializing them back would
                # fail with a `KeyError` looking up the defining module.
                (; ip = data.ip, path = data.path)
            end
        end
    finally
        Distributed.rmprocs(reference_pid)
    end

    batched_by_threads = map(thread_counts) do n
        @info "Spawning worker with $n thread(s)..."
        pid = _spawn_worker(n, project)
        try
            Distributed.remotecall_fetch(
                pid,
                benchmark_type,
                solver_benchmarks_path,
                num_samples,
                problem_kwargs,
                tol,
            ) do bt, path, ns, pk, t
                SolverBenchmarks = include(path)
                Base.invokelatest() do
                    problem_type = bt === :qp ? SolverBenchmarks.QuadraticProgramBenchmark() :
                        SolverBenchmarks.TrajectoryGameBenchmark()
                    extra = isnothing(pk) ? (;) : (; problem_kwargs = pk)
                    data = SolverBenchmarks.benchmark_throughput(
                        problem_type;
                        num_samples = ns,
                        tol = t,
                        run_sequential_ip = false,
                        run_path = false,
                        extra...,
                    )
                    (;
                        nthreads = Threads.nthreads(),
                        num_samples = data.num_samples,
                        tol = data.tol,
                        batched = data.batched,
                    )
                end
            end
        finally
            Distributed.rmprocs(pid)
        end
    end

    (; reference, batched_by_threads)
end

"Print a thread-scaling summary table from `thread_scaling_benchmark` data."
function thread_scaling_summary(data)
    (; reference, batched_by_threads) = data
    rate(t) = batched_by_threads[1].num_samples / t
    @info "BatchedInteriorPoint CPU thread scaling over $(batched_by_threads[1].num_samples) problems, tol=$(batched_by_threads[1].tol):"
    baseline = first(batched_by_threads).batched.total_time
    for d in batched_by_threads
        @info string(
            rpad("$(d.nthreads) thread(s)", 14),
            "batched ", rpad(string(round(d.batched.total_time; digits = 3), " s"), 10),
            "throughput ", rpad(string(round(rate(d.batched.total_time); digits = 1), " prob/s"), 16),
            "solved ", d.batched.num_solved, "/", d.num_samples, "   ",
            "speedup vs 1 thread: ", round(baseline / d.batched.total_time; digits = 2), "×",
        )
    end
    @info string(
        "(reference, thread-count-invariant) PATH: ",
        round(reference.path.total_time; digits = 3), " s,   ",
        "sequential InteriorPoint: ", round(reference.ip.total_time; digits = 3), " s",
    )
    data
end
