""" Throughput benchmark: solve a whole *batch* of problems that share one MCP
structure (differing only in their parameters θ).

This is the regime the batched `BatchedInteriorPoint` solver targets — many small/medium
instances at once — and it is where CPU multithreading pays off: the batched solver
factorizes/solves all instances in parallel across threads, whereas PATH (and the
unbatched `InteriorPoint`) must process them one at a time on a single thread.

The comparison reports total wall-clock to clear `num_samples` problems:
  • PATH                      — sequential, single-threaded (baseline)
  • InteriorPoint (unbatched) — sequential
  • BatchedInteriorPoint      — one threaded call over the (nθ × N) parameter matrix

Run with several threads to see the batched speedup, e.g. `julia -t 4`. On
heterogeneous CPUs (Apple silicon) prefer `-t <#performance-cores>`.

NOTE: `BatchedInteriorPoint` requires kernel evaluators, built with `SerialForm`
codegen. For the QP these scale with the (dense) symbolic Hessian block, so keep
`num_primals` modest — this targets the "many small" regime. Large `num_primals`
(≳100) can make kernel-evaluator compilation very slow (see D2 in
`docs/gpu_kkt_design.md`).
"""
function benchmark_throughput(
    benchmark_type = QuadraticProgramBenchmark();
    num_samples = 256,
    problem_kwargs = (; num_primals = 32, num_inequalities = 16),
    batched_mcp = nothing,
    path_mcp = nothing,
    tol = 1e-4,
)
    @info "Generating random problems..."
    problem = generate_test_problem(benchmark_type; problem_kwargs...)
    hasproperty(problem, :K) || error(
        "benchmark_throughput currently supports callable-`K` problems " *
        "(e.g. QuadraticProgramBenchmark).",
    )

    rng = Random.MersenneTwister(1)
    θs = map(1:num_samples) do _
        generate_random_parameter(benchmark_type; rng, problem_kwargs...)
    end
    Θ = reduce(hcat, θs)                  # (nθ × N) — column b is instance b
    parameter_dimension = size(Θ, 1)

    # Batched-capable IP MCP (kernel evaluators are required by BatchedInteriorPoint).
    @info "Generating batched IP MCP (with kernel evaluators)..."
    batched_mcp = @something(
        batched_mcp,
        MixedComplementarityProblems.PrimalDualMCP(
            problem.K,
            problem.lower_bounds,
            problem.upper_bounds;
            parameter_dimension,
            compute_kernel_evaluators = true,
        )
    )

    @info "Generating PATH MCP..."
    path_mcp = @something(
        path_mcp,
        ParametricMCPs.ParametricMCP(
            (z, θ) -> problem.K(z; θ),
            problem.lower_bounds,
            problem.upper_bounds,
            parameter_dimension,
        )
    )

    # Warm up (compile) each solver before timing.
    @info "Warming up solvers..."
    MixedComplementarityProblems.solve(
        MixedComplementarityProblems.BatchedInteriorPoint(),
        batched_mcp,
        Θ[:, 1:1];
        tol,
    )
    MixedComplementarityProblems.solve(
        MixedComplementarityProblems.InteriorPoint(),
        batched_mcp,
        θs[1];
        tol,
    )
    ParametricMCPs.solve(path_mcp, θs[1]; warn_on_convergence_failure = false)

    # --- Batched IP: one threaded call over the whole batch. ---
    @info "Solving batch with BatchedInteriorPoint ($(Threads.nthreads()) threads)..."
    local batched_sol
    t_batched = @elapsed batched_sol = MixedComplementarityProblems.solve(
        MixedComplementarityProblems.BatchedInteriorPoint(),
        batched_mcp,
        Θ;
        tol,
    )
    n_batched = count(==(:solved), batched_sol.status)

    # --- Unbatched IP: sequential, single instance at a time. ---
    @info "Solving sequentially with InteriorPoint..."
    t_ip = @elapsed n_ip = count(θs) do θ
        MixedComplementarityProblems.solve(
            MixedComplementarityProblems.InteriorPoint(),
            batched_mcp,
            θ;
            tol,
        ).status == :solved
    end

    # --- PATH: sequential, single-threaded. ---
    @info "Solving sequentially with PATH..."
    t_path = @elapsed n_path = count(θs) do θ
        ParametricMCPs.solve(
            path_mcp,
            θ;
            warn_on_convergence_failure = false,
        ).status == PATHSolver.MCP_Solved
    end

    (;
        batched_mcp,
        path_mcp,
        num_samples,
        nthreads = Threads.nthreads(),
        tol,
        batched = (; total_time = t_batched, num_solved = n_batched),
        ip = (; total_time = t_ip, num_solved = n_ip),
        path = (; total_time = t_path, num_solved = n_path),
    )
end

"Print a throughput summary from `benchmark_throughput` data."
function throughput_summary(data)
    (; num_samples, nthreads) = data
    rate(t) = num_samples / t                      # problems / second
    row(name, d) = @info string(
        rpad(name, 26),
        "total ", round(d.total_time; digits = 3), " s   ",
        "throughput ", round(rate(d.total_time); digits = 1), " prob/s   ",
        "solved ", d.num_solved, "/", num_samples,
    )

    @info "Throughput over $num_samples problems on $nthreads thread(s), tol=$(data.tol):"
    row("PATH (1 thread)", data.path)
    row("InteriorPoint (seq)", data.ip)
    row("BatchedInteriorPoint", data.batched)
    @info string(
        "Batched speedup: ",
        round(data.path.total_time / data.batched.total_time; digits = 2),
        "× vs PATH,  ",
        round(data.ip.total_time / data.batched.total_time; digits = 2),
        "× vs sequential InteriorPoint",
    )

    (;
        batched_vs_path = data.path.total_time / data.batched.total_time,
        batched_vs_ip = data.ip.total_time / data.batched.total_time,
    )
end
