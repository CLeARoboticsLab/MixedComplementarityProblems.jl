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

Works for both benchmark types: the `QuadraticProgramBenchmark` (callable `K`) and the
`TrajectoryGameBenchmark` (symbolic `K` with internal η — solved with the `:internal`
regularization scheme). Run with several threads to see the batched speedup, e.g.
`julia -t 4`. On heterogeneous CPUs (Apple silicon) prefer `-t <#performance-cores>`.

NOTE: `BatchedInteriorPoint` requires kernel evaluators, built with `SerialForm` codegen,
whose compile time grows with the size of the symbolic KKT system (D2 in
`docs/gpu_kkt_design.md`). For the QP keep `num_primals` modest (its symbolic Hessian
block is dense); for the trajectory game keep the `horizon` modest — large horizons make
kernel-evaluator compilation slow.
"""
function benchmark_throughput(
    benchmark_type = QuadraticProgramBenchmark();
    num_samples = 256,
    problem_kwargs = (; num_primals = 32, num_inequalities = 16),
    batched_mcp = nothing,
    path_mcp = nothing,
    tol = 1e-4,
    device = KernelAbstractions.CPU(),
    run_batched = true,
    run_sequential_ip = true,
    run_path = true,
    use_initial_guess = true,
)
    @info "Generating random problems..."
    problem = generate_test_problem(benchmark_type; problem_kwargs...)

    rng = Random.MersenneTwister(1)
    θs = map(1:num_samples) do _
        generate_random_parameter(benchmark_type; rng, problem_kwargs...)
    end
    Θ_host = reduce(hcat, θs)              # (nθ × N) — column b is instance b
    Θ = Adapt.adapt(device, Θ_host)        # moved to `device` once; reused for every solve
    parameter_dimension = size(Θ, 1)

    # Solve with the additive `:identity` scheme (full ∇F + η·I). Even problems that carry
    # an internal η (trajectory games) want this: their KKT systems need every row
    # regularized, which `:identity` provides (the batched assembler augments the pattern
    # to the full diagonal); the `:internal` primal-only regularization is insufficient
    # and diverges on them. `internally_regularized` only affects how the MCPs are built
    # (the game's K carries η, which PATH strips).
    internally_regularized =
        hasproperty(problem, :η_symbolic) && !isnothing(problem.η_symbolic)
    regularize_linear_solve = :identity

    # Batched-capable IP MCP (kernel evaluators are required by BatchedInteriorPoint).
    # Needed for BOTH `run_batched` and `run_sequential_ip` (the sequential comparison
    # reuses this same mcp, not a separate one). Callable-`K` problems (QP) use the
    # function constructor; symbolic problems (games, via `build_mcp_components`) use the
    # symbolic constructor with their internal η.
    if run_batched || run_sequential_ip
        @info "Generating batched IP MCP (with kernel evaluators)..."
        batched_mcp = @something(
            batched_mcp,
            if hasproperty(problem, :K)
                MixedComplementarityProblems.PrimalDualMCP(
                    problem.K,
                    problem.lower_bounds,
                    problem.upper_bounds;
                    parameter_dimension,
                    compute_kernel_evaluators = true,
                )
            else
                MixedComplementarityProblems.PrimalDualMCP(
                    problem.K_symbolic,
                    problem.z_symbolic,
                    problem.θ_symbolic,
                    problem.lower_bounds,
                    problem.upper_bounds;
                    η_symbolic = internally_regularized ? problem.η_symbolic : nothing,
                    compute_kernel_evaluators = true,
                )
            end
        )
    end

    # PATH solves the unregularized system, so strip any internal η from the symbolic K.
    if run_path
        @info "Generating PATH MCP..."
        path_mcp = @something(
            path_mcp,
            if hasproperty(problem, :K)
                ParametricMCPs.ParametricMCP(
                    (z, θ) -> problem.K(z; θ),
                    problem.lower_bounds,
                    problem.upper_bounds,
                    parameter_dimension,
                )
            else
                K_symbolic =
                    internally_regularized ?
                    Vector{Symbolics.Num}(
                        Symbolics.substitute.(
                            problem.K_symbolic,
                            Ref(Dict(problem.η_symbolic => 0.0)),
                        ),
                    ) : problem.K_symbolic
                ParametricMCPs.ParametricMCP(
                    K_symbolic,
                    problem.z_symbolic,
                    problem.θ_symbolic,
                    problem.lower_bounds,
                    problem.upper_bounds,
                )
            end
        )
    end

    # Per-`benchmark_type` initial guess for the batched solve (e.g. the trajectory
    # game's zero-input rollout — see `generate_initial_guess` in
    # `trajectory_game_benchmark.jl`); `nothing` (cold zero start) for benchmark types
    # that don't override it, matching prior behavior. Set `use_initial_guess = false` to
    # force the batched solve to cold-start too, matching the (cold) sequential IP and
    # PATH baselines for an apples-to-apples comparison.
    X₀ = (run_batched && use_initial_guess) ?
        generate_initial_guess(benchmark_type, batched_mcp, Θ, device; problem_kwargs...) :
        nothing

    # Warm up (compile) only the solvers being run.
    @info "Warming up solvers..."
    run_batched && MixedComplementarityProblems.solve(
        MixedComplementarityProblems.BatchedInteriorPoint(),
        batched_mcp,
        Θ[:, 1:1];
        tol,
        regularize_linear_solve,
        device,
        X₀ = isnothing(X₀) ? nothing : X₀[:, 1:1],
    )
    run_sequential_ip && MixedComplementarityProblems.solve(
        MixedComplementarityProblems.InteriorPoint(),
        batched_mcp,
        θs[1];
        tol,
        regularize_linear_solve,
    )
    run_path && ParametricMCPs.solve(
        path_mcp,
        θs[1];
        convergence_tolerance = tol,
        warn_on_convergence_failure = false,
    )

    # --- Batched IP: one call over the whole batch, on `device`. ---
    batched = if run_batched
        @info "Solving batch with BatchedInteriorPoint (device = $(typeof(device)), $(Threads.nthreads()) threads)..."
        local batched_sol
        t_batched = @elapsed batched_sol = MixedComplementarityProblems.solve(
            MixedComplementarityProblems.BatchedInteriorPoint(),
            batched_mcp,
            Θ;
            tol,
            regularize_linear_solve,
            device,
            X₀,
        )
        (; total_time = t_batched, num_solved = count(==(:solved), batched_sol.status))
    end

    # --- Unbatched IP: sequential, single instance at a time. ---
    ip = if run_sequential_ip
        @info "Solving sequentially with InteriorPoint..."
        t_ip = @elapsed n_ip = count(θs) do θ
            MixedComplementarityProblems.solve(
                MixedComplementarityProblems.InteriorPoint(),
                batched_mcp,
                θ;
                tol,
                regularize_linear_solve,
            ).status == :solved
        end
        (; total_time = t_ip, num_solved = n_ip)
    end

    # --- PATH: sequential, single-threaded. ---
    path = if run_path
        @info "Solving sequentially with PATH..."
        t_path = @elapsed n_path = count(θs) do θ
            ParametricMCPs.solve(
                path_mcp,
                θ;
                convergence_tolerance = tol,
                warn_on_convergence_failure = false,
            ).status == PATHSolver.MCP_Solved
        end
        (; total_time = t_path, num_solved = n_path)
    end

    (;
        batched_mcp = run_batched || run_sequential_ip ? batched_mcp : nothing,
        path_mcp,
        num_samples,
        nthreads = Threads.nthreads(),
        device,
        tol,
        batched,
        ip,
        path,
    )
end

"Print a throughput summary from `benchmark_throughput` data. Solvers that weren't run
(their field is `nothing`) are skipped."
function throughput_summary(data)
    (; num_samples, nthreads, device) = data
    rate(t) = num_samples / t                      # problems / second
    row(name, d) = @info string(
        rpad(name, 26),
        "total ", round(d.total_time; digits = 3), " s   ",
        "throughput ", round(rate(d.total_time); digits = 1), " prob/s   ",
        "solved ", d.num_solved, "/", num_samples,
    )

    @info "Throughput over $num_samples problems (BatchedInteriorPoint device = $(typeof(device)), $nthreads thread(s)), tol=$(data.tol):"
    !isnothing(data.path) && row("PATH (1 thread)", data.path)
    !isnothing(data.ip) && row("InteriorPoint (seq)", data.ip)
    !isnothing(data.batched) && row("BatchedInteriorPoint", data.batched)

    batched_vs_path =
        !isnothing(data.batched) && !isnothing(data.path) ?
        data.path.total_time / data.batched.total_time : nothing
    batched_vs_ip =
        !isnothing(data.batched) && !isnothing(data.ip) ?
        data.ip.total_time / data.batched.total_time : nothing

    messages = String[]
    !isnothing(batched_vs_path) && push!(messages, "$(round(batched_vs_path; digits = 2))× vs PATH")
    !isnothing(batched_vs_ip) &&
        push!(messages, "$(round(batched_vs_ip; digits = 2))× vs sequential InteriorPoint")
    !isempty(messages) && @info string("Batched speedup: ", join(messages, ",  "))

    (; batched_vs_path, batched_vs_ip)
end
