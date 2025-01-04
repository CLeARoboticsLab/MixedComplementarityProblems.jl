"Benchmark interior point solver against PATH on a bunch of random test problems."
function benchmark(
    benchmark_type;
    num_samples = 100,
    problem_kwargs = (;),
    ip_mcp = nothing,
    path_mcp = nothing,
    ip_kwargs = (; tol = 1e-6),
)
    # Generate problem and random parameters.
    @info "Generating random problems..."
    problem = generate_test_problem(benchmark_type; problem_kwargs...)

    rng = Random.MersenneTwister(1)
    θs = map(1:num_samples) do _
        generate_random_parameter(benchmark_type; rng, problem_kwargs...)
    end

    # Generate corresponding MCPs.
    @info "Generating IP MCP..."
    parameter_dimension = length(first(θs))
    ip_mcp = if !isnothing(ip_mcp)
        ip_mcp
    elseif hasproperty(problem, :K)
        # Generated a callable problem.
        MixedComplementarityProblems.PrimalDualMCP(
            problem.K,
            problem.lower_bounds,
            problem.upper_bounds;
            parameter_dimension,
        )
    else
        # Generated a symbolic problem.
        MixedComplementarityProblems.PrimalDualMCP(
            problem.K_symbolic,
            problem.z_symbolic,
            problem.θ_symbolic,
            problem.lower_bounds,
            problem.upper_bounds,
        )
    end

    @info "Generating PATH MCP..."
    path_mcp = if !isnothing(path_mcp)
        path_mcp
    elseif hasproperty(problem, :K)
        # Generated a callable problem.
        ParametricMCPs.ParametricMCP(
            problem.K,
            problem.lower_bounds,
            problem.upper_bounds,
            parameter_dimension,
        )
    else
        # Generated a symbolic problem.
        ParametricMCPs.ParametricMCP(
            problem.K_symbolic,
            problem.z_symbolic,
            problem.θ_symbolic,
            problem.lower_bounds,
            problem.upper_bounds
        )
    end

    # Warm up the solvers.
    @info "Warming up IP solver..."
    MixedComplementarityProblems.solve(
        MixedComplementarityProblems.InteriorPoint(),
        ip_mcp,
        first(θs);
        ip_kwargs...,
    )

    @info "Warming up PATH solver..."
    ParametricMCPs.solve(path_mcp, first(θs); warn_on_convergence_failure = false)

    # Solve and time.
    ip_data = @showprogress desc = "Solving IP MCPs..." map(θs) do θ
        elapsed_time = @elapsed sol = MixedComplementarityProblems.solve(
            MixedComplementarityProblems.InteriorPoint(),
            ip_mcp,
            θ;
            ip_kwargs...,
        )

        (; elapsed_time, success = sol.status == :solved)
    end

    path_data = @showprogress desc = "Solving PATH MCPs..." map(θs) do θ
        # Solve and time.
        elapsed_time = @elapsed sol =
            ParametricMCPs.solve(path_mcp, θ; warn_on_convergence_failure = false)

        (; elapsed_time, success = sol.status == PATHSolver.MCP_Solved)
    end

    (; ip_mcp, path_mcp, ip_data, path_data)
end

"Compute summary statistics from solver benchmark data."
function summary_statistics(data)
    accumulate_stats(solver_data) = begin
        (; success_rate = fraction_solved(solver_data), runtime_stats(solver_data)...)
    end

    stats =
        (; ip = accumulate_stats(data.ip_data), path = accumulate_stats(data.path_data))
    @info "IP runtime is $(100(stats.ip.μ / stats.path.μ)) % that of PATH."

    stats
end

"Estimate mean and standard deviation of runtimes for all problems."
function runtime_stats(solver_data)
    filtered_times =
        map(datum -> datum.elapsed_time, filter(datum -> datum.success, solver_data))
    μ = Statistics.mean(filtered_times)
    σ = Statistics.stdm(filtered_times, μ)

    (; μ, σ)
end

"Compute fraction of problems solved."
function fraction_solved(solver_data)
    Statistics.mean(datum -> datum.success, solver_data)
end
