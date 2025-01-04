""" Generate a random (convex) quadratic problem of the form
                               min_x 0.5 xᵀ M x - ϕᵀ x
                               s.t.  Ax - b ≥ 0.

NOTE: the problem may not be feasible!
"""
function generate_test_problem(
    ::QuadraticProgramBenchmark;
    num_primals = 100,
    num_inequalities = 100,
)
    G(x, y; θ) =
        let
            (; M, A, ϕ) = unpack_parameters(
                QuadraticProgramBenchmark(),
                θ;
                num_primals,
                num_inequalities,
            )
            M * x - ϕ - A' * y
        end

    H(x, y; θ) =
        let
            (; A, b) = unpack_parameters(
                QuadraticProgramBenchmark(),
                θ;
                num_primals,
                num_inequalities,
            )
            A * x - b
        end

    K(z, θ) =
        let
            x = z[1:num_primals]
            y = z[(num_primals + 1):end]

            [G(x, y; θ); H(x, y; θ)]
        end

    unconstrained_dimension = num_primals
    constrained_dimension = num_inequalities
    lower_bounds = [fill(-Inf, num_primals); fill(0, num_inequalities)]
    upper_bounds = fill(Inf, num_primals + num_inequalities)

    (; K, lower_bounds, upper_bounds)
end

"Generate a random parameter vector Θ corresponding to a convex QP."
function generate_random_parameter(
    ::QuadraticProgramBenchmark;
    rng,
    num_primals = 100,
    num_inequalities = 100,
    sparsity_rate = 0.9,
)
    bernoulli = Distributions.Bernoulli(1 - sparsity_rate)

    M = let
        P =
            randn(rng, num_primals, num_primals) .*
            rand(rng, bernoulli, num_primals, num_primals)
        P' * P
    end

    A =
        randn(rng, num_inequalities, num_primals) .*
        rand(rng, bernoulli, num_inequalities, num_primals)
    b = randn(rng, num_inequalities)
    ϕ = randn(rng, num_primals)

    [reshape(M, length(M)); reshape(A, length(A)); b; ϕ]
end

"Unpack a parameter vector θ into the components of a convex QP."
function unpack_parameters(::QuadraticProgramBenchmark, θ; num_primals, num_inequalities)
    M = reshape(θ[1:(num_primals^2)], num_primals, num_primals)
    A = reshape(
        θ[(num_primals^2 + 1):(num_primals^2 + num_inequalities * num_primals)],
        num_inequalities,
        num_primals,
    )

    b =
        θ[(num_primals^2 + num_inequalities * num_primals + 1):(num_primals^2 + num_inequalities * (num_primals + 1))]
    ϕ = θ[(num_primals^2 + num_inequalities * (num_primals + 1) + 1):end]

    (; M, A, b, ϕ)
end
