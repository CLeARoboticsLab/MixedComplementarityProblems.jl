using Test: @testset, @test
using MixedComplementarityProblems
const MCP = MixedComplementarityProblems
using KernelAbstractions: CPU
using SparseArrays: sparse, nonzeros, nnz
using LinearAlgebra: norm, transpose, I
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using Zygote: Zygote
using Random: Random

@testset "BatchedMCP" begin
    """ Batched/GPU KKTSystem layer + BatchedInteriorPoint solver, exercised on the
    KernelAbstractions CPU backend, on the QP
                       min_x 0.5 xᵀ M x - θᵀ x  s.t.  A x - b ≥ 0.
    """
    M = [2.0 1 0; 1 2 1; 0 1 2]
    A = Matrix(1.0I, 3, 3)
    b = [1.0, 1, 1]
    G(x, y; θ) = M * x - θ - transpose(A) * y
    H(x, y; θ) = A * x - b
    mcp = MCP.PrimalDualMCP(
        G,
        H;
        unconstrained_dimension = 3,
        constrained_dimension = 3,
        parameter_dimension = 3,
        compute_sensitivities = true,
        compute_kernel_evaluators = true,
    )

    dev = CPU()
    n, m = 3, 3
    d = n + 2m
    B = 64

    # A batch of evaluation points (Y, S elementwise positive: interior).
    Random.seed!(1)
    Θ = randn(n, B)
    X = randn(n, B)
    Y = rand(m, B) .+ 0.1
    S = rand(m, B) .+ 0.1
    ϵ = fill(0.1, B)

    @testset "materialize" begin
        cache = MCP.materialize(mcp, MCP.BatchedSparse(), dev; batch_size = B)
        @test cache.d == d
        @test cache.batch_size == B
        # cache picks up the full nonzero pattern of the unbatched Jacobian
        # (nnz(::SparseFunction) == length(.rows), the count of structural nonzeros).
        @test cache.nnz == nnz(mcp.∇F_z!)
        @test size(cache.nzval) == (cache.nnz, B)
        # diag_nz entries really are on the diagonal of the shared pattern.
        @test all(cache.rows[cache.diag_nz] .== cache.cols[cache.diag_nz])
    end

    @testset "residual! matches unbatched F!" begin
        cache = MCP.materialize(mcp, MCP.BatchedSparse(), dev; batch_size = B)
        F = zeros(d, B)
        MCP.residual!(F, mcp, X, Y, S, Θ, ϵ; device = dev)
        for bb in 1:B
            buf = zeros(d)
            mcp.F!(buf, X[:, bb], Y[:, bb], S[:, bb]; θ = Θ[:, bb], ϵ = ϵ[bb])
            @test F[:, bb] ≈ buf
        end
    end

    @testset "jacobian! matches unbatched ∇F_z!, plus diagonal η" begin
        cache = MCP.materialize(mcp, MCP.BatchedSparse(), dev; batch_size = B)

        # η = 0: nonzero values match the unbatched sparse Jacobian exactly.
        MCP.jacobian!(cache, mcp, X, Y, S, Θ, ϵ, zeros(B); device = dev)
        for bb in 1:B
            Jbuf = mcp.∇F_z!.result_buffer
            mcp.∇F_z!(Jbuf, X[:, bb], Y[:, bb], S[:, bb]; θ = Θ[:, bb], ϵ = ϵ[bb])
            @test nonzeros(Jbuf) ≈ cache.nzval[:, bb]
        end

        # η > 0 lands only on the structurally-present diagonal entries.
        MCP.jacobian!(cache, mcp, X, Y, S, Θ, ϵ, fill(0.7, B); device = dev)
        Jbuf = mcp.∇F_z!.result_buffer
        mcp.∇F_z!(Jbuf, X[:, 1], Y[:, 1], S[:, 1]; θ = Θ[:, 1], ϵ = ϵ[1])
        diff = cache.nzval[:, 1] .- nonzeros(Jbuf)
        offdiag = setdiff(1:cache.nnz, cache.diag_nz)
        @test all(isapprox.(diff[cache.diag_nz], 0.7; atol = 1e-12))
        @test all(abs.(diff[offdiag]) .< 1e-12)
    end

    @testset "factorize!/ldiv! solve the batched Newton system" begin
        cache = MCP.materialize(mcp, MCP.BatchedSparse(), dev; batch_size = B)
        F = zeros(d, B)
        δz = zeros(d, B)
        MCP.residual!(F, mcp, X, Y, S, Θ, ϵ; device = dev)
        MCP.jacobian!(cache, mcp, X, Y, S, Θ, ϵ, zeros(B); device = dev)
        MCP.factorize!(cache)
        MCP.ldiv!(δz, cache, -F)
        for bb in 1:B
            Asp = sparse(cache.rows, cache.cols, cache.nzval[:, bb], d, d)
            @test norm(Asp * δz[:, bb] + F[:, bb], Inf) < 1e-10
        end
    end

    @testset "BatchedInteriorPoint matches unbatched at B=1" begin
        θ = [-0.5, 0.3, 0.4]
        unb = MCP.solve(MCP.InteriorPoint(), mcp, θ)
        bat = MCP.solve(MCP.BatchedInteriorPoint(), mcp, reshape(θ, :, 1); device = dev)
        @test bat.status[1] == :solved
        # Both target tol = 1e-4 on the KKT residual; solutions agree to ~tol.
        @test isapprox(bat.x[:, 1], unb.x; atol = 1e-3)
        @test isapprox(bat.y[:, 1], unb.y; atol = 1e-3)
        @test isapprox(bat.s[:, 1], unb.s; atol = 1e-3)
    end

    @testset "BatchedInteriorPoint solves a genuine batch" begin
        tol = 1e-4
        sol = MCP.solve(MCP.BatchedInteriorPoint(), mcp, Θ; device = dev, tol)
        @test all(sol.status .== :solved)
        @test maximum(sol.kkt_error) ≤ tol
        for bb in 1:B
            x = sol.x[:, bb]
            y = sol.y[:, bb]
            Hval = A * x - b
            @test all(abs.(M * x - Θ[:, bb] - transpose(A) * y) .< 5e-3)  # G ≈ 0
            @test all(Hval .> -5e-3)                                       # H ≥ 0
            @test all(y .> -5e-3)                                          # y ≥ 0
            @test abs(sum(y .* Hval)) < 5e-3                               # complementarity
        end
    end

    @testset "solve_jacobian_θ matches unbatched sensitivities" begin
        # Converge each instance (unbatched) and collect the primal-dual point + ϵ.
        X = zeros(n, B)
        Y = zeros(m, B)
        S = zeros(m, B)
        ϵ = zeros(B)
        sols = map(1:B) do bb
            sol = MCP.solve(MCP.InteriorPoint(), mcp, Θ[:, bb])
            X[:, bb] = sol.x
            Y[:, bb] = sol.y
            S[:, bb] = sol.s
            ϵ[bb] = sol.ϵ
            sol
        end

        ∂z∂θ = MCP.solve_jacobian_θ(mcp, X, Y, S, Θ, ϵ; device = dev)
        @test size(∂z∂θ) == (d, n, B)

        # Each slice matches the unbatched QR-based sensitivity to machine precision.
        for bb in 1:B
            ref = MCP.AutoDiff._solve_jacobian_θ(mcp, sols[bb], Θ[:, bb])
            @test isapprox(∂z∂θ[:, :, bb], ref; atol = 1e-9)
        end

        # And the full z(θ) Jacobian agrees with finite differences on a sample instance.
        zofθ =
            θ -> let sol = MCP.solve(MCP.InteriorPoint(), mcp, θ)
                [sol.x; sol.y; sol.s]
            end
        fd = FiniteDiff.finite_difference_jacobian(zofθ, Θ[:, 1])
        @test isapprox(∂z∂θ[:, :, 1], fd; atol = 1e-5)
    end

    @testset "BatchedInteriorPoint is differentiable through the batch" begin
        # Scalar loss over a whole batch of solves; gradient is wrt the (nθ × B) Θ.
        loss =
            Θ -> let sol = MCP.solve(MCP.BatchedInteriorPoint(), mcp, Θ; device = dev)
                sum(sol.x .^ 2) + sum(sol.y .^ 2)
            end

        ∇_reverse = only(Zygote.gradient(loss, Θ))
        ∇_forward = only(Zygote.gradient(Θ -> Zygote.forwarddiff(loss, Θ), Θ))
        ∇_forwarddiff = ForwardDiff.gradient(loss, Θ)
        ∇_finitediff = FiniteDiff.finite_difference_gradient(loss, Θ)

        # Reverse and forward both contract the same analytic ∂z∂θ, so they agree to
        # ~machine precision; only the finite-difference checks warrant a loose atol.
        @test isapprox(∇_reverse, ∇_forward; atol = 1e-9)
        @test isapprox(∇_reverse, ∇_finitediff; atol = 1e-3)
        @test isapprox(∇_forwarddiff, ∇_finitediff; atol = 1e-3)
    end
end
