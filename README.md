# MixedComplementarityProblems.jl

[![CI](https://github.com/CLeARoboticsLab/MixedComplementarityProblems.jl/actions/workflows/test.yml/badge.svg)](https://github.com/CLeARoboticsLab/MixedComplementarityProblems.jl/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-BSD-new)](https://opensource.org/license/bsd-3-clause)

This package provides an easily-customizable interface for expressing mixed complementarity problems (MCPs) which are defined in terms of an arbitrary vector of parameters. `MixedComplementarityProblems` implements a reasonably high-performance interior point method for solving these problems, and integrates with `ChainRulesCore` and `ForwardDiff` to enable automatic differentiation of solutions with respect to problem parameters.

As of `v0.2.2`, `MixedComplementarityProblems.jl` implements CPU multithreading and GPU-parallelized solvers as well, enabled via `KernelAbstractions.jl`. Check out the benchmarking README [here](https://github.com/CLeARoboticsLab/MixedComplementarityProblems.jl/blob/main/benchmark/README.md) for more details.

If you find this project useful in your work, please cite the accompanying [paper](https://arxiv.org/pdf/2608.00959):
```
@article{fridovich2026mcps,
    title={MixedComplementarityProblems.jl: A Fast, Batched, Open-Source Interior Point Solver for Mixed Complementarity Problems},
    author={David Fridovich-Keil},
    year={2026},
    journal={arXiv preprint arXiv:2608.00959}
}
```

## What are MCPs?

Mixed complementarity problems (MCPs) are a class of mathematical program, and they arise in a wide variety of application problems. In particular, one way they can arise is via the KKT conditions of nonlinear programs and noncooperative games. This package provides a utility for constructing MCPs from (parameterized) games, cf. `src/game.jl` for further details. To see the connection between KKT conditions and MCPs, read the next section.

## Why this package?

As discussed below, this package replicates functionality already available in [ParametricMCPs](https://github.com/JuliaGameTheoreticPlanning/ParametricMCPs.jl). Our intention here is to provide an easily customizable and open-source solver with efficiency and reliability that is at least comparable with the [PATH](https://pages.cs.wisc.edu/~ferris/path.html) solver which `ParametricMCPs` uses under the hood (actually, it hooks into the interface to the `PATH` binaries which is provided by another wonderful package, [PATHSolver](https://github.com/chkwon/PATHSolver.jl)). Hopefully, users will find it useful to modify the interior point solver provided in this package for their own application problems, use it for highly parallelized implementations (since it is in pure Julia), etc.

## Installation

`MixedComplementarityProblems` is a registered package and can be installed with the standard Julia package manager as follows:
```julia
] add MixedComplementarityProblems
```

## Quickstart guide

Suppose we have the following quadratic program:
```displaymath
min_x 0.5 xᵀ M x - θᵀ x
s.t. Ax - b ≥ 0.
```

The KKT conditions for this problem can be expressed as follows:
```displaymath
G(x, y; θ) = Mx - θ - Aᵀ y = 0
H(x, y; θ) = Ax - b ≥ 0
y ≥ 0
yᵀ H(x, y; θ) = 0,
```
where `y` is the Lagrange multiplier associated to the constraint `Ax - b ≥ 0` in the original problem.

This is precisely a MCP, whose standard form is:
```displaymath
G(x, y; θ) = 0
0 ≤ y ⟂ H(x, y; θ) ≥ 0.
```

Now, we can encode this problem and solve it using `MixedComplementarityProblems` as follows:

```julia
using MixedComplementarityProblems

M = [2 1; 1 2]
A = [1 0; 0 1]
b = [1; 1]
θ = rand(2)

G(x, y; θ) = M * x - θ - A' * y
H(x, y; θ) = A * x - b

mcp = MixedComplementarityProblems.PrimalDualMCP(
    G,
    H;
    unconstrained_dimension = size(M, 1),
    constrained_dimension = length(b),
    parameter_dimension = size(M, 1),
)
sol = MixedComplementarityProblems.solve(MixedComplementarityProblems.InteriorPoint(), mcp, θ)
```

The solver can easily be warm-started from a given initial guess:
```julia
sol = MixedComplementarityProblems.solve(
    MixedComplementarityProblems.InteriorPoint(),
    mcp,
    θ;
    x₀ = # your initial guess
    y₀ = # your **positive** initial guess
)
```

Note that the initial guess for the $y$ variable must be elementwise positive. This is because we are using an interior point method; for further details, refer to `src/solver.jl`.

Finally, `MixedComplementarityProblems` integrates with `ChainRulesCore` and `ForwardDiff` so you can differentiate through the solver itself! For example, suppose we wanted to find the value of $\theta$ in the problem above which solves
```displaymath
min_{θ, x, y} f(x, y)
s.t. (x, y) solves MCP(θ).
```

We could do so by initializing with a particular value of $\theta$ and then iteratively descending the gradient $\nabla_\theta f$, which we can easily compute via:
```julia
mcp = MixedComplementarityProblems.PrimalDualMCP(
    G,
    H;
    unconstrained_dimension = size(M, 1),
    constrained_dimension = length(b),
    parameter_dimension = size(M, 1),
    compute_sensitivities = true,
)

function f(θ)
    sol = MixedComplementarityProblems.solve(MixedComplementarityProblems.InteriorPoint(), mcp, θ)

    # Some example objective function that depends on `x` and `y`.
    sum(sol.x .^ 2) + sum(sol.y .^ 2)
end

∇f = only(Zygote.gradient(f, θ))
```

## Batched solving (CPU multithreading and GPU)

Many applications need to solve a whole *batch* of MCPs that share the same structure but
differ only in their parameters `θ` — e.g. sampling many initial conditions of a
trajectory game, or sweeping many instances of a parameterized program. For this,
`MixedComplementarityProblems` provides the `BatchedInteriorPoint` solver, which solves the
entire batch in a single call, parallelized across CPU threads or an NVIDIA GPU via a
single [`KernelAbstractions.jl`](https://github.com/JuliaGPU/KernelAbstractions.jl)
backend. The unbatched `InteriorPoint` solver above is unchanged; the batched solver is a
separate, additive entry point.

Two things differ from the unbatched setup:

1. Build the MCP with `compute_kernel_evaluators = true`. The batched solver needs these
   device-portable evaluators, so they are opt-in (they add to build time).
2. Stack your parameters into an `(nθ × B)` matrix `Θ`, where column `b` is the parameter
   vector of instance `b` (`B` is the batch size).

### CPU (multithreaded)

Start Julia with several threads — `julia -t 4`. On heterogeneous CPUs (e.g. Apple
silicon) use `-t <#performance-cores>`; oversubscribing past the physical performance
cores plateaus or regresses throughput. Then, reusing the QP from the quickstart above:

```julia
using MixedComplementarityProblems

M = [2 1; 1 2]
A = [1 0; 0 1]
b = [1; 1]

G(x, y; θ) = M * x - θ - A' * y
H(x, y; θ) = A * x - b

# Note the `compute_kernel_evaluators = true`, required by the batched solver.
mcp = MixedComplementarityProblems.PrimalDualMCP(
    G,
    H;
    unconstrained_dimension = size(M, 1),
    constrained_dimension = length(b),
    parameter_dimension = size(M, 1),
    compute_kernel_evaluators = true,
)

# Build a batch of B = 256 parameter vectors as an (nθ × B) matrix.
B = 256
Θ = rand(size(M, 1), B)   # column b is instance b's parameter vector

sol = MixedComplementarityProblems.solve(
    MixedComplementarityProblems.BatchedInteriorPoint(),
    mcp,
    Θ,
)
```

The result is a named tuple `(; status, x, y, s, kkt_error, ϵ, outer_iters, total_iters)`.
`status` is a length-`B` vector of `:solved` / `:failed`, and `x`, `y`, `s` are batched
`(· × B)` matrices — column `b` is instance `b`'s solution. For example, the solution of
instance 5 is `sol.x[:, 5]`, and `count(==(:solved), sol.status)` counts how many instances
converged.

The batched solver mirrors the unbatched schedule and accepts the same kinds of options,
per-instance where appropriate — e.g. `tol`, warm starts (`X₀`, `Y₀`, `S₀`, with `Y₀`, `S₀`
elementwise positive), and `regularize_linear_solve` (`:identity` default / `:internal` /
`:none`). Trajectory games (built via `ParametricGame(...; compute_kernel_evaluators =
true)`) are supported and solve with the default `:identity` scheme.

> **Note on batches with hard/infeasible instances.** A batch typically mixes easy, hard,
> and infeasible instances. Instances that neither converge nor diverge are declared
> `:failed` and frozen after `max_stall_rounds` (default 5) consecutive non-converging
> outer rounds, so a few stragglers do not drag the whole batch to the iteration ceiling.
> Tune `max_stall_rounds` against your problem distribution.

### GPU (NVIDIA, via cuDSS)

The GPU path uses the exact same solver call — you only (1) load the CUDA/cuDSS extension,
(2) move the parameter matrix to the device, and (3) pass `device = CUDABackend()`. The
GPU backend is provided by a package extension that activates when both `CUDA` and `CUDSS`
are loaded (both ship artifacts only for NVIDIA-capable platforms, so the base package
loads and runs unchanged on non-NVIDIA machines):

```julia
using MixedComplementarityProblems
using CUDA, CUDSS   # loading both activates the GPU (cuDSS) backend extension

# `mcp` is built exactly as in the CPU example (compute_kernel_evaluators = true) — its
# symbolic structure is device-independent, so the same MCP works on CPU and GPU.

Θ_gpu = CuArray(Θ)   # move the (nθ × B) parameter matrix to the GPU

sol = MixedComplementarityProblems.solve(
    MixedComplementarityProblems.BatchedInteriorPoint(),
    mcp,
    Θ_gpu;
    device = CUDA.CUDABackend(),
)
```

The returned `x`, `y`, `s` live on the GPU (as `CuArray`s); bring them back with
`Array(sol.x)` if you need them on the host.

> If you would rather write device-generic code (one code path that runs on either CPU or
> GPU), move parameters with `Adapt.adapt(device, Θ)` instead of `CuArray` — that is the
> pattern the benchmarks use. `Adapt` is not a dependency of this package, so you would need
> to add it to your own project (`] add Adapt`).

> **Performance status.** Both batched backends clear a batch far faster than sequential
> `PATH` (the trajectory game runs ~77–177× faster on the multithreaded CPU, ~26–57× on the
> GPU, across batch sizes 64–4096). Between the two backends the story is regime-dependent:
> the GPU wins end-to-end only on large *dense* per-instance systems (e.g. randomly generated
> QPs with ≳128 primal variables, roughly 3×) or large batches of them, while the **CPU is
> faster on the trajectory game** at all tested horizons. The GPU's batched sparse
> factorization does become cheaper *per instance* than threaded KLU as the per-instance
> system grows, but the CPU's active-set skip — it factorizes only the still-active instances
> each Newton step, whereas cuDSS always processes the whole batch — keeps the CPU ahead
> end-to-end on the game. See the [benchmarking
> README](https://github.com/CLeARoboticsLab/MixedComplementarityProblems.jl/blob/main/benchmark/README.md)
> for the full breakdown and numbers.

## A fancier demo

If you would like to get a better sense of the kinds of problems `MixedComplementarityProblems` was built for, check out the example in `examples/lane_change.jl`. This problem encodes a two-player game in which each player is driving a car and wishes to choose a trajectory that tracks a preferred lane center, maintains a desired speed, minimizes control actuation effort, and avoids collision with the other player. The problem is naturally expressed as a noncooperative game, and encoded as a mixed complementarity problem.

To run the example, activate the `examples` environment
```julia
] activate examples
```
and then, from within the examples directory run
```julia
include("TrajectoryExamples")
TrajectoryExamples.run_lane_change_example()
```

This will generate a video animation and save it as `sim_steps.mp4`, and it should show two vehicles smoothly changing lanes and avoiding collisions. Once compiled, the entire example should run in a few seconds (including time to save everything).

## Acknowledgement and future plans

This project inherits many key ideas from [ParametricMCPs](https://github.com/JuliaGameTheoreticPlanning/ParametricMCPs.jl), which provides essentially identical functionality but which currently only supports the (closed-source, but otherwise excellent) [PATH](https://pages.cs.wisc.edu/~ferris/path.html) solver. Ultimately, this `MixedComplementarityProblems` will likely merge with `ParametricMCPs` to provide an identical frontend and allow users a flexible choice of backend solver. Currently, `MixedComplementarityProblems` replicates a substantially similar interface as that provided by `ParametricMCPs`, but there are some (potentially annoying) differences that users should take care to notice, e.g., in the function signature for `solve(...)`.

## Other related projects

If you are curious about other related software, consider checking out [JuliaGameTheoreticPlanning](https://github.com/orgs/JuliaGameTheoreticPlanning/repositories).
