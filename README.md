# MixedComplementarityProblems.jl

[![CI](https://github.com/CLeARoboticsLab/MixedComplementarityProblems.jl/actions/workflows/test.yml/badge.svg)](https://github.com/CLeARoboticsLab/MixedComplementarityProblems.jl/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-BSD-new)](https://opensource.org/license/bsd-3-clause)

This package provides an easily-customizable interface for expressing mixed complementarity problems (MCPs) which are defined in terms of an arbitrary vector of parameters. `MixedComplementarityProblems` implements a reasonably high-performance interior point method for solving these problems, and integrates with `ChainRulesCore` and `ForwardDiff` to enable automatic differentiation of solutions with respect to problem parameters.

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

## A fancier demo

If you'd like to get a better sense of the kinds of problems `MixedComplementarityProblems` was built for, check out the example in `examples/lane_change.jl`. This problem encodes a two-player game in which each player is driving a car and wishes to choose a trajectory that tracks a preferred lane center, maintains a desired speed, minimizes control actuation effort, and avoids collision with the other player. The problem is naturally expressed as a noncooperative game, and encoded as a mixed complementarity problem.

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
