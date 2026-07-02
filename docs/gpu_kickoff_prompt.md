# GPU work kickoff prompt

Paste the block below into a fresh Claude Code session running **on the remote NVIDIA
machine** (inside `tmux`, on branch `gpu-version-for-real`) to jump-start the cuDSS
batched-sparse solver work. It is self-contained — it assumes no memory of prior work and
points at the files that carry the design.

---

```
We're adding a GPU batched-sparse linear solver to this package
(MixedComplementarityProblems.jl), a pure-Julia interior-point MCP solver. The CPU
batched solver already works and ships in v0.2.x; the GPU path is scaffolded but not
implemented. Your job is to implement and validate it on this machine's NVIDIA GPU.

You should be on branch `gpu-version-for-real` (based off the merged gpu-version work and
still containing gpu_prototype/, which has earlier exploration worth skimming). Confirm
this before starting.

START BY READING (do not skip — they carry the full design):
  1. docs/gpu_kkt_design.md — the architecture. Focus on option D1(a) (one shared
     symbolic factorization across the batch, via NVIDIA cuDSS) and the GPU-specific
     notes near the end (synchronize hoisting, CSC→CSR).
  2. ext/MixedComplementarityProblemsCUDSSExt.jl — the extension scaffold. Its top
     docstring is a detailed DESIGN block; the three method bodies currently call
     `_todo(...)` and error. These are exactly what you implement.
  3. src/batched_solver.jl — the CPU reference. The verbs you're implementing on the
     GPU (`_materialize_linsolve`, `factorize!`, `ldiv!`) already have working CPU
     (KLU) implementations here. The interior-point solve loop and the residual/Jacobian
     assembly kernels are device-portable (KernelAbstractions) and must be reused
     unchanged — you are ONLY adding the CUDABackend methods in the extension.

BEFORE WRITING ANY CODE, verify the environment and report back:
  - `julia --project=. -e 'using CUDA; @show CUDA.functional(); CUDA.versioninfo()'`
  - `nvidia-smi` (driver + GPU)
  - That CUDA + CUDSS instantiate/precompile in this project (they're weakdeps; the
     extension MixedComplementarityProblemsCUDSSExt loads only when both are present).
  - Check the INSTALLED CUDSS.jl version and confirm its batched (uniform) API — the
     DESIGN block assumes ~0.6–0.8; the exact CudssSolver / cudss(...) batched calling
     convention must be confirmed against what's actually installed, not assumed.

CONSTRAINTS:
  - Do NOT modify the CPU solve path or the unbatched InteriorPoint solver.
  - The data-layout subtlety is real: jacobian! fills cache.nzval in CSC order; cuDSS
     wants CSR. Compute the CSC→CSR permutation ONCE on the host at analyze time, move
     it to the device, and gather per-factorization on-device. The DESIGN block has the
     recipe.
  - For GPU, hoist KernelAbstractions.synchronize out of the per-iteration
     residual!/jacobian! (see the NOTE in src/batched_solver.jl) — sync only at the
     host convergence read-back.

VALIDATE by mirroring the CPU tests in test/batched.jl: solve a batch on the GPU and
check the results match the CPU/KLU path (same problems, same tolerances) — including
the multi-RHS ldiv! used by parameter sensitivities (solve_jacobian_θ).

First give me the environment report and a short implementation plan; don't start editing
until we've looked at the cuDSS API together.
```
