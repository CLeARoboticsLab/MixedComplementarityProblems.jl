# Experiments data (technical report §VII)

Benchmark data behind the Experiments section of `docs/paper/main.tex`, collected 2026-07-30.
Hardware/software: see `run_metadata.txt` (AMD Ryzen 9 7950X, 16 physical / 32 logical cores;
NVIDIA RTX 4090 24 GB; Julia 1.12.6). All runs used `julia -t 32 --project=benchmark/gpu`.

## Methodology
- **Cold vs warm start.** We report both: *cold* (all solvers from their default start) is the
  fair, apples-to-apples comparison against PATH and the unbatched solver, which are always cold;
  *warm* (the batched solve seeded with `generate_initial_guess`, the trajectory game's zero-input
  rollout) reflects realistic receding-horizon robotics use. Warm only affects the game (the QP has
  no initial-guess override). Controlled by the `use_initial_guess` kwarg on
  `benchmark_throughput` and the `WARMSTART` env var in `scripts/run_benchmarks.jl`.
- **Distributions, not point estimates.** The QP's random draws are often infeasible by
  construction, making per-instance solve times bimodal (fast feasible / slow-to-diverge
  infeasible). Report mean±std, or violins when bimodal, from the raw per-sample CSVs — not a
  single number. Batched sub-second timings are also noisy (GPU especially at large horizon), so
  we sample many repetitions.

## Key findings
- **Batched solver vs PATH.** Both batched backends clear a batch far faster than sequential PATH
  (game: CPU ~60–90×, GPU ~20–55×).
- **CPU vs GPU is regime-dependent, and the CPU wins on the game.** The GPU only wins end-to-end on
  large *dense* per-instance systems (128-primal QP: 2–3×) or large QP batches. This **corrects
  PR #55's headline** ("GPU beats CPU 2.5–2.8× on the game"), which inverted the ratio direction /
  over-generalized a per-call result. The GPU *factorization kernel* does cross over CPU at large d
  (`percall_timing.csv`), but the CPU's active-set skip — it factorizes only the shrinking active
  subset each Newton step, while cuDSS always processes the whole batch — keeps the CPU ahead
  end-to-end (`active_fraction_T50.csv`: GPU/CPU 0.77×→8.1× as the active set shrinks 1024→32).
- **KLU default.** Switching the unbatched interior-point solver's linear solve from UMFPACK to KLU
  (in-place refactor reusing the symbolic analysis) is ~4–6× faster per solve at identical
  reliability (`table1_summary.txt`), bringing the unbatched solver to ≈ parity with PATH on the QP
  and ~50× faster on the game. Now the package default.

## Files
| file | what |
|---|---|
| `experiments.csv` | single-shot cold results: throughput, gpu_scaling, problem_size, horizon (cold+warm). `warm_start` column. |
| `experiments_warm_reference.csv` | earlier full warm sweep (old schema, no `warm_start` col; QP rows cold-valid, game rows warm). |
| `per_instance.csv` | raw per-instance solve time + status for PATH & unbatched IP (N=1024, QP+game) — for violins / bimodality. |
| `per_instance_table1.csv` | Table I raw: per-instance PATH vs IP(UMFPACK) vs IP(KLU). `table1_summary.txt` has the summary. |
| `per_rep.csv` | raw per-repetition full-batch wall-clock for batched CPU/GPU (+ PATH totals), across the batch-size sweep. |
| `throughput_klu.csv` | clean single-run throughput (PATH vs sequential IP vs batched CPU) with the KLU default — the source for the report's Table II. |
| `thread_scaling.csv` | batched throughput vs CPU thread count (1–32), QP and game — the source for the thread-scaling figure. |
| `percall_timing.csv` | per-call `jacobian!`/`factorize!`/`ldiv!` CPU-vs-GPU vs horizon (d) — the kernel crossover. |
| `active_fraction_T50.csv` | GPU-vs-CPU vs active-set size at fixed d=3500 — the mechanism behind CPU winning end-to-end. |
| `confirm_game_T30_median.txt` | median-of-5 game T=30 CPU-vs-GPU (cold+warm). |
| `run_metadata.txt` | hardware / git / Julia snapshot. |
| `scripts/` | the driver + analysis scripts (repo root auto-resolved, or set `REPO_ROOT`). |

## Reproduce
From the repo root, one GPU stage at a time (single GPU — never overlap):
```
WARMSTART=0 STAGE=throughput   julia -t 32 --project=benchmark/gpu benchmark/results/scripts/run_benchmarks.jl
#   STAGE ∈ {throughput, gpu_scaling, problem_size, horizon}
julia -t 32 --project=benchmark/gpu benchmark/results/scripts/percall.jl        # kernel crossover
julia -t 32 --project=benchmark/gpu benchmark/results/scripts/active_frac.jl     # active-set mechanism
julia -t 32 --project=benchmark/gpu benchmark/results/scripts/table1_klu.jl      # Table I (KLU vs UMFPACK)
julia -t 32 --project=benchmark/gpu benchmark/results/scripts/raw_data.jl        # STAGE=throughput_raw|scaling_raw
```
