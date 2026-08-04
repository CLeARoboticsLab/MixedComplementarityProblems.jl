# Recomputes every specific number cited in the Experiments section of
# docs/paper/main.tex directly from the raw files in benchmark/results/, and diffs
# each against the value currently asserted in the paper (hardcoded below as
# `claimed=`). Dependency-free (stdlib only) since benchmark/Project.toml carries no
# CSV/DataFrames dependency and these files are plain comma-joined rows (no
# quoting/escaping), matching how the generating scripts wrote them.
#
# Usage: julia benchmark/results/scripts/verify_paper_numbers.jl
#
# NOTE: the `claimed=` values are transcribed from main.tex at the time this script
# was written. If the prose/tables change, update the corresponding `claimed=` here
# too — this script does not read main.tex itself.

using Statistics: mean, median, std
using Printf: @sprintf, @printf

const HERE = @__DIR__
const RESULTS = normpath(joinpath(HERE, ".."))

# ---------------------------------------------------------------------------
# Minimal CSV reader: header row + comma-joined fields, numeric where parseable.
# ---------------------------------------------------------------------------
function readcsv(name)
    lines = readlines(joinpath(RESULTS, name))
    header = [Symbol(replace(h, "/" => "_")) for h in split(lines[1], ",")]
    rows = NamedTuple[]
    for line in lines[2:end]
        isempty(line) && continue
        fields = split(line, ",")
        vals = map(fields) do f
            isempty(f) && return missing
            v = tryparse(Float64, f)
            v === nothing ? String(f) : v
        end
        push!(rows, NamedTuple{Tuple(header)}(Tuple(vals)))
    end
    rows
end

matches(r, kw) = all(getproperty(r, k) == v for (k, v) in pairs(kw))
filt(rows; kw...) = filter(r -> matches(r, kw), rows)
onlyrow(rows; kw...) = only(filt(rows; kw...))

const NCHECKS = Ref(0)
const NFAIL = Ref(0)

function check(name, claimed, actual; rtol = 0.03)
    NCHECKS[] += 1
    ok = isapprox(claimed, actual; rtol)
    ok || (NFAIL[] += 1)
    @printf("%-8s %-62s paper=%-10s data=%-10s\n",
        ok ? "ok" : "MISMATCH", name, string(claimed), @sprintf("%.4g", actual))
    ok
end

function checkeq(name, claimed, actual)
    NCHECKS[] += 1
    ok = claimed == actual
    ok || (NFAIL[] += 1)
    @printf("%-8s %-62s paper=%-10s data=%-10s\n",
        ok ? "ok" : "MISMATCH", name, string(claimed), string(actual))
    ok
end

section(title) = println("\n== ", title, " ", "="^max(1, 76 - length(title)))

# ===========================================================================
section("Table I — single-instance reliability & speed (per_instance_table1.csv)")
# ===========================================================================
t1 = readcsv("per_instance_table1.csv")

# Every (solved, mean, std) triple in Table I's body, plus the median asserted in its
# caption (only given there for the QP row; `nothing` where the paper doesn't assert one).
const TABLE1_CLAIMS = Dict(
    ("qp", "path")         => (solved = 437, mean = 3.9, std = 3.6, median = 4.1),
    ("qp", "ip_umfpack")   => (solved = 431, mean = 17.7, std = 19.6, median = 2.5),
    ("qp", "ip_klu")       => (solved = 431, mean = 4.7, std = 6.3, median = 0.6),
    ("game", "path")       => (solved = 996, mean = 45.2, std = 22.3, median = nothing),
    ("game", "ip_umfpack") => (solved = 1024, mean = 5.5, std = 4.4, median = nothing),
    ("game", "ip_klu")     => (solved = 1024, mean = 0.9, std = 0.5, median = nothing),
)

stats = Dict()
for problem in ("qp", "game"), solver in ("path", "ip_umfpack", "ip_klu")
    rs = filt(t1; problem, solver)
    ms = 1000 .* getproperty.(rs, :time_s)
    solved = count(r -> r.solved == "true", rs)
    stats[(problem, solver)] = (; n = length(rs), solved, mean = mean(ms), median = median(ms), std = std(ms))
    r = stats[(problem, solver)]
    @printf("  %-4s %-11s solved=%4d/%-4d  mean=%7.3fms  median=%7.3fms  std=%7.3fms\n",
        problem, solver, r.solved, r.n, r.mean, r.median, r.std)

    claim = TABLE1_CLAIMS[(problem, solver)]
    checkeq("Table I $problem/$solver solved", claim.solved, r.solved)
    check("Table I $problem/$solver mean ms", claim.mean, r.mean)
    check("Table I $problem/$solver std ms", claim.std, r.std)
    # caption medians are rounded to 1 decimal place; at ~0.6ms that's a wider relative
    # band than the default rtol, so widen it here rather than loosen the default globally.
    claim.median === nothing || check("Table I $problem/$solver median ms (caption)", claim.median, r.median; rtol = 0.04)
end

qp_median_speedup = stats[("qp", "path")].median / stats[("qp", "ip_klu")].median
game_median_speedup = stats[("game", "path")].median / stats[("game", "ip_klu")].median
check("text: median speedup vs PATH, qp (~7x)", 7.0, qp_median_speedup)
check("text: median speedup vs PATH, game (~55x)", 55.0, game_median_speedup)

qp_klu_vs_umf = stats[("qp", "ip_umfpack")].median / stats[("qp", "ip_klu")].median
game_klu_vs_umf = stats[("game", "ip_umfpack")].median / stats[("game", "ip_klu")].median
println("  KLU vs UMFPACK (median ratio): qp=$(round(qp_klu_vs_umf, digits=2))x  ",
    "game=$(round(game_klu_vs_umf, digits=2))x  (text claims 4--6x)")

# ===========================================================================
section("Table II — batched CPU throughput vs. PATH (throughput_klu.csv)")
# ===========================================================================
t2 = readcsv("throughput_klu.csv")
qp_path = onlyrow(t2; problem = "qp", solver = "path")
qp_seq = onlyrow(t2; problem = "qp", solver = "ip_klu")
qp_batch = onlyrow(t2; problem = "qp", solver = "batched_cpu", warm_start = "false")
game_path = onlyrow(t2; problem = "game", solver = "path")
game_seq = onlyrow(t2; problem = "game", solver = "ip_klu")
game_batch = onlyrow(t2; problem = "game", solver = "batched_cpu", warm_start = "false")

for r in (qp_path, qp_seq, qp_batch, game_path, game_seq, game_batch)
    @printf("  %-5s %-12s total=%7.4fs  solved=%d/%d\n",
        r.problem, r.solver, r.total_time_s, Int(r.num_solved), Int(r.num_samples))
end

# Every (solved, total_time_s) pair in Table II's body, for all six rows.
# PATH totals reflect commit 79f12df's matched-tolerance rerun (convergence_tolerance=1e-4,
# was PATH's stricter default 1e-6) — negligible per-instance effect, but Table II's PATH
# row/speedups were updated in main.tex to the re-timed values.
const TABLE2_CLAIMS = Dict(
    "qp path" => (row = qp_path, solved = 437, total = 3.97),
    "qp seq" => (row = qp_seq, solved = 431, total = 5.61),
    "qp batch" => (row = qp_batch, solved = 424, total = 0.58),
    "game path" => (row = game_path, solved = 996, total = 46.4),
    "game seq" => (row = game_seq, solved = 1024, total = 0.97),
    "game batch" => (row = game_batch, solved = 1024, total = 0.44),
)
for (label, claim) in TABLE2_CLAIMS
    checkeq("Table II $label solved", claim.solved, Int(claim.row.num_solved))
    check("Table II $label total (s)", claim.total, claim.row.total_time_s)
end

check("Table II qp speedup vs PATH (6.8x)", 6.8, qp_path.total_time_s / qp_batch.total_time_s)
check("Table II qp seq/PATH (0.7x)", 0.7, qp_path.total_time_s / qp_seq.total_time_s)
check("Table II game speedup vs PATH (105.5x)", 105.5, game_path.total_time_s / game_batch.total_time_s)
check("text: unbatched already ~48x on game", 47.8, game_path.total_time_s / game_seq.total_time_s)
check("text: threading-only speedup, qp (9.7x)", 9.7, qp_seq.total_time_s / qp_batch.total_time_s)
check("text: threading-only speedup, game (2.2x)", 2.2, game_seq.total_time_s / game_batch.total_time_s)

# ===========================================================================
section("CPU thread scaling (thread_scaling.csv)")
# ===========================================================================
ts = readcsv("thread_scaling.csv")
for problem in ("qp", "game")
    rs = sort(filt(ts; problem), by = r -> r.nthreads)
    for r in rs
        @printf("  %-4s threads=%2d  throughput=%7.1f/s\n", problem, Int(r.nthreads), r.throughput_per_s)
    end
    println("  $problem peak throughput: $(maximum(r.throughput_per_s for r in rs))/s at ",
        "$(rs[argmax([r.throughput_per_s for r in rs])].nthreads) threads")
end
check("qp throughput @1 thread", 371.0, onlyrow(ts; problem = "qp", nthreads = 1.0).throughput_per_s)
check("qp throughput @4 threads", 1211.0, onlyrow(ts; problem = "qp", nthreads = 4.0).throughput_per_s)
check("game throughput @1 thread", 725.0, onlyrow(ts; problem = "game", nthreads = 1.0).throughput_per_s)
check("game throughput @4 threads", 2486.0, onlyrow(ts; problem = "game", nthreads = 4.0).throughput_per_s)

# ===========================================================================
section("GPU vs. CPU — batch-size and problem-size sweeps")
# ===========================================================================
# Statistic policy for this section: MEDIAN throughout, from per_rep.csv wherever
# repeated trials exist. As of commit 79f12df's `problem_size_raw` stage, that now
# covers: the QP batch-size sweep (gpu_scaling, p32i16, all B), the T=10 horizon point
# (both already had >=30 reps), the larger-problem-size sweep (p64i32/p128i64 @ B=1024,
# 5 reps), and the FULL horizon sweep (T=20/30/40/50 @ B=1024, cold+warm, 5 reps each).
# Only the problem-size sweep's B=4096 points (n_x=64,128) remain single-run (that stage
# only added reps at B=1024); labeled accordingly below. The old one-off
# `confirm_game_T30_median.txt` recheck is superseded by this per_rep.csv coverage (it
# used a different set of 5 runs and is no longer read here).
ex = readcsv("experiments.csv")
pr = readcsv("per_rep.csv")

function median_reps(; kw...)
    rs = filt(pr; kw...)
    isempty(rs) && return nothing
    (;
        n = length(rs), median = median(getproperty.(rs, :total_time_s)),
        solved = Int(first(rs).num_solved), total = Int(first(rs).num_samples),
    )
end

qp_b4096_cpu_reps = median_reps(problem = "qp", size = "p32i16", solver = "batched", device = "cpu", warm_start = "false", num_samples = 4096.0)
qp_b4096_gpu_reps = median_reps(problem = "qp", size = "p32i16", solver = "batched", device = "gpu", warm_start = "false", num_samples = 4096.0)
println("  QP B=4096: median of $(qp_b4096_cpu_reps.n) CPU reps = $(round(qp_b4096_cpu_reps.median, digits=4))s, ",
    "median of $(qp_b4096_gpu_reps.n) GPU reps = $(round(qp_b4096_gpu_reps.median, digits=4))s")
check("QP GPU speedup at B=4096 (1.8x, median of $(qp_b4096_cpu_reps.n) reps)",
    1.8, qp_b4096_cpu_reps.median / qp_b4096_gpu_reps.median)

println("  problem-size sweep (n_x=64,128 @ B=1024: median of 5 reps; @ B=4096: single run)")
sizelabel = Dict(32.0 => "p32i16", 64.0 => "p64i32", 128.0 => "p128i64")
for nx in (32.0, 64.0, 128.0), B in (1024.0, 4096.0)
    cpu_reps = median_reps(problem = "qp", size = sizelabel[nx], solver = "batched", device = "cpu", warm_start = "false", num_samples = B)
    gpu_reps = median_reps(problem = "qp", size = sizelabel[nx], solver = "batched", device = "gpu", warm_start = "false", num_samples = B)
    if !isnothing(cpu_reps) && cpu_reps.n > 1
        ratio = cpu_reps.median / gpu_reps.median
        @printf("  qp n_x=%-4g B=%-5g  cpu/gpu=%.2fx (median of %d reps)  solved=%d/%d\n",
            nx, B, ratio, cpu_reps.n, cpu_reps.solved, cpu_reps.total)
    else
        gpu = onlyrow(ex; experiment = "problem_size", problem = "qp", num_primals = nx, num_samples = B, device = "gpu")
        cpu = onlyrow(ex; experiment = "problem_size", problem = "qp", num_primals = nx, num_samples = B, device = "cpu")
        ratio = cpu.total_time_s / gpu.total_time_s
        @printf("  qp n_x=%-4g B=%-5g  cpu/gpu=%.2fx (single run)  solved=%d/%d\n",
            nx, B, ratio, Int(gpu.num_solved), Int(B))
    end
end
nx128_reps = (
    cpu = median_reps(problem = "qp", size = "p128i64", solver = "batched", device = "cpu", warm_start = "false", num_samples = 1024.0),
    gpu = median_reps(problem = "qp", size = "p128i64", solver = "batched", device = "gpu", warm_start = "false", num_samples = 1024.0),
)
nx128_ratio = nx128_reps.cpu.median / nx128_reps.gpu.median
check("text: QP n_x=128 GPU speedup (roughly 3x, median of 5)", 3.0, nx128_ratio; rtol = 0.05)

nx32 = median_reps(problem = "qp", size = "p32i16", solver = "batched", device = "cpu", warm_start = "false", num_samples = 1024.0)
nx64 = median_reps(problem = "qp", size = "p64i32", solver = "batched", device = "cpu", warm_start = "false", num_samples = 1024.0)
nx128 = median_reps(problem = "qp", size = "p128i64", solver = "batched", device = "cpu", warm_start = "false", num_samples = 1024.0)
check("text: QP solved fraction 41% (n_x=32)", 0.41, nx32.solved / 1024)
check("text: QP solved fraction 96% (n_x=64, median of 5)", 0.96, nx64.solved / 1024)
check("text: QP solved fraction 100% (n_x=128, median of 5)", 1.00, nx128.solved / 1024)

println("\n  horizon sweep (T=10: >=30 reps; T=20/30/40/50: median of 5 reps, cold+warm)")
horizons = Dict()
for T in (10.0, 20.0, 30.0, 40.0, 50.0), warm in (false, true)
    sz = T == 10.0 ? "T10" : "T$(Int(T))"
    cpu_reps = median_reps(problem = "game", size = sz, solver = "batched", device = "cpu", warm_start = string(warm), num_samples = 1024.0)
    gpu_reps = median_reps(problem = "game", size = sz, solver = "batched", device = "gpu", warm_start = string(warm), num_samples = 1024.0)
    isnothing(cpu_reps) && continue
    horizons[(T, warm)] = (cpu = cpu_reps, gpu = gpu_reps)
    @printf("  game T=%-3g warm=%-5s  CPU=%7.4fs(n=%d)  GPU=%7.4fs(n=%d)  GPU/CPU=%.2fx  solved=%d/%d\n",
        T, warm, cpu_reps.median, cpu_reps.n, gpu_reps.median, gpu_reps.n,
        gpu_reps.median / cpu_reps.median, cpu_reps.solved, cpu_reps.total)
end

t30 = horizons[(30.0, true)]
check("text: game T=30 CPU total (4.1s, median of $(t30.cpu.n))", 4.1, t30.cpu.median)
check("text: game T=30 GPU total (8.0s, median of $(t30.gpu.n))", 8.0, t30.gpu.median)
check("text: game T=30 GPU/CPU ratio (1.9x slower)", 1.9, t30.gpu.median / t30.cpu.median)
check("text: game T=30 solved fraction ~92% (paper text; per_rep.csv gives ~93%)", 0.92, t30.cpu.solved / 1024; rtol = 0.02)

t50 = horizons[(50.0, true)]
check("text: game T=50 CPU total (12.7s, median of $(t50.cpu.n))", 12.7, t50.cpu.median)
check("text: game T=50 GPU total (16.3s, median of $(t50.gpu.n))", 16.3, t50.gpu.median)
check("text: game T=50 solved fraction ~57% (median of 5)", 0.57, t50.cpu.solved / 1024; rtol = 0.05)

# ===========================================================================
section("Per-call kernel crossover (percall_timing.csv)")
# ===========================================================================
pc = readcsv("percall_timing.csv")
for r in sort(pc, by = r -> (r.d, r.device))
    ratio_here = begin
        other = onlyrow(pc; horizon = r.horizon, device = r.device == "CPU" ? "GPU" : "CPU")
        r.device == "CPU" ? other.jacfac_ms / r.jacfac_ms : r.jacfac_ms / other.jacfac_ms
    end
    @printf("  T=%-3g d=%-5g %-3s  jacfac=%6.2fms  (GPU/CPU=%.3g)\n", r.horizon, r.d, r.device, r.jacfac_ms, ratio_here)
end
cpu2100 = onlyrow(pc; horizon = 30.0, device = "CPU")
gpu2100 = onlyrow(pc; horizon = 30.0, device = "GPU")
cpu3500 = onlyrow(pc; horizon = 50.0, device = "CPU")
gpu3500 = onlyrow(pc; horizon = 50.0, device = "GPU")
cpu4900 = onlyrow(pc; horizon = 70.0, device = "CPU")
gpu4900 = onlyrow(pc; horizon = 70.0, device = "GPU")
println("  d=2100 GPU/CPU=$(round(gpu2100.jacfac_ms/cpu2100.jacfac_ms, digits=3)) (still >1, GPU slower)")
println("  d=3500 GPU/CPU=$(round(gpu3500.jacfac_ms/cpu3500.jacfac_ms, digits=3)) (now <1, GPU faster)")
println("  => crossover lies between d=2100 and d=3500; text claims d\\approx 2500")
check("text: GPU ~1.4x cheaper per-step by d=4900", 1.4, cpu4900.jacfac_ms / gpu4900.jacfac_ms)

# ===========================================================================
section("Active-set mechanism at d=3500 (active_fraction_T50.csv)")
# ===========================================================================
af = readcsv("active_fraction_T50.csv")
for r in sort(af, by = r -> -r.active_k)
    @printf("  active_k=%4d  jacfac_GPU/CPU=%.3g\n", Int(r.active_k), r.jacfac_GPU_CPU)
end
full = onlyrow(af; active_k = 1024.0)
sparse_ = onlyrow(af; active_k = 32.0)
check("text: GPU per-step advantage at full batch (1.3x)", 1.3, 1 / full.jacfac_GPU_CPU)
check("text: GPU per-step disadvantage at active=32 (8x)", 8.0, sparse_.jacfac_GPU_CPU)

# ===========================================================================
section("Summary")
# ===========================================================================
println("$(NCHECKS[] - NFAIL[])/$(NCHECKS[]) checks passed.")
NFAIL[] > 0 && (println("$(NFAIL[]) MISMATCH(es) — see above."); exit(1))
