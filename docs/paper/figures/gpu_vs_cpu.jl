""" Generates the GPU-vs-CPU figure (§VII-D, fig:gpu): two panels, both plotting the same
quantity (CPU/GPU throughput ratio = CPU total time / GPU total time; >1 means GPU faster,
<1 means CPU faster). Left: vs. batch size B at the base per-instance size. Right: vs.
per-instance size at fixed B=1024 -- QP's n_x and the game's horizon T live on different
scales, so the right panel uses a twin x-axis (QP on bottom, game on top) sharing the
same y-axis, rather than conflating two different units on one axis.

All values are median-of-N from benchmark/results/per_rep.csv (see
verify_paper_numbers.jl's `median_reps` for the same pattern) -- N ranges from 5 (the
larger problem sizes / long horizons, added in commit 79f12df) to 30 (the base
configs). The game side uses warm_start=true, matching the paper's choice to restrict
the game comparison to the warm-started regime (§VII-D).

Each curve is shown with a shaded band: a CONSERVATIVE envelope
[min(CPU)/max(GPU), max(CPU)/min(GPU)] over the N repeated trials, not a statistical
confidence interval on the ratio itself (CPU and GPU reps are independent, unpaired
repeated trials of the same fixed computation, so there is no natural per-rep pairing
to bootstrap a tighter interval from). This is deliberately the widest reasonable band
given the two marginals; the true variability in the ratio is likely somewhat smaller.

Run with: julia --project=benchmark docs/paper/figures/gpu_vs_cpu.jl
Output:   docs/paper/figures/gpu_vs_cpu.pdf
"""

using Statistics: median
using CairoMakie
using CairoMakie: rich, subscript

const RESULTS = normpath(joinpath(@__DIR__, "..", "..", "..", "benchmark", "results"))

function readcsv(path)
    lines = readlines(path)
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
filt(rows; kw...) = filter(r -> all(getproperty(r, k) == v for (k, v) in pairs(kw)), rows)

pr = readcsv(joinpath(RESULTS, "per_rep.csv"))

"Conservative [lo, med, hi] envelope for the CPU/GPU total-time ratio over N repeated trials."
function speedup_stats(; kw...)
    cpu = filt(pr; device = "cpu", kw...)
    gpu = filt(pr; device = "gpu", kw...)
    (isempty(cpu) || isempty(gpu)) && return (; lo = NaN, med = NaN, hi = NaN)
    cpu_t = getproperty.(cpu, :total_time_s)
    gpu_t = getproperty.(gpu, :total_time_s)
    (;
        lo = minimum(cpu_t) / maximum(gpu_t),
        med = median(cpu_t) / median(gpu_t),
        hi = maximum(cpu_t) / minimum(gpu_t),
    )
end

"Draw a shaded min/max band plus the median line+markers for one series."
function plot_series!(ax, xs, stats; color)
    band!(ax, xs, getproperty.(stats, :lo), getproperty.(stats, :hi); color = (color, 0.2))
    lines!(ax, xs, getproperty.(stats, :med); color, linewidth = 2)
    scatter!(ax, xs, getproperty.(stats, :med); color, markersize = 10)
end

fig = Figure(; size = (900, 380))

# ---- Left: vs. batch size B, at base per-instance size (p32i16 / T10) --------------
axL = Axis(
    fig[1, 1];
    xlabel = "batch size B", ylabel = "CPU/GPU throughput ratio",
    xscale = log10,
)
hlines!(axL, [1]; color = :black, linestyle = :dash, linewidth = 1)

qp_B = [64.0, 256.0, 1024.0, 4096.0, 16384.0]
qp_stats_B = [speedup_stats(; problem = "qp", size = "p32i16", solver = "batched", warm_start = "false", num_samples = B) for B in qp_B]
plot_series!(axL, qp_B, qp_stats_B; color = :steelblue)

game_B = [64.0, 256.0, 1024.0, 4096.0]
game_stats_B = [speedup_stats(; problem = "game", size = "T10", solver = "batched", warm_start = "false", num_samples = B) for B in game_B]
plot_series!(axL, game_B, game_stats_B; color = :firebrick)

# Manual legend entries (plot_series! doesn't tag `label`s, since the band shouldn't get one).
Legend(
    fig[1, 1], [LineElement(; color = :steelblue), LineElement(; color = :firebrick)],
    ["Random QP", "Trajectory game"]; tellwidth = false, tellheight = false, halign = :left, valign = :top, margin = (10, 10, 10, 10),
)

# ---- Right: vs. per-instance size at fixed B = 1024 (twin x-axis) -----------------
axR = Axis(
    fig[1, 2];
    xlabel = rich("QP primal dimension ", "n", subscript("x")),
)
axR_top = Axis(fig[1, 2]; xaxisposition = :top, xlabel = "trajectory-game horizon T")
hidespines!(axR_top)
hideydecorations!(axR_top)
linkyaxes!(axR, axR_top)
hlines!(axR, [1]; color = :black, linestyle = :dash, linewidth = 1)

qp_nx = [32.0, 64.0, 128.0]
qp_sizes = ["p32i16", "p64i32", "p128i64"]
qp_stats_nx = [speedup_stats(; problem = "qp", size = sz, solver = "batched", warm_start = "false", num_samples = 1024.0) for sz in qp_sizes]
plot_series!(axR, qp_nx, qp_stats_nx; color = :steelblue)
xlims!(axR, 16, 144)

game_T = [10.0, 20.0, 30.0, 40.0, 50.0]
game_sizes = ["T10", "T20", "T30", "T40", "T50"]
game_stats_T = [speedup_stats(; problem = "game", size = sz, solver = "batched", warm_start = "true", num_samples = 1024.0) for sz in game_sizes]
plot_series!(axR_top, game_T, game_stats_T; color = :firebrick)
xlims!(axR_top, 5, 55)

outpath = joinpath(@__DIR__, "gpu_vs_cpu.pdf")
save(outpath, fig)
@info "Wrote $outpath"
@info "left QP" qp_B qp_stats_B
@info "left game" game_B game_stats_B
@info "right QP" qp_nx qp_stats_nx
@info "right game" game_T game_stats_T
