""" Generates Fig. 1(b): total wall-clock to clear a batch of trajectory-game instances,
PATH (sequential) vs. our CPU-multithreaded and GPU batched solvers, vs. batch size.

Uses the SAME statistic policy as benchmark/results/scripts/verify_paper_numbers.jl:
median over whatever repeated trials were actually collected in
benchmark/results/per_rep.csv (10 reps for the batched CPU/GPU points; PATH has only
1-3 reps at each size, so its "median" is over however many exist there — labeled on
the plot).

Run with: julia --project=benchmark docs/paper/figures/teaser_throughput.jl
Output:   docs/paper/figures/teaser_throughput.pdf
"""

using Statistics: median
using CairoMakie

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

matches(r, kw) = all(getproperty(r, k) == v for (k, v) in pairs(kw))
filt(rows; kw...) = filter(r -> matches(r, kw), rows)

pr = readcsv(joinpath(RESULTS, "per_rep.csv"))

function median_time(; kw...)
    rs = filt(pr; kw...)
    isempty(rs) && return (; n = 0, median = NaN)
    (; n = length(rs), median = median(getproperty.(rs, :total_time_s)))
end

const SIZES = [64.0, 256.0, 1024.0, 4096.0]

series = Dict(
    "PATH" => [median_time(; problem = "game", size = "T10", solver = "path", warm_start = "false", num_samples = B) for B in SIZES],
    "CPU (batched)" => [median_time(; problem = "game", size = "T10", solver = "batched", device = "cpu", warm_start = "false", num_samples = B) for B in SIZES],
    "GPU (batched)" => [median_time(; problem = "game", size = "T10", solver = "batched", device = "gpu", warm_start = "false", num_samples = B) for B in SIZES],
)

for (label, pts) in series, (B, p) in zip(SIZES, pts)
    @info "$label B=$(Int(B)): median of $(p.n) reps = $(round(p.median, digits=4))s"
end

fig = Figure(; size = (620, 250))
ax = Axis(
    fig[1, 1];
    xlabel = "batch size B",
    ylabel = "total wall-clock time (s)",
    xscale = log2,
    yscale = log10,
    xticks = (SIZES, string.(Int.(SIZES))),
)

colors = Dict("PATH" => :gray30, "CPU (batched)" => :steelblue, "GPU (batched)" => :firebrick)
markers = Dict("PATH" => :circle, "CPU (batched)" => :rect, "GPU (batched)" => :utriangle)

for label in ("PATH", "CPU (batched)", "GPU (batched)")
    ys = [p.median for p in series[label]]
    lines!(ax, SIZES, ys; color = colors[label], label, linewidth = 2)
    scatter!(ax, SIZES, ys; color = colors[label], marker = markers[label], markersize = 12)
end
axislegend(ax; position = :rb)

outpath = joinpath(@__DIR__, "teaser_throughput.pdf")
save(outpath, fig)
@info "Wrote $outpath"
