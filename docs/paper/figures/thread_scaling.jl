""" Generates the thread-scaling figure (§VII-C, fig:threadscaling): batched-solver
throughput (instances/s) vs. CPU thread count, one curve per problem family, at a fixed
batch size of N = 1024. Reads directly from benchmark/results/thread_scaling.csv (already
one row per (problem, nthreads) — no repeated trials needed here, it's the throughput
number itself, not a wall-clock total).

Run with: julia --project=benchmark docs/paper/figures/thread_scaling.jl
Output:   docs/paper/figures/thread_scaling.pdf
"""

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
filt(rows; kw...) = filter(r -> all(getproperty(r, k) == v for (k, v) in pairs(kw)), rows)

ts = readcsv(joinpath(RESULTS, "thread_scaling.csv"))

fig = Figure(; size = (480, 360))
ax = Axis(
    fig[1, 1];
    xlabel = "CPU threads", ylabel = "throughput (instances/s)",
    xscale = log2, xticks = ([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"]),
)
vlines!(ax, [16]; color = :gray, linestyle = :dash, linewidth = 1, label = "16 physical cores")

colors = Dict("qp" => :steelblue, "game" => :firebrick)
labels = Dict("qp" => "Random QP", "game" => "Trajectory game")
for problem in ("qp", "game")
    rs = sort(filt(ts; problem), by = r -> r.nthreads)
    xs = getproperty.(rs, :nthreads)
    ys = getproperty.(rs, :throughput_per_s)
    lines!(ax, xs, ys; color = colors[problem], linewidth = 2, label = labels[problem])
    scatter!(ax, xs, ys; color = colors[problem], markersize = 10)
end
axislegend(ax; position = :rb)

outpath = joinpath(@__DIR__, "thread_scaling.pdf")
save(outpath, fig)
@info "Wrote $outpath"
