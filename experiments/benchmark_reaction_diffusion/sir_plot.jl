using JLD
using CairoMakie
using TuePlots
using OrdinaryDiffEq
using LaTeXStrings

import BayesExpIntExperiments as BEIE
import BayesExpIntExperiments: C1, C2, get_label, PlotTheme, get_alg_style

DIR = "experiments/benchmark_reaction_diffusion"
# data = load(joinpath(DIR, "workprecisiondata.jld"))
data = load(joinpath(DIR, "sir_results.jld"))
results = data["results"]

# NU = results["NU"]
NU = 3

x = :nf
xlabel = String(x)
# xlabel = "Number of steps"
y = :final
ylabel = String(y)
# ylabel = "Error"

algs = (
    # "Tsit5",
    # "BS3",
    # "Rosenbrock23",
    "EK0+IWP($NU)",
    "EKL+IWP($NU)",
    "EK1+IWP($NU)",
    "EK0+IOUP($NU)",
    "EK1+IOUP($NU)",
    "EK0+IOUP($NU)+RB",
    "EK1+IOUP($NU)+RB",
)

set_theme!(
    merge(
        Theme(
            Axis=(
                titlesize=7,
                xticks=LogTicks(WilkinsonTicks(3)),
                yticks=LogTicks(WilkinsonTicks(3)),
                xlabelsize=7,
                ylabelsize=7,
            ),
            Legend=(; labelsize=7),
        ),
        PlotTheme,
        Theme(
            TuePlots.SETTINGS[:NEURIPS];
            font=false,
            fontsize=true,
            figsize=true,
            thinned=true,
            # width_coeff=0.35,
            nrows=1, ncols=2,
            # subplot_height_to_width_ratio=1/TuePlots.GOLDEN_RATIO,
            # subplot_height_to_width_ratio=1,
        ),
    ),
)

fig = Figure()

# Solution plot
prob, L = BEIE.prob_rd_1d_sir()
ref_sol = solve(prob, RadauIIA5(), abstol=1e-20, reltol=1e-20)
Array(ref_sol)
gl = fig[1, 1] = GridLayout()
Label(fig[1, 1, Top()],
    text=rich(rich("a. ", font="Times Bold"),
        rich("SIR ODE solution", font="Times")),
    fontsize=7)
for i in 1:3
    d = length(ref_sol.u[1]) ÷ 3
    ax_sol = Axis(
        gl[1, i],
        # xticks=1:length(prob.u0),
        xticks=([0.5, d + 0.5], ["0", "$d"]),
        # xticksvisible=false,
        xticklabelsvisible=false,
        yticks=[prob.tspan[1], prob.tspan[2]],
        xlabel="x",
        ylabel=i == 1 ? "t" : "",
        yticklabelsvisible=i == 1,
    )
    heatmap!(
        ax_sol,
        1:d,
        ref_sol.t,
        Array(ref_sol)[d*(i-1)+1:d*i, :];
        colormap=[:Blues, :Oranges, :Greens][i],
        # colorrange=(0, 1),
        fxaa=false,
    )
end
colgap!(gl, 2)

# # Work-precision
sclines = Dict()
ax = Axis(
    fig[1, 2];
    xscale=log10,
    yscale=log10,
    yticklabelsvisible=true,
    xlabel=xlabel,
    ylabel=ylabel,
    title=rich(rich("b. ", font="Times Bold"),
        rich("Work-precision diagram", font="Times")),
)
for alg in algs
    wp = results[alg]
    scl = scatterlines!(
        ax,
        [r[x] for r in wp],
        [r[y] for r in wp];
        label=alg,
        get_alg_style(alg)...,
        color=(get_alg_style(alg).color, 0.5),
        markercolor=get_alg_style(alg).color,
    )
    sclines[alg] = scl
end

# Calibration
ax_cal = Axis(
    fig[1, 3];
    xscale=log10,
    yscale=log10,
    yticklabelsvisible=false,
    xlabel="Normalised χ² statistic",
    title=rich(rich("c. ", font="Times Bold"),
        rich("Uncertainty calibration", font="Times")),
)
d = length(prob.u0)
vlines!(ax_cal, [1], color=:gray, linestyle=:dash, linewidth=1)
for alg in algs
    wp = results[alg]
    scl = scatterlines!(
        ax_cal,
        [r[:chi2_final] / d for r in wp],
        # [r[x] for r in wp],
        [r[y] for r in wp];
        label=alg,
        get_alg_style(alg)...,
        color=(get_alg_style(alg).color, 0.5),
        markercolor=get_alg_style(alg).color,
    )
end

leg = Legend(
    fig[:, end+1],
    [sclines[k] for k in algs],
    [get_label(k) for k in algs],
)

colgap!(fig.layout, 5)
# ylims!(ax, 1e-5, 1e5) # NU = 1
# ylims!(ax, 1e-10, 1e5)  # NU = 2
xlims!(ax_cal, 1e-10, 1e10)

save("../bayes-exp-int/figures/sir_reaction_diffusion.pdf", fig, pt_per_unit=1)
