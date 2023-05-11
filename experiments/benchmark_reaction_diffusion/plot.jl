using JLD
using CairoMakie
using TuePlots
using OrdinaryDiffEq
using Distributions
using LaTeXStrings

import BayesExpIntExperiments as BEIE
import BayesExpIntExperiments: C1, C2, get_label, PlotTheme, get_alg_style

DIR = "experiments/benchmark_reaction_diffusion"
data = load(joinpath(DIR, "workprecisiondata.jld"))
results = data["results"]

# NU = results["NU"]
NU = 2

x = :nsteps
xlabel = String(x)
xlabel = "Number of steps"
y = :final
ylabel = "Final error"

algs = (
    # "Tsit5",
    # "BS3",
    # "Rosenbrock23",
    # "Rosenbrock32",
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
prob, L = BEIE.prob_rd_1d_fisher()
@info "plot.jl" length(prob.u0)
d = length(prob.u0)
ref_sol = solve(prob, RadauIIA5(), abstol=1e-20, reltol=1e-20)
Array(ref_sol)
ax_sol = Axis(
    fig[1, 1],
    xticks=([0.5, length(prob.u0) + 0.5], ["0", "$d"]),
    # xticksvisible=false,
    # xticklabelsvisible=false,
    yticks=[prob.tspan[1], prob.tspan[2]],
    xlabel="Space [x]",
    ylabel="Time [t]",
    title=rich(rich("a. ", font="Times New Roman Bold"),
        # rich("Reaction-diffusion model", font="Times New Roman")),
        rich("ODE solution", font="Times New Roman")),
)
CairoMakie.heatmap!(
    ax_sol,
    1:length(ref_sol.u[1]),
    ref_sol.t,
    Array(ref_sol);
    # log.(ref_sol.t[2:end]),
    # Array(ref_sol)[:, 2:end];
    colormap=:thermal, colorrange=(0, 1),
    fxaa=false,
)

# Work-precision
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
    wp = results[alg][2:end]
    scl = scatterlines!(
        ax,
        [(prob.tspan[2] - prob.tspan[1]) / r[:dt] for r in wp],
        [r[y] for r in wp];
        label=alg,
        get_alg_style(alg)...,
        color=(get_alg_style(alg).color, 0.5),
        markercolor=get_alg_style(alg).color,
    )
    sclines[alg] = scl
end

ax_cal = Axis(
    fig[1, 3];
    xscale=log10,
    yscale=log10,
    yticklabelsvisible=false,
    xlabel="Normalised χ² statistic",
    title=rich(rich("c. ", font="Times Bold"),
        rich("Uncertainty calibration", font="Times")),
)
# dist = Chisq(length(prob.u0))
vlines!(
    ax_cal,
    # [mean(dist)],
    [1],
    color=:gray, linestyle=:dash,
    linewidth=1,
)
# 99% confidence interval
# ql, qr = quantile(dist, [0.005, 0.995])
# vlines!(
#     ax_cal,
#     [ql, qr],
#     color=:black, linestyle=:dash,
#     linewidth=0.5,
# )
for alg in algs
    wp = results[alg][2:end]
    scl = scatterlines!(
        ax_cal,
        [r[:chi2_final] / d for r in wp],
        [r[y] for r in wp];
        label=alg,
        get_alg_style(alg)...,
        color=(get_alg_style(alg).color, 0.5),
        markercolor=get_alg_style(alg).color,
    )
    sclines[alg] = scl
end

# axislegend(ax)
leg = Legend(
    fig[:, end+1],
    [sclines[k] for k in algs],
    [get_label(k) for k in algs],
    labelfont="Times",
)

colgap!(fig.layout, 5)
# ylims!(ax, 1e-5, 1e5) # NU = 1
# xlims!(ax, 2e-3, 2e0)  # NU = 2
CairoMakie.ylims!(ax, nothing, 1e5)  # NU = 2
CairoMakie.ylims!(ax_cal, nothing, 1e5)  # NU = 2
CairoMakie.xlims!(ax_cal, 1e-10, 1e10)  # NU = 2
CairoMakie.xlims!(ax, 1e0, nothing)  # NU = 2

save("../bayes-exp-int/figures/reaction_diffusion.pdf", fig, pt_per_unit=1)
