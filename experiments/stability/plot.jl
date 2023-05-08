using JLD
using CairoMakie
using TuePlots
using OrdinaryDiffEq
using LaTeXStrings

import BayesExpIntExperiments: alg_styles, C1, C2, Labels, PlotTheme

DIR = "experiments/stability"
data = load(joinpath(DIR, "workprecisiondata.jld"))
wps = data["wps"]

NU = parse(Int, collect(filter(x -> startswith(x, "EK0+IWP"), keys(wps)))[1][9])
algs = (
    # "Tsit5",
    # "BS3",
    "EK0+IWP($NU)",
    "EK1+IWP($NU)",
    "EK0+IOUP($NU)",
    "EK1+IOUP($NU)",
)

set_theme!(
    merge(
        Theme(),
        PlotTheme,
        Theme(
            TuePlots.SETTINGS[:NEURIPS];
            font=false,
            fontsize=true,
            figsize=true,
            thinned=true,
            # width_coeff=0.35,
            nrows=2, ncols=4,
            # subplot_height_to_width_ratio=1/TuePlots.GOLDEN_RATIO,
            # subplot_height_to_width_ratio=1,
        ),
    ),
)

fig = Figure()
sclines = Dict()
ax = Axis(
    fig[1, 1];
    # yticks=[1e-10, 1e-5, 1e-0],
    # xticks=steps == "fixed" ?
    #        [1e0, 1e1, 1e2] :
    #        [1e1, 1e2, 1e3, 1e4],
    xscale=log10,
    yscale=log10,
    yticklabelsvisible=true,
    # title=L"\dot{y} = - y + 10^{%$(Int(log10(b)))} \cdot y^2",
    xlabel="dt",
    # xlabel="nf",
    ylabel="l2 error",
)
for alg in algs
    wp = wps[alg]
    scl = scatterlines!(
        ax,
        [r[:dt] for r in wp],
        # [r[:nf] for r in wp],
        [r[:l2] for r in wp];
        label=alg,
        alg_styles[alg]...,
        color=(alg_styles[alg].color, 0.5),
        markercolor=alg_styles[alg].color,
    )
    sclines[alg] = scl
end

# leg = Legend(
#     fig[:, end+1],
#     [sclines[k] for k in algs],
#     [labels[k] for k in algs],
# )
axislegend(ax)

colgap!(fig.layout, 5)
# ylims!(ax, 1e-5, 1e5) # NU = 1
ylims!(ax, 1e-10, 1e5)  # NU = 2

# save("../bayes-exp-int/figures/gradual_nonlinearity_$steps.pdf", fig, pt_per_unit=1)
save("plot.pdf", fig, pt_per_unit=1)
