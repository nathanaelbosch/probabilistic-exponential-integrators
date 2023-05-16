using JLD
using CairoMakie
using TuePlots
using OrdinaryDiffEq
using Distributions
using LaTeXStrings

import BayesExpIntExperiments as BEIE
import BayesExpIntExperiments: C1, C2, get_label, PlotTheme, get_alg_style, LINEALPHA

DIR = "experiments/benchmark_reaction_diffusion"
data = load(joinpath(DIR, "burgers_results.jld"))
results = data["results"]

# NU = results["NU"]
NU = 2

x = :dt
x2 = :nf
xlabel =
    x == :nsteps ? "Number of steps" :
    x == :nf ? "Number of f evaluations" :
    x == :dt ? "Step size" :
    String(x)
x2label =
    x2 == :chi2_final ? "Normalised χ² statistic" :
    x2 == :nf ? "Number of f evaluations" :
    x2 == :time ? "Runtime [s]" :
    String(x2)
y = :final
ylabel = String(y)
ylabel = "Final error"

algs = (
    # "Tsit5",
    # "BS3",
    # "Rosenbrock23",
    # "Rosenbrock32",
    "EK0+IWP($NU)",
    "EK1+IWP($NU)",
    "EKL+IWP($NU)",
    "EK0+IOUP($NU)",
    # "EK1+IOUP($NU)",
    # "EK0+IOUP($NU)+RB",
    "EK1+IOUP($NU)+RB",
)

set_theme!(
    merge(
        Theme(
            figure_padding=(-7, 12, 1, 0),
            Axis=(
                # titlesize=7,
                xticks=LogTicks(WilkinsonTicks(3)),
                yticks=LogTicks(WilkinsonTicks(3)),
                # xlabelsize=7,
                # ylabelsize=7,
                # xticklabelsize=7,
                # yticklabelsize=7,
            ),
            # Legend=(; labelsize=7),
        ),
        PlotTheme,
        Theme(
            TuePlots.SETTINGS[:NEURIPS];
            font=false,
            fontsize=true,
            figsize=true,
            thinned=true,
            # width_coeff=0.35,
            nrows=1.1, ncols=3,
            # subplot_height_to_width_ratio=1/TuePlots.GOLDEN_RATIO,
            # subplot_height_to_width_ratio=1,
        ),
    ),
)

fig = Figure()

# Solution plot
prob, L = BEIE.prob_burgers(; N=100)
@info "plot.jl" length(prob.u0)
d = length(prob.u0)
ref_sol = solve(prob, RadauIIA5(), abstol=1e-20, reltol=1e-20, saveat=0.01)
Array(ref_sol)
gl = fig[1, 1] = GridLayout()
ax_sol = Axis(
    gl[1, 1],
    xticks=([0.5, length(prob.u0) + 0.5], ["0", "1"]),
    yticks=[prob.tspan[1], prob.tspan[2]],
    xlabel="Space",
    ylabel="Time",
    title=rich(rich("a. ", font="Times New Roman Bold"),
        rich("ODE solution", font="Times New Roman")),
)
hm = CairoMakie.heatmap!(
    ax_sol,
    1:length(ref_sol.u[1]),
    ref_sol.t,
    Array(ref_sol);
    colormap=:balance, colorrange=(-0.5, 0.5),
    fxaa=false,
)
# Add colorbar
cb = Colorbar(gl[1, 2], hm, size=3)
colgap!(gl, 3)
colsize!(gl, 1, Aspect(1, 1.0))

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
        x == :nsteps ?
        [(prob.tspan[2] - prob.tspan[1]) / r[:dt] for r in wp] :
        [r[x] for r in wp],
        [r[y] for r in wp];
        label=alg,
        get_alg_style(alg)...,
        color=(get_alg_style(alg).color, LINEALPHA),
        markercolor=get_alg_style(alg).color,
    )
    sclines[alg] = scl
end

ax2 = Axis(
    fig[1, 3];
    xscale=log10,
    yscale=log10,
    yticklabelsvisible=false,
    xlabel=x2label,
    # title=rich(rich("c. ", font="Times Bold"),
    #     rich("Uncertainty calibration", font="Times")),
)
(x2 == :chi2_final) && vlines!(ax2, [1], color=:gray, linestyle=:dash, linewidth=1)
for alg in algs
    wp = results[alg][2:end]
    if x2 == :chi2_final && !(:chi2_final in keys(wp[1]))
        continue
    end
    scl = scatterlines!(
        ax2,
        (x2 == :chi2_final) ? [r[x2] / d for r in wp] : [r[x2] for r in wp],
        [r[y] for r in wp];
        label=alg,
        get_alg_style(alg)...,
        color=(get_alg_style(alg).color, LINEALPHA),
        markercolor=get_alg_style(alg).color,
    )
    sclines[alg] = scl
end

leg = Legend(fig[:, end+1], [sclines[k] for k in algs], [get_label(k) for k in algs],
    tellwidth=true)

colgap!(fig.layout, 1, 5)
colgap!(fig.layout, 2, 3)
colgap!(fig.layout, 3, 0)
linkyaxes!(ax, ax2)
(x2 == :chi2_final) && CairoMakie.xlims!(ax2, 1e-8, 1e20)

save("../bayes-exp-int/figures/burgers.pdf", fig, pt_per_unit=1)
