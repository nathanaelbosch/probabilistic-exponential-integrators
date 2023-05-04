using JLD
using CairoMakie
using TuePlots
using OrdinaryDiffEq
using LaTeXStrings

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

C1, C2 = Makie.wong_colors()[1:2]
alg_styles = Dict(
    "Tsit5" => (color=:gray,
        # linestyle=:solid,
        marker=:dtriangle),
    "BS3" => (color=:gray,
        # linestyle=:solid,
        marker=:utriangle),
    "EK0+IWP($NU)" => (color=C1,
        # linestyle=:solid,
        marker=:diamond),
    "EK1+IWP($NU)" => (color=C1,
        # linestyle=:dash,
        marker=:pentagon),
    "EK0.5+IWP($NU)" => (color=C1,
        # linestyle=:dot,
        marker=:hexagon),
    "EK0+IOUP($NU)" => (color=C2,
        # linestyle=:solid,
        marker=:star4),
    "EK1+IOUP($NU)" => (color=C2,
        # linestyle=:dash,
        marker=:star5),
)
labels = Dict(
    "Tsit5" => L"\text{Tsit5}",
    "BS3" => L"\text{BS3}",
    "EK0+IWP($NU)" => L"\text{EK0 & IWP(%$NU)}",
    "EK1+IWP($NU)" => L"\text{EK1 & IWP(%$NU)}",
    "EK0.5+IWP($NU)" => L"\text{EK0.5 & IWP(%$NU)}",
    "EK0+IOUP($NU)" => L"\text{EK0 & IOUP(%$NU)}",
    "EK1+IOUP($NU)" => L"\text{EK1 & IOUP(%$NU)}",
)

T1 = Theme(
    TuePlots.SETTINGS[:NEURIPS];
    font=false,
    fontsize=true,
    figsize=true,
    thinned=true,
    # width_coeff=0.35,
    nrows=1, ncols=1,
    # subplot_height_to_width_ratio=1/TuePlots.GOLDEN_RATIO,
    # subplot_height_to_width_ratio=1,
)
T2 = Theme(
    # Axis=(
    # xlabelsize=8,
    # ylabelsize=8,
    # titlesize=8
    # ),
    Label=(
        halign=:left,
        tellwidth=false,
        # tellheight=false,
        justification=:left,
        padding=(12, 0, 1, 0),
        # font="Times New Roman",
    ),
    ScatterLines=(
        markersize=8,
        linewidth=3,
        strokewidth=0.2,
    ),
    Legend=(;
        labelsize=8,
        patchlabelgap=-8,
        patchsize=(10, 10),
        framevisible=false,
    ),
)
set_theme!(merge(T2, T1))

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
