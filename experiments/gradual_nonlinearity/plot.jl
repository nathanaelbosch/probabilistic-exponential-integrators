using JLD
using CairoMakie
using TuePlots
using OrdinaryDiffEq
# using Plots
using LaTeXStrings

# DIR = @__DIR__
DIR = "experiments/gradual_nonlinearity"
data = load(joinpath(DIR, "workprecisiondata.jld"))
# steps = "adaptive"
steps = "fixed"
wpss = data["wpss_$steps"]

bs = (0.00001, 0.001, 0.1)
NU = 2
algs = (
    # "Tsit5",
    # "BS3",
    "EK0+IWP($NU)",
    "EK1+IWP($NU)",
    "EK0.5+IWP($NU)",
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
    nrows=2, ncols=4,
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
axes = []
sclines = Dict()
for i in 1:length(bs)
    b = bs[i]
    wps = wpss[b]
    ax = Axis(
        fig[1, i];
        # yticks=[1e-10, 1e-5, 1e-0],
        xticks=steps == "fixed" ?
               [1e0, 1e1, 1e2] :
               [1e1, 1e2, 1e3, 1e4],
        xscale=log10,
        yscale=log10,
        yticklabelsvisible=i == 1,
        title=L"\dot{y} = - y + 10^{%$(Int(log10(b)))} \cdot y^2",
        xlabel="N",
        # xlabel="nf",
        ylabel=i == 1 ? "L2 error" : "",
    )
    push!(axes, ax)
    for alg in algs
        wp = wps[alg]
        scl = scatterlines!(
            ax,
            [r[:nsteps] for r in wp],
            # [r[:nf] for r in wp],
            [r[:L2] for r in wp];
            label=alg,
            alg_styles[alg]...,
            color=(alg_styles[alg].color, 0.5),
            markercolor=alg_styles[alg].color,
        )
        sclines[alg] = scl
    end
end
linkyaxes!(axes...)
linkxaxes!(axes...)
if steps == "fixed"
    ylims!.(axes, Ref((1e-13, 1e1)))
end

# for i in 1:length(bs)
#     b = bs[i]
#     Label(fig.layout[1, i, TopLeft()],
#           L"\dot{y} = -y + %$b y^2")
# end

leg = Legend(
    fig[:, end+1],
    [sclines[k] for k in algs],
    [labels[k] for k in algs],
)

colgap!(fig.layout, 5)

# Plots problems
function f!(du, u, p, t)
    @. du = p.a * u + p.b * u^2
end
u0 = [1.0]
tspan = (0.0, 10.0)
for (i, b) in enumerate(bs)
    p = (a=-1, b=b)
    prob = ODEProblem(f!, u0, tspan, p)
    sol = solve(prob, Rodas5P())

    ax = Axis(
        fig[1, i], width=Relative(0.4), height=Relative(0.3),
        # halign=0.95, valign=0.95,
        halign=0.05, valign=0.05,
        backgroundcolor=:white,
        xticksvisible=false,
        yticksvisible=false,
        xticklabelsvisible=false,
        yticklabelsvisible=false,
        xgridvisible=false,
        ygridvisible=false,
    )
    lines!(ax, sol.t, Array(sol)[:], color=(:black, 0.8), linewidth=1)
    # magic to bring the inset plot to front
    # copy-pasted from: https://discourse.julialang.org/t/inset-axis-in-makie/70209
    translate!(ax.scene, 0, 0, 10)
    elements = keys(ax.elements)
    filtered = filter(ele -> ele != :xaxis && ele != :yaxis, elements)
    foreach(ele -> translate!(ax.elements[ele], 0, 0, 9), filtered)
end

save("../bayes-exp-int/figures/gradual_nonlinearity_$steps.pdf", fig, pt_per_unit=1)
