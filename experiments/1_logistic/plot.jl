using JLD
using CairoMakie
using TuePlots
using OrdinaryDiffEq
using LaTeXStrings

import BayesExpIntExperiments: get_alg_style, get_label, PlotTheme

DIR = @__DIR__
data = load(joinpath(DIR, "workprecisiondata.jld"))
steps = "fixed"
wpss = data["wpss_$steps"]

bs = (1e-10, 1e-7, 1e-4, 1e-1) |> reverse
NU = 2
algs = (
    "EK0+IWP($NU)",
    "EK1+IWP($NU)",
    "EKL+IWP($NU)",
    "EK0+IOUP($NU)",
)

set_theme!(
    merge(
        Theme(Axis=(; yticks=([1e0, 1e-10, 1e-20], ["10⁰", "10⁻¹⁰", "10⁻²⁰"]))),
        PlotTheme,
        Theme(
            TuePlots.SETTINGS[:NEURIPS];
            font=false,
            fontsize=true,
            figsize=true,
            thinned=true,
            nrows=1, ncols=3,
        ),
    ),
)

fig = Figure(
    figure_padding=(0, 8, 0, 0),
)
axes = []
sclines = Dict()
for i in 1:length(bs)
    b = bs[i]
    wps = wpss[b]
    ax = Axis(
        fig[1, i];
        xticks=steps == "fixed" ?
               [1e0, 1e1, 1e2] :
               [1e1, 1e2, 1e3, 1e4],
        xscale=log10,
        yscale=log10,
        yticklabelsvisible=i == 1,
        title=L"\dot{y} = - y + 10^{%$(Int(log10(b)))} \cdot y^2",
        xlabel="Number of steps",
        ylabel=i == 1 ? "Final error" : "",
    )
    push!(axes, ax)
    for alg in algs
        wp = wps[alg]
        scl = scatterlines!(
            ax,
            [r[:nsteps] for r in wp],
            [r[:final] for r in wp];
            label=alg,
            get_alg_style(alg)...,
            color=(get_alg_style(alg).color, BEIE.LINEALPHA),
            markercolor=get_alg_style(alg).color,
        )
        sclines[alg] = scl
    end
end
linkyaxes!(axes...)
linkxaxes!(axes...)
if steps == "fixed"
    ylims!.(axes, Ref((1e-23, 1e3)))
end

leg = Legend(
    fig[:, end+1],
    [sclines[k] for k in algs],
    [get_label(k) for k in algs],
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
        fig[1, i], width=Relative(0.4), height=Relative(0.25),
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

filename = joinpath(DIR, "logistic.pdf")
save(filename, fig, pt_per_unit=1)
@info "Saved figure to $filename"
