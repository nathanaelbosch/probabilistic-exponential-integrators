using LinearAlgebra
using OrdinaryDiffEq, ProbNumDiffEq, DiffEqDevTools
using JLD
using CairoMakie
using TuePlots
using LaTeXStrings
using DataStructures

import BayesExpIntExperiments as BEIE

DT = 4 // 2
NU = 2

function f!(du, u, p, t)
    @. du = p.a * u + p.b * u^2
end
u0 = [1.0]
tspan = (0.0, 10.0)

b = 0.1
bs = [0.0001, 0.001, 0.01, 0.9, 0.99]
bs = [(1.0 ./ 10.0 .^ (4:-0.5:0.1))..., 0.9, 0.99]
results = DefaultDict(() -> Float64[])
setups = (
    "EK0+IWP" => (prob, EK0(prior=IWP(NU), diffusionmodel=FixedDiffusion())),
    "EK1+IWP" => (prob, EK1(prior=IWP(NU), diffusionmodel=FixedDiffusion())),
    "EK05+IWP" => (prob_badjac, EK1(prior=IWP(NU), diffusionmodel=FixedDiffusion())),
    "EK0+IOUP" => (prob, EK0(prior=IOUP(NU, -1), diffusionmodel=FixedDiffusion())),
    "EK1+IOUP" => (prob, EK1(prior=IOUP(NU, -1), diffusionmodel=FixedDiffusion())),
)
for b in bs
    @info "b = $b"
    push!(results["bs"], b)
    p = (a=-1, b=b)
    prob = ODEProblem(f!, u0, tspan, p)
    prob_badjac = ODEProblem(ODEFunction(f!, jac=(J, u, p, t) -> (J .= p.a)), u0, tspan, p)

    ref_sol = solve(prob, RadauIIA5(), abstol=1e-20, reltol=1e-20)

    for (name, (_prob, alg)) in setups
        sol = solve(_prob, alg, adaptive=false, dt=DT)
        errsol = appxtrue(sol, ref_sol; dense_errors=false, timeseries_errors=true)
        push!(results[name], errsol.errors[:l2])
    end
end

############################################################################################
# Plot the data
############################################################################################
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
        # patchsize=(10, 10),
    ),
)
set_theme!(merge(T2, T1))

C1, C2 = Makie.wong_colors()[1:2]
alg_styles = Dict(
    "EK0+IWP" => (color=C1, marker=:diamond),
    "EK1+IWP" => (color=C1, marker=:pentagon),
    "EK05+IWP" => (color=C1, marker=:hexagon),
    "EK0+IOUP" => (color=C2, marker=:star4),
    "EK1+IOUP" => (color=C2, marker=:star5),
)

fig = Figure()
ax = Axis(
    fig[1, 1];
    xscale=log10,
    yscale=log10,
    title=L"\dot{y} = -y + b y^2",
    # xlabel="Number of steps",
    xlabel="b",
    ylabel="Mean square error",
)
sclines = []
for name in first.(setups)
    scl = scatterlines!(
        ax, results["bs"], results[name]; label=name,
        alg_styles[name]...,
        color=(alg_styles[name].color, 0.8),
        markercolor=alg_styles[name].color,
    )
    push!(sclines, scl)
end

leg = Legend(
    fig[1, 1],
    sclines,
    collect(first.(setups)),
    margin=(10, 10, 10, 10),
    halign=:right,
    valign=:top,
    framevisible=false,
)
# axislegend(ax)

save("fig.pdf", fig, pt_per_unit=1)
