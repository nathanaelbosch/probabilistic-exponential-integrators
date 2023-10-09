using ProbNumDiffEq, DiffEqCallbacks, OrdinaryDiffEq, LinearAlgebra
import Plots
Plots.theme(:default;
    markersize=1,
    markerstrokewidth=0.1,
)

DIR = @__DIR__

function f(du, u, p, t)
    du[1] = -0.5u[1] + 20u[2]
    du[2] = -20u[2]
end
u0 = [0.0, 1.0]
tspan = (0.0, 3.0)
prob = ODEProblem(f, u0, tspan)
L = [-0.5 20; 0 -20]

ref_sol = solve(prob, ImplicitEuler(), abstol=1e-10, reltol=1e-10)

sol_ek0 = solve(prob, EK0());
sol_ek0.destats

sol_ek1 = solve(prob, EK1());
sol_ek1.destats

order = 3
diffusionmodel = FixedDiffusion()

dt_ek0 = 0.01
dt_ek1 = 0.25
dt_ek0_ioup = 0.5
SMOOTH = true
sol_ek0 = solve(
    prob,
    EK0(; order, diffusionmodel, smooth=SMOOTH);
    adaptive=false,
    dt=dt_ek0,
    dense=SMOOTH,
);
sol_ek1 = solve(
    prob,
    EK1(; order, diffusionmodel, smooth=SMOOTH);
    adaptive=false,
    dt=dt_ek1,
    dense=SMOOTH,
);
sol_ek0_ioup = solve(prob, ExpEK(; L, diffusionmodel, smooth=SMOOTH);
    adaptive=false, dt=dt_ek1, dense=SMOOTH);

############################################################################################
using CairoMakie, TuePlots, LaTeXStrings, ColorSchemes
import BayesExpIntExperiments: PlotTheme

COLORS = ColorSchemes.tableau_10.colors
ALPHA = 0.5

set_theme!(
    merge(
        Theme(
            Axis=(;
                titlesize=8
            ),
            Lines=(;
                linewidth=0.5,
                linestyle=:dash,
            ),
            Series=(
                linewidth=0.5,
                solid_color=:gray,
            ),
            ScatterLines=(; markersize=2, linewidth=0.5),
        ),
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

fig = Figure()
ax_ek0 = Axis(
    fig[1, 1];
    yticks=[0, 1],
    xticks=[0, tspan[2]],
    title=rich(rich("a. ", font="Times New Roman Bold"),
        rich("Explicit method (not A-stable)", font="Times New Roman")),
    xlabel=L"t",
    ylabel=L"y(t)",
)
ax_ek1 = Axis(
    fig[1, 2];
    yticks=[0, 1],
    xticks=[0, tspan[2]],
    yticklabelsvisible=false,
    title=rich(rich("b. ", font="Times New Roman Bold"),
        rich("Semi-implicit method (A-stable)", font="Times New Roman")),
    xlabel=L"t",
)
ax_ek0_ioup = Axis(
    fig[1, 3]; yticks=[0, 1], xticks=[0, tspan[2]],
    yticklabelsvisible=false,
    title=rich(rich("c. ", font="Times New Roman Bold"),
        rich("Exponential integrator (L-stable)", font="Times New Roman")),
    xlabel=L"t",
)

dense_ts = tspan[1]:0.01:tspan[2]
vecvec2mat(vv) = hcat(vv...)

for (i, (ax, sol)) in
    enumerate(((ax_ek0, sol_ek0), (ax_ek1, sol_ek1), (ax_ek0_ioup, sol_ek0_ioup)))
    series!(ax, ref_sol.t, vecvec2mat(ref_sol.u), solid_color=:black, linestyle=:dash)
    xlims!(ax, tspan)
    ylims!(ax, -0.1, 1.2)
    us = sol.pu
    means = sol.u |> vecvec2mat
    stddevs = sqrt.(vecvec2mat(diag.(us.Σ)))
    for j in 1:2
        scatterlines!(ax, sol.t, vecvec2mat(sol.u)[j, :], color=COLORS[i])
        fill_between!(
            ax,
            sol.t,
            means[j, :] - 1.96stddevs[j, :],
            means[j, :] + 1.96stddevs[j, :],
            color=(COLORS[i], 0.25),
        )
    end
end

text!(ax_ek0, 0, 0, text=L"dt=%$dt_ek0",
    fontsize=8, align=(:left, :bottom), offset=(7, 1.5))
text!(ax_ek1, 0, 0, text=L"dt=%$dt_ek1",
    fontsize=8, align=(:left, :bottom), offset=(10, 1.5))
text!(ax_ek0_ioup, 0, 0, text=L"dt=%$dt_ek1",
    fontsize=8, align=(:left, :bottom), offset=(10, 1.5))

trim!(fig.layout)
filename = joinpath(DIR, "1_stability.pdf")
save(filename, fig, pt_per_unit=1)
@info "Saved figure to $filename"
