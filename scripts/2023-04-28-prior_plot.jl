using ProbNumDiffEq, LinearAlgebra, Random
import ProbNumDiffEq: Gaussian
using CairoMakie, TuePlots, LaTeXStrings, ColorSchemes

import BayesExpIntExperiments: PlotTheme

COLORS = ColorSchemes.tableau_10.colors
# COLORS = ColorSchemes.Set1_3
ALPHA = 0.5

set_theme!(
    merge(
        Theme(
            Axis=(;
                titlesize=8
                # xgridvisible=false,
                # ygridvisible=false,
            ),
            Lines=(;
                linewidth=0.5,
                linestyle=:dash,
            ),
            Series=(
                linewidth=0.5,
                solid_color=:gray,
            ),
            # Label=(
            #     halign=:left,
            #     tellwidth=false,
            #     # tellheight=false,
            #     justification=:left,
            #     padding=(12, 0, 1, 0),
            #     # font="Times New Roman",
            # ),
        ),
        PlotTheme,
        Theme(
            TuePlots.SETTINGS[:NEURIPS];
            font=false,
            fontsize=true,
            figsize=true,
            thinned=true,
            # width_coeff=0.35,
            nrows=1, ncols=3,
            # subplot_height_to_width_ratio=1/TuePlots.GOLDEN_RATIO,
            # subplot_height_to_width_ratio=1,
        ),
    ),
)

α = 0.2
L_undamped = [0 -2π; 2π 0]
L = L_undamped - Diagonal([α, α])
f!(du, u, p, t) = mul!(du, L, u)
u0 = [1.0; 1.0]
tspan = (0.0, 2.0)
prob = ODEProblem{true,SciMLBase.FullSpecialize()}(f!, u0, tspan)

d, q = 2, 1
# κ² = 1e0
κ² = 1e0
D = d * (q + 1)
E0 = [1 0] * ProbNumDiffEq.projection(d, q)(0)

integ = init(prob, EK1(order=q))
x0 = integ.cache.x.μ

function simulate(A, Q, N; randinit=true)
    x = randinit ?
        rand(Gaussian(zeros(D), 0.5I)) :
        copy(x0)
    ys = zeros(N, d)
    ys[1, :] .= E0 * x
    for i in 2:N
        x = rand(Gaussian(A * x, κ² * Symmetric(Q)))
        ys[i, :] .= E0 * x
    end
    return ys
end
function predict(A, Q, N; randinit=true)
    m = randinit ? zeros(D) : copy(x0)
    C = randinit ? 0.5I(D) : zeros(D, D)
    ms = zeros(N, D)
    Cs = zeros(N, D, D)
    ms[1, :] .= m
    Cs[1, :, :] .= C
    for i in 2:N
        m, C = A * m, A * C * A' + κ² * Q
        ms[i, :] .= m
        Cs[i, :, :] .= C
    end
    return ms, Cs
end

T, dt = 5, 1 // 50
ts = 0:dt:T
N = T ÷ dt + 1
M = 10

fig = Figure()
ax_iwp = Axis(
    fig[1, 1];
    yticks=[-3, 0, 3],
    xticks=[0, T],
    # xticklabelsvisible = false,
    title=rich(rich("a. ", font="Times New Roman Bold"),
        rich("Integrated Wiener process", font="Times New Roman")),
    xlabel="t",
    ylabel="y(t)",
)
ax_ioup = Axis(
    fig[1, 2];
    yticks=[-3, 0, 3],
    xticks=[0, T],
    yticklabelsvisible=false,
    title=rich(rich("b. ", font="Times New Roman Bold"),
        rich("Integrated Ornstein-Uhlenbeck", font="Times New Roman")),
    xlabel="t",
)
ax_ioup_init = Axis(
    fig[1, 3]; yticks=[-3, 0, 3], xticks=[0, T],
    yticklabelsvisible=false,
    title=rich(rich("c. ", font="Times New Roman Bold"),
        rich("IOUP + initial value", font="Times New Roman")),
    xlabel="t",
)
# rowgap!(fig.layout, 10)
# colgap!(fig.layout, 10)

A, Q = ProbNumDiffEq.discretize(ProbNumDiffEq.IWP(d, q), float(dt))
Q = Matrix(Q)
ms, Cs = predict(A, Q, N; randinit=true)
lines!(ax_iwp, ts, ms[:, 1], color=COLORS[1])
fill_between!(
    ax_iwp,
    ts,
    ms[:, 1] - 2sqrt.(Cs[:, 1, 1]),
    ms[:, 1] + 2sqrt.(Cs[:, 1, 1]),
    color=(COLORS[1], 0.25),
)
for _ in 1:M
    ys_iwp = simulate(A, Q, N)
    series!(ax_iwp, ts, ys_iwp', solid_color=(COLORS[1], ALPHA))
end

A, Q = ProbNumDiffEq.discretize(ProbNumDiffEq.IOUP(d, q, L), dt)
Q = Matrix(Q)
ms, Cs = predict(A, Q, N; randinit=true)
lines!(ax_ioup, ts, ms[:, 1], color=COLORS[2])
fill_between!(
    ax_ioup,
    ts,
    ms[:, 1] - 2sqrt.(Cs[:, 1, 1]),
    ms[:, 1] + 2sqrt.(Cs[:, 1, 1]),
    color=(COLORS[2], 0.25),
)
for _ in 1:M
    ys_ioup = simulate(A, Q, N)
    series!(ax_ioup, ts, ys_ioup', solid_color=(COLORS[2], ALPHA))
end

ms, Cs = predict(A, Q, N; randinit=false)
lines!(ax_ioup_init, ts, ms[:, 1], color=COLORS[3])
fill_between!(
    ax_ioup_init,
    ts,
    ms[:, 1] - 2sqrt.(Cs[:, 1, 1]),
    ms[:, 1] + 2sqrt.(Cs[:, 1, 1]),
    color=(COLORS[3], 0.25),
)
for _ in 1:M
    ys_ioup = simulate(A, Q, N; randinit=false)
    series!(ax_ioup_init, ts, ys_ioup', solid_color=(COLORS[3], ALPHA))
end

xlims!(ax_iwp, (0, T))
xlims!(ax_ioup, (0, T))
xlims!(ax_ioup_init, (0, T))
ylims!(ax_iwp, (-3, 3))
ylims!(ax_ioup, (-3, 3))
ylims!(ax_ioup_init, (-3, 3))

save("../bayes-exp-int/figures/priors.pdf", fig, pt_per_unit=1)
# save("plot.pdf", fig, pt_per_unit=1)
