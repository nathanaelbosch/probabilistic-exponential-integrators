using LinearAlgebra
using OrdinaryDiffEq, ProbNumDiffEq, DiffEqDevTools, ForwardDiff
using JLD
using CairoMakie
using TuePlots
using LaTeXStrings
using DataStructures

import BayesExpIntExperiments as BEIE

function fitzhughnagumo(du, u, p, t)
    v = u[1]
    w = u[2]
    a, b, τinv, l = p
    du[1] = v - v^3 / 3 - w + l
    du[2] = τinv * (v + a - b * w)
end
u0 = [1.0, 1.0]
tspan = (0.0, 50.0)
p = [0.7, 0.8, 1 / 12.5, 0.5]
prob = ODEProblem(fitzhughnagumo, u0, tspan, p)
sol = solve(prob, Tsit5())

# Solution plot
plot_ts = tspan[1]:0.1:tspan[2]
vecvec2mat(x) = hcat(x...)

C1, C2 = Makie.wong_colors()[1:2]
NU = 1
κ = 1e-2
# κ = 1.0

fig = Figure(resolution=(600, 400))
ax = Axis(fig[1, 1], xlabel="t", ylabel="u")
series!(plot_ts, vecvec2mat(sol.(plot_ts)), solid_color=:black, linestyle=:dash)

# Plot the prior
integ = init(prob, EK1(prior=IWP(NU), diffusionmodel=FixedDiffusion(κ, false)))
push!(integ.sol.diffusions, integ.cache.default_diffusion)
push!(integ.sol.x_smooth, integ.sol.x_filt[1])
function plot_pn_sol!(ax, us, ts; name, color, kwargs...)
    means = us.μ |> vecvec2mat
    stds = sqrt.(vecvec2mat(diag.(us.Σ)))
    series!(ts, means; solid_color=color, name)
    for i in 1:length(u0)
        fill_between!(ax, ts, means[i, :] - 2 * stds[i, :], means[i, :] + 2 * stds[i, :],
            color=(color, 0.1))
    end
end
plot_pn_sol!(ax, integ.sol(plot_ts).u, plot_ts; name="IWP", color=C1)

L = ForwardDiff.jacobian(u -> (du = copy(u); fitzhughnagumo(du, u, p, 0.0); du), u0)
integ = init(prob, EK1(prior=IOUP(NU, L), diffusionmodel=FixedDiffusion(κ, false)))
push!(integ.sol.diffusions, integ.cache.default_diffusion)
push!(integ.sol.x_smooth, integ.sol.x_filt[1])
plot_pn_sol!(ax, integ.sol(plot_ts).u, plot_ts; name="IOUP", color=C2)

# Plot the correction
ax2 = Axis(fig[1, 2], xlabel="t", ylabel="u")
series!(plot_ts, vecvec2mat(sol.(plot_ts)), solid_color=:black, linestyle=:dash)

function get_t_est(alg, t)
    integ = init(remake(prob, tspan=(prob.tspan[1], t)), alg, adaptive=false, dt=2t)
    solve!(integ)
    return integ.sol.pu[end]
end

iwp_filtests = StructArray(
    [
    get_t_est(EK1(prior=IWP(NU), diffusionmodel=FixedDiffusion(κ, false)), t)
    for t in plot_ts[2:end]
])
plot_pn_sol!(ax2, iwp_filtests, plot_ts[2:end]; color=C1, name="IWP")
ioup_filtests = StructArray(
    [
    get_t_est(EK1(prior=IOUP(NU, L), diffusionmodel=FixedDiffusion(κ, false)), t)
    for t in plot_ts[2:end]
])
plot_pn_sol!(ax2, ioup_filtests, plot_ts[2:end]; color=C2, name="IOUP")

ylims!(ax, -5, 5)
ylims!(ax2, -5, 5)

save("fig.pdf", fig, px_per_unit=1)
