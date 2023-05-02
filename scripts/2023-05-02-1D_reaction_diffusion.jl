#=
A quick demo for how to use the new reaciton diffusion definitions by making some gifs
=#
using OrdinaryDiffEq, Plots
import BayesExpIntExperiments as BEIE

############################################################################################
# Fisher / Spruce-Budworm gif
############################################################################################
prob, L = BEIE.prob_rd_1d_fisher(; N=32);
sol = solve(prob, RadauIIA5(), abstol=1e-10, reltol=1e-10, saveat=1);

# Create and save the gif
anim = @animate for i in 1:length(sol)
    plot(sol[i], ylim=(0, 1), label="", title="t = $(sol.t[i])", ylabel="u(x, t)", xlabel="x")
end
gif(anim, "1dreactiondiffusion.gif", fps=15)

############################################################################################
# SIR gif
############################################################################################
prob, _ = BEIE.prob_rd_1d_sir(; N=64);
sol = solve(prob, Vern9(), abstol=1e-10, reltol=1e-10, saveat=1);

# Create and save the gif
anim = @animate for i in 1:length(sol)
    plot(
        reshape(sol[i], :, 3),
        ylim=(0, prob.p.reaction!.P),
        label="",
        title="t = $(sol.t[i])",
        ylabel="u(x, t)",
        xlabel="x",
    )
end
gif(anim, "1dreactiondiffusion.gif", fps=30)

# Or as a heatmap:
sol_arr = reshape(Array(sol), :, 3, length(sol))
p1 = heatmap(sol_arr[:, 1, :], xticks=false, ylabel="x", title="S")
p2 = heatmap(sol_arr[:, 2, :], xticks=false, ylabel="x", title="I")
p3 = heatmap(sol_arr[:, 3, :], xlabel="t", ylabel="x", title="R")
plot(p1, p2, p3, layout=(3, 1))
