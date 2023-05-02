#=
A quick demo for how to use the new reaciton diffusion definitions by making some gifs
=#
using OrdinaryDiffEq, LinearAlgebra, ForwardDiff, SimpleUnPack, SparseArrays, ProbNumDiffEq
using Plots
import BayesExpIntExperiments as BEIE

############################################################################################
# Fisher / Spruce-Budworm gif
############################################################################################
prob, L = BEIE.prob_rd_1d_fisher(; N=32);
sol_acc = solve(prob, RadauIIA5(), abstol=1e-10, reltol=1e-10, saveat=1);

# Create and save the gif
anim = @animate for t in 0:prob.tspan[2]
    plot(sol_acc(t), ylim=(0, 1), label="", title="t = $t", ylabel="u(x, t)", xlabel="x")
end
gif(anim, "1dreactiondiffusion.gif", fps=30)

############################################################################################
# SIR gif
############################################################################################
prob, _ = BEIE.prob_rd_1d_sir(; N=64);
sol_acc = solve(prob, Vern9(), abstol=1e-10, reltol=1e-10);

# Create and save the gif
anim = @animate for t in 0:prob.tspan[2]
    plot(
        sol_acc(t),
        ylim=(0, prob.p.reaction!.P),
        label="",
        title="t = $t",
        ylabel="u(x, t)",
        xlabel="x",
    )
end
gif(anim, "1dreactiondiffusion.gif", fps=30)
