#=
Some plots to compare the PN solvers on the reaction diffusion problems

TL;DR:
- at first signt the IOUP seems helpful!
- the standard EK0 fails miserably
- the EK1 is not so bad though
=#
using LinearAlgebra
using OrdinaryDiffEq, ProbNumDiffEq, Plots
import BayesExpIntExperiments as BEIE

N = 16
# prob, L = BEIE.prob_rd_1d_fisher(; N);
prob, L = BEIE.prob_rd_1d_sir(; N);
prob_badjac, _ = BEIE.prob_rd_1d_fisher(; N, fakejac=true);
sol_acc = solve(prob, RadauIIA5(), abstol=1e-10, reltol=1e-10);
NU = 3

############################################################################################
# PN: Adaptive steps
############################################################################################
# sol_ek0_iwp = solve(prob, EK0(prior=IWP(NU))); # diverges
sol_ek1_iwp = solve(prob, EK1(prior=IWP(NU)));
sol_ek05_iwp = solve(prob_badjac, EK1(prior=IWP(NU)));
sol_ek0_ioup = solve(prob, EK0(prior=IOUP(NU, L)));
sol_ek1_ioup = solve(prob, EK1(prior=IOUP(NU, L)));
p1 = plot(xlabel="t", ylabel="error", title="Adaptive steps", yscale=:log10)
function plot_errs!(p, sol; kwargs...)
    err = sol.u - sol_acc.(sol.t)
    plot!(p, sol.t[2:end], map(norm, err[2:end]); marker=:o, kwargs...)
end
# plot_errs!(p1, sol_ek0_iwp, label="EK0 IWP")
plot_errs!(p1, sol_ek1_iwp, label="EK1 IWP")
plot_errs!(p1, sol_ek05_iwp, label="EK0.5 IWP")
plot_errs!(p1, sol_ek0_ioup, label="EK0 IOUP")
plot_errs!(p1, sol_ek1_ioup, label="EK1 IOUP")

############################################################################################
# PN: Fixed steps
############################################################################################
# EK0 only works with ridicoulously small steps like dt=0.0001
# sol_ek0_iwp = solve(prob, EK0(prior=IWP(NU), diffusionmodel=FixedDiffusion()), adaptive=false, dt=0.1); # diverges
sol_ek1_iwp = solve(
    prob,
    EK1(prior=IWP(NU), diffusionmodel=FixedDiffusion()),
    adaptive=false,
    dt=0.1,
);
sol_ek05_iwp = solve(
    prob_badjac,
    EK1(prior=IWP(NU), diffusionmodel=FixedDiffusion()),
    adaptive=false,
    dt=0.1,
);
sol_ek0_ioup = solve(
    prob,
    EK0(prior=IOUP(NU, L), diffusionmodel=FixedDiffusion()),
    adaptive=false,
    dt=0.1,
);
sol_ek1_ioup = solve(
    prob,
    EK1(prior=IOUP(NU, L), diffusionmodel=FixedDiffusion()),
    adaptive=false,
    dt=0.1,
);
p2 = plot(xlabel="t", ylabel="error", title="Fixed steps", yscale=:log10)
# plot_errs!(p2, sol_ek0_iwp, label="EK0 IWP")
plot_errs!(p2, sol_ek1_iwp, label="EK1 IWP")
plot_errs!(p2, sol_ek05_iwp, label="EK0.5 IWP")
plot_errs!(p2, sol_ek0_ioup, label="EK0 IOUP")
plot_errs!(p2, sol_ek1_ioup, label="EK1 IOUP")

plot(p1, p2, layout=(2, 1))
