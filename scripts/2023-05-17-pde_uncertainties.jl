using ProbNumDiffEq, LinearAlgebra
import Plots

import BayesExpIntExperiments as BEIE

prob, L = BEIE.prob_burgers();
sol = solve(
    prob,
    EK1(prior=IOUP(2, update_rate_parameter=true), diffusionmodel=FixedDiffusion(), smooth=false),
    dense=false,
    adaptive=false,
    dt=5e-2,
);
# sol = solve(prob, EK1(prior=IOUP(3, update_rate_parameter=true)), adaptive=false, dt=1e-2);

vals = hcat(sol.u...)
stds = sqrt.(hcat(diag.(sol.pu.Σ)...))

Plots.heatmap(
    1:length(sol.u[1]),
    sol.t,
    stds',
    # colormap=:balance,
    # clims=(-0.5, 0.5),
)
