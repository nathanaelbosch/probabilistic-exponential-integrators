using OrdinaryDiffEq, ProbNumDiffEq
using JLD

import BayesExpIntExperiments as BEIE

DIR = "experiments/benchmark_reaction_diffusion"

prob, L = BEIE.prob_rd_1d_sir(N=4)
ref_sol = solve(prob, RadauIIA5(), abstol=1e-20, reltol=1e-20)
# import Plots
# Plots.plot(ref_sol)
# solve(prob, EK1());

# Visualize this a bit
import Plots
anim = Plots.@animate for t in 0.0:1:prob.tspan[2]
    Plots.plot(
        reshape(ref_sol(t), :, 3),
        ylim=(0, prob.p.reaction!.P),
        label="",
        title="t = $t",
        ylabel="u(x, t)",
        xlabel="x",
    )
end
Plots.gif(anim, "1dreactiondiffusion.gif", fps=15)

DM = FixedDiffusion()
# DM = DynamicDiffusion()

# Fixed steps:
dts = 1.0 ./ 10.0 .^ (-1//2:1//2:2)
abstols = reltols = zero(dts)

# Adaptive steps:
# abstols = 1.0 ./ 10.0 .^ (4:13)
# reltols = 1.0 ./ 10.0 .^ (1:10)
# abstols = 1.0 ./ 10.0 .^ (4:12)
# reltols = 1.0 ./ 10.0 .^ (1:9)
# dts = nothing

SAVE = false
wp_fun(prob, alg) = BEIE.MyWorkPrecision(
    prob, alg, abstols, reltols;
    appxsol=ref_sol,
    timeseries_errors=SAVE,
    dts=dts,
    verbose=true,
    dense_errors=SAVE,
    save_everystep=SAVE,
    dense=SAVE,
)

results = Dict()
# results["Tsit5"] = wp_fun(prob, Tsit5())
# @info "Tsit5 done"
# results["BS3"] = wp_fun(prob, BS3())
# @info "BS3 done"
# results["Rosenbrock23"] = wp_fun(prob, Rosenbrock23())
# @info "Rosenbrock23 done"
# results["Rosenbrock32"] = wp_fun(prob, Rosenbrock32())
# @info "Rosenbrock32 done"

NUS = (
    1,
    2,
    3,
    4,
)
ALGS = Dict(
    :EK0 => EK0,
    :EK1 => EK1,
)
PRIORS = Dict(
    :IWP => (nu -> IWP(nu)),
    :IOUP => (nu -> IOUP(nu, L)),
    :IOUPRB => (nu -> IOUP(nu, update_rate_parameter=true)),
)
for nu in NUS, (alg_sym, alg) in ALGS, (prior_sym, prior) in PRIORS
    # if alg_sym == :EK0 && prior_sym == :IWP
    #     continue
    # end
    str = if prior_sym == :IOUPRB
        "$alg_sym+IOUP($nu)+RB"
    else
        "$alg_sym+$prior_sym($nu)"
    end
    @info "start $str"
    results[str] = wp_fun(prob, alg(prior=prior(nu), diffusionmodel=DM, smooth=SAVE))
end

save(joinpath(DIR, "sir_results.jld"), "results", results)
