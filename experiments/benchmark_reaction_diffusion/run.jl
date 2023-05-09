using OrdinaryDiffEq, ProbNumDiffEq
using JLD

import BayesExpIntExperiments as BEIE

DIR = "experiments/benchmark_reaction_diffusion"

prob, L = BEIE.prob_rd_1d_fisher()
prob_appxjac = ODEProblem(ODEFunction(prob.f.f, jac=(J, u, p, t) -> (J .= L)),
                          prob.u0, prob.tspan, prob.p)
ref_sol = solve(prob, RadauIIA5(), abstol=1e-20, reltol=1e-20)
# import Plots
# Plots.plot(ref_sol)
# solve(prob, EK1(prior=IOUP(3, L)), adaptive=false, dt=1e-1);
solve(prob, EK1(prior=IOUP(3, L)), adaptive=false, dt=1e-1);
# using ProfileView
# @profview solve(prob, EK1(prior=IOUP(3, L)), adaptive=false, dt=1e-1);
# @profview for _ in 1:100 solve(prob, EK1(prior=IOUP(3, L)), adaptive=false, dt=1e-1) end

# Visualize this a bit
import Plots
anim = Plots.@animate for t in 0.0:0.1:prob.tspan[2]
    Plots.plot(
        ref_sol(t),
        ylim=(0, 1),
        label="",
        title="t = $t",
        ylabel="u(x, t)",
        xlabel="x",
    )
end
Plots.gif(anim, "1dreactiondiffusion.gif", fps=15)

DM = FixedDiffusion()

# Fixed steps:
dts = 1.0 ./ 10.0 .^ (0:1//2:5//2)
abstols = reltols = zero(dts)

# Adaptive steps:
# abstols = 1.0 ./ 10.0 .^ (4:13)
# reltols = 1.0 ./ 10.0 .^ (1:10)
# dts = nothing

wp_fun(prob, alg) = BEIE.MyWorkPrecision(
    prob, alg, abstols, reltols;
    appxsol=ref_sol,
    timeseries_errors=true,
    dense_errors=true,
    dts=dts,
)

results = Dict()
results["Tsit5"] = wp_fun(prob, Tsit5())
@info "Tsit5 done"
results["BS3"] = wp_fun(prob, BS3())
@info "BS3 done"
results["Rosenbrock23"] = wp_fun(prob, Rosenbrock23())
@info "Rosenbrock23 done"
results["Rosenbrock32"] = wp_fun(prob, Rosenbrock32())
@info "Rosenbrock32 done"

NUS = (1, 2, 3, 4)

for nu in NUS, extrapolation_jacobian in (:Z, :L, :F), correction_jacobian in (:Z, :L, :F)

    alg, _prob = if correction_jacobian == :Z
        EK0, prob
    elseif correction_jacobian == :L
        EK1, prob_appxjac
    elseif correction_jacobian == :F
        EK1, prob
    end

    prior = if extrapolation_jacobian == :Z
        IWP(nu)
    elseif extrapolation_jacobian == :L
        IOUP(nu, L)
    elseif extrapolation_jacobian == :F
        IOUP(nu, update_rate_parameter=true)
    end

    alg_str = Dict(:Z=>"EK0", :L=>"EKL", :F=>"EK1")[correction_jacobian]
    prior_str = Dict(:Z=>"IWP($nu)", :L=>"IOUP($nu)", :F=>"IOUP($nu)+RB")[extrapolation_jacobian]

    str = "$alg_str+$prior_str"
    @info "start $str"
    results[str] = wp_fun(prob, alg(prior=prior, diffusionmodel=DM))
    save(joinpath(DIR, "workprecisiondata.jld"), "results", results)
end
