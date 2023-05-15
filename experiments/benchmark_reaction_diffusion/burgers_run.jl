using OrdinaryDiffEq, ProbNumDiffEq
using JLD

import BayesExpIntExperiments as BEIE

DIR = "experiments/benchmark_reaction_diffusion"

prob, L = BEIE.prob_burgers()
@info "burgers_run with this size:" length(prob.u0)
prob_appxjac = ODEProblem(ODEFunction(prob.f.f, jac=(J, u, p, t) -> (J .= L)),
                          prob.u0, prob.tspan, prob.p)
ref_sol = solve(prob, RadauIIA5(), abstol=1e-20, reltol=1e-20)
# import Plots
# Plots.plot(ref_sol)
# solve(prob, EK1());

sol = solve(prob, EK1(prior=IWP(3), smooth=false, diffusionmodel=FixedDiffusion()),
    dense=false, adaptive=false, dt=1e-1)

DM = FixedDiffusion()
# DM = DynamicDiffusion()

# Fixed steps:
# dts = 1.0 ./ 10.0 .^ (-1//2:1//2:2)
dts = 1.0 ./ 10.0 .^ (-1//2:1//2:6//2)
abstols = reltols = zero(dts)

FINAL = true
wp_fun(prob, alg; kwargs...) = BEIE.MyWorkPrecision(
    prob, alg, abstols, reltols;
    appxsol=ref_sol,
    timeseries_errors=!FINAL,
    dts=dts,
    verbose=false,
    dense_errors=!FINAL,
    save_everystep=!FINAL,
    dense=!FINAL,
    kwargs...
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

NUS = (
    # 1,
    2,
    # 3,
    # 4,
)

for nu in NUS
    for extrapolation_jacobian in (:Z, :L, :F), correction_jacobian in (:Z, :L, :F)
        if extrapolation_jacobian == :L && correction_jacobian == :L
            # IOUP + EKL is currently already covered by IOUP + EK0
            continue
        end
        if extrapolation_jacobian == :F && correction_jacobian == :L
            # Doing Rosenbrock but operating on an approximat Jacobian doesn't really make sense
            continue
        end
        if extrapolation_jacobian == :L && correction_jacobian != :Z
            # IOUP with global L just with EK0 for now (== EKL)
            continue
        end
        if extrapolation_jacobian == :F && correction_jacobian != :F
            # Rosenbrock just with full EK1 for now
            continue
        end

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

        alg_str = Dict(:Z => "EK0", :L => "EKL", :F => "EK1")[correction_jacobian]
        prior_str =
            Dict(:Z => "IWP($nu)", :L => "IOUP($nu)", :F => "IOUP($nu)+RB")[extrapolation_jacobian]

        str = "$alg_str+$prior_str"
        @info "start $str"
        results[str] = wp_fun(_prob, alg(prior=prior, diffusionmodel=DM, smooth=!FINAL),
                              name=str)
    end
    save(joinpath(DIR, "burgers_results.jld"), "results", results)
end
