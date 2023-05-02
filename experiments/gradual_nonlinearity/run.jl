#=
Goal of the experiment: Show that the IOUP is more useful as the model is more linear
=#
using LinearAlgebra
using OrdinaryDiffEq, ProbNumDiffEq, DiffEqDevTools
using JLD

import BayesExpIntExperiments as BEIE

DIR = @__DIR__

function f!(du, u, p, t)
    @. du = p.a * u + p.b * u^2
end
u0 = [1.0]
tspan = (0.0, 10.0)
p = (a=-1, b=0.0001)
prob = ODEProblem(f!, u0, tspan, p)
prob_badjac = ODEProblem(ODEFunction(f!, jac=(J, u, p, t) -> (J .= p.a)), u0, tspan, p)
ref_sol = solve(prob, RadauIIA5(), abstol=1e-20, reltol=1e-20);
# plot(ref_sol)

# plot(solve(prob, RadauIIA5(), abstol=1e-8, reltol=1e-8, p=(a=-1, b=0.001)))
# plot!(solve(prob, RadauIIA5(), abstol=1e-8, reltol=1e-8, p=(a=-1, b=0.01)))
# plot!(solve(prob, RadauIIA5(), abstol=1e-8, reltol=1e-8, p=(a=-1, b=0.1)))
# plot!(solve(prob, RadauIIA5(), abstol=1e-8, reltol=1e-8, p=(a=-1, b=0.9)))
# plot!(solve(prob, RadauIIA5(), abstol=1e-8, reltol=1e-8, p=(a=-1, b=0.99)))

# PN
abstols = 1.0 ./ 10.0 .^ (4:13)
reltols = 1.0 ./ 10.0 .^ (1:10)

function get_wps(b)
    p = (a=-1, b=b)
    prob = ODEProblem(f!, u0, tspan, p)
    prob_badjac = ODEProblem(ODEFunction(f!, jac=(J, u, p, t) -> (J .= p.a)), u0, tspan, p)
    ref_sol = solve(prob, RadauIIA5(), abstol=1e-20, reltol=1e-20)

    wps = Dict()

    wp_fun(prob, alg) = BEIE.MyWorkPrecision(
        prob, alg, abstols, reltols;
        appxsol=ref_sol,
        timeseries_errors=true,
    )

    wps["EK0+IWP(3)"] = wp_fun(prob, EK0(prior=IWP(3)))
    wps["EK1+IWP(3)"] = wp_fun(prob, EK1(prior=IWP(3)))
    wps["EK0.5+IWP(3)"] = wp_fun(prob_badjac, EK1(prior=IWP(3)))
    wps["EK0+IOUP(3)"] = wp_fun(prob, EK0(prior=IOUP(3, prob.p.a)))
    wps["EK1+IOUP(3)"] = wp_fun(prob, EK1(prior=IOUP(3, prob.p.a)))

    return wps
end

wpss = Dict(
    0.001 => get_wps(0.001),
    0.01 => get_wps(0.01),
    0.1 => get_wps(0.1),
    0.9 => get_wps(0.9),
)

save(joinpath(DIR, "workprecisiondata.jld"), "wpss", wpss)
