using LinearAlgebra
using OrdinaryDiffEq, ProbNumDiffEq, DiffEqDevTools
using JLD

import BayesExpIntExperiments as BEIE

DIR = "experiments/stability"

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
ref_sol = solve(prob, RadauIIA5(), abstol=1e-20, reltol=1e-20)

dts = 1.0 ./ 10.0 .^ (-2:1//4:2)
abstols = reltols = zero(dts)

wp_fun(prob, alg) = BEIE.MyWorkPrecision(
    prob, alg, abstols, reltols;
    appxsol=ref_sol,
    timeseries_errors=true,
    dense_errors=true,
    dts=dts,
)

NU = 2
DM = FixedDiffusion()

wps = Dict()
wps["Tsit5"] = wp_fun(prob, Tsit5())
wps["BS3"] = wp_fun(prob, BS3())
wps["EK0+IWP($NU)"] = wp_fun(prob, EK0(prior=IWP(NU), diffusionmodel=DM))
wps["EK1+IWP($NU)"] = wp_fun(prob, EK1(prior=IWP(NU), diffusionmodel=DM))
wps["EK0+IOUP($NU)"] = wp_fun(prob, EK0(prior=IOUP(NU, update_rate_parameter=true), diffusionmodel=DM))
wps["EK1+IOUP($NU)"] = wp_fun(prob, EK1(prior=IOUP(NU, update_rate_parameter=true), diffusionmodel=DM))

save(joinpath(DIR, "workprecisiondata.jld"), "wps", wps)
