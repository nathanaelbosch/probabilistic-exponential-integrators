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
# p = (a=-1, b=0.0001)
# prob = ODEProblem(f!, u0, tspan, p)
# prob_badjac = ODEProblem(ODEFunction(f!, jac=(J, u, p, t) -> (J .= p.a)), u0, tspan, p)
# ref_sol = solve(prob, RadauIIA5(), abstol=1e-20, reltol=1e-20);
# plot(ref_sol)

# plot(solve(prob, RadauIIA5(), abstol=1e-8, reltol=1e-8, p=(a=-1, b=0.001)))
# plot!(solve(prob, RadauIIA5(), abstol=1e-8, reltol=1e-8, p=(a=-1, b=0.01)))
# plot!(solve(prob, RadauIIA5(), abstol=1e-8, reltol=1e-8, p=(a=-1, b=0.1)))
# plot!(solve(prob, RadauIIA5(), abstol=1e-8, reltol=1e-8, p=(a=-1, b=0.9)))
# plot!(solve(prob, RadauIIA5(), abstol=1e-8, reltol=1e-8, p=(a=-1, b=0.99)))

# PN
NU = 2

function get_wps(b; abstols, reltols, dts, dm)
    p = (a=-1, b=b)
    prob = ODEProblem(f!, u0, tspan, p)
    prob_badjac = ODEProblem(ODEFunction(f!, jac=(J, u, p, t) -> (J .= p.a)), u0, tspan, p)
    ref_sol = solve(remake(prob, u0=big.(prob.u0)), RadauIIA5(), abstol=1e-30, reltol=1e-30)

    wps = Dict()

    wp_fun(prob, alg) = BEIE.MyWorkPrecision(
        prob, alg, abstols, reltols;
        appxsol=ref_sol,
        timeseries_errors=true,
        dense_errors=true,
        dts=dts,
    )

    wps["Tsit5"] = wp_fun(prob, Tsit5())
    wps["BS3"] = wp_fun(prob, BS3())
    wps["EK0+IWP($NU)"] = wp_fun(prob, EK0(prior=IWP(NU), diffusionmodel=dm))
    wps["EK1+IWP($NU)"] = wp_fun(prob, EK1(prior=IWP(NU), diffusionmodel=dm))
    wps["EK0.5+IWP($NU)"] = wp_fun(prob_badjac, EK1(prior=IWP(NU), diffusionmodel=dm))
    wps["EK0+IOUP($NU)"] = wp_fun(prob, EK0(prior=IOUP(NU, prob.p.a), diffusionmodel=dm))
    wps["EK1+IOUP($NU)"] = wp_fun(prob, EK1(prior=IOUP(NU, prob.p.a), diffusionmodel=dm))

    return wps
end

dts = 1.0 ./ 10.0 .^ (-1:1//4:1)
abstols = reltols = zero(dts)
dm = FixedDiffusion()
bs = (1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1 - 1e-1, 1 - 1e-2)
wpss_fixed = Dict((b => get_wps(b; abstols, reltols, dts, dm)) for b in bs)

# dm = DynamicDiffusion()
# abstols = 1.0 ./ 10.0 .^ (4:13)
# reltols = 1.0 ./ 10.0 .^ (1:10)
# wpss_adaptive = Dict(
#     0.00001 => get_wps(0.00001; abstols, reltols, dm, dts=nothing),
#     0.0001 => get_wps(0.0001; abstols, reltols, dm, dts=nothing),
#     0.001 => get_wps(0.001; abstols, reltols, dm, dts=nothing),
#     0.01 => get_wps(0.01; abstols, reltols, dm, dts=nothing),
#     0.1 => get_wps(0.1; abstols, reltols, dm, dts=nothing),
#     0.9 => get_wps(0.9; abstols, reltols, dm, dts=nothing),
# )

save(joinpath(DIR, "workprecisiondata.jld"),
    "wpss_fixed", wpss_fixed,
    # "wpss_adaptive", wpss_adaptive
)
