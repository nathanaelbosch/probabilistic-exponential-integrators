#=
Goal of the experiment: Show that the IOUP is more useful as the model is more linear
=#
using LinearAlgebra
using OrdinaryDiffEq, ProbNumDiffEq, Plots
using DiffEqDevTools
import BayesExpIntExperiments as BEIE
using LaTeXStrings

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

function simple_plot_wps(wps)
    p = plot()
    for (name, kwargs) in (
        "EK0+IWP(3)" => (color=:red, linestyle=:solid),
        "EK1+IWP(3)" => (color=:red, linestyle=:dash),
        "EK0.5+IWP(3)" => (color=:red, linestyle=:dot),
        "EK0+IOUP(3)" => (color=:blue, linestyle=:solid),
        "EK1+IOUP(3)" => (color=:blue, linestyle=:dash),
    )
        wp = wps[name]
        plot!(
            [r[:nsteps] for r in wp],
            [r[:l2] for r in wp];
            label=name,
            xlabel="number of steps",
            ylabel="L2 error",
            kwargs...,
        )
    end
    plot!(yscale=:log10, xscale=:log10)
    return p
end

wpss = Dict(
    0.001 => get_wps(0.001),
    0.01 => get_wps(0.01),
    0.1 => get_wps(0.1),
    0.9 => get_wps(0.9),
)

p1 = simple_plot_wps(wpss[0.001])
plot!(title=L"\dot{y} = -y + 0.001 y^2", legend=false)
p2 = simple_plot_wps(wpss[0.01])
plot!(title=L"\dot{y} = -y + 0.01 y^2", legend=false, ylabel="", yticks=false)
p3 = simple_plot_wps(wpss[0.1])
plot!(title=L"\dot{y} = -y + 0.1 y^2", legend=false, ylabel="", yticks=false)
p4 = simple_plot_wps(wpss[0.9])
plot!(title=L"\dot{y} = -y + 0.9 y^2", ylabel="", yticks=false)
plot(p1, p2, p3, p4, layout=(1, 4), size=(1200, 300), link=:y)
