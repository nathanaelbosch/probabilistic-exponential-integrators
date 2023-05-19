using ProbNumDiffEq, OrdinaryDiffEq
using ODEProblemLibrary
import Plots

function vanderpol(du, u, p, t)
    μ = p[1]
    du[1] = u[2]
    du[2] = μ * (1 - u[1]^2) * u[2] - u[1]
end
u0 = [2.0, 0.0]
tspan = (0.0, 500.0)
p = [500.0]

prob = ODEProblem(vanderpol, u0, tspan, p)
refsol = solve(prob, RadauIIA5(), abstol=1e-20, reltol=1e-20)

sol = solve(prob, EK1(prior=IWP(5)));
Plots.plot(sol, ylims=(-3, 3))
sum(abs2, sol[end] - refsol[end])
sol.destats

sol = solve(prob, EK1(prior=IOUP(5, update_rate_parameter=true), smooth=false), dense=false);
Plots.plot!(sol, ylims=(-3, 3))
sum(abs2, sol[end] - refsol[end])
sol.destats

@btime solve(prob, EK1(prior=IWP(5), smooth=false), dense=false);
@btime solve(prob, EK1(prior=IOUP(5, update_rate_parameter=true), smooth=false), dense=false);
