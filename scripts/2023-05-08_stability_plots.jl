#=
This script reproduces the plots from
https://math.stackexchange.com/questions/1466978/advantage-of-l-stability-compared-to-a-stability
=#
using ProbNumDiffEq, DiffEqCallbacks, OrdinaryDiffEq
import Plots
Plots.theme(:default;
    markersize=1,
    markerstrokewidth=0.1,
)

function f(du, u, p, t)
    du[1] = -0.5u[1] + 20u[2]
    du[2] = -20u[2]
end
u0 = [0.0, 1.0]
tspan = (0.0, 3.0)
prob = ODEProblem(f, u0, tspan)
L = [-0.5 20; 0 -20]

ref_sol = solve(prob, ImplicitEuler(), abstol=1e-10, reltol=1e-10)

sol_ek0 = solve(prob, EK0());
sol_ek0.destats

sol_ek1 = solve(prob, EK1());
sol_ek1.destats

Plots.plot(sol_ek0, denseplot=false, marker=:o, markersize=1, markerstrokewidth=0.1)
Plots.plot!(sol_ek1, denseplot=false, marker=:o, markersize=1, markerstrokewidth=0.1)

############################################################################################
# Fig 1
dt_small, dt_large = 0.01, 0.1
order = 1
diffusionmodel = FixedDiffusion()
callback = PresetTimeCallback([0.5], integ -> SciMLBase.set_proposed_dt!(integ, dt_large))

sol_ek0 = solve(prob, EK0(; order, diffusionmodel); adaptive=false, dt=dt_small, callback);
sol_ek1 = solve(prob, EK1(; order, diffusionmodel); adaptive=false, dt=dt_small, callback);
L = [-0.5 20; 0 -20]
sol_ek0_ioup = solve(prob, EK0(; prior=IOUP(order, L), diffusionmodel);
    adaptive=false, dt=dt_small, callback);

Plots.plot(ref_sol, ylims=(0, 1), color=:black, linestyle=:dash, label="ref")
Plots.plot!(sol_ek0, denseplot=false, marker=:o, ribbon=0, label="EK0+IWP", color=1)
Plots.plot!(sol_ek1, denseplot=false, marker=:o, ribbon=0, label="EK1+IWP", color=2)
Plots.plot!(sol_ek0_ioup, denseplot=false, marker=:o, ribbon=0, label="EK0+IOUP", color=3)

############################################################################################
# Fig 2
sol_ek0 = solve(prob, EK0(; order, diffusionmodel); adaptive=false, dt=dt_large);
sol_ek1 = solve(prob, EK1(; order, diffusionmodel); adaptive=false, dt=dt_large);
sol_ek0_ioup = solve(prob, EK0(; prior=IOUP(order, L), diffusionmodel);
    adaptive=false, dt=dt_large);

Plots.plot(ref_sol, ylims=(0, 1), color=:black, linestyle=:dash, label="ref")
Plots.plot!(sol_ek0, denseplot=false, marker=:o, ribbon=0, label="EK0+IWP", color=1)
Plots.plot!(sol_ek1, denseplot=false, marker=:o, ribbon=0, label="EK1+IWP", color=2)
Plots.plot!(sol_ek0_ioup, denseplot=false, marker=:o, ribbon=0, label="EK0+IOUP", color=3)
