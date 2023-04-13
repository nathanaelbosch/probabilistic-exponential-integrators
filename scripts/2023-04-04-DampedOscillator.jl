#=
Simple linear ODE: A damped oscillator

TL;DR:
- The IOUP perfectly solves this problem, and even gets near-zero covariances
- The IOUP with only an approximate Jacobian needs much fewer steps than the IWP
- The IWP is still a bit faster than the one with approximate Jacobian, due to exp
=#
using ProbNumDiffEq, BenchmarkTools, ProfileView, Plots, LinearAlgebra, SciMLBase

α = 5e-1
A_undamped = [0 -2π; 2π 0]
A = A_undamped - Diagonal([α, α])
f!(du, u, p, t) = mul!(du, A, u)
u0 = [1.0; 1.0]
tspan = (0.0, 2.0)

prob = ODEProblem{true,SciMLBase.FullSpecialize()}(f!, u0, tspan)

NU = 3
SMOOTH = false
sol_ek0 = solve(prob, EK0(prior=IWP(NU)));
p1 = plot(sol_ek0, label="IWP", color=[1 2])
scatter!(p1, sol_ek0, label="", denseplot=false, color=[1 2])

sol_ek1 = solve(prob, EK1(prior=IWP(NU)));
p2 = plot(sol_ek1, label="IWP", color=[1 2])
scatter!(p2, sol_ek1, label="", denseplot=false, color=[1 2])

sol2 = solve(prob, EK0(prior=IOUP(NU, A)));
p3 = plot(sol2, label="IOUP exact", color=[1 2])
scatter!(p3, sol2, label="", denseplot=false, color=[1 2])

sol3 = solve(prob, EK0(prior=IOUP(NU, A_undamped)));
p4 = plot(sol3, label="IOUP appx", color=[1 2])
scatter!(p4, sol3, label="", denseplot=false, color=[1 2])

plot(p1, p2, p3, p4, layout=(4, 1))

ref = solve(prob, EK0(), abstol=1e-14, reltol=1e-14);
mse(u) = sum(abs2, u - ref.u[end])
@info "errors" iwp_ek0 = mse(sol_ek0.u[end]) iwp_ek1 = mse(sol_ek1.u[end]) exact =
    mse(sol2.u[end]) approx = mse(sol3.u[end])

@info "destats" iwp_ek0 = sol_ek0.destats iwp_ek1 = sol_ek1.destats exact = sol2.destats approx =
    sol3.destats

t_iwp_ek0 = @elapsed solve(prob, EK0(prior=IWP(NU)));
t_iwp_ek1 = @elapsed solve(prob, EK1(prior=IWP(NU)));
t_exact = @elapsed solve(prob, EK0(prior=IOUP(NU, A)));
t_approx = @elapsed solve(prob, EK0(prior=IOUP(NU, A_undamped)));
@info "runtimes" t_iwp_ek0 t_iwp_ek1 t_exact t_approx

###########################################################################################
# Profiling
###########################################################################################
# @profview for _ in 1:1000
#     solve(prob, EK0(prior=IWP(NU)))
# end
# @profview for _ in 1:1000
#     solve(prob, EK0(prior=IOUP(NU, A_undamped)))
# end

###########################################################################################
# What about fixed steps?
###########################################################################################
DT = 1e-2
sol_ek0 = solve(prob, EK0(prior=IWP(NU)), adaptive=false, dt=DT);
sol_ek1 = solve(prob, EK1(prior=IWP(NU)), adaptive=false, dt=DT);
sol2 = solve(prob, EK0(prior=IOUP(NU, A)), adaptive=false, dt=DT);
sol3 = solve(prob, EK0(prior=IOUP(NU, A_undamped)), adaptive=false, dt=DT);

# p1 = plot(sol_ek0, label="IWP", color=[1 2])
# scatter!(p1, sol_ek0, label="", denseplot=false, color=[1 2])
# p2 = plot(sol2, label="IOUP exact", color=[1 2])
# scatter!(p2, sol2, label="", denseplot=false, color=[1 2])
# p3 = plot(sol3, label="IOUP appx", color=[1 2])
# scatter!(p3, sol3, label="", denseplot=false, color=[1 2])
# plot(p1, p2, p3, layout=(3, 1))

@info "errors" iwp_ek0 = mse(sol_ek0.u[end]) iwp_ek1 = mse(sol_ek1.u[end]) exact =
    mse(sol2.u[end]) approx = mse(sol3.u[end])
# IOUP with approx jacobian has lower error

t_iwp_ek0 = @elapsed solve(prob, EK0(prior=IWP(NU)), adaptive=false, dt=DT);
t_iwp_ek1 = @elapsed solve(prob, EK1(prior=IWP(NU)), adaptive=false, dt=DT);
t_exact = @elapsed solve(prob, EK0(prior=IOUP(NU, A)), adaptive=false, dt=DT);
t_approx = @elapsed solve(prob, EK0(prior=IOUP(NU, A_undamped)), adaptive=false, dt=DT);
@info "runtimes" t_iwp_ek0 t_iwp_ek1 t_exact t_approx

# IOUP has higher runtime
# @profview for _ in 1:1000
#     solve(prob, EK0(prior=IWP(NU)), adaptive=false, dt=1e-1)
# end
# @profview for _ in 1:1000
#     solve(prob, EK0(prior=IOUP(NU, A_undamped)), adaptive=false, dt=1e-1)
# end
# profiling is a bit unclear here; in any case there is still some stuff to do
