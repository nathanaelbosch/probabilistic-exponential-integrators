#=
Re-write of the reaction-diffusion problem from the other file, which makes the linear
part more explicit (and correct actually)

Playing around gave me the following insights:
- In the EK1 case there are not really any benefits it seems
- For _very_ high step sizes, the exponential solver can sometimes be the only useful one.
  The others just have super high error - even the EK1! This needs small NU.
=#
using ProbNumDiffEq, BenchmarkTools, ProfileView, Plots, LinearAlgebra, SciMLBase
using SparseArrays, Distributions
using OrdinaryDiffEq

DT = 0.02
DX = 0.1
DIFFUSION = 0.01

# This generates a finite-difference-discretized laplace operator
# that can be applied to a vector my matrix-vector multiplication
function laplace_1d(state, dx)
    state_size = length(state)
    spdiagm(
        -1 => fill(one(eltype(state)), state_size - 1),
        0 => fill(-2.0 * one(eltype(state)), state_size),
        1 => fill(one(eltype(state)), state_size - 1),
    ) / (dx^2)
end

function linear_part(state, dx; diffusion)
    # basically laplace + neumann borders + diffusion
    Δ = laplace_1d(state, dx)
    Δ[1, 1] += Δ[1, 2]
    Δ[end, end] += Δ[end, end-1]
    @. Δ *= diffusion
    return Δ
end

u0 = [0.01, 0.5, 0.8, 0.2]
Δ = linear_part(u0, DX; diffusion=DIFFUSION)
tspan = (0.0, 10.0)

function f!(du, u, p, t)
    mul!(du, Δ, u)
    @. du += u * (1 - u)
end
prob = ODEProblem{true,SciMLBase.FullSpecialize()}(f!, u0, tspan)
# ref = solve(remake(prob, u0=big.(u0)), RadauIIA5(), abstol=1e-30, reltol=1e-30);
# ref = solve(prob, RadauIIA5(), abstol=1e-20, reltol=1e-20);

NU = 3
sol_iwp = solve(prob, EK0(prior=IWP(NU)), adaptive=false, dt=DT);
# @btime sol_iwp = solve(prob, EK0(prior=IWP(NU)), adaptive=false, dt=DT);
err_iwp = sum(x -> x'x, ref.(sol_iwp.t) .- sol_iwp.u)

sol_iwp2 = solve(prob, EK1(prior=IWP(NU)), adaptive=false, dt=DT);
# @btime sol_iwp2 = solve(prob, EK1(prior=IWP(NU)), adaptive=false, dt=DT);
err_iwp2 = sum(x -> x'x, ref.(sol_iwp2.t) .- sol_iwp2.u)

sol_ioup = solve(prob, EK0(prior=IOUP(NU, Δ)), adaptive=false, dt=DT);
# @btime sol_ioup = solve(prob, EK0(prior=IOUP(NU, Δ)), adaptive=false, dt=DT);
err_ioup = sum(x -> x'x, ref.(sol_ioup.t) .- sol_ioup.u)

sol_ioup2 = solve(prob, EK1(prior=IOUP(NU, Δ)), adaptive=false, dt=DT);
# @btime sol_ioup2 = solve(prob, EK1(prior=IOUP(NU, Δ)), adaptive=false, dt=DT);
err_ioup2 = sum(x -> x'x, ref.(sol_ioup2.t) .- sol_ioup2.u)
