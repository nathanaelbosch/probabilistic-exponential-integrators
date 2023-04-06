#=
...
=#
using ProbNumDiffEq, BenchmarkTools, ProfileView, Plots, LinearAlgebra, SciMLBase
using SparseArrays, Distributions
using OrdinaryDiffEq

dt = 0.02
tspan = (0.0, 10.0)


# This generates a finite-difference-discretized laplace operator
# that can be applied to a vector my matrix-vector multiplication
function laplace_1d(state, dx)
    state_size = length(state)
    spdiagm(
        -1 => fill(one(eltype(state)), state_size - 1),
        0 => fill(-2.0 * one(eltype(state)), state_size),
        1 => fill(one(eltype(state)), state_size - 1)
    ) / (dx^2)
end


# Startpoints for 4 interacting logistic ODEs (Spruce-Budworm model)
u0 = [0.01, 0.5, 0.8, 0.2]

# The non-linear part
function logistic_f(u, p ,t)
    return u .* (one(eltype(u)) .- u)
end


# The linear part
function diffuse_f(u, p, t)
    diffusion_constant, dx = p
    # Add ghost nodes
    padded_u = zeros(eltype(u), length(u) + 2)
    # This encode zero-Neumann boundary conditions (i.e. zero flux) >>
    padded_u[begin] = u[begin]
    padded_u[end] = u[end]
    # <<
    padded_u[2:end-1] .= u
    Δ = laplace_1d(padded_u, dx)
    padded_u = diffusion_constant * Δ * padded_u
    return padded_u[2:end-1]
end

# Now kiss
function f!(du, u, p, t)
    du .= logistic_f(u, p, t) + diffuse_f(u, p, t)
end


NU = 5
DX = 0.1

prob_with_diffusion = ODEProblem{true,SciMLBase.FullSpecialize()}(f!, u0, tspan, [0.01, DX])
prob_without_diffusion = ODEProblem{true,SciMLBase.FullSpecialize()}(f!, u0, tspan, [0.0, DX])

# Solve with Radau
radau_sol_with_diffusion = solve(prob_with_diffusion, RadauIIA3(), adaptive=false, dt=dt)
radau_sol_without_diffusion = solve(prob_without_diffusion, RadauIIA3(), adaptive=false, dt=dt)

plot(
    plot(radau_sol_with_diffusion, title="diffusion = 0.01", label=hcat(["x=$i" for i in 1:length(u0)]...)),
    plot(radau_sol_without_diffusion, title="diffusion = 0.0", label=hcat(["x=$i" for i in 1:length(u0)]...)),
)

# Solve with EK1 (IWP)
EK1_sol_with_diffusion = solve(prob_with_diffusion, EK1(), adaptive=false, dt=dt)
EK1_sol_without_diffusion = solve(prob_without_diffusion, EK1(), adaptive=false, dt=dt)

plot(
    plot(EK1_sol_with_diffusion, title="diffusion = 0.01", label=hcat(["x=$i" for i in 1:length(u0)]...)),
    plot(EK1_sol_without_diffusion, title="diffusion = 0.0", label=hcat(["x=$i" for i in 1:length(u0)]...)),
)


# Solve with EK1 (IOUP)
EK1_ioup_sol_with_diffusion = solve(prob_with_diffusion, EK1(prior=IOUP(NU, laplace_1d(u0))), adaptive=false, dt=dt)
EK1_ioup_sol_without_diffusion = solve(prob_without_diffusion, EK1(prior=IOUP(NU, laplace_1d(u0))), adaptive=false, dt=dt)

plot(
    plot(EK1_ioup_sol_with_diffusion, title="diffusion = 0.01", label=hcat(["x=$i" for i in 1:length(u0)]...)),
    plot(EK1_ioup_sol_without_diffusion, title="diffusion = 0.0", label=hcat(["x=$i" for i in 1:length(u0)]...)),
)

