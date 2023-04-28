#=
Goal: Use the `SplitODEProblem` type
=#
using ProbNumDiffEq, LinearAlgebra, Plots

α = 5e-1
A = [1 1; 1 1]
f2!(du, u, p, t) = @. du = u * (1 - u)
f!(du, u, p, t) = begin
    f2!(du, u, p, t)
    mul!(du, A, u, 1, 1)
end
u0 = [-0.5; 1.5]
tspan = (0.0, 3.0)

prob1 = ODEProblem(f!, u0, tspan)
sol = solve(prob1, EK1(prior=IOUP(3, update_rate_parameter=true)));
sol.destats
plot(sol)

prob2 = SplitODEProblem(MatrixOperator(A), f2!, u0, tspan)
sol = solve(prob2, EK0(initialization=ClassicSolverInit()));
sol = solve(prob2, EK1(initialization=ClassicSolverInit()));
sol = solve(prob2, EK1(prior=IOUP(3, update_rate_parameter=true)));
sol.destats
plot(sol)
