"""
One-dimensional laplace operator, discretized with finite differences.
Includes boundary conditions
"""
function laplace_1d(x; dx, boundary_condition="zero-neumann")
    lap = zeros(eltype(x), length(x))
    lap[2:end-1] = (x[1:end-2] .+ x[3:end] .- 2x[2:end-1]) ./ dx^2
    if boundary_condition == "zero-neumann"
        lap[1] = (x[1] + x[2] - 2x[1]) ./ dx^2
        lap[end] = (x[end] + x[end-1] - 2x[end]) ./ dx^2
    elseif boundary_condition == "zero-dirichlet"
        lap[1] = (0 + x[2] - 2x[1]) ./ dx^2
        lap[end] = (0 + x[end-1] - 2x[end]) ./ dx^2
    elseif boundary_condition == "periodic"
        lap[1] = (x[end] + x[2] - 2x[1]) ./ dx^2
        lap[end] = (x[1] + x[end-1] - 2x[end]) ./ dx^2
    elseif boundary_conditions == "none"
        # nothing to do
    else
        error("Boundary $boundary_condition is not known or implemented.")
    end
    return lap
end

raw"""
One-dimensional partial differential equation of the form
```math
\partial_t u = D \partial_x^2 u + f(u),
```
with `u: \mathbb{R}^d \times \mathbb{R} \to \mathbb{R}^d`,
diffusion parameter `D \in \mathbb{R}`,
and reaction term `f: \mathbb{R}^d \to \mathbb{R}^d`.

In code, the reaction term, diffusion parameter, and discretized laplacian are all passed
to the vector field through the parameter variable `p`.
"""
function reaction_diffusion_vector_field_1d(du, u, p, t)
    @unpack reaction!, diffusion_parameter, ∇² = p
    reaction!(du, u) # writes the reaction term into du
    mul!(du, ∇², u, diffusion_parameter, 1) # diffuses with ∇² and the diffusion parameter
end

function prob_rd_1d(
    ; u0, reaction!,
    dx=0.02,
    diffusion=0.01,
    tspan=(0.0, 3.0),
    boundary_condition="zero-neumann",
    fakejac=false, # to overwrite f.jac with just the linear part
    n_components=1,
)
    @assert length(u0) % n_components == 0
    ∇² =
        ForwardDiff.jacobian(
            x -> laplace_1d(x; dx, boundary_condition),
            u0[1:length(u0)÷n_components],
        ) |> sparse
    ∇² = kron(I(n_components), ∇²)
    p = (; reaction!, ∇²=∇², diffusion_parameter=diffusion)

    L = diffusion * ∇² |> Matrix

    f = if fakejac
        ODEFunction(reaction_diffusion_vector_field_1d, jac=(J, u, p, t) -> (J .= L))
    else
        reaction_diffusion_vector_field_1d
    end

    prob = ODEProblem{true,SciMLBase.FullSpecialize()}(f, u0, tspan, p)

    return prob, L
end

prob_rd_1d_fisher(; N=100, Omega=(0, 1), tspan=(0, 2), diffusion=0.25, kwargs...) = begin
    xs = range(Omega[1], Omega[2], length=N)
    dx = (Omega[2] - Omega[1]) / N
    u0 = @. 1 - (exp(xs * 30 - 10) / (1 + exp(xs * 30 - 10)))
    reaction!(du, u) = (@. du = u * (1 - u))
    return prob_rd_1d(; u0, reaction!, tspan, diffusion, dx, kwargs...)
end

function uux!(du, u; dx, boundary_condition="zero-dirichlet")
    du[2:end-1] = @. (u[3:end] .^ 2 - u[1:end-2] .^ 2) / (4 * dx)
    if boundary_condition == "zero-dirichlet"
        du[1] = u[2] .^ 2 / (4 * dx)
        du[end] = -u[end-1] .^ 2 / (4 * dx)
    else
        error("Boundary $boundary_condition is not known or implemented.")
    end
    return nothing
end
function prob_burgers(; N=250, Omega=(0, 1), tspan=(0.0, 1.0), kwargs...)
    # u_t = -u u_x + u_xx
    dx = (Omega[2] - Omega[1]) // N
    xs = range(Omega[1], Omega[2], length=N)
    u0 = @. sin(3π * xs)^3 * (1 - xs)^(3 / 2)
    ∇² =
        ForwardDiff.jacobian(
            x -> laplace_1d(x; dx, boundary_condition="zero-dirichlet"), u0) |> sparse
    L = 0.075 * ∇²

    function f(du, u, p, t)
        @unpack dx, L = p
        uux!(du, u; dx)
        mul!(du, L, u, 1, -1)
    end

    prob = ODEProblem{true,SciMLBase.FullSpecialize()}(f, u0, tspan, (dx=dx, L=L))
    return prob, Matrix(L)
end
