#=
Reaction diffusion models
=#

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
    elseif boundary_condition == "periodic"
        lap[1] = (x[end] + x[2] - 2x[1]) ./ dx^2
        lap[end] = (x[1] + x[end-1] - 2x[end]) ./ dx^2
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
    tspan=(0.0, 10.0),
    boundary_condition="zero-neumann",
    fakejac=false, # to overwrite f.jac with just the linear part
    n_components=1,
)
    @assert length(u0) % n_components == 0
    ∇² = ForwardDiff.jacobian(
        x -> laplace_1d(x; dx, boundary_condition),
        u0[1:length(u0)÷n_components]
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

"""A sensible default for relatively nice fisher / spruce-budworm plots"""
_logistic_init(N) = begin
    xs = range(-5, 20, length=N)
    return @. 1 - (exp(xs) / (1 + exp(xs)))
end

prob_rd_1d_fisher(; N, kwargs...) = begin
    reaction!(du, u) = (@. du = u * (1 - u))
    u0 = _logistic_init(N)
    return prob_rd_1d(; u0, reaction!, kwargs...)
end

prob_rd_1d_newell_whitehead_segel(; N, kwargs...) = begin
    reaction!(du, u) = (@. du = u * (1 - u)^2)
    u0 = _logistic_init(N)
    return prob_rd_1d(; u0, reaction!, kwargs...)
end

prob_rd_1d_zeldovich_frank_kamenetskii(; N, β=1, kwargs...) = begin
    reaction!(du, u) = (@. du = u * (1 - u) * exp(-β * (1 - u)))
    u0 = _logistic_init(N)
    return prob_rd_1d(; u0, reaction!, kwargs...)
end

prob_rd_1d_sir(; N, diffusion=0.02, β=0.3, γ=0.07, P=1000.0,
    tspan=(0.0, 150.0), dx=0.5, kwargs...) = begin
    reaction!(du, u) = begin
        S, dS = view(u, 1:N), view(du, 1:N)
        I, dI = view(u, N+1:2N), view(du, N+1:2N)
        R, dR = view(u, 2N+1:3N), view(du, 2N+1:3N)
        @. dS = -β * S * I / P
        @. dI = β * S * I / P - γ * I
        @. dR = γ * I
    end

    # xs = range(-5, 15, length=N)
    # I0 = @. 200 * exp(-(xs .^ 2) ./ 1 .^ 2) + 1
    xs = range(0, 2π, length=N)
    noise = rand(N)
    I0 = @. 100 * (sin(xs)+0.5noise)^2 + 1
    S0 = P * ones(N) - I0
    R0 = zeros(N)
    u0 = vcat(S0, I0, R0)

    return prob_rd_1d(; u0, reaction!, diffusion, tspan, dx, n_components=3, kwargs...)
end
