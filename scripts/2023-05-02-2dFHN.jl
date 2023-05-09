using OrdinaryDiffEq, LinearAlgebra, SimpleUnPack
using Plots

# Define the 2D laplacian operator for discretized space
function laplace(f; dx)
    lap = zeros(eltype(f), size(f))
    lap[2:end-1, 2:end-1] =
        (
            f[1:end-2, 2:end-1]
            .+
            f[3:end, 2:end-1]
            .+
            f[2:end-1, 1:end-2]
            .+
            f[2:end-1, 3:end]
            .-
            4f[2:end-1, 2:end-1]
        ) ./ dx^2
    return lap
end

# Define the vector field
function fhn(duv, uv, p, t)
    @unpack Du, Dv, λ, σ, κ, τ, dx = p
    u, v = view(uv, :, :, 1), view(uv, :, :, 2)
    du, dv = view(duv, :, :, 1), view(duv, :, :, 2)
    Lu = laplace(u; dx)
    Lv = laplace(v; dx)
    @. du = Du * Lu + λ * u - u^3 - κ - σ * v
    @. dv = (Dv * Lv + u - v) / τ
end

N = 128
u0 = rand(N, N);
v0 = rand(N, N);
uvw0 = cat(u0, v0, dims=3);

# Reaction diffusion model parameters
Du, Dv, f, k = 1e-2, 1e-2, 0.03, 0.07
p = (
    τ=0.1,
    Du=2.8e-4,
    Dv=5e-3,
    λ=1.0,
    κ=0.005,
    σ=1.0,
    dx=0.02,
)

# Set up problem and solve
T = 30
tspan = (0.0, T)
prob = ODEProblem(fhn, uvw0, tspan, p);
sol = solve(prob, Vern9(), abstol=1e-8, reltol=1e-8, saveat=1);
# heatmap(sol[end][:, :, 1])

# Create and save the gif
anim = @animate for i in 1:length(sol)
    heatmap(sol[i][:, :, 1], aspect_ratio=:equal, color=:viridis, cbar=false, title="u")
end
gif(anim, "fhn.gif", fps=15)
