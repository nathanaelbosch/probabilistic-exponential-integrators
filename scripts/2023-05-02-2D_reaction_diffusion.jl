using OrdinaryDiffEq, LinearAlgebra, ForwardDiff, SimpleUnPack, SparseArrays, ProbNumDiffEq
using Plots

mat2vec(m) = reshape(m, length(m))
vec2mat(v) = (M = Int(sqrt(length(v))); reshape(v, M, M))

function laplace_2d(x; dx=0.05)
    lap = zeros(eltype(x), size(x))
    lap[2:end-1, 2:end-1] =
        (
            x[1:end-2, 2:end-1]
            .+
            x[3:end, 2:end-1]
            .+
            x[2:end-1, 1:end-2]
            .+
            x[2:end-1, 3:end]
            .-
            4x[2:end-1, 2:end-1]
        ) ./ dx^2
    return lap
end

N = 16
x_mat = rand(N, N);
x_vec = x_mat[:];
∇² = ForwardDiff.jacobian(laplace_2d, x_mat);
∇²s = sparse(∇²)
@assert ∇² * x_vec ≈ laplace_2d(x_mat)[:]

function reaction_diffusion_2d(duv, uv, p, t; N²=N^2, ∇²=∇²s)
    @unpack diffu, diffv, Ru, Rv = p
    u, v = view(uv, 1:N²), view(uv, N²+1:2N²)
    du, dv = view(duv, 1:N²), view(duv, N²+1:2N²)
    # reaction
    Ru(du, u, v)
    Rv(dv, u, v)
    # diffusion
    mul!(du, ∇², u, diffu, 1)
    mul!(dv, ∇², v, diffv, 1)
end

u0 = rand(N, N)[:];
v0 = rand(N, N)[:];
uv0 = [u0; v0];

# Gray-Scott
f = 0.038
k = 0.061
Ru_gs(du, u, v; f=f, k=k) = @. du = -u * v^2 + f * (1 - u)
Rv_gs(dv, u, v; f=f, k=k) = @. dv = u * v^2 - (f + k) * v
p_gs = (diffu=2e-4, diffv=1e-4, Ru=Ru_gs, Rv=Rv_gs)
# prob = ODEProblem(reaction_diffusion_2d, uv0, (0.0, 500.0), p_gs);

# FitzHugh-Nagumo
Ru_fhn(du, u, v; λ=1.0, κ=0.005, σ=1.0) =
    (@. du = λ * u - u^3 - κ - σ * v)
Rv_fhn(dv, u, v; τ=0.1) =
    (@. dv = (u - v) / τ)
p_fhn = (diffu=2.8e-4, diffv=5e-3 / 0.1, Ru=Ru_fhn, Rv=Rv_fhn)
prob = ODEProblem(reaction_diffusion_2d, uv0, (0.0, 50.0), p_fhn);

sol = solve(prob, Rosenbrock23(), saveat=1);

# Create and save the gif
anim = @animate for i in 1:length(sol)
    heatmap(
        vec2mat(view(sol[i], 1:N^2)),
        aspect_ratio=:equal,
        color=:viridis,
        cbar=false,
        title="u",
    )
end
gif(anim, "fhn.gif", fps=15)
