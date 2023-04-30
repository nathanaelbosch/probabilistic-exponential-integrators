using OrdinaryDiffEq, LinearAlgebra
using Plots

# Define the 2D laplacian operator for discretized space
function laplace(f)
    lap = zeros(eltype(f), size(f))
    lap[2:end-1, 2:end-1] = (
        f[1:end-2, 2:end-1]
        .+ f[3:end, 2:end-1]
        .+ f[2:end-1, 1:end-2]
        .+
        f[2:end-1, 3:end]
        .-
        4f[2:end-1, 2:end-1]
    )
    return lap
end

# Define the vector field
function gray_scott(duv, uv, p, t)
    Du, Dv, f, k = p
    u, v = view(uv, :, :, 1), view(uv, :, :, 2)
    du, dv = view(duv, :, :, 1), view(duv, :, :, 2)
    Lu = laplace(u)
    Lv = laplace(v)
    @. du = Du * Lu - u * v^2 + f * (1 - u)
    @. dv = Dv * Lv + u * v^2 - (f + k) * v
end

# using Images
# img = load("logo.png");
# matrix = Gray.(img) |> channelview |> (x -> float.(x))
# h, w = size(matrix);
# if h < w
#     padding = zeros(Float64, w - h, w)
#     padding = zeros(Float64, w - h, w)
#     matrix = cat(dims=1, matrix, padding)
# elseif w < h
#     padding = zeros(Float64, h - w, h)
#     matrix = cat(dims=2, matrix, padding)
# end
# N = size(matrix, 1)

# Set up initial conditions
# N = 512
# u0 = zeros(N, N);
# v0 = zeros(N, N);
# u0[N÷2-5:N÷2+5, N÷2-5:N÷2+5] .= 1;
# v0[N÷2-5:N÷2+5, N÷2-5:N÷2+5] .= 0;
u0 = rand(N, N);
v0 = rand(N, N);
# circles:
u0 = zeros(N, N);
# v0 = zeros(N, N);
xc, yc = N ÷ 2, N ÷ 2
for i in 1:N
    for j in 1:N
        r = sqrt((i - xc)^2 + (j - yc)^2)
        if r <= N ÷ 10
            u0[i, j] = 0.0
            # v0[i, j] = 1.0
        elseif r <= N ÷ 8
            u0[i, j] = 1.0
            # v0[i, j] = 0.25
        elseif r <= N ÷ 6
            u0[i, j] = 0.0
            # v0[i, j] = 0.0
        elseif r <= N ÷ 4
            u0[i, j] = 0.5
            # v0[i, j] = 0.25
        elseif r <= N ÷ 2
            u0[i, j] = 0.2
            # v0[i, j] = 0.1
        end
    end
end
# Stripes
# u0 = zeros(N, N);
# v0 = zeros(N, N);
# for i in 1:N
#     for j in 1:N
#         if j % 10 <= 4
#             u0[i, j] = 0.5
#             v0[i, j] = 0.25
#         else
#             u0[i, j] = 0.2
#             v0[i, j] = 0.1
#         end
#     end
# end
u0 .= matrix[end:-1:begin, :]
uvw0 = cat(u0, v0, dims=3);

# Reaction diffusion model parameters
# Du, Dv, f, k = 2e-5, 1e-5, 0.03, 0.07
# Du, Dv, f, k = 1e-2, 1e-2, 0.062, 0.0609
Du, Dv, f, k = 1e-2, 1e-2, 0.03, 0.07
p = (Du, Dv, f, k)

# Set up problem and solve
T = 5000
tspan = (0.0, T)
prob = ODEProblem(gray_scott, uvw0, tspan, p);
sol = solve(prob, Vern9(), abstol=1e-8, reltol=1e-8, saveat=10);

# Create and save the gif
anim = @animate for i in 1:length(sol)
    heatmap(sol[i][:, :, 1], aspect_ratio=:equal, color=:viridis, cbar=false, title="u")
end
gif(anim, "grayscott.gif", fps=15)
