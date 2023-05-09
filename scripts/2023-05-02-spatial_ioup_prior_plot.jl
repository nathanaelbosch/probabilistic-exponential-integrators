using OrdinaryDiffEq, LinearAlgebra, ForwardDiff, SimpleUnPack, SparseArrays, ProbNumDiffEq
using Plots
import BayesExpIntExperiments as BEIE

N = 32
prob, L = BEIE.prob_rd_1d_fisher(; N);

############################################################################################
# Can I plot the resulting prior? Yes!
############################################################################################
NU = 1
# prior = IOUP(length(u0), NU, zero(L)) # This gives an IWP; neighbors get very different
prior = IOUP(length(prob.u0), NU, L) # This gives the exp-int prior; neighbors are quite close
A, Q = ProbNumDiffEq.discretize(prior, 1);
Q = Matrix(Q);
function simulate(A, Q, N)
    x = rand(ProbNumDiffEq.Gaussian(zeros(size(A, 1)), 0.1I))
    ys = zeros(N, size(A, 1))
    ys[1, :] .= x
    for i in 2:N
        x = rand(ProbNumDiffEq.Gaussian(A * x, 1e-2 * Symmetric(Q)))
        ys[i, :] .= x
    end
    return ys
end
ys = simulate(A, Q, 5000);

E0 = ProbNumDiffEq.projection(length(prob.u0), NU)(0);
anim = @animate for i in 1:size(ys, 1)
    plot(E0 * ys[i, :])
end
gif(anim, "prior.gif", fps=60)

xs = range(-5, 5, length=100)
u0 = @. exp(-(xs .^ 2) ./ 0.5 .^ 2)
