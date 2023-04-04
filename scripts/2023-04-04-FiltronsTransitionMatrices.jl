#=
Tried implementing the transition matrix with phi functions as suggested in Filtron's draft.

TL;DR: In all the cases I tried, it's slower than just calling `exponential!`.
The code also got a bit verbose because I tried optimizing away allocations etc.
=#
using ProbNumDiffEq, LinearAlgebra, ExponentialUtilities

d, q = 2, 3

iwp = ProbNumDiffEq.IWP(d, q)
iwpsde = ProbNumDiffEq.to_1d_sde(iwp)
Fiwp, Liwp = kron(iwpsde.F, I(d)), kron(iwpsde.L, I(d))

L = rand(d, d)
Fioup = Fiwp + kron(Diagonal([zeros(q); 1]), L)

h = 0.1
Phi_iwp = exp(Fiwp * h)
Phi_ioup = exp(Fioup * h)

A = kron(ProbNumDiffEq.to_1d_sde(ProbNumDiffEq.IWP(d, q - 1)).F, I(d))
phis = ExponentialUtilities.phi(L * h, q)
Phi12 = vcat((h .^ (q:-1:1) .* phis[end:-1:2])...)
[exp(A * h) Phi12; zero(Phi12)' phis[1]]

C = copy(Fioup)
method = ExpMethodHigham2005()
cache = ExponentialUtilities.alloc_mem(C, method)
Phi_vanilla(h; C=C, Fioup=Fioup, method=method, cache=cache) = begin
    mul!(C, Fioup, h)
    ExponentialUtilities.exponential!(C, method, cache)
end

C_phis = ExponentialUtilities.phi(L * h, q)
C_L = copy(L)
expAh = exp(A * h)
M = [expAh Phi12; zero(Phi12)' phis[1]]
Phi_ft(h; C_L=C_L, L=L, C_phis=C_phis, d=d, q=q, M=M) = begin
    mul!(C_L, L, h)
    phis = ExponentialUtilities.phi!(C_phis, C_L, q)
    # Phi12 = vcat((h .^ (q:-1:1) .* phis[end:-1:2])...)
    for i in 0:q-1
        @. M[i*d+1:(i+1)*d, d*q+1:end] = h * phis[end-i]
    end
    M
end

h = 0.1
@btime Phi_vanilla(h; C, Fioup, method, cache);
@btime Phi_ft(h; C_L, L, C_phis, q, M, d);
# @profview for _ in 1:10_000 Phi_ft(h; C_L, L, C_phis, q, M, d) end
