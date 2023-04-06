#=
Tried implementing the transition matrix with phi functions as suggested in Filtron's draft.

TL;DR: In all the cases I tried, it's slower than just calling `exponential!`.
The code also got a bit verbose because I tried optimizing away allocations etc.
=#
using ProbNumDiffEq, LinearAlgebra, ExponentialUtilities, BenchmarkTools, ProfileView

d, q = 4, 3

iwp = ProbNumDiffEq.IWP(d, q)
iwpsde = ProbNumDiffEq.to_1d_sde(iwp)
Fiwp, Liwp = kron(iwpsde.F, I(d)), kron(iwpsde.L, I(d))

L = Symmetric(rand(d, d))
Fioup = Fiwp + kron(Diagonal([zeros(q); 1]), L)

h = 0.1
Phi_ioup = exp(Fioup * h) # this is the correct transition matrix

############################################################################################
# 1. A more efficient implementation of the vanilla exp approach
############################################################################################
C = copy(Fioup);
method = ExpMethodHigham2005()
cache = ExponentialUtilities.alloc_mem(C, method);
Phi_vanilla(h; C=C, Fioup=Fioup, method=method, cache=cache) = begin
    mul!(C, Fioup, h)
    ExponentialUtilities.exponential!(C, method, cache)
end

@assert Phi_ioup ≈ Phi_vanilla(h)

############################################################################################
# 2. Filtron's suggested implementation via phi functions
############################################################################################
A_iwp_smaller = kron(ProbNumDiffEq.to_1d_sde(ProbNumDiffEq.IWP(d, q - 1)).F, I(d));
C_phis = ExponentialUtilities.phi(L, q);
Phi12 = vcat((h .^ (q:-1:1) .* C_phis[end:-1:2])...);
C_L = copy(L);
expAh = exp(A_iwp_smaller * h); # could be computed in closed form since it's an IWP
M = [expAh Phi12; zero(Phi12)' C_phis[1]];
Phi_ft(h; C_L=C_L, L=L, C_phis=C_phis, d=d, q=q, M=M) = begin
    # mul!(C_L, L, h)
    phis = ExponentialUtilities.phi!(C_phis, L * h, q)
    # Phi12 = vcat((h .^ (q:-1:1) .* phis[end:-1:2])...)
    for i in 0:q
        @. M[i*d+1:(i+1)*d, d*q+1:end] = h^(q - i) * phis[end-i]
    end
    M
end
@assert Phi_ioup ≈ Phi_ft(h)

############################################################################################
# Benchmarks
############################################################################################
@btime Phi_vanilla(h; C, Fioup, method, cache);
@btime Phi_ft(h; C_L, L, C_phis, q, M, d);
#=
Some results:
- d, q = 2, 3: 5us vs 11us => vanilla exp is faster
- d, q = 100, 3: 12ms vs 70us => vanilla exp is faster
- d, q = 2, 30: 133us vs 120us => the smart one can actually be faster! for huge q though
=#

############################################################################################
# What if L has structure?
############################################################################################
L_sym = Symmetric(rand(d, d))
L_dense = L_sym |> Matrix
@btime exp(L_dense);
@btime exp(L_sym);
@btime ExponentialUtilities.exponential!(copy(L_dense));
@btime ExponentialUtilities.exponential!(copy(L_sym));
@btime ExponentialUtilities.phi(L_dense, q);
@btime ExponentialUtilities.phi(L_sym, q);
C_phis = ExponentialUtilities.phi(L_sym, q);
@btime ExponentialUtilities.phi!(C_phis, L_sym, q)

L_diag = Diagonal(L_dense);
@btime exp(L_diag);
@btime ExponentialUtilities.phi(L_diag, q);
C_phis = ExponentialUtilities.phi(L_diag, q);
@btime ExponentialUtilities.phi!(C_phis, L_diag * h, q)

# It's not quite clear to me when any of this helps
