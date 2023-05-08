using ProbNumDiffEq, LinearAlgebra, ExponentialUtilities, BenchmarkTools, ProfileView, FastBroadcast

d, q = 16, 3

iwp = ProbNumDiffEq.IWP(d, q);
iwpsde = ProbNumDiffEq.to_1d_sde(iwp);
Fiwp, Liwp = kron(iwpsde.F, I(d)), kron(iwpsde.L, I(d));

# L = Symmetric(rand(d, d))
L = rand(d, d);
Fioup = Fiwp + kron(Diagonal([zeros(q); 1]), L);

h = 0.1
Phi_ioup = exp(Fioup * h); # this is the correct transition matrix

Fh = Fioup * h;
@btime exp($Fh);
Lh = L * h;
@btime exp($Lh);

function get_fastexp(mat)
    C = copy(mat)
    method = ExpMethodHigham2005()
    cache = ExponentialUtilities.alloc_mem(C, method)
    return h -> ExponentialUtilities.exponential!(mul!(C, mat, h), method, cache)
end
exp_Fioup = get_fastexp(Fioup)
@assert exp_Fioup(h) ≈ Phi_ioup

exp_L = get_fastexp(L)
@btime exp_Fioup(h);
@btime exp_L(h);

out = zeros(d * (q + 1), d);
Lh = L * h;
function compute_phis(Lh, d, q)
    phi = exp(Lh)
    phis = [copy(phi)]
    Lh_fac = lu(Lh)
    for i in 1:q
        # phi = Lh_fac \ (phi - 1 / factorial(i - 1) * I)
        f = 1 / factorial(i - 1)
        for j in 1:d
            phi[j, j] -= f
        end
        ldiv!(Lh_fac, phi)
        push!(phis, copy(phi))
    end
    return phis
end
function phis2mat(phis)
    out = zeros(d * (q + 1), d)
    # need to be filled in blocks from bottom to top
    for i in 1:q
        out[(i-1)*d+1:i*d, :] = phis[end-i] .* h^(q + 1 - i)
    end
    out[q*d+1:end, :] = phis[1]
    return out
end

C = copy(L);
method = ExpMethodHigham2005();
cache = ExponentialUtilities.alloc_mem(C, method);
out = zeros(d * (q + 1), d);
L_fac = lu(L);
umask = UpperTriangular(ones(d, d)) |> Matrix;
lmask = 1 .- umask;
function fast_compute_phis(out, L, h, C, method, cache, d, q, Lfac, Lmask, Umask)
    @. C = L * h
    phi = ExponentialUtilities.exponential!(C, method, cache)
    out[q*d+1:end, :] = phi

    @. Lfac.factors *= (Lmask + Umask * h)

    for i in 1:q

        f = 1 / factorial(i - 1)
        for j in 1:d
            phi[j, j] -= f
        end
        ldiv!(Lfac, phi)

        @. out[d*(q-i)+1:d*(q-i+1), :] = phi * h^i
    end
    return out
end

@assert fast_compute_phis(out, L, h, C, method, cache, d, q, lu(L), lmask, umask) ≈ Phi_ioup[:, end-d+1:end]
@btime fast_compute_phis(out, L, h, C, method, cache, d, q, lu(L), lmask, umask);
@btime fast_compute_phis(out, L, h, C, method, cache, d, q, _LU, lmask, umask) setup=(_LU=lu(L));
@btime exp_Fioup(h);
