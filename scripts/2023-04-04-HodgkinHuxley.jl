#=
A small self-contained implementation of a Hodgkin-Huxley ODE and some small evaluation

TL;DR:
- Rosenbrock-style IOUP consistently needs less steps to reach similar or lower errors
- But, runtimes are higher
=#
using ProbNumDiffEq,
    BenchmarkTools, ProfileView, SciMLBase, ForwardDiff, OrdinaryDiffEq, LinearAlgebra

function get_hh_ivp(; tspan=(0.0, 100.0), p=[20, 15])
    I(t) =
        if t isa ProbNumDiffEq.Taylor1 || t isa ProbNumDiffEq.TaylorN
            return zero(t)
        else
            return (10 <= t <= 90) ? 500one(t) * 1e-6 : zero(t)
        end

    αm(V, VT) = -0.32 * (V - VT - 13) / (exp(-(V - VT - 13) / 4) - 1)
    βm(V, VT) = 0.28 * (V - VT - 40) / (exp((V - VT - 40) / 5) - 1)

    αn(V, VT) = -0.032 * (V - VT - 15) / (exp(-(V - VT - 15) / 5) - 1)
    βn(V, VT) = 0.5 * exp(-(V - VT - 10) / 40)

    αh(V, VT) = 0.128 * exp(-(V - VT - 17) / 18)
    βh(V, VT) = 4 / (1 + exp(-(V - VT - 40) / 5))

    # const model params
    ENa = 53
    EK = -107
    area = 15e-5
    C = 1
    Eleak = -70
    VT = -60
    gleak = 0.1
    V0 = -70

    # initial value
    # Would be the solution when di/dt = 0:
    m∞(V, VT) = 1 / (1 + βm(V, VT) / αm(V, VT))
    n∞(V, VT) = 1 / (1 + βn(V, VT) / αn(V, VT))
    h∞(V, VT) = 1 / (1 + βh(V, VT) / αh(V, VT))
    p∞(V) = 1 / (1 + exp(-(V + 35) / 10))
    u0 = [V0, m∞(V0, VT), n∞(V0, VT), h∞(V0, VT)]

    function HodgkinHuxley!(du, u, p, t)
        V, m, n, h = u
        gNa, gK = p

        # channel gating
        du[2] = dm = (αm(V, VT) * (1 - m) - βm(V, VT) * m)
        du[3] = dn = (αn(V, VT) * (1 - n) - βn(V, VT) * n)
        du[4] = dh = (αh(V, VT) * (1 - h) - βh(V, VT) * h)

        # currents
        INa = gNa * m^3 * h * (V - ENa) * area
        IK = gK * n^4 * (V - EK) * area
        Ileak = gleak * (V - Eleak) * area
        Cm = C * area
        du[1] = dV = -(Ileak + INa + IK - I(t)) / Cm
    end

    prob = ODEProblem{true,SciMLBase.FullSpecialize()}(HodgkinHuxley!, u0, tspan, p)
    return prob
end

prob = get_hh_ivp()
ref = solve(prob, Vern9(), abstol=1e-10, reltol=1e-10)

NU = 3
ALG = EK0

sol_iwp = solve(prob, ALG(prior=IWP(NU), smooth=false), dense=false);
sol_iwp.destats
norm(sol_iwp.u[end] - ref.u[end])
@btime solve(prob, ALG(prior=IWP(NU), smooth=false), dense=false);

sol_ioup =
    solve(prob, ALG(prior=IOUP(NU, update_rate_parameter=true), smooth=false), dense=false);
sol_ioup.destats
norm(sol_ioup.u[end] - ref.u[end])
@btime solve(
    prob,
    ALG(prior=IOUP(NU, update_rate_parameter=true), smooth=false),
    dense=false,
);

# Profileview shows what's taking time: The matrix exponential is expensive
@profview for _ in 1:10
    solve(prob, ALG(prior=IOUP(NU, update_rate_parameter=true), smooth=false), dense=false)
end
