function chi2(gaussian_estimate, actual_value)
    μ, Σ = gaussian_estimate
    diff = μ - actual_value
    chi2_pinv = diff' * pinv(Matrix(Σ)) * diff
    return chi2_pinv
end

function MyWorkPrecision(
    prob,
    alg,
    abstols,
    reltols,
    args...;
    appxsol,
    dts=nothing,
    dense_errors=false,
    timeseries_errors=false,
    kwargs...,
)
    results = []
    for (i, (atol, rtol)) in enumerate(zip(abstols, reltols))

        sol_call() =
            isnothing(dts) ?
            solve(prob, alg, abstol=atol, reltol=rtol, args...; kwargs...) :
            solve(
                prob,
                alg,
                abstol=atol,
                reltol=rtol,
                adaptive=false,
                tstops=prob.tspan[1]:dts[i]:prob.tspan[2],
                args...;
                kwargs...,
            )

        sol = sol_call()
        errsol =
            sol isa ProbNumDiffEq.ProbODESolution ?
            appxtrue(
                mean(sol),
                appxsol;
                dense_errors=dense_errors,
                timeseries_errors=timeseries_errors,
            ) :
            appxtrue(
                sol,
                appxsol;
                dense_errors=dense_errors,
                timeseries_errors=timeseries_errors,
            )

        # Time
        tbest = 9999999999999999
        for i in 1:5
            t = @elapsed sol_call()
            tbest = min(tbest, t)
        end

        r = Dict(
            :final => errsol.errors[:final],
            :time => tbest,
            :nf => isnothing(errsol.destats) ? nothing : errsol.destats.nf,
            :njacs => isnothing(errsol.destats) ? nothing : errsol.destats.njacs,
            :nsteps => length(sol)-1,
        )
        if dense_errors
            r[:L2] = errsol.errors[:L2]
        end
        if timeseries_errors
            r[:l2] = errsol.errors[:l2]
        end
        if sol isa ProbNumDiffEq.ProbODESolution
            r[:chi2_final] = chi2(sol.pu[end], appxsol.(sol.t[end]))[1]
        end

        push!(results, r)
    end
    return results
end
