function chi2(gaussian_estimate, actual_value)
    μ, Σ = gaussian_estimate
    diff = μ - actual_value
    chi2_pinv = if any(isnan.(Σ.R))
        NaN
    else
        diff' * pinv(Matrix(Σ)) * diff
    end
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
    verbose=false,
    kwargs...,
)
    results = []
    for (i, (atol, rtol)) in enumerate(zip(abstols, reltols))
        if verbose
            if !isnothing(dts)
                @info "i=$i" dts[i]
            else
                @info "i=$i" atol rtol
            end
        end

        sol_call() =
            if isnothing(dts)
                solve(prob, alg, abstol=atol, reltol=rtol, args...; kwargs...)
            else
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
            end

        try
            sol = sol_call()
            errsol = if sol isa ProbNumDiffEq.ProbODESolution
                appxtrue(
                    mean(sol),
                    appxsol;
                    dense_errors=dense_errors,
                    timeseries_errors=timeseries_errors,
                )
            else
                appxtrue(
                    sol,
                    appxsol;
                    dense_errors=dense_errors,
                    timeseries_errors=timeseries_errors,
                )
            end

            # Time
            tbest = Inf
            for i in 1:3
                t = @elapsed sol_call()
                tbest = min(tbest, t)
            end

            r = Dict(
                :final => errsol.errors[:final],
                :time => tbest,
                :nf => isnothing(errsol.destats) ? nothing : errsol.destats.nf,
                :njacs => isnothing(errsol.destats) ? nothing : errsol.destats.njacs,
                :nsteps => length(sol) - 1,
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
            if !isnothing(dts)
                r[:dt] = dts[i]
            end

            @info "result" r
            push!(results, r)
        catch e
            if e isa InterruptException
                throw(e)
            end
            @warn "Run failed!"
            r = Dict(
                :final => NaN,
                :time => NaN,
                :nf => NaN,
                :njacs => NaN,
                :nsteps => NaN,
            )
            if !isnothing(dts)
                r[:dt] = dts[i]
            end
            if dense_errors
                r[:L2] = NaN
            end
            if timeseries_errors
                r[:l2] = NaN
            end
            if alg isa ProbNumDiffEq.AbstractEK
                r[:chi2_final] = NaN
            end
            push!(results, r)
        end
    end
    return results
end
