printname(::AbstractExperiment; plural = false) = plural ? "experiments" : "experiment"

function Base.summary(io::IO, experiment::AbstractExperiment)
    config = get_config(experiment)
    overrides = get_overrides(config)
    experiment_specific = ""
    pn = printname(experiment)
    if has_dependency(experiment)
        w = isempty(experiment_specific) ? "with" : "\nand"
        experiment_specific *= "$w initial conditions obtained from another $pn"
    end
    if !isempty(overrides)
        experiment_specific *= "\nwith the following overrides:\n" * join(overrides, '\n')
    end
    if isempty(experiment_specific)
        experiment_specific *= "with no overrides"
    end

    model = get_model(experiment)

    printstyled(io, nameof(typeof(experiment)), color = :blue, bold = true)
    println(io, " for ", nameof(model), ' ', experiment_specific, '.')
end

function Base.show(io::IO, ::MIME"text/plain", experiment::AbstractExperiment)
    summary(io, experiment)
end

function Base.show(io::IO, ::MIME"text/plain", experiment::Experiment)
    summary(io, experiment)
    prob_f = get_prob(experiment)
    model = get_model(experiment)
    tspan = timespan(experiment)
    prob = get_prob(experiment)
    println(io, "The simulation of this ", printname(experiment), " is given by:")
    summary(io, prob)
    print(io, "\ntimespan: ", timespan(experiment))
end

function Base.show(io::IO, ::MIME"text/plain", experiment::DiscreteExperiment)
    summary(io, experiment)
    println(io, "timespan: ", timespan(experiment))
    println(io, "dt: ", step_size(experiment))
end

function Base.summary(io::IO, es::AbstractExperimentCollection)
    pn = printname(first(es), plural = length(es) > 1)
    printstyled(io, nameof(typeof(es)), color = :blue, bold = true)
    print(io,
        " collection with ",
        length(es) > 1 ? length(es) : "one", " ",
        pn, '.')
end

function Base.show(io::IO, ::MIME"text/plain", es::AbstractExperimentCollection)
    summary(io, es)
end

function Base.summary(io::IO, ist::InternalStorage)
    search_space = get_search_space(ist)
    lb = lowerbound(ist)
    ub = upperbound(ist)
    println(io, "Search space of length $(length(search_space)): ")
    print(io, search_space)
end

function Base.show(io::IO, ::MIME"text/plain", mc::InternalStorage)
    summary(io, mc)
end

function Base.summary(io::IO, prob::AbstractInverseProblem)
    experiments = get_experiments(prob)
    n = length(experiments)
    pn = printname(first(experiments), plural = n > 1)
    exp_description = n == 1 ? "one $pn" : "$n $pn"
    printstyled(io, nameof(typeof(prob)), color = :blue, bold = true)
    print(io,
        #   " for ",
        #   nameof(prob.model), ' ',
        " with ",
        exp_description)
end

function Base.show(io::IO, ::MIME"text/plain", prob::AbstractInverseProblem)
    summary(io, prob)
    ist = get_internal_storage(prob)
    n = length(get_search_space(ist))
    println(io, " with $n elements in the search space.")
end

function Base.show(io::IO, ::MIME"text/plain", ada::AbstractDesignAnalysis)
    summary(io, ada)
    println(io, '.')
end

function Base.summary(io::IO, alg::AbstractCalibrationAlgorithm)
    printstyled(io, nameof(typeof(alg)), color = :blue)
end

function Base.show(io::IO, ::MIME"text/plain", alg::AbstractCalibrationAlgorithm)
    summary(io, alg)
    maxiters = get_maxiters(alg)
    maxtime = get_maxtime(alg)
    optimizer = get_optimizer(alg)
    println(io,
        " method, optimizing with $(nameof(typeof(optimizer))).\nmaxiters = $maxiters and maxtime = $maxtime")
end

function failure_retcode(retcode::ReturnCode.T)
    retcode == ReturnCode.DtNaN || retcode == ReturnCode.DtLessThanMin ||
        retcode == ReturnCode.Unstable || retcode == ReturnCode.InitialFailure ||
        retcode == ReturnCode.ConvergenceFailure || retcode == ReturnCode.Failure ||
        retcode == ReturnCode.Infeasible
end

function Base.summary(io::IO, c::CalibrationResult)
    elapsed = c.elapsed
    print(io, "Calibration result")
    if !ismissing(elapsed)
        t = pretty_time(elapsed)
        print(io, " computed in $t")
    end
    if c.original isa SciMLBase.OptimizationSolution
        final_objective = sprint(show, c.original.objective, context = :compact => true)
        retcode = c.retcode
        printstyled(io, ". Final objective value: ")
        printstyled(io,
            final_objective,
            color = isinf(c.original.objective) ? :red : :normal)
        printstyled(io, ".\n")
        color = if successful_retcode(retcode)
            :green
        elseif failure_retcode(retcode)
            :red
        else
            :blue
        end
        if !isempty(c)
            printstyled(io, "Optimization ended with ")
            printstyled(io, retcode, color = color)
            printstyled(io, " Return Code and returned.")
        else
            printstyled(io, "Optimization ended with ")
            printstyled(io, retcode, color = color)
            printstyled(io, " Return Code.")
        end
    else
        # assuming imported result
        print(io, " imported from ", nameof(typeof(c.original)), ".")
    end
end

function Base.show(io::IO, ::MIME"text/plain", c::CalibrationResult)
    summary(io, c)
    println(io, '\n')
    !isempty(c) && pretty_table(io, c)
end

function Base.summary(io::IO, ens::ParameterEnsemble)
    elapsed = ens.elapsed
    print(io, "Parametric uncertainty ensemble of length $(length(ens))")
    if !ismissing(elapsed)
        t = pretty_time(elapsed)
        println(io, " computed in $t.")
    else
        println(io, '.')
    end
end

function Base.summary(io::IO, ens::MCMCResult)
    elapsed = ens.elapsed
    print(io, "MCMC result of length $(length(ens))")
    if !ismissing(elapsed)
        t = pretty_time(elapsed)
        println(io, " computed in $t.")
    else
        println(io, '.')
    end
end

function Base.show(io::IO, ::MIME"text/plain", ens::AbstractParametricUncertaintyEnsemble)
    summary(io, ens)
    objectives = map(i -> i.original.objective, ens)
    col_names = vcat("Final Objective", Tables.columnnames(ens)...)
    pretty_table(io,
        hcat(objectives, reduce(hcat, ens.u)'),
        header = col_names
    )
end

function Base.show(io::IO, ::MIME"text/plain", ::NoData)
    print(io, "NoData()")
end

function import_res(res::Vector, prob;
        header = collect(Symbol.(search_space_names(prob))))
    t = Tables.table(permutedims(res); header)
    rows = Tables.rowtable(t)
    CalibrationResult(only(rows), prob, nothing, res, missing, nothing, missing)
end

"""
    import_res(res, prob)

Import a CalibrationResult object ([`CalibrationResult`](@ref)) from tabular data or a vector.
If a table (in the sense of Tables.jl) in provided, it is assumed to have one row.
Note that since a `Vector` is one column and calibration results are one row, the input is internally permuted.

## Arguments

  - `res`: tabular data that contains the results of a calibration.
    This could be e.g a `DataFrame`, a `CSV.File` or a `Vector`.
  - `prob`: an [`InverseProblem`](@ref) object.

## Example

```
julia> import_res([1,2], prob)
Calibration result imported from Array.

┌────┬────┐
│ k1 │ c1 │
├────┼────┤
│  1 │  2 │
└────┴────┘
```
"""
function import_res(res, prob)
    expected_header = collect(Symbol.(search_space_names(prob)))
    header = Tables.columnnames(res)
    @assert expected_header==header "Expected $expected_header, got $header"
    rows = NamedTuple.(pairs.(Tables.rows(res)))
    @assert length(rows)==1 "Calibration results are one single row, got $(length(rows))"
    CalibrationResult(only(rows),
        prob,
        nothing,
        collect(only(rows)),
        missing,
        nothing,
        missing)
end

function import_ps(ps::AbstractMatrix, prob;
        header = collect(Symbol.(search_space_names(prob))),
        result_type = ParameterEnsemble)
    t = Tables.table(ps; header)
    rows = Tables.rowtable(t)
    result_type(rows, prob, missing, missing)
end

"""
    import_ps(ps, prob)

Import a ParameterEnsemble object ([`ParameterEnsemble`](@ref)) from tabular data.

## Arguments

  - `ps`: tabular data that contains the parameters of the population.
    This could be e.g a `DataFrame` or a `CSV.File`.
  - `prob`: an [`InverseProblem`](@ref) object.
"""
function import_ps(ps, prob; result_type = ParameterEnsemble)
    expected_header = collect(Symbol.(search_space_names(prob)))
    header = Tables.columnnames(ps)
    @assert expected_header==header "Expected $expected_header, got $header"
    rows = NamedTuple.(pairs.(Tables.rows(ps)))
    result_type(rows, prob, missing, missing)
end
