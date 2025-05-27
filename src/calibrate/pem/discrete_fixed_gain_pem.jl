"""
    DiscreteFixedGainPEM(alpha)

This model transformation implements the prediction error method as a discrete callback
that brings the solution closer to the data at the corresponding timepoints as the integrator steps through
with a factor that depends on `alpha`, which is a scalar value from 0 to 1. If `alpha` is 0, then this is equivalent with not
applying any correction, while `alpha=1` means that we start each step from the data.

If there is data for the unknowns, it is directly used.
If there is data for observed variables, corresponding equations which are linear with respect to the unknowns are used
to estimate other unknowns in those equations for which there is no data. This is only the case if the system of equations
are invertible with respect to those other unknowns.

It is also useful for calibrating unstable systems, as pure simulation methods cannot track the true solution over longer time intervals and diverge.
"""
struct DiscreteFixedGainPEM{A} <: AbstractPredictionErrorMethod
    alpha::A
end

function discretefixedgainpem_nonlinearproblem(model, md, experiment_data)
    # No observed data
    isempty(md.observed_vars_data) && return nothing, [], []

    eqs = md.observed_eqs[md.observed_idxs]

    # Do fixed point substitution to get `observed = f(states)`
    sub_eqs = fixed_point_substitute(md.observed_vars, md.observed_eqs, eqs)

    # Filter out the linear eqs
    linear_eqs = sub_eqs[map(
        Base.Fix2(ModelingToolkit.isaffine, md.unknowns_vars), getproperty.(
            sub_eqs, :rhs))]

    # Make variable to parameter in the equations
    combined_dict = Dict(
        (get_parent_symbol.(md.observed_vars_data) .=>
            ModelingToolkit.toparam.(get_parent_symbol.(md.observed_vars_data)))...,
        (get_parent_symbol.(md.unknowns_vars_data) .=>
            ModelingToolkit.toparam.(get_parent_symbol.(md.unknowns_vars_data)))...,
        get_parent_symbol(md.iv) => ModelingToolkit.toparam(get_parent_symbol(md.iv))
    )

    # Substitute variables as parameters
    corrected_linear_eqs = Equation[]
    for linear_eq in linear_eqs
        eq = substitute(linear_eq, combined_dict)
        push!(corrected_linear_eqs, eq)
    end

    estimatable_unknowns, estimatable_unknowns_idxs = find_estimatable_unknowns(
        md.unknowns_vars, md.unknowns_idxs, corrected_linear_eqs)

    # Set the non linear system
    sys = setup_nonlinearsystem(corrected_linear_eqs, estimatable_unknowns,
        vcat(md.iv, md.observed_vars_data, md.unknowns_vars_data, md.ps))

    # Error when cannot use PEM
    (length(unknowns(sys)) != 0 && length(md.unknowns_vars_data) == 0) &&
        throw(ErrorException("Cannot use PEM as either there are no data for the unknowns OR observed cannot be used to invert and estimate unknowns."))

    # Warn when the observed data is not used (but there will be `unknowns` data)
    if length(unknowns(sys)) != 0
        @warn "Observed data cannot be used to invert unknowns because of no unique solution."
        return nothing, [], []
    end

    # If it is `VectorLike`, it has to be observed data
    initial_observed_vars_data = if data_shape(typeof(experiment_data)) isa VectorLike
        [experiment_data[1]]
    else
        experiment_data[md.observed_idxs_data, 1]
    end
    # Build the non linear least squares problem
    np = NonlinearLeastSquaresProblem(sys, [],
        [(md.observed_vars_data .=> initial_observed_vars_data)...,
            (md.unknowns_vars_data .=> experiment_data[md.unknowns_idxs_data, 1])...,
            md.iv => experiment_data.time[1], (md.ps .=>
                getindex.((default_values(model),), md.ps))...])

    return np, estimatable_unknowns, estimatable_unknowns_idxs
end

function discretefixedgainpem_setp(::Nothing, md, shape)
    nothing
end

function discretefixedgainpem_setp(np::NonlinearLeastSquaresProblem, md, ::MatrixLike)
    setsym(np, [md.observed_vars_data..., md.unknowns_vars_data..., md.iv])
end

function discretefixedgainpem_setp(np::NonlinearLeastSquaresProblem, md, ::VectorLike)
    setsym(np, [md.observed_vars_data..., md.iv])
end

# Main function to build the callback for PEM
function prediction_error_callback(
        pem::DiscreteFixedGainPEM, experiment_data::TimeSeriesData, model)
    alpha = pem.alpha
    md = pem_metadata(experiment_data, model)
    np, estimatable_unknowns, estimatable_unknowns_idxs = discretefixedgainpem_nonlinearproblem(
        model, md, experiment_data)
    setp_func = discretefixedgainpem_setp(np, md, data_shape(typeof(experiment_data)))

    ## This is needed as `experiment_data` behaves differently having one or multiple states
    ## Remove this after that is fixed
    function get_affect_function(::MatrixLike)
        return function affect!(integrator)
            i = findfirst(==(integrator.t), experiment_data.time)
            if !isempty(md.observed_vars_data) && !isnothing(np)
                setp_func(np,
                    [experiment_data[md.observed_idxs_data, i]...,
                        experiment_data[md.unknowns_idxs_data, i]..., integrator.t])
                sol = solve(np)
                !SciMLBase.successful_retcode(sol) &&
                    throw(ErrorException("Not able to invert observed variables into states for PEM."))
                vals = sol[estimatable_unknowns]
                @views integrator.u[estimatable_unknowns_idxs] += alpha * (
                    vals - integrator.u[estimatable_unknowns_idxs])
            end
            @views integrator.u[md.unknowns_idxs] += alpha * (
                experiment_data[md.unknowns_idxs_data, i] - integrator.u[md.unknowns_idxs])
        end
    end

    function get_affect_function(::VectorLike)
        return function affect!(integrator)
            i = findfirst(==(integrator.t), experiment_data.time)
            if !isempty(md.observed_vars_data) && !isnothing(np)
                setp_func(np, [experiment_data[i], integrator.t])
                sol = solve(np)
                !SciMLBase.successful_retcode(sol) &&
                    throw(ErrorException("Not able to invert observed variables into states for PEM."))
                vals = sol[estimatable_unknowns]
                @views integrator.u[estimatable_unknowns_idxs] += alpha * (vals -
                                                                   integrator.u[estimatable_unknowns_idxs])
            else
                @views integrator.u[md.unknowns_idxs] += alpha * (experiment_data[[i]] -
                                                          integrator.u[md.unknowns_idxs])
            end
        end
    end

    affect_func = get_affect_function(data_shape(typeof(experiment_data)))
    return PresetTimeCallback(experiment_data.time,
        affect_func,
        save_positions = (false, false))
end
