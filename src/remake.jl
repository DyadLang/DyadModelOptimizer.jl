function remake_timeseries_data(experiment,
        data,
        model,
        saveat,
        tspan,
        save_names,
        keep_data)
    if ismissing(data)
        save_idxs = get_save_idxs(experiment)
        original_data = hasdata(experiment) ? get_data(experiment).original : nothing
        redo_data = false
        old_save_names = isnothing(save_idxs) ? get_saved_model_variables(experiment) :
                         states(model)[save_idxs]
        save_names = if ismissing(save_names)
            old_save_names
        else
            redo_data = true
            save_names
        end
        saveat = if ismissing(saveat)
            get_saveat(experiment)
        else
            redo_data = true
            saveat
        end
        experiment_data = if redo_data
            timeseries_data(original_data, save_names, saveat, tspan)
        else
            get_data(experiment)
        end

    else
        save_names = ismissing(save_names) ? determine_save_names(data, model) : save_names
        saveat = ismissing(saveat) ? determine_saveat(data) : saveat
        experiment_data = timeseries_data(data, save_names, saveat, tspan)
    end
    if !keep_data
        experiment_data = NoData()
    end

    return experiment_data, save_names, saveat
end

"""
    remake(experiment; kwargs...)

Re-construct an [`Experiment`](@ref) with new values for the fields specified by the keyword arguments.

# Positional Arguments

  - `experiment`: [`Experiment`](@ref) object.

# Keyword Arguments

  - `data`: A `DataSet` or a tabular data object.
  - `model`: An `ODESystem` describing the model that we are using for the experiment. The model needs to define defaults
    for all initial conditions and all parameters.

Rest of the keyword arguments are the same as [`Experiment`](@ref).
"""
function SciMLBase.remake(experiment::Experiment;
        data = missing,
        model = get_model(experiment),
        overrides = get_overrides(experiment),
        constraints = get_constraints(experiment),
        model_transformations = missing,
        callback = get_callbacks(experiment),
        tspan = timespan(experiment),
        save_names = missing,
        saveat = missing,
        constraints_ts = get_constraints_ts(experiment),
        loss_func = get_loss_func(experiment),
        postprocess = get_postprocess(experiment),
        noise_priors = get_config(experiment).noise_priors,
        likelihood = get_config(experiment).likelihood,
        reduction = get_config(experiment).reduction,
        dependency = get_dependency(experiment),
        name = nameof(experiment),
        alg = get_solve_alg(experiment),
        keep_data = true,
        do_not_replace_sol = Val(false),
        prob_kwargs = (;),
        kwargs...)
    experiment_data, save_names, saveat = remake_timeseries_data(experiment, data, model,
        saveat, tspan, save_names,
        keep_data)
    if ismissing(model_transformations)
        translate_model_transformations = false
        # don't duplicate callbacks
        model_transformations = get_model_transformations(experiment)
    else
        # model transformations changed, add the new callbacks
        translate_model_transformations = true
    end
    dependency = dependency isa UUID ? get_dependency(experiment) : dependency

    config = construct_config(
        experiment_data, model, overrides, constraints, constraints_ts,
        tspan, model_transformations,
        callback, dependency, save_names, saveat, loss_func, postprocess,
        noise_priors, likelihood, reduction, name, alg,
        # we need to take care to not create the callbacks for model transformations twice
        do_not_replace_sol, prob_kwargs, kwargs, translate_model_transformations)

    return Experiment(config, tspan)
end

"""
    remake(experiment; kwargs...)

Re-construct an [`DiscreteExperiment`](@ref) with new values for the fields specified by the keyword arguments.

# Positional Arguments

  - `experiment`: [`DiscreteExperiment`](@ref) object.

# Keyword Arguments

  - `data`: A `DataSet` or a tabular data object.
  - `model`: An `ODESystem` describing the model that we are using for the experiment. The model needs to define defaults
    for all initial conditions and all parameters.

Rest of the keyword arguments are the same as [`DiscreteExperiment`](@ref).
"""
function SciMLBase.remake(experiment::DiscreteExperiment;
        data = missing,
        model = get_model(experiment),
        overrides = get_overrides(experiment),
        constraints = get_constraints(experiment),
        model_transformations = missing,
        callback = get_callbacks(experiment),
        tspan = timespan(experiment),
        dt = step_size(experiment),
        save_names = missing,
        saveat = missing,
        constraints_ts = get_constraints_ts(experiment),
        loss_func = get_loss_func(experiment),
        postprocess = get_postprocess(experiment),
        noise_priors = get_config(experiment).noise_priors,
        likelihood = get_config(experiment).likelihood,
        reduction = get_config(experiment).reduction,
        dependency = get_dependency(experiment),
        name = nameof(experiment),
        alg = get_solve_alg(experiment),
        keep_data = true,
        do_not_replace_sol = Val(false),
        prob_kwargs = (;),
        kwargs...)
    if ismissing(model_transformations)
        translate_model_transformations = false
        # don't duplicate callbacks
        model_transformations = get_model_transformations(experiment)
    else
        # model transformations changed, add the new callbacks
        translate_model_transformations = true
    end
    experiment_data, save_names, saveat = remake_timeseries_data(experiment, data, model,
        saveat, tspan, save_names,
        keep_data)
    dependency = dependency isa UUID ? get_dependency(experiment) : dependency

    config = construct_config(
        experiment_data, model, overrides, constraints, constraints_ts,
        tspan, model_transformations,
        callback, dependency, save_names, saveat, loss_func, postprocess,
        noise_priors, likelihood, reduction, name, alg,
        # we need to take care to not create the callbacks for model transformations twice
        do_not_replace_sol, prob_kwargs, kwargs, translate_model_transformations)

    return DiscreteExperiment(config, tspan, dt)
end

function remake_ss_data(experiment, data, model, save_names, keep_data)
    if ismissing(save_names)
        save_idxs = get_save_idxs(experiment)
        save_names = isnothing(save_idxs) ? get_saved_model_variables(experiment) :
                     unknowns(model)[save_idxs]
    else
        # _, save_idxs = determine_what_to_save(model, save_names)
        save_names = determine_save_names(data, model, false)
    end
    if ismissing(data)
        experiment_data = get_data(experiment)
    else
        experiment_data = maybe_steadystate(data, save_names)
    end
    if !keep_data
        experiment_data = NoData()
    end
    return experiment_data, save_names
end

"""
    remake(experiment; kwargs...)

Re-construct an [`SteadyStateExperiment`](@ref) with new values for the fields specified by the keyword arguments.

# Positional Arguments

  - `experiment`: [`SteadyStateExperiment`](@ref) object.

# Keyword Arguments

  - `data`: A `DataSet` or a tabular data object.
  - `model`: An `ODESystem` describing the model that we are using for the experiment. The model needs to define defaults
    for all initial conditions and all parameters.

Rest of the keyword arguments are the same as [`SteadyStateExperiment`](@ref).
"""
function SciMLBase.remake(experiment::SteadyStateExperiment;
        data = missing,
        model = get_model(experiment),
        overrides = get_overrides(experiment),
        constraints = get_constraints(experiment),
        model_transformations = missing,
        callback = get_callbacks(experiment),
        save_names = missing,
        constraints_ts = get_constraints_ts(experiment),
        loss_func = get_loss_func(experiment),
        postprocess = get_postprocess(experiment),
        noise_priors = get_config(experiment).noise_priors,
        likelihood = get_config(experiment).likelihood,
        reduction = get_config(experiment).reduction,
        dependency = get_dependency(experiment),
        name = nameof(experiment),
        alg = get_solve_alg(experiment),
        keep_data = true,
        prob_kwargs = (;),
        kwargs...)
    if ismissing(model_transformations)
        translate_model_transformations = false
        # don't duplicate callbacks
        model_transformations = get_model_transformations(experiment)
    else
        # model transformations changed, add the new callbacks
        translate_model_transformations = true
    end
    experiment_data, save_names = remake_ss_data(experiment, data, model, save_names,
        keep_data)
    dependency = dependency isa UUID ? get_dependency(experiment) : dependency
    # @debug ModelingToolkit.get_parameter_dependencies(model)

    config = construct_config(
        experiment_data, model, overrides, constraints, constraints_ts,
        nothing, model_transformations,
        callback, dependency, save_names, (), loss_func, postprocess,
        noise_priors, likelihood, reduction, name, alg,
        # we need to take care to not create the callbacks for model transformations twice
        Val(false), prob_kwargs, kwargs, translate_model_transformations)

    return SteadyStateExperiment(config)
end

function SciMLBase.remake(vp::ParameterEnsemble; u = vp.u, prob = vp.prob, alg = vp.alg,
        elapsed = missing)
    ParameterEnsemble(u, prob, alg, elapsed)
end

function virtual_experiment(experiment::AbstractExperiment, model; kwargs...)
    remake(experiment, model; keep_data = false, kwargs...)
end

"""
    remake_experiments(invprob::InverseProblem; kwargs...)

Remake all experiments of an inverse problem. Internally calls [`remake`](@ref) on each experiment.

# Positional Arguments

  - invprob: [`InverseProblem`](@ref).

# Keyword Arguments

These are passed to [`remake`](@ref) of each experiment.
"""
function remake_experiments(invprob::AbstractInverseProblem; kwargs...)
    experiments = []
    for experiment in get_experiments(invprob)
        remade_experiment = remake(experiment; kwargs...)
        push!(experiments, remade_experiment)
    end
    ss = get_search_space(invprob)
    experiments = parameterless_type(typeof(get_experiments(invprob)))(experiments)
    InverseProblem(experiments, ss)  # TODO: @Sebastian: add missing kwargs.
end
