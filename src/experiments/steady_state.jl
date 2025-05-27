struct SteadyStateExperiment{FWD, S, C <: ExperimentConfig} <: AbstractExperiment{FWD, S}
    config::C
end

function SteadyStateExperiment(config::C) where {FWD, S, C <: ExperimentConfig{FWD, S}}
    SteadyStateExperiment{FWD, S, C}(config)
end

"""
    SteadyStateExperiment(data, model; kwargs...)

Describes a experiment that is ran until a steady state is reached. This object can be initialized
in the same way as a [`Experiment`](@ref) object, with the only difference being that `data` needs
to be a `Vector` here. The `data` in this case represents the values of the saved states when
the system has reached its steady state. The simulation for this experiment type corresponds
to solving a `SteadyStateProblem`.

See [the SciML documentation](https://docs.sciml.ai/DiffEqDocs/stable/types/steady_state_types/#Steady-State-Problems) for background information on steady state problems.
"""
function SteadyStateExperiment(data, model::AbstractTimeDependentSystem;
        overrides = Pair[],
        constraints = [],
        constraints_ts = (),
        model_transformations = (),
        callback = CallbackSet(),
        depvars = determine_save_names(data, model, false),
        save_names = nothing,
        loss_func = determine_err(data),
        noise_priors = Distributions.InverseGamma(2, 3),
        likelihood = get_Normal_likelihood(save_names, noise_priors),
        reduction = identity,
        dependency = nothing,
        name = "SteadyStateExperiment",
        postprocess = identity,
        alg = missing,
        prob_kwargs = (;),
        kwargs...)

    # _, save_idxs = determine_what_to_save(model, save_names)
    experiment_data = maybe_steadystate(data, depvars)

    config = construct_config(
        experiment_data, model, overrides, constraints, constraints_ts,
        nothing, model_transformations,
        callback, dependency, depvars, (), loss_func, postprocess,
        noise_priors, likelihood, reduction, name, alg,
        # we need to take care to not create the callbacks for model transformations twice
        Val(false), prob_kwargs, kwargs)

    SteadyStateExperiment(config)
end

function SteadyStateExperiment(data::AbstractVector{<:Pair},
        model::AbstractTimeDependentSystem;
        kwargs...)
    _data = NamedTuple(data)
    SteadyStateExperiment(_data, model; kwargs...)
end

function SteadyStateExperiment(data::DataFrameRow,
        model::AbstractTimeDependentSystem;
        kwargs...)
    # convenience constructor for DataFrameRows
    _data = [Symbol(k) => [v] for (k, v) in zip(names(data), collect(data))]
    SteadyStateExperiment(_data, model; kwargs...)
end

function maybe_steadystate(data, save_names)
    if symbolic_type(save_names) == NotSymbolic()
        SteadyStateData(data, Symbol.(save_names))
    else
        SteadyStateData(data, Symbol(save_names))
    end
end

function maybe_steadystate(data, save_names::AbstractVector)
    SteadyStateData(data, Symbol.(save_names))
end
maybe_steadystate(::Nothing, save_names) = NoData()
maybe_steadystate(::Nothing, ::AbstractVector) = NoData()

# SteadyStateExperiment doesn't have tspan
timespan(::SteadyStateExperiment, x, invprob) = nothing
timespan(::SteadyStateExperiment, ::CalibrationResult, prob) = nothing
timespan(::SteadyStateExperiment) = nothing

function postprocess_initial_experiment(ss_sol::SciMLBase.ODESolution,
        ss_experiment,
        prob,
        x)
    save_idxs = get_save_idxs(ss_experiment)
    saveat = get_saveat(ss_experiment)
    sol = replace_sol(ss_experiment, ss_sol, saveat, save_idxs)
    ss_u0 = get_postprocess(ss_experiment)(ss_sol)
    err = compute_error(ss_experiment, prob, x, sol)

    return ss_u0, err
end

function postprocess_initial_experiment(ss_sol::SciMLBase.NonlinearSolution,
        ss_experiment,
        prob,
        x)
    save_idxs = get_save_idxs(ss_experiment)
    if !isnothing(save_idxs)
        sol = ss_sol[save_idxs]
    else
        sol = ss_sol
    end
    ss_u0 = get_postprocess(ss_experiment)(ss_sol)
    err = compute_error(ss_experiment, prob, x, sol)

    return ss_u0, err
end

postprocess_initial_experiment(::Nothing, sse, prob, x) = nothing, Inf

function initial_simulation(prob, x)
    ss_experiment = get_experiments(prob).initial
    ss_prob = setup_problem(ss_experiment, prob, x)
    ss_sol = try
        alg = get_solve_alg(ss_experiment)

        kwargs = get_kwargs(ss_experiment)

        solve(ss_prob, alg; kwargs...)
    catch e
        SOLVE_ERR_WARN && @warn "solve failed with $(typeof(e))" maxlog=3 exception=e
    end

    postprocess_initial_experiment(ss_sol, ss_experiment, prob, x)
end
