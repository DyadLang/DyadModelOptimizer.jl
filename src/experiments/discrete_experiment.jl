struct DiscreteExperiment{FWD, S, C <: ExperimentConfig{FWD}, T, DT} <:
       AbstractExperiment{FWD, S}
    config::C
    tspan::T
    dt::DT
end

step_size(exp::DiscreteExperiment) = exp.dt

function DiscreteExperiment(config::C,
        tspan::Tuple{A, B},
        dt::DT) where {FWD, S, C <: ExperimentConfig{FWD, S}, A, B, DT}
    DiscreteExperiment{FWD, S, C, Tuple{A, B}, DT}(config, tspan, dt)
end

"""
    DiscreteExperiment(data, model, dt; kwargs...)

Describes a experiment that is simulated in discrete time with the time increment `dt`.
This object can be initialized in the same way as an [`Experiment`](@ref) object,
with the only difference being that `dt` is an additional argument here. The simulation
for this experiment type corresponds to solving a `DiscreteProblem`.

See [the SciML documentation](https://docs.sciml.ai/DiffEqDocs/stable/types/discrete_types/) for background information on discrete time problems.
"""
function DiscreteExperiment(data, model, dt;
        constraints = [],
        u0 = Pair{Num, Nothing}[],
        params = Pair{Num, Nothing}[],
        model_transformations = (),
        callback = CallbackSet(),
        indepvar = :timestamp,
        tspan = determine_tspan(data, indepvar),
        save_names = determine_save_names(data, model),
        saveat = determine_saveat(data, indepvar),
        constraints_ts = determine_constraints_ts(constraints, saveat, tspan),
        loss_func = determine_err(data),
        postprocess = last,
        noise_priors = Distributions.InverseGamma(2, 3),
        likelihood = get_Normal_likelihood(save_names, noise_priors),
        reduction = identity,
        dependency = nothing,
        name = "Experiment",
        alg = missing,
        # TODO: simplify writing error functions
        do_not_replace_sol = Val(false),
        prob_f_kwargs = (;),
        kwargs...)
    config = construct_config(data, model, constraints, constraints_ts,
        tspan, u0, params, model_transformations,
        callback, dependency, save_names, saveat, loss_func, postprocess,
        noise_priors, likelihood, reduction, name, alg,
        do_not_replace_sol, prob_f_kwargs, kwargs, indepvar)
    return DiscreteExperiment(config, tspan, dt)
end

function setup_problem(::DiscreteExperiment, model, f, u0, tspan, p, callback, ist, x)
    DiscreteProblem{true, FullSpecialize}(f, u0, tspan, p; callback)
end

setup_prob(model) = DiscreteFunction{true, FullSpecialize}(model)

get_solve_alg(::DiscreteExperiment) = FunctionMap()
