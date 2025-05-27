
abstract type AbstractExperiment{FWD, S} end

include("config.jl")

struct Experiment{FWD, S, C <: ExperimentConfig{FWD}, T} <: AbstractExperiment{FWD, S}
    config::C
    tspan::T
end

function Experiment(config::C,
        tspan::Tuple{A, B}) where {FWD, S, C <: ExperimentConfig{FWD, S}, A, B}
    Experiment{FWD, S, C, Tuple{A, B}}(config, tspan)
end

include("data.jl")
include("determine_kwargs.jl")
include("loss.jl")
include("design_config.jl")
include("steady_state.jl")
include("discrete_experiment.jl")
include("experiment_collections.jl")
include("experiment_api.jl")
include("datasets.jl")
include("utils.jl")

"""
    Experiment(data, model; kwargs...)

The `Experiment` describes an experiment in which `data` was obtained. The dynamics of the investigated
system are represented in `model` as an [ODESystem](https://mtk.sciml.ai/stable/systems/ODESystem/#ODESystem).
The experiment is used within the optimization problem, as part of [`InverseProblem`](@ref) to fit
the unknown `model` parameters and initial conditions to `data`.
If there is no data or if no data is needed, one can just use `Experiment(model; kwargs...)`, i.e.
just avoid passing the `data` argument. If the data is not passed, then one must provide the `tspan`
argument, representing the timespan used for integrating the differential equations of the model, otherwise this is inferred
from the data.

When simulating an `Experiment`, the parameters and initial conditions that are used in the equations are based on
the following hierarchy: If the parameter is fixed by the experiment, that has the highest priority, otherwise
if the parameter is to be estimated (i.e. it's present in the search space), than the estimated value is used,
otherwise, the default value obtained from the model definition is used. When we say that a parameter is fixed
by the experiment, that is meant to reflect the conditions under which the experiment was conducted.
In this way we can have one experiment estimating a parameter that is known (or fixed) in another one at the same time.
For example let's consider that we have two separate experiments in the inverse problem.
The first experiment is characterized by knowing one parameter value, say `a=1`.
This means that for the first experiment we'll have to fix the value of the known parameter to its known value.
The value of this parameter is not known for the other experiment. We want to find `a` and also make use of what we know
from experiment 1, so in this case we can set `params=[a=>1]` only for experiment 1 and have `a` in the search space for experiment 2.
With this configuration, when we simulate the experiments, the tuned value for `a` will be ignored in the first experiment
and the fixed (i.e. `a=1`) value will be used, while the second experiment will make use of it. Since the first experiment
also contributes to the overall objective value associated with the inverse problem, we are making use of the information from
the first experiment where the (globally) unknown parameter is known in a particular case.
In order to specify the fixed parameters or initial conditions, one can use the `params` keyword argument for the
parameters (e.g. `params = [a => specific_value, b => other_value]`) and `u0` for the initial
conditions (e.g. `u0 = [state_name => custom_initial_value]`). The fixed values for the parameters and initial conditions can also
be parametrized. For example if `a` is in the search space, we can have the initial condition for a state `u1` to be fixed to `2*a`.
In this case the value will depend on the tuned value of `a` and will be different based on what tuned values are tried for `a`, but
the relation `u1=2a` will always hold.

The contribution of the `Experiment` to the cost function is computed using the [`squaredl2loss`](@ref) function
by default, but this can be changed by using the `err` keyword argument
(e.g. `loss_func = (tuned_vals, sol, data) -> compute_error`). The function requires 3 arguments,
the tuned values of the parameters or initial conditions (i.e. what was provided as search space),
the solution of the experiment and the data and is expected to return a scalar value corresponding
to the loss of the experiment.

## Positional arguments

  - `data`: A `DataSet` or a tabular data object. If there is no data or no data is needed, this can be omitted.
  - `model`: An `ODESystem` describing the model that we are using for the experiment. The model needs to define defaults
    for all initial conditions and all parameters.

## Keyword arguments

  - `overrides`: a vector of `Pair`s indicating overrides passed to the `ODEProblem` constructor (e.g. `overrides = [unknown_name => value]`).
  - `constraints`: a vector of equations representing equality or inequality constraints using model variables which are required to be satisfied during optimization.
  - `model_transformations`: Apply some transformations to the model used in this experiment,
    such as using [`DiscreteFixedGainPEM`](@ref) (e.g. `model_transformations = (DiscreteFixedGainPEM(alpha),)`)
  - `callback`: A callback or a callback set to be used during the simulation.
    See https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/ for more details.
  - `tspan`: the timespan to use when integrating the equations. If data is passed, than it is automatically determined.
  - `depvars`: the names of model variables or data columns to use from the given data. By default all model parameters and initial
    conditions that are present in the data are used. This argument should be used only if one wants to use only a subset of the available data.
  - `saveat`: the times at which the solution of the differential equations will be saved. If the data is passed,
    the times for which we have data will be used and this argument does not need to be passed. If this argument is provided when using data,
    care must be taken in the experiment loss function, such that the correct time points are used.
  - `constraints_ts`: the times at which time dependent constraints should be evaluated. Defaults to `saveat`.
  - `loss_func`: the contribution to the loss corresponding to this experiment (e.g. `loss_func = (tuned_vals, sol, data) -> compute_error`).
  - `prob_kwargs`: A `NamedTuple` indicating keyword arguments to be passed to the `ODEProblem` constructor.
    See [the ModelingToolkit docs](https://docs.sciml.ai/ModelingToolkit/stable/systems/ODESystem/#SciMLBase.ODEProblem-Tuple%7BModelingToolkit.AbstractODESystem,%20Vararg%7BAny%7D%7D)
    for more details.
  - `dependency`: This keyword can be be assigned to a variable representing an other (previously defined) experiment to express the
    fact that the initial conditions for this experiment depend on the solution of the given experiment. For example if one
    experiment (e.g. `experiment_ss`) defines a steady state, we can use that for the definition of the initial conditions
    for a subsequent experiment with `dependency=experiment_ss`.

If additional keywords are passed, they will be forwarded to the `solve` call from DifferentialEquations. For example,
one can pass `alg=Tsit5()` to specify what solver will be used. More information about supported
arguments can be found [here](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/#solver_options).
"""
function Experiment(data,
        model::AbstractTimeDependentSystem;
        overrides = Pair[],
        constraints = [],
        model_transformations = (),
        callback = CallbackSet(),
        indepvar = :timestamp,
        tspan = determine_tspan(data, indepvar),
        depvars = determine_save_names(data, model),
        saveat = determine_saveat(data, indepvar),
        constraints_ts = determine_constraints_ts(constraints, saveat, tspan),
        loss_func = determine_err(data),
        postprocess = last,
        noise_priors = Distributions.InverseGamma(2, 3),
        likelihood = get_Normal_likelihood(depvars, noise_priors),
        reduction = identity,
        dependency = nothing,
        name = "Experiment",
        alg = missing,
        # TODO: simplify writing error functions
        do_not_replace_sol = Val(false),
        prob_kwargs = (;),
        kwargs...)
    config = construct_config(
        data, model, overrides, constraints, constraints_ts,
        tspan, model_transformations,
        callback, dependency, depvars, saveat, loss_func, postprocess,
        noise_priors, likelihood, reduction, name, alg,
        do_not_replace_sol, prob_kwargs, kwargs, indepvar)
    return Experiment(config, tspan)
end

function extend_with_dependencies(model, u0, params)
    isempty(u0) && isempty(params) && return model
    deps = ODESystem(
        [], ModelingToolkit.get_iv(model); parameter_dependencies = params,
        defaults = u0,
        name = :deps)
    _model = isnothing(get_parent(model)) ? model : get_parent(model)
    return structural_simplify(extend(deps, _model, name = nameof(model)))
end

"""
    JuliaHubJob(; node_specs, auth, batch_image, dataset_name, alg)

This is the container for storing various pieces required for submitting batch jobs on JuliaHub using [JuliaHub.jl](https://help.juliahub.com/julia-api/stable/).

## Keyword Arguments

  - `node_specs`: Specifications of the required compute in JuliaHub as a NamedTuple like `ncpu`, `memory` etc. Look at https://help.juliahub.com/julia-api/stable/reference/job-submission/#Compute-configuration for different configurations available.
  - `auth`: Authentication object of type `JuliaHub.Authentication` for verification while performing various operations on Juliahub.
  - `batch_image`: Job image to be used for the batch job. This is of type `JuliaHub.BatchImage`.
  - `dataset_name`: Name of the dataset in which the result of the computation is serialised and uploaded.
  - `alg`: Algorithm object used in either [`calibrate`](@ref) or [`parametric_uq`](@ref).
"""
Base.@kwdef struct JuliaHubJob{S, A, B, D, AL}
    node_specs::S
    auth::A
    batch_image::B
    dataset_name::D
    alg::AL
end

function setup_prob(model::ODESystem, u0map, psmap, tspan, kwargs)
    @debug "u0map: $u0map"
    @debug "psmap: $psmap"
    ODEProblem{true, FullSpecialize}(model, u0map, tspan, psmap; kwargs...)
end

function setup_prob(model::ODESystem, u0map, psmap, ::Nothing, kwargs)
    SteadyStateProblem{true}(model, u0map, psmap; kwargs...)
end

function timeseries_data(data::Vector{DataFrame}, save_names, indepvar, saveat, tspan,
        expect_match = false)
    length(data) == 1 &&
        return timeseries_data(only(data), save_names, indepvar, saveat, tspan)
    # For replicate data we don't expect the tspan and saveat to match the time in the data
    experiment_data = map(
        df -> timeseries_data(df, save_names, indepvar, saveat, tspan, expect_match),
        data)
    # if length(unique(typeof.(experiment_data))) > 1  # Replicate data may differ in the parameter that indicates mismatch.
    #     error("Replicate measurements don't have same data type!")
    # end
    ReplicateData(experiment_data, Symbol.(save_names))
end

function timeseries_data(data, save_names, indepvar, saveat, tspan, expect_match = true)
    if isnothing(save_names)
        # we want to save data, but it doesn't have the correct format
        # the user has to write the error function; this is not officially supported
        @debug "Bypassing user data interpretation. You must handle the error function!"
        return data
    end

    data_elt = data_eltype(data)
    sym_save_names = if save_names isa Vector{<:Pair}
        Symbol.(first.(save_names))
    elseif save_names isa Union{Vector, SaveAllUnknowns}
        Symbol.(save_names)
    else
        Symbol(save_names)
    end
    column_names = if save_names isa Vector{<:Pair}
        Symbol.(last.(save_names))
    else
        sym_save_names
    end

    @debug "column_names: $column_names"
    unused_columns = filter(
        x -> !(x in column_names) && x != indepvar, Tables.columnnames(data))
    !isempty(unused_columns) &&
        @warn "Columns $unused_columns in the data cannot be used as they are not present in the model."
    if data_elt <: Union{Number, Missing}
        TimeSeriesData(
            data, sym_save_names, column_names, indepvar, saveat, tspan, expect_match)
    elseif data_elt <: Tuple
        BoundsData(
            data, sym_save_names, column_names, indepvar, saveat, tspan, expect_match)
    elseif isnothing(data_elt)
        timeseries_data(nothing, sym_save_names, indepvar, saveat, tspan, expect_match)
    else
        error("Data format not supported!")
    end
end

timeseries_data(::Nothing, save_names, indepvar, saveat, tspan, expect_match) = NoData()

Base.nameof(experiment::AbstractExperiment) = get_name(experiment)
