get_config(experiment::AbstractExperiment) = experiment.config

get_model(ec::ExperimentConfig) = ec.model
get_model(e::AbstractExperiment) = get_model(get_config(e))

get_model_cache_id(ec::ExperimentConfig) = ec.model_cache_id
get_model_cache_id(e::AbstractExperiment) = get_model_cache_id(get_config(e))

get_overrides(ec::ExperimentConfig) = ec.overrides
get_u0(ec::ExperimentConfig) = ec.u0map
get_params(ec::ExperimentConfig) = ec.psmap
get_overrides(e::AbstractExperiment) = get_overrides(get_config(e))
get_u0(e::AbstractExperiment) = get_u0(get_config(e))
get_params(e::AbstractExperiment) = get_params(get_config(e))

get_constraints(ec::ExperimentConfig) = ec.constraints
get_constraints(e::AbstractExperiment) = get_constraints(get_config(e))

get_constraints_ts(ec::ExperimentConfig) = ec.constraints_ts
get_constraints_ts(e::AbstractExperiment) = get_constraints_ts(get_config(e))

get_data(ec::ExperimentConfig) = ec.data
get_data(experiment) = get_data(get_config(experiment))

get_name(tc::ExperimentConfig) = tc.name
get_name(experiment) = get_name(get_config(experiment))

get_postprocess(ec::ExperimentConfig) = ec.postprocess
get_postprocess(experiment::AbstractExperiment) = get_postprocess(get_config(experiment))

get_likelihood(ec::ExperimentConfig) = ec.likelihood
get_likelihood(e::AbstractExperiment) = get_likelihood(get_config(e))

# state_names are unknowns(model), so that they are in the same order as the unknowns in sol
# so that unknowns in sol and noise are paired correctly in the likelihood distribution
function get_noise_priors(t::AbstractExperiment, state_names)
    get_noise_priors(get_config(t), state_names)
end
function get_noise_priors(ec::ExperimentConfig, state_names)
    get_noise_priors(ec.noise_priors, state_names)
end

# For consistency, get_noise_priors always returns a vector.
# So the single Distributions.Sampleable is wrapped in a vector in this dispatch.
get_noise_priors(prior::Distributions.Sampleable, state_names) = [prior]

function get_noise_priors(prior_pairs::Vector{<:Pair}, state_names)
    priors = Distributions.Sampleable[]
    for s in state_names
        idx = findfirst(x -> isequal(first(x), s), prior_pairs)
        if !isnothing(idx)
            push!(priors, last(prior_pairs[idx]))
        end
    end
    # Tighten distributions type
    priors_tight = map(identity, priors)

    return priors_tight
end

get_solve_alg(::ODEProblem) = DefaultODEAlgorithm(autodiff = AutoForwardDiff())
get_solve_alg(::SteadyStateProblem) = FastShortcutNonlinearPolyalg()

get_solve_alg(ec::ExperimentConfig) = ec.alg
get_solve_alg(e::AbstractExperiment) = get_solve_alg(get_config(e))

get_save_idxs(ec::ExperimentConfig) = ec.save_idxs
get_save_idxs(e::AbstractExperiment) = get_save_idxs(get_config(e))

get_saveat(ec::ExperimentConfig) = ec.saveat
get_saveat(e::AbstractExperiment) = get_saveat(get_config(e))

function get_saveat(t::AbstractExperiment, x, prob)
    saveat = get_saveat(t)
    substitute_symbolic_saveat(saveat, x, prob)
end

get_model_transformations(ec::ExperimentConfig) = ec.model_transformations
get_model_transformations(e::AbstractExperiment) = get_model_transformations(get_config(e))

get_saved_model_variables(ec::ExperimentConfig) = ec.saved_model_variables
get_saved_model_variables(t::AbstractExperiment) = get_saved_model_variables(get_config(t))

get_save_everystep(ec::ExperimentConfig) = ec.save_everystep
get_save_everystep(e::AbstractExperiment) = get_save_everystep(get_config(e))
get_save_start(ec::ExperimentConfig) = ec.save_start
get_save_start(e::AbstractExperiment) = get_save_start(get_config(e))
get_save_end(ec::ExperimentConfig) = ec.save_end
get_save_end(e::AbstractExperiment) = get_save_end(get_config(e))

get_callbacks(ec::ExperimentConfig) = ec.callbacks
get_callbacks(e::AbstractExperiment) = get_callbacks(get_config(e))
get_callbacks(e::AbstractExperiment, x) = get_callbacks(get_config(e))

get_uuid(ec::ExperimentConfig) = ec.uuid
get_uuid(e::AbstractExperiment) = get_uuid(get_config(e))

get_prob(ec::ExperimentConfig) = ec.prob
get_prob(e::AbstractExperiment) = get_prob(get_config(e))

get_kwargs(ec::ExperimentConfig) = ec.kwargs
get_kwargs(e::AbstractExperiment) = get_kwargs(get_config(e))

get_loss_func(ec::ExperimentConfig) = ec.loss_func
get_loss_func(e::AbstractExperiment) = get_loss_func(get_config(e))

function get_diff_variables(model)
    only.(get_variables.(getproperty.(diff_equations(model), :lhs)))
end

has_initial(x) = iscall(unwrap(x)) && operation(unwrap(x)) isa Initial
peel_init(x) = wrap(only(arguments(unwrap(x))))

function split_overrides(overrides, model)
    psmap = filter(x -> is_parameter(model, x[1]), overrides)
    u0map = filter(x -> !is_parameter(model, x[1]), overrides)
    # workaround Initial not working in psmap
    for p in psmap
        if has_initial(p[1])
            push!(u0map, peel_init(p[1]) => p[2])
        end
    end

    u0map, psmap
end

warn_size(sol, ::NoData, name) = nothing

function warn_size(sol, data, name)
    ss = size(sol)
    sd = size(data)

    if length(ss) == 1 && ss ≠ sd
        @warn "Size of solution $ss and data $sd are not the same for $name"
    end
end

determine_dependency(::Nothing) = nothing
determine_dependency(dependency::AbstractExperiment) = get_uuid(dependency)
determine_dependency(dependency::UUID) = dependency

get_dependency(tc::ExperimentConfig) = tc.u0_dependency
get_dependency(e::AbstractExperiment) = get_dependency(get_config(e))
get_dependency(::Type{<:AbstractExperiment{D}}) where {D} = D

has_dependency(::AbstractExperiment) = false
has_dependency(::AbstractExperiment{<:UUID}) = true

saved_values(::Type{<:AbstractExperiment{FWD, S}}) where {FWD, S} = S

function promote_to_common_eltype(x)
    # bail out if x is empty
    isempty(x) && return x
    T = promote_type((typeof.(x))...)
    map(T, x)
end

function promote_to_concrete(x, T)
    E = eltype(x)
    # @debug "E: $E, T: $T"
    if isconcretetype(T)
        if isconcretetype(E)
            return promote_type(E, T)
        elseif isempty(x)
            return T
        else
            better_x_type = eltype(map(identity, x))
            return promote_type(better_x_type, T)
        end
    else
        if isconcretetype(E)
            # we have to try fixing again later
            return promote_type(E, T)
        else
            better_x_type = eltype(map(identity, x))
            return promote_type(better_x_type, T)
        end
    end
end

# function alias_and_transform(experiment, invprob, x_original)
#     # mc = get_cache(invprob, experiment)
#     # Alias local parameters
#     # @debug "x before replacement: $x_original"
#     # x_aliased = alias_x(x_original, experiment, mc)
#     x = copy(x_original)#copy(x_aliased)
#     # @debug "x after replacement: $x_aliased"
#     transform_params!(:inverse, x, invprob)
#     # @debug "transformed part of: $x"
#     return x
# end

function match_search_space_order(x::AbstractVector{<:Pair}, ss)
    map(x) do (k, v)
        i = indexof(k, collect(ss))
        if !isnothing(i)
            last(x[i])
        else
            error("index of $k not found in $(ss)!")
        end
    end
end

function match_search_space_order(x::NamedTuple, ss)
    match_search_space_order(collect(pairs(x)), ss)
end
