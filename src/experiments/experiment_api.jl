function setup_problem(experiment::AbstractExperiment, invprob,
        x_original::AbstractVector = initial_state(Any, invprob),
        default_u0 = Val(:default);
        tspan = timespan(experiment))#, x_original, invprob)) # FIXME: figure this out
    x = copy(x_original)
    transform_params!(:inverse, x, invprob)

    # @debug "x: $x"
    ist = get_internal_storage(invprob)

    prob0 = prepare_initial_prob(experiment, invprob, x, default_u0)

    setter = ist.param_setter
    i = find_index(experiment, invprob)
    isnothing(i) && error("Experiment not found!")
    (u0, p) = setter(prob0, i, x)
    # @debug "updated params: $(p.tunable)"
    # @debug "u0 before remake: $u0"

    if isnothing(tspan)
        # avoid overwriting internally assigned tspan with nothing
        # https://github.com/JuliaComputing/DyadModelOptimizer.jl/issues/1005
        prob = remake(prob0; u0, p)
    else
        prob = remake(prob0; u0, p, tspan)
    end
    # @debug prob.u0

    return prob
end

function prepare_initial_prob(experiment, invprob, x, default_u0)
    prob0 = get_prob(experiment)

    u0 = determine_initial_u0(experiment, x, invprob, default_u0)
    # @debug u0 name=nameof(experiment)

    if u0 isa Val{:default}
        # @debug "early exit for $(nameof(experiment))"
        return prob0
    end

    config = get_config(experiment)

    (newu0, p) = config.dependency_handling.set_dependent_u0(prob0, u0)
    # @debug newu0

    remake(prob0; u0 = newu0, p)
end

function get_updated_param(p, experiment, invprob, x)
    prob = setup_problem(experiment, invprob, x)
    integ = init(prob, get_solve_alg(experiment))

    # TODO cache the getp
    getsym(prob, p)(integ)
end

function setup_problem(ex::AbstractExperiment, prob, x::CalibrationResult,
        default_u0 = Val(:default);
        tspan = timespan(ex))#, x, prob)) # FIXME
    # we don't need to apply the direct transformation, as x.original is
    # not transformed back to the linear space
    setup_problem(ex, prob, x.original, default_u0; tspan)
end

function setup_problem(ex::AbstractExperiment, prob,
        x::Union{AbstractVector{<:Pair}, NamedTuple},
        default_u0 = Val(:default);
        tspan = timespan(ex))#, x, prob)) # FIXME
    ist = get_internal_storage(prob)
    bi = ist.bi
    ordered_x = match_search_space_order(x, variable_symbols(bi))
    # @debug ordered_x
    ist = get_internal_storage(prob)
    # x′ = transform_params(:direct, ordered_x, prob)
    # @debug "x after transform: $x′"
    setup_problem(ex, prob, ordered_x, default_u0; tspan)
end

# this experiment needs ss_u0, and it was provided
function determine_initial_u0(experiment::AbstractExperiment, x, invprob, default_u0)
    default_u0
end

# this experiment needs ss_u0, but it wasn't provided
function determine_initial_u0(experiment::AbstractExperiment{<:UUID}, x, invprob,
        ::Val{:default})
    experiments = get_experiments(invprob)
    uuid = get_dependency(experiment)
    config = get_config(experiment)
    i = findfirst(t -> get_uuid(t) == uuid, experiments)
    if !isnothing(i)
        required_experiment = experiments[i]
        @debug "solving experiment $(nameof(required_experiment)) is needed."
        sol = trysolve(required_experiment, invprob, x)
        isnothing(sol) && error("solve failed")
        u = get_postprocess(required_experiment)(sol)
        @debug "postprocessed u: $u" name=nameof(required_experiment)
        config.dependency_handling.getu0_vals(u)
    else
        @error "Couldn't find dependency for $(nameof(experiment)). Using defaults."
        get_default_u0(invprob, x, experiment)
    end
end

"""
    simulate(experiment::AbstractExperiment, prob::InverseProblem, x)

Simulate the given `experiment` using optimization-state point `x`, which contains values for each
parameter and initial condition that is optimized in [`InverseProblem`](@ref) `prob`.
"""
function DyadInterface.simulate(
        experiment::AbstractExperiment, invprob::AbstractInverseProblem,
        x = initial_state(Any, invprob),
        default_u0 = Val(:default);
        tspan = timespan(experiment),#, x, invprob), # FIXME: figure this out
        saveat = get_saveat(experiment),#, x, invprob),
        save_idxs = get_save_idxs(experiment),
        alg = get_solve_alg(experiment),
        kwargs = get_kwargs(experiment),
        extra_kwargs...)
    prob = setup_problem(experiment, invprob, x, default_u0; tspan)
    merged_kwargs = isempty(extra_kwargs) ? kwargs : merge(kwargs, extra_kwargs)

    # We want to help inference and provide the alg only via a positional arg

    solve(prob, alg;
        save_idxs,
        saveat,
        merged_kwargs...)
end

function DyadInterface.simulate(ex::AbstractExperiment, res::CalibrationResult;
        tspan = timespan(ex),# res.original, res.prob), # FIXME: figure this out
        saveat = get_saveat(ex, res.original, res.prob),
        kwargs...)
    # we don't need to apply the direct transformation, as x.original is
    # not transformed back to the linear space
    # @debug x′
    simulate(ex, res.prob, res.original; tspan, saveat, kwargs...)
end

function trysolve(experiment, invprob, x, default_u0 = Val(:default))
    try
        simulate(experiment, invprob, x, default_u0)
    catch e
        e isa UNEXPECTED_EXCEPTION && rethrow()
        SOLVE_ERR_WARN && @warn "solve errored with $(typeof(e))" maxlog=5 exception=e
    end
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(trysolve), args...)
    rrule_via_ad(config, simulate, args...)
end

function replace_sol(experiment::T, sol, args...) where {T <: AbstractExperiment}
    replace_sol(saved_values(T), experiment, sol, args...)
end

function replace_sol(::Type{Val{true}}, experiment, sol)
    model_vars = get_saved_model_variables(experiment)
    get_vals = get_config(experiment).get_saved_vals
    u = get_vals(sol)

    # only the 2 arg constructor has adjoints
    DiffEqArray(u, sol.t)
end

function replace_sol(::Type{Val{true}}, experiment::SteadyStateExperiment, sol)
    model_vars = get_saved_model_variables(experiment)
    get_vals = get_config(experiment).get_saved_vals
    u = get_vals(sol)
end

function replace_sol(::Type{Val{true}}, experiment, sol, saveat_slice, save_idxs)
    model_vars = get_saved_model_variables(experiment)
    if length(model_vars) == 1
        model_vars = only(model_vars)
        u = Vector{eltype(sol)}(undef, length(saveat_slice))
    else
        u = Vector{Vector{eltype(sol)}}(undef, length(saveat_slice))
    end
    syms = variable_symbols(sol)
    obs = get_config(experiment).get_obs
    @inbounds for (i, t) in enumerate(saveat_slice)
        @views u[i] = obs(sol(t), parameter_values(sol), t)
    end

    # only the 2 arg constructor has adjoints
    DiffEqArray(u, saveat_slice)
end

replace_sol(::Type{Nothing}, experiment, sol) = sol

function replace_sol(::Type{Nothing}, experiment, sol, saveat_slice, save_idxs)
    sol(saveat_slice, idxs = save_idxs)
end

function replace_sol(::Type{Nothing}, experiment, sol, ::Tuple{}, save_idxs)
    @views sol[save_idxs, :]
end

replace_sol(::Type{Nothing}, experiment, sol, ::Tuple{}, ::Nothing) = sol

@inline function compute_error(experiment,
        prob,
        x,
        sol,
        data = get_data(experiment);
        loss_func = get_loss_func(experiment))
    sol = replace_sol(experiment, sol)
    # warn_size(sol, data, nameof(experiment))
    loss_func_wrapper(loss_func, prob, x, sol, data)
end

@inline function compute_error(experiment,
        prob,
        x,
        sol,
        data,
        saveat_slice,
        save_idxs;
        loss_func = get_loss_func(experiment))
    sol = replace_sol(experiment, sol, saveat_slice, save_idxs)
    # warn_size(sol, data, nameof(experiment))
    loss_func_wrapper(loss_func, prob, x, sol, data)
end

@inline function loss_func_wrapper(loss_func, prob, x, sol, data; kw...)
    loss_func_wrapper(has_fast_indexing(loss_func), loss_func, prob, x, sol, data; kw...)
end

function loss_func_wrapper(::Val{false}, loss_func, prob, x, sol, data; kw...)
    loss_func(to_dict(x, prob), sol, data; kw...)
end
function loss_func_wrapper(::Val{true}, loss_func, prob, x, sol, data; kw...)
    loss_func(x, sol, data; kw...)
end

@inline gsa_reduction(experiment, sol) = experiment.config.reduction(sol)

hasdata(experiment::AbstractExperiment) = hasdata(get_data(experiment))
has_fast_indexing(::Any) = Val(false)
