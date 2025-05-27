get_internal_storage(c::AbstractDesignAnalysis) = c.internal_storage

function search_space_defaults(c::AbstractDesignAnalysis)
    search_space_defaults(get_internal_storage(c))
end

search_space_names(a::AbstractDesignAnalysis) = search_space_names(get_internal_storage(a))

get_defaults(a::AbstractDesignAnalysis) = get_defaults(get_internal_storage(a))

function get_default_u0(::AbstractDesignAnalysis, x, experiment)
    state_values(get_prob(experiment))
end

function get_the_only_model(prob::AbstractDesignAnalysis)
    # TODO: Delete this and refactor API for multiple models
    models = collect_all(get_model, get_experiments(prob))
    if length(models) ≠ 1
        error("This function only works with one model in the inverse problem.")
    end
    # all models are the same
    only(models)
end

"""
    function get_model(res::CalibrationResult)

This function retrieves the model associated with a `CalibrationResult`.
"""
get_model(res::CalibrationResult) = get_the_only_model(res.prob)

get_experiments(a::AbstractDesignAnalysis) = a.experiments
get_experiments(ps::AbstractParametricUncertaintyEnsemble) = get_experiments(ps.prob)
get_experiments(res::CalibrationResult) = get_experiments(res.prob)

model_unknowns(a, ex::AbstractExperiment) = variable_symbols(get_model(ex))

function model_parameters(a::AbstractDesignAnalysis, ex::AbstractExperiment)
    parameter_symbols(get_model(ex))
end

function get_ss_transformations(a::AbstractDesignAnalysis)
    get_ss_transformation(get_internal_storage(a))
end

get_ss_transformation(ist::InternalStorage) = ist.ss_transformation

function get_Normal_likelihood(save_names, noise_priors::Vector{<:Pair})
    #=
    This is the default likelihood, set to a (multivariate) Normal distribution.
    The dimensionality of the distribution depends on the number of unknowns saved at each timepoint.
    u is the state vector at each timepoint with length(u) = length(save_names).
    s is the vector of noise_priors, in this case the standard deviation of the Normal.
    We can only have length(s)==length(u) in this dispatch,
    each pair in noise_priors matches a state in save_names
    =#
    if length(noise_priors) == 1
        if length(save_names) == 1
            return (u, s) -> Distributions.Normal(u, only(s))
        else
            # doe we need abs2?
            return (u, s) -> Distributions.MvNormal(u, Diagonal(only(s)))
        end
    else
        # do we need abs2?
        return (u, s) -> Distributions.MvNormal(u, Diagonal(map(abs2, s)))
    end
end

function get_Normal_likelihood(save_names, noise_prior::Distributions.Sampleable)
    #=
    This is the default likelihood, set to a (multivariate) Normal distribution.
    This dispatch is the case of a single noise parameter for all unknowns.
    =#
    if length(save_names) == 1
        return (u, s) -> Distributions.Normal(u, only(s))
    else
        return (u, s) -> Distributions.MvNormal(u, only(s) * Distributions.I)
    end
end

function get_Normal_likelihood(::Nothing, noise_priors::Distributions.Sampleable)
    (u, s) -> error("No likelihood defined.")
end

function determine_batch_size(::EnsembleDistributed, n)
    nw = nworkers()
    b = n ÷ (nw > 0 ? nw : 1)

    return b == 0 ? 1 : b
end

function check_retcode(sol::AbstractSciMLSolution, name)
    if !successful_retcode(sol.retcode)
        @ignore_derivatives SOLVE_FAIL_WARN &&
                            @warn "Solve failed with $(sol.retcode) for $name." maxlog=3
        return Inf
    else
        zero(eltype(sol))
    end
end

check_retcode(sol, name) = zero(eltype(sol))
check_retcode(::Nothing, name) = Inf

subval(t, d) = value(substitute(t, d))

# TODO: refactor this
function ss_unknowns_dict(mc, x)
    defs = mc.params_defaults
    sx = zip(search_space_parameters(mc), x)
    # merge the defaults with the current values from x
    merge(defs, Dict(map(((s, v),) -> s => v, sx)))
end

timespan(e::AbstractExperiment) = e.tspan

function substitute_symbolic_tspan(tspan::Union{Tuple{T, <:Num}, Tuple{<:Num, T}}, x,
        mc) where {T}
    d = ss_unknowns_dict(mc, x)
    substitute_symbolic_tspan(tspan, d)
end

substitute_symbolic_tspan(t, x, prob) = t
function substitute_symbolic_tspan(t::Tuple{T, <:Num}, dict) where {T}
    (first(t), subval(last(t), dict))
end
function substitute_symbolic_tspan(t::Tuple{<:Num, T}, dict) where {T}
    (subval(first(t), dict), last(t))
end
function substitute_symbolic_tspan(t::NTuple{2, <:Num}, dict)
    (subval(first(t), dict), subval(last(t), dict))
end

function substitute_symbolic_saveat(saveat::Union{Num, Vector{Num}}, x, prob)
    d = ss_unknowns_dict(prob, x)
    substitute_symbolic_saveat(saveat, d)
end

substitute_symbolic_saveat(saveat::Num, d) = subval(saveat, d)
substitute_symbolic_saveat(saveat::Vector{Num}, d) = subval.(saveat, (d,))
substitute_symbolic_saveat(saveat, x, prob) = saveat

warn_num_tspan(::Any) = nothing
function warn_num_tspan(::Tuple{T, <:Num}) where {T}
    @warn "Could not obtain a concrete value for the timespan."
end
function warn_num_tspan(::Tuple{<:Num, T}) where {T}
    @warn "Could not obtain a concrete value for the timespan."
end
function warn_num_tspan(::NTuple{2, <:Num})
    @warn "Could not obtain a concrete value for the timespan."
end
warn_num_tspan(::Nothing) = nothing

indexof(sym::Symbol, syms) = findfirst(isequal(sym), Symbol.(syms))
indexof(sym, syms) = findfirst(isequal(sym), syms)
indexof(sym, eqs::Vector{Equation}) = findfirst(isequal(sym), getproperty.(eqs, :lhs))

function indexof(sym::Symbol, eqs::Vector{Equation})
    findfirst(isequal(sym), Symbol.(getproperty.(eqs, :lhs)))
end

ispresent(sym, syms) = !isnothing(indexof(sym, syms))

function col_in_vars(col_name, model, model_unknowns)
    !isnothing(find_corresponding_model_var(col_name, model, model_unknowns))
end

function observed_names(model)
    obs_eq = observed(model)
    vars = [] # avoid specialization

    for eq in obs_eq
        push!(vars, eq.lhs)
    end

    return vars
end

function variable_index_in_data(variable, data, model)
    findfirst(isequal(variable), parse_variable.((model,), string.(data.save_names)))
end

function find_corresponding_model_var(var, args...)
    find_corresponding_model_var(string(var), args...)
end

function find_corresponding_model_var(
        var::AbstractString,
        model,
        model_unknowns,
        obs = observed_names(model)
)
    # Is this needed? I don't remember why this was added...
    # but now due to https://github.com/JuliaSymbolics/Symbolics.jl/issues/1364
    # we have inconsistencies in what get_variables returns as we might get the t or not
    # filter!(x -> length(get_variables(x)) == 1, obs)
    v = vcat(model_unknowns, obs)
    idx = find_corresponding_model_var_index(var, model, v)
    isnothing(idx) && return nothing
    return v[idx]
end

function find_corresponding_model_var_index(var::AbstractString, model, vars)
    model_names = map(string, vars)::Vector{String} # help type inference

    state_name = replace(var, "₊" => ".")
    model_names = replace.(model_names, ("₊" => ".",))
    exact_match = findfirst(==(state_name), model_names)

    # we can't have a state name matching multiple model names
    # as that would mean that model names are not unique
    if !isnothing(exact_match)
        return exact_match
    end

    # try to find partial matches where the independent variable
    # dependence is omitted
    t = string(ModelingToolkit.get_iv(model))
    re = Regex("^$(state_name)(\\($t\\))?\\z")
    partial_matches = findall(n -> !isnothing(match(re, n)), model_names)

    if length(partial_matches) == 1
        only(partial_matches)
    elseif isempty(partial_matches)
        nothing
    else
        # length(partial_matches) > 1
        error("$var matches more than one model name: $(join(model_names[partial_matches], ", ", " and ")).")
    end
end

function find_unknowns_in_data(data, model, model_unknowns::Vector)
    # we overwrite the last argument with an empty vector to just look at the unknowns and not
    # include the observed
    map(x -> find_corresponding_model_var(x, model, model_unknowns, []),
        string.(data.save_names))
end

function unknowns_to_idxs(state_names, model)
    idxs = indexof.(state_names, (unknowns(model),))
    @debug "idxs for model unknowns: $idxs"
    if length(idxs) == 1
        only(idxs)
    elseif idxs isa Vector
        idxs
    else
        collect(idxs)
    end
end

function unknowns_to_idxs(::SaveAllUnknowns, model)
    nothing
end

function unknowns_to_idxs(state_names, model, experiment)
    save_idxs = get_save_idxs(experiment)
    idxs = if !(ismissing(state_names) || isnothing(state_names))
        unknowns_to_idxs(state_names, model)
    elseif isnothing(save_idxs)
        return nothing
    else
        eachindex(unknowns(model))
    end

    if isnothing(save_idxs)
        return unknowns_to_idxs(state_names, model)
    end

    d = setdiff(idxs, save_idxs)
    if !isempty(d)
        unknowns_diff = getindex.((unknowns(model),), d)
        msg = join(string.(unknowns_diff), ", ", " and ") *
              " were not saved with save_names."
        throw(ErrorException(msg))
    end

    return idxs
end

function params_to_idxs(param_names, model)
    idxs = indexof.(param_names, (parameters(model),))

    if length(idxs) == 1
        only(idxs)
    else
        idxs
    end
end

function to_dict(x, prob)
    Dict(map(Symbol, get_search_space(prob)) .=> visible_params_part(x, prob))
end

# The model variables that appear in the data
function recorded_model_vars(get_cached_vars, experiment, invprob)
    model = get_model(experiment)
    vars = get_cached_vars(model)
    # Get the variables in data
    vars_in_data = get_saved_model_variables(experiment)

    # Get the indices of vars in data
    idxs = filter(!isnothing, map(x -> findfirst(isequal(x), vars), vars_in_data))

    vars[idxs]
end

function reset_time!(integrator; erase_sol = false)
    # @debug "tstops before reset_time!: $(integrator.opts.tstops)"
    # reinit! sets integrator.t to prob.tspan[1] by default
    # and in the case of CVODE_BDF it will also
    # `handle_callback_modifiers!(integrator)` which will set the
    # time in the CVODE integrator to integrator.t (updated to the new value)
    reinit!(integrator, integrator.u;
        t0 = integrator.sol.prob.tspan[1],
        tstops = integrator.sol.prob.tspan,
        erase_sol)
    # The tstop corresponding to the end of the interval is hit
    # in the `solver_step(integrator, tstop)` and this callback is
    # called after in `handle_callbacks!(integrator)`.
    # Because the loop that integrates until the first tstop
    # has the following condition
    # `integrator.tdir * (integrator.t - first(integrator.opts.tstops)) < -1e6eps()`
    # and we modify the time in reinit!, we also have to add an
    # additional tstop at the same time as the modified time
    # in order to interrupt the loop correctly.
    # This additional tstop is then deleted when the loop is interrupted and
    # and `handle_tstop!(integrator)` is be called.
    # @debug "tstops after reinit!: $(integrator.opts.tstops)"
    # push!(integrator.opts.tstops, integrator.sol.prob.tspan[1])
    # @debug "tstops is now: $(integrator.opts.tstops)"
    nothing
end

function pretty_time(sec)
    t = if sec > 3600
        canonicalize(Minute(round(sec / 60)))
    elseif sec > 30
        canonicalize(Second(round(sec)))
    elseif sec > 0.01
        canonicalize(Millisecond(round(sec * 1e3)))
    else
        canonicalize(Microsecond(round(sec * 1e6)))
    end

    join(lstrip.(split(string(t), ',')), ", ", " and ")
end

function finite_bounds(prob)
    !all(isinf.(vcat(lowerbound(prob), upperbound(prob))))
end

"""
    initial_guess(alg, prob)

Return the initial guess for the elements of the search space for a
given combination of calibration algorithm (`alg`) and inverse problem (`prob`).
"""
function initial_guess(alg, prob)
    ist = get_internal_storage(prob)
    x0 = initial_state(alg, prob)
    names = collect(Symbol.(search_space_names(prob)))

    return names .=> visible_params_part(x0, ist)
end

function find_by_description(desc::AbstractString, model)
    idxs = findall(==(desc), getdescription.(keys(ModelingToolkit.defaults(model))))
    @assert !isempty(idxs) ""
    @assert length(idxs)==1 "More than 1 thing fits the given description"
end
find_by_description(desc, model) = desc
