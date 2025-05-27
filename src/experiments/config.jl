struct ExperimentConfig{
    FWD,
    S,
    D,
    M,
    IM,
    O,
    U,
    P,
    C,
    CT,
    MT,
    DH,
    PU0,
    GU0,
    SU0,
    SAU,
    SA,
    SI,
    SMV,
    GSV,
    GO,
    CB,
    LF,
    PP,
    NP,
    L,
    R,
    A,
    PR,
    KW
}
    u0_dependency::FWD
    saved_values::S
    data::D
    model::M
    initial_model::IM
    overrides::O
    u0map::U
    psmap::P
    constraints::C
    constraints_ts::CT
    model_transformations::MT
    dependency_handling::DH
    parametric_u0::PU0
    getu0_defs::GU0
    set_parametric_u0::SU0
    set_all_u0s::SAU
    saveat::SA
    save_idxs::SI  # Indices of the solution to be saved, sorted to match data. `Nothing` if we need the full solution.
    saved_model_variables::SMV
    get_saved_vals::GSV
    get_obs::GO
    callbacks::CB
    loss_func::LF
    postprocess::PP
    noise_priors::NP
    likelihood::L
    reduction::R
    name::String
    uuid::UUID
    alg::A
    prob::PR
    kwargs::KW
end

function ExperimentConfig(
        data, model, initial_model, overrides, u0map, psmap, constraints, constraints_ts,
        tspan, parametric_u0, getu0_defs, set_parametric_u0,
        set_all_u0s, dependency, dependency_handling,
        save_names, saveat, callbacks, loss_func, postprocess,
        noise_priors, likelihood, reduction, name, alg,
        prob, model_transformations,
        do_not_replace_sol, kwargs)
    u0_dependency = determine_dependency(dependency)

    saved_values, save_idxs,
    saved_model_variables = saving_behavior(model, save_names, tspan,
        name, do_not_replace_sol)

    # cached getsym for replace_sol
    if (saved_model_variables isa AbstractVector && length(saved_model_variables) == 1)
        model_var = only(saved_model_variables)
        get_saved_vals = getsym(prob, model_var)
        if is_observed(prob, model_var)
            get_obs = SymbolicIndexingInterface.observed(prob, model_var)
        else
            @debug "observed not available for $model_var"
            get_obs = nothing
        end
    elseif !isnothing(saved_model_variables)
        get_saved_vals = getsym(prob, saved_model_variables)
        can_observe = if saved_model_variables isa AbstractVector
            any(Base.Fix1(is_observed, prob), saved_model_variables)
        else
            # scalar case
            is_observed(prob, saved_model_variables)
        end
        if can_observe
            get_obs = SymbolicIndexingInterface.observed(prob, saved_model_variables)
        else
            @debug "observed not available for $saved_model_variables"
            get_obs = nothing
        end
    else
        @debug "getsym and observed are not available"
        get_saved_vals = nothing
        get_obs = nothing
    end

    ExperimentConfig(u0_dependency,
        saved_values,
        data,
        model,
        initial_model,
        overrides,
        u0map,
        psmap,
        constraints,
        constraints_ts,
        model_transformations,
        dependency_handling,
        parametric_u0,
        getu0_defs,
        set_parametric_u0,
        set_all_u0s,
        saveat,
        save_idxs,
        saved_model_variables,
        get_saved_vals,
        get_obs,
        callbacks,
        loss_func,
        postprocess,
        noise_priors,
        likelihood,
        reduction,
        name,
        uuid4(),
        alg,
        prob,
        kwargs
    )
end

function construct_config(data, initial_model, overrides, constraints, constraints_ts,
        tspan, model_transformations,
        callback, dependency, save_names, saveat, loss_func, postprocess,
        noise_priors, likelihood, reduction, name, alg,
        do_not_replace_sol, prob_kwargs, kwargs, indepvar = :timestamp,
        translate_model_transformations = true)
    # We get the timepoints at which we save from the available data.
    if data isa AbstractExperimentData
        experiment_data = data
    else
        experiment_data = timeseries_data(data, save_names, indepvar, saveat, tspan)
    end

    callbacks = CallbackSet(callback)
    if translate_model_transformations
        for mt in model_transformations
            @debug "Transforming $(nameof(typeof(mt)))"
            cb = callback_form_transformation(mt, experiment_data, initial_model)
            if !isnothing(cb)
                callbacks = CallbackSet(callbacks, cb)
            else
                continue
            end
        end
    end

    @debug "overrides: $overrides"

    new_guess = []
    for o in overrides
        # Add guesses for all overrides that don't already have that
        # This ensures that parameter overrides that are passed will
        # opt in to parameter initialization
        k = o[1]
        val = haskey(defaults(initial_model), o[1]) ? o[1] : o[2]
        if !haskey(guesses(initial_model), k)
            push!(new_guess, k => val)
        end
    end
    if !isempty(new_guess)
        @debug "Adding extra guesses: $new_guess"
        g = merge(guesses(initial_model), Dict(new_guess))
        @set! initial_model.guesses = g
    end

    u0map, psmap = split_overrides(overrides, initial_model)

    defs = defaults(initial_model)
    parametric_u0 = filter(
        k -> haskey(defs, k) && !(symbolic_type(defs[k]) isa NotSymbolic),
        unknowns(initial_model))
    @debug "Parametric initial conditions: $parametric_u0"

    if !isempty(model_transformations)
        new_model = augment_model(model_transformations[1], initial_model)
    else
        new_model = initial_model
    end
    for mt in model_transformations[2:end]
        new_model = augment_model(mt, new_model)
    end

    prob = setup_prob(
        new_model, u0map, psmap, tspan, (; callback = callbacks, prob_kwargs...))

    for mt in model_transformations
        prob = augment_prob(mt, prob)
    end

    if isnothing(prob.f.initialization_data)
        @debug "No initialization problem created."
    end

    alg = if ismissing(alg)
        default_alg = get_solve_alg(prob)
        prepare_alg(default_alg, prob.u0, prob.p, prob)
    else
        prepare_alg(alg, prob.u0, prob.p, prob)
    end

    defs = default_values(prob)

    if !isnothing(dependency)
        # Only non-parametric differential variables can be set via dependencies
        non_parametric_diff_vars = filter(
            k -> haskey(defs, k) && !haskey(Dict(overrides), k) &&
                     (symbolic_type(defs[k]) isa NotSymbolic),
            get_diff_variables(new_model))

        @debug "Non-parametric differential variables: $non_parametric_diff_vars"
        u0_defs = getindex.((default_values(prob),), non_parametric_diff_vars)
        @debug "Defaults for dependent u0s: $u0_defs"
        dependent_prob = get_prob(dependency)
        source_vars = variable_symbols(dependent_prob) ∩ non_parametric_diff_vars
        @debug "Source vars for u0 dependencies: $source_vars"
        getu0_vals = getsym(prob, source_vars)
        set_dependent_u0 = setsym_oop(prob, Initial.(source_vars))

        dependency_handling = (; source_vars, getu0_vals, set_dependent_u0)
    else
        dependency_handling = nothing
    end

    u0_defs = getindex.((default_values(prob),), parametric_u0)
    @debug "Defaults for parametric u0s: $u0_defs"
    getu0_defs = getsym(prob, u0_defs)
    set_parametric_u0 = setsym_oop(prob, Initial.(parametric_u0))
    set_all_u0s = setsym_oop(prob, Initial.(get_diff_variables(new_model)))

    return ExperimentConfig(
        experiment_data, new_model, initial_model, overrides, u0map, psmap, constraints, constraints_ts,
        tspan, parametric_u0, getu0_defs, set_parametric_u0, set_all_u0s,
        dependency, dependency_handling,
        save_names, saveat, callbacks, loss_func, postprocess,
        noise_priors, likelihood, reduction, name, alg,
        prob, model_transformations,
        do_not_replace_sol, kwargs)
end

function determine_what_to_save(model, save_names::Vector{<:Pair}, model_unknowns)
    saved_model_variables = first.(save_names)
    save_idxs = collect(indexof.(saved_model_variables, (model_unknowns,)))
    return saved_model_variables, save_idxs
end

function determine_what_to_save(model, save_names::Vector, model_unknowns)
    saved_model_variables = find_corresponding_model_var.(save_names,
        (model,),
        (model_unknowns,))
    save_idxs = collect(indexof.(saved_model_variables, (model_unknowns,)))

    return saved_model_variables, save_idxs
end

function determine_what_to_save(model, save_names, model_unknowns)
    saved_model_variables = find_corresponding_model_var(save_names,
        model,
        model_unknowns)
    save_idxs = indexof(saved_model_variables, model_unknowns)

    return saved_model_variables, save_idxs
end

function saving_behavior(model,
        save_names,
        tspan,
        name,
        do_not_replace_sol::Val{T} = Val(false)) where {T}
    # haskey(kwargs, :save_idxs) &&
    #     @warn "specifying save_idxs will overwrite the automatically computed one and might lead to errors"
    model_unknowns = unknowns(model)
    @debug "save_names: $save_names"

    saved_model_variables, save_idxs = determine_what_to_save(model,
        save_names,
        model_unknowns)

    if !isnothing(saved_model_variables)
        msg = saved_model_variables isa AbstractVector ? join(saved_model_variables, ", ") :
              saved_model_variables
        @debug "saved model variables in $name: $msg"
    else
        @debug "saved_model_variables is nothing. Is this ok?"
    end

    if tspan isa Number
        tspan = (zero(tspan), tspan)
    end

    saved_values = nothing
    @debug "save_idxs before normalization: $save_idxs"

    save_idxs, saved_values = normalize_save_idxs(save_idxs,
        saved_values,
        model_unknowns, # TODO: how does this work with DAEs?
        save_names,
        do_not_replace_sol)

    if save_idxs isa Vector && length(save_idxs) == 1
        save_idxs = only(save_idxs)
    end

    @debug "save_idxs: $save_idxs"
    @debug "saved_values: $saved_values"

    saved_values, save_idxs, saved_model_variables
end

function normalize_save_idxs(save_idxs::Vector{Int},
        saved_values,
        model_unknowns,
        save_names,
        ::Val{S} = Val(false)) where {S}
    if save_idxs == collect(1:length(model_unknowns)) && save_idxs ≠ [1]
        # if all unknowns are saved we can just use the default
        @debug "saving all the unknowns ($model_unknowns), no need to use save_idx"
        save_idxs = nothing
    end
    save_idxs, saved_values
end

function normalize_save_idxs(save_idxs::Int,
        saved_values,
        model_unknowns,
        save_names,
        ::Val{S} = Val(false)) where {S}
    save_idxs, saved_values
end

function normalize_save_idxs(save_idxs::Vector,
        saved_values,
        model_unknowns,
        save_names,
        ::Val{S} = Val(false)) where {S}
    #check for save_names that were not found
    not_found = findall(isnothing, save_idxs)
    if !isempty(not_found)
        @debug "$(join(save_names[not_found], ", ")) were not found in the unknowns of the model"
        # save all unknowns and compute the observed after with replace_sol
        save_idxs = nothing
        saved_values = S ? saved_values : Val(true)
    end

    save_idxs, saved_values
end

function normalize_save_idxs(save_idxs::Nothing,
        saved_values,
        model_unknowns,
        save_names,
        ::Val{T} = Val(false)) where {T}
    if T
        @debug "unknowns were not identified, assuming all are observed."
        saved_values = Val(true)
    end

    save_idxs, saved_values
end

function setup_u0_idx(model, u0)
    u0_idx = Int[]
    model_unknowns = unknowns(model)

    for tu0 in u0
        i = indexof(tu0[1], model_unknowns)
        isnothing(i) && error("$(tu0[1]) was not found in the model unknowns.")

        push!(u0_idx, i)
    end

    return u0_idx
end

function setup_params_idx(model, params)
    params_idx = Int[]

    for tp in params
        # @debug tp
        i = parameter_index(model, tp[1])
        isnothing(i) && error("$(tp[1]) was not found in the model parameters.")

        push!(params_idx, i)
    end

    return params_idx
end

function setup_params_alias_idx(model, params)
    idx_map = Vector{Pair{Int, Int}}(undef, 0)
    model_parameters = parameters(model)

    for (k, v) in params
        i = indexof(k, model_parameters)
        j = indexof(v, model_parameters)
        isnothing(i) && error("$k not found in the model parameters.")
        isnothing(j) && error("$v not found in the model parameters.")
        push!(idx_map, i => j)
    end

    return idx_map
end

function setup_u0_alias_idx(model, u0)
    idx_map = Vector{Pair{Int, Int}}(undef, 0)
    model_unknowns = unknowns(model)
    model_parameters = parameters(model)

    for (k, v) in u0
        symbolic_type(unwrap(v)) === NotSymbolic() && continue
        i = indexof(k, model_unknowns)
        j = indexof(v, model_parameters)
        isnothing(i) && error("$k not found in the model unknowns.")
        isnothing(j) && error("$v not found in the model parameters. " *
              "Note that initial conditions must be aliased to parameters, not other unknowns.")
        push!(idx_map, i => j)
    end

    return idx_map
end
