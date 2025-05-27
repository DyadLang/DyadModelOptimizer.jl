"""
    InternalStorage(experiments, search_space)

The internal storage stores all the information about the model,
the search space in which the solution is defined and also builds
an efficient cache mapping the names of the elements of the search space
to their integer index in the problem state vector.

A model is described by the equations, the unknowns and parameters.
Besides that we can also have the initial values for the unknowns
and the parameter values.
When we optimize a model cost function to fit data, we will describe
the state of the optimization problem as the search space to be explored.
The search space will be a subset of the above model information
in which we look for a solution inside user defined bounds.
"""
struct InternalStorage{B, BI, PG, PS, I, T, S}
    bi::BI
    param_getter::PG
    param_setter::PS
    # search space
    initial_guess::I            # default values for the elements in the searches space
    bounds::B                   # we use the type for dispatch in `boundstype`
    ss_transformation::T        # transformations applied to each of the elements of the search space
    search_space::S   # full search space (what the user provided)
end

function InternalStorage(experiments::AbstractExperimentCollection,
        search_space)
    #=
    we have 2 separate needs for defaults
    - we need to define the initial state of the inverse problem
    - we need to be able to define the underlying problem to solve for the experiments
    for the first case we have the defaults_override,
    for the second case we need at least the types of the initial conditions
    and parameters and the values for any parameters that determine initial conditions or
    other parametrized parts of the problem, such as the timespan.
    =#
    for ex in experiments
        model_transformations = get_model_transformations(ex)
        for mt in model_transformations
            append!(search_space, get_tunable_params(mt, ex))
        end
    end
    @debug "Augmented search_space: $search_space"

    all_tunables, transforms, transformed_tunables = get_tunables(search_space, experiments)

    bi = BatchedInterface(ntuple(length(experiments)) do i
        experiment = experiments[i]
        prob = get_prob(experiment)

        tunables = all_tunables[i]
        @debug "tunables for experiment $i: $tunables"
        (prob, tunables)
    end...
    )
    param_getter = getsym(bi)
    param_setter = setsym_oop(bi)

    initial_vals = param_getter(get_prob.(experiments)...)

    isempty(initial_vals) && !isempty(search_space) &&
        error("Obtained an empty search space specification from a non-empty search space.\n" *
              "Check that the search space elements are not fixed in all $(printname(experiments[1], plural=true))")

    transformed_idxs = variable_index.((bi,), transformed_tunables)
    @debug "search space transformations at: $transformed_idxs"
    # apply search space transformation
    for (t, i) in zip(transforms, transformed_idxs)
        @views initial_vals[i] = t.direct.(initial_vals[i])
    end
    @debug "initial guess for the search space: $initial_vals"

    bounds = []

    # variable_symbols(bi) gives the scalarized version of the search space
    for k in variable_symbols(bi)
        b = getbounds(k)
        # check if bounds are valid
        if !(b isa Distributions.Sampleable)
            lb, ub = b
            if lb > ub
                throw(ArgumentError("The given lowerbound ($lb) for $k is greater than the upperbound ($ub)."))
            end
        end
        # @debug b
        if b isa Tuple{<:AbstractVector, <:AbstractVector}
            # we need to get the bound for the array element
            _k = Symbolics.unwrap(k)
            @assert iscall(_k) "Unable to figure out array index for $_k"
            idx = arguments(_k)[2]
            @debug "array index for $_k is $idx"
            push!(bounds, (b[1][idx], b[2][idx]))
        else
            push!(bounds, b)
        end
    end
    # promote to the eltype of the initial guess
    # @debug eltype(initial_vals)
    bounds = map(Base.Fix2(promote_bound, eltype(initial_vals)), bounds)
    # @debug bounds

    ss_transformations = isempty(transformed_tunables) ? nothing :
                         (; transformed_tunables, transforms)

    InternalStorage(
        bi,
        param_getter,
        param_setter,
        initial_vals,
        bounds,
        ss_transformations,
        search_space)
end

promote_bound(b::Tuple{<:Number, <:Number}, T) = T.(b)
promote_bound(b::Tuple{<:AbstractVector, <:AbstractVector}, T) = (T.(b[1]), T.(b[2]))
promote_bound(b::Priors, T) = b

function get_tunables(search_space, experiments)
    models = get_model.(experiments)
    # while we could do all parameters and rely on Initials, we also get extra initials
    # for dummy derivatives which lead to false positives in checking for duplicate descriptions
    model_symbolic_vars = [get_all_parameters(models); get_all_unknowns(models)]
    descriptions = collect_descriptions(model_symbolic_vars)
    desc_map = Dict(descriptions .=> model_symbolic_vars)

    registered_transformations = Dict([
        :log => (direct = log, inverse = exp),
        :log10 => (direct = log10, inverse = exp10),
        :identity => (direct = identity, inverse = identity)
    ])

    all_tunables = []
    transforms = []
    transformed_tunables = []

    for (i, model) in enumerate(models)
        experiment = experiments[i]
        tunables = []
        defs = default_values(model)

        for (_k, v) in search_space
            if _k isa AbstractString
                if count(==(_k), descriptions) > 1
                    error("Description $_k is not unique!")
                end
                if !haskey(desc_map, _k)
                    error("Description \"$_k\" not found in any of the models.")
                end
                k = desc_map[_k]
                @debug "translated $_k to $k"
            else
                k = _k
            end
            if is_variable(model, k)
                @debug "Transforming $k to Initial."
                k = Initial(k)
            end
            # fixed_params = first.(get_params(experiment))
            # fixed_u0 = first.(get_u0(experiment))
            ex_overrides = first.(get_overrides(experiment))
            @debug "overrides for experiment $i: $ex_overrides"

            lb_ub = get_bounds(v)
            if v isa Tuple && (last(v) isa Symbol)
                t = last(v)
                # @debug t
                if isnothing(findfirst(isequal(k), transformed_tunables))
                    push!(transformed_tunables, k)
                    push!(transforms, registered_transformations[t])
                end

                if !(first(v) isa Distributions.Sampleable)
                    # distributions are assumed to be already transformed
                    lb, ub = lb_ub

                    direct_t = registered_transformations[t][1]
                    can_transform_bound(direct_t, lb, "lower bound", k)
                    can_transform_bound(direct_t, ub, "upper bound", k)

                    # k = Symbolics.wrap(setmetadata(
                    #     Symbolics.unwrap(k), VariableDefaultValue, direct_t.(defs[k])))
                    # k = Symbolics.wrap(setmetadata(Symbolics.unwrap(k),
                    #     ModelingToolkit.VariableBounds, (direct_t.(lb), direct_t.(ub))))
                    lb_ub = direct_t.(lb), direct_t.(ub)
                end
            end

            if has_initial(k) && is_variable(model, peel_init(k))
                if !ispresent(peel_init(k), ex_overrides)
                    prob = get_prob(experiment)
                    if v isa Tuple && first(v) isa VectorOrScalar &&
                       (length(v) == 3 && !(last(v) isa Symbol)) ||
                       (length(v) == 4 && last(v) isa Symbol)
                        # @debug "$k -> $(first(v))"
                        setsym(prob, k)(prob, first(v))
                    end
                    k = Symbolics.wrap(setmetadata(
                        Symbolics.unwrap(k), ModelingToolkit.VariableBounds, (lb_ub)))
                    push!(tunables, k)
                end
            elseif is_parameter(model, k)
                # normal parameter
                if ispresent(k, ex_overrides)
                    if !istunable(k)
                        @warn "$k is marked as not tunable, but it was included in the search space"
                    end
                else
                    if v isa Tuple && first(v) isa VectorOrScalar &&
                       (length(v) == 3 && !(last(v) isa Symbol)) ||
                       (length(v) == 4 && last(v) isa Symbol)
                        # change value in the cache
                        @debug "got explicit default for $k: $(first(v))"
                        prob = get_prob(experiment)
                        setsym(prob, k)(prob, first(v))
                    end

                    k = Symbolics.wrap(setmetadata(Symbolics.unwrap(k),
                        ModelingToolkit.VariableBounds, (lb_ub)))
                    push!(tunables, k)
                end
            end
            # error checking
            for key in tunables
                idxs = findall(isequal(key), tunables)
                @assert length(idxs)==1 "Repeated elements in the search space are not allowed. Found $(length(idxs)) entries for $key."
            end
            !haskey(defs, k) &&
                error("Unable to get default for `$k`. Please provide a default " *
                      "in the model or an initial guess in the search space. For providing defaults in the search space, " *
                      "refer to the docstring of InverseProblem for more details.")
        end

        # @debug "pushing $tunables"
        push!(all_tunables, map(identity, tunables))
    end

    return all_tunables, transforms, transformed_tunables
end

function collect_descriptions(all_syms)
    [symbolic_type(sym) isa NotSymbolic ? sym :
     has_initial(sym) ? getdescription(peel_init(sym)) : getdescription(sym)
     for sym in all_syms]
end

function collect_all(f, experiments)
    things = ()
    for experiment in experiments
        x = f(experiment)
        things = combine_collection(things, x)
    end

    return things
end

combine_collection(v, x) = (v..., x)
combine_collection(v, x::Tuple) = (v..., x...)

function get_all_unknowns(models)
    all_u0s = reduce(vcat, [unknowns(model) for model in models])
    check_for_contradictory_descriptions(all_u0s)
    unique(all_u0s)
end

function get_all_parameters(models; full = false, initial_parameters = false)
    if full
        all_ps = reduce(vcat, [full_parameters(model) for model in models])
    else
        all_ps = reduce(vcat, [parameters(model; initial_parameters) for model in models])
    end
    check_for_contradictory_descriptions(all_ps)
    unique(all_ps)
end

function get_all_observed(models)
    all_obs = reduce(vcat, [observed_names(model) for model in models])
    check_for_contradictory_descriptions(all_obs)
    unique(all_obs)
end

function check_for_contradictory_descriptions(syms::Vector)
    for sym in syms
        # can't check for non symbolic things
        symbolic_type(sym) isa NotSymbolic && continue
        my_description = getdescription(sym)
        idxs = findall(Base.Fix1(isequal, sym), syms)
        if !isnothing(idxs)
            n_duplicates = length(idxs)
            n_duplicates > 1 && @debug "$n_duplicates duplicates for $sym"
            syms_equal_to_me = syms[idxs]
            problem_idx = findfirst(!=(my_description), getdescription.(syms_equal_to_me))
            # @debug problem_idx
            if !isnothing(problem_idx)
                desc = getdescription(syms_equal_to_me[problem_idx])
                error("Found 2 or more contradictory descriptions for $sym: " *
                      "$my_description and $desc.\nPlease use different symbolic variables or different descriptions.")
            end
        end
    end
end

# can't check for strings
check_for_contradictory_descriptions(syms::Vector{<:AbstractString}) = nothing

function push_transforms!(ss_direct_transformation, ss_inverse_transformation, k, v)
    registered_transformations = Dict([
        :log => (log, exp),
        :log10 => (log10, exp10),
        :identity => (identity, identity)
    ])

    if v isa Tuple && last(v) isa Symbol
        t = last(v)
        if haskey(registered_transformations, t)
            direct_t = registered_transformations[t][1]
            inverse_t = registered_transformations[t][2]
            if v[1] isa Number
                lb, ub = get_bounds(v)
                can_transform_bound(direct_t, lb, "lower bound", k)
                can_transform_bound(direct_t, ub, "upper bound", k)
                if length(v) == 4
                    @assert lb≤v[1]≤ub "Initial guess $(v[1]) is outside of the bounds ($lb, $ub)"
                    can_transform_bound(direct_t, lb, "initial guess", k)
                end
            end
            push!(ss_direct_transformation, direct_t)
            push!(ss_inverse_transformation, inverse_t)
        else
            supported = join(keys(registered_transformations), ", ", " and ")
            error("Transformation $t not supported. Supported transformations are $supported.")
        end
    else
        push!(ss_direct_transformation, identity)
        push!(ss_inverse_transformation, identity)
    end
end

function can_transform_bound(transform, bound, name, key)
    try
        transform(bound)
    catch e
        if e isa DomainError
            msg = "Encountered an invalid value for the $transform transformation of the $name: $key => $bound"
            throw(DomainError(bound, msg))
        else
            rethrow()
        end
    end
end

can_transform_bound(::typeof(identity), args...) = nothing

function to_opt!(opt, syms)
    for s in syms
        if istunable(s)
            push!(opt, s)
        end
    end
end

function get_what_to_optimize(model::AbstractTimeDependentSystem)
    sts = unknowns(model)
    params = parameters(model)

    opt = Vector{Num}()
    to_opt!(opt, sts)
    to_opt!(opt, params)

    return opt
end

"""
    determine_search_space(model)

Read the search space from the MTK model definition
"""
function determine_search_space(model::AbstractTimeDependentSystem)
    bounds = get_bounds(model)
    opt = get_what_to_optimize(model)
    if !isempty(bounds) && !isempty(opt)
        filter!(x -> ispresent(first(x), opt), bounds)
        return bounds
    else
        @error "The model doesn't have the required metadata."
    end
end

get_search_space(ist::InternalStorage) = ist.search_space

# get the symbolic optimization state vector
search_space_names(ist::InternalStorage) = variable_symbols(ist.bi)

"""
    search_space_defaults(ist::InternalStorage)

Get the default values for the elements of the search space.
"""
function search_space_defaults(ist::InternalStorage)
    x = ist.initial_guess
    bounds = get_bounds(ist)
    isempty(bounds) && return Float64[]
    # we need to ensure that picking from a distribution
    # picks different values everytime instead
    # of using the same cached values every time
    [pick_x0(bound, xi) for (bound, xi) in zip(bounds, x)]
end

pick_x0(::Tuple, x) = x

function pick_x0(bounds::Distributions.Sampleable, x)
    # For distributions we assume that the user provided the log transformed distribution
    rand(bounds)
end

function visible_params_part(x, ist::InternalStorage)
    end_idx = length(ist.initial_guess)
    @views values(x)[1:end_idx]
end

function visible_params_part(x, prob::AbstractDesignAnalysis)
    visible_params_part(x, get_internal_storage(prob))
end

function internal_params_part(x, ist::InternalStorage)
    start_idx = length(search_space_names(ist)) + 1
    # @debug "start_idx for internal params part: $start_idx"
    # the index cache is a permutation of 1:number_of_search_space_elements
    @views values(x)[start_idx:end]
end

function internal_params_part(mt::AbstractModelTransformation, x, ist)
    idxs = ist.hidden_params_index_cache[get_uuid(mt)]
    # @debug "idxs: $idxs"

    view(internal_params_part(x, ist), idxs)
end

get_penalty(ist::InternalStorage) = ist.penalty
