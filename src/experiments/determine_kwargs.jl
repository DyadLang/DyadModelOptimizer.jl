function determine_tspan(data, indepvar)
    # This is the first kwarg function that gets called
    # We should first check here if the time is ok
    t = Tables.getcolumn(data, indepvar)

    if issorted(t)
        (zero(Base.promote_eltype(t)), last(t))
    elseif issorted(t, rev = true)
        tspan = (first(t), last(t))
        @info "reverse time order detected, assuming tspan is $tspan"
        tspan
    else
        error("The independent variable column in the data is not sorted. Are you sure $indepvar is the correct column name?")
    end
end

determine_tspan(::Nothing, indepvar) = error("No timespan provided and no data available.")

function determine_tspan(data::Vector{DataFrame}, indepvar)
    if length(data) == 0
        error("No timespan provided and no data available.")
        ()
    else
        # TODO: use indepvar
        ts = vcat(first.(eachcol.(data))...)
        (zero(Base.promote_eltype(ts)), max(ts...))
    end
end

struct SaveAllUnknowns{T}
    unknowns::T
end

Base.length(s::SaveAllUnknowns) = length(s.unknowns)
Base.iterate(s::SaveAllUnknowns, i...) = iterate(s.unknowns, i...)

# no data, default to saving all unknowns
determine_save_names(::Nothing, model, skip_t_col = true) = SaveAllUnknowns(unknowns(model))
# can't determine the names
# determine_save_names(::AbstractArray, model) = NoNamesProvided(length(unknowns(model)))

function determine_save_names(data::Vector{<:DataFrame}, model, skip_t_col = true)
    all_names = determine_save_names.(data, (model,), skip_t_col)
    ref = first(all_names)
    for n in all_names[2:end]
        @assert reduce(isequal, [ref, n]) "Not all replicates have the same column names!"
    end

    return ref
end

function determine_save_names(data, model, skip_t_col = true)
    data_as_dict_warn(data)
    all_cols = Tables.columnnames(data)
    if skip_t_col
        t_col = first(all_cols)
        colnames = setdiff(all_cols, [t_col])
        length(all_cols) - 1 ≠ length(colnames) && error("Column names are not unique!")
    else
        colnames = collect(all_cols)
    end

    # select only the column names that are present in the model
    model_unknowns = unknowns(model)
    cols = filter(col -> col_in_vars(col, model, model_unknowns), string.(colnames))
    @debug "Selected column names: $(join(cols, ", "))"

    if isempty(cols)
        @warn "No column names matched model names."
        nothing
    else
        map(Symbol, cols)
    end
end

function determine_saveat(data::T, indepvar = :timestamp) where {T}
    if Tables.istable(T)
        Tables.getcolumn(data, indepvar)
    else
        @debug "try data_shape fallback"
        determine_saveat(data_shape(T), data, indepvar)
    end
end

function determine_saveat(::MatrixLike, data, indepvar)
    @debug "Determining saveat for MatrixLike data"
    # user_saveat = get(kwargs, :saveat, nothing)
    data_saveat = isnothing(data) ? nothing : Tables.getcolumn(data, indepvar)
    # !isnothing(user_saveat) && @debug "user given saveat: $user_saveat"
    isnothing(data_saveat) ? () : data_saveat
    # isnothing(user_saveat) && isnothing(data_saveat) && return ()
    # isnothing(user_saveat) && return data_saveat
    # isnothing(data_saveat) && return user_saveat
    # !isequal(user_saveat, data_saveat) && @warn "The saveat keyword argument is different from the first column of the data."
    # sort!(unique!(vcat(user_saveat, data_saveat)))
    # user_saveat
end

function determine_saveat(::VectorLike{<:DataFrame}, data, indepvar)
    n = length(data)
    @debug "Determining saveat for replicate data with $n replicates"
    saveat = Vector{Vector{Float64}}(undef, n)
    for (i, df) in enumerate(data)
        s = determine_saveat(df, indepvar)
        @debug "Replicte $i: $s"
        saveat[i] = s
    end
    sort!(unique!(vcat(saveat...)))
end

# steady state data doesn't have saveat
determine_saveat(::VectorLike, data) = ()
determine_saveat(::Nothing, data) = ()

determine_saveat(::NoData) = ()

determine_saveat(data::TimeSeriesData) = data.time
determine_saveat(::SteadyStateData) = ()

function determine_constraints_ts(constraints, saveat, tspan)
    if isempty(constraints)
        saveat
    else
        if isempty(saveat)
            error("Please provide saveat or constraints_ts.")
        else
            if saveat isa Number
                range(tspan..., step = saveat)
            else
                saveat
            end
        end
    end
end

function determine_saveat(data::ReplicateData)
    @debug "saveat from ReplicateData"
    saveat = determine_saveat.(data)
    sort!(unique!(vcat(saveat...)))
end

determine_err(::AbstractArray{<:Tuple}) = ARMLoss
determine_err(::Nothing) = (x, sol, data) -> error("Experiment has no data!")

function determine_err(d::Vector{DataFrame})
    errs = determine_err.(d)
    if all(errs .== ARMLoss)
        return ARMLoss
    elseif all(errs .== squaredl2loss)
        return squaredl2loss
    else
        error("Replicate measurements don't have same data type!")
    end
end

function determine_err(data)
    data_elt = data_eltype(data)
    @debug "data eltype: $data_elt"
    if data_elt <: Union{Number, Missing}
        squaredl2loss
    elseif data_elt <: Tuple
        ARMLoss
    elseif isnothing(data_elt)
        error("There is no data and no error/loss function was provided. Please provide one.")
    else
        error("Data format not supported!")
    end
end

function data_eltype(data)
    if isempty(data)
        @warn "Empty data"
        return nothing
    end
    @assert Tables.istable(typeof(data)) "The given data is not compatible with Tables.jl, please convert it to something compatible."
    sch = Tables.schema(data)
    isnothing(sch) && error("Tables without a schema are not currently supported.")
    col_types = sch.types
    @assert length(col_types)>1 "At least 2 columns are needed, one for the time and one for the state."

    reduce(promote_type, Base.tail(col_types))
end

function determine_loss_func(running_cost, args...)
    @debug "symbolic_type for $running_cost: $(symbolic_type(running_cost))"
    determine_loss_func(symbolic_type(running_cost), running_cost, args...)
end

function determine_loss_func(::Nothing, args...)
    error("keyword argument `running_cost` not assigned.")
end

function determine_loss_func(::NotSymbolic, running_cost, reduction, args...)
    function loss_func(x, sol, data)
        reduction(running_cost(x, sol, data))
    end
end

function determine_loss_func(
        ::Any, running_cost, reduction, model, overrides, tspan, prob_kwargs)
    u0map, psmap = split_overrides(overrides, model)
    prob = setup_prob(model, u0map, psmap, tspan, prob_kwargs)
    obs = SymbolicIndexingInterface.observed(prob, running_cost)
    let obs = obs
        function loss_func(x, sol, data)
            reduction(obs.(sol.u, (parameter_values(sol),), sol.t))
        end
    end
end
