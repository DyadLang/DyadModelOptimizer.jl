# Get what `unknowns` and `observed` are present in data
function find_unknowns_and_observed(experiment_data, model, unknowns_vars, observed_vars)
    idxs = map(
        x -> find_corresponding_model_var_index(
            x, model, vcat(unknowns_vars, observed_vars)),
        string.(experiment_data.save_names))

    # filtering is done if names in data doesn't match symbols in model. It gives warning at the experiment level
    res = map(
        i -> [index_unknowns_observed(length(unknowns_vars), i),
            mark_unknowns_observed(length(unknowns_vars), i)],
        filter(!isnothing, idxs))
    return first.(res), last.(res)
end

# Get what `unknowns` to estimate from non linear system/problem
function find_estimatable_unknowns(unknowns_vars, unknowns_idxs, eqs)
    estimatable_unknowns_idxs = zeros(Bool, length(unknowns_vars))
    for eq in eqs
        mask = occursin.(unknowns_vars, (eq.lhs,))
        estimatable_unknowns_idxs[mask] .= true
        mask = occursin.(unknowns_vars, (eq.rhs,))
        estimatable_unknowns_idxs[mask] .= true
    end
    estimatable_unknowns_idxs[unknowns_idxs] .= false
    return unknowns_vars[estimatable_unknowns_idxs], findall(estimatable_unknowns_idxs)
end

function setup_nonlinearsystem(eqs, vars, ps)
    @named ns = NonlinearSystem(eqs, vars, ps)
    return structural_simplify(ns; fully_determined = false)
end

function get_parent_symbol(var)
    uvar = Symbolics.unwrap(var)
    # Check if its an element of a symbolic array
    if iscall(uvar) && operation(uvar) == getindex
        return arguments(uvar)[1]
    end
    return var
end

function fixed_point_substitute(observed_vars, observed_eqs, eqs)
    expr_dict = Dict(observed_vars .=> getproperty.(observed_eqs, :rhs))
    rhs_exprs = map(
        x -> ModelingToolkit.fixpoint_sub(x.rhs, expr_dict), eqs)
    return [eqs[i].lhs ~ rhs_exprs[i] for i in eachindex(rhs_exprs)]
end

# Get indices in `unknowns`/`observed` array, data and their corresponding variables
function get_indices_and_vars(vars, index_mapping, unknowns_or_observed, key)
    filtered = filter(x -> x[2] == key,
        collect(zip(index_mapping, unknowns_or_observed, 1:length(index_mapping))))
    idxs = first.(filtered)
    idxs_data = last.(filtered)
    vars_data = vars[idxs]
    return idxs, idxs_data, vars_data
end

# index in `unknowns` or `observed` array
function index_unknowns_observed(l, i)
    i > l ? i - l : i
end

# Mark `unknown` as true and `observed` as false
function mark_unknowns_observed(l, i)
    i > l ? false : true
end
