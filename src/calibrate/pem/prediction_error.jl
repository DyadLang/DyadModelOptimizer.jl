include("utils.jl")
include("discrete_fixed_gain_pem.jl")

function prediction_error_callback(method::AbstractPredictionErrorMethod, ::NoData, model)
    @error "Cannot use Predictive Error Method as there is no data"
    return nothing
end

# Wrapper function to get all the metadata necessary for PEM
function pem_metadata(experiment_data, model)
    # Get the unknowns from the model
    unknowns_vars = unknowns(model)

    # Get the observed from the model
    observed_eqs = observed(model)
    observed_vars = getproperty.(observed_eqs, :lhs)

    # Get the unknown/observed mapping in data
    index_mapping, unknowns_or_observed = find_unknowns_and_observed(
        experiment_data, model, unknowns_vars, observed_vars)

    # Get the unknowns present in data and indices in the data matrix to which it corresponds
    unknowns_idxs, unknowns_idxs_data, unknowns_vars_data = get_indices_and_vars(
        unknowns_vars,
        index_mapping, unknowns_or_observed, true)

    # Get the observed present in data and indices in the data matrix to which it corresponds
    observed_idxs, observed_idxs_data, observed_vars_data = get_indices_and_vars(
        observed_vars,
        index_mapping, unknowns_or_observed, false)

    # parameters of the model
    ps = parameters(model)

    # Independent variables
    iv = ModelingToolkit.get_iv(model)

    return (; iv, ps, observed_eqs,
        observed_idxs, observed_vars, observed_vars_data, observed_idxs_data,
        unknowns_idxs, unknowns_vars, unknowns_vars_data, unknowns_idxs_data)
end
