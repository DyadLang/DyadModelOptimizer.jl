function cost_contribution(alg::AbstractCollocationAlgorithm,
        experiment::AbstractExperiment,
        prob,
        x,
        alg_cache = nothing)
    # Get collocation data
    alg_cache = reuse_cache(alg_cache, alg, prob)

    _, _, du, du_est = collocate_data(alg, experiment, prob, x, alg_cache)

    # Get the error function
    err = error_function(alg, experiment, du, du_est, prob, x)
    return err
end

initialize_cache(alg::AbstractCollocationAlgorithm, prob) = create_alg_cache(alg, prob)

reuse_cache(alg_cache, alg, prob) = alg_cache
reuse_cache(::Nothing, alg, prob) = create_alg_cache(alg, prob)

function algorithm_cache(alg::AbstractCollocationAlgorithm, experiment, invprob)
    # Get data
    data = get_data(experiment)

    # Get model
    model = get_model(experiment)

    # Compute states from observed
    data_or_ns = collocation_cache(model, data, experiment, invprob)

    if data_or_ns isa Tuple
        nothing, nothing, nothing, data_or_ns
    else
        # Check if collocation is possible
        data_idxs = check_possible_collocation(model,
            data_or_ns,
            model_unknowns(invprob, experiment))
        # @debug data_idxs
        # Get the time points
        saveat = get_saveat(experiment)
        # Collocate data
        collocate_data(alg, data_or_ns, saveat)..., data_idxs, data_or_ns
    end
end

###################### Collocate data functions ######################

function recompute_collocation(data_or_ns::Tuple,
        data_idxs,
        u_est,
        du_est,
        alg,
        experiment,
        invprob,
        x)
    u0s, ns = data_or_ns
    model = get_model(experiment)
    data = get_data(experiment)

    new_data = compute_states_from_observed(model, data, experiment, invprob, u0s, ns, x)
    data_idxs = check_possible_collocation(model,
        new_data,
        model_unknowns(invprob, experiment))

    saveat = get_saveat(experiment)

    u_est, du_est = collocate_data(alg, new_data, saveat)

    return u_est, du_est, new_data, data_idxs
end

function recompute_collocation(data_or_ns,
        data_idxs,
        u_est,
        du_est,
        alg,
        experiment,
        invprob,
        x)
    u_est, du_est, data_or_ns, data_idxs
end

function collocate_data(alg::AbstractCollocationAlgorithm,
        experiment::AbstractExperiment,
        invprob,
        x,
        cache)
    # Collocate data
    u_est, du_est, data_idxs, new_data = cache[get_uuid(experiment)]

    u_est, du_est, new_data, data_idxs = recompute_collocation(new_data, data_idxs, u_est,
        du_est,
        alg,
        experiment,
        invprob,
        x)

    # Get the time points
    saveat = get_saveat(experiment)

    # Get the problem
    prob = setup_problem(experiment, invprob, x)

    # Compute the derivatives from the model
    du = derivatives_from_model(prob, invprob, experiment, x, u_est, data_idxs, saveat)
    return new_data, u_est, du, du_est
end

function collocate_data(alg::KernelCollocation, data::AbstractMatrix, tpoints)
    𝟙 = oneunit(eltype(data))
    𝟘 = zero(eltype(data))

    ξ₁ = [𝟙; 𝟘]
    ξ₂ = [𝟘; 𝟙; 𝟘]
    n = length(tpoints)
    bandwidth = alg.bandwidth === nothing ?
                (n^(-1 / 5)) * (n^(-3 / 35)) * ((log(n))^(-1 / 16)) : alg.bandwidth

    WY = similar(data, n, size(data, 1))    # W_t * Y in (Liang and Wu, 2009)
    WT₁ = similar(data, n, 2)               # W_t * T_1,t in (ibid.)
    WT₂ = similar(data, n, 3)               # W_t * T_2,t in (ibid.)
    T₂WT₂ = similar(data, 3, 3)
    T₁WT₁ = similar(data, 2, 2)
    α_hat = map(tpoints) do _t
        T₁ = construct_t1(_t, tpoints)
        T₂ = construct_t2(_t, tpoints)
        W = construct_w(_t, tpoints, bandwidth, alg.kernel)
        mul!(WY, W, data')
        mul!(WT₁, W, T₁)
        mul!(WT₂, W, T₂)
        mul!(T₂WT₂, T₂', WT₂)
        mul!(T₁WT₁, T₁', WT₁)
        (det(T₂WT₂) ≈ 0.0 || det(T₁WT₁) ≈ 0.0) &&
            error("Collocation failed with bandwidth $bandwidth. Please choose a higher bandwidth")
        ξ₂' * ((T₂' * WT₂) \ T₂') * WY, (ξ₁' * ((T₁' * WT₁) \ T₁')) * WY
    end
    du_est = reduce(hcat, transpose.(first.(α_hat)))
    u_est = reduce(hcat, transpose.(last.(α_hat)))
    return u_est, du_est
end

function collocate_data(alg::SplineCollocation, data::AbstractMatrix, tpoints)
    u_est = zero(data)
    du_est = zero(data)
    @inbounds for d1 in axes(data, 1)
        # Pass in `data.data[d1, :]` instead of `data[d1, :]` as it will fail with BSplineInterpolation
        # TODO: fix abstract array interface for time series data to remove this
        interpolation = alg.interp(data.data[d1, :], tpoints, alg.interp_args...)
        @views u_est[d1, :] .= interpolation.(tpoints)
        @views du_est[d1, :] .= derivative.((interpolation,), tpoints)
    end
    return u_est, du_est
end

function collocate_data(alg::NoiseRobustCollocation, data::AbstractMatrix, tpoints)
    du_est = zero(data)
    @inbounds for i in axes(data, 1)
        @views du_est[i, :] = tvdiff(data[i, :],
            alg.diff_iters,
            alg.α;
            alg.tvdiff_kwargs...)
    end
    u_est = zero(data)
    @inbounds for i in axes(data, 1)
        @views u_est[i, :] = cumul_integrate(tpoints, du_est[i, :])
        @views u_est[i, :] .+= data[i, 1]
    end
    return u_est, du_est
end

function collocate_data(alg::AbstractCollocationAlgorithm, data::AbstractVector, tpoints)
    u_est, du_est = collocate_data(alg, reshape(data, 1, :), tpoints)
    return u_est, du_est
end

###################### Get Collocation data ######################
"""
A container for storing the original and the collocated data - both states and its derivatives, which can be used for visualization and validation purposes.
"""
struct CollocationData{U, UE, DU, DUE, T, N}
    u::U
    u_est::UE
    du::DU
    du_est::DUE
    time::T
    names::N
end

"""
    CollocationData(alg, experiment, invprob, x)

Get the collocated data by passing in `alg` - collocation algorithm, experiment, inverse problem and the parameter which can be either be
a vector of symbol value pairs, named tuple or vector of numbers. This will return a `CollocationData` object which contains estimated states and derivatives along
with the data and derivatives computed from the model.
"""
function CollocationData(alg::AbstractCollocationAlgorithm,
        experiment::AbstractExperiment,
        invprob,
        x::Union{AbstractVector{<:Pair}, NamedTuple})
    ist = get_internal_storage(invprob)
    ss = search_space_names(ist)
    ordered_x = copy(match_search_space_order(x, ss))
    x′ = transform_params!(:direct, ordered_x, invprob)
    CollocationData(alg, experiment, invprob, x′)
end

function CollocationData(alg::AbstractCollocationAlgorithm,
        experiment::AbstractExperiment,
        invprob,
        x)
    cache = Dict(get_uuid(experiment) => algorithm_cache(alg, experiment, invprob))
    data, u_est, du, du_est = collocate_data(alg, experiment, invprob, x, cache)
    CollocationData(
        typeof(data.data) <: AbstractVector ? reshape(data.data, 1, :) :
        data.data,
        u_est,
        du,
        du_est,
        data.time,
        data.save_names)
end

"""
    CollocationData(alg, experiment, invprob)

Get the collocated data by passing in `alg` - collocation algorithm and experiment.
This will return a `CollocationData` object which contains estimated states and derivatives along with the data.
As the parameters are not passed, derivatives from the model cannot be computed.
"""
function CollocationData(alg::AbstractCollocationAlgorithm,
        experiment::AbstractExperiment,
        invprob)
    model = get_model(experiment)
    data = get_data(experiment)
    new_data = collocation_cache(model, data, experiment, invprob)
    check_possible_collocation(model, new_data, model_unknowns(invprob, experiment))
    u_est, du_est = collocate_data(alg, new_data.data, new_data.time)
    CollocationData(
        typeof(new_data.data) <: AbstractVector ? reshape(new_data.data, 1, :) :
        new_data.data,
        u_est,
        nothing,
        du_est,
        new_data.time,
        new_data.save_names)
end

###################### Util functions ######################

function collocation_cache(model, data, experiment, invprob)
    # mc = get_cache(invprob, experiment)
    model = get_model(experiment)
    # Get the observed and state variables
    observed_vars = observed_names(model)
    vars = variable_symbols(model)

    # Get the variables in data
    vars_in_data = get_saved_model_variables(experiment)

    # Get the indices of observed vars in data
    idxs_observed = filter(x -> !isnothing(x),
        map(x -> findfirst(isequal(x), observed_vars), vars_in_data))

    # If no observed vars, return
    isempty(idxs_observed) && return data

    # If x is not passed, cannot compute the states
    # isnothing(x) && error("Cannot compute states from observed with no parameters passed.")
    # Get the indices of states in data
    idxs_states = filter(!isnothing,
        map(x -> findfirst(isequal(x), vars), vars_in_data))

    # Get the states and observed vars for which we have data
    known_states = vars[idxs_states]
    known_observed = observed_vars[idxs_observed]

    # Get the unknown states
    unknown_states = vars[filter(x -> !in(x, idxs_states), 1:length(vars))]

    # Get the indices of data to identify which variable is a state and which is observed var
    idxs_observed_data = map(x -> findfirst(isequal(x), vars_in_data), known_observed)
    idxs_state_data = map(x -> findfirst(isequal(x), vars_in_data), known_states)

    # Get the equations of the observed for which we have data
    filtered_eqs = observed(model)[idxs_observed]

    # Replace all the observed variables in the RHS till we get only states
    expr_dict = Dict(observed_vars .=> getproperty.(observed(model), :rhs))
    rhs_exprs = map(x -> ModelingToolkit.fixpoint_sub(getproperty(x, :rhs), expr_dict),
        filtered_eqs)

    # New equations where observed = f(states)
    new_filtered_eqs = [getproperty(filtered_eqs[i], :lhs) ~ rhs_exprs[i]
                        for i in eachindex(rhs_exprs)]

    # unknown states in the equations
    u0s = filter(x -> ispresent(x, unknown_states),
        unique(reduce(vcat, get_variables.(rhs_exprs))))

    # All states in the equations are known
    if isempty(u0s)
        return TimeSeriesData{Val{true}}(data[idxs_state_data, :],
            data.time,
            Symbol.(known_states),
            data[idxs_state_data, :])
    end

    # More unknowns than equations
    length(u0s) > length(new_filtered_eqs) &&
        error("Cannot compute states from observed as the number of equations is not sufficient")

    # Parameters are t, states for which we have data, observed vars, parameters of the system
    iv = ModelingToolkit.get_iv(model)
    ps_model = parameter_symbols(model)
    ps = vcat(iv, vars_in_data, ps_model)

    # Construct the non linear system
    defs = Dict(ps_model .=> getindex.((defaults(model),), ps_model))
    @named ns = NonlinearSystem(new_filtered_eqs, u0s, ps, defaults = defs)

    return u0s, ns
end

function compute_states_from_observed(model, data, experiment, invprob, u0s, ns, x)
    model = get_model(experiment)
    vars_in_data = get_saved_model_variables(experiment)

    # Get the states and observed vars for which we have data
    known_states = recorded_model_vars(variable_symbols, experiment, invprob)

    known_observed = recorded_model_vars(observed_names, experiment, invprob)

    # Get the indices of data to identify which variable is a state and which is observed var
    idxs_observed_data = map(x -> findfirst(isequal(x), vars_in_data), known_observed)
    idxs_state_data = map(x -> findfirst(isequal(x), vars_in_data), known_states)

    u = zeros(eltype(x), length(u0s), length(data.time))

    iv = ModelingToolkit.get_iv(model)
    ps_model = parameter_symbols(model)

    # Construct the non linear problem
    ns_prob = NonlinearProblem(complete(ns), [], [(vars_in_data .=> 0.0)..., iv => 0.0])

    # Get the parameters

    x_cp = copy(x)
    transform_params!(:inverse, x_cp, invprob)
    # p = get_params(mc, experiment, x)
    p = get_updated_param(ps_model, experiment, invprob, x)
    parameter_map = ps_model .=> p

    # Compute the jacobian
    jacobian = ModelingToolkit.calculate_jacobian(ns)
    det_jacobian = det(jacobian)
    substitute(det_jacobian, parameter_map) == Num(0.0) &&
        error("Cannot do collocation as the states cannot be computed from the observed data.")

    # Solve the non linear problem at each time point
    for i in eachindex(data.time)
        new_prob = remake(ns_prob,
            p = [(known_observed .=> data.data[idxs_observed_data, i])...,
                (known_states .=> data.data[idxs_state_data, i])..., iv => data.time[i],
                parameter_map...])
        sol = solve(new_prob)
        @views u[:, i] = sol.u
    end

    # Construct `TimeSeriesData`
    TimeSeriesData{Val{true}}(vcat(u, data[idxs_state_data, :]),
        data.time,
        Symbol.(vcat(u0s, known_states)),
        vcat(u, data[idxs_state_data, :]))
end

function check_possible_collocation(model, data, model_unknowns)
    # @debug model_unknowns
    solved_unknowns_in_data = find_unknowns_in_data(data, model, model_unknowns)
    # @debug solved_unknowns_in_data
    data_idxs = filter(x -> !isnothing(x),
        map(x -> findfirst(isequal(x), model_unknowns), solved_unknowns_in_data))

    # @debug data_idxs
    # No states are in the data
    if isempty(data_idxs)
        error("Cannot use Collocation as the data does not have any states.")
    end

    varvardeps = varvar_dependencies(asgraph(model), variable_dependencies(model))
    for varvar in varvardeps.badjlist
        # Data is missing for the dependent states of this state
        if !all(map(x -> in(x, data_idxs), varvar))
            error("Cannot use Collocation as the data is missing state(s) which is needed for derivative computation.")
        end
    end

    # return the indices of the states in the model
    return data_idxs
end

function derivatives_from_model(prob, invprob, experiment, x, u_est, data_idxs, saveat)
    # Set up derivative container
    unknowns = model_unknowns(invprob, experiment)
    du = zeros(eltype(x), length(unknowns), size(u_est, 2))
    u = zeros(eltype(x), length(unknowns), size(u_est, 2))
    # @debug size(u)
    # @debug size(u_est)
    @views u .= prob.u0
    @views u[data_idxs, :] .= u_est

    # Loop to get derivatives
    @inbounds for i in axes(u_est, 2)
        du_view = @view du[:, i]
        u_view = @view u[:, i]
        prob.f(du_view, u_view, parameter_values(prob), saveat[i])
    end
    return view(du, data_idxs, :)
end

function error_function(alg::AbstractCollocationAlgorithm,
        experiment::AbstractExperiment,
        du,
        du_est,
        prob,
        x)
    if alg.cutoff[1] + alg.cutoff[2] >= 1.0
        error("No points in the timeseries for the loss computation")
    end
    length_time_series = size(du, 2)
    start_idx = Int(round(alg.cutoff[1] * length_time_series + 1))
    end_idx = Int(round(length_time_series - alg.cutoff[2] * length_time_series))
    loss_func = get_loss_func(experiment)
    loss_func_wrapper(loss_func,
        prob,
        x,
        view(du, :, start_idx:end_idx),
        view(du_est, :, start_idx:end_idx))
end
