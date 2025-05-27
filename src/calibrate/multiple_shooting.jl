"""
    cost_contribution(alg, experiment::AbstractExperiment, prob, x)

Compute the contribution of the given `experiment` to the total cost associated
with the inverse problem `prob` using the given calibration algorithm (`alg`).

## Positional arguments

  - `alg`: the calibration algorithm that formulates the objective function.
  - `experiment`: the experiment for which we compute the contribution.
  - `prob`: the [`InverseProblem`](@ref) defining the objective.
  - `x`: the values of the tuned parameters or initial conditions.
"""
function cost_contribution(::SingleShooting, experiment::AbstractExperiment, prob,
        x = initial_state(Any, prob),
        alg_cache = nothing;
        loss_func = get_loss_func(experiment))
    sol = trysolve(experiment, prob, x)
    isinf(check_retcode(sol, nameof(experiment))) && return Inf
    return compute_error(experiment, prob, x, sol; loss_func)
end

function cost_contribution(alg::DataShooting,
        experiment::AbstractExperiment,
        prob,
        x = search_space_defaults(prob, true),
        alg_cache = nothing)
    # TODO: update it to symbolic timespan when working
    time_intervals = split_timespan(alg, timespan(experiment),
        get_saveat(experiment))
    u0s = data_shoot_u0s(time_intervals, experiment, x, prob)

    multiple_shoot_loss(alg, x, u0s, prob, experiment, time_intervals)
end

function data_shoot_u0s(time_intervals, experiment, x, prob)
    ist = get_internal_storage(prob)
    all_model_unknowns = model_unknowns(prob, experiment)
    n_states = length(all_model_unknowns)
    n_saved = count_saved_states(experiment, ist)
    save_idxs = get_save_idxs(experiment)
    # TODO: maybe use additional internal parameters to fit missing initial conditions

    u0s = zeros(eltype(x), n_states, length(time_intervals) - 1)
    # @debug "n_states = $n_states, lt: $(length(time_intervals))"
    # The u0s are assumed to be in order of the states in the model,
    # but the data doesn't guarantee that.
    saved_vars = get_saved_model_variables(experiment)
    idxs = Vector{Int}(undef, n_states)
    i = 1
    for var in saved_vars
        vi = indexof(var, all_model_unknowns)
        if !isnothing(vi)
            idxs[i] = vi
            i += 1
        end
    end
    @assert i - 1==n_states "All states of the system must be known for DataShooting, got $i states instead of $n_states."

    for (i, (t0, t_end)) in enumerate(time_intervals[2:end])
        data = get_data(experiment)
        @views u0s[:, i] = data_subset(data, t0, t_end)[idxs, 1]
    end

    return u0s
end

function cost_contribution(alg::MultipleShooting,
        experiment::AbstractExperiment,
        prob,
        x = initial_state(alg, prob),
        alg_cache = nothing)
    ist = get_internal_storage(prob)
    time_intervals = split_timespan(alg, timespan(experiment),#, x, prob), # FIXME
        get_saveat(experiment))
    u0s = internal_params_part(alg, experiment, x, prob)
    multiple_shoot_loss(alg, x, u0s, prob, experiment, time_intervals)
end

continuity_type(::Type{<:MultipleShooting{C}}) where {C} = C
continuity_type(::Type{<:DataShooting}) = ModelStatesPenalty

function split_timespan(alg::MultipleShooting, (t0, t_end), saveat)
    n = alg.trajectories
    Δt = (t_end - t0) / n

    [(t0 + (i - 1) * Δt, t0 + i * Δt) for i in 1:n]
end

function split_timespan(alg::DataShooting, (t0, t_end), saveat)
    interval_saveat_points = alg.groupsize

    T = typeof(t0)
    ts = Vector{Tuple{T, T}}(undef, 0)
    t1 = t0
    t2 = saveat[1]
    included_saveat = 0
    for i in eachindex(saveat)
        t2 = saveat[i]
        included_saveat += 1
        # @debug "t1 = $t1\nt2 = $t2\nincluded_saveat = $included_saveat"
        # @debug "ts: $ts"
        t2 == t1 && continue
        if included_saveat == interval_saveat_points
            push!(ts, (t1, t2))
            t1 = t2
            included_saveat = 1
        end
    end

    if included_saveat > 1
        push!(ts, (t1, t_end))
    end

    return ts
end

function tspan_subset_idx(time, t0, t_end)
    ascending_time = first(time) < last(time)
    if ascending_time
        t0_idx = findfirst(t -> t ≥ t0, time)
        t_end_idx = findlast(t -> t ≤ t_end, time)
    else
        t0_idx = findfirst(t -> t ≤ t0, time)
        t_end_idx = findlast(t -> t ≥ t_end, time)
    end

    # sanity checks

    @assert !isnothing(t0_idx) "invalid starting time for the segment: $t0. Data starts at $(first(time))"
    @assert !isnothing(t_end_idx) "invalid ending time for the segment: $t0. Data ends at $(last(time))"

    return t0_idx, t_end_idx
end

function data_subset(data::T, t0, t_end) where {T <: AbstractExperimentData}
    data_subset(data_shape(T), data, t0, t_end)
end
data_subset(::NoData, t0, t_end) = NoData()

function data_subset(::MatrixLike, data, t0, t_end)
    t0_idx, t_end_idx = tspan_subset_idx(data.time, t0, t_end)

    @views data[:, t0_idx:t_end_idx]
end

function data_subset(::VectorLike, data, t0, t_end)
    t0_idx, t_end_idx = tspan_subset_idx(data.time, t0, t_end)

    @views data[t0_idx:t_end_idx]
end

function segment_loss(sol, i, experiment, time_intervals, u0s, alg, invprob, x)
    (t0, t_end) = time_intervals[i]
    full_data = get_data(experiment)
    data = data_subset(full_data, t0, t_end)
    isinf(check_retcode(sol, nameof(experiment))) && return Inf
    # @debug "segment $i"
    # @debug "length(sol): $(length(sol))"
    save_idxs = get_save_idxs(experiment)
    saveat = get_saveat(experiment)
    if !isempty(saveat)
        t0_idx, t_end_idx = tspan_subset_idx(saveat, t0, t_end)
        saveat_slice = @views saveat[t0_idx:t_end_idx]
    else
        saveat_slice = sol.t
    end
    # even though we pass saveat, we still need to use interpolation
    # as we also pass save_end=true, so depending on the saveat
    # the solution might have one extra point compared to the data
    replaced_sol = replace_sol(experiment, sol, saveat_slice, save_idxs)
    # @debug replaced_sol.u

    loss = if (hasdata(experiment) && isempty(data))
        zero(eltype(sol))
    else
        data_stats = isa(full_data, NoData) ? (;) :
                     (
            data_mean = mean(full_data, dims = 2), data_std = std(full_data, dims = 2))
        loss_func_wrapper(
            get_loss_func(experiment), invprob, x, replaced_sol, data; data_stats...)
    end
    # @debug "data loss: $(ForwardDiff.value(loss))"
    loss += continuity_penalty(alg, sol, replaced_sol, u0s, i, experiment)

    return loss
end

function continuity_penalty(alg::T, sol, rsol, u0s, i, ex) where {T}
    continuity_penalty(continuity_type(T), alg, sol, rsol, u0s, i, ex)
end

function continuity_penalty(::Type{<:ModelStatesPenalty}, alg, sol, rsol, u0s, i, ex)
    continuity_penalty = zero(eltype(sol))
    c = alg.continuity
    if i ≤ size(u0s, 2)
        next_u0 = view(u0s, :, i)
        # @debug "computing model states discontinuity penalty for the segment no. $i for $(sol.u[end]) and $(next_u0)"
        continuity_penalty = c.continuitylossfun(nothing, sol.u[end], next_u0) *
                             c.continuitylossweight
        # @debug "continuity penalty: $continuity_penalty"
    end
    return continuity_penalty
end

function continuity_penalty(::Type{<:ConstraintBased}, alg, sol, rsol, u0s, i, ex)
    zero(eltype(sol))
end

function ensemble_setup(x, u0s, invprob, experiment, time_intervals)
    # TODO: verify correctness with fixed u0
    prob = setup_problem(experiment, invprob, x;
        tspan = time_intervals[1])
    initial_u0 = prob.u0

    function prob_func(prob, i, repeat)
        if i == 1
            u0 = initial_u0
        else
            # u0s are the initial conditions for the shooting
            # intervals starting with the 2nd one, as for the
            # first we already know the initial conditions
            # from the experiment
            u0 = u0s[:, i - 1]
        end
        # @debug "initial prob.u0: $(prob.u0)"
        tunables_type = eltype(SciMLStructures.canonicalize(Tunable(), prob.p)[1])
        promoted_u0 = promote_type(tunables_type, eltype(u0)).(u0)
        # @debug "$i-th u0: $(promoted_u0)"
        new_u0, new_p = get_config(experiment).set_all_u0s(prob, promoted_u0)
        # @debug new_u0
        # @debug new_p.tunable
        remake(prob; u0 = new_u0, p = new_p, tspan = time_intervals[i],
            initializealg = BrownFullBasicInit())
    end

    return prob, prob_func
end

function multiple_shoot_ensemble(alg, x, u0s, invprob, experiment, time_intervals)
    prob, prob_func = ensemble_setup(x, u0s, invprob, experiment, time_intervals)

    function output_func(sol, i)
        (segment_loss(sol, i, experiment, time_intervals, u0s, alg, invprob, x), false)
    end

    ensembleprob = EnsembleProblem(prob; prob_func, output_func, safetycopy = false)
    ensemblealg = alg.ensemblealg

    # we can avoid saving everywhere as we'll only need the points at saveat
    # for the loss function
    # note that it is very important to save the end of the solution in order to
    # correctly enforce the continuity between segments (either via penalty or constraints)
    solve(ensembleprob, get_solve_alg(experiment), ensemblealg;
        trajectories = length(time_intervals), saveat = get_saveat(experiment),
        save_end = true,
        get_kwargs(experiment)...)
end

function multiple_shoot_ensemble(r::CalibrationResult, experiment::AbstractExperiment)
    x = r.original
    invprob = r.prob
    alg = r.alg
    time_intervals = split_timespan(alg, timespan(experiment),#, x, invprob), # FIXME
        get_saveat(experiment))
    u0s = internal_params_part(alg, experiment, x, invprob)

    prob, prob_func = ensemble_setup(x, u0s, invprob, experiment, time_intervals)

    ensembleprob = EnsembleProblem(prob; prob_func, safetycopy = false)
    ensemblealg = alg.ensemblealg

    # we don't pass saveat here because we want to show the full solution,
    # not just the points that are matching the data
    solve(ensembleprob, get_solve_alg(experiment), ensemblealg;
        trajectories = length(time_intervals),
        get_kwargs(experiment)...)
end

function multiple_shoot_loss(alg, x, u0s, invprob, experiment, time_intervals)
    sim = try
        multiple_shoot_ensemble(alg, x, u0s, invprob, experiment, time_intervals)
    catch e
        e isa UNEXPECTED_EXCEPTION && rethrow()
        SOLVE_FAIL_WARN && @warn "solve failed with $(typeof(e))" maxlog=5 exception=e
        return Inf
    end

    sum(sim)
end

function multiple_shoot_loss(r::CalibrationResult, experiment::AbstractExperiment)
    x = r.original
    invprob = r.prob
    alg = r.alg
    time_intervals = split_timespan(alg, timespan(experiment, x, invprob),
        get_saveat(experiment))
    u0s = internal_params_part(alg, experiment, x, invprob)

    sim = try
        multiple_shoot_ensemble(alg, r, u0s, invprob, experiment, time_intervals)
    catch e
        SOLVE_FAIL_WARN && @warn "solve failed with $(typeof(e))" maxlog=3 exception=e
        return Inf
    end

    sum(sim)
end

function internal_alg_params(alg::MultipleShooting, prob)
    experiments = get_experiments(prob)
    n_u0s = 0
    for experiment in experiments
        n_diff_vars = length(get_diff_variables(get_model(experiment)))
        # n_starts = length(get_starts(experiment, alg))
        tspan = timespan(experiment)
        saveat = get_saveat(experiment)
        n_intervals = length(split_timespan(alg, tspan, saveat))
        # we know the initial conditions for the first shooting
        # segment, they are the experiment initial conditions
        n_u0s += n_diff_vars * (n_intervals - 1)
    end
    ss_defs = search_space_defaults(prob)
    T = isempty(ss_defs) ? Float64 : reduce(promote_type, eltype.(ss_defs))
    u0s = zeros(T, n_u0s)
    # Only one trajectory - becomes equivalent to SingleShooting
    n_u0s == 0 && return u0s
    idx = 1
    n_u0s = 0
    for experiment in experiments
        n_diff_vars = length(get_diff_variables(get_model(experiment)))
        tspan = timespan(experiment)
        saveat = get_saveat(experiment)
        n_intervals = length(split_timespan(alg, tspan, saveat))
        n_u0s = n_diff_vars * (n_intervals - 1)
        @views u0s[idx:(idx + n_u0s - 1)] .= initialize_segments(alg.initialization,
            alg,
            experiment,
            prob,
            n_u0s,
            T)
        idx += n_u0s
    end
    return u0s
end

function initialize_segments(::DefaultSimulationInitialization,
        alg,
        experiment,
        prob,
        n_u0s,
        T)
    u0s = zeros(T, n_u0s)
    sim = simulate(experiment, prob, save_idxs = nothing)
    diff_vars = get_diff_variables(get_model(experiment))
    n_diff_vars = length(diff_vars)
    tspan = timespan(experiment)
    saveat = get_saveat(experiment)
    intervals = split_timespan(alg, tspan, saveat)
    u0s_view = @views u0s
    u0s_view_matrix = reshape(u0s_view, n_diff_vars, :)
    values = sim(first.(intervals[2:end]), idxs = diff_vars)
    for i in axes(u0s_view_matrix, 1)
        u0s_view_matrix[i, :] .= getindex.(values.u, i)
    end
    return u0s
end

function initialize_segments(::DataInitialization, alg, experiment, prob, n_u0s, T)
    u0s = zeros(T, n_u0s)
    data = get_data(experiment)
    model = get_model(experiment)
    # mc = get_cache(prob, experiment)
    model_vars = get_saved_model_variables(experiment)
    mu = get_diff_variables(model)
    initial_var_vals = getsym(get_prob(experiment), mu)(get_prob(experiment))

    # Get a mask of what states are present in the data in the states vector
    ms_masks = map(x -> ispresent(x, model_vars), mu)
    if !reduce(|, ms_masks)
        @warn "No model variable found in the data, segment initialization defaulting to 0s."
    end

    # Get indices of states in the data
    model_vars_indices = findall(map(x -> ispresent(x, mu), model_vars))

    splines = if data.data isa Vector
        [alg.initialization.interpolation(data.data, data.time)]
    else
        [alg.initialization.interpolation(data.data[i, :], data.time)
         for i in axes(data.data, 1)]
    end
    tspan = timespan(experiment)
    saveat = get_saveat(experiment)
    intervals = split_timespan(alg, tspan, saveat)
    n_diff_vars = length(mu)

    u0s_view = @views u0s
    u0s_view_matrix = reshape(u0s_view, n_diff_vars, :)
    j = 1
    for i in axes(u0s_view_matrix, 1)
        if ms_masks[i]
            u0s_view_matrix[i, :] .= splines[model_vars_indices[j]](first.(intervals[2:end]))
            j += 1
        else
            u0s_view_matrix[i, :] .= initial_var_vals[i]
        end
    end
    return u0s
end

function initialize_segments(::RandomInitialization, alg, experiment, prob, n_u0s, T)
    u0s = zeros(T, n_u0s)
    # mc = get_cache(prob, experiment)
    model = get_model(experiment)
    initial_states = state_values(get_prob(experiment))
    n_diff_vars = length(get_diff_variables(get_model(experiment)))
    u0s_view = @views u0s
    u0s_view_matrix = reshape(u0s_view, n_diff_vars, :)
    for i in axes(u0s_view_matrix, 1)
        u0s_view_matrix[i, :] .= initial_states[i] * rand()
    end
    return u0s
end

function continuity_states(::Type{<:ModelStatesPenalty}, ex, ist)
    length(get_diff_variables(get_model(ex)))
end

function compute_residual(alg::T, prob) where {T <: MultipleShooting}
    compute_residual(continuity_type(T), alg, prob)
end

compute_residual(::Any, alg, prob) = nothing

function compute_residual(::Type{<:ConstraintBased}, alg::MultipleShooting, prob)

    # for each experiment, where we have n states, we must satisfy the following
    # constraints between the endpoins of the segments
    #
    # segment_1     segment_2   …   segment_m       segment_m+1
    #   λ1      =     λ_n+1         λ_(m-1)n+1  =     λ_m*n+1
    #   λ2      =     λ_n+2         λ_(m-1)n+2  =     λ_m*n+2
    #           ⋮               ⋱               ⋮
    #   λn      =     λ_2n      …   λ_m*n       =     λ_(m+1)*n

    function (res, x, p)
        experiments = get_experiments(prob)
        idx = 1
        for experiment in experiments
            time_intervals = split_timespan(alg, timespan(experiment),#, x, prob), # FIXME
                get_saveat(experiment))
            u0s = internal_params_part(alg, experiment, x, prob)

            eprob, prob_func = ensemble_setup(x, u0s, prob, experiment, time_intervals)

            ensembleprob = EnsembleProblem(eprob; prob_func, safetycopy = false)
            ensemblealg = alg.ensemblealg

            sim = solve(ensembleprob, get_solve_alg(experiment), ensemblealg;
                trajectories = length(time_intervals), save_everystep = false,
                save_end = true,
                get_kwargs(experiment)...)

            n_diff_vars = length(get_diff_variables(get_model(experiment)))
            n_intervals = length(time_intervals) - 1

            # @debug "n states: $n_diff_vars"
            # @debug "n segments: $n_intervals"
            @views res_view = res[idx:(idx + (n_diff_vars * n_intervals) - 1)]
            mat_res = reshape(res_view, n_diff_vars, n_intervals)
            for m in 1:n_intervals
                # @debug m
                # this is a bit weird
                mat_res[:, m] = sim.u[m].u[end][1:n_diff_vars] - view(u0s, :, m)
            end
            idx += n_diff_vars * n_intervals
        end
    end
end
