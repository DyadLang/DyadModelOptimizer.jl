struct PlotExperimentData{T}
    args::T
end

@recipe function f(plt::PlotExperimentData)
    experiment, prob, variables, sol = plt.args

    @series begin
        # this should be refactored
        # https://github.com/SciML/SciMLBase.jl/issues/10
        xlabel --> string(ModelingToolkit.get_iv(get_model(experiment)))
        !hasdata(experiment) &&
            throw(ErrorException("$(nameof(typeof(experiment))) named $(nameof(experiment)) does not have data."))
        get_data(experiment), variables, get_model(experiment)
    end
end

@recipe function f(td::AbstractExperimentData,
        variables::Union{<:AbstractVector, <:Tuple},
        model)
    i = 0
    for var in variables
        i += 1
        @series begin
            color --> i
            td, var, model
        end
    end
end

@recipe f(::NoData, variable::Union{<:AbstractVector, <:Tuple}, model) = error("No data to plot.")

@recipe function f(data::T, variable, model) where {T <: AbstractExperimentData}
    @series begin
        seriestype := :scatter
        legend --> false
        color --> 1
        label --> "data for $variable"
        data_shape(T), data, variable, model
    end
end

@recipe function f(::MatrixLike, data::TimeSeriesData, variable, model)
    idx = variable_index_in_data(variable, data, model)
    data.time, data[idx, :]
end

@recipe function f(::VectorLike, data::SteadyStateData, variable, model)
    idx = variable_index_in_data(variable, data, model)
    seriestype := :scatter
    xlabel --> "Model variables"
    markershape --> :rect
    ms --> 10
    [string.(variable)], data[idx, :]
end

@recipe function f(::VectorLike, data::TimeSeriesData, variable, model)
    # TODO find a better solution here
    # @assert isequal(string_to_state(string(only(data.save_names)), model), state) "$state was not saved in the data."
    data.time, data
end

@recipe function f(::MatrixLike, data::BoundsData, variable, model)
    idx = variable_index_in_data(variable, data, model)
    bounds = data[idx, :]
    lb = first.(bounds)
    ub = last.(bounds)
    t = data.time

    @series begin
        seriestype := :path
        linestyle --> :dash
        t, lb
    end
    @series begin
        seriestype := :path
        linestyle --> :dash
        t, ub
    end
end

struct PlotExperimentSolution{T}
    args::T
end

@recipe function f(plt::PlotExperimentSolution)
    experiment, prob, vars_to_plot, sol = plt.args

    if experiment isa SteadyStateExperiment
        for (i, st) in enumerate(vars_to_plot)
            label = [string(st)]
            @series begin
                seriestype := :scatter
                markershape --> :star5
                ms --> 10
                lab --> st
                label, [sol[st]]
            end
        end
    else
        # @debug "Plotting experiment solution"
        @series begin
            legend --> false
            if isnothing(get_save_idxs(experiment))
                idxs --> vars_to_plot
            else
                # symbolic indexing does not respect save_idxs
                # https://github.com/SciML/SciMLBase.jl/issues/37
                all_model_variables = variable_symbols(get_prob(experiment))
                variable_indices = map(
                    Base.Fix1(variable_index, sol),
                    vars_to_plot)
                save_idxs = get_save_idxs(experiment)
                Δvars = setdiff(variable_indices, save_idxs)
                @debug "variable_indices: $variable_indices"
                !isempty(Δvars) && error("$((all_model_variables[Δvars])) not saved")
                if variable_indices isa Vector
                    # look for the index of the states in the saved states
                    labels --> permutedims(all_model_variables[variable_indices])
                    idxs --> variable_indices
                else
                    # plot the first and only element
                    label --> all_model_variables[variable_indices]
                    idxs --> findfirst(==(variable_indices), save_idxs)
                end
            end
            sol
        end
    end
end

struct PlotExperimentExtras{T}
    args::T
end

@recipe function f(::PlotExperimentExtras{<:Tuple{<:AbstractExperiment, <:Any}}) end

# experiment plot recipe
@recipe function f(experiment::AbstractExperiment, prob::AbstractInverseProblem,
        x = initial_state(Any, prob);
        tspan = timespan(experiment),#, x, prob), # FIXME
        saveat = get_saveat(experiment),#, x, prob),
        show_data = false,
        show_doses = false,
        save_idxs_override = nothing,
        states = get_saved_model_variables(experiment))
    # remake the experiment to save everything regardless of the data,
    # so that we can pick with `states` variables that are not in the data
    sol = simulate(experiment, prob, x; tspan, saveat, save_idxs = save_idxs_override)
    @series begin
        PlotExperimentSolution((experiment, prob, states, sol))
    end

    show_data && @series begin
        PlotExperimentData((experiment, prob, states, sol))
    end

    show_doses && @series begin
        PlotExperimentExtras((experiment, sol))
    end
end

@recipe function f(experiment::AbstractExperiment, res::CalibrationResult)
    @series begin
        experiment, res.prob, res
    end
end

@userplot struct PlotAllExperiments{T <: Tuple{<:CalibrationResult}}
    args::T
end

@recipe function f(plt::PlotAllExperiments)
    r = plt.args[1]

    experiments = get_experiments(r.prob)

    for i in 1:length(experiments)
        @series begin
            experiments[i], r
        end
    end
end

"""
    confidenceplot(experiment::AbstractExperiment, ps::AbstractParametricUncertaintyEnsemble; confidence = 0.8, kwargs...)

Plots the trajectories for each state of `experiment` for a given confidence value of the quantile.

## Arguments

  - `experiment`: [`Experiment`](@ref) object.
  - `ps`: [`ParameterEnsemble`](@ref) object obtained from doing [`parametric_uq`](@ref).

## Keyword Arguments:

  - `confidence`: Defaults to `0.8`. A scalar value that shows the level of confidence (n-th quantile in loss) that the obtained plot is a good fit in comparison to the actual experimental data, out of the generated samples.
  - `show_data`: `Bool`, defaults to `true`, in order to show the dergee of fit with the actual data. Determines whether data of `experiment` is also plotted. If `true` data is plotted as a scatter plot on top of the state trajectories.
  - `states`: a `Vector` of model states, whose trajectories are plotted. Defaults to all saved states in `experiment`.
  - `kwargs`: These kwargs get forwarded to Plots.jl's `plot` function. Some of the useful ones can be `size`, `layout` etc.
"""
@userplot struct ConfidencePlot{
    T <: Tuple{<:AbstractExperiment,
    <:AbstractParametricUncertaintyEnsemble}}
    args::T
end

@recipe function f(cp::ConfidencePlot; confidence = 0.8,
        show_data = true,
        show_doses = false,
        states = get_saved_model_variables(cp.args[1]),
        save_idxs_override = nothing)
    experiment, ps = cp.args
    prob = ps.prob

    x, cost_var = get_params_at_confidence(confidence, ps)
    sol = simulate(experiment, prob, x, save_idxs = save_idxs_override)

    @series begin
        plot_title --> "Cost Value: $cost_var"
        PlotExperimentSolution((experiment, prob, states, sol))
    end

    show_data && @series begin
        PlotExperimentData((experiment, prob, states, sol))
    end

    show_doses && @series begin
        PlotExperimentExtras((experiment, sol))
    end
end

function get_params_at_confidence(confidence, ps)
    # TODO: use pre-computed cost values
    perm = sortperm(get_costvals(ps))
    sorted_vpop = ps[perm, :]
    sorted_cost = get_costvals(ps)[perm, :]
    vpop_confidence_idx = round(quantile(collect(1:size(ps, 1)), 1 - confidence))
    x = sorted_vpop[Int[vpop_confidence_idx][1]]
    plot_cost_value = sorted_cost[Int[vpop_confidence_idx][1]]

    cost_var = @sprintf "%.3e" plot_cost_value

    x, cost_var
end

function ensemble_subset(sim, states)
    u = [DiffEqArray(s[states], s.t) for s in sim]
    EnsembleSolution(u, sim.elapsedTime, sim.converged)
end

function ensemble_subset(sim, states::Vector)
    u = map(sim) do sol
        DiffEqArray([[sol[s][i] for s in states] for i in eachindex(sol.t)], sol.t)
    end
    EnsembleSolution(u, sim.elapsedTime, sim.converged)
end

@recipe function f(vp::AbstractParametricUncertaintyEnsemble,
        experiment::AbstractExperiment;
        summary = true,
        show_data = false,
        show_doses = false,
        states = get_saved_model_variables(experiment),
        quantiles = [0.05, 0.95],
        saveat_reduction = x -> only(unique(x)),
        save_idxs_override = nothing)
    # remake the experiment to save everything regardless of the data,
    # so that we can pick with `states` variables that are not in the data
    # experiment = remake(experiment, save_idxs = nothing)
    sim = solve_ensemble(vp, experiment; saveat_reduction, save_idxs = save_idxs_override)

    if summary && length(vp) ≠ 1
        labels = states isa Vector ? string.(permutedims(states)) : string(states)
        subset_sim = ensemble_subset(sim, states)
        summ = EnsembleSummary(subset_sim; quantiles)

        @series begin
            # idxs --> states
            labels --> labels
            summ
        end
    else
        @series begin
            # If states is a Term, we need to wrap it since it's not iterable
            sts = states isa Vector ? states : [states]
            idxs --> sts
            sim
        end
    end

    show_data && @series begin
        PlotExperimentData((experiment, vp.prob, states, sim.u[1]))
    end

    show_doses && @series begin
        PlotExperimentExtras((experiment, sim.u[1]))
    end
end

@recipe function f(vp::AbstractParametricUncertaintyEnsemble;
        layout = (1, 1),
        show_comparison_value <: Union{Bool, Vector{Int64}, Vector{Float64}},
        max_freq = [])
    table = DataFrame(vp)
    layout := layout
    comparison_values = []

    for i in 1:length(Tables.columns(vp))
        @series begin
            seriestype := :histogram
            bins --> 10
            color --> i
            lab --> string(Tables.columnnames(vp)[i])
            table[:, i]
        end
    end

    if typeof(show_comparison_value) <: Union{Vector{Int64}, Vector{Float64}}
        if isempty(max_freq)
            error("`max_freq ` kwarg is missing. Please provide this arg input...")
        else
            for i in 1:length(Tables.columns(vp))
                @series begin
                    linestyle := :dash
                    lw --> 3
                    color --> "Black"
                    lab --> "Comparison value for " * string(Tables.columnnames(vp)[i])
                    [show_comparison_value[i], show_comparison_value[i]], [0, max_freq[i]]
                end
            end
        end
    elseif show_comparison_value == true
        comparison_values = search_space_defaults(vp.prob)

        if isempty(max_freq)
            error("`max_freq ` kwarg is missing. Please provide this arg input...")
        else
            for i in 1:length(Tables.columns(vp))
                @series begin
                    linestyle := :dash
                    lw --> 3
                    color --> "Black"
                    lab --> "Comparison value for " * string(Tables.columnnames(vp)[i])
                    [comparison_values[i], comparison_values[i]], [0, max_freq[i]]
                end
            end
        end
    end
end

function skip_ss(experiments)
    filter(experiments.experiments) do e
        !(e isa SteadyStateExperiment)
    end
end

@recipe function f(ps::AbstractParametricUncertaintyEnsemble, prob::AbstractInverseProblem;
        experiment_names = get_name.(skip_ss(get_experiments(prob))),
        layout = (length(experiment_names), 1),
        summary = true,
        show_data = false,
        show_doses = false,
        quantiles = [0.05, 0.95])
    experiments = skip_ss(get_experiments(prob))
    layout := layout
    for (i, experiment) in enumerate(experiments)
        if get_name(experiment) in experiment_names
            @series begin
                subplot := i
                title := "$(get_name(experiment))"
                summary --> summary
                quantiles --> quantiles
                show_data --> show_data
                show_doses --> show_doses
                ps, experiment
            end
        end
    end
end

"""
    plot_shooting_segments(experiment::AbstractExperiment, r; kwargs...)

Plots each segment of the trajectory by simulating the `experiment` using the parameters and initial conditions present in `r` which is of type [`CalibrationResult`](@ref).
This is used for visualizing results obtained using [`MultipleShooting`](@ref) for calibration.

# Arguments

  - `experiment`: [`Experiment`](@ref) object.
  - `r`: [`CalibrationResult`](@ref) object.

# Keyword Arguments

  - `kwargs`: These kwargs get forwarded to Plots.jl's `plot` function. Some of the useful ones can be `size`, `layout` etc.
"""
@userplot struct Plot_Shooting_Segments{
    T <: Tuple{<:AbstractExperiment, <:CalibrationResult},
}
    args::T
end

@recipe function f(pss::Plot_Shooting_Segments;
        states = get_saved_model_variables(pss.args[1]),
        show_data = hasdata(pss.args[1]))
    experiment, r = pss.args
    sim = multiple_shoot_ensemble(r, experiment)
    prob = r.prob

    @series begin
        # If states is a Term, we need to wrap it since it's not iterable
        sts = states isa Vector ? states : [states]
        idxs --> sts
        sim
    end

    show_data && @series begin
        PlotExperimentData((experiment, prob, states, sim.u[1]))
    end
end

"""
    confidence_plot_shooting_segments(experiment::AbstractExperiment, ps::AbstractParametricUncertaintyEnsemble; kwargs...)

Plots the trajectories for each state of `experiment` for a given confidence value of the quantile.

## Keyword Arguments:

  - `confidence`: Defaults to `0.8`. A scalar value that shows the level of confidence that the obtained plot is a good fit in comparison to the actual experimental data, out of the generated samples.
  - `show_data`: `Bool`, defaults to `true`, in order to show the dergee of fit with the actual data. Determines whether data of `experiment` is also plotted. If `true` data is plotted as a scatter plot on top of the state trajectories.
  - `kwargs`: These kwargs get forwarded to Plots.jl's `plot` function. Some of the useful ones can be `size`, `layout` etc.
"""
@userplot struct Confidence_Plot_Shooting_Segments{
    T <: Tuple{<:AbstractExperiment,
    <:AbstractParametricUncertaintyEnsemble
}}
    args::T
end

@recipe function f(cp::Confidence_Plot_Shooting_Segments; confidence = 0.8,
        show_data = true,
        show_doses = false)
    experiment, vp = cp.args
    states = get_saved_model_variables(experiment)
    # perm = sortperm(get_costvals(vp))
    # sorted_vpop = vp[perm, :]
    # sorted_cost = get_costvals(vp)[perm, :]
    # vpop_confidence_idx = round(quantile(collect(1:size(vp, 1)), 1 - confidence))
    # x = sorted_vpop[Int[vpop_confidence_idx][1]]
    # plot_cost_value = sorted_cost[Int[vpop_confidence_idx][1]]
    x, cost_var = get_params_at_confidence(confidence, vp)

    y = 1
    for i in 1:length(vp)
        if vp[i][1] == x
            y = i
            break
        end
    end

    # cost_var = @sprintf "%.3e" plot_cost_value
    sol = multiple_shoot_ensemble(vp[y], experiment)
    prob = vp[y].prob

    @series begin
        plot_title --> "Cost Value: $cost_var"
        # If states is a Term, we need to wrap it since it's not iterable
        sts = states isa Vector ? states : [states]
        idxs --> sts
        sol
    end

    show_data && @series begin
        PlotExperimentData((experiment, prob, states, sol[1]))
    end

    show_doses && @series begin
        PlotExperimentExtras((experiment, sol[1]))
    end
end

"""
    convergenceplot(r; full = false, stack = false, kwargs...)

Plots the loss history.

# Arguments

  - `r`: Single or a vector of [`CalibrationResult`](@ref) objects.

# Keyword Arguments

  - `full`: Boolean variable to indicate whether the full loss history is to be plotted or not. If set to `false`, monotonically non increasing curve is plotted, i.e., loss is updated only when it decreases.
  - `stack`: Boolean variable to plot different loss histories in the same plot or in separate plots.
  - `kwargs`: These kwargs get forwarded to Plots.jl's `plot` function. Some of the useful ones can be `size`, `layout` etc.
"""
@userplot struct ConvergencePlot{T}
    args::T
end

@recipe function f(cp::ConvergencePlot{<:Tuple{<:CalibrationResult}}; full = false)
    r = only(cp.args)

    if full
        lh = r.loss_history
    else
        lh = monotonous_convergence_curve(r.loss_history)
    end

    @series begin
        lh
    end
end

@recipe function f(cp::ConvergencePlot{<:Tuple{<:Vector{<:CalibrationResult}}};
        stack = false,
        full = false)
    rs = only(cp.args)

    i1 = 1
    i2 = 0
    for r in rs
        if full
            lh = r.loss_history
        else
            lh = monotonous_convergence_curve(r.loss_history)
        end
        i1 += i2
        i2 += length(lh)
        name = string(nameof(typeof(r.alg.optimizer)))
        if stack
            @series begin
                ylabel --> "Objective value"
                xlabel --> "Optimizer iterations"
                label --> name
                lh
            end
        else
            @series begin
                ylabel --> "Objective value"
                xlabel --> "Optimizer iterations"
                label --> name
                i1:i2, lh
            end
        end
    end
end

@recipe function f(cd::CollocationData; vars = "states")
    layout --> (size(cd.u_est, 1), 1)
    legend --> :best
    derivatives_model_present = isnothing(cd.du) ? false : true
    for i in axes(cd.u_est, 1)
        if vars == "states"
            @series begin
                subplot := i
                label := ["u" "u_est"]
                title := cd.names[i]
                cd.time, [view(cd.u, i, :), view(cd.u_est, i, :)]
            end
        elseif vars == "derivatives"
            @series begin
                subplot := i
                label := derivatives_model_present ? ["du" "du_est"] : "du_est"
                title := cd.names[i]
                cd.time,
                derivatives_model_present ? [view(cd.du, i, :), view(cd.du_est, i, :)] :
                [view(cd.du_est, i, :)]
            end
        else
            error("Choose plotting between \"states\" and \"derivatives\"")
        end
    end
end

function monotonous_convergence_curve(full_loss_history)
    lh = similar(full_loss_history)
    l1, rest_of_lh = Iterators.peel(full_loss_history)
    lh[1] = l1
    last_loss = l1
    for (i, current_loss) in enumerate(rest_of_lh)
        if current_loss < last_loss
            # i starts from 1, but we start with the second
            # element in lh
            lh[i + 1] = current_loss
            last_loss = current_loss
        else
            lh[i + 1] = last_loss
        end
    end

    return lh
end

get_problem(vp::AbstractParametricUncertaintyEnsemble) = vp.prob
get_method(vp::AbstractParametricUncertaintyEnsemble) = vp.alg.method

function get_costvals(vp::AbstractParametricUncertaintyEnsemble)  # Includes penalty
    cost = objective(get_problem(vp), get_method(vp))
    cost.(vp.u)
end
