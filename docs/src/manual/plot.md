# Plot Functions

There are plot recipes and user plots defined in DyadModelOptimizer for easy plotting of results:

## Calibration

i.
```julia
plot(experiment::AbstractExperiment, prob::InverseProblem, x = initial_state(prob))
```

Plots the trajectories for each state of `experiment`, when point `x` is used to provide parameter and/or initial condition values that are optimized in `prob`. The last argument, `x`, defaults to the initial guess for the values of each parameter and initial condition to be optimized, as they were specified during model definition. `x` can be a `NamedTuple` or [`CalibrationResult`](@ref) object.

Arguments:

- `experiment`: [`Experiment`](@ref) object.
- `prob`: [`InverseProblem`](@ref) object.
- `x`: parameters/initial conditions included in the optimization.

ii.
```@docs
plot_shooting_segments
```

iii.
```julia
plot(cd::CollocationData; vars="states", kwargs...)
```

Plots the collocated data stored in `CollocationData` object against the ground truth (if available)

Arguments:

- `cd`: [`CollocationData`](@ref) object.

Keyword Arguments:

- `vars`: This can be either "states" or "derivatives" to plot the states and derivatives of it respectively.
- `kwargs`: These kwargs get forwarded to Plots.jl's `plot` function. Some of the useful ones can be `size`, `layout` etc.

vi.
```@docs
convergenceplot
```

## Parametric Uncertainty Quantification

i.
```julia
plot(ps::AbstractParametricUncertaintyEnsemble, experiment::AbstractExperiment; summary = true, quantile = [0.05, 0.95], show_data = false, kwargs...)
```

Plot the trajectories corresponding to each set of parameters (and/or initial conditions) from an ensemble given as result of parametric uncertainty quantification (i.e. results from `parametric_uq`).

Arguments:

- `ps`: [`ParameterEnsemble`](@ref) object obtained from doing [`parametric_uq`](@ref).
- `experiment`: [`Experiment`](@ref) object.

Keyword Arguments:

- `summary`: `Bool`, defaults to `true`. Determines whether summary statistics of the trajectories are plotted. If `true`, a mean trajectory is shown with a band around it representing a lower and upper quantile of the state distribution at each saved timepoint.
- `quantile`: Defaults to `[0.05, 0.95]`. A vector of two elements, corresponding to the lower and upper quantile of the distribution of each state at each saved timepoint, to be plotted if `summary == true`.
- `states`: a `Vector` of model states, whose trajectories are plotted. Defaults to all saved states in `experiment`.
- `show_data`: `Bool`, defaults to `false`. Determines whether data of `experiment` is also plotted. If `true` data is plotted as a scatter plot on top of the state trajectories.
- `kwargs`: These kwargs get forwarded to Plots.jl's `plot` function. Some of the useful ones can be `size`, `layout` etc.

ii.
```julia
plot(ps::AbstractParametricUncertaintyEnsemble, prob::InverseProblem; summary = true, quantile = [0.05, 0.95], show_data = false, kwargs...)
```

Plots the state trajectories of experiments that are part of [`InverseProblem`](@ref) `prob`,
using an ensemble given as result of parametric uncertainty quantification (i.e. results from `parametric_uq`).

Each experiment is shown in a separate subplot.

Arguments:

- `ps`: [`ParameterEnsemble`](@ref) object obtained from doing [`parametric_uq`](@ref).
- `prob`: [`InverseProblem`](@ref) object.

Keyword Arguments:

- `experiment_names`: `Vector` containing the names of experiments to be plotted. These experiments need to be part of `prob`.
- `layout`: `Tuple{Int, Int}`. Determines how experiments are shown on the plotting window. Defaults to one experiment per row. The `Tuple` should look like `(number_of_rows, number_of_columns)`.
- `summary`: `Bool`, defaults to `true`. Determines whether summary statistics of the trajectories are plotted. If `true`, a mean trajectory is shown with a band around it representing a lower and upper quantile of the state distribution at each saved timepoint.
- `quantile`: Defaults to `[0.05, 0.95]`. A vector of two elements, corresponding to the lower and upper quantile of the distribution of each state at each saved timepoint, to be plotted if `summary == true`.
- `show_data`: `Bool`, defaults to `false`. Determines whether data of each plotted experiment is also shown. If `true` data is plotted as a scatter plot on top of the state trajectories.
- `kwargs`: These kwargs get forwarded to Plots.jl's `plot` function. Some of the useful ones can be `size` etc.

iii.
```@docs
confidenceplot
```

iv.
```@docs
confidence_plot_shooting_segments
```
