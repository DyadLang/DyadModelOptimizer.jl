"""
    DesignConfiguration(model; kwargs...)

The `DesignConfiguration` represents the target for a design optimization problem, making use of [`Experiment`](@ref) internally.
The dynamics of the investigated system are represented in `model` as an [ODESystem](https://docs.sciml.ai/ModelingToolkit/stable/systems/ODESystem/).
The configuration is used within an optimization problem corresponding to a [`InverseProblem`](@ref) to define the
target objective that is to be achieved using the tunable parameters and initial conditions of the system.

The contribution of the `DesignConfiguration` to the cost function is computed using the integrated running cost,
expressed here by the `reduction` and `running_cost` keyword arguments. The `running_cost` computes or specifies symbolically
the elementwise running cost, i.e. the value of the running cost for each saved element of the solution and the `reduction`
receives those values as a single argument and returns the corresponding (integrated) value.
In the symbolic form, the `running_cost` keyword argument accepts a symbolic expression using variables and parameters corresponding to the model
and is internally transformed into a function that evaluates the expression based on a given ODE solution that corresponds to a design configuration.
The functional form for the `running_cost` is a function that requires 3 arguments,
the tuned values of the parameters or initial conditions (i.e. what was provided as the search space),
the solution to the `ODEProblem` corresponding to the design configuration and the last argument `data`, which can be used to access additional information
and can be provided via the `data` keyword argument. The function is expected to return a scalar value corresponding
to the loss that is associated to the design configuration defined by the tuned values passed in the first argument.
If there is no easy way of expressing the loss function with `running_cost` and `reduction`, one can directly provide the `loss_func`
keyword argument from [`Experiment`](@ref).

The cost function corresponding to the design configuration forms the objective function for the optimization problem defined by the
inverse problem. The tuned values of the parameters that are tried during the optimization are then used to solve the `ODEProblem`
corresponding to the design configuration. The solution is provided to the running cost and the available timepoints are defined by
`saveat`.

Constraints that define the design configuration can be provided using the `constraints` keyword argument in the form of a vector of
equations or inequalities. If the expression contains time dependent variables, then the expression will be automatically discretized and
evaluated at `constraints_ts`, which is by default the same as `saveat`. If the constraints should be evaluated at different times
from the running cost, such as when a denser discretization is needed around an event, the `constraints_ts` keyword can be used to provide
an arbitrary vector of timepoints to be used.

## Positional arguments

  - `model`: An `ODESystem` describing the model that we are using for the design configuration. The model needs to define defaults
    for all initial conditions and all parameters.

## Keyword arguments

  - `constraints`: a vector of equations representing equality or inequality constraints.
  - `u0`: fix the initial conditions for the experiment (e.g. `u0 = [state_name => custom_initial_value]`), see [`Experiment`](@ref) for more details.
  - `params`: fix the parameters for the experiment (e.g. `params = [p1 => specific_value, ... p3 => other_value]`), see [`Experiment`](@ref) for more details.
  - `model_transformations`: Apply some transformations to the model used in this experiment,
    such as using [`DiscreteFixedGainPEM`](@ref) (e.g. `model_transformations = (DiscreteFixedGainPEM(alpha),)`).
  - `callback`: A callback or a callback set to be used during the simulation.
    See https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/ for more details.
  - `tspan`: the timespan to use when integrating the equations.
  - `saveat`: the times at which the solution should be saved using interpolations. Defaults to saving where the integrator stops.
    This controls the timepoints when the running cost is evaluated.
  - `constraints_ts`: the times at which time dependent constraints should be evaluated. Defaults to `saveat`.
  - `running_cost`: the contribution to the loss corresponding to this design configuration (e.g. `running_cost = (tuned_vals, sol, data) -> (sol[sys.var1] - ref_val)^2`
    or `running_cost = (sys.var1 - ref_val)^2)`.
  - `reduction`: this function is applied to the result of the `running_cost`, acting like an integration, and is expected to return a scalar. By default `mean` is used.
  - `loss_func`: if one wants to override the `reduction(running_cost)` description of the loss function, a function with the `(tuned_vals, sol, data)` signature can be passed.
  - `prob_kwargs`: A `NamedTuple` indicating keyword arguments to be passed to the `ODEProblem` constructor.
    See [the ModelingToolkit docs](https://docs.sciml.ai/ModelingToolkit/stable/systems/ODESystem/#SciMLBase.ODEProblem-Tuple%7BModelingToolkit.AbstractODESystem,%20Vararg%7BAny%7D%7D)
    for more details.
  - `dependency`: This keyword can be be assigned to a variable representing an other (previously defined) experiment to express the
    fact that the initial conditions for this experiment depend on the solution of the given experiment. For example if one
    experiment (e.g. `experiment_ss`) defines a steady state, we can use that for the definition of the initial conditions
    for a subsequent experiment with `dependency=experiment_ss`.

If additional keywords are passed, they will be forwarded to the `solve` call. For example,
one can pass `alg=Tsit5()` to specify what solver will be used. More information about supported
arguments can be found [here](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/#solver_options).
"""
function DesignConfiguration(
        model::AbstractTimeDependentSystem;
        overrides = Pair[],
        constraints = [],
        model_transformations = (),
        callback = CallbackSet(),
        tspan,
        saveat = (),
        constraints_ts = determine_constraints_ts(constraints, saveat, tspan),
        running_cost = nothing,
        loss_func = nothing,
        prob_kwargs = (;),
        reduction = mean,
        postprocess = last,
        dependency = nothing,
        name = "DesignConfiguration",
        alg = missing,
        data = nothing,
        kwargs...)
    loss_func = isnothing(loss_func) ?
                determine_loss_func(
        running_cost, reduction, model, overrides, tspan, prob_kwargs) : loss_func
    Experiment(data, model;
        overrides,
        constraints,
        model_transformations,
        callback,
        tspan,
        saveat,
        constraints_ts,
        reduction,
        loss_func,
        postprocess,
        dependency,
        name,
        alg,
        prob_kwargs,
        depvars = isnothing(data) ? determine_save_names(data, model) : nothing,
        kwargs...)
end
