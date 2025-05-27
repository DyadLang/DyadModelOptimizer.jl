"""
    objective(prob, alg)

When considering the [`InverseProblem`](@ref) in the context of calibration, we can define
an objective function that associates a number to a vector of possible search space values.
This number is a measure of how close is the given vector to a solution of the inverse problem.
The objective for an inverse problem is defined as a sum of the individual contributions from
each experiment.

```math
\\sum_{i=1}^{N} loss(i).
```

In the above `loss(i)` is the individual contribution of the `i`th experiment and this can be
computed via [`cost_contribution`](@ref) and is determined by the `loss_func` passed in the corresponding
experiment constructor.

Calling `objective(prob, alg)` will return a function of the search space values, so in order to
use it, it is best to store it in a variable as follows:

```julia
cost = objective(prob, alg);
```

In order to obtain a particular value that corresponds to some search space values, one has to
input a `NamedTuple` or a vector of pairs to the above mentioned function.

As an example, one would call

```julia
cost([a => 1.0, b => 3.0])
```

where the search space in the inverse problem was defined as containing the `a` and `b` parameters,
i.e. `[a => (0.0, 10.0), b => (-5.0, 5.0)]`.

The above function also accepts calibration results as input, so if we have a calibration result
obtained from something like

```julia
r = calibrate(prob, alg)
```

then we can compute the cost value corresponding to the calibration result using `cost(r)`, where
`cost` is the above mentioned variable.

The `cost` function returned by `objective` can also be called with 0 arguments, i.e. `cost()`,
in which case the cost will be computed using the initial state of the optimization problem,
which uses the default values present in the model for all parameters and initial conditions.

## Positional arguments

  - `prob`: the [`InverseProblem`](@ref) defining the objective
  - `alg`: the calibration algorithm that formulates the objective function
"""
function objective(prob::AbstractInverseProblem{<:AbstractExperimentCollection},
        alg::AbstractCalibrationAlgorithm)
    function cost(_x = initial_state(alg, prob),
            p = calibration_parameters(alg, prob))
        # x is the input from the OptimizationProblem. It can contain
        # both parameters and states (u0) for the ODEProblem
        invprob, c_alg, alg_cache = p
        experiments = get_experiments(invprob)

        # tighten eltype in case of user error
        # fallback to Float64 in case of empty params here in order to avoid
        # handling empty x downstream
        x = isconcretetype(eltype(_x)) ? _x : isempty(_x) ? Float64[] : map(identity, _x)

        err = initial_cost_val(x)
        for ex in experiments
            err += cost_contribution(c_alg, ex, invprob, x, alg_cache)
        end

        return err
    end

    cost(x::CalibrationResult, p) = cost(x.original, p)

    return cost
end

function calibration_parameters(alg, prob)
    prob, alg, initialize_cache(alg, prob)
end

initialize_cache(::AbstractCalibrationAlgorithm, prob) = nothing

function create_alg_cache(alg, prob)
    Dict(get_uuid(e) => algorithm_cache(alg, e, prob) for e in get_experiments(prob))
end

initial_cost_val(::AbstractVector{T}) where {T} = zero(T)
initial_cost_val(::AbstractVector{<:Pair{N, T}}) where {N, T <: Number} = zero(T)
initial_cost_val(::NamedTuple{names, T}) where {names, T} = zero(eltype(T))
