# Neural Automated Model Discovery for Autocompleting Models with Prior Structural Information

When calibrating models to data we can have uncertainties about the parameter values, but also about the structure of the model. It could happen that no matter how we try to fit the data, there are no good enough parametrizations, which might point out that we need to adjust the equations of the model.

This is what Universal Differential Equations (UDEs for short) try to solve. Starting from the model and the data that you have, we find the minimal mechanistic extension that would provide a better fit. In this tutorial, we will show how to use the DyadModelOptimizer to extend a partially correct model and auto-complete it to find the missing physics. [^1]

The model that we will consider will be the Lotka-Volterra equations. These equations are given by:

```math
\begin{aligned}
\frac{dx}{dt} &= \alpha x - \beta x y      \\
\frac{dy}{dt} &= -\delta y + \gamma x y    \\
\end{aligned}
```

This is a model of rabbits and wolves. ``\alpha x`` is the exponential growth of rabbits
in isolation, ``-\beta x y`` and ``\gamma x y`` are the interaction effects of wolves
eating rabbits, and ``-\delta y`` is the term for how wolves die hungry in isolation.

Now assume that we have never seen rabbits and wolves in the same room. We only know the
two effects ``\alpha x`` and ``-\delta y``. Can we use Scientific Machine Learning to
automatically discover an extension to what we already know? That is what we will solve
with the universal differential equation.

## First steps

For this tutorial we will need the following packages:

| Module                                                                                                     | Description                                                                                |
|:---------------------------------------------------------------------------------------------------------- |:------------------------------------------------------------------------------------------ |
| [DyadModelOptimizer](https://help.juliahub.com/jsmo/stable/)                                           | The high-level library used to formulate our problem and perform automated model discovery |
| [ModelingToolkit](https://docs.sciml.ai/ModelingToolkit/stable/)                                           | The symbolic modeling environment                                                          |
| [OrdinaryDiffEq](https://docs.sciml.ai/DiffEqDocs/stable/) (DifferentialEquations.jl)                      | The numerical differential equation solvers                                                |
| [CSV](https://csv.juliadata.org/stable/) and [DataFrames](https://dataframes.juliadata.org/stable/)        | We will read our experimental data from .csv files                                         |
| [DataSets](https://help.juliahub.com/juliahub/stable/tutorials/datasets_intro/)                            | We will load our experimental data from datasets on JuliaHub                               |
| [OptimizationOptimisers](https://docs.sciml.ai/Optimization/stable/optimization_packages/optimisers/)      | The optimization solver package with `Adam`                                                |
| [OptimizationOptimJL](https://docs.sciml.ai/Optimization/stable/optimization_packages/optim/)              | The optimization solver package with `LBFGS`                                               |
| [LineSearches](https://julianlsolvers.github.io/LineSearches.jl/latest/)                                   | Line search routines for non linear optimizers                                             |
| [DataDrivenDiffEq](https://docs.sciml.ai/DataDrivenDiffEq/stable/)                                         | The symbolic regression interface                                                          |
| [DataDrivenSparse](https://docs.sciml.ai/DataDrivenDiffEq/stable/libs/datadrivensparse/sparse_regression/) | The sparse regression symbolic regression solvers                                          |
| [Plots](https://docs.juliaplots.org/stable/)                                                               | The plotting and visualization library                                                     |
| [StableRNGs](https://docs.juliaplots.org/stable/)                                                          | Stable random seeding                                                                      |

Besides that we'll also need the following Julia standard libraries:

| Module        | Description                      |
|:------------- |:-------------------------------- |
| LinearAlgebra | Required for the `norm` function |
| Statistics    | Required for the `mean` function |

```@example autocomplete
using DyadModelOptimizer
using ModelingToolkit
using OrdinaryDiffEq
using CSV, DataFrames
using DataSets
using OptimizationOptimisers: Adam
using OptimizationOptimJL: LBFGS
using LineSearches: BackTracking
using DataDrivenDiffEq
using DataDrivenSparse
using Plots
gr(fmt=:png) # hide
using StableRNGs
using LinearAlgebra
using Statistics
using Test # hide
using SymbolicIndexingInterface # hide
```

## Problem setup

We will now define the incomplete model using ModelingToolkit and read in the training data (that corresponds to the correct model data + some small normally distributed noise).

```@setup autocomplete
function lotka()
    iv = only(@variables(t))
    sts = @variables x(t)=3.1 y(t)=1.5
    ps = @parameters α=1.3 β=0.9 γ=0.8 δ=1.8
    ∂ = Differential(t)

    eqs = [
        ∂(x) ~ α*x - β*x*y,
        ∂(y) ~ -δ*y + γ*x*y
    ]

    @named lotka = ODESystem(eqs, iv, sts, ps)
end

function generate_noisy_data(model, tspan = (0.0, 1.0), n = 5;
        params = [],
        u0 = [],
        rng = StableRNG(1111),
        kwargs...)
    prob = ODEProblem(model, u0, tspan, params)
    prob = remake(prob, u0 = 5.0f0 * rand(rng, length(prob.u0)))
    saveat = range(prob.tspan..., length = n)
    sol = solve(prob; saveat, kwargs...)
    X = Array(sol)
    x̄ = mean(X, dims = 2)
    noise_magnitude = 5e-3
    Xₙ = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))
    return DataFrame(hcat(sol.t, transpose(Xₙ)),
        vcat(:timestamp, getname.(variable_symbols(sol)))), sol
end
```

```@example autocomplete
function incomplete_lotka()
    iv = only(@variables(t))
    sts = @variables x(t)=5.0 y(t)=5.0
    ps = @parameters α=1.3 β=0.9 γ=0.8 δ=1.8
    ∂ = Differential(t)

    eqs = [
        ∂(x) ~ α * x,
        ∂(y) ~ - δ * y,
    ]

    return ODESystem(eqs, iv, sts, ps; name = :lotka)
end

rng = StableRNG(1111)
data, true_sol = generate_noisy_data(lotka(), (0., 5), 21; alg = Vern7(), abstol = 1e-12, reltol = 1e-12, rng) # hide
training_dataset = dataset("juliasimtutorials/lotka_data")
data = open(IO, training_dataset) do io
    CSV.read(io, DataFrame)
end

scatter(data.timestamp, [data."x(t)" data."y(t)"], color = :red, label = ["Noisy Data" nothing])
plot!(true_sol, alpha = 0.75, color = :black, label = ["True Data" nothing]) # hide
```

## Training the neural network

We will now train the neural network by calibrating its parameters to the training data.

```@example autocomplete
# We use the incomplete model in the experiment definition
incomplete_model = complete(incomplete_lotka())
@unpack x, y = incomplete_model
experiment = Experiment(data, incomplete_model, alg = Vern9(), abstol = 1e-8, reltol = 1e-8, u0 = [x=>data."x(t)"[1], y=>data."y(t)"[1]])
```

In order to add a neural network to the output of the model, we specify the neural network to be used via the `neural_network` keyword argument.

The `neural_network` keyword argument accepts a [Lux.jl](https://lux.csail.mit.edu/) model. For ease of use, the DyadModelOptimizer provides functions that can build such models easily, such as [`multi_layer_feed_forward`](@ref). As our model has 2 states, the input and output of the neural network will be 2.

```@example autocomplete
prob = InverseProblem(experiment, []; neural_network = multi_layer_feed_forward(2, 2), nn_rng = rng)
```

We now train the neural network by calibrating the inverse problem in two stages:

```@example autocomplete
r1 = calibrate(prob, SingleShooting(; maxiters = 5000, optimizer = Adam()))
```

We now use this optimization result as an initial guess for a second optimization using LBFGS and use BackTracking linesearch.

```@example autocomplete
r2 = calibrate(r1, SingleShooting(; maxiters = 1000, optimizer = LBFGS(linesearch = BackTracking())))
```

## Visualizing the training results

Let's visualize how well was the neural network trained:

```@example autocomplete
pl_losses = convergenceplot([r1, r2], yscale = :log10)
```

Next, we compare the original data to the output of the UDE predictor.

```@example autocomplete
pl_trajectory = plot(experiment, prob, r2, show_data = true, legend = :bottomleft)
```

Let's see how well the unknown term has been approximated:

```@example autocomplete
X̂ = simulate(experiment, prob, r2)
β = ModelingToolkit.defaults(incomplete_model)[incomplete_model.β]
γ = ModelingToolkit.defaults(incomplete_model)[incomplete_model.γ]
# Ideal unknown interactions of the predictor
Ȳ = [-β * (X̂[1, :] .* X̂[2, :])'; γ * (X̂[1, :] .* X̂[2, :])']
# Neural network guess
Ŷ = network_prediction(prob)(X̂, r2)

ts = X̂.t

pl_reconstruction = plot(ts, transpose(Ŷ), xlabel = "t", ylabel = "U(x,y)", color = :red, label = ["UDE Approximation" nothing])
plot!(ts, transpose(Ȳ), color = :black, label = ["True Interaction" nothing])
```
and we can also take a look at the reconstruction error
```@example autocomplete
# Plot the error
pl_reconstruction_error = plot(ts, norm.(eachcol(Ȳ - Ŷ)), yaxis = :log, xlabel = "t", ylabel = "L2-Error", label = nothing, color = :red)
pl_missing = plot(pl_reconstruction, pl_reconstruction_error, layout = (2, 1))

pl_overall = plot(pl_trajectory, pl_missing)
```

## Symbolic interpretation

We can now use the training results to autocomplete our model with symbolic terms that match
the data. This can be done with the `autocomplete` function, which is compatible with
DataDrivenDiffEq algorithms.
For this example we will use the `ADMM` algorithm from DataDrivenSparse and we will assume that our missing terms come from a polynomial basis of degree 4.
In order to improve the accuracy of the symbolic interpretation, we process the neural network predictions in batches and we normalize the results using the Z-score transformation.

```@example autocomplete
alg = ADMM(1e-1)
data_processing = DataProcessing(split = 0.9, batchsize = 30, shuffle = true)
res = autocomplete(r2, alg;
    basis_generator = (polynomial_basis, 4),
    digits = 1,
    normalize = DataNormalization(ZScoreTransform),
    data_processing)
```

The autocomplete function returns a calibration result as the new model parameters are calibrated to best match the data. We can extract the autocompleted model using `get_model`.

```@example autocomplete
get_model(res)
```

In order to evaluate how good is our new model, we can try making predictions outside of the training data by solving for a larger timespan.

```@example autocomplete
ex = only(get_experiments(res))
plot(ex, res, tspan = (0,50.), saveat = 0.1, show_data = true, legend = :topright)
```

[^1]: Derived from https://docs.sciml.ai/Overview/stable/showcase/missing_physics/ from https://github.com/SciML/SciMLDocs, MIT licensed, see repository for details
