# Custom Loss Functions

This tutorial assumes that you have read the [getting started tutorial](@ref getting_started_page). By default DyadModelOptimizer uses the [`squaredl2loss`](@ref), but this may always not be the best choice. In this tutorial, we will work with the same model as the getting started tutorial but use a custom loss function for minimization.

## Julia Environment

For this tutorial, we will use the following packages:

| Module                                                                                         | Description                                                  |
|:---------------------------------------------------------------------------------------------- |:------------------------------------------------------------ |
| [DyadModelOptimizer](https://help.juliahub.com/jsmo/stable/)                               | This is used to formulate our inverse problem and solve it   |
| [ModelingToolkit](https://docs.sciml.ai/ModelingToolkit/stable/)                               | The symbolic modeling environment                            |
| [ModelingToolkitStandardLibrary](https://docs.sciml.ai/ModelingToolkitStandardLibrary/stable/) | Library for using standard modeling components               |
| [OrdinaryDiffEq](https://docs.sciml.ai/DiffEqDocs/stable/)                                     | The numerical differential equation solvers                  |
| [Statistics](https://docs.julialang.org/en/v1/stdlib/Statistics/)                              | Library for standard statistics functions                    |
| [DataSets](https://help.juliahub.com/juliahub/stable/tutorials/datasets_intro/)                | We will load our experimental data from datasets on JuliaHub |
| [Plots](https://docs.juliaplots.org/stable/)                                                   | The plotting and visualization library                       |

```@example loss_tutorial
using DyadModelOptimizer
using ModelingToolkit
using ModelingToolkit: t_nounits as t
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks: Sine
using OrdinaryDiffEq
using Statistics
using DyadData
using Plots
gr(fmt=:png) # hide
using Test #hide
```

## Model Setup

```@example loss_tutorial
function create_model(; C₁ = 3e-5, C₂ = 1e-6)
    @named resistor1 = Resistor(R = 5.0)
    @named resistor2 = Resistor(R = 2.0)
    @named capacitor1 = Capacitor(C = C₁)
    @named capacitor2 = Capacitor(C = C₂)
    @named source = Voltage()
    @named input_signal = Sine(frequency = 100.0)
    @named ground = Ground()
    @named ampermeter = CurrentSensor()

    eqs = [connect(input_signal.output, source.V)
        connect(source.p, capacitor1.n, capacitor2.n)
        connect(source.n, resistor1.p, resistor2.p, ground.g)
        connect(resistor1.n, capacitor1.p, ampermeter.n)
        connect(resistor2.n, capacitor2.p, ampermeter.p)]

    @named circuit_model = ODESystem(eqs, t,
        systems = [
            resistor1, resistor2, capacitor1, capacitor2,
            source, input_signal, ground, ampermeter,
        ])
end
C₂ = 1e-5 # hide
model = create_model()
sys = complete(structural_simplify(model))
```

## Data Setup

However, in this tutorial, the data is noisier. Furthermore, the strength of the noise seems to depend on the absolute value of the current $i$ going through the ampermeter. The data can be read in using the `dataset` function.

```@example loss_tutorial
data = DyadDataset("juliasimtutorials", "circuit_data_heterogeneous", independent_var="timestamp", dependent_vars=["ampermeter.i(t)"])
```

For [heteroscedastic](https://en.wikipedia.org/wiki/Homoscedasticity_and_heteroscedasticity) noise, a [weighted variant of l2loss](https://en.wikipedia.org/wiki/Weighted_least_squares) outperforms the default l2loss. Specifically, we assume that the standard deviation of the measurement noise increases linearly as `i` increases.

## Implement Custom Loss Function

We can implement such a weighted loss ourselves using the `loss_func` keyword of `experiment`. `loss_func` should always be a function with 3 inputs, the values of the tuned parameter or initial conditions values, the solution of the model, at the point in the search space given by the first argument and the data. Here the solution will always only contain the states and timepoints present in the dataset, e.g. state voltages will not be present in the solution. The dataset is transformed from a `DataFrame` to a `Matrix`. Both inputs are ordered in the same way, i.e. `solution[i,j]` corresponds to the same state and timepoint as `dataset[i,j]`. We can then easily implement losses using broadcasting.

```@example loss_tutorial
function weighted_loss(tuned_vals, solution, dataset)
    σ² = var(dataset)
    sum(((solution .- dataset) .^ 2) ./ σ²)
end
```

## Defining Experiment and InverseProblem

In order to create an [`Experiment`](@ref), we will use the default initial values of the states and parameters of our model. These are our initial guesses which will be used to optimize the inverse problem in order to fit the given data. We also pass in the custom loss function that we defined above.

```@example loss_tutorial
@unpack capacitor1, capacitor2 = model
experiment = Experiment(data, sys; overrides = [capacitor2.v => 0.0], alg = Rodas5P(), abstol = 1e-6, reltol = 1e-5, loss_func = weighted_loss)
```

Once we have created the experiment, the next step is to create an [`InverseProblem`](@ref). This inverse problem, requires us to provide the search space as a vector of pairs corresponding to the parameters that we want to recover and the assumption that we have for their respective bounds.

```@example loss_tutorial
prob = InverseProblem(experiment, [capacitor2.C => (1.e-7, 1e-3)])
```

## Calibration

Now, lets use [`SingleShooting`](@ref) for calibration. To do this, we first define an algorithm `alg` and then call [`calibrate`](@ref) with the `prob` and `alg`.

```@example loss_tutorial
alg = SingleShooting(maxiters = 10^3)
r = calibrate(prob, alg)
@test only(r)≈C₂ rtol=1e-1 # hide
r # hide
```

## Visualization

Let us visualize the calibrated results. We can plot the simulation using the calibrated parameters and compare it against the data.

```@example loss_tutorial
plot(experiment, prob, r, show_data = true, legend = true)
```

We see that this calibrated current fits the data well!
