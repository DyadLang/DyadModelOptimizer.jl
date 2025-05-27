# Calibration of a Dynamical System with Observed Data with Prediction Error Method

In this tutorial, we will show how to calibrate length of a pendulum using prediction error method.

Prediction error methods (PEM) are good at smoothing loss landscapes which eases the calibration process. Without PEM, this problem has tons of local minima in the loss landscape which makes the calibration process difficult to converge to the correct value.

## Julia Environment

For this tutorial we will need the following packages:

| Module                                                                                              | Description                                                                                |
|:--------------------------------------------------------------------------------------------------- |:------------------------------------------------------------------------------------------ |
| [DyadModelOptimizer](https://help.juliahub.com/jsmo/stable/)                                       | The high-level library used to formulate our problem and perform automated model discovery |
| [ModelingToolkit](https://docs.sciml.ai/ModelingToolkit/stable/)                                    | The symbolic modeling environment                                                          |                                     |
| [CSV](https://csv.juliadata.org/stable/) and [DataFrames](https://dataframes.juliadata.org/stable/) | We will read our experimental data from .csv files                                         |
| [DataSets](https://help.juliahub.com/juliahub/stable/tutorials/datasets_intro/)                     | We will load our experimental data from datasets on JuliaHub                               |
| [Plots](https://docs.juliaplots.org/stable/)                                                        | The plotting and visualization library                                                     |

```@example pendulum_pem
using DyadModelOptimizer
using ModelingToolkit
import ModelingToolkit: D_nounits as D, t_nounits as t
using CSV, DataFrames
using DyadData
using Plots
gr(fmt=:png) # hide
using Test # hide
```

## Data Setup

Let us import the dataset. This dataset contains data for the observed variables of the system.

```@example pendulum_pem
pendulum_dataset = DyadDataset("juliasimtutorials", "pendulum_data", independent_var="timestamp", dependent_vars=["m"])

data = build_dataframe(pendulum_dataset)

first(data, 5)
```

## Model Setup

We will now define the model using ModelingToolkit. The model is a simple pendulum fixed at one end and mass on the other. We will also assume mass of the string is negiglible. `x₁` denotes the displacement and `x₂` denotes the velocity. We are measuring `3*x₁` which is the data collected.

```@example pendulum_pem
@variables x₁(t)=0.0 x₂(t)=3.0 m(t)
@parameters L = 2.0
@constants g = 9.81
tspan = (0.0, 20.0)
eqs = [
    D(x₁) ~ x₂,
    D(x₂) ~ -(g / L) * sin(x₁),
    m ~ 3 * x₁
]
@named model = ODESystem(eqs, t, [x₁, x₂], [L]; tspan)
sys = structural_simplify(model)
```

## Defining Experiment and InverseProblem

In order to create an [`Experiment`](@ref), we will use the default initial values of the states and parameters of our model. These are our initial guesses which will be used to optimize the inverse problem in order to fit the given data. To use prediction error method, we pass [`DiscreteFixedGainPEM`](@ref) in the `model_transformations` keyword argument. [`DiscreteFixedGainPEM`](@ref) automatically tries to use all the data present for guiding the simulation - both unknowns and observed. If observed data is present, it tries to solve the system of equations to estimate unknowns, provided the observed equations corresponding to the data are linear and invertible.

```@example pendulum_pem
experiment_no_pem = Experiment(data, sys; abstol = 1e-8, reltol = 1e-6)
experiment_pem = Experiment(data, sys; abstol = 1e-6, reltol = 1e-6, model_transformations = [DiscreteFixedGainPEM(0.1)])
```

Once we have created the experiment, the next step is to create an [`InverseProblem`](@ref). This inverse problem, requires us to provide the search space as a vector of pairs corresponding to the parameters that we want to recover and the assumption that we have for their respective bounds.

```@example pendulum_pem
prob_no_pem = InverseProblem(experiment_no_pem, [L => (0.0, 3.0)])
prob_pem = InverseProblem(experiment_pem, [L => (0.0, 3.0)])
```

## Calibration

The true length of the pendulum is 0.2.

### SingleShooting

Let us first try to solve this problem using [`SingleShooting`](@ref) without PEM. To do this, we first define an algorithm `alg` and then call [`calibrate`](@ref) with the `prob_no_pem` and `alg`.

```@example pendulum_pem
alg = SingleShooting(maxiters = 1000, maxtime=300)
r_no_pem = calibrate(prob_no_pem, alg)
@test !(r_no_pem.u[1] ≈ 0.2) # hide
r_no_pem # hide
```

We can see the calibrated length of the pendulum is far from the correct value.

### SingleShooting with Prediction Error Method

Now, let us try it with PEM.

```@example pendulum_pem
alg = SingleShooting(maxiters = 1000, maxtime=300)
r_pem = calibrate(prob_pem, alg)
@test r_pem.u[1] ≈ 0.2 rtol=1e-4 # hide
r_pem # hide
```

We can see it calibrates correctly!

## Visualization

Let us plot the result to confirm visually.

```@example pendulum_pem
plot(experiment_no_pem, prob_no_pem, r_no_pem, show_data = true, ms = 1.0, size = (1000, 300))
```

```@example pendulum_pem
plot(experiment_pem, prob_pem, r_pem, show_data = true, ms = 1.0, size = (1000, 300))
```

We can see the simulation results without PEM do not match the data at all but with PEM matches the data really well.
