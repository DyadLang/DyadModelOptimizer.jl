# Calibrate a Functional Mock-up Unit (FMU)

In this example, we will calibrate a Functional Mock-up Unit (FMU), following [Functional Mockup Interface](https://fmi-standard.org/) v2.0, to experiment data using the [`SingleShooting`](@ref) method.

The example is based on a Coupled-Clutches model that is available as an FMU. This [Coupled-Clutches FMU](https://github.com/modelica/fmi-cross-check/tree/master/fmus/2.0/cs/linux64/MapleSim/2021.2/CoupledClutches) is a model of a clutch system with three clutches.

## Julia environment

For this example, we will use the following packages:

| Module                                                                                               | Description                                                |
|:---------------------------------------------------------------------------------------------------- |:---------------------------------------------------------- |
| [DyadModelOptimizer](https://help.juliahub.com/jsmo/stable/)                                     | This is used to formulate our inverse problem and solve it |
| [FMI](https://thummeto.github.io/FMI.jl/dev/) and [FMIImport](https://thummeto.github.io/FMI.jl/dev/)| Library for importing and simulating FMUs                  |
| [CSV](https://csv.juliadata.org/stable/) and [DataFrames](https://dataframes.juliadata.org/stable/)  | We will read our experimental data from .csv files         |
| [Optimization](https://docs.sciml.ai/Optimization/stable/)                                           | High level interface package for using optimizers          |
| [OptimizationNLopt](https://docs.sciml.ai/Optimization/stable/optimization_packages/nlopt/)          | The optimization solver package with `GN_ISRES`            |
| [Plots](https://docs.juliaplots.org/stable/)                                                         | The plotting and visualization library                     |

```@example calibFMU
using DyadModelOptimizer
using FMI, FMIImport
using DataFrames, CSV
using Optimization, OptimizationNLopt
using Plots
gr(fmt=:png) # hide
using Test # hide
```

## Model Setup

We start off by loading the FMU using [FMI.jl](https://github.com/ThummeTo/FMI.jl) - an open-source julia package which enables working with FMUs. We specify the type of the FMU as `FMI.fmi2TypeModelExchange` since we intend to simualate the FMU in Model Exchange mode.

```@example calibFMU
fmu = fmiLoad(joinpath(@__DIR__, "../assets/CoupledClutches_ME.fmu");
    type = FMI.fmi2TypeModelExchange)
```

!!! note
    Note that since the FMU support needs both FMI and FMIImport, both packages need to be loaded.

## Data Setup

We load the data from a CSV file using the DataFrames.jl package. The data is a time series of the states of the system, such as the angular velocity and acceleration of the inertias. The true values of the parameters are `freqHz = 2.0` and `T2 = 1.0`.

```@example calibFMU
data = CSV.read(joinpath(@__DIR__, "../assets/CoupledClutches_data.csv"), DataFrame)
freqHz_test = 2.0 # hide
T2_test = 1.0 # hide
data[1:5, :] # hide
```

## Defining Experiment and InverseProblem

We next use the data to setup an experiment. We specify the time span of the [`Experiment`](@ref) and the relative tolerance of the solver.

```@example calibFMU
experiment = Experiment(data, fmu; tspan = (0.0, 1.5), reltol = 1e-8)
```

Next, We define an [`InverseProblem`](@ref) of interest to us where we specify the parameters we want to recover and their bounds.

```@example calibFMU
prob = InverseProblem(experiment, ["freqHz" => (0.05, 3.0), "T2" => (0.2, 1.5)])
```

## Calibration

We run the calibration using the [`calibrate`](@ref) function. We specify the [`SingleShooting`](@ref) algorithm to use for the calibration. We also specify the `adtype` to be `NoAD()` as we will use a gradient free optimizer for the calibration process. We use [`GN_ISRES`](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms#isres-improved-stochastic-ranking-evolution-strategy) as the optimizer.

```@example calibFMU
opt = OptimizationNLopt.NLopt.GN_ISRES()
alg = SingleShooting(maxiters = 30_000, optimizer = opt)
r = calibrate(prob, alg, adtype = Optimization.SciMLBase.NoAD())
@test r[1] ≈ freqHz_test # hide
@test r[2] ≈ T2_test # hide
r
```

We can see the values of the calibration matches the true values of the parameters, i.e., `freqHz = 2.0` and `T2 = 1.0` from the data.

## Visualization

We can now plot the simulation using the calibrated parameters and compare it against the data.

```@example calibFMU
plot(experiment, prob, r, show_data = true, legend = :best, ms = 0.2, layout = (8, 1), size = (1000, 1600), left_margin = 5Plots.mm)
```
