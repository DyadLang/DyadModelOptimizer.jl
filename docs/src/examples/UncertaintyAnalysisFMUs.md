# Parametric Uncertainty Quantification with Functional Mockup Units (FMUs)

This example demonstrates how use Functional Mockup Units (FMUs), following [Functional Mockup Interface](https://fmi-standard.org/) v2.0, to perform parametric uncertainty quantification with DyadModelOptimizer.

The example is based on a Coupled-Clutches model that is available as an FMU. This [Coupled-Clutches FMU](https://github.com/modelica/fmi-cross-check/tree/master/fmus/2.0/cs/linux64/MapleSim/2021.2/CoupledClutches) is a model of a clutch system with three clutches.

## Julia Environment

For this example, we will use the following packages:

| Module                                                                                               | Description                                                |
|:---------------------------------------------------------------------------------------------------- |:---------------------------------------------------------- |
| [DyadModelOptimizer](https://help.juliahub.com/jsmo/stable/)                                     | This is used to formulate our inverse problem and solve it |
| [FMI](https://thummeto.github.io/FMI.jl/dev/) and [FMIImport](https://thummeto.github.io/FMI.jl/dev/)| Library for importing and simulating FMUs                  |
| [CSV](https://csv.juliadata.org/stable/) and [DataFrames](https://dataframes.juliadata.org/stable/)  | We will read our experimental data from .csv files         |
| [Plots](https://docs.juliaplots.org/stable/)                                                         | The plotting and visualization library                     |

```@example FMU
using DyadModelOptimizer
using FMI, FMIImport
using CSV, DataFrames
using Plots
gr(fmt=:png) # hide
using Test #hide
```

## Model Exchange

### Model Setup

Let us first demonstrate the use of the Coupled-Clutches FMU in Model Exchange mode.

We start of by loading the FMU using [FMI.jl](https://github.com/ThummeTo/FMI.jl) - an open-source julia package which enables working with FMUs. We specify the type of the FMU as `FMI.fmi2TypeModelExchange` since we intend to simualate the FMU in Model Exchange mode.

```@example FMU
fmu = fmiLoad(joinpath(@__DIR__, "../assets/CoupledClutches_ME.fmu"); type = FMI.fmi2TypeModelExchange)
```

!!! note
    Note that since the FMU support needs both FMI and FMIImport, both packages need to be loaded.

### Data Setup

We load the data from a CSV file using the DataFrames.jl package. The data is a time series of the states of the system, such as the angular velocity and acceleration of the inertias.

```@example FMU
data = CSV.read(joinpath(@__DIR__, "../assets/CoupledClutches_data.csv"), DataFrame)
data[1:5, :] # hide
```

### Defining Experiment and InverseProblem

We next use the data to setup an [`Experiment`](@ref). We specify the time span in the [`Experiment`](@ref) and the relative tolerance of the solver.

```@example FMU
experiment = Experiment(data, fmu; tspan = (0.0, 1.5), reltol = 1e-5)
```

Next, we specify an [`InverseProblem`](@ref) of interest to us where we specify the parameters we want to recover and their bounds.

```@example FMU
invprob = InverseProblem(experiment, ["freqHz" => (0.05, 3), "T2" => (0.2, 1.5)])
```

### Parametric Uncertainty Quantification

We now run the parameteric uncertainty quantification using the [`parametric_uq`](@ref) function. We specify the [`InverseProblem`](@ref) and the [`StochGlobalOpt`](@ref) algorithm. We also specify the sample size for the analysis.

```@example FMU
results = parametric_uq(invprob, StochGlobalOpt(; maxiters = 3000), sample_size = 20)
```

!!! note

    Note that we need a sample size of at least 5000 to get a good result and this low sample size is just for demonstration. Same with the number of `maxiters` parameter of `StochGlobalOpt`.

### Visualization

We can see the results of the analysis by plotting its histogram.

```@example FMU
plot(results, layout = (2, 1), bins = 8, show_comparison_value = false, legend = :best, size = (1000, 1000))
```

We can see the distribution for both the parameters is concentrated near their true values.

## Co-Simulation Mode

### Model Setup

Let us now import the FMU in Co-Simulation mode.

```@example FMU
fmu = fmiLoad(joinpath(@__DIR__, "../assets/CoupledClutches_CS.fmu"); type = FMI.fmi2TypeCoSimulation)
```

### Defining Experiment and InverseProblem

The steps for using FMUs in Co-Simulation mode are similar to those for Model Exchange mode. The only difference is that we specify the type of the FMU as `FMI.fmi2TypeCoSimulation` and we use the [`DiscreteExperiment`](@ref) instead of the [`Experiment`](@ref) to setup the experiment as Co-Simulation FMUs are discrete time models and specify the step size (here `1e-2`) compulsorily.

```@example FMU
de = DiscreteExperiment(data, fmu, 1e-2; tspan = (0.0, 1.5))
invprob = InverseProblem(de, ["freqHz" => (0.05, 3), "T2" => (0.2, 1.5)])
```

### Parametric Uncertainty Quantification

The rest of the code remains the same as for the Model Exchange FMU.

```@example FMU
results = parametric_uq(invprob, StochGlobalOpt(; maxiters = 3000), sample_size = 20)
```

### Visualization

We can see the results of the analysis by plotting its histogram.

```@example FMU
plot(results, layout = (2, 1), bins = 8, show_comparison_value = false, legend = :best, size = (1000, 1000))
```

Again, we can see the distribution for both the parameters is concentrated near their true values.
