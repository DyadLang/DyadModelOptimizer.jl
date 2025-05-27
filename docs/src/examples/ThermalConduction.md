# [Global Model Calibration of Thermal Conduction](@id thermal_conduction)

## Thermal conduction model

Here we present an example of integrating the [ModelingToolkitStandardLibrary](https://docs.sciml.ai/ModelingToolkitStandardLibrary/stable/) with DyadModelOptimizer by using the thermal conduction example. This example demonstrates the thermal response of two masses connected by a conducting element. The two masses have the same unknown heat capacity but different initial temperatures (`T1=100 [°C]`, `T2=0 [°C]`). The mass with the higher temperature will cool off while the mass with the lower temperature heats up. They will each asymptotically approach the calculated temperature `T_final` that results from dividing the total initial energy in the system by the sum of the heat capacities of each element. The goal of this example is to find the unknown heat capacitance and also quantify its uncertainty.

## Julia environment

For this example, we will need the following packages:

| Module                                                                                         | Description                                                                                |
|:---------------------------------------------------------------------------------------------- |:------------------------------------------------------------------------------------------ |
| [DyadModelOptimizer](https://help.juliahub.com/jsmo/stable/)                               | The high-level library used to formulate our problem and perform automated model discovery |
| [ModelingToolkit](https://docs.sciml.ai/ModelingToolkit/stable/)                               | The symbolic modeling environment                                                          |
| [ModelingToolkitStandardLibrary](https://docs.sciml.ai/ModelingToolkitStandardLibrary/stable/) | Library for using standard modeling components                                             |
| [OrdinaryDiffEq](https://docs.sciml.ai/DiffEqDocs/stable/)                                     | The numerical differential equation solvers                                                |
| [DataFrames](https://dataframes.juliadata.org/stable/)                                         | For converting simulation into a Dataframe                                                  |
| [Plots](https://docs.juliaplots.org/stable/)                                                   | The plotting and visualization library                                                     |

```@example thermal
using DyadModelOptimizer
using ModelingToolkit
using ModelingToolkit: t_nounits as t
using ModelingToolkitStandardLibrary.Thermal
using OrdinaryDiffEq
using DataFrames
using Plots
gr(fmt=:png) # hide
using Test # hide
```

## Model setup

```@example thermal
C = 15.0
@named mass1 = HeatCapacitor(C = C, T = 373.15)
@named mass2 = HeatCapacitor(C = C, T = 273.15)
@named conduction = ThermalConductor(G = 10)
@named Tsensor1 = TemperatureSensor()
@named Tsensor2 = TemperatureSensor()

connections = [
    connect(mass1.port, conduction.port_a),
    connect(conduction.port_b, mass2.port),
    connect(mass1.port, Tsensor1.port),
    connect(mass2.port, Tsensor2.port),
]

@named model = ODESystem(connections, t, systems = [mass1, mass2, conduction, Tsensor1, Tsensor2])
sys = structural_simplify(model)
```

## Data Setup

Now, lets simulate to generate some data.

```@example thermal
prob = ODEProblem(sys, [mass1.C => 14.0, mass2.C => 14.0], (0, 100.0))
sol = solve(prob, Tsit5())
data = DataFrame(sol)
first(data, 5)
```

## Defining Experiment and InverseProblem

We next use the data to setup an [`Experiment`](@ref).

```@example thermal
experiment = Experiment(data, sys)
```

Next, we specify an [`InverseProblem`](@ref) of interest to us where we specify the parameters we want to recover and their bounds.

```@example thermal
invprob = InverseProblem(experiment, [mass1.C => (10, 20), mass2.C => (10, 20)])
```

## Parametric Uncertainty Quantification

We now run the parameteric uncertainty quantification using the [`parametric_uq`](@ref) function. We specify the [`InverseProblem`](@ref) and the [`StochGlobalOpt`](@ref) algorithm. We also specify the sample size for the analysis.

```@example thermal
ps = parametric_uq(invprob, StochGlobalOpt(maxiters = 1000), sample_size = 50)
```

## Visualization

We can see the results of the analysis by plotting its histogram.

```@example thermal
plot(ps, layout = (2, 1), bins = 10, max_freq = [20, 20], legend = :best, size = (1000, 1000))
```

We can see the distribution for the parameters is concentrated near their true values.
