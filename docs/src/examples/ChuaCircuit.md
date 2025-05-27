# [Calibration of an Unstable Circuit Model using Prediction Error Method](@id chua_circuit)

In this example, we present the creation of a custom component is demonstrated via the [Chua's circuit](https://en.wikipedia.org/wiki/Chua%27s_circuit). The circuit is a simple circuit that shows chaotic behaviour. Except for a non-linear resistor every other component already is part of `ModelingToolkitStandardLibrary.Electrical`. We can then seamlessly plug this model with DyadModelOptimizer for calibration.

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

```@example chuacircuit
using DyadModelOptimizer
using ModelingToolkit
import ModelingToolkit: D_nounits as D, t_nounits as t
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Electrical: OnePort
using OrdinaryDiffEq
using DataFrames
using Plots
gr(fmt=:png) # hide
using Test # hide
```

## Model Setup

The first step is to use the pre defined components defined in the Electrical Toolkit in the ModelingToolkit library. We can also define custom components such as the "Non linear resistor" as defined below. One advantage of using ModelingToolkit is being directly able to use custom components out of the box. We can define a Resistor component, a Capacitor component, an Inductor component etc. ModelingToolkit defined models can be seamlessly integrated with the solvers from DifferentialEquations.jl.

```@example chuacircuit
function NonlinearResistor(; name, Ga, Gb, Ve)
    @named oneport = OnePort()
    @unpack v, i = oneport
    pars = @parameters Ga=Ga Gb=Gb Ve=Ve
    eqs = [
        i ~ ifelse(v < -Ve,
            Gb * (v + Ve) - Ga * Ve,
            ifelse(v > Ve,
                Gb * (v - Ve) + Ga * Ve,
                Ga * v)),
    ]
    extend(ODESystem(eqs, t, [], pars; name = name), oneport)
end

@named L = Inductor(L = 18)
@named Ro = Resistor(R = 12.5e-3)
@named G = Conductor(G = 0.565)
@named C1 = Capacitor(C = 10, v = 4)
@named C2 = Capacitor(C = 100)
@named Nr = NonlinearResistor(Ga = -0.757576, Gb = -0.409091, Ve = 1)
@named Gnd = Ground()

connections = [connect(L.p, G.p)
    connect(G.n, Nr.p)
    connect(Nr.n, Gnd.g)
    connect(C1.p, G.n)
    connect(L.n, Ro.p)
    connect(G.p, C2.p)
    connect(C1.n, Gnd.g)
    connect(C2.n, Gnd.g)
    connect(Ro.n, Gnd.g)]

@named model = ODESystem(connections, t, systems = [L, Ro, G, C1, C2, Nr, Gnd])
sys = structural_simplify(model)
```

## Data Setup

Let us simulate using a stiff ODE solver `Rodas4` and use this data for calibration.

```@example chuacircuit
@unpack L, C2 = model
prob = ODEProblem(sys, [L.i => 0.0, C2.v => 0.0], (0, 5e4), [Ro.R => 11e-3, C1.C => 9.3, C2.C => 102.5], saveat = 10)
sol = solve(prob, Rodas4())
data = DataFrame(sol)
first(data, 5)
```

## Defining Experiment and InverseProblem

This system is unstable and it can be difficult to simulate it for different sets of parameters. To mitigate this, we will use Prediction Error Method, where the simulation is guided by the data such that the trajectory won't diverge and this should help with the calibration process.

In order to create an [`Experiment`](@ref), we will use the default initial values of the states and parameters of our model. These are our initial guesses which will be used to optimize the inverse problem in order to fit the given data. To use Prediction Error Method, we also need to pass it in the `model_transformations` keyword in the constructor.

```@example chuacircuit
experiment = Experiment(data, sys, overrides = [L.i => 0.0, C2.v => 0.0], model_transformations = [DiscreteFixedGainPEM(0.2)], alg = Rodas4())
```

Argument passed to `DiscreteFixedGainPEM` is the amount of correction needed during simulation. `1.0` represents completely using the data and `0.0` represents completely ignoring the data. Typically, we should use this be about 0.2-0.3 to help guide the simulation.

The next step is to define an [`InverseProblem`](@ref) by specifying the parameters we want to optimize and the search space of those parameters.

```@example chuacircuit
invprob = InverseProblem(experiment, [Ro.R => (9.5e-3, 13.5e-3), C1.C => (9, 11), C2.C => (95, 105)])
```

## Calibration

We will use [`SingleShooting`](@ref) as our calibration algorithm. To calibrate, we simply call [`calibrate`](@ref) with our inverse problem and calibration algorithm.

```@example chuacircuit
alg = SingleShooting(maxiters = 100)
r = calibrate(invprob, alg)
```

As we see above, the parameters recovered match the true parameters!

## Visualization

We can now plot the simulation using the calibrated parameters and compare it against the data.

```@example chuacircuit
plot(experiment, invprob, r, show_data = true, legend = :best, ms = 0.2, layout = (3, 1), size = (1000, 900))
```
