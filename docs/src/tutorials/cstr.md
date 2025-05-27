# [Calibration of a Dynamical System using Collocation Methods](@id cstr)

Collocation Methods are techniques where the loss function minimizes the derivatives instead of the states or observed measurements in the data. This technique can be quite useful when the model is known to be unstable or takes a lot of time to simulate. This is because this suite of techniques does not involve simulating the model at all as derivatives are estimated from the data and derivatives for those time points are obtained from the model using the right hand side function. This also has a limitation - if we have partial observability, i.e., if we have data to only a few states or observed variables, there is no guarantee whether we can obtain the derivatives from the model. In this case, estimation of rest of the states from the data is internally handled and if it is not possible, it errors out gracefully.

In this tutorial, we will show how to use collocation methods for calibration of a dynamical system namely [Continuous Stirred Tank Reactor](https://en.wikipedia.org/wiki/Continuous_stirred-tank_reactor). This dynamical system is a common model representing a chemical reactor.

## Julia Environment

For this tutorial we will need the following packages:

| Module                                                                                              | Description                                                                                |
|:--------------------------------------------------------------------------------------------------- |:------------------------------------------------------------------------------------------ |
| [DyadModelOptimizer](https://help.juliahub.com/jsmo/stable/)                                    | The high-level library used to formulate our problem and perform automated model discovery |
| [ModelingToolkit](https://docs.sciml.ai/ModelingToolkit/stable/)                                    | The symbolic modeling environment                                                          |
| [ModelingToolkitStandardLibrary](https://docs.sciml.ai/ModelingToolkitStandardLibrary/stable/)      | Library for using standard modeling components                                             |
| [OrdinaryDiffEq](https://docs.sciml.ai/DiffEqDocs/stable/)                                          | The numerical differential equation solvers                                                |
| [CSV](https://csv.juliadata.org/stable/) and [DataFrames](https://dataframes.juliadata.org/stable/) | We will read our experimental data from .csv files                                         |
| [DataSets](https://help.juliahub.com/juliahub/stable/tutorials/datasets_intro/)                     | We will load our experimental data from datasets on JuliaHub                               |
| [DataInterpolations](https://docs.sciml.ai/DataInterpolations/stable/)                              | Library for creating interpolations from the data                                          |
| [Plots](https://docs.juliaplots.org/stable/)                                                        | The plotting and visualization library                                                     |

```@example cstr
using DyadModelOptimizer
using ModelingToolkit
import ModelingToolkit: D_nounits as D, t_nounits as t
using ModelingToolkitStandardLibrary.Blocks: RealInput, TimeVaryingFunction
using OrdinaryDiffEq
using CSV, DataFrames
using DyadData
using DataInterpolations: AkimaInterpolation, CubicSpline
using Plots
gr(fmt=:png) # hide
using Test # hide
```

## Data Setup

We recorded some simulation data from a MPC Controller for this model using JuliaSimControl as illustrated [here](https://help.juliahub.com/juliasimcontrol/stable/examples/cstr_mpc/). Let us use this data for calibration.

```@example cstr
cstr_dataset = DyadDataset("juliasimtutorials", "cstr_mpc_data",
    independent_var="timestamp",
    dependent_vars=["model.Cₐ", "model.Cᵦ", "model.Tᵣ", "model.Tₖ", "model.F", "model.Q̇"])

data = build_dataframe(cstr_dataset)

first(data, 5)
```

## Model Setup

```@example cstr
f = CubicSpline(data[!, "model.F"], data[!, "timestamp"])
q = CubicSpline(data[!, "model.Q̇"], data[!, "timestamp"])

@variables Cₐ(t)=0.8 Cᵦ(t)=0.5 Tᵣ(t)=134.14 Tₖ(t)=130.0
@named finput = RealInput()
@named qinput = RealInput()
F = finput.u
Q̇ = qinput.u

ps = @parameters K0_ab=1.287e12 K0_bc=1.287e12 K0_ad=9.043e9 R_gas=8.31446e-3 E_A_ab=9758.3 E_A_bc=9758.3 E_A_ad=8560.0 Hᵣ_ab=4.2 Hᵣ_bc=-11.0 Hᵣ_ad=-41.85 Rou=0.9342 Cp=3.01 Cpₖ=2.0 Aᵣ=0.215 Vᵣ=1.0 m_k=2.5 T_in=130.0 K_w=4032.0 C_A0=(5.7+4.5)/2.0*1.0

eqs = [
    D(Cₐ) ~ F*(C_A0 - Cₐ)-(K0_ab * exp((-E_A_ab)/((Tᵣ+273.15))))*Cₐ - (K0_ad * exp((-E_A_ad)/((Tᵣ+273.15))))*abs2(Cₐ)
    D(Cᵦ) ~ -F*Cᵦ + (K0_ab * exp((-E_A_ab)/((Tᵣ+273.15))))*Cₐ - (K0_bc * exp((-E_A_bc)/((Tᵣ+273.15))))*Cᵦ
    D(Tᵣ) ~ (((K0_ab * exp((-E_A_ab)/((Tᵣ+273.15))))*Cₐ*Hᵣ_ab + (K0_bc * exp((-E_A_bc)/((Tᵣ+273.15))))*Cᵦ*Hᵣ_bc +
                (K0_ad * exp((-E_A_ad)/((Tᵣ+273.15))))*abs2(Cₐ)*Hᵣ_ad)/(-Rou*Cp)) + F*(T_in-Tᵣ) + (((K_w*Aᵣ)*(-(Tᵣ-Tₖ)))/(Rou*Cp*Vᵣ))
    D(Tₖ) ~ (Q̇ + K_w*Aᵣ*(Tᵣ-Tₖ))/(m_k*Cpₖ)
]

@named model = ODESystem(eqs, t, [Cₐ, Cᵦ, Tᵣ, Tₖ], ps, systems=[finput, qinput])

fsrc = TimeVaryingFunction(f; name = :F)
qsrc = TimeVaryingFunction(q; name = :Q̇)

eqs = [
    ModelingToolkit.connect(model.finput, fsrc.output),
    ModelingToolkit.connect(model.qinput, qsrc.output),
]

@named cstr = ODESystem(eqs, t; systems = [model, fsrc, qsrc])
sys = structural_simplify(complete(cstr))
nothing # hide
```

## Defining Experiment and InverseProblem

In order to create an [`Experiment`](@ref), we will use the default initial values of the states and parameters of our model. These are our initial guesses which will be used to optimize the inverse problem in order to fit the given data.

```@example cstr
experiment = Experiment(data, sys, loss_func = meansquaredl2loss)
```

Once we have created the experiment, the next step is to create an [`InverseProblem`](@ref). This inverse problem, requires us to provide the search space as a vector of pairs corresponding to the parameters that we want to recover and the assumption that we have for their respective bounds. Here we are trying to calibrate the values of `m_k` which is the mass of the coolant and `Vᵣ` which is the volume of the reactor.

```@example cstr
prob = InverseProblem(experiment, [sys.model.m_k => (0.01, 10.0), sys.model.Vᵣ => (0.01, 20.0)])
```

## Calibration

### SingleShooting

Let us first try to solve this problem using [`SingleShooting`](@ref). To do this, we first define an algorithm `alg` and then call [`calibrate`](@ref) with the `prob` and `alg`.

```@example cstr
alg = SingleShooting(maxiters = 1000)
r = calibrate(prob, alg)
```

The true values of the parameters are - `m_k` has a value of 5.0 and `Vᵣ` has a value of 10.0. We can see that the calibrated parameters do not match the true values.

### Collocation

Now, let us try to do calibration process using a collocation method. For this tutorial, we will use [`SplineCollocation`](@ref) with [Akima Interpolation](https://en.wikipedia.org/wiki/Akima_spline). To do this, we have to define the algorithm very similar to how we did with [`SingleShooting`](@ref).

```@example cstr
alg = SplineCollocation(maxtime = 100, interp = AkimaInterpolation, cutoff = (0.05, 0.0))
```

Before we do calibration, we can visualize whether the estimated states match the data and estimated derivatives match from the model. This is an important step to check before calibration to visually verify whether the fit is accurate or not. To do this we use [`CollocationData`](@ref), and call it with the algorithm, experiment and inverse problem we defined and also pass in the true value of parameters.

!!! info

    For all problems, we won't have access to true values of parameters and we won't be able to pass in anything while calling [`CollocationData`](@ref) constructor. In those cases, we cannot see the derivatives from the model as it depends on the values of the parameters.

```@example cstr
collocated_data = CollocationData(alg, experiment, prob, [5.0, 10.0])
```

To plot it, we do:

For states,

```@example cstr
plot(collocated_data, vars = "states", layout = (4, 1), size = (1000, 1000))
```

For derivatives,

```@example cstr
plot(collocated_data, vars = "derivatives", layout = (4, 1), size = (1000, 1000))
```

We can see the derivatives match well except around the beginning. This is expected with collocation methods as they won't fit the data perfectly around the edges. To mitigate this, we use `cutoff` argument in the `alg` which tells us how much data to cut in the beginning and end for computing loss.

We can now proceed with calibration.

```@example cstr
r = calibrate(prob, alg)
```

We can see that the calibrated parameters are quite close to the true values!
