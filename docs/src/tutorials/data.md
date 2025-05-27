# Experiment Data Interface

The data ingested by DyadModelOptimizer needs a particular format. Let us first go through the rules and then demonstrate them with an example.

## Julia Environment

For this tutorial, we will use the following packages:

| Module                                                                                         | Description                                                  |
|:---------------------------------------------------------------------------------------------- |:------------------------------------------------------------ |
| [DyadModelOptimizer](https://help.juliahub.com/jsmo/stable/)                               | DyadModelOptimizer is used to formulate our problem      |
| [ModelingToolkit](https://docs.sciml.ai/ModelingToolkit/stable/)                               | The symbolic modeling environment                            |
| [ModelingToolkitStandardLibrary](https://docs.sciml.ai/ModelingToolkitStandardLibrary/stable/) | Library for using standard modeling components               |
| [OrdinaryDiffEq](https://docs.sciml.ai/DiffEqDocs/stable/)                                     | The numerical differential equation solvers                  |
| [DataSets](https://help.juliahub.com/juliahub/stable/tutorials/datasets_intro/)                | We will load our experimental data from datasets on JuliaHub |

```@example data_format
using DyadModelOptimizer
using ModelingToolkit
using ModelingToolkit: t_nounits as t
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks: Sine
using OrdinaryDiffEq
using DyadData
```

## Data Format

The rules are as follows:

 1. Data which can either be simulations or real world data are bunch of timeseries for all the states/observables arranged in a tabular format which has a [`Tables.jl`](https://tables.juliadata.org/stable/) interface such as `DataFrame`, `CSV.File`, `NamedTuple` containing vectors for each columns etc.
 2. First column of every table should be named as "timestamp". This column is the series of time points. When we have steady state problems, we can have `Inf` in the timestamp column to signify that it is steady state data.
 3. The names for the rest of the columns should match the names of the states or algebraic variables of the corresponding model. Note that:
    i. The independent variable suffix (usually "(t)"), which `ModelingToolkit` generates is not required, it can either be present or omitted.
    ii. For models with sub-systems, the variables should be namespaced according to the system they belong to. The character used to delimit the namespace can either be "." or "₊".

Now, let us go through an example which is the same as the example in the [getting started page](@ref getting_started_page).

```@example data_format
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

model = create_model()
sys = structural_simplify(model)
```

We can see the unknowns of the model by:

```@example data_format
unknowns(sys)
```

We can see that states of the model are defined in an interpretable manner. For example,"capacitor2₊v(t)" means the voltage across `capacitor2`. So, using the rules defined above, the name of this column in the dataset can be:

  - "capacitor2₊v(t)"
  - "capacitor2₊v"
  - "capacitor2.v(t)"
  - "capacitor2.v"

All the above names map to the same state in the model.

## Data Storage

The data can be saved in any format on disk as long we can deserialize it in the formats mentioned above.

For ease, DyadModelOptimizer also works with DyadData, which provides an interface for working with JuliaHub datasets as well as local files and raw data.

Let us demonstrate this with an example. We will use the dataset from [getting started page](@ref getting_started_page).

```@example data_format
data = DyadDataset("juliasimtutorials", "circuit_data", independent_var="timestamp", dependent_vars=["ampermeter.i(t)"])
experiment = Experiment(data, sys; overrides = [sys.capacitor2.v => 0.0], alg = Rodas4(), abstol = 1e-6, reltol = 1e-5)
```

We can see that we only need to pass in the `DataSet` object directly into the [`Experiment`](@ref) constructor to use the data.

## DyadData interface

```@docs
DyadData.DyadDataset
DyadData.build_dataframe
```
