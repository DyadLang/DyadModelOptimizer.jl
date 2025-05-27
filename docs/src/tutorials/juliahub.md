# Launching Batch Jobs in JuliaHub

DyadModelOptimizer provides an interface to launch batch jobs on JuliaHub using [JuliaHub.jl](https://help.juliahub.com/julia-api/stable/). This is particularly useful for scaling compute to multiple threads/processes like for [`parametric_uq`](@ref), [`MultipleShooting`](@ref) where it can be massively parallelized. In this tutorial, let us walk through in the process of launching these batch jobs. We will use the same example as in Getting Started page, i.e., [De-Sauty bridge](@ref getting_started).

## Julia Environment

For this tutorial we will need the following packages:

| Module                                                                                         | Description                                                                                |
|:---------------------------------------------------------------------------------------------- |:------------------------------------------------------------------------------------------ |
| [DyadModelOptimizer](https://help.juliahub.com/jsmo/stable/)                               | The high-level library used to formulate our problem and perform automated model discovery |
| [ModelingToolkit](https://docs.sciml.ai/ModelingToolkit/stable/)                               | The symbolic modeling environment                                                          |
| [ModelingToolkitStandardLibrary](https://docs.sciml.ai/ModelingToolkitStandardLibrary/stable/) | Library for using standard modeling components                                             |
| [OrdinaryDiffEq](https://docs.sciml.ai/DiffEqDocs/stable/)                                     | The numerical differential equation solvers                                                |
| [DataSets](https://help.juliahub.com/juliahub/stable/tutorials/datasets_intro/)                 | We will load our experimental data from datasets on JuliaHub                               |
| [Plots](https://docs.juliaplots.org/stable/)                                                   | The plotting and visualization library                                                      |
| [JuliaHub](https://help.juliahub.com/julia-api/stable/)                                         | Package for a programmatic Julia interface to the JuliaHub platform                        |
| [Serialization](https://docs.julialang.org/en/v1/stdlib/Serialization/)                         | Julia Standard library for serializing and deserializing files                              |

```@example juliahub
using DyadModelOptimizer
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks
using OrdinaryDiffEq
using DyadData
using Plots
using JuliaHub
using Serialization
gr(fmt = :png) # hide
using Test # hide
```

Before we launch any job, we need to authenticate to JuliaHub which can be done using `JuliaHub.authenticate`. We will use this `auth` object for submitting jobs.

```@example juliahub
auth = JuliaHub.authenticate()
nothing # hide
```

## Data Setup

We can read tabular experimental data where the model names are matching column names in the table.

```@example juliahub
data = DyadDataset("juliasimtutorials", "circuit_data", independent_var="timestamp", dependent_vars=["ampermeter.i(t)"])
```

## Model Setup

We will now define the De Sauty circuit model.

```@example juliahub
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

## Defining Experiment and InverseProblem

In order to create an [`Experiment`](@ref), we will use the default initial values of the states and parameters of our model. These are our initial guesses which will be used to optimize the inverse problem in order to fit the given data.

```@example juliahub
@unpack capacitor1, capacitor2 = model
experiment = Experiment(data, sys; overrides = [capacitor2.v => 0.0], alg = Rodas4(), abstol = 1e-6, reltol = 1e-5)
```

Once we have created the experiment, the next step is to create an [`InverseProblem`](@ref). This inverse problem, requires us to provide the search space as a vector of pairs corresponding to the parameters that we want to recover and the assumption that we have for their respective bounds. Here we are trying to calibrate the values of `capacitor2.C` which is the capacitance of `capacitor2`.

```@example juliahub
prob = InverseProblem(experiment, [capacitor2.C => (1.e-7, 1e-3)])
```

## Parametric Uncertainty Quantification

Now, lets do parameteric uncertainty quantification using [`parametric_uq`](@ref). Here is the part where we need to provide JuliaHub batch job specifications using [`JuliaHubJob`](@ref).

[`JuliaHubJob`](@ref) is a wrapper for all algorithms (like [`SingleShooting`](@ref), [`MultipleShooting`](@ref), [`parametric_uq`](@ref)) to add compute specifications like number of cpus, memory etc. (see [here](https://help.juliahub.com/julia-api/stable/reference/job-submission/#JuliaHub.submit_job) for more details), batch image to use, name of the `JuliaHub.dataset` where the results would be uploaded and the authentication token.

```@example juliahub
alg = StochGlobalOpt(
    method = SingleShooting(maxiters = 10^3),
    parallel_type = EnsembleDistributed())

specs = (ncpu = 8,
        memory = 64,
        nnodes = 1,
        process_per_node = false)

alg_juliahub = JuliaHubJob(; auth,
    batch_image = JuliaHub.batchimage("juliasim-batch", "JuliaSim - Stable"),
    node_specs = specs, dataset_name = "desauty",
    alg = alg)
```

Once we define this, running [`parametric_uq`](@ref) or [`calibrate`](@ref) is the same except instead of passing `alg`, we pass in `alg_juliahub`. This returns a job object as the batch job will run asynchronously.

```julia
sample_size = 100
ps_job = parametric_uq(prob, alg_juliahub; sample_size = sample_size)
```

```@example juliahub
sample_size = 100 # hide
ps = parametric_uq(prob, alg; sample_size = sample_size) # hide
nothing # hide
```

Once a job is launched, we can see the progress and logs on the JuliaHub platform. When the job is finished, we can download the dataset which corresponds to the results and inspect it.

```julia
JuliaHub.download_dataset("desauty", "./ps"; auth)
ps = deserialize("ps")
```

```@example juliahub
ps # hide
```

## Visualization

We can see the results of the analysis by plotting its histogram.

```@example juliahub
plot(ps, bins = 100, legend = :best)
```

This way we can launch batch jobs for long standing computations in a seamless manner.
