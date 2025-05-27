# [Experiments](@id trials_page)

In DyadModelOptimizer, the different variations of the model to be ran are called "experiments". For example, one experiment may specify that the model should be solved with a driving voltage of 10V pulse, while the next experiment specifies that the driving voltage is a 5V pulse. Each experiment is then optionally tied to a dataset which, when defined in an inverse problem, specifies a multi-simulation optimization problem that the further functions (`calibrate`, `parametric_uq`, etc.) generate solutions for. The type of experiment is used to signify what the data corresponds to measuring, i.e. whether the experiment is used to match data of time series, or steady states, etc.

## Experiment Types

The following describes the types of experiments which can be generated.

```@docs
Experiment
DiscreteExperiment
SteadyStateExperiment
```

## Design optimization

For design optimization problems, the `DesignConfiguration` API can be used.

```@docs
DesignConfiguration
```

## Simulation and Analysis Functions

To better understand and debug experiments, the experiments come with associated analysis functions to allow for easy investigation of the results in an experiment-by-experiment form. The following functions help the introspection of such experiments.

```@docs
DyadInterface.simulate(::DyadModelOptimizer.AbstractExperiment, ::DyadModelOptimizer.AbstractInverseProblem, ::Any, ::Any)
```

## Loss Functions

By default, the loss function associated with a experiment against its data is the standard Euclidean distance, also known as the L2 loss. However, `DyadModelOptimizer` provides alternative loss definitions to allow for customizing the fitting strategy.

```@docs
squaredl2loss
l2loss
meansquaredl2loss
root_meansquaredl2loss
norm_meansquaredl2loss
zscore_meansquaredl2loss
ARMLoss
```

## Remake

```@docs
remake
```

## DyadInterface API

```@docs
CalibrationAnalysisSpec
DyadInterface.run_analysis
DyadInterface.AbstractAnalysisSolution
```
