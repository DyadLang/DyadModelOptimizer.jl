# Calibrating Models to Data

Calibrating models to data, or finding parameters which make a model a sufficiently close fit to data, is part of the core functionality of DyadModelOptimizer. The `calibrate` function is designed as an automated mechanism for performing such model calibration in an accurate and efficient manner. It uses alternative methods for building the cost functions, also called [transcription methods](https://en.wikipedia.org/wiki/Trajectory_optimization#Terminology), such as multiple shooting and collocation to improve the stability of the training process, along with mixing in techniques for sampling initial conditions, to ensure that an accurate fit can be generate with little to no effort.

!!! note

    `calibrate` is designed for finding a single best fitting parameter set for a model.
    For a set of potential fits to characterize fitting uncertainty, see the
    [documentation on the `parametric_uq` function for parametric uncertainty quantification](@ref parametric_uq)

## Objective function definition

```@docs
objective
cost_contribution
```

## The `calibrate` Function

```@docs
calibrate
```

## Ipopt Optimizer

```@docs
IpoptOptimizer
```

## Calibration Algorithms

### Single Shooting

```@docs
SingleShooting
```

### Multiple Shooting

```@docs
MultipleShooting
```

#### [Initialization Methods](@id ms_init)

```@docs
DefaultSimulationInitialization
DataInitialization
RandomInitialization
```

### Data Shooting

```@docs
DataShooting
```

### Collocation methods

Collocation methods work by computing the derivatives of the data instead of integrating the equations.
For more details on how they work, take a look at [the collocation manual page](@ref collocation_page).

!!! note

    Collocation methods work with data by design, so they can only be used when you have data in the loss functions.

## Model transformations

### Prediction Error Methods

```@docs
DiscreteFixedGainPEM
```
