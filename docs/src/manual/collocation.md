# [Collocation Methods](@id collocation_page)

## Introduction

Many methods for solving inverse problems involving ODE's need costly ODE solvers at each iteration to compute the loss. E.g. for L2 loss:

``\mathrm{Loss}_{\mathrm{L2}} = \left\|u - \hat{u}_{\mathrm{solver}}(u_0, t,p) \right\|_2``

Where ``u`` is the observed data, ``\hat{u}_{\mathrm{solver}}(u_0, t,p)`` is the ODE solver prediction of ``u``, given initial conditions ``u_0``, timespan ``t`` and parameters ``p``.

In order to reduce the computational cost, a collocation method may be used [^1], [^2]. Collocation methods circumvents the need for an ODE solver by estimating the derivatives of the data. The loss is computed between these estimated derivatives and the system dynamics. E.g.

``\mathrm{Loss}_{\mathrm{L2}} = \left\|\hat{u}' - f(u, t,p) \right\|_2``

Where ``\hat{u}'`` is the estimated derivatives of the observed data.

DyadModelOptimizer implements three different collocation methods to estimate the derivative of the data: Kernel Collocation, Spline Collocation and Noise Robust Collocation.

## Kernel Collocation

```@docs
KernelCollocation
```

## Spline Collocation

```@docs
SplineCollocation
```

## Noise Robust Collocation

```@docs
NoiseRobustCollocation
```

## CollocationData for plotting

```@docs
CollocationData
```

[^1]: Roesch, Elisabeth, Christopher Rackauckas, and Michael PH Stumpf. "Collocation based training of neural ordinary differential equations." Statistical Applications in Genetics and Molecular Biology 20, no. 2 (2021): 37-49.
[^2]: Liang, Hua, and Hulin Wu. "Parameter estimation for differential equation models using a framework of measurement error in regression models." Journal of the American Statistical Association 103, no. 484 (2008): 1570-1583. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2631937/
