# Results interface

```@docs
CalibrationResult
ParameterEnsemble
```

## Saving calibration results

Since calibration results behave like tables, we can easily export the results to as `.csv` file using

```julia
CSV.write(io, res)
```

We can also read back the results in with [`import_res`](@ref). Note that reconstructing the result requires the corresponding inverse problem to be defined.

```julia
prob = InverseProblem(...)
csv = CSV.File(fn)
icsv = import_res(csv, prob)
```

# Importing results

```@docs
import_res
import_ps
```

## Using imported calibration results

We can use imported calibration results just like we had computed them in the current session.
For example they can be used for simulating the results of an experiment given the known calibrated parameters:

```julia
csv = CSV.File(fn)
icsv = import_res(csv, prob)
isol = simulate(experiment, prob, icsv)
```

We can also use [`import_res`](@ref) to continue a calibration later. For example if we already have a calibration result saved in a file, we can continue the calibration from the imported results.

```julia
csv = CSV.File(fn)
icsv = import_res(csv, prob)
calibrate(icsv, SingleShooting(maxiters = 10^3))
```
