# Version 3.1.1

## Bug fixes
- Fix GSA errors due to the missing `samples` keyword argument.

# Version 3.1.0

## New features
- plot recipe (`JuliaComputing/JuliaSimModelOptimizer.jl#232`): `plot(::AbstractQSPResult, ::InverseProblem; trial_names=get_name.(prob.trials))), kwargs...)` to plot each trial whose name is in `trial_names`, for every virtual patients in `AbstractQSPResult`.

## Bug fixes
- Fix plotting bounds data (`JuliaComputing/JuliaSimModelOptimizer.jl#236`): fix plot recipes breaking when keyword argument `show_data=true` and data is `(lower, upper)` bounds. Now `show_data` will plot two dashed lines for lower and upper bounds respectively.
- Fix a bug in the `plot(vp, trial)` plot recipe where `saveat_reduction` would not be forwarded to the underlying `solve_ensemble` call.
- Fix color and legend bugs in `plot(vp, trial)` and `plot(trial, prob, x)`.
- Fix `plot(vp, trial, summary = true)` not respecting the `states` argument.
- Remove OptimizationBBO specific kwargs from the `vpop` solve call. This will make it possible to use other optimization methods.

## Docs
- Plot Function page (`JuliaComputing/JuliaSimModelOptimizer.jl#232`): API page with docstrings for each plot recipe. Uses placeholder function names to pass docstrings to `@docs`.

## Tests
- Group all plot recipe tests in a `test_plots` function that runs the `@testset` (`JuliaComputing/JuliaSimModelOptimizer.jl#236`).

## Other
- Removed separate `Trial` and `SteadyStateTrial` constructors when data is `(lower, upper)` bounds (`JuliaComputing/JuliaSimModelOptimizer.jl#236`).

# Version 3.0.2

## New features
- `DDS` subsampling method (`JuliaComputing/JuliaSimModelOptimizer.jl#221`): Custom-made subsampling method that matches the histogram of a virtual population to some input reference histogram for each one of the considered model states.

## Bug fixes
- Fix indexing a `vp::MCMCResult` with an `::AbstractVector` (`JuliaComputing/JuliaSimModelOptimizer.jl#228`). This fixes using `vp::MCMCResult` in `subsample`.

# Version 3.0.1

## Bug fixes
- Fix compat (`JuliaComputing/JuliaSimModelOptimizer.jl#215`): A [Symbolics.jl issue](https://github.com/JuliaSymbolics/Symbolics.jl/issues/670) prompted us to add a compat bound on Symbolics to ~4.9. It was [fixed](https://github.com/JuliaSymbolics/Symbolics.jl/pull/671) with Symbolics v4.10.2 and the constraint was removed.

## Other
- README updates (`JuliaComputing/JuliaSimModelOptimizer.jl#213`)

# Version 3.0.0

## New features
- MCMC Refactor (`JuliaComputing/JuliaSimModelOptimizer.jl#188`): Trials now contain a `likelihood` function and `noise_priors` for the scale parameters of typical likelihoods (e.g. standard deviation terms in a Normal). Likelihoods are the closest to a Bayesian equivalent for the `err` function of standard optimization, so makes sense to keep them on the trial level. This way users can adapt the `likelihood`s and `noise_priors` according to what is measured in a trial. The assumption is that each `Trial` has its own `noise_priors`, either a common one for all `save_idxs` or one per `save_idxs`.
- Add timespan optimization (`JuliaComputing/JuliaSimModelOptimizer.jl#205`): The timespan  can now be specified symbolically. If the parameters used are in the search space, the timespan will be computed using the parameters from the optimization. The `saveat` can also be specified similarly.

## Breaking changes
- Remove trial caching (`JuliaComputing/JuliaSimModelOptimizer.jl#195`)
- Remove MCMCModelCache (`JuliaComputing/JuliaSimModelOptimizer.jl#197`)
- Simplify API (`JuliaComputing/JuliaSimModelOptimizer.jl#198`)
- Refactor Subsample API (`JuliaComputing/JuliaSimModelOptimizer.jl#202`)
- Rename QSPCost and QSPSensitivity (`JuliaComputing/JuliaSimModelOptimizer.jl#212`)

### Upgrade steps
- Trials can no longer be cached. This feature was not documented and not used by anyone. Removing this makes the `QSPCost` / `InverseProblem` thread safe.
- The `solve_trial` and the trial plot recipe no longer have the `ss_trial` keyword argument for specifying the steady state trial for a trial with `forward_u0=true`. This is now automatically retrieved by the `solve_trial` function if a trial needs it.
- `subsample(alg, vp, trial; kwargs...)` is now the subsampling API. Note that the `alg` argument has been moved to the first position. Each subsampling algorithm alg (e.g. MAPEL, ARM, etc) has now a `Sampler` callable struct that is returned by `get_sampler(alg, vp, trial)`. Users can then call the `Sampler` with the original `vp` to get a subsampled `vp`.
```julia
alg = MAPEL(binning_function, reference_weights, N_patients_to_subsample)
vp_subsampled = subsample(alg, vp, trial)

# ___OR___

sampler = get_sampler(alg, vp, trial)
idxs = sampler()
vp_subsampled = vp[idxs]
```
- `QSPCost(model, trials; search_space)` and `QSPSensitivity(model, trials; parameter_space)` have been replaced by `InverseProblem(trials, model, search_space)` and `SensitivityProblem(trials, model, parameter_space)`. Note that the order of the arguments has changed and that the `search_space` is no longer a keyword argument. Instead of building cost functions we now build the corresponding problems to be solved and the functions corresponding to the problems are conceptually separate. The `InverseProblem` is still a callable struct, but this will be deprecated in a future version and it is not part of the public API. We will have a dedicated function for evaluating the cost corresponding to a problem.

## Bug fixes
- Fix bugs (`JuliaComputing/JuliaSimModelOptimizer.jl#207`): Fix `vpop_prob` and cost errors from importing vpops. The cost function now works on named tuples that would arise form the Tables.jl interface.

## Docs
- MCMC documentation (`JuliaComputing/JuliaSimModelOptimizer.jl#199`)
- Update docs links (`JuliaComputing/JuliaSimModelOptimizer.jl#206`)
- Refactor Docs (`JuliaComputing/JuliaSimModelOptimizer.jl#209`)

## Other
- CompatHelper: add new compat entry for MCMCChains at version 5, (keep existing compat) (`JuliaComputing/JuliaSimModelOptimizer.jl#196`)
- Leucine model updates (`JuliaComputing/JuliaSimModelOptimizer.jl#203`)
