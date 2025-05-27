# [Subsampling Parameter Ensembles](@id subsample_page)

While the [`parametric_uq`](@ref) function returns a `VirtualPopulation` ensemble of parameter values which correspond to relatively good fits to the data, in many cases this return is referred to as a "plausible population", i.e. a set of potentially good parameters which may or may not reflect the statistical effects of the population.

For example, say that for the data series that is being fit, 50% of the population is known to be fast matabolizers and the other 50% is comprised of slow matabolizers (characterized by some parameter or measurement in the model). A virtual population method, `vpop`, will return a plausible population of `N` plausible parameters but there is no guarantee that it captured the distribution of fast and slow metabolizers from the actual population. The purpose of a `subsample` algorithm is to downsample from `N` to `M` to find a subpopulation which captures important statistical quantities better than the plausible population.

## The `subsample` Function

```@docs
subsample
```

## Accessing the Sampler

Each subsampling algorithm internally uses a `Sampler` object to perform subsampling on a plausible population `vp`. One can call the `subsample` method multiple times to rerun the subsampling process and produce a different population each time. However, this might be computationally expensive, as methods like `ARM` need to solve an optimization problem before producing a subsampled population.

Users can access the internal `Sampler` object directly and call it multiple times to generate different populations. The `Sampler` is initialized with all the necessary information that the respective subsampling algorithm needs. Thus, by using this object, one can avoid the computational cost of running the entire `subsample` method multiple times, if multiple subsampled populations are required.

```@docs
get_sampler
```
