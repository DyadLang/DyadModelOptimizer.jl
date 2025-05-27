# TODO:
# define the Abstract types that are referenced in the docstrings,
# e.g. AbstractSubsamplingAlgorithm, AbstractVirtualPopulation, or change the docstrings

"""
    function subsample(alg::AbstractSubsamplingAlgorithm, ps::AbstractParametricUncertaintyEnsemble,
                    experiment::AbstractExperiment; kwargs)

Subsamples a parameter ensemble `ps` in order to create a new ensemble
which satisfies the given constraints as specified by the sampling algorithm
`alg` defined on a given `experiment`.
"""
function subsample(alg, ps, experiment; kwargs...)
    sampler = get_sampler(alg, ps, experiment; kwargs...)
    idxs = sampler()

    return ps[idxs]
end

"""
    function get_sampler(alg::AbstractSubsamplingAlgorithm, ps::AbstractParametricUncertaintyEnsemble,
        experiment::AbstractExperiment; kwargs)

Returns a sampler that, when called, produces a subset of indices from the parameter ensemble `ps`.
The sampled indices satisfy the overall given constraints
as specified by the sampling algorithm `alg` defined on a given `experiment`.

## Example

```julia
ps = parametric_uq(prob, alg; population_size = 100)
subsample_alg = WeightSampler()
sampler = get_sampler(subsample_alg, vp, experiment)
ps_subsampled = vp[sampler()]
```
"""
function get_sampler end

struct WeightSampler{W, I}
    # Use this sampler for methods with a given number n of patients to subsample.
    # w is a Weights() collection, one for each patient in the original virtual/plausible population
    # idxs is a collection of eachindex(virtual/plausible population) to be subsampled
    w::W
    idxs::I
    n::Int
end

(f::WeightSampler)() = sample(f.idxs, f.w, f.n)

"""
    RefWeights(; binning, reference_weights, n::Int)

A reference-weight based algorithm that calculates assigns weights to each plausible
patient and then uses them to subsample the plausible population into a virtual population.
Uses a binning function with reference weights for each bin to choose a subsample of the
plausible patient population which bins with the same frequency as the reference.

## Keyword Arguments

  - `binning`: A function `binning(sol)` which returns an integer representing the bin the
    patient belongs to.
  - `reference_weights`: A vector containing the probability of a patient belonging to
    each bin in the actual population. These weights need to sum to 1.
  - `n`: Number of plausible patients to subsample to the virtual population.

## References

Schmidt BJ, Casey FP, Paterson T, Chan JR. Alternate virtual populations elucidate the type
I interferon signature predictive of the response to rituximab in rheumatoid arthritis. BMC
Bioinformatics. 2013 Jul 10;14:221. doi: 10.1186/1471-2105-14-221. PMID: 23841912;
PMCID: PMC3717130.
"""
struct RefWeights{F, R}
    binning::F
    reference_weights::R
    n::Int
end

function RefWeights(; binning, reference_weights, n)
    @assert sum(reference_weights)≈1 "Reference weights do not sum to 1"
    RefWeights(binning, reference_weights, n)
end

(f::RefWeights)(vp) = f.binning(vp)

function get_sampler(alg::RefWeights, vp, experiment)
    sim = solve_ensemble(vp, experiment)
    reference_weights = alg.reference_weights
    response_types = keys(reference_weights)
    responses = [alg(sol) for sol in sim]
    # Compute how many responses do we have for each response type
    counts = [count(==(r), responses) for r in response_types]

    # Compute weights for the plausible patients
    # as (reference weights) / (current weights)
    # The current weights are counts / length(vp)
    response_weights = reference_weights ./ counts * length(vp)

    # Re-weighted responses
    w = Weights([response_weights[r] for r in responses])

    return WeightSampler(w, eachindex(vp), alg.n)
end

function KSStatistic(state, CDF)
    N = length(state)
    F_empirical = [count(state .<= x) for x in state] ./ N
    F_cdf = CDF.(state)
    maximum(abs.(F_empirical - F_cdf))
end

"""
    DDS(; reference, n::Int, nbins::Int)

Discretized Density Sampling (DDS) performs subsampling by matching the histogram of each model
state under consideration to a reference histogram. The method matches the frequency of plausible patients
that fall within each bin to the frequency of the same bin in the reference distribution.
This way, the final Virtual Population has bin frequencies (or probabilities) equal to those
of the reference distribution for each one of the considered model states.

## Keyword Arguments

  - `reference`: `Vector` of pairs. The first element of each pair is a model state and the
    second element is a `NamedTuple` containing two fields:

      + `dist`: a reference distribution that the algorithm will match for the given state.
        This could be any distribution, see [`TransformedBeta`](@ref) or [the Distributions.jl documentation](https://juliastats.org/Distributions.jl/stable/univariate/)) for more options.
      + `t`: the timepoint along the state trajectory where the state should match the reference `dist`.
        This does not have to coincide with a `saveat` timepoint of the considered experiment.

  - `n`: Number of plausible patients to subsample to the virtual population.
  - `nbins`: Number of bins to discretize the reference distributions into.

## Example

State `y1` at timepoint 5 [modelled units of time] should be distributed as a `Beta(2,6)` distribution
within the bounds `[1, 5]`, which is a [`TransformedBeta`](@ref) distribution,
and state `y2` should be distributed as an `InverseGamma(2,2)` at the 10 timepoint.
The number of patients is set to 100 and number of bins for each state is 20.

```julia
ref = [
    y1 => (dist = TransformedBeta(Beta = Beta(2, 6), lb = 2, ub = 6), t = 5),
    y2 => (dist = InverseGamma(2, 2), t = 10)
]
alg = DDS(reference = ref, n = 100, nbins = 20)
```

!!! tip "Tip : Visualizing distributions"

    One can easily visualize what a reference distribution looks like for a given bin size by running

    ```julia
    using StatsPlots

    reference_distribution = Normal(0, 1)
    number_of_bins = 20
    number_of_samples = 1000
    histogram(rand(reference_distribution, number_of_samples), nbins = number_of_bins)
    ```
"""
Base.@kwdef struct DDS{T}
    reference::T
    n::Int
    nbins::Int
end

function get_sampler(alg::DDS, vp, experiment)
    model = get_model(experiment)
    d = [last(ref).dist for ref in alg.reference]
    x_reference = rand.(d, 5_000) # just get a high number of samples for an accurate histogram
    # TO DO :
    # instead of nbins we can use AbstractRanges as edges
    # edges would create exactly the number of bins the users ask for
    # whereas nbins is only a suggestion, which might change internally to fit axis ticks nicely
    h = fit(Histogram, (x_reference...,); alg.nbins)
    edges = h.edges
    nbins = size(h.weights) # get actual number of bins, which might differ from alg.nbins
    W_reference = normalize(h).weights

    sim = solve_ensemble(vp, experiment)
    x_sim = map(alg.reference) do ref
        state = first(ref)
        state_idx = states_to_idxs([state], model)
        t = last(ref).t
        [sol(t)[state_idx] for sol in sim]
    end

    h_sim = fit(Histogram, (x_sim...,), edges)
    W_sim = normalize(h_sim).weights

    W = W_reference ./ W_sim
    #W[isinf.(W)] .= 0.0 # set the weight for bins where there were no Plausible Patients to zero

    x_sim = hcat(x_sim...)
    x_tpl = map(r -> (r...,), eachrow(x_sim))
    bin_idxs = binindex.(Ref(h), x_tpl)
    W_vp = map(bin_idxs) do idx
        any(iszero, idx) || any(idx .> nbins) ? 0.0 : W[CartesianIndex(idx)]
    end

    return WeightSampler(Weights(W_vp), eachindex(vp), alg.n)
end
