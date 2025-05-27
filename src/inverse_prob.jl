struct InverseProblem{E, I} <: AbstractInverseProblem{E}
    experiments::E
    internal_storage::I

    @doc """
        InverseProblem(experiments, search_space)

    The `InverseProblem` represents the problem of finding some parameters and/or initial conditions
    that best fit what we know about our system(s). The known information is expressed in the form of
    an [`Experiment`](@ref), or a collection of such objects. The `search_space` is a `Vector` of pairs.
    Each pair consists of a parameter or initial condition that we need to find and its lower and upper bounds.
    If the keys of the pairs are symbolic variables, they are assumed to be part of one of the models in the
    inverse problem. If they are a `String`, then they are assumed to represent a description and it is required
    for that description to be unique across models.
    The value can be:

    - a tuple of 2 numbers representing lower and upper bounds
    - a tuple of 3 numbers, where the first will be the initial guess and the last 2 will be lower and upper bounds
    - a tuple of 3 elements, where the first 2 are numbers and represent lower and upper bounds and the last element
      is a `Symbol`, specifying a model transformation (:log, :log10 or :identity)
    - a tuple of 4 elements, where the first number is the initial guess, the last 2 will be lower and upper bounds
      and the last is a `Symbol`, specifying a model transformation (:log, :log10 or :identity)
    - a distribution (`<:Distribuitions.Sampleable`)
    - a tuple of a distribution (`<:Distribuitions.Sampleable`) and a `Symbol`,
      specifying a model transformation (:log, :log10 or :identity)

    Each experiment defines an error function that can expresses how well a certain combination of `search_space` values
    fits the corresponding data or how well does that particular configuration match a certain objective.

    All the `experiments` are taken into account when finding the values in the `search_space`, their individual
    contributions being summed when computing the overall metric for how good a particular combination of `search_space`
    values is. This overall metric can be obtained by calling [`objective`](@ref).

    ## Positional arguments
    - `experiments`: the experiments defining the inverse problem
    - `search_space`: the search space for the inverse problem representing the unknown parameters or initial conditions to be tuned
    """
    function InverseProblem(experiments::AbstractVector, search_space)
        experiments = to_collection(experiments) # TODO get rid of this
        ist = InternalStorage(experiments, search_space)
        new{typeof(experiments), typeof(ist)}(experiments, ist)
    end

    function InverseProblem(experiments, search_space)
        ist = InternalStorage(experiments, search_space)
        new{typeof(experiments), typeof(ist)}(experiments, ist)
    end

    function InverseProblem(experiment::AbstractExperiment, search_space)
        experiments = to_collection([experiment]) # TODO get rid of this
        ist = InternalStorage(experiments, search_space)
        new{typeof(experiments), typeof(ist)}(experiments, ist)
    end
end

get_uuid(mt::AbstractModelTransformation) = mt.uuid

function to_collection(experiments::AbstractVector)
    to_collection(typeof(first(experiments)), experiments)
end
function to_collection(::Type{<:AbstractExperiment}, experiments)
    IndependentExperiments(experiments)
end

function get_search_space(prob::AbstractDesignAnalysis)
    get_search_space(get_internal_storage(prob))
end

get_penalty(prob::AbstractDesignAnalysis) = get_penalty(get_internal_storage(prob))
