abstract type AbstractExperimentCollection{T} end

get_experiments(aec::AbstractExperimentCollection) = aec.experiments

Base.eltype(::AbstractExperimentCollection{T}) where {T} = T

Base.length(aec::AbstractExperimentCollection) = length(get_experiments(aec))

function Base.iterate(aec::AbstractExperimentCollection, state...)
    iterate(get_experiments(aec), state...)
end

Base.keys(aec::AbstractExperimentCollection) = keys(get_experiments(aec))

Base.@propagate_inbounds function Base.getindex(aec::AbstractExperimentCollection,
        i::Integer)
    get_experiments(aec)[i]
end

Base.firstindex(aec::AbstractExperimentCollection) = firstindex(get_experiments(aec))
Base.lastindex(aec::AbstractExperimentCollection) = lastindex(get_experiments(aec))

abstract type AbstractIndependentExperiments{T} <: AbstractExperimentCollection{T} end

struct IndependentExperiments{T <: AbstractExperiment} <: AbstractIndependentExperiments{T}
    experiments::Vector{T}
end

"""
    IndependentExperiments(experiments...)

This experiment collection type indicates that each experiment can be solved individually
and that there is no interaction between them. This experiment type is automatically
created it the experiments are passed as a `Vector` (i.e. [experiment1, experiment2]).
"""
IndependentExperiments(experiments...) = IndependentExperiments(map(identity,
    [experiments...]))
function IndependentExperiments(experiments::AbstractVector)
    IndependentExperiments(map(identity, experiments))
end
