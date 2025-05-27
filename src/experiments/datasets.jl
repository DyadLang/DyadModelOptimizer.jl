function Experiment(dataset::DyadDataset, model::AbstractTimeDependentSystem; kwargs...)
    data = build_dataframe(dataset)
    Experiment(data, model; indepvar = Symbol(dataset.independent_var), kwargs...)
end

function SteadyStateExperiment(
        dataset::DyadDatapoint, model::AbstractTimeDependentSystem; kwargs...)
    data = build_dataframe(dataset)
    SteadyStateExperiment(data, model; kwargs...)
end

for et in [:Experiment, :SteadyStateExperiment]
    @eval begin
        function ($et)(model::AbstractTimeDependentSystem; kwargs...)
            ($et)(nothing, model; kwargs...)
        end
    end
end
