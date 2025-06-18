abstract type AbstractCalibrationAnalysisSolution <: AbstractAnalysisSolution end

struct CalibrationAnalysisSolution{S, R} <: AbstractCalibrationAnalysisSolution
    spec::S
    r::R
end

function Base.show(io::IO, m::MIME"text/plain", spec::AbstractCalibrationAnalysisSolution)
    print(io, "Calibration Analysis solution for $(nameof(spec.spec)): ")
    show(io, m, spec.r)
end

function get_experiments(oas::AbstractCalibrationAnalysisSolution)
    get_experiments(oas.r.prob)
end

function experiment_by_name(name::Symbol, oas::AbstractCalibrationAnalysisSolution)
    experiments = get_experiments(oas)
    if length(experiments) > 1
        idx = parse(
            Int, match(r"CalibrationResult[a-zA-Z]*Experiment([0-9]*)", string(name))[1])
        # @debug "experiment idx: $idx"
        experiments[idx]
    else
        only(experiments)
    end
end

function get_calibration_results_metadata(oas::AbstractCalibrationAnalysisSolution)
    exps = get_experiments(oas)
    ex_idxs = eachindex(exps)

    # Plots
    plt_names = [Symbol("CalibrationResultPlotExperiment$i") for i in ex_idxs]
    plt_types = Any[ArtifactType.PlotlyPlot for i in ex_idxs]
    plt_titles = ["Calibration results for experiment $i" for i in ex_idxs]
    plt_descriptions = ["Simulation results using the calibration parameters for experiment $i."
                        for i in ex_idxs]

    # Parameter tables

    # TODO: if an experiment has an override of something that is also in the search space,
    # we need to replace the result with the override given value and not show something from the optimization
    # This can happen if we have more than one experiment and one experiment sets an override on something
    # that another experiment is adding to the search space.
    # The plots correctly handle this since `simulate` is used.
    # This is a longstanding issue: https://github.com/JuliaComputing/DyadModelOptimizer.jl/issues/111
    tbl_names = [Symbol("CalibrationResultTableExperiment$i") for i in ex_idxs]
    tbl_types = Any[ArtifactType.DataFrame for i in ex_idxs]
    tbl_titles = ["Calibrated parameters for experiment $i" for i in ex_idxs]
    tbl_descriptions = ["Parameter values for experiment $i." for i in ex_idxs]

    artifacts = [ArtifactMetadata(name, type, title, description)
                 for (name, type, title, description) in zip(
        vcat(plt_names, tbl_names), vcat(plt_types, tbl_types), vcat(
            plt_titles, tbl_titles),
        vcat(plt_descriptions, tbl_descriptions))]

    push!(artifacts,
        ArtifactMetadata(:ConvergencePlot, ArtifactType.PlotlyPlot, "Convergence plot",
            "Loss function evolution during calibration"))

    @assert length(ex_idxs)==1 "allowed_symbols is not clearly defined on multiple experiments"
    allowed_symbols = getname.(variable_symbols(get_prob(only(exps))))

    AnalysisSolutionMetadata(artifacts, allowed_symbols)
end

function DyadInterface.AnalysisSolutionMetadata(res::CalibrationAnalysisSolution)
    get_calibration_results_metadata(res)
end

function DyadInterface.artifacts(oas::CalibrationAnalysisSolution, name::Symbol)
    experiment = experiment_by_name(name, oas)
    if startswith(string(name), "CalibrationResultPlotExperiment")
        plot(experiment, oas.r, show_data = true, legend = true, saveat = ())
    elseif startswith(string(name), "CalibrationResultTableExperiment")
        DataFrame(oas.r) # TODO: this should depend on the experiment too
    elseif name == :ConvergencePlot
        convergenceplot(oas.r, yscale = :log10)
    else
        error("Result type $name not recognized!")
    end
end

function DyadInterface.customizable_visualization(
        oas::AbstractCalibrationAnalysisSolution, vizdef::PlotlyVisualizationSpec)
    plotallexperiments(oas.r, idxs = vizdef.symbols, show_data = true)
end
