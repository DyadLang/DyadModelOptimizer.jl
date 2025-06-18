abstract type AbstractCalibrationAnalysisSpec <: AbstractAnalysisSpec end

"""
    CalibrationAnalysisSpec(;
        name,
        model,
        abstol,
        reltol,
        data,
        N_cols,
        depvars_cols,
        depvars_names,
        N_tunables,
        search_space_names,
        search_space_lb],
        search_space_ub,
        calibration_alg,
        optimizer_maxiters
    )

Calibration analysis specification that describes how to run parameter estimation via [`calibrate`](@ref).
The structure is an interface specification corresponding to
a [base analysis type in Dyad](https://symmetrical-adventure-kqpeovn.pages.github.io/design/analysis/).

When one creates an analysis in Dyad, the definition would look like

```
analysis LotkaVolterraCalibrationAnalysis
  extends CalibrationAnalysis(data=DyadDataset("file://lotka.csv", independent_var = "t", dependent_vars = ["x"]), N_cols = 1, depvars_cols=["x"], depvars_names=["x"], N_tunables = 1, search_space_names=["α"], search_space_lb=[1.1], search_space_ub=[2], calibration_alg="SingleShooting", optimizer_maxiters=100)

  model = LotkaVolterra()
end
```

# Keyword Arguments

  - `name`: The name of the analysis.
  - `model`: An `ODESystem` representing the model that will be used for numerical integration.
  - `alg`: The ODE integrator to use as a symbol. Possible options are: `:auto` (default), `:Rodas5P`, `:FBDF`, `:Tsit5`.
  - `start`: The start time for the integration. Defaults to 0.0.
  - `stop`: The end time for the integration.
  - `abstol`: Absolute tolerance to use during the simulation.  Defaults to `1e-8`.
  - `reltol`: Relative tolerance to use during the simulation.  Defaults to `1e-8`.
  - `saveat`: Timepoints to save the solution at or `0` (to let the integrator decide, default).
  - `dtmax`: The maximum allowed timestep or `0` (to let the integrator decide, default).
  - `data`: A DyadDataset from DyadData.
  - `indepvar`: The independent variable in the data that was passed.
  - `N_cols`: The number of dependent variable columns in the data.
  - `depvars_cols`: A vector of column names for dependent variables.
  - `depvars_names`: A vector of names in the model that map to the `depvars_cols`. Note that they need to be in the same order.
  - `loss_func`: The loss function to use for comparing data to simulation results. Defaults to "l2loss". Options are: "l2loss", "norm_meansquaredl2loss", "meansquaredl2loss, "squaredl2loss".
  - `N_tunables`: The number of tunable parameters.
  - `search_space_names`: The names of the tunable parameters (as a vector of strings).
  - `search_space_lb`: A vector of values for the lower bound of the tunable parameters.
  - `search_space_ub`: A vector of values for the upper bound of the tunable parameters.
  - `calibration_alg`: The calibration algorithm to use. Available options are: "SingleShooting", "MultipleShooting", "SplineCollocation" & "KernelCollocation".
  - `multiple_shooting_trajectories`: If `MultipleShooting` is selected, specify the number of trajectories. See [`MultipleShooting`](@ref) for more details.
  - `pem_gain`: Gain factor for Prediction Error Method. If this is greater than 0, the [`DiscreteFixedGainPEM`](@ref) model transformation is applied.
  - `optimizer`: The optimizer to use. Available options are: "auto", "Ipopt", "BBO".
  - `optimizer_maxiters`: The maximum number of iterations for the optimizer.
  - `optimizer_maxtime`: The maximum number of (real-world) seconds for the optimizer to run.
"""
@kwdef struct CalibrationAnalysisSpec{
    M, S <: AbstractString, D <: DyadDataset, T1, T2, T3, T4} <:
              AbstractCalibrationAnalysisSpec
    name::Symbol
    model::M
    alg::S = "auto"
    start::T1 = 0.0
    stop::T1 = 0.0
    abstol::T2 = 1e-8
    reltol::T3 = 1e-8
    saveat::Union{T1, Vector{<:T1}} = 0.0
    dtmax::T1 = 0.0
    data::D
    N_cols::Int
    depvars_cols::Vector{S}
    depvars_names::Vector{S}
    loss_func::S = "l2loss"
    N_tunables::Int
    search_space_names::Vector{S}
    search_space_lb::Vector{T4}
    search_space_ub::Vector{T4}
    calibration_alg::S
    multiple_shooting_trajectories::Int = 0
    pem_gain::Float64 = 0.0
    optimizer::S = "auto"
    optimizer_abstol::Float64 = 1e-4
    optimizer_maxiters::Int
    optimizer_maxtime::Float64 = 0.0
    optimizer_verbose::Bool = false
end

function translate_optimizer(spec::AbstractCalibrationAnalysisSpec)
    optimizer_name = spec.optimizer

    if optimizer_name == "Ipopt" || optimizer_name == "auto"
        IpoptOptimizer(; verbose = spec.optimizer_verbose, tol = spec.optimizer_abstol)
    elseif optimizer_name == "BBO"
        BBO_adaptive_de_rand_1_bin_radiuslimited()
    else
        error("optimizer $optimizer_name not supported yet.")
    end
end

function get_calibration_alg(spec::AbstractCalibrationAnalysisSpec)
    optimizer = translate_optimizer(spec)
    calibration_alg_name = spec.calibration_alg
    abstol = spec.optimizer_abstol
    maxiters = iszero(spec.optimizer_maxiters) ? nothing : spec.optimizer_maxiters
    maxtime = iszero(spec.optimizer_maxtime) ? nothing : spec.optimizer_maxtime
    verbose = spec.optimizer_verbose

    common_kwargs = if !(spec.optimizer == "auto" || spec.optimizer == "Ipopt")
        (; abstol, maxiters, maxtime, verbose, optimizer)
    else
        # Ipopt needs special handling
        (; maxiters, maxtime, optimizer)
    end

    if calibration_alg_name == "SingleShooting"
        SingleShooting(; common_kwargs...)
    elseif calibration_alg_name == "MultipleShooting"
        trajectories = spec.multiple_shooting_trajectories
        MultipleShooting(; trajectories, common_kwargs...)
    elseif calibration_alg_name == "SplineCollocation"
        SplineCollocation(; common_kwargs...)
    elseif calibration_alg_name == "KernelCollocation"
        KernelCollocation(; common_kwargs...)
    else
        error("Calibration alg $calibration_alg_name not yet supported.")
    end
end

function get_search_space(spec::AbstractCalibrationAnalysisSpec)
    @assert ModelingToolkit.isscheduled(spec.model) "Expected the model to be structurally simplified at this point."
    [parse_variable(spec.model, var) => (lb, ub)
     for (var, lb, ub) in zip(
        spec.search_space_names, spec.search_space_lb, spec.search_space_ub)]
end

function get_model_transformations(spec::CalibrationAnalysisSpec)
    if !iszero(spec.pem_gain)
        [DiscreteFixedGainPEM(spec.pem_gain)]
    else
        []
    end
end

function Base.show(io::IO, m::MIME"text/plain", spec::AbstractCalibrationAnalysisSpec)
    print(io, "Calibration Analysis specification for ")
    printstyled(io, "$(nameof(spec))\n", color = :green, bold = true)

    println(io, "\nSearch space: ", join(get_search_space(spec), " "))

    println(io, "Calibration alg: ", spec.calibration_alg)
end

function setup_loss_func(spec::AbstractCalibrationAnalysisSpec)
    if spec.loss_func == "l2loss"
        l2loss
    elseif spec.loss_func == "norm_meansquaredl2loss"
        norm_meansquaredl2loss
    elseif spec.loss_func == "meansquaredl2loss"
        meansquaredl2loss
    elseif spec.loss_func == "squaredl2loss"
        squaredl2loss
    elseif spec.loss_func == "zscore_meanabsl1loss"
        zscore_meanabsl1loss
    else
        error("loss function name $(spec.loss_func) not recognized")
    end
end

function setup_experiment(spec::AbstractCalibrationAnalysisSpec;
        loss_func = setup_loss_func(spec),
        model_transformations = get_model_transformations(spec),
        verbose = true)
    model = spec.model
    ex_config = ODEProblemConfig(spec)
    # translate depvars to model variables
    @assert ModelingToolkit.isscheduled(model) "Expected the model to be structurally simplified at this point."
    depvars = [parse_variable(model, var) => col_name
               for (col_name, var) in zip(spec.depvars_cols, spec.depvars_names)]
    @debug "translated depvars in analysis: $depvars"
    data = build_dataframe(spec.data)
    indepvar = Symbol(spec.data.independent_var)

    saveat = if isempty(ex_config.saveat)
        # no saveat provided, using data as fallback
        determine_saveat(data, indepvar)
    else
        ex_config.saveat
    end

    tspan = if ex_config.tspan == (0.0, 0.0)
        determine_tspan(data, indepvar)
    else
        ex_config.tspan
    end
    if iszero(ex_config.dtmax)
        dtmax = tspan[end] - tspan[begin]
    else
        dtmax = ex_config.dtmax
    end

    Experiment(data, model;
        name = string(nameof(spec)),
        tspan,
        saveat,
        ex_config.alg,
        ex_config.abstol,
        ex_config.reltol,
        indepvar,
        depvars,
        loss_func,
        model_transformations,
        dtmax,
        verbose
    )
end

function setup_invprob(spec::AbstractCalibrationAnalysisSpec; verbose = true)
    model = get_simplified_model(spec)
    @set! spec.model = model

    ex = setup_experiment(spec; verbose)

    search_space = get_search_space(spec)
    @debug "running analysis with search_space = $search_space"
    InverseProblem(ex, search_space)
end

CalibrationAnalysis(; kwargs...) = run_analysis(CalibrationAnalysis(; kwargs...))

function DyadInterface.run_analysis(spec::CalibrationAnalysisSpec)
    # prepare
    @debug "preparing"
    invprob = setup_invprob(spec)
    calibration_alg = get_calibration_alg(spec)

    # run
    @debug "running calibration"
    r = calibrate(invprob, calibration_alg)

    # post-process
    @debug "preparing result"
    res = CalibrationAnalysisSolution(spec, r)
    return res
end
