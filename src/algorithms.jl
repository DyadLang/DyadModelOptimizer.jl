abstract type AbstractParametricUncertaintyAlgorithm end
abstract type AbstractCalibrationAlgorithm end

abstract type AbstractParametricBayesianAlgorithm <: AbstractParametricUncertaintyAlgorithm end
abstract type AbstractMultiCalibrationAlgorithm <: AbstractParametricUncertaintyAlgorithm end
abstract type AbstractMultipleShootingAlgorithm <: AbstractCalibrationAlgorithm end
abstract type AbstractCollocationAlgorithm <: AbstractCalibrationAlgorithm end
abstract type AbstractMultipleShootingInitializationMethod end

const MOI_OPTIMIZER_TYPE = Union{
    MOI.AbstractOptimizer,
    MOI.OptimizerWithAttributes
}

get_optimizer(a::AbstractCalibrationAlgorithm) = a.optimizer
function get_maxiters_optimizer(opt::MOI_OPTIMIZER_TYPE, maxiters)
    MOI.get(opt, MOI.RawOptimizerAttribute("max_iter"))
end
get_maxiters_optimizer(opt, maxiters) = maxiters
function get_maxiters(a::AbstractCalibrationAlgorithm)
    get_maxiters_optimizer(a.optimizer, a.maxiters)
end
get_maxtime_optimizer(opt, maxtime) = maxtime
function get_maxtime_optimizer(opt::MOI_OPTIMIZER_TYPE, maxtime)
    MOI.get(opt, MOI.RawOptimizerAttribute("max_wall_time"))
end
function get_maxtime(a::AbstractCalibrationAlgorithm)
    get_maxtime_optimizer(a.optimizer, a.maxtime)
end
get_kwargs(a::AbstractCalibrationAlgorithm) = a.kwargs
get_kwargs(a::AbstractMultiCalibrationAlgorithm) = get_kwargs(a.method)
get_maxiters(a::AbstractParametricBayesianAlgorithm) = a.maxiters

"""
    IpoptOptimizer(;
        verbose = false,
        tol = 1e-4,
        max_iter = 1000,
        acceptable_iter = 250,
        acceptable_tol = 1e-3,
        hessian_approximation = "limited-memory", kwargs...)

A convenience constructor to create an `optimizer = Ipopt.Optimizer()` and set options. Other options can be passed as kwargs.
See https://coin-or.github.io/Ipopt/OPTIONS.html for information about each option. The defaults provided here are more relaxed than Ipopt defaults.
"""
function IpoptOptimizer(;
        verbose = false,
        tol = 1e-4,
        max_iter = 1000,
        acceptable_iter = 250,
        acceptable_tol = 1e-3,
        hessian_approximation = "limited-memory", kwargs...)
    return MOI.OptimizerWithAttributes(Ipopt.Optimizer,
        "tol" => tol, "max_iter" => isnothing(max_iter) ? 1000 : max_iter,
        "acceptable_tol" => acceptable_tol,
        "acceptable_iter" => isnothing(max_iter) ? 250 : acceptable_iter,
        "print_level" => verbose isa Bool ? verbose * 5 : verbose,
        "hessian_approximation" => hessian_approximation,
        [string(k) => v for (k, v) in kwargs]...)
end

function check_maxiters_maxtime!(optimizer, maxiters, maxtime)
    if isnothing(maxiters) && isnothing(maxtime)
        error("At least one of `maxiters` or `maxtime` must be provided.")
    end
    return maxiters, maxtime
end

function check_maxiters_maxtime!(optimizer::MOI_OPTIMIZER_TYPE, maxiters, maxtime)
    if isnothing(maxiters) && isnothing(maxtime)
        error("At least one of `maxiters` or `maxtime` must be provided.")
    end
    if !isnothing(maxiters)
        MOI.set(optimizer, MOI.RawOptimizerAttribute("max_iter"), maxiters)
    end
    if !isnothing(maxtime)
        # maxtime needs to be a float
        MOI.set(optimizer,
            MOI.RawOptimizerAttribute("max_wall_time"),
            convert(AbstractFloat, maxtime))
    end
    # This is done to as maxtime and maxiters are set in the optimizer itself
    return nothing, nothing
end

"""
    SingleShooting(;maxiters = nothing, maxtime = nothing, optimizer = IpoptOptimizer(; max_iter = maxiters), kwargs...)

Single shooting is the simplest transcription method for building the loss function. In this case we solve all the model equation and then we compare the solutions against the data with the user given error functions.

The optimization method will get the cost function as the sum of the individual previously mentioned costs. To change the optimization method, you can use the `optimizer` keyword argument.
The default optimization method is `Ipopt.Optimizer` which can be used from [`OptimizationMOI`](https://docs.sciml.ai/Optimization/stable/optimization_packages/mathoptinterface/).
We also have a convenience constructor [`IpoptOptimizer`](@ref) for defining it. The maximum number of iterations for the optimizer is given by `maxiters`, which is a required keyword argument.
Atleast one of `maxiters` or `maxtime` is required. Optimization stops when either one of the condition is met. Atleast one of `maxiters` or `maxtime` is required. If both of them are given, optimization stops after either of condition is first met.

## Keyword Arguments

  - `maxiters`: Maximum numbers of iterations when using the method with a time stepping optimizer.
  - `maxtime`: Maximum time for the optimization process.
  - `optimizer`: An Optimization.jl algorithm for the global optimization. Defaults to `Ipopt.Optimizer` constructed using [`IpoptOptimizer`](@ref).
  - `kwargs`: Keyword arguments passed onto `Optimization.solve`.
"""
struct SingleShooting{I, T, O, K} <: AbstractMultipleShootingAlgorithm
    maxiters::I
    maxtime::T
    optimizer::O
    kwargs::K
end

function SingleShooting(; maxiters = nothing, maxtime = nothing,
        optimizer = IpoptOptimizer(; max_iter = maxiters),
        kwargs...)
    maxiters, maxtime = check_maxiters_maxtime!(optimizer, maxiters, maxtime)
    SingleShooting(maxiters, maxtime, optimizer, kwargs)
end

"""
    DataShooting(;maxiters = nothing, maxtime = nothing, optimizer = IpoptOptimizer(; max_iter = maxiters), groupsize, ensemblealg = EnsembleThreads(), continuitylossfun = squaredl2loss, continuitylossweight = 100, kwargs...)

`DataShooting` is a variant of the multiple shooting method (see [`MultipleShooting`](@ref) for more details), where we use the data for the internal initial conditions between segments instead of having the optimizer find them.
If the data is considered noisy, it might be better to let the optimizer find the initial conditions, as we would inevitably introduce errors if we use `DataShooting`.
To specify the group size, which is the number of time points in each trajectory where data points on both ends are taken as initial conditions, we can use `groupsize` keyword argument. The continuity of the resulting solution ensemble is enforced using a continuity penalty between segments.
The continuity losss function can be changed via `continuitylossfun`, defaulting to [`squaredl2loss`](@ref) and the penalty used for the continuity is weighted via a the `continuitylossweight`, which defaults to 100.

The optimization method will get the cost function as the sum of the individual previously mentioned costs. To change the optimization method, you can use the `optimizer` keyword argument.
The default optimization method is `Ipopt.Optimizer` which can be used from [`OptimizationMOI`](https://docs.sciml.ai/Optimization/stable/optimization_packages/mathoptinterface/).
We also have a convenience constructor [`IpoptOptimizer`](@ref) for defining it. The maximum number of iterations for the optimizer is given by `maxiters`, which is a required keyword argument.
Atleast one of `maxiters` or `maxtime` is required. If both of them are given, optimization stops after either of condition is first met.

## Keyword Arguments

  - `maxiters`: Maximum numbers of iterations when using the method with a time stepping optimizer.
  - `maxtime`: Maximum time for the optimization process.
  - `optimizer`: An Optimization.jl algorithm for the global optimization. Defaults to `Ipopt.Optimizer` constructed using [`IpoptOptimizer`](@ref).
  - `groupsize`: Required. Number of time points in each trajectory.
  - `ensemblealg`: Parallelization method for the ensemble problem. Defaults to `EnsembleThreads()`.
  - `continuitytype`: Type of continuity enforced. This can be `ModelStatesPenalty` which adds a term in the loss which corresponds to continuity loss given by `continuitylossfun` or it can be `ConstraintBased` where the continuity equations are added as constraints for the optimizer.
  - `continuitylossfun`: Loss function to compute continuity penalty. This is only applicable if `continuitytype` is `ModelStatesPenalty`.
  - `continuitylossweight`: Weight multiplied with the continuity loss term in the total loss. This is only applicable if `continuitytype` is `ModelStatesPenalty`.
  - `initialization`: Initialization method of the segments. This is only applicable if `continuitytype` is `ModelStatesPenalty`.
  - `kwargs`: Keyword arguments passed onto `Optimization.solve`.
"""
struct DataShooting{C, I, T, O, E, K} <: AbstractMultipleShootingAlgorithm
    continuity::C
    maxiters::I
    maxtime::T
    optimizer::O
    ensemblealg::E
    groupsize::Int
    kwargs::K
end

function DataShooting(; maxiters = nothing,
        maxtime = nothing,
        optimizer = IpoptOptimizer(; max_iter = maxiters),
        ensemblealg = EnsembleThreads(),
        groupsize::Int,
        continuitylossfun = squaredl2loss,
        continuitylossweight = 100,
        kwargs...)
    maxiters, maxtime = check_maxiters_maxtime!(optimizer, maxiters, maxtime)
    DataShooting(ModelStatesPenalty(continuitylossfun, continuitylossweight),
        maxiters, maxtime, optimizer, ensemblealg, groupsize,
        kwargs)
end

abstract type AbstractMultipleShootingContinuityMethod end

struct SavedStatesPenalty{CF, CW} <: AbstractMultipleShootingContinuityMethod
    continuitylossfun::CF
    continuitylossweight::CW
end

struct ModelStatesPenalty{CF, CW} <: AbstractMultipleShootingContinuityMethod
    continuitylossfun::CF
    continuitylossweight::CW
end

struct ConstraintBased <: AbstractMultipleShootingContinuityMethod end

"""
    DataInitialization(; interpolation = CubicSpline)

This is used for initializing segments for doing calibration using [`MultipleShooting`](@ref). It is used for those states where data is present and the values are obtained by fitting using `interpolation`.
For other states, initial condition for that state is used for all the segments.

## Keyword Arguments

  - interpolation: Interpolation method from [DataInterpolations](https://docs.sciml.ai/DataInterpolations/stable/) which can be passed to fit data for obtaining initial values for the segments.
"""
struct DataInitialization{I} <: AbstractMultipleShootingInitializationMethod
    interpolation::I
end

function DataInitialization(; interpolation = CubicSpline)
    DataInitialization(interpolation)
end

"""
    DefaultSimulationInitialization()

This is used for initializing segments for doing calibration using [`MultipleShooting`](@ref).
The values for all the segments are computed by simulating the model passed in the [`Experiment`](@ref) with default parameters and using its `ODESolution`.
"""
struct DefaultSimulationInitialization <: AbstractMultipleShootingInitializationMethod end

"""
    RandomInitialization()

This is used for initializing segments for doing calibration using [`MultipleShooting`](@ref).
The values for all the segments are computed as a random value from 0 to 1 multiplied by the initial condition for that state.
"""
struct RandomInitialization <: AbstractMultipleShootingInitializationMethod end

struct MultipleShooting{C, I, M, O, E, T, IN, K} <: AbstractMultipleShootingAlgorithm
    continuity::C
    maxiters::I
    maxtime::M
    optimizer::O
    ensemblealg::E
    trajectories::T
    initialization::IN
    kwargs::K
end

"""
    MultipleShooting(;maxiters = nothing, maxtime = nothing, optimizer = IpoptOptimizer(; max_iter = maxiters), trajectories, ensemblealg = EnsembleThreads(), continuitytype = ModelStatesPenalty, continuitylossfun = squaredl2loss, continuitylossweight = 100, initialization = DefaultSimulationInitialization(), kwargs...)

Multiple shooting is a calibration algorithm in which we split the solves of the model equations in several parts, as it can be easier to avoid local minima if we fit the individual segments and them match them up.
The number of groups can be controlled via the `trajectories` keyword argument. The initial conditions between the segments are added to the optimization as extra hidden parameters that have to be fit.
The continuity of the resulting solution ensemble is enforced using a continuity penalty between segments.

The continuity loss function can be changed via `continuitylossfun`, defaulting to [`squaredl2loss`](@ref) and the penalty used for the continuity is weighted via a the `continuitylossweight`, which defaults to 100.
All segments can be initialized using `initialization` keyword argument. It defaults to `DefaultSimulationInitialization()`, i.e. the values are obtained by simulating the model passed in the [`Experiment`](@ref) with default parameters.
For information on other initializations, refer [Initialization Methods](@ref ms_init) section in the manual.

The optimization method will get the cost function as the sum of the individual previously mentioned costs. To change the optimization method, you can use the `optimizer` keyword argument.
The default optimization method is `Ipopt.Optimizer` which can be used from [`OptimizationMOI`](https://docs.sciml.ai/Optimization/stable/optimization_packages/mathoptinterface/).
We also have a convenience constructor [`IpoptOptimizer`](@ref) for defining it. The maximum number of iterations for the optimizer is given by `maxiters`, which is a required keyword argument.
Atleast one of `maxiters` or `maxtime` is required. If both of them are given, optimization stops after either of condition is first met.

## Keyword Arguments

  - `maxiters`: Maximum numbers of iterations when using the method with a time stepping optimizer.
  - `maxtime`: Maximum time for the optimization process.
  - `optimizer`: An Optimization.jl algorithm for the global optimization. Defaults to `Ipopt.Optimizer` which can be used from [`OptimizationMOI`](https://docs.sciml.ai/Optimization/stable/optimization_packages/mathoptinterface/).
  - `trajectories`: Required. This is number of segments that the whole time span is split into.
  - `ensemblealg`: Parallelization method for the ensemble problem. Defaults to `EnsembleThreads()`.
  - `continuitytype`: Type of continuity enforced. This can be `ModelStatesPenalty` which adds a term in the loss which corresponds to continuity loss given by `continuitylossfun` or it can be `ConstraintBased` where the continuity equations are added as constraints for the optimizer.
  - `continuitylossfun`: Loss function to compute continuity penalty. This is only applicable if `continuitytype` is `ModelStatesPenalty`.
  - `continuitylossweight`: Weight multiplied with the continuity loss term in the total loss. This is only applicable if `continuitytype` is `ModelStatesPenalty`.
  - `initialization`: Initialization method of the segments. This is only applicable if `continuitytype` is `ModelStatesPenalty`.
  - `kwargs`: Keyword arguments passed onto `Optimization.solve`.
"""
function MultipleShooting(;
        maxiters = nothing,
        maxtime = nothing,
        optimizer = IpoptOptimizer(; max_iter = maxiters, hessian_approximation = "exact"),
        ensemblealg = EnsembleThreads(),
        trajectories,
        continuitytype = ModelStatesPenalty,
        continuitylossfun = squaredl2loss,
        continuitylossweight = 100,
        initialization = DefaultSimulationInitialization(),
        kwargs...)
    maxiters, maxtime = check_maxiters_maxtime!(optimizer, maxiters, maxtime)
    if continuitytype <: ConstraintBased
        continuity = continuitytype()
    else
        continuity = continuitytype(continuitylossfun, continuitylossweight)
    end
    MultipleShooting(continuity,
        maxiters,
        maxtime,
        optimizer,
        ensemblealg,
        trajectories,
        initialization,
        kwargs)
end

"""
    StochGlobalOpt(; method = SingleShooting(optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters = 10_000), parallel_type = EnsembleSerial())

A stochastic global optimization method.

## Keyword Arguments

  - `parallel_type`: Choice of parallelism. Defaults to `EnsembleSerial()` or serial computation.
    For more information on ensemble parallelism types, see the [ensemble types from SciML](https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/#EnsembleAlgorithms)
  - `method`: a calibration algorithm to use during the runs.
    Defaults to `SingleShooting(optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters = 10_000)`. i.e. single shooting method with a black-box differential evolution method from BlackBoxOptimization.
    For more information on choosing optimizers, see [Optimization.jl](https://docs.sciml.ai/Optimization/stable/).
"""
struct StochGlobalOpt{P, M} <: AbstractMultiCalibrationAlgorithm
    parallel_type::P
    method::M
end

function StochGlobalOpt(; parallel_type = EnsembleSerial(),
        method = SingleShooting(optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited(),
            maxiters = 10_000),
        kwargs...)
    StochGlobalOpt(parallel_type, method)
end

get_optimizer(s::StochGlobalOpt) = get_optimizer(s.method)
get_maxiters(s::StochGlobalOpt) = get_maxiters(s.method)

"""
    MCMCOpt(; maxiters, discard_initial = max(Int(round(maxiters/2)), 1000), parallel_type = EnsembleSerial(), sampler = Turing.NUTS(0.65), hierarchical = false)

A Markov-Chain Monte Carlo (MCMC) method for solving the inverse problem.  The method approximates the posterior distribution of the model parameters given the data with a number of samples equal to `maxiters`.

## Keyword Arguments

  - `maxiters`: Required. Represents the number of posterior distribution samples.
  - `parallel_type`: Choice of parallelism. Defaults to `EnsembleSerial()` or serial computation. Determines the number of chains of samples that will be drawn.
    For more information on ensemble parallelism types, see the [ensemble types from SciML](https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/#EnsembleAlgorithms).
  - `discard_initial`: The number of initial samples to discard. Defaults to the maximum between half of the `maxiters` samples and 1000, as a general recommendation.
    This default is the same as the number of samples that `sampler` draws during its warm-up phase, when the default `NUTS` sampler is used.
  - `sampler`: The sampler used for the MCMC iterations. Defaults to `Turing.NUTS` for the No-U-Turn Sampler variant of Hamiltonian Monte Carlo.
    For more information on sampler algorithms, see [the Turing.jl documentation](https://turinglang.org/library/AdvancedHMC/stable/api/)
  - `hierarchical`: Defaults to `false`. Controls whether hierarchical MCMC inference will be used.
    If it is set to `true`, each [`Experiment`](@ref) object of the [`InverseProblem`](@ref) is treated as a separate subject and all subjects are assumed to come from the same population distribution.
    Population-level parameters are generated internally and their posterior is also inferred from data.
    If it is set to `false`, it is assumed that the same parameters are used for each subject and thus inference is performed on a lower-dimensional space.
"""
struct MCMCOpt{P, HIER, S} <: AbstractParametricBayesianAlgorithm
    maxiters::Int
    discard_initial::Int
    parallel_type::P
    sampler::S

    function MCMCOpt(;
            maxiters,
            discard_initial = max(Int(round(maxiters / 2)), 1000),
            parallel_type = EnsembleSerial(),
            sampler = Turing.NUTS(0.65),
            hierarchical = false)
        new{typeof(parallel_type), Val{hierarchical}, typeof(sampler)}(maxiters,
            discard_initial,
            parallel_type,
            sampler)
    end
end

"""
    KernelCollocation(; maxiters = nothing, maxtime = nothing,  optimizer = IpoptOptimizer(; max_iter = maxiters), kernel = EpanechnikovKernel(), bandwidth = nothing, cutoff = (0.0, 0.0), kwargs...)

A kernel based collocation method for solving the inverse problem. The derivatives of the data are estimated with a local linear regression using a kernel.
The loss is then generated from the error between the estimated derivatives and the time derivatives predicted by solving the ODE system.
Atleast one of `maxiters` or `maxtime` is required. If both of them are given, optimization stops after either of condition is first met.

## Keyword Arguments

  - `maxiters`: Maximum numbers of iterations when using the method with a time stepping optimizer.
  - `maxtime`: Maximum time for the optimization process.
  - `optimizer`: An Optimization.jl algorithm for the global optimization. Defaults to `Ipopt.Optimizer` which can be used from [`OptimizationMOI`](https://docs.sciml.ai/Optimization/stable/optimization_packages/mathoptinterface/).
  - `kernel`: The kernel used to estimate the derivatives of the data. For more information, see: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2631937/.
  - `bandwidth`: The bandwidth used for computing kernel weights for local kernel regression.
  - `cutoff`: The fraction of data in the beginning and end which is removed for loss computation.
  - `kwargs`: Keyword arguments passed onto `Optimization.solve`.
"""
struct KernelCollocation{I, T, O, K, B, C, KW} <: AbstractCollocationAlgorithm
    maxiters::I
    maxtime::T
    optimizer::O
    kernel::K
    bandwidth::B
    cutoff::C
    kwargs::KW
end

function KernelCollocation(; maxiters = nothing,
        maxtime = nothing,
        optimizer = IpoptOptimizer(; max_iter = maxiters),
        kernel = EpanechnikovKernel(),
        bandwidth = nothing,
        cutoff = (0.0, 0.0),
        kwargs...)
    maxiters, maxtime = check_maxiters_maxtime!(optimizer, maxiters, maxtime)
    KernelCollocation(maxiters, maxtime, optimizer, kernel, bandwidth, cutoff, kwargs)
end

"""
    SplineCollocation(; maxiters = nothing, maxtime = nothing, optimizer = IpoptOptimizer(; max_iter = maxiters), interp = Datainterpolations.CubicSpline, interp_args = (), cutoff = (0.0, 0.0), kwargs...)

A spline based collocation method for solving the inverse problem. The derivatives of the data are estimated by intepolating the data with a spline.
The loss is then generated from the error between the estimated derivatives and the time derivatives predicted by solving the ODE system.
Atleast one of `maxiters` or `maxtime` is required. If both of them are given, optimization stops after either of condition is first met.

## Keyword Arguments

  - `maxiters`: Maximum numbers of iterations when using the method with a time stepping optimizer.
  - `maxtime`: Maximum time for the optimization process.
  - `optimizer`: An Optimization.jl algorithm for the global optimization. Defaults to `Ipopt.Optimizer` which can be used from [`OptimizationMOI`](https://docs.sciml.ai/Optimization/stable/optimization_packages/mathoptinterface/).
  - `interp`: The interpolation function used to estimate the derivatives of the data. For more information, see: [DataInterpolations.jl](https://docs.sciml.ai/DataInterpolations/stable/).
  - `interp_args`: Extra arguments apart from states and time to be provided for methods like `BSplineInterpolation` or `BSplineApprox` of [DataInterpolations.jl](https://docs.sciml.ai/DataInterpolations/stable/).
  - `cutoff`: The fraction of data in the beginning and end which is removed for loss computation.
  - `kwargs`: Keyword arguments passed onto `Optimization.solve`.
"""
struct SplineCollocation{IT, T, O, I, IA, C, K} <: AbstractCollocationAlgorithm
    maxiters::IT
    maxtime::T
    optimizer::O
    interp::I
    interp_args::IA
    cutoff::C
    kwargs::K
end

function SplineCollocation(; maxiters = nothing,
        maxtime = nothing,
        optimizer = IpoptOptimizer(; max_iter = maxiters),
        interp = CubicSpline,
        interp_args = (),
        cutoff = (0.0, 0.0),
        kwargs...)
    maxiters, maxtime = check_maxiters_maxtime!(optimizer, maxiters, maxtime)
    SplineCollocation(maxiters, maxtime, optimizer, interp, interp_args, cutoff, kwargs)
end

"""
    NoiseRobustCollocation(; maxiters = nothing, maxtime = nothing, diff_iters, dx, α, optimizer = IpoptOptimizer(; max_iter = maxiters), tvdiff_kwargs = (), cutoff = (0.0, 0.0), kwargs...)

A noise robust collocation method for solving the inverse problem. The derivatives of the data are estimated using *total variational regularized numerical differentiation* (`tvdiff`).
The loss is then generated from the error between the estimated derivatives and the time derivatives predicted by the ODE solver.
Atleast one of `maxiters` or `maxtime` is required. If both of them are given, optimization stops after either of condition is first met.

For more information about the `tvdiff` implementation, see: [NoiseRobustDifferentiation.jl](https://github.com/adrhill/NoiseRobustDifferentiation.jl).

## Keyword Arguments

  - `maxiters`: Required. Maximum numbers of iterations when using the method with a time stepping optimizer.
  - `optimizer`: An Optimization.jl algorithm for the global optimization. Defaults to `Ipopt.Optimizer` which can be used from [`OptimizationMOI`](https://docs.sciml.ai/Optimization/stable/optimization_packages/mathoptinterface/).
  - `diff_iters`: Required. Number of iterations to run the main loop of the differentiation algorithm.
  - `α`: Required. Regularization parameter. Higher values increase regularization strength and improve conditioning.
  - `tvdiff_kwargs`: The keyword arguments passed on to `NoiseRobustDifferentiation.tvdiff` function for estimating derivatives.
  - `cutoff`: The fraction of data in the beginning and end which is removed for loss computation.
  - `kwargs`: Keyword arguments passed onto `Optimization.solve`.
"""
struct NoiseRobustCollocation{I, T, O, TK, C, K} <: AbstractCollocationAlgorithm
    maxiters::I
    maxtime::T
    optimizer::O
    diff_iters::Int
    α::Float64
    tvdiff_kwargs::TK
    cutoff::C
    kwargs::K
end

function NoiseRobustCollocation(; maxiters = nothing,
        maxtime = nothing,
        optimizer = IpoptOptimizer(; max_iter = maxiters),
        diff_iters::Int,
        α::Float64,
        tvdiff_kwargs = (),
        cutoff = (0.0, 0.0),
        kwargs...)
    maxiters, maxtime = check_maxiters_maxtime!(optimizer, maxiters, maxtime)
    NoiseRobustCollocation(maxiters,
        maxtime,
        optimizer,
        diff_iters,
        α,
        tvdiff_kwargs,
        cutoff,
        kwargs)
end
