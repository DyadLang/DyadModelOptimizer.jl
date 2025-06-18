module DyadModelOptimizer

export InverseProblem,
       Experiment, DiscreteExperiment, SteadyStateExperiment,
       DesignConfiguration,
       IndependentExperiments, # TODO: remove this one too
       DataShooting, MultipleShooting, SingleShooting, JuliaHubJob,
       ConstraintBased, ModelStatesPenalty,
       DataInitialization, DefaultSimulationInitialization, RandomInitialization,
       StochGlobalOpt,
#    MCMCOpt, TransformedBeta,
       DDS, RefWeights,
       CalibrationResult, ParameterEnsemble, MCMCResult,
       calibrate, objective, IpoptOptimizer,
       DiscreteFixedGainPEM,
       parametric_uq, solve_ensemble, subsample, get_sampler,
       cost_contribution,
       virtual_experiment,
       squaredl2loss, l2loss, meansquaredl2loss, root_meansquaredl2loss,
       zscore_meansquaredl2loss, norm_meansquaredl2loss, zscore_meanabsl1loss, ARMLoss,
       import_ps, import_res,
#    import_petab, export_petab, create_petab_template,
       EpanechnikovKernel, UniformKernel, TriangularKernel, QuarticKernel, TriweightKernel,
       TricubeKernel, GaussianKernel, CosineKernel, LogisticKernel, SigmoidKernel,
       SilvermanKernel,
       KernelCollocation, SplineCollocation, NoiseRobustCollocation, CollocationData,
       convergenceplot, plot_shooting_segments,
       confidenceplot, confidence_plot_shooting_segments,
       lowerbound, upperbound,
       get_model, get_experiments, initial_guess, remake_experiments

# SBMLToolkit re-exports
# export readSBML, DefaultImporter, ReactionSystemImporter, ODESystemImporter

# DyadInterface re-exports
export simulate

# Dyad related exports
export CalibrationAnalysisSpec, AbstractCalibrationAnalysisSpec, CalibrationAnalysis

# SciMLBase re-exports
export remake

using Distributed: pmap, workers, CachingPool, nworkers, RemoteChannel
using DyadInterface: DyadInterface, simulate, AbstractAnalysisSpec,
                     ODEProblemConfig, get_simplified_model,
                     AbstractAnalysisSolution,
                     AnalysisSolutionMetadata, ArtifactType,
                     ArtifactMetadata,
                     PlotlyVisualizationSpec, Attribute
using DyadData: DyadDataset, DyadDatapoint, build_dataframe
using LinearAlgebra: norm, normalize, Diagonal, mul!, dot, det
using Printf: @sprintf
# SciML
using CommonSolve: CommonSolve, solve, solve!, init
using SciMLBase: SciMLBase, remake,
                 ODEProblem, DiscreteProblem, ODEFunction, DiscreteFunction,
                 OptimizationProblem, OptimizationFunction, ODESolution,
                 EnsembleProblem, EnsembleSolution, EnsembleSummary, SteadyStateProblem,
                 NonlinearProblem, NonlinearLeastSquaresProblem, NonlinearFunction,
                 AbstractNonlinearSolution, DEProblem, AbstractSciMLSolution,
                 FullSpecialize,
                 EnsembleSerial, EnsembleDistributed, EnsembleThreads,
                 parameterless_type,
                 terminate!, reinit!, auto_dt_reset!, u_modified!, add_tstop!,
                 DiscreteCallback, CallbackSet,
                 tmap,
                 allowsbounds,
                 ReturnCode, successful_retcode, convert,
                 AbstractDEAlgorithm
using DiffEqBase: prepare_alg
using SymbolicIndexingInterface: SymbolicIndexingInterface,
                                 symbolic_type, NotSymbolic, getname,
                                 variable_index, parameter_index,
                                 parameter_symbols, variable_symbols,
                                 all_variable_symbols,
                                 state_values, parameter_values, default_values,
                                 BatchedInterface, setsym_oop, setsym, getsym,
                                 is_variable, is_parameter, is_observed
using SciMLStructures: SciMLStructures, Tunable
using ModelingToolkit: ModelingToolkit, AbstractTimeDependentSystem,
                       ODESystem, ConstraintsSystem, NonlinearSystem,
                       defaults, guesses, parameters, unknowns, observed, equations,
                       getbounds, istunable, tunable_parameters, Initial,
                       generate_function,
                       structural_simplify, complete,
                       getdescription,
                       @parameters, @named, toparam, calculate_jacobian, isautonomous,
                       extend, get_ps, get_parent, asgraph,
                       varvar_dependencies, variable_dependencies,
                       constraints, full_parameters, diff_equations,
                       parse_variable, iscomplete
using ModelingToolkitStandardLibrary: ModelingToolkitStandardLibrary
using Symbolics: Symbolics, Num, Equation, Differential, substitute, value,
                 get_variables, VariableDefaultValue, wrap, unwrap, VariableSource
using SymbolicUtils: operation, setmetadata, arguments, iscall, Sym
using Expronicon: Substitute
using RuntimeGeneratedFunctions: RuntimeGeneratedFunctions
using DiffEqCallbacks: DiffEqCallbacks, PresetTimeCallback
using NonlinearSolve: FastShortcutNonlinearPolyalg
# Arrays
using LazyArrays: LazyArray, @~
using ArraysOfArrays: deepview
using ComponentArrays: ComponentArray
# Optimization state vector
using RecursiveArrayTools: ArrayPartition, DiffEqArray, VectorOfArray, AbstractDiffEqArray
# bounds
using StructArrays: StructArray
# These deps are needed for providing defaults
using ADTypes: AutoForwardDiff
using OrdinaryDiffEqDefault: DefaultODEAlgorithm
using OrdinaryDiffEqCore: BrownFullBasicInit
using OrdinaryDiffEqFunctionMap: FunctionMap
using OptimizationBBO: BBO_adaptive_de_rand_1_bin_radiuslimited
using OptimizationMOI: MOI
using Ipopt: Ipopt, Optimizer
using DataInterpolations: CubicSpline
# Optimization
using Optimization: Optimization, OptimizationState
using Distributions: Distributions
# Progress
using ProgressLogging: ProgressLogging, Progress, ProgressLevel
# Logging
using Logging: Debug, Warn, disable_logging
# Pretty print
using PrettyTables: pretty_table
using Dates: canonicalize, Minute, Second, Millisecond, Microsecond
# Plotting
using RecipesBase: RecipesBase, @recipe, @series, @userplot, plot
# Data & stats
using Tables: Tables, AbstractRow
using DataFrames: DataFrames,
                  DataFrame, DataFrameRow,
                  rename, rename!, Not, unstack, select, select!, nrow, nonunique
using CSV: CSV
using JSONSchema: Schema, validate
using StructTypes: StructTypes, Struct
using RelocatableFolders: @path
# Statistics
using Statistics: mean, std
using StatsBase: sample, Weights, ecdf, fit, Histogram, binindex, quantile
using Random: Random, AbstractRNG
using QuasiMonteCarlo: QuasiMonteCarlo, LatinHypercubeSample
# UUID for each experiment
using UUIDs: uuid4, UUID
using Setfield: @set, @set!
# Calibrate deps
using DataInterpolations: derivative
using NoiseRobustDifferentiation: tvdiff
# AD
using ChainRulesCore: ChainRulesCore, rrule_via_ad, RuleConfig, NoTangent,
                      @ignore_derivatives
using GenericLinearAlgebra: svd # necessary when doing AD through SSRootfind steady state solver
using ForwardDiff: ForwardDiff
using ChainRulesCore: ChainRulesCore, rrule_via_ad, RuleConfig, NoTangent,
                      @ignore_derivatives
# Petab import
# using SBMLToolkit: create_var,
# SBMLToolkit, readSBML,
# DefaultImporter, ReactionSystemImporter, ODESystemImporter
# using SBML: SBML
using NumericalIntegration
import Preferences

using PkgAuthentication: PkgAuthentication
using JuliaHubData: JuliaHubData
import JuliaHub
using Tar: Tar
using TOML: TOML
using Serialization: serialize

using PrecompileTools: @setup_workload, @compile_workload

RuntimeGeneratedFunctions.init(@__MODULE__)

abstract type AbstractDesignAnalysis end
abstract type AbstractInverseProblem{E} <: AbstractDesignAnalysis end

include("algorithms.jl")
include("results.jl")
include("model_transformations.jl")
include("experiments/experiment.jl")
include("inverse_prob.jl")
include("config/config.jl")
include("objectives.jl")
include("calibrate/calibrate.jl")
include("parametric_uq.jl")
include("subsample.jl")
include("utils.jl")
include("remake.jl")
include("plot_recipes.jl")
include("io.jl")
include("dyad/dyad_interface.jl")
# include("petab/petab.jl")
# precompile should be always last
include("precompile.jl")

const SOLVE_ERR_WARN = Preferences.@load_preference("SolveErrorWarning", true)
const SOLVE_FAIL_WARN = Preferences.@load_preference("SolveFailureWarning", true)
const UNEXPECTED_EXCEPTION = Union{InterruptException, MethodError, ErrorException,
    ArgumentError, UndefKeywordError, UndefRefError, UndefVarError, BoundsError,
    KeyError, TypeError}

end
