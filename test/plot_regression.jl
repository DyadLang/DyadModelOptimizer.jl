using DyadModelOptimizer
using OrdinaryDiffEqTsit5, OrdinaryDiffEqRosenbrock
using ModelingToolkit
using ModelingToolkit: t_nounits as t
using Random
using Plots
using Test
using VisualRegressionTests
using JSMOPlotReferenceImages
using DataInterpolations
using SteadyStateDiffEq
using OptimizationBBO

include("reactionsystem.jl")

const RNG_SEED = 1234
const N_population = 50
const uq_alg = StochGlobalOpt(method = SingleShooting(maxiters = 10,
    optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited()))

# Derived from https://github.com/JuliaPlots/Plots.jl, MIT licensed, see repository for details
# Using a global const Vector{::PlotExample} is based on the way Plots.jl does its visual regression testing :
# https://github.com/JuliaPlots/Plots.jl/blob/b0b99d280de2022f69ee5817bc24899cc92e20ad/src/examples.jl
struct PlotExample
    header::AbstractString
    desc::AbstractString
    plot::Function
end

function generate_experiment(model; tspan = (0.0, 1.0), kwargs...)
    data = generate_data(model, tspan)
    Experiment(data, model; tspan, alg = Tsit5(), name = "Trial", kwargs...) # alg is hardcoded for reproducibility
end

function generate_param_ic_ss()
    @variables s2(t)
    @parameters c1
    [c1 => (0.0, 5.0), Initial(s2) => (1.5, 3.0)]
end

function generate_param_ss()
    @variables s2(t)
    [Initial(s2) => (1.5, 3.0)]
end

function generate_ic_ss()
    @parameters c1
    [c1 => (0.0, 5.0)]
end

const _examples = [
    PlotExample("plot(experiment::AbstractExperiment, prob::InverseProblem)",
        "experiment with default kwargs, prob with 1 parameter and 1 initial condition in the search space",
        function ()
            # 1
            model = extended_reactionsystem()
            experiment = generate_experiment(model)
            ss = generate_param_ic_ss()
            prob = InverseProblem([experiment], ss)

            plot(experiment, prob)
        end),
    PlotExample(
        "plot(experiment::AbstractExperiment, prob::InverseProblem; show_data = true)",
        "experiment with default kwargs, prob with 1 parameter and 1 initial condition in the search space",
        function ()
            # 2
            model = extended_reactionsystem()
            experiment = generate_experiment(model)
            ss = generate_param_ic_ss()
            prob = InverseProblem([experiment], ss)

            plot(experiment, prob; show_data = true)
        end),
    missing, # 3
    PlotExample(
        "plot(experiment::AbstractExperiment, prob::InverseProblem; states = [s1, s2])",
        """
        experiment with default kwargs, prob with 1 parameter and 1 initial condition in the search space.
        Plotting a subset of all saved states of experiment.
        """,
        function ()
            # 4
            @variables s1(t) s2(t)
            model = extended_reactionsystem()
            experiment = generate_experiment(model)
            ss = generate_param_ic_ss()
            prob = InverseProblem([experiment], ss)

            plot(experiment, prob; states = [s1, s2])
        end),
    PlotExample(
        "plot(param_ens::ParameterEnsemble, experiment::AbstractExperiment; summary = false)",
        """
        param_ens with 1 parameter and 1 initial condition in the search space, experiment with default kwargs.
        Plotting all trajectories from param_ens, not summaries.
        """,
        function ()
            # 5
            model = extended_reactionsystem()
            experiment = generate_experiment(model)
            ss = generate_param_ic_ss()
            prob = InverseProblem([experiment], ss)

            Random.seed!(RNG_SEED)
            param_ens = parametric_uq(prob, uq_alg, sample_size = N_population)

            plot(param_ens, experiment; summary = false)
        end),
    PlotExample(
        "plot(param_ens::ParameterEnsemble, experiment::AbstractExperiment; summary = true)",
        """
        param_ens with 1 parameter and 1 initial condition in the search space, experiment with default kwargs.
        Plotting trajectory summaries from param_ens with quantiles=[0.05, 0.95].
        """,
        function ()
            # 6
            model = extended_reactionsystem()
            experiment = generate_experiment(model)
            ss = generate_param_ic_ss()
            prob = InverseProblem(experiment, ss)

            Random.seed!(RNG_SEED)
            param_ens = parametric_uq(prob, uq_alg, sample_size = N_population)

            plot(param_ens, experiment; summary = true)
        end),
    PlotExample(
        "plot(param_ens::ParameterEnsemble, experiment::AbstractExperiment; summary = true, show_data = true)",
        """
        param_ens with 1 parameter and 1 initial condition in the search space, experiment with default kwargs.
        Plotting trajectory summaries from param_ens with quantiles=[0.05, 0.95] and data.
        """,
        function ()
            # 7
            model = extended_reactionsystem()
            experiment = generate_experiment(model)
            ss = generate_param_ic_ss()
            prob = InverseProblem([experiment], ss)

            Random.seed!(RNG_SEED)
            param_ens = parametric_uq(prob, uq_alg, sample_size = N_population)

            plot(param_ens, experiment; summary = true, show_data = true)
        end),
    missing, missing, # 8, 9
    PlotExample(
        "plot(param_ens::ParameterEnsemble, experiment::AbstractExperiment; summary = true, states = [s1, s2])",
        """
        param_ens with 1 parameter and 1 initial condition in the search space, experiment with default kwargs.
        Plotting trajectory summaries from param_ens with quantiles=[0.05, 0.95]
        and a subset of all saved states of experiment.
        """,
        function ()
            # 10
            model = extended_reactionsystem()
            @unpack s1, s2 = model
            experiment = generate_experiment(model)
            ss = generate_param_ic_ss()
            prob = InverseProblem([experiment], ss)

            Random.seed!(RNG_SEED)
            param_ens = parametric_uq(prob, uq_alg, sample_size = N_population)

            plot(param_ens, experiment; summary = true, states = [s1, s2])
        end),
    missing, # 11
    PlotExample("plot(param_ens::ParameterEnsemble, prob::InverseProblem; summary = true)",
        """
        prob with 1 experiment and search space with 1 parameter and 1 initial condition.
        Plotting trajectory summaries from param_ens with quantiles=[0.05, 0.95].
        """,
        function ()
            # 12
            model = extended_reactionsystem()
            experiment = generate_experiment(model)
            ss = generate_param_ic_ss()
            prob = InverseProblem([experiment], ss)

            Random.seed!(RNG_SEED)
            param_ens = parametric_uq(prob, uq_alg, sample_size = N_population)

            plot(param_ens, prob; summary = true)
        end),
    PlotExample("plot(param_ens::ParameterEnsemble, prob::InverseProblem; summary = true)",
        """
        prob with 2 independent experiments and search space with 1 parameter and 1 initial condition.
        experiments have different tspan and names, otherwise default kwargs.
        Plotting trajectory summaries from param_ens with quantiles=[0.05, 0.95].
        """,
        function ()
            # 13
            model = extended_reactionsystem()
            experiments = [
                generate_experiment(model; tspan = (0, 1), name = "Trial 1"),
                generate_experiment(model; tspan = (0, 5), name = "Trial 2")
            ]
            ss = generate_param_ic_ss()
            prob = InverseProblem(experiments, ss)

            Random.seed!(RNG_SEED)
            param_ens = parametric_uq(prob, uq_alg, sample_size = N_population)

            plot(param_ens, prob; summary = true)
        end),
    PlotExample("plot(param_ens::ParameterEnsemble, prob::InverseProblem; summary = true)",
        """
        prob with 2 independent experiments and search space with 1 parameter and 1 initial condition.
        experiments have different tspan, saved states and names otherwise default kwargs.
        Plotting trajectory summaries from param_ens with quantiles=[0.05, 0.95].
        """,
        function ()
            # 14
            model = extended_reactionsystem()
            @unpack s1, s2 = model
            experiments = [
                generate_experiment(model;
                    tspan = (0, 1),
                    depvars = [s1, s2],
                    name = "Trial 1"),
                generate_experiment(model; tspan = (0, 5), name = "Trial 2")
            ]
            ss = generate_param_ic_ss()
            prob = InverseProblem(experiments, ss)

            Random.seed!(RNG_SEED)
            param_ens = parametric_uq(prob, uq_alg, sample_size = N_population)

            plot(param_ens, prob; summary = true)
        end),
    PlotExample(
        "plot(param_ens::ParameterEnsemble, prob::InverseProblem; summary=true, show_data = true)",
        """
        prob with 2 independent experiments and search space with 1 parameter and 1 initial condition.
        experiments have different tspan, saved states and names otherwise default kwargs.
        Plotting trajectory summaries from param_ens with quantiles=[0.05, 0.95] and experiment data.
        """,
        function ()
            # 15
            model = extended_reactionsystem()
            @unpack s1, s2 = model
            experiments = [
                generate_experiment(model;
                    tspan = (0, 1),
                    depvars = [s1, s2],
                    name = "Trial 1"),
                generate_experiment(model; tspan = (0, 5), name = "Trial 2")
            ]
            ss = generate_param_ic_ss()
            prob = InverseProblem(experiments, ss)

            Random.seed!(RNG_SEED)
            param_ens = parametric_uq(prob, uq_alg, sample_size = N_population)

            plot(param_ens, prob; summary = true, show_data = true)
        end),
    missing, # 16
    PlotExample(
        "plot(param_ens::ParameterEnsemble, prob::InverseProblem; summary = true, layout = (1,2))",
        """
        prob with 2 independent experiments and search space with 1 parameter and 1 initial condition.
        experiments have different tspan, saved states and names otherwise default kwargs.
        Plotting trajectory summaries from param_ens with quantiles=[0.05, 0.95].
        Subplots are arranged on a single row with 2 columns.
        """,
        function ()
            # 17
            model = extended_reactionsystem()
            @unpack s1, s2 = model
            experiments = [
                generate_experiment(model;
                    tspan = (0, 1),
                    depvars = [s1, s2],
                    name = "Trial 1"),
                generate_experiment(model; tspan = (0, 5), name = "Trial 2")
            ]
            ss = generate_param_ic_ss()
            prob = InverseProblem(experiments, ss)

            Random.seed!(RNG_SEED)
            param_ens = parametric_uq(prob, uq_alg, sample_size = N_population)

            plot(param_ens, prob; summary = true, layout = (1, 2))
        end),
    PlotExample("plot(experiment::AbstractExperiment, prob::InverseProblem)",
        """
        experiment with one unknown in depvars, prob with 1 parameter and 1 initial condition in the search space.
        Plotting all saved states of the experiment.
        """,
        function ()
            # 18
            model = extended_reactionsystem()
            @unpack s1, s2 = model
            experiment = generate_experiment(model, depvars = [s1])
            ss = generate_param_ic_ss()
            prob = InverseProblem(experiment, ss)

            plot(experiment, prob; show_data = true)
        end),
    PlotExample(
        "plot(experiment::SteadyStateExperiment, prob::InverseProblem, r::CalibrationResult, states)",
        """
        steady state experiment with all states saved, forward ordering of states in plot function
        """,
        function ()
            # 19
            Random.seed!(RNG_SEED)
            model = reactionsystem()
            @unpack k1, c1 = model
            @unpack s1, s2, s1s2 = model

            params = [c1 => 3.0]
            u0 = [s2 => 1.0, s1 => 2.0]
            data = generate_data(model; params)
            ss_data = data[end, 2:end]
            t1 = SteadyStateExperiment(ss_data, model;
                alg = DynamicSS(Rodas5()),
                prob_kwargs = (jac = true,),
                overrides = params)

            prob = InverseProblem(t1,
                [Initial(s2) => (1.5, 3), k1 => (0, 5)])
            r = calibrate(prob,
                SingleShooting(maxiters = 10^4,
                    optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited()))

            plot(t1, prob, r, legend = true, show_data = true, states = [s1, s1s2])
        end),
    PlotExample(
        "plot(experiment::SteadyStateExperiment, prob::InverseProblem, r::CalibrationResult, states)",
        """
        steady state experiment with all states saved, reverse ordering of states in plot function
        """,
        function ()
            # 20
            Random.seed!(RNG_SEED)
            model = reactionsystem()
            @unpack k1, c1 = model
            @unpack s1, s2, s1s2 = model

            params = [c1 => 3.0]
            u0 = [s2 => 1.0, s1 => 2.0]
            data = generate_data(model; params)
            ss_data = data[end, 2:end]
            t1 = SteadyStateExperiment(ss_data, model;
                alg = DynamicSS(Rodas5()),
                prob_kwargs = (jac = true,),
                overrides = params)

            prob = InverseProblem(t1,
                [Initial(s2) => (1.5, 3), k1 => (0, 5)])
            r = calibrate(prob,
                SingleShooting(maxiters = 10^4,
                    optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited()))

            plot(t1, prob, r, legend = true, show_data = true, states = [s2, s1])
        end),
    PlotExample(
        "plot(experiment::SteadyStateExperiment, prob::InverseProblem, r::CalibrationResult, states)",
        """
        steady state experiment with 2 unknowns in depvars, forward ordering of states in depvars, reverse ordering of states in plot function
        """,
        function ()
            # 21
            Random.seed!(RNG_SEED)
            model = reactionsystem()
            @unpack k1, c1 = model
            @unpack s1, s2, s1s2 = model

            params = [c1 => 3.0]
            u0 = [s2 => 1.0, s1 => 2.0]
            data = generate_data(model; params)
            ss_data = data[end, 2:end]
            t1 = SteadyStateExperiment(ss_data, model;
                alg = DynamicSS(Rodas5()),
                prob_kwargs = (jac = true,),
                overrides = params,
                depvars = [s1s2, s2])

            prob = InverseProblem(t1,
                [Initial(s2) => (1.5, 3), k1 => (0, 5)])
            r = calibrate(prob,
                SingleShooting(maxiters = 10^4,
                    optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited()))

            plot(t1, prob, r, legend = :outerright, show_data = true, states = [s1s2, s2])
        end),
    PlotExample(
        "plot(experiment::SteadyStateExperiment, prob::InverseProblem, r::CalibrationResult, states)",
        """
        steady state experiment with 2 states in depvars, forward ordering of states in depvars, reverse ordering of states in plot function
        """,
        function ()
            # 22
            Random.seed!(RNG_SEED)
            model = reactionsystem()
            @unpack k1, c1 = model
            @unpack s1, s2, s1s2 = model

            params = [c1 => 3.0]
            u0 = [s2 => 1.0, s1 => 2.0]
            data = generate_data(model; params)
            ss_data = data[end, 2:end]
            t1 = SteadyStateExperiment(ss_data, model;
                alg = DynamicSS(Rodas5()),
                prob_kwargs = (jac = true,),
                overrides = params,
                depvars = [s1s2, s2])

            prob = InverseProblem(t1,
                [Initial(s2) => (1.5, 3), k1 => (0, 5)])
            r = calibrate(prob,
                SingleShooting(maxiters = 10^4,
                    optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited()))

            plot(t1, prob, r, legend = :outerright, show_data = true, states = [s2, s1s2])
        end),
    PlotExample("plot(collocation_data::CollocationData; vars = \"states\")",
        "Plotting Collocated states data when alg, experiment, invprob, and parameters is also passed as a vector of symbol value pairs into CollocationData constructor",
        function ()
            # 23
            Random.seed!(RNG_SEED)
            model = reactionsystem()
            n_save = 300
            @unpack c1 = model
            true_c = 2.5
            data = generate_noisy_data(model,
                (0.0, 1.0),
                n_save;
                noise_std = 0.05,
                params = [c1 => true_c])
            experiment = Experiment(data, model, tspan = (0.0, 1.0))
            invprob = InverseProblem([experiment], [c1 => (0, 5)])
            alg = KernelCollocation(maxiters = 10^3, cutoff = (0.1, 0.0), bandwidth = 0.4)
            collocation_data = CollocationData(alg, experiment, invprob, [c1 => true_c])
            plot(collocation_data; vars = "states")
        end),
    PlotExample("plot(collocation_data::CollocationData; vars = \"derivatives\")",
        "Plotting Collocated derivatives data when alg, experiment, invprob, and parameters is also passed as a vector of symbol value pairs into CollocationData constructor",
        function ()
            # 24
            Random.seed!(RNG_SEED)
            model = reactionsystem()
            n_save = 300
            @unpack c1 = model
            true_c = 2.5
            data = generate_noisy_data(model,
                (0.0, 1.0),
                n_save;
                noise_std = 0.05,
                params = [c1 => true_c])
            experiment = Experiment(data, model, tspan = (0.0, 1.0))
            invprob = InverseProblem([experiment], [c1 => (0, 5)])
            alg = KernelCollocation(maxiters = 10^3, cutoff = (0.1, 0.0), bandwidth = 0.4)
            collocation_data = CollocationData(alg, experiment, invprob, [c1 => true_c])
            plot(collocation_data; vars = "derivatives")
        end),
    PlotExample("plot(collocation_data::CollocationData; vars = \"states\")",
        "Plotting Collocated states data when alg, experiment, invprob, and parameters is also passed as a vector of numbers into CollocationData constructor",
        function ()
            # 25
            model = reactionsystem()
            n_save = 300
            @unpack c1 = model
            true_c = 2.5
            data = generate_data(model, (0.0, 1.0), n_save; params = [c1 => true_c])
            experiment = Experiment(data, model, tspan = (0.0, 1.0))
            invprob = InverseProblem([experiment], [c1 => (0, 5)])
            alg = SplineCollocation(maxiters = 10^3, interp = CubicSpline)
            collocation_data = CollocationData(alg, experiment, invprob, [true_c])
            plot(collocation_data; vars = "states")
        end),
    PlotExample("plot(collocation_data::CollocationData; vars = \"derivatives\")",
        "Plotting Collocated derivatives data when alg, experiment, invprob, and parameters is also passed as a vector of numbers into CollocationData constructor",
        function ()
            # 26
            model = reactionsystem()
            n_save = 300
            @unpack c1 = model
            true_c = 2.5
            data = generate_data(model, (0.0, 1.0), n_save; params = [c1 => true_c])
            experiment = Experiment(data, model, tspan = (0.0, 1.0))
            invprob = InverseProblem([experiment], [c1 => (0, 5)])
            alg = SplineCollocation(maxiters = 10^3, interp = CubicSpline)
            collocation_data = CollocationData(alg, experiment, invprob, [true_c])
            plot(collocation_data; vars = "derivatives")
        end),
    PlotExample("plot(collocation_data::CollocationData; vars = \"states\")",
        "Plotting Collocated states data when only passing alg and experiment into CollocationData constructor",
        function ()
            # 27
            Random.seed!(RNG_SEED)
            model = reactionsystem()
            n_save = 300
            @unpack c1 = model
            true_c = 2.5
            data = generate_noisy_data(model,
                (0.0, 1.0),
                n_save;
                noise_std = 0.05,
                params = [c1 => true_c])
            experiment = Experiment(data, model, tspan = (0.0, 1.0))
            invprob = InverseProblem([experiment], [c1 => (0, 5)])
            alg = NoiseRobustCollocation(maxiters = 10^3,
                diff_iters = 500,
                α = 5.0,
                cutoff = (0.3, 0.1),
                tvdiff_kwargs = (diff_kernel = "square", scale = "large"))
            collocation_data = CollocationData(alg, experiment, invprob)
            plot(collocation_data; vars = "states")
        end),
    PlotExample("plot(collocation_data::CollocationData; vars = \"derivatives\")",
        "Plotting Collocated derivatives data when only passing alg and experiment",
        function ()
            # 28
            Random.seed!(RNG_SEED)
            model = reactionsystem()
            n_save = 300
            @unpack c1 = model
            true_c = 2.5
            data = generate_noisy_data(model,
                (0.0, 1.0),
                n_save;
                noise_std = 0.05,
                params = [c1 => true_c])
            experiment = Experiment(data, model, tspan = (0.0, 1.0))
            invprob = InverseProblem([experiment], [c1 => (0, 5)])
            alg = NoiseRobustCollocation(maxiters = 10^3,
                diff_iters = 500,
                α = 5.0,
                cutoff = (0.3, 0.1),
                tvdiff_kwargs = (diff_kernel = "square", scale = "large"))
            collocation_data = CollocationData(alg, experiment, invprob)
            plot(collocation_data; vars = "derivatives")
        end)
]

function generate_images(examples)
    for (i, ex) in enumerate(examples)
        ismissing(ex) && continue # skip plot with dosing
        p = ex.plot()
        savefig(p, "ref$i.png")
    end
end

function test_reference_plots(examples)
    default_version = v"11.1.0"
    @testset "Plot references" begin
        for (i, ex) in enumerate(examples)
            ismissing(ex) && continue # skip plot with dosing
            @debug "Testing ref$i"
            fn = reference_file(i, default_version)
            @plottest ex.plot fn popup=false tol=0
        end
    end
end

ENV["GKSwstype"] = "100" # avoid any interactive output
test_reference_plots(_examples)
