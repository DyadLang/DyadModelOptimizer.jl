using Test
using DataFrames
using DyadModelOptimizer
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEqTsit5
using Plots
using OptimizationBBO

include("reactionsystem.jl")

@testset "Plottting level of confidence for good fit" begin
    @testset "Experiment1" begin
        model = reactionsystem()
        @unpack s1, s1s2, s2, k1, c1 = model
        experiment_ps = [k1 => 1.5,
            c1 => 2.5
        ]

        data = generate_data(model, (0.0, 1.0), 50, params = [k1 => 1.5, c1 => 2.5])
        experiment = Experiment(data, model, tspan = (0.0, 1.0))
        prob = InverseProblem(experiment, [k1 => (0, 5), c1 => (0, 5)])

        ps = parametric_uq(prob,
            StochGlobalOpt(method = SingleShooting(maxiters = 10,
                optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited())),
            sample_size = 50)
        @test_nowarn confidenceplot(experiment,
            ps,
            confidence = 0.6,
            legend = true,
            show_data = true)
        @test_nowarn confidenceplot(experiment,
            ps,
            confidence = 0.6,
            legend = true,
            show_data = true,
            states = [s1s2, s1])
        @test_nowarn confidenceplot(experiment,
            ps,
            confidence = 0.6,
            legend = true,
            show_data = true,
            states = [s1])
    end
    @testset "Experiment2" begin
        states = @variables S(t) I(t) R(t)
        ps = @parameters β γ

        eqs = [D(S) ~ -β * S * I,
            D(I) ~ β * S * I - γ * I,
            D(R) ~ γ * I]

        defs = Dict(S => 0.99,
            I => 0.01,
            R => 0.0,
            β => 0.1,
            γ => 0.05)

        model = complete(ODESystem(eqs, t, states, ps; defaults = defs, name = :model))

        experiment_inits = [
            I => 0.01,
            R => 0.0
        ]

        experiment_ps = [β => 0.3,
            γ => 0.06
        ]
        experiment_tspan = (0.0, 100.0)

        experiment_n = 100
        experiment_saveat = range(experiment_tspan[1],
            experiment_tspan[2],
            length = experiment_n)
        experiment_prob = ODEProblem(model, experiment_inits, experiment_tspan,
            experiment_ps,
            saveat = experiment_saveat)
        experiment_sol = solve(experiment_prob, Tsit5(), reltol = 1e-8, abstol = 1e-8)
        experiment_data = DataFrame(experiment_sol)
        experiment = Experiment(experiment_data, model;
            overrides = [experiment_inits; experiment_ps],
            tspan = experiment_tspan,
            alg = Tsit5()            # reltol=1e-8, abstol=1e-8
        )

        prob = InverseProblem(experiment,
            [S => (0.5, 1.5), β => (0.1, 0.4), γ => (0.04, 0.08)])
        ps = parametric_uq(prob,
            StochGlobalOpt(method = SingleShooting(maxiters = 20,
                optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited())),
            sample_size = 5_000)

        @test_nowarn confidenceplot(experiment,
            ps,
            confidence = 0.1,
            show_data = true,
            legend = true)
        @test_nowarn confidenceplot(experiment,
            ps,
            confidence = 0.1,
            show_data = true,
            legend = true,
            states = [I, R])
        @test_nowarn confidenceplot(experiment,
            ps,
            confidence = 0.1,
            show_data = true,
            legend = true,
            states = [I, R, S])
    end
end
