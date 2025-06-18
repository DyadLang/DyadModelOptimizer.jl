using DyadModelOptimizer
using DyadData
using Test
using SteadyStateDiffEq
using ModelingToolkit
using ModelingToolkit: t_nounits as t
using OrdinaryDiffEqTsit5

include("../reactionsystem.jl")

@testset "TimeSeriesData" begin
    data = DyadDataset("reaction_system_data.csv", independent_var = "t",
        dependent_vars = ["u1", "u2", "u3"])
    model = reactionsystem()

    experiment = Experiment(data, model, indepvar = :t,
        depvars = [model.s1 => "u1", model.s1s2 => "u3", model.s2 => "u2"])
    @test DyadModelOptimizer.timespan(experiment) == (0.0, 1.0)

    @unpack k1, c1 = model

    @testset "Optimize all parameters" begin
        prob = InverseProblem(experiment, [k1 => (0, 5), c1 => (0, 5)])

        cost = objective(prob, SingleShooting(maxiters = 1))
        # the cost might not be exactly 0 due to the fact that the saved dataset was generated
        # with different package versions than what's currently running,
        # so floating point operations might be slightly different
        @test cost()≈0 atol=1e-9
    end
end

@testset "SteadyStateExperiments" begin
    model = reactionsystem()
    @unpack k1, c1 = model
    @variables s1(t) s2(t) s1s2(t)

    params = [c1 => 3.0]
    overrides = [s2 => 1.0, s1 => 2.0]
    data = DyadDataset("reaction_system_data.csv", independent_var = "t",
        dependent_vars = ["u1", "u2", "u3"])
    ss_data = DyadDatapoint(
        "reaction_system_data_end.csv", variable_names = ["s1(t)", "s1s2(t)", "s2(t)"])
    ex1 = SteadyStateExperiment(ss_data, model;
        alg = DynamicSS(Tsit5()),
        overrides = params)
    @test nameof(ex1) == "SteadyStateExperiment"

    ex2 = Experiment(data, model; overrides, dependency = ex1, indepvar = :t,
        depvars = [model.s1 => "u1", model.s1s2 => "u3", model.s2 => "u2"])
    ex3 = Experiment(data, model; overrides, indepvar = :t,
        depvars = [model.s1 => "u1", model.s1s2 => "u3", model.s2 => "u2"])

    prob = InverseProblem([ex1, ex2, ex3],
        [s2 => (1.5, 3), k1 => (0, 5)])
    cost = objective(prob, SingleShooting(maxiters = 1))

    @test !iszero(cost())

    @testset "Iteration" begin
        trials = get_experiments(prob)
        @test length(trials) == 3

        @test trials[1] === ex1
        @test trials[2] === ex2
        @test trials[3] === ex3

        @test first(trials) === ex1
        @test last(trials) === ex3

        @test trials[begin] === ex1
        @test trials[end] === ex3
    end

    @testset "Automated steady state computation" begin
        x = DyadModelOptimizer.initial_state(Any, prob)
        sol1 = simulate(ex1, prob)
        sol2 = simulate(ex2, prob)
        ss_u0 = DyadModelOptimizer.has_dependency(ex2) ? sol1[[s1s2]] : Val(:default)
        sol2_ss = simulate(ex2, prob, x, ss_u0)
        ss_u0 = DyadModelOptimizer.has_dependency(ex3) ? sol1[[s1s2]] : Val(:default)
        sol3 = simulate(ex3, prob, x, ss_u0)

        # only s1s2 will be obtained from the steady state
        # because the other states are fixed by the experiment
        @test sol2[s1s2, 1] == sol1[s1s2]
        @test sol2[s1, 1] == 2.0
        @test sol2[s2, 1] == 1.0
        @test sol2.u == sol2_ss.u
        # experiment3 doesn't use the steady state
        @test sol3[s1s2, 1] ≠ sol1[s1s2]
    end
end
