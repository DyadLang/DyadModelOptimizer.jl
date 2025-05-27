using DyadModelOptimizer
using OrdinaryDiffEqTsit5
using ModelingToolkit
using SteadyStateDiffEq
using SciMLBase: successful_retcode
using DataFrames
using Test
using JET

include("../reactionsystem.jl")

@testset "IndependentExperiments constructors" begin
    model = reactionsystem()
    @unpack c1 = model
    overrides = [c1 => 3.0]
    experiment1 = Experiment(nothing, model, tspan = (0.0, 1.0))
    experiment2 = Experiment(nothing, model; overrides, tspan = (0.0, 2.0))

    @test IndependentExperiments(experiment1, experiment2) isa
          IndependentExperiments{<:Experiment}
    @test IndependentExperiments([experiment1, experiment2]) isa
          IndependentExperiments{<:Experiment}

    tv = []
    push!(tv, experiment1)
    push!(tv, experiment2)

    @test IndependentExperiments(tv) isa IndependentExperiments{<:Experiment}
    @test IndependentExperiments(tv...) isa IndependentExperiments{<:Experiment}
end

@testset "SteadyStateExperiments" begin
    model = reactionsystem()
    @unpack k1, c1, s1, s2, s1s2 = model

    params = [c1 => 3.0]
    u0 = [s2 => 1.0, s1 => 2.0]
    data = generate_data(model; params)
    ss_data = data[end, 2:end]
    ex1 = SteadyStateExperiment(ss_data, model;
        alg = DynamicSS(Tsit5()),
        overrides = params)
    @test nameof(ex1) == "SteadyStateExperiment"

    ex2 = Experiment(data, model; overrides = u0, dependency = ex1)
    ex3 = Experiment(data, model; overrides = u0)

    prob = InverseProblem([ex1, ex2, ex3],
        [Initial(s2) => (1.5, 3), k1 => (0, 5)])
    cost = objective(prob, SingleShooting(maxiters = 1))
    # Run JET error analysis
    @test_call target_modules=(DyadModelOptimizer,) cost()
    @test !iszero(cost())

    @testset "Iteration" begin
        experiments = get_experiments(prob)
        @test length(experiments) == 3

        @test experiments[1] === ex1
        @test experiments[2] === ex2
        @test experiments[3] === ex3

        @test first(experiments) === ex1
        @test last(experiments) === ex3

        @test experiments[begin] === ex1
        @test experiments[end] === ex3
    end

    @testset "Automated steady state computation" begin
        x = DyadModelOptimizer.initial_state(Any, prob)
        sol1 = simulate(ex1, prob)
        sol2 = simulate(ex2, prob)
        # note that explicit `default_u0` is not public API and
        # now the default_i0 has to be only the u0s that would actually
        # be changed, not all of them
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

@testset "ChainedExperiments" begin
    model = reactionsystem()
    @unpack k1, c1, s1, s2, s1s2 = model

    params = [c1 => 3.0]
    u0 = [s2 => 1.0, s1 => 2.0]
    data = generate_data(model; params)
    ss_data = data[end, 2:end]
    ex1 = SteadyStateExperiment(ss_data, model;
        alg = DynamicSS(Tsit5()),
        overrides = params)
    @test nameof(ex1) == "SteadyStateExperiment"

    ex2 = Experiment(data, model; overrides = u0, dependency = ex1)
    ex3 = Experiment(data, model; overrides = u0, dependency = ex2)

    prob = InverseProblem([ex1, ex2, ex3],
        [Initial(s2) => (1.5, 3), k1 => (0, 5)])
    cost = objective(prob, SingleShooting(maxiters = 1))
    # Run JET error analysis
    @test_call target_modules=(DyadModelOptimizer,) cost()
    @test !iszero(cost())

    @testset "Iteration" begin
        experiments = get_experiments(prob)
        @test length(experiments) == 3

        @test experiments[1] === ex1
        @test experiments[2] === ex2
        @test experiments[3] === ex3

        @test first(experiments) === ex1
        @test last(experiments) === ex3

        @test experiments[begin] === ex1
        @test experiments[end] === ex3
    end

    @testset "Automated u0 computation" begin
        x = DyadModelOptimizer.search_space_defaults(prob)
        sol1 = simulate(ex1, prob)
        sol2 = simulate(ex2, prob)
        sol2_ss = simulate(ex2, prob, x, sol1[[s1s2]])
        sol3 = simulate(ex3, prob)
        # only s1s2 will be obtained from the steady state
        # because the other states are fixed by the experiment
        @test sol2[s1s2, 1] == sol1[s1s2]
        @test sol2[s1, 1] == 2.0
        @test sol2[s2, 1] == 1.0
        @test sol2.u == sol2_ss.u
        # experiment3 starts from experiment2
        @test sol3[s1s2, 1] == sol2[s1s2, end]
    end
end
