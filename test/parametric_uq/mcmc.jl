using Test
using DyadModelOptimizer
using OrdinaryDiffEq, SteadyStateDiffEq
using ModelingToolkit
using DataFrames
using Turing
using Plots
using JET

include("../reactionsystem.jl")
include("../plot.jl")

@testset "Two trials + MCMC" begin
    model = reactionsystem()
    @unpack k1, c1, s1, s2 = model
    data1 = generate_noisy_data(model)
    trial1 = Experiment(data1, model, tspan = (0.0, 1.0), save_names = [s1, s2],
        alg = TRBDF2())
    tspan, u0, params = (0.0, 2.0), [s2 => 1.0], [c1 => 3.0]
    data2 = generate_noisy_data(model, tspan, 10; u0, params)
    trial2 = Experiment(data2, model; u0, params, tspan, alg = TRBDF2())

    prob = InverseProblem([trial1, trial2],
        [s2 => (1.5, 3), k1 => (0, 5)])

    # Run JET error analysis
    @test_call target_modules=(DyadModelOptimizer,) parametric_uq(prob,
        MCMCOpt(maxiters = 5),
        sample_size = 1)
    vp = parametric_uq(prob, MCMCOpt(maxiters = 5), sample_size = 5)
    @test length(vp) == 5
    @test all(p -> all(.!iszero.(values(p))), vp)

    sim = solve_ensemble(vp, trial1)
    @test sim.converged

    test_nowarn_plots(vp, prob, trial1)
end

@testset "Two trials + MCMC + Noise priors" begin
    model = reactionsystem()
    @unpack k1, c1, s1, s2 = model
    data1 = generate_noisy_data(model)
    trial1 = Experiment(data1, model,
        tspan = (0.0, 1.0),
        save_names = [s1, s2],
        alg = TRBDF2(),
        noise_priors = [s1 => Gamma(2, 2), s2 => InverseGamma(1, 3)])
    tspan, u0, params = (0.0, 2.0), [s2 => 1.0], [c1 => 3.0]
    data2 = generate_noisy_data(model, tspan, 10; u0, params)
    trial2 = Experiment(data2, model;
        u0,
        params,
        tspan,
        alg = TRBDF2(),
        noise_priors = Exponential(1))

    prob = InverseProblem([trial1, trial2],
        [s2 => (1.5, 3), k1 => (0, 5)])

    # Run JET error analysis
    @test_call target_modules=(DyadModelOptimizer,) parametric_uq(prob,
        MCMCOpt(maxiters = 5),
        sample_size = 1)
    vp = parametric_uq(prob, MCMCOpt(maxiters = 5), sample_size = 5)
    @test length(vp) == 5
    @test all(p -> all(.!iszero.(values(p))), vp)

    sim = solve_ensemble(vp, trial1)
    @test sim.converged

    test_nowarn_plots(vp, prob, trial1)
end

@testset "Two trials + MCMC + Distributions search space" begin
    model = reactionsystem()
    @unpack k1, c1, s1, s2 = model
    data1 = generate_noisy_data(model)
    trial1 = Experiment(data1, model, tspan = (0.0, 1.0), save_names = [s1, s2],
        alg = TRBDF2())
    tspan, u0, params = (0.0, 2.0), [s2 => 1.0], [c1 => 3.0]
    data2 = generate_noisy_data(model, tspan, 10; u0, params)
    trial2 = Experiment(data2, model; u0, params, tspan, alg = TRBDF2())

    prob = InverseProblem([trial1, trial2],
        [
            s2 => TransformedBeta(Beta = Beta(2, 5), lb = 1.5, ub = 3.0),
            k1 => TransformedBeta(Beta = Beta(4, 3), lb = 0.0, ub = 5.0)
        ])

    # Run JET error analysis
    @test_call target_modules=(DyadModelOptimizer,) parametric_uq(prob,
        MCMCOpt(maxiters = 5),
        sample_size = 1)
    vp = parametric_uq(prob, MCMCOpt(maxiters = 5), sample_size = 5)
    @test length(vp) == 5
    @test all(p -> all(.!iszero.(values(p))), vp)

    sim = solve_ensemble(vp, trial1)
    @test sim.converged

    test_nowarn_plots(vp, prob, trial1)
end

@testset "SteadyStateExperiments + MCMC" begin
    model = reactionsystem()
    @unpack k1, c1, s2 = model
    data1 = generate_noisy_data(model)
    trial1 = SteadyStateExperiment(collect(data1[end, 2:end]), model)

    tspan, u0, params = (0.0, 2.0), [s2 => 1.0], [c1 => 3.0]
    data2 = generate_noisy_data(model, tspan, 10; u0, params)
    trial2 = Experiment(data2, model; u0, params, tspan, dependency = true)

    prob = InverseProblem(SteadyStateExperiments(trial1, trial2),
        [s2 => (1.5, 3), k1 => (0, 5)])

    vp = parametric_uq(prob, MCMCOpt(maxiters = 10, discard_initial = 5), sample_size = 5)
    @test length(vp) == 5
    @test all(p -> all(.!iszero.(values(p))), vp)

    sim = solve_ensemble(vp, trial2)
    @test sim.converged

    test_nowarn_plots(vp, prob, trial2)
end

@testset "Two trials + hierarchical MCMC" begin
    model = reactionsystem()
    @unpack k1, c1, s1, s2 = model
    data1 = generate_noisy_data(model)
    trial1 = Experiment(data1, model, tspan = (0.0, 1.0), save_names = [s1, s2],
        alg = TRBDF2())
    tspan, u0, params = (0.0, 2.0), [s2 => 1.0], [c1 => 3.0]
    data2 = generate_noisy_data(model, tspan, 10; u0, params)
    trial2 = Experiment(data2, model; u0, params, tspan, alg = TRBDF2())

    prob = InverseProblem([trial1, trial2],
        [s2 => (1.5, 3), k1 => (0, 5)])

    # Run JET error analysis
    @test_call target_modules=(DyadModelOptimizer,) parametric_uq(prob,
        MCMCOpt(maxiters = 5,
            hierarchical = true),
        sample_size = 1)
    vp = parametric_uq(prob, MCMCOpt(maxiters = 5, hierarchical = true), sample_size = 5)

    @test length(vp) == 5
    @test all(p -> all(.!iszero.(values(p))), vp)
    @test all([Symbol("α_", sts) in Turing.names(vp.original)
               for sts in DyadModelOptimizer.search_space_names(prob)])
    @test all([Symbol("β_", sts) in Turing.names(vp.original)
               for sts in DyadModelOptimizer.search_space_names(prob)])

    sim = solve_ensemble(vp, trial1)
    @test sim.converged

    test_nowarn_plots(vp, prob, trial1)
end
