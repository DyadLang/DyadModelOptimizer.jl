using Test
using DyadModelOptimizer
using OrdinaryDiffEqTsit5
using OptimizationOptimJL
using OptimizationBBO
using ModelingToolkit
using DataFrames
using Plots
using LineSearches
using JET

include("../reactionsystem.jl")
include("../plot.jl")

@testset "One experiment" begin
    model = reactionsystem()
    @unpack k1, c1 = model
    data = generate_data(model, (0.0, 1.0), 10, params = [k1 => 2, c1 => 3])
    experiment = Experiment(data, model)

    prob = InverseProblem(experiment, [k1 => (0, 5), c1 => (0, 5)])

    # broken = VERSION < v"1.10.0-alpha1" # possible JET bug with QuasiMonteCarlo@0.3
    @test_call target_modules=(DyadModelOptimizer,) parametric_uq(prob,
        StochGlobalOpt(method = SingleShooting(maxiters = 10,
            optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited())),
        sample_size = 1)
    ps = parametric_uq(prob,
        StochGlobalOpt(method = SingleShooting(maxiters = 10^4,
            optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited())),
        sample_size = 50)
    @test length(ps) == 50
    @test !iszero(ps)

    sim = solve_ensemble(ps, experiment)
    @test sim.converged

    test_nowarn_plots(ps, prob, experiment)
end

@testset "Two experiments" begin
    model = reactionsystem()
    @unpack k1, c1, s1, s2 = model
    data1 = generate_noisy_data(model)
    experiment1 = Experiment(data1, model, tspan = (0.0, 1.0), depvars = [s1, s2],
        alg = Tsit5())
    tspan, u0, params = (0.0, 2.0), [s2 => 1.0], [c1 => 3.0]
    data2 = generate_noisy_data(model, tspan, 10; u0, params)
    experiment2 = Experiment(data2, model; overrides = [u0; params], tspan)

    prob = InverseProblem([experiment1, experiment2],
        [s2 => (1.5, 3), k1 => (0, 5)])

    ps = parametric_uq(prob,
        StochGlobalOpt(method = SingleShooting(maxiters = 10,
            optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited())),
        sample_size = 50)
    @test length(ps) == 50
    @test !iszero(ps)

    sim = solve_ensemble(ps, experiment1)
    @test sim.converged

    test_nowarn_plots(ps, prob, experiment1)
end

@testset "SteadyStateExperiments" begin
    model = reactionsystem()
    @unpack k1, c1, s2 = model
    data1 = generate_noisy_data(model)
    experiment1 = SteadyStateExperiment(
        (; s1 = [data1[end, 2]], s1s2 = [data1[end, 3]], s2 = [data1[end, 4]]), model)
    tspan, u0, params = (0.0, 2.0), [s2 => 1.0], [c1 => 3.0]
    data2 = generate_noisy_data(model, tspan, 10; u0, params)
    experiment2 = Experiment(
        data2, model; dependency = experiment1, overrides = [u0; params], tspan)

    prob = InverseProblem([experiment1, experiment2],
        [s2 => (1.5, 3), k1 => (0, 5)])

    @test_call target_modules=(DyadModelOptimizer,) parametric_uq(prob,
        StochGlobalOpt(method = SingleShooting(maxiters = 10,
            optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited())),
        sample_size = 1)
    ps = parametric_uq(prob,
        StochGlobalOpt(method = SingleShooting(maxiters = 10,
            optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited())),
        sample_size = 50)
    @test length(ps) == 50
    @test !iszero(ps)

    sim = solve_ensemble(ps, experiment2)
    @test sim.converged

    test_nowarn_plots(ps, prob, experiment2)
end

@testset "save_names with observed states" begin
    model = reactionsystem_obs()
    @unpack k1, c1, s1, s3 = model
    tspan = (0.0, 1.0)
    prob = ODEProblem(model, [], tspan, [])
    saveat = range(prob.tspan..., length = 10)
    sol = solve(prob, Tsit5(); saveat)

    data = DataFrame(:timestamp => sol.t, :s1 => sol[s1], :s3 => sol[s3])
    experiment = Experiment(data, model, tspan = (0.0, 1.0))

    invprob = InverseProblem(experiment, [k1 => (0, 5), c1 => (0, 5)])
    alg = StochGlobalOpt(method = SingleShooting(maxiters = 100,
        optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited()))

    sample_size = 20
    ps = parametric_uq(invprob, alg; sample_size)

    @test length(ps) == sample_size
    @test !iszero(ps)

    sim = solve_ensemble(ps, experiment)
    @test sim.converged

    test_nowarn_plots(ps, invprob, experiment)
end

@testset "One experiment + ARMLoss bounds" begin
    model = reactionsystem()
    @unpack k1, c1, s1, s2, s1s2 = model
    df = DataFrame("timestamp" => [0.0, 1.0],
        "s1(t)" => [(0.1, 3.0), (0.1, 3.0)],
        "s2(t)" => [(0.5, 3.5), (0.5, 3.5)],
        "s1s2(t)" => [(0.0, 5.0), (0.0, 5.0)])
    experiment = Experiment(df, model, tspan = (0.0, 1.0))

    prob = InverseProblem(experiment, [k1 => (0, 5), c1 => (0, 5)])

    ps = parametric_uq(prob,
        StochGlobalOpt(method = SingleShooting(maxiters = 10,
            optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited())),
        sample_size = 50)
    @test length(ps) == 50
    @test !iszero(ps)

    sim = solve_ensemble(ps, experiment)
    @test sim.converged

    test_nowarn_plots(ps, prob, experiment)
end

@testset "using MultipleShooting as the method" begin
    model = reactionsystem()
    @unpack k1, c1 = model
    data = generate_data(model, (0.0, 1.0), 10, params = [k1 => 2, c1 => 3])
    experiment = Experiment(data, model, abstol = 1e-6, reltol = 1e-6)

    prob = InverseProblem(experiment, [k1 => (0, 5), c1 => (0, 5)])

    ps = parametric_uq(prob,
        StochGlobalOpt(method = MultipleShooting(maxiters = 10^3,
            trajectories = 3,
            optimizer = LBFGS(linesearch = BackTracking()))),
        sample_size = 10)
    @test length(ps) == 10
    # k1 and c1 are not identifiable, but k1 * c1 ≈ 6
    @test all(isapprox.(Tables.columns(ps)[:k1] .* Tables.columns(ps)[:c1], 6, rtol = 1e-4))

    sim = solve_ensemble(ps, experiment)
    @test sim.converged

    test_nowarn_plots(ps, prob, experiment)
end
