using Test
using Distributed
using SciMLBase: EnsembleDistributed
addprocs(2, exeflags = "-t 2 --project=$(Base.active_project())")

@everywhere using DyadModelOptimizer
using ModelingToolkit
using DyadModelOptimizer: get_params
using DataFrames
using OptimizationOptimJL
using Tables

include("reactionsystem.jl")

@testset "One experiment (StochGlobalOpt)" begin
    model = reactionsystem()
    data = generate_data(model, (0.0, 1.0), params = [model.c1 => 4])
    experiment = Experiment(data, model)
    prob = InverseProblem(experiment, [model.k1 => (0, 5), model.c1 => (0, 5)])

    alg = StochGlobalOpt(parallel_type = EnsembleDistributed())
    ps = parametric_uq(prob, alg, sample_size = 200)

    @test length(ps) == 200
end

# TODO enable after MCMC is updated
# @testset "One experiment (MCMC)" begin
#     model = reactionsystem()
#     @parameters k1, c1
#     data = generate_data(model)

#     noise_level = 0.5
#     rd = randn(size(data[:, Not("timestamp")])) .* noise_level
#     data[:, Not("timestamp")] .+= rd

#     experiment = Experiment(data, model, tspan = (0.0, 1.0), alg = TRBDF2())
#     prob = InverseProblem([experiment], [k1 => (0, 5), c1 => (0, 5)])

#     N_chains = nworkers()
#     ps = parametric_uq(prob, MCMCOpt(maxiters = 5, parallel_type = EnsembleDistributed());
#         sample_size = 5, N_chains)

#     # make sure the MCMCChains object actually contains the correct number of chains
#     @test size(ps.original)[3] == N_chains
# end

# @testset "Two experiments (hierarchical MCMC)" begin
#     model = reactionsystem()
#     @parameters k1, c1
#     @variables t s1(t) s2(t)

#     data1 = generate_noisy_data(model)
#     experiment1 = Experiment(data1, model, tspan = (0.0, 1.0), save_names = [s1, s2],
#         alg = TRBDF2())
#     tspan, u0, params = (0.0, 2.0), [s2 => 1.0], [c1 => 3.0]
#     data2 = generate_noisy_data(model, tspan, 10; u0, params)
#     experiment2 = Experiment(data2, model; u0, params, tspan, alg = TRBDF2())

#     prob = InverseProblem([experiment1, experiment2],
#         [s2 => (1.5, 3), k1 => (0, 5)])

#     N_chains = nworkers()
#     ps = parametric_uq(prob,
#         MCMCOpt(maxiters = 5, parallel_type = EnsembleDistributed());
#         sample_size = 5,
#         N_chains)

#     # make sure the MCMCChains object actually contains the correct number of chains
#     @test size(ps.original)[3] == N_chains
# end

@testset "EnsembleSerial for segments" begin
    model = reactionsystem()
    data = generate_data(model, (0.0, 1.0), 50)
    experiment = Experiment(data, model)
    prob = InverseProblem(experiment, [model.k1 => (0, 5), model.c1 => (0, 5)])

    alg = StochGlobalOpt(
        method = MultipleShooting(trajectories = 5,
            ensemblealg = EnsembleSerial(),
            maxiters = 100),
        parallel_type = EnsembleDistributed())
    ps = parametric_uq(prob, alg, sample_size = 20)
    t = Tables.columns(ps)
    @test all(isapprox.(t[:k1] .* t[:c1], 2; rtol = 1e-4))
end

@testset "EnsembleThreads for segments" begin
    model = reactionsystem()
    data = generate_data(model, (0.0, 1.0), 50)
    experiment = Experiment(data, model)
    prob = InverseProblem(experiment, [model.k1 => (0, 5), model.c1 => (0, 5)])

    alg = StochGlobalOpt(
        method = MultipleShooting(trajectories = 5,
            ensemblealg = EnsembleThreads(),
            maxiters = 100),
        parallel_type = EnsembleDistributed())
    ps = parametric_uq(prob, alg, sample_size = 20)

    t = Tables.columns(ps)
    @test all(isapprox.(t[:k1] .* t[:c1], 2; rtol = 1e-4))
end
