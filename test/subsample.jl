using Test
using ModelingToolkit
using DyadModelOptimizer
using DyadModelOptimizer: KSStatistic
using OrdinaryDiffEqTsit5
using DataFrames
using StatsBase
using Distributions
using OptimizationOptimJL, OptimizationBBO

include("reactionsystem.jl")

@testset "RefWeights" begin
    model = reactionsystem()

    data1 = generate_data(model)
    trial1 = Experiment(data1, model, tspan = (0.0, 1.0), alg = Tsit5())
    tspan, u0, params = (0.0, 2.0), [model.s2 => 1.0], [model.c1 => 3.0]
    data2 = generate_data(model, tspan, 10; u0, params)
    trial2 = Experiment(data2, model; overrides = [u0; params], tspan)

    prob = InverseProblem([trial1, trial2],
        [Initial(model.s2) => (1.5, 3), model.k1 => (0, 5)])

    N = 5000
    vp = parametric_uq(prob,
        StochGlobalOpt(method = SingleShooting(maxiters = 10,
            optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited())),
        sample_size = N)
    @test length(vp) == N

    binning(sol) = sol(0.5)[1] > 1.2 ? 1 : 2
    ref_w = [0.5, 0.5]
    subsample_size = 1000
    alg = RefWeights(; binning, reference_weights = ref_w, n = subsample_size)

    vp_sub = subsample(alg, vp, trial1)

    sim = solve_ensemble(vp, trial1)
    initial_bins = [binning(sol) for sol in sim]
    initial_hist = fit(Histogram, initial_bins, nbins = 2)
    @test initial_hist.weights ./ N≈[0.80, 0, 0.20] rtol=0.1

    sub = solve_ensemble(vp_sub, trial1)
    bins = [binning(sol) for sol in sub]
    hist = fit(Histogram, bins, nbins = 2)
    @test hist.weights ./ subsample_size≈[0.5, 0, 0.5] rtol=0.1
end

# TODO enable after updating MCMC
# @testset "RefWeights + MCMC" begin
#     model = reactionsystem()
#     @parameters k1, c1
#     @variables t s2(t)
#     data1 = generate_data(model)
#     trial1 = Experiment(data1, model, tspan = (0.0, 1.0), alg = Tsit5())
#     tspan, u0, params = (0.0, 2.0), [s2 => 1.0], [c1 => 3.0]
#     data2 = generate_data(model, tspan, 10; u0, params)
#     trial2 = Experiment(data2, model; u0, params, tspan)

#     noise_level = 0.5
#     rd1 = randn(size(data1[:, Not("timestamp")])) .* noise_level
#     data1[:, Not("timestamp")] .+= rd1
#     rd2 = randn(size(data2[:, Not("timestamp")])) .* noise_level
#     data2[:, Not("timestamp")] .+= rd2

#     prob = InverseProblem([trial1, trial2],
#         [s2 => (1.5, 3), k1 => (0, 5)])

#     N = 2000
#     vp = parametric_uq(prob, MCMCOpt(maxiters = N, discard_initial = 0), sample_size = N)
#     @test length(vp) == N

#     binning(sol) = sol(0.5)[1] > 1.2 ? 1 : 2
#     ref_w = [0.5, 0.5]
#     subsample_size = 1000
#     alg = RefWeights(; binning, reference_weights = ref_w, n = subsample_size)

#     vp_sub = subsample(alg, vp, trial1)

#     sub = solve_ensemble(vp_sub, trial1)
#     bins = [binning(sol) for sol in sub]
#     hist = fit(Histogram, bins, nbins = 2)
#     @test hist.weights ./ subsample_size≈[0.5, 0, 0.5] rtol=0.1
# end

# @testset "DDS" begin
#     model = reactionsystem()
#     @parameters k1, c1
#     @variables t s1(t) s1s2(t) s2(t)
#     data1 = generate_data(model)
#     trial1 = Experiment(data1, model, tspan = (0.0, 1.0), alg = Tsit5())
#     tspan, u0, params = (0.0, 2.0), [s2 => 1.0], [c1 => 3.0]
#     data2 = generate_data(model, tspan, 10; u0, params)
#     trial2 = Experiment(data2, model; u0, params, tspan)

#     prob = InverseProblem([trial1, trial2],
#         [s2 => (1.5, 3), k1 => (0, 5)])

#     N = 5_000
#     vp = parametric_uq(prob,
#         StochGlobalOpt(method = SingleShooting(maxiters = 10,
#             optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited())),
#         sample_size = N)
#     @test length(vp) == N

#     dist_s1 = TransformedBeta(Beta = Beta(6, 2), lb = 0, ub = 1.5)
#     dist_s1s2 = TransformedBeta(Beta = Beta(1, 6), lb = 2.4, ub = 3.2)
#     ref = [
#         s1 => (dist = dist_s1, t = 1.0),
#         s1s2 => (dist = dist_s1s2, t = 0.5)
#     ]
#     alg = DDS(reference = ref, n = 1_000, nbins = 20)
#     vp_sub = subsample(alg, vp, trial1)

#     ecdf_s1 = ecdf(rand(dist_s1, 5_000))
#     ecdf_s1s2 = ecdf(rand(dist_s1s2, 5_000))

#     sim = solve_ensemble(vp, trial1)
#     df = [DataFrame(sol) for sol in sim]
#     data = reduce(append!, df)
#     d_s1 = data[data[!, :timestamp] .== 1.0, "s1"]
#     d_s1s2 = data[data[!, :timestamp] .== 0.5, "s1s2"]

#     sim_sub = solve_ensemble(vp_sub, trial1)
#     df_sub = [DataFrame(sol) for sol in sim_sub]
#     data_sub = reduce(append!, df_sub)
#     d_s1_sub = data_sub[data_sub[!, :timestamp] .== 1.0, "s1"]
#     d_s1s2_sub = data_sub[data_sub[!, :timestamp] .== 0.5, "s1s2"]

#     # Test whether the subsampled distributions are closer to the reference ones
#     # compared to the ones from parametric_uq.
#     init_KS_s1 = KSStatistic(d_s1, ecdf_s1)
#     init_KS_s1s2 = KSStatistic(d_s1s2, ecdf_s1s2)
#     final_KS_s1 = KSStatistic(d_s1_sub, ecdf_s1)
#     final_KS_s1s2 = KSStatistic(d_s1s2_sub, ecdf_s1s2)

#     @test final_KS_s1 - init_KS_s1 < 0.0
#     @test final_KS_s1s2 - init_KS_s1s2 < 0.0
# end
