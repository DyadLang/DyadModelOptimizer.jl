using Test
using DyadModelOptimizer
using DyadModelOptimizer: indexof, get_params_idx,
                          get_noise_priors, get_likelihood
using Distributions

include("reactionsystem.jl")

@testset "No experiment param - No optimization param" begin
    model = reactionsystem()

    c1_opt = defaults(model)[model.c1] + 1.0
    k1_opt = defaults(model)[model.k1] + 1.0
    s1_opt = defaults(model)[model.s1] + 1.0

    experiment = Experiment(nothing, model; tspan = (0.0, 5.0))
    prob = InverseProblem(experiment, [model.s1 => (0.0, 2.0)])
    mc = get_cache(prob, experiment)

    x = [s1_opt]
    pvt = get_params(mc, experiment, x)
    pv = getparams(model)
    p = parameters(model)

    @test all(pvt .== pv)
end

@testset "No experiment param - Optimization params" begin
    model = reactionsystem()

    c1_opt = defaults(model)[model.c1] + 1.0
    k1_opt = defaults(model)[model.k1] + 1.0

    experiment = Experiment(nothing, model; tspan = (0.0, 5.0))
    prob = InverseProblem(experiment, [model.k1 => (0, 5), model.c1 => (0, 5)])
    mc = get_cache(prob, experiment)

    x = [k1_opt, c1_opt]
    pvt = get_params(mc, experiment, x)
    pv = getparams(model)
    p = parameters(model)

    c1_idx = indexof(model.c1, p)
    @test pvt[c1_idx] == c1_opt
    k1_idx = indexof(model.k1, p)
    @test pvt[k1_idx] == k1_opt
end

@testset "Experiment param - Optimization params" begin
    model = reactionsystem()

    c1_fixed = defaults(model)[model.c1] + 2.0
    c1_opt = defaults(model)[model.c1] + 1.0
    k1_opt = defaults(model)[model.k1] + 1.0

    params = [model.c1 => c1_fixed]
    experiment = Experiment(nothing, model; params, tspan = (0.0, 5.0))
    prob = InverseProblem(experiment, [model.c1 => (0, 5), model.k1 => (0, 5)])
    mc = get_cache(prob, experiment)

    x = [c1_opt, k1_opt]
    pvt = get_params(mc, experiment, x)
    pv = getparams(model)
    p = parameters(model)

    c1_idx = indexof(model.c1, p)
    @test pvt[c1_idx] == c1_fixed
    k1_idx = indexof(model.k1, p)
    @test pvt[k1_idx] == k1_opt
end

@testset "Experiment params - No optimization param" begin
    model = reactionsystem()

    c1_fixed = defaults(model)[model.c1] + 1.0
    k1_fixed = defaults(model)[model.k1] + 1.0
    s1_opt = defaults(model)[model.s1] + 1.0

    params = [model.c1 => c1_fixed, model.k1 => k1_fixed]
    experiment = Experiment(nothing, model; params, tspan = (0.0, 5.0))
    prob = InverseProblem([experiment], [model.s1 => (0.0, 2.0)])
    mc = get_cache(prob, experiment)

    x = [s1_opt]
    pvt = get_params(mc, experiment, x)
    pv = getparams(model)
    p = parameters(model)

    c1_idx = indexof(model.c1, p)
    @test pvt[c1_idx] == c1_fixed
    k1_idx = indexof(model.k1, p)
    @test pvt[k1_idx] == k1_fixed
end

@testset "MCMC utils" begin
    model = reactionsystem()

    np1 = InverseGamma(5, 3)
    trial1 = Experiment(nothing, model;
        tspan = (0.0, 5.0),
        noise_priors = np1)
    @test np1 == only(get_noise_priors(trial1, states(model)))
    @test get_likelihood(trial1)(zeros(3), rand(np1)) isa IsoNormal

    ss_trial1 = SteadyStateExperiment(nothing, model;
        noise_priors = np1)
    @test np1 == only(get_noise_priors(ss_trial1, states(model)))
    @test get_likelihood(ss_trial1)(zeros(3), rand(np1)) isa IsoNormal

    np2 = [
        model.s2 => Gamma(2, 2),
        model.s1 => InverseGamma(5, 3),
        model.s1s2 => Exponential(5)
    ]
    trial2 = Experiment(nothing, model;
        tspan = (0.0, 5.0),
        noise_priors = np2)
    @test all([InverseGamma(5, 3), Exponential(5), Gamma(2, 2)] .==
              get_noise_priors(trial2, states(model)))
    @test get_likelihood(trial2)(zeros(3), rand.(last.(np2))) isa DiagNormal

    ss_trial2 = SteadyStateExperiment(nothing, model;
        noise_priors = np2)
    @test all([InverseGamma(5, 3), Exponential(5), Gamma(2, 2)] .==
              get_noise_priors(ss_trial2, states(model)))
    @test get_likelihood(ss_trial2)(zeros(3), rand.(last.(np2))) isa DiagNormal
end
