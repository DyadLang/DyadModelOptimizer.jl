using Test
using DyadModelOptimizer
using ModelingToolkit
using DataFrames
using Statistics
using DataInterpolations
using StableRNGs
using Random
using OptimizationMOI, Ipopt
Random.seed!(StableRNG(42), 42)

include("../reactionsystem.jl")
model = reactionsystem()
model_with_observed = reactionsystem_obs()
n_save = 300
true_c = 2.5

function test_reaction_system_collocation(data, alg, rtol_calibrate, rtol_param_uq)
    two_states_data = data[!, ["timestamp", "s1(t)", "s2(t)"]]
    one_state_data = data[!, ["timestamp", "s1(t)"]]
    @unpack c1 = model
    @testset "Complete Data" begin
        experiment = Experiment(data, model, tspan = (0.0, 1.0))
        invprob = InverseProblem([experiment], [c1 => (0, 5)])
        r = calibrate(invprob, alg)
        @test only(r)≈true_c rtol=rtol_calibrate

        ps = parametric_uq(invprob,
            StochGlobalOpt(method = alg),
            sample_size = 1)
        @test only(ps[1])≈true_c rtol=rtol_param_uq
    end
    @testset "Two States Data" begin
        experiment = Experiment(two_states_data, model, tspan = (0.0, 1.0))
        invprob = InverseProblem([experiment], [c1 => (0, 5)])
        r = calibrate(invprob, alg)
        @test only(r)≈true_c rtol=rtol_calibrate

        ps = parametric_uq(invprob,
            StochGlobalOpt(method = alg),
            sample_size = 1)
        @test only(ps[1])≈true_c rtol=rtol_param_uq
    end
    @testset "One State Data" begin
        experiment = Experiment(one_state_data, model, tspan = (0.0, 1.0))
        invprob = InverseProblem([experiment], [c1 => (0, 5)])
        @test_throws ErrorException calibrate(invprob, alg)
    end

    # @testset "Cache" begin
    #     experiment = Experiment(two_states_data, model, tspan = (0.0, 1.0))
    #     invprob = InverseProblem([experiment], [c1 => (0, 5)])

    #     cost = objective(invprob, alg)
    #     @test !isnothing(cost.alg_cache)
    # end
end

function test_reaction_system_observed_collocation(data, alg, rtol_calibrate, rtol_param_uq)
    three_vars_data = data[!, ["timestamp", "s1(t)", "s2(t)", "s3(t)"]]
    two_vars_data = data[!, ["timestamp", "s1(t)", "s3(t)"]]
    one_var_data = data[!, ["timestamp", "s3(t)"]]
    @unpack c1 = model_with_observed
    @testset "Both States and Observed Data" begin
        experiment = Experiment(three_vars_data, model_with_observed, tspan = (0.0, 1.0))
        invprob = InverseProblem([experiment], [c1 => (0, 5)])
        r = calibrate(invprob, alg)
        @test only(r)≈true_c rtol=rtol_calibrate

        ps = parametric_uq(invprob,
            StochGlobalOpt(method = alg),
            sample_size = 1)
        @test only(ps[1])≈true_c rtol=rtol_param_uq
    end
    @testset "One State and Observed Data" begin
        experiment = Experiment(two_vars_data, model_with_observed, tspan = (0.0, 1.0))
        invprob = InverseProblem([experiment], [c1 => (0, 5)])
        r = calibrate(invprob, alg)
        @test only(r)≈true_c rtol=rtol_calibrate

        ps = parametric_uq(invprob,
            StochGlobalOpt(method = alg),
            sample_size = 1)
        @test only(ps[1])≈true_c rtol=rtol_param_uq
    end
    @testset "Only Observed Data" begin
        experiment = Experiment(one_var_data, model_with_observed, tspan = (0.0, 1.0))
        invprob = InverseProblem([experiment], [c1 => (0, 5)])
        @test_throws ErrorException calibrate(invprob, alg)
    end
end

@unpack c1 = model
data = generate_data(model, (0.0, 1.0), n_save; params = [c1 => true_c])
data_noise = generate_noisy_data(model,
    (0.0, 1.0),
    n_save;
    noise_std = 0.03,
    params = [c1 => true_c])

@unpack c1 = model_with_observed
data_obs = generate_observed_data(model_with_observed,
    (0.0, 1.0),
    n_save;
    params = [c1 => true_c])

@testset "Reaction System" begin
    @testset "Only States" begin
        @unpack c1 = model
        @testset "Data with no Noise" begin
            @testset "KernelCollocation" begin
                alg = KernelCollocation(maxiters = 10^3)
                test_reaction_system_collocation(data, alg, 1e-2, 1e-2)
            end
            @testset "SplineCollocation" begin
                alg = SplineCollocation(maxiters = 10^3, interp = CubicSpline)
                test_reaction_system_collocation(data, alg, 1e-2, 1e-2)

                # This is just to test `BSplineInterpolation` for collocation as it breaks if we pass time series data directly
                alg = SplineCollocation(maxiters = 10^3,
                    interp = BSplineInterpolation,
                    interp_args = (3, :Uniform, :Average))
                test_reaction_system_collocation(data, alg, 2e-2, 2e-2)
            end
            @testset "NoiseRobustCollocation" begin
                alg = NoiseRobustCollocation(maxiters = 10^3,
                    diff_iters = 500,
                    α = 0.001,
                    cutoff = (0.1, 0.0),
                    tvdiff_kwargs = (diff_kernel = "square", scale = "large"))
                test_reaction_system_collocation(data, alg, 1e-2, 1e-2)
            end
        end
        @testset "Data with noise" begin
            @testset "KernelCollocation" begin
                alg = KernelCollocation(maxiters = 10^3,
                    cutoff = (0.1, 0.0),
                    bandwidth = 0.4)
                test_reaction_system_collocation(data_noise, alg, 2e-2, 2e-2)
            end
            @testset "NoiseRobustCollocation" begin
                alg = NoiseRobustCollocation(maxiters = 10^3,
                    diff_iters = 500,
                    α = 5.0,
                    cutoff = (0.3, 0.1),
                    tvdiff_kwargs = (diff_kernel = "square", scale = "large"))
                test_reaction_system_collocation(data_noise, alg, 2e-2, 2e-2)
            end
        end
    end
    @testset "Data with States and Observed" begin
        @unpack c1 = model_with_observed
        @testset "Data with no Noise" begin
            @testset "KernelCollocation" begin
                alg = KernelCollocation(maxiters = 10^3)
                test_reaction_system_observed_collocation(data_obs, alg, 1e-2, 1e-2)
            end
            @testset "SplineCollocation" begin
                alg = SplineCollocation(maxiters = 10^3, interp = CubicSpline)
                test_reaction_system_observed_collocation(data_obs, alg, 1e-2, 1e-2)
            end
            @testset "NoiseRobustCollocation" begin
                alg = NoiseRobustCollocation(maxiters = 10^3,
                    diff_iters = 500,
                    α = 0.001,
                    cutoff = (0.1, 0.0),
                    tvdiff_kwargs = (diff_kernel = "square", scale = "large"))
                test_reaction_system_observed_collocation(data_obs, alg, 1e-2, 1e-2)
            end
        end
    end
end

@testset "Single State data" begin
    tend = 10.0
    ts = 0.0:0.1:tend
    true_p = 0.5
    data = DataFrame("timestamp" => ts, "x" => exp.(true_p * ts))
    t = ModelingToolkit.t_nounits
    D = ModelingToolkit.D_nounits
    @variables x(t) = 1.0
    @parameters p = 2.0
    model = complete(ODESystem([D(x) ~ p * exp(p * t)], t, name = :model))
    experiment = Experiment(data, model; tspan = (0.0, tend))
    invprob = InverseProblem(experiment, [p => (0.0, 10.0)])
    opt = IpoptOptimizer(;
        max_iter = 2000,
        tol = 1e-8,
        acceptable_tol = 1e-8)
    alg = KernelCollocation(maxiters = 2000, optimizer = opt)
    r = calibrate(invprob, alg)
    @test only(r)≈true_p rtol=1e-3
end

@testset "No Data because of cutoff" begin
    @unpack c1 = model
    data = generate_noisy_data(model,
        (0.0, 1.0),
        n_save;
        noise_std = 0.05,
        params = [c1 => true_c])
    cutoff = (0.5, 0.6)
    algs = [KernelCollocation(maxiters = 10^3, cutoff = cutoff),
        SplineCollocation(maxiters = 10^3, interp = CubicSpline, cutoff = cutoff),
        NoiseRobustCollocation(maxiters = 10^3,
            diff_iters = 500,
            α = 10.0,
            cutoff = cutoff)]
    experiment = Experiment(data, model, tspan = (0.0, 1.0))
    invprob = InverseProblem([experiment], [c1 => (0, 5)])
    @testset "$(nameof(typeof(alg)))" for alg in algs
        @test_throws ErrorException calibrate(invprob, alg)
    end
end
