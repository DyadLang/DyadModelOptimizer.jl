module CalibrateTests

using Test
using DyadModelOptimizer
using DyadModelOptimizer: get_params, get_saveat, get_model,
                          get_data, get_saveat, get_solve_alg, get_config,
                          initial_state, timespan, lowerbound, upperbound
using DataFrames
using ModelingToolkit
using Optimization
using OptimizationNLopt
using OptimizationOptimJL, OptimizationBBO
using JET
using DataInterpolations

using ModelingToolkit: t_nounits as t, D_nounits as D

include("../reactionsystem.jl")

@testset "JET error analysis" begin
    model = reactionsystem()
    data = generate_data(model)

    experiment = Experiment(data, model, tspan = (0.0, 1.0))
    invprob = InverseProblem([experiment], [])

    # TODO: remove skip=true when fixed
    @test_call target_modules=(DyadModelOptimizer,) calibrate(invprob,
        DataShooting(maxiters = 10^3,
            groupsize = 3)) skip=true
end

@testset "SingleShooting" begin
    model = reactionsystem()
    params = [model.c1 => 3.5]
    data = generate_data(model; params)

    experiment = Experiment(data, model, tspan = (0.0, 1.0))
    invprob = InverseProblem(experiment, [model.c1 => (0.1, 5)])

    r = calibrate(invprob, SingleShooting(maxiters = 10))
    @test only(r)≈3.5 rtol=1e-4
end

@testset "SingleShooting with local method" begin
    model = reactionsystem()
    @unpack c1 = model

    data = generate_data(model, params = [c1 => 3])

    experiment = Experiment(data, model, tspan = (0.0, 1.0), abstol = 1e-6)
    invprob = InverseProblem([experiment], [c1 => (0.1, 5)])

    alg = SingleShooting(maxiters = 10^3, optimizer = NLopt.G_MLSL_LDS(),
        local_method = NLopt.LD_LBFGS())

    c = cost_contribution(alg, experiment, invprob, [3])
    @test c≈0 atol=1e-7

    r = calibrate(invprob, alg)

    @test only(r)≈3 rtol=1e-4

    alg = SingleShooting(maxiters = 10^3, optimizer = NLopt.G_MLSL_LDS(),
        local_method = NLopt.LD_LBFGS())

    c = cost_contribution(alg, experiment, invprob, [3])
    @test c≈0 atol=1e-7

    # The algorithm TikTak does not support callbacks
    r = calibrate(invprob, alg)

    @test only(r)≈3 rtol=1e-4

    # cross check with parametric_uq
    ps = parametric_uq(invprob,
        StochGlobalOpt(method = SingleShooting(maxiters = 10^3,
            optimizer = NLopt.G_MLSL_LDS(),
            local_method = NLopt.LD_LBFGS())), sample_size = 1)
    @test only(ps[1])≈3 rtol=1e-4
end

@testset "SingleShooting using different loss Functions" begin
    model = reactionsystem()
    @unpack c1 = model
    data = generate_data(model, params = [c1 => 3.0])
    data2 = data[!, ["timestamp", "s1(t)"]]
    alg = SingleShooting(maxiters = 1000)
    loss_functions = [
        squaredl2loss,
        l2loss,
        meansquaredl2loss,
        root_meansquaredl2loss,
        norm_meansquaredl2loss
    ]
    @testset "$loss_function" for loss_function in loss_functions
        _data = [("Single state", data), ("All states", data2)]
        @testset "$(d[1])" for d in _data
            experiment = Experiment(
                d[2], model, tspan = (0.0, 1.0), loss_func = loss_function)
            invprob = InverseProblem(experiment, [c1 => (0, 5)])
            r = calibrate(invprob, alg)
            @test only(r)≈3.0 rtol=1e-4
        end
    end
end

@testset "DataShooting" begin
    model = reactionsystem()
    data = generate_data(model)

    experiment = Experiment(data, model, tspan = (0.0, 1.0))
    invprob = InverseProblem(experiment, [model.c1 => (0, 5)])

    alg = DataShooting(maxiters = 10^3, groupsize = 3)

    c = cost_contribution(alg, experiment, invprob, initial_state(alg, invprob))
    @test c≈0 atol=1e-7

    r = calibrate(invprob, alg)

    @test only(r)≈2 rtol=1e-4

    # cross check with parametric_uq
    ps = parametric_uq(invprob,
        StochGlobalOpt(method = SingleShooting(maxiters = 10^3,
            optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited())),
        sample_size = 1)
    @test only(ps[1])≈2 rtol=1e-4
end

@testset "DataShooting with data in different order" begin
    model = reactionsystem()
    original_data = generate_data(model)
    data = original_data[!, [1, 3, 2, 4]]

    experiment = Experiment(data, model, tspan = (0.0, 1.0))
    invprob = InverseProblem(experiment, [model.c1 => (0, 5)])

    alg = DataShooting(maxiters = 10^3, groupsize = 3)

    c = cost_contribution(alg, experiment, invprob, initial_state(alg, invprob))
    @test c≈0 atol=1e-7

    r = calibrate(invprob, alg)

    @test only(r)≈2 rtol=1e-4

    # cross check with parametric_uq
    ps = parametric_uq(invprob,
        StochGlobalOpt(method = SingleShooting(maxiters = 10^3,
            optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited())),
        sample_size = 1)
    @test only(ps[1])≈2 rtol=1e-4
end

@testset verbose=true "MultipleShooting" begin
    @testset "data saveat does not coincide with shooting interval endpoints" begin
        model = reactionsystem()
        data = generate_data(model, (0.0, 1), 11)

        @unpack c1 = model

        experiment = Experiment(data, model, tspan = (0.0, 1.0))
        invprob = InverseProblem(experiment, [c1 => (0.1, 5)])

        alg = MultipleShooting(maxiters = 10^5, trajectories = 3,
            continuitylossweight = 10^2)

        c = cost_contribution(alg, experiment, invprob, initial_state(alg, invprob))
        # the initial internal u0s are bad guesses, so just do a sanity check
        @test c > 0

        r = calibrate(invprob, alg)

        @test only(r)≈2 rtol=1e-4

        # cross check with parametric_uq
        ps = parametric_uq(invprob,
            StochGlobalOpt(method = SingleShooting(maxiters = 10^3,
                optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited())),
            sample_size = 1)
        @test only(ps[1])≈2 rtol=1e-4
    end

    # https://github.com/JuliaComputing/DyadModelOptimizer.jl/issues/453
    @testset "vector data without all model states" begin
        @variables A(t) = 10.0
        @variables B(t) = 0.0
        @parameters k = 0.5

        eqs = [D(A) ~ -k * A,
            D(B) ~ k * A]

        model = complete(ODESystem(eqs, t, name = :model))

        tspan = (0.0, 10.0)
        k_true = 0.4
        prob = ODEProblem(model, [], tspan, [k => k_true])
        true_sol = solve(prob, Tsit5(), saveat = 1.0)
        data = DataFrame("timestamp" => true_sol.t, "B" => true_sol[B])

        experiment = Experiment(data, model)
        search_space = [k => (0.1, 10.0)]
        invprob = InverseProblem(experiment, search_space)
        alg = SingleShooting(maxiters = 10^4)

        r = calibrate(invprob, alg)
        @test only(r)≈k_true rtol=1e-6

        alg = MultipleShooting(maxiters = 10^3, trajectories = 2,
            continuitylossweight = 1e-8)
        r = calibrate(invprob, alg)
        @test only(r)≈k_true rtol=1e-6
    end

    @testset "experiment fixes u0" begin
        model = reactionsystem()
        @unpack c1, s1 = model

        data = generate_data(model, (0.0, 1), params = [c1 => 4], u0 = [s1 => 3], 11)

        experiment = Experiment(data, model, overrides = [s1 => 3], tspan = (0.0, 1.0))
        invprob = InverseProblem(experiment, [c1 => (0.1, 5)])

        alg = MultipleShooting(maxiters = 10^5, trajectories = 3,
            continuitylossweight = 10^2)

        c = cost_contribution(alg, experiment, invprob, initial_state(alg, invprob))
        # the initial internal u0s are bad guesses, so just do a sanity check
        @test c > 0

        r = calibrate(invprob, alg)

        @test only(r)≈4 rtol=1e-4

        sol = simulate(experiment, invprob, r)
        @test sol[s1][1] == 3

        # cross check with parametric_uq
        ps = parametric_uq(invprob,
            StochGlobalOpt(method = SingleShooting(maxiters = 10^3,
                optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited())),
            sample_size = 1)
        @test only(ps[1])≈4 rtol=1e-4
    end

    @testset "Multiple Experiments" begin
        model = reactionsystem()
        @unpack c1, k1 = model
        data1 = generate_data(model, (0.0, 1), params = [c1 => 4.0, k1 => 2.0], 11)
        data2 = generate_data(model, (0.0, 1), params = [k1 => 2.0, c1 => 4.0], 11)
        experiment1 = Experiment(data1, model, tspan = (0.0, 1.0))
        experiment2 = Experiment(data2, model, tspan = (0.0, 1.0))
        invprob = InverseProblem([experiment1, experiment2],
            [c1 => (0.1, 5.0), k1 => (0.1, 5.0)])
        @testset "$i" for i in [ModelStatesPenalty, ConstraintBased]
            alg = MultipleShooting(maxiters = 100, trajectories = 2, continuitytype = i)
            r = calibrate(invprob, alg)
            @test r[1] * r[2]≈8.0 rtol=1e-3
        end
    end

    @testset verbose=true "Initialization" begin
        model = reactionsystem()
        @unpack c1, s1 = model
        data = generate_data(model, (0.0, 1), params = [c1 => 4], u0 = [s1 => 3], 11)
        two_states_data = data[!, ["timestamp", "s1(t)", "s2(t)"]]
        experiment = Experiment(data, model, overrides = [s1 => 3], tspan = (0.0, 1.0))
        experiment_2 = Experiment(two_states_data,
            model,
            overrides = [s1 => 3],
            tspan = (0.0, 1.0))
        invprob = InverseProblem(experiment, [c1 => (0, 5)])
        invprob_2 = InverseProblem(experiment_2, [c1 => (0, 5)])
        @testset "Default Initialization" begin
            @testset "trajectories = $t" for t in [1, 3]
                alg = MultipleShooting(maxiters = 100,
                    trajectories = t,
                    continuitylossweight = 100)
                r = calibrate(invprob, alg)
                @test only(r)≈4 rtol=1e-4
            end
        end
        @testset "Data Initialization" begin
            @testset "All States" begin
                @testset "trajectories = $t" for t in [1, 3]
                    alg = MultipleShooting(maxiters = 100,
                        trajectories = t,
                        continuitylossweight = 100)
                    r = calibrate(invprob, alg)
                    @test only(r)≈4 rtol=1e-4
                end
            end
            @testset "Two states" begin
                @testset "trajectories = $t" for t in [1, 3]
                    alg = MultipleShooting(maxiters = 100,
                        trajectories = t,
                        continuitylossweight = 100)
                    r = calibrate(invprob, alg)
                    @test only(r)≈4 rtol=1e-4
                end
            end
        end
        @testset "Random Initialization" begin
            @testset "trajectories = $t" for t in [1, 3]
                alg = MultipleShooting(maxiters = 100,
                    trajectories = t,
                    continuitylossweight = 100)
                r = calibrate(invprob, alg)
                @test only(r)≈4 rtol=1e-4
            end
        end
    end
end

end # CalibrateTests
