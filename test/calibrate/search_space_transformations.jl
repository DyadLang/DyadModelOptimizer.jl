using Test
using DyadModelOptimizer
using DyadModelOptimizer: get_params, get_saveat, get_model,
                          get_data, get_saveat, get_solve_alg, get_config,
                          initial_state, timespan, lowerbound, upperbound,
                          apply_ss_transform
using DataFrames
using ModelingToolkit
using Optimization
using OptimizationNLopt
using OptimizationOptimJL, OptimizationBBO
using OptimizationMOI, Ipopt
using JET
using DataInterpolations
using SciMLBase: successful_retcode
using SymbolicIndexingInterface

include("../reactionsystem.jl")

@testset "SingleShooting" begin
    @testset "log transformed 1 element search space" begin
        model = reactionsystem()
        @unpack c1 = model

        data = generate_data(model, params = [c1 => 3])

        experiment = Experiment(data, model, tspan = (0.0, 1.0))
        invprob = InverseProblem(experiment, [c1 => (0, 5)])

        alg = SingleShooting(maxiters = 10^3)

        c = cost_contribution(alg, experiment, invprob, [3])
        # this should solve the same problem as the data generation
        @test iszero(c)

        r = calibrate(invprob, alg)

        @test only(r)≈3 rtol=1e-4

        invprob_log = InverseProblem(experiment, [c1 => (0, 5, :log)])

        # @test only(get_transformed_states(invprob_log)) == log(2)
        # @test only(get_states(invprob_log)) == 2  # Check `get_transformed_states` did not mutate states
        x0 = initial_state(alg, invprob)
        x0′ = initial_state(alg, invprob_log)
        @test apply_ss_transform(:inverse, x0′, invprob_log) == x0

        c2 = cost_contribution(alg, experiment, invprob_log, [log(3)])
        @test c2≈0 atol=1e-20

        r2 = calibrate(invprob_log, alg)

        @test only(r2)≈3 rtol=1e-4

        # cross check with parametric_uq
        ps = parametric_uq(invprob,
            StochGlobalOpt(method = SingleShooting(maxiters = 10^3)), progress = nothing, sample_size = 1)
        @test only(ps[1])≈3 rtol=1e-4

        @testset "invalid bounds" begin
            @test_throws DomainError InverseProblem(experiment,
                [c1 => (-1, 5, :log)])
            @test_throws DomainError InverseProblem(experiment,
                [c1 => (-10, -5, :log)])
            @test_throws ArgumentError InverseProblem(experiment, [c1 => (5, 1)])
        end
    end

    @testset "log transformed 2 element search space" begin
        model = reactionsystem()
        @unpack c1, k1 = model

        data = generate_data(model, params = [c1 => 3, k1 => 1.5])

        experiment = Experiment(data, model, tspan = (0.0, 1.0))
        invprob = InverseProblem(experiment, [c1 => (0, 5), k1 => (1, 2)])

        alg = SingleShooting(maxiters = 10^3)

        c = cost_contribution(alg, experiment, invprob, [3, 1.5])
        @test c≈0 atol=1e-7

        r = calibrate(invprob, alg)

        # c1 and k1 are not identifiable, but c1*k1 is
        @test r[:c1] * r[:k1]≈4.5 rtol=1e-4

        @testset "log log" begin
            invprob_log = InverseProblem(experiment,
                [c1 => (0, 5, :log), k1 => (1, 2, :log)])

            x0 = initial_state(alg, invprob)
            x0′ = initial_state(alg, invprob_log)
            @test apply_ss_transform(:inverse, x0′, invprob_log) == x0

            c2 = cost_contribution(alg, experiment, invprob_log, [log(3), log(1.5)])
            @test c2≈0 atol=1e-20

            r2 = calibrate(invprob_log, alg)

            # c1 and k1 are not identifiable, but c1*k1 is
            @test r2[:c1] * r2[:k1]≈4.5 rtol=1e-4
        end

        @testset "identity log" begin
            invprob_log = InverseProblem(experiment,
                [c1 => (0, 5), k1 => (1, 2, :log)])

            x0 = initial_state(alg, invprob)
            x0′ = initial_state(alg, invprob_log)
            @test apply_ss_transform(:inverse, x0′, invprob_log) == x0

            c2 = cost_contribution(alg, experiment, invprob_log, [3, log(1.5)])
            @test c2≈0 atol=1e-20

            r2 = calibrate(invprob_log, alg)

            # c1 and k1 are not identifiable, but c1*k1 is
            @test r2[:c1] * r2[:k1]≈4.5 rtol=1e-4
        end

        @testset "parametric_uq cross check" begin
            invprob_log = InverseProblem(experiment,
                [c1 => (0, 5), k1 => (1, 2, :log)])

            ps = parametric_uq(invprob_log,
                StochGlobalOpt(method = SingleShooting(maxiters = 10^4, maxtime = 100,
                    optimizer = NLopt.G_MLSL_LDS(),
                    local_method = NLopt.LN_NELDERMEAD())), sample_size = 1)
            @test ps[1][:k1] * ps[1][:c1]≈4.5 rtol=1e-4
        end

        @testset "x is not changed" begin
            invprob_log = InverseProblem(experiment,
                [c1 => (0, 5), k1 => (1, 2, :log)])

            sol1 = simulate(experiment, invprob_log)
            sol2 = simulate(experiment, invprob_log)

            @test all(sol1.u .== sol2.u)
            @test parameter_values(sol1) == parameter_values(sol2)

            x = [c1 => 3, k1 => 1.5]
            sol1 = simulate(experiment, invprob_log, x)
            @test all(isequal.(x, [c1 => 3, k1 => 1.5]))

            x = [3, 1.5]
            sol1 = simulate(experiment, invprob_log, x)
            @test x == [3, 1.5]
        end

        @testset "ordering of transformations" begin
            function reactionsystem()
                t = t_nounits
                sts = @variables s1(t)=1.2 s1s2(t)=2.0 s2(t)=2.0
                ps = @parameters k1=-1.0 c1=2.5
                eqs = [(Differential(t))(s1) ~ -0.25 * c1 * k1 * s1 * s2
                       (Differential(t))(s1s2) ~ 0.25 * c1 * k1 * s1 * s2
                       (Differential(t))(s2) ~ -0.25 * c1 * k1 * s1 * s2]

                return complete(ODESystem(eqs, t; name = :reactionsystem))
            end

            model = reactionsystem()

            @unpack s1, s2 = model

            data = generate_data(model, params = [s1 => 1.3, c1 => 3, k1 => 1.0])
            experiment = Experiment(data, model, tspan = (0.0, 1.0))

            invprob = InverseProblem(experiment,
                [k1 => (-5, 5), Initial(s1) => (1, 3), c1 => (2, 5)])
            invprob_log = InverseProblem(experiment,
                [k1 => (-5, 5.0), Initial(s1) => (0.01, 5.0, :log), c1 => (2, 5)])

            alg = SingleShooting(maxiters = 100,
                optimizer = IpoptOptimizer(;
                    max_iter = 100,
                    tol = 1e-8,
                    acceptable_tol = 1e-8))

            x0 = initial_state(alg, invprob) # s1 k1 c1
            x0′ = initial_state(alg, invprob_log)
            @test apply_ss_transform(:inverse, x0′, invprob_log) == x0

            c2 = cost_contribution(
                alg, experiment, invprob_log, [k1 => 1.0, c1 => 3, Initial(s1) => log(1.3)])
            @test iszero(c2)

            r1 = calibrate(invprob, alg)
            r2 = calibrate(invprob_log, alg)

            @test r1.original.objective < 1e-8
            @test r2.original.objective < 1e-8

            @testset "correct results" begin
                @test r1[Symbol("Initial(s1(t))")]≈1.3 rtol=1e-4
                @test r1[:k1] * r1[:c1]≈3 rtol=1e-4

                @test r2[Symbol("Initial(s1(t))")]≈1.3 rtol=1e-4
                @test r2[:k1] * r2[:c1]≈3 rtol=1e-4
            end

            sol1 = simulate(experiment, r1)
            @test successful_retcode(sol1.retcode)
            sol2 = simulate(experiment, r2)
            @test successful_retcode(sol2.retcode)
            sol3 = simulate(experiment, invprob_log, r2)
            @test successful_retcode(sol3.retcode)
            @test all(isapprox.(sol1.u, sol2.u, rtol = 1e-5))
            @test all(isapprox.(sol1.u, sol3.u, rtol = 1e-5))
            # cross check with parametric_uq
            ps = parametric_uq(invprob,
                StochGlobalOpt(method = SingleShooting(maxiters = 10^4, maxtime = 100,
                    optimizer = NLopt.G_MLSL_LDS(),
                    local_method = NLopt.LN_NELDERMEAD())), sample_size = 1)
            @test ps[1][:k1] * ps[1][:c1]≈3 rtol=1e-4
            @test ps[1][Symbol("Initial(s1(t))")]≈1.3 rtol=1e-4
        end
    end
end

@testset "MultipleShooting" begin
    @testset "data saveat does not coincide with shooting interval endpoints" begin
        model = reactionsystem()
        data = generate_data(model, (0.0, 1), 11)

        @unpack c1 = model

        experiment = Experiment(data, model, tspan = (0.0, 1.0))
        invprob = InverseProblem(experiment, [c1 => (0, 5)])

        alg = MultipleShooting(maxiters = 10^5, trajectories = 3,
            continuitylossweight = 10^2)

        c = cost_contribution(alg, experiment, invprob, initial_state(alg, invprob))
        # the initial internal u0s are bad guesses, so just do a sanity check
        @test c > 0

        x0 = initial_state(alg, invprob)
        # the internal params are not transformed
        @test x0[2:end] == DyadModelOptimizer.internal_params(alg, invprob)

        r = calibrate(invprob, alg)
        @test only(r)≈2 rtol=1e-4

        invprob_log = InverseProblem(experiment, [c1 => (0, 5, :log10)])

        r_log = calibrate(invprob_log, alg)
        @test only(r_log)≈2 rtol=1e-4

        # cross check with parametric_uq
        ps = parametric_uq(invprob,
            StochGlobalOpt(method = SingleShooting(maxiters = 10^3,
                optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited())),
            sample_size = 1)
        @test only(ps[1])≈2 rtol=1e-4
    end
end
