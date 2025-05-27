using DyadModelOptimizer
using DyadModelOptimizer: get_internal_storage, get_saveat
using OrdinaryDiffEqTsit5
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using SteadyStateDiffEq
using SciMLBase: successful_retcode
using SymbolicIndexingInterface
using DataFrames
using Test
using JET

include("../reactionsystem.jl")

getk1(sol) = getp(sol, :k1)(sol)
getc1(sol) = getp(sol, :c1)(sol)

function reactionsystem_local_alias()
    ps = @parameters k1=1.0 c1=2 c1_cond1=2.0 c1_cond2=2
    sts = @variables s1(t)=2.0 s1s2(t)=2.0 s2(t)=2c1
    eqs = [D_nounits(s1) ~ -0.25 * c1 * k1 * s1 * s2
           D_nounits(s1s2) ~ 0.25 * c1 * k1 * s1 * s2
           D_nounits(s2) ~ -0.25 * c1 * k1 * s1 * s2]

    return structural_simplify(ODESystem(eqs,
        t,
        sts,
        ps;
        name = :reactionsystem))
end

@testset "Aliasing" begin
    @testset "only alias params, both k and v in search space" begin
        model = reactionsystem_local_alias()
        @unpack c1, c1_cond1 = model
        params = [c1 => 3.0]
        data1 = generate_data(model; params)
        experiment1 = Experiment(data1, model)
        params = [c1 => 1.0]
        data2 = generate_data(model, (0.0, 2.0), 11; params)
        experiment2 = Experiment(
            data2, model; overrides = [c1 => c1_cond1], tspan = (0.0, 2))

        prob = InverseProblem([experiment1, experiment2],
            [c1 => (0, 5), c1_cond1 => (0, 5)])
        # This should solve the exact same problem as the data generation
        cost = objective(prob, SingleShooting(maxiters = 1))
        @test cost([3, 1.0]) == 0

        r = calibrate(prob, SingleShooting(maxiters = 100))
        @test r.u≈[3, 1.0] rtol=1e-3
    end

    @testset "only alias params, k in search space" begin
        model = reactionsystem_local_alias()
        @unpack c1, c1_cond1, k1 = model
        params = [c1 => 3.0]
        data1 = generate_data(model; params)
        experiment1 = Experiment(data1, model)
        params = [c1 => 2.0] # c1_cond1 defaults to 2
        data2 = generate_data(model, (0.0, 2.0), 11; params)
        experiment2 = Experiment(
            data2, model; overrides = [c1 => c1_cond1], tspan = (0.0, 2))

        prob = InverseProblem([experiment1, experiment2], [c1 => (0, 5)])
        # This should solve the exact same problem as the data generation
        alg = SingleShooting(maxiters = 1)
        cost = objective(prob, alg)

        @testset "individual experiment setup" begin
            sol1 = simulate(experiment1, prob, [3.0])
            @test getk1(sol1) == 1
            @test getc1(sol1) == 3
            @test sol1[unknowns(model), 1] == collect(data1[1, 2:end])
            sol2 = simulate(experiment2, prob, [3.0])
            @test getc1(sol2) == 2
            @test sol2[unknowns(model), 1] == collect(data2[1, 2:end])
        end
        @test cost([3.0]) == 0

        r = calibrate(prob, SingleShooting(maxiters = 100))
        @test r.u≈[3] rtol=1e-4
    end

    @testset "only alias params, k in search space, v fixed" begin
        model = reactionsystem_local_alias()
        @unpack c1, c1_cond1 = model
        params = [c1 => 3.0]
        data1 = generate_data(model; params)
        experiment1 = Experiment(data1, model)
        params = [c1 => 1.5]
        data2 = generate_data(model, (0.0, 2.0), 11; params)
        experiment2 = Experiment(
            data2, model; overrides = [c1 => c1_cond1, c1_cond1 => 1.5],
            tspan = (0.0, 2))

        prob = InverseProblem([experiment1, experiment2], [c1 => (0, 5)])
        # This should solve the exact same problem as the data generation
        alg = SingleShooting(maxiters = 1)
        cost = objective(prob, alg)

        @testset "individual experiment setup" begin
            sol1 = simulate(experiment1, prob, [3.0])
            @test getc1(sol1) == 3.0
            @test sol1[unknowns(model), 1] == collect(data1[1, 2:end])
            sol2 = simulate(experiment2, prob, [2.5])
            # deoesn't matter what's in  x, c1 is fixed
            @test getc1(sol2) == 1.5
            @test sol2[unknowns(model), 1] == collect(data2[1, 2:end])
        end
        @test cost([3.0]) == 0

        r = calibrate(prob, SingleShooting(maxiters = 100))
        @test r.u≈[3] rtol=1e-4
    end

    @testset "only alias params, k and v not in search space" begin
        model = reactionsystem_local_alias()
        @unpack c1, c1_cond1, k1 = model
        params = [c1 => 3.0, k1 => 2.5]
        data1 = generate_data(model; params)
        experiment1 = Experiment(data1, model, overrides = [c1 => 3])
        params = [c1 => 2.0, k1 => 2.5]
        data2 = generate_data(model, (0.0, 2.0), 11; params)
        experiment2 = Experiment(
            data2, model; overrides = [c1 => c1_cond1], tspan = (0.0, 2))

        prob = InverseProblem([experiment1, experiment2], [k1 => (0, 5)])
        # This should solve the exact same problem as the data generation
        alg = SingleShooting(maxiters = 1)
        cost = objective(prob, alg)

        @testset "individual experiment setup" begin
            sol1 = simulate(experiment1, prob, [2.5])
            @test getk1(sol1) == 2.5
            @test getc1(sol1) == 3.0
            @test sol1[unknowns(model), 1] == collect(data1[1, 2:end])
            sol2 = simulate(experiment2, prob, [2.5])
            @test getk1(sol2) == 2.5
            @test sol2[unknowns(model), 1] == collect(data2[1, 2:end])
        end
        @test cost([2.5]) == 0

        r = calibrate(prob, SingleShooting(maxiters = 100))
        @test r.u≈[2.5] rtol=1e-4
    end

    @testset "only alias params, v in search space" begin
        model = reactionsystem_local_alias()
        @unpack c1, c1_cond1 = model
        params = [c1 => 3.0]
        data1 = generate_data(model; params)
        experiment1 = Experiment(data1, model; overrides = params)
        params = [c1 => 1.0]
        data2 = generate_data(model, (0.0, 2.0), 11; params)
        experiment2 = Experiment(data2, model; overrides = [c1 => c1_cond1, c1_cond1 => 1],
            tspan = (0.0, 2))

        prob = InverseProblem([experiment1, experiment2], [c1_cond1 => (0, 5)])
        # This should solve the exact same problem as the data generation
        alg = SingleShooting(maxiters = 1)
        cost = objective(prob, alg)

        @testset "individual experiment setup" begin
            sol1 = simulate(experiment1, prob, [3.0])
            @test getc1(sol1) == 3
            @test sol1[unknowns(model), 1] == collect(data1[1, 2:end])
            sol2 = simulate(experiment2, prob, [1])
            @test getc1(sol2) == 1
            @test sol2[unknowns(model), 1] == collect(data2[1, 2:end])
        end
        @test cost([1]) == 0
        # no calibration, we test here that we respect the priority
        # defs < optimized < locally fixed
    end

    @testset "only alias params, v in search space, concrete defaults" begin
        ps = @parameters k1=1.0 c1=2.0 c1_cond1=2.0
        sts = @variables s1(t)=2.0 s1s2(t)=2.0 s2(t)=2.0
        eqs = [D_nounits(s1) ~ -0.25 * c1 * k1 * s1 * s2
               D_nounits(s1s2) ~ 0.25 * c1 * k1 * s1 * s2
               D_nounits(s2) ~ -0.25 * c1 * k1 * s1 * s2]

        model = structural_simplify(ODESystem(eqs,
            t,
            sts,
            ps;
            name = :reactionsystem))

        @unpack c1, c1_cond1 = model
        params = [c1 => 3.0]
        data1 = generate_data(model; params)
        experiment1 = Experiment(data1, model; overrides = [c1 => c1_cond1])

        prob = InverseProblem(experiment1, [c1_cond1 => (0, 5)])
        # This should solve the exact same problem as the data generation
        alg = SingleShooting(maxiters = 1)
        cost = objective(prob, alg)

        @testset "individual experiment setup" begin
            sol1 = simulate(experiment1, prob, [3.0])
            @test getc1(sol1) == 3
            @test sol1[unknowns(model), 1] == collect(data1[1, 2:end])
        end
        @test cost([3]) == 0
        r = calibrate(prob, SingleShooting(maxiters = 10))
        @test only(r)≈3 rtol=1e-4
    end

    @testset "only alias params, k not used" begin
        model = reactionsystem_local_alias()
        @unpack c1, c1_cond1, c1_cond2 = model
        params = [c1 => 3.0]
        data1 = generate_data(model; params)
        experiment1 = Experiment(data1, model; overrides = [c1 => c1_cond1])
        params = [c1 => 1.0]
        data2 = generate_data(model, (0.0, 2.0), 11; params)
        experiment2 = Experiment(data2, model; overrides = [c1 => c1_cond2],
            tspan = (0.0, 2))

        prob = InverseProblem([experiment1, experiment2],
            [c1_cond1 => (0, 5),
                c1_cond2 => (0, 5)])
        # This should solve the exact same problem as the data generation
        alg = SingleShooting(maxiters = 1)
        cost = objective(prob, alg)

        @testset "individual experiment setup" begin
            sol1 = simulate(experiment1, prob, [3.0, 1])
            @test getc1(sol1) == 3
            @test sol1[unknowns(model), 1] == collect(data1[1, 2:end])
            sol2 = simulate(experiment2, prob, [3, 1])
            @test getc1(sol2) == 1
            @test sol2[unknowns(model), 1] == collect(data2[1, 2:end])
        end
        @test cost([3, 1]) == 0

        r = calibrate(prob, SingleShooting(maxiters = 100))
        @test r.u≈[3, 1.0] rtol=1e-3
    end

    @testset "mix aliasig with fixing params" begin
        model = reactionsystem_local_alias()
        @unpack c1, c1_cond1, k1 = model
        params = [c1 => 3.0, k1 => 0.9]
        data1 = generate_data(model; params)
        experiment1 = Experiment(data1, model, overrides = [k1 => 0.9])
        params = [c1 => 1.0, k1 => 1.5]
        data2 = generate_data(model, (0.0, 2.0), 11; params)
        experiment2 = Experiment(data2, model; overrides = [c1 => c1_cond1, k1 => 1.5])

        prob = InverseProblem([experiment1, experiment2],
            [c1 => (0, 5), c1_cond1 => (0, 5)])
        # This should solve the exact same problem as the data generation
        alg = SingleShooting(maxiters = 1)
        cost = objective(prob, alg)

        @testset "individual experiment setup" begin
            sol1 = simulate(experiment1, prob, [3.0, 0.9])
            @test getc1(sol1) == 3
            @test getk1(sol1) == 0.9
            @test sol1[unknowns(model), 1] == collect(data1[1, 2:end])
            sol2 = simulate(experiment2, prob, [1, 1.0])
            @test getc1(sol2) == 1.0
            @test getk1(sol2) == 1.5
            # data was generated with c1 = 1
            @test sol2[unknowns(model), 1] == collect(data2[1, 2:end])
        end

        @test cost([3, 1.0]) == 0

        r = calibrate(prob, SingleShooting(maxiters = 100))
        @test r.u≈[3, 1.0] rtol=1e-4
    end

    @testset "u0" begin
        model = reactionsystem_local_alias()
        @unpack s1, c1_cond1 = model
        u0 = [s1 => 3.0]
        data1 = generate_data(model; u0)
        experiment1 = Experiment(data1, model)
        u0 = [s1 => 1.0]
        data2 = generate_data(model, (0.0, 2.0), 11; u0)
        # technically initial conditions are still some parameters,
        # so we alias them to parameters
        experiment2 = Experiment(data2, model; overrides = [s1 => c1_cond1])

        prob = InverseProblem([experiment1, experiment2],
            [Initial(s1) => (0, 5), c1_cond1 => (0, 5)])
        # This should solve the exact same problem as the data generation
        cost = objective(prob, SingleShooting(maxiters = 1))
        @test cost([3, 1.0]) == 0

        r = calibrate(prob, SingleShooting(maxiters = 100))
        @test r.u≈[3, 1.0] rtol=1e-4
    end

    @testset "u0 and params" begin
        model = reactionsystem_local_alias()
        @unpack s1, c1, c1_cond1, k1 = model
        u0 = [s1 => 3.0]
        params = [c1 => 3.0]
        data1 = generate_data(model; u0, params)
        experiment1 = Experiment(data1, model; overrides = params)
        u0 = [s1 => 1.0]
        params = [c1 => 1.0]
        data2 = generate_data(model, (0.0, 2.0), 11; u0, params)
        # technically initial conditions are still some parameters,
        # so we alias them to parameters
        experiment2 = Experiment(data2,
            model;
            overrides = [s1 => c1_cond1, c1 => c1_cond1])

        prob = InverseProblem([experiment1, experiment2],
            [Initial(s1) => (0, 5), c1_cond1 => (0, 5)])
        # This should solve the exact same problem as the data generation
        alg = SingleShooting(maxiters = 1)

        @testset "individual experiment setup" begin
            sol1 = simulate(experiment1, prob, [3.0, 3])
            @test getc1(sol1) == 3
            @test getk1(sol1) == 1
            @test sol1[unknowns(model), 1] == collect(data1[1, 2:end])
            @test cost_contribution(alg, experiment1, prob, [3, 1]) == 0
            sol2 = simulate(experiment2, prob, [1, 1.0])
            @test getc1(sol2) == 1.0
            @test getk1(sol2) == 1.0
            @test sol2[unknowns(model), 1] == collect(data2[1, 2:end])
            @test cost_contribution(alg, experiment2, prob, [3, 1]) == 0
            sol2 = simulate(experiment2, prob, [4, 1.3])
            # test that the value from the optimizer is ignored and that
            # the value of c1_cond1 is used
            @test sol2[s1, 1] == 1.3
        end

        cost = objective(prob, alg)
        @test cost([3.0, 1.0]) == 0
    end
end

@testset "Parametrized initial conditions" begin
    @testset "parameter in search space" begin
        ps = @parameters k1=1.0 c1=2.0
        sts = @variables s1(t)=2.0 s1s2(t)=2.0 s2(t)=2 * k1
        eqs = [D_nounits(s1) ~ -0.25 * c1 * k1 * s1 * s2
               D_nounits(s1s2) ~ 0.25 * c1 * k1 * s1 * s2
               D_nounits(s2) ~ -0.25 * c1 * k1 * s1 * s2]

        model = structural_simplify(ODESystem(eqs,
            t,
            sts,
            ps;
            name = :reactionsystem))

        data1 = generate_data(model, params = [k1 => 3.2])
        experiment1 = Experiment(data1, model, tspan = (0.0, 1.0))

        alg = SingleShooting(maxiters = 100)
        prob = InverseProblem(experiment1, [k1 => (0, 4)])

        cost = objective(prob, alg)
        @test iszero(cost([3.2]))

        sol = simulate(experiment1, prob, [3.0])
        @test sol[s2, 1] == 6

        r = calibrate(prob, alg)
        @test only(r)≈3.2 rtol=1e-6
    end

    @testset "parameter not in search space" begin
        ps = @parameters k1=1.0 c1=2.0
        sts = @variables s1(t)=2.0 s1s2(t)=2.0 s2(t)=2 * k1
        eqs = [D_nounits(s1) ~ -0.25 * c1 * k1 * s1 * s2
               D_nounits(s1s2) ~ 0.25 * c1 * k1 * s1 * s2
               D_nounits(s2) ~ -0.25 * c1 * k1 * s1 * s2]

        model = structural_simplify(ODESystem(eqs,
            t,
            sts,
            ps;
            name = :reactionsystem))

        data1 = generate_data(model, u0 = [s1 => 2.1])
        experiment1 = Experiment(data1, model, tspan = (0.0, 1.0))

        alg = SingleShooting(maxiters = 100)
        prob = InverseProblem(experiment1, [Initial(s1) => (0, 4)])

        cost = objective(prob, alg)
        @test iszero(cost([2.1]))

        sol = simulate(experiment1, prob)
        @test sol[s2, 1] == 2.0
        sol = simulate(experiment1, prob, [1.23])
        @test sol[s1, 1] == 1.23

        r = calibrate(prob, alg)
        @test only(r)≈2.1 rtol=1e-6
    end

    @testset "parameter not in search space, default value is an expression" begin
        ps = @parameters c1=2.0 k1=c1 / 2
        sts = @variables s1(t)=2.0 s1s2(t)=2.0 s2(t)=2 * k1
        eqs = [D_nounits(s1) ~ -0.25 * c1 * k1 * s1 * s2
               D_nounits(s1s2) ~ 0.25 * c1 * k1 * s1 * s2
               D_nounits(s2) ~ -0.25 * c1 * k1 * s1 * s2]

        model = structural_simplify(ODESystem(eqs,
            t,
            sts,
            ps;
            name = :reactionsystem))

        data1 = generate_data(model, u0 = [s1 => 2.5])
        experiment1 = Experiment(data1, model, tspan = (0.0, 1.0))

        alg = SingleShooting(maxiters = 100)
        prob = InverseProblem(experiment1, [Initial(s1) => (0, 4)])

        cost = objective(prob, alg)
        @test iszero(cost([2.5]))

        sol = simulate(experiment1, prob)
        @test sol[s2, 1] == 2

        r = calibrate(prob, alg)
        @test only(r)≈2.5 rtol=1e-6
    end

    @testset "parameter not in search space, experiment sets parameter" begin
        ps = @parameters c1=2.0 k1=2.0
        sts = @variables s1(t)=2.0 s1s2(t)=2.0 s2(t)=2 * k1
        eqs = [D_nounits(s1) ~ -0.25 * c1 * k1 * s1 * s2
               D_nounits(s1s2) ~ 0.25 * c1 * k1 * s1 * s2
               D_nounits(s2) ~ -0.25 * c1 * k1 * s1 * s2]

        model = structural_simplify(ODESystem(eqs,
            t,
            sts,
            ps;
            name = :reactionsystem))

        data1 = generate_data(model, params = [k1 => 3], u0 = [s1 => 2.5])
        experiment1 = Experiment(data1, model, tspan = (0.0, 1.0), overrides = [k1 => 3])

        alg = SingleShooting(maxiters = 100)
        prob = InverseProblem(experiment1, [Initial(s1) => (0, 4)])

        x = DyadModelOptimizer.initial_state(alg, prob)
        sol = simulate(experiment1, prob, x)
        @test sol[s2, 1] == 6 # 2*k1

        cost = objective(prob, alg)
        @test iszero(cost([2.5]))
        r = calibrate(prob, alg)
        @test only(r)≈2.5 rtol=1e-6
    end

    @testset "parameter in search space, experiment sets parameter" begin
        ps = @parameters c1=2.0 k1=2.0
        sts = @variables s1(t)=2.0 s1s2(t)=2.0 s2(t)=2 * k1
        eqs = [D_nounits(s1) ~ -0.25 * c1 * k1 * s1 * s2
               D_nounits(s1s2) ~ 0.25 * c1 * k1 * s1 * s2
               D_nounits(s2) ~ -0.25 * c1 * k1 * s1 * s2]

        model = structural_simplify(ODESystem(eqs, t; name = :reactionsystem))

        data1 = generate_data(model, params = [k1 => 3], u0 = [s1 => 2.5])
        experiment1 = Experiment(data1, model, tspan = (0.0, 1.0), overrides = [k1 => 3])
        data1 = generate_data(model)
        experiment2 = Experiment(data1, model, tspan = (0.0, 1.0))

        alg = SingleShooting(maxiters = 100)
        prob = InverseProblem(experiment1, [Initial(s1) => (0, 4), k1 => (0, 5)])

        x = [0, 0.0]
        sol = simulate(experiment1, prob, x)
        @test sol[unknowns(model)[end], 1] == 6 # 2*k1

        cost = objective(prob, alg)
        @test iszero(cost([2.5]))
        r = calibrate(prob, alg)
        @test first(r)≈2.5 rtol=1e-6
    end

    @testset "make initial condition parametric in one experiment, parameter in search space" begin
        ps = @parameters k = 2
        sts = @variables x(t) = 1.5

        eqs = [
            D(x) ~ -k * x
        ]
        model = structural_simplify(ODESystem(eqs, t, sts, ps; name = :model))

        data = generate_data(model, params = [k => 3], u0 = [x => 3])
        tspan = (0.0, 1.0)

        experiment = Experiment(data, model, overrides = [x => k])
        ss = [k => (0.1, 10.0)]
        prob = InverseProblem(experiment, ss)
        sol = simulate(experiment, prob, [1.0])

        @test sol.u[1] == 1

        r = calibrate(prob, SingleShooting(maxiters = 100))
        @test only(r)≈3 rtol=1e-6
    end

    @testset "parametrized param" begin
        ps = @parameters k1=1.0 c1=k1 / 2
        sts = @variables s1(t)=2.0 s1s2(t)=2.0 s2(t)=2 * k1
        eqs = [D_nounits(s1) ~ -0.25 * c1 * k1 * s1 * s2
               D_nounits(s1s2) ~ 0.25 * c1 * k1 * s1 * s2
               D_nounits(s2) ~ -0.25 * c1 * k1 * s1 * s2]

        model = structural_simplify(ODESystem(eqs, t; name = :reactionsystem))

        data1 = generate_data(model, params = [c1 => 3])
        experiment1 = Experiment(data1, model, tspan = (0.0, 1.0))

        alg = SingleShooting(maxiters = 10)
        prob = InverseProblem(experiment1, [c1 => (0, 4)])

        cost = objective(prob, alg)
        @test iszero(cost([3]))

        sol = simulate(experiment1, prob, [3.0])
        @test sol[s2, 1] == 2
        @test sol.ps[[k1, c1]] == [1, 3]

        r = calibrate(prob, alg)
        @test only(r)≈3 rtol=1e-5
    end
end

@testset "Non-independent parameters" begin
    @testset "parameter in search space" begin
        ps = @parameters k1=1.0 c1=2.0 k2
        sts = @variables s1(t)=2.0 s1s2(t)=2.0 s2(t)=2
        eqs = [D_nounits(s1) ~ -0.25 * c1 * k1 * s1 * s2
               D_nounits(s1s2) ~ 0.25 * c1 * k1 * s1 * s2
               D_nounits(s2) ~ -0.25 * c1 * k2 * s1 * s2]

        model = structural_simplify(ODESystem(eqs,
            t,
            sts,
            ps;
            parameter_dependencies = [k2 => 2k1],
            name = :reactionsystem))

        data1 = generate_data(model, params = [k1 => 1.5])
        experiment = Experiment(data1, model, tspan = (0.0, 1.0))

        alg = SingleShooting(maxiters = 100)
        prob = InverseProblem(experiment, [k1 => (0, 4)])

        cost = objective(prob, alg)
        @test iszero(cost([1.5]))

        sol = simulate(experiment, prob, [3.0])
        @test sol.ps[k2] == 6
        @test sol[s2] ≠ sol[s1]

        r = calibrate(prob, alg)
        @test only(r)≈1.5 rtol=1e-5
    end

    @testset "parameter fixed by experiment" begin
        ps = @parameters k1=1.0 c1=2.0 k2
        sts = @variables s1(t)=2.0 s1s2(t)=2.0 s2(t)=2
        eqs = [D_nounits(s1) ~ -0.25 * c1 * k1 * s1 * s2
               D_nounits(s1s2) ~ 0.25 * c1 * k1 * s1 * s2
               D_nounits(s2) ~ -0.25 * c1 * k2 * s1 * s2]

        model = structural_simplify(ODESystem(
            eqs, t, sts, ps; parameter_dependencies = [k2 => 2k1],
            name = :reactionsystem))

        data1 = generate_data(model)
        experiment = Experiment(data1, model, overrides = [k1 => 3], tspan = (0.0, 1.0))

        alg = SingleShooting(maxiters = 1)
        prob = InverseProblem(experiment, [c1 => (0, 4)])

        sol = simulate(experiment, prob, [2])
        @test sol.ps[[k1, c1, k2]] == [3, 2, 6]
    end
end
