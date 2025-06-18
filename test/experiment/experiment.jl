using DyadModelOptimizer
using DyadModelOptimizer: get_saveat, initial_state
using OrdinaryDiffEqVerner, OrdinaryDiffEqTsit5
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using SteadyStateDiffEq
using SciMLBase: successful_retcode
using Distributions
using DataFrames
using Test
using JET
using DyadInterface

include("../reactionsystem.jl")

@testset "One experiment" begin
    model = reactionsystem()
    data = generate_data(model)
    experiment = Experiment(data, model)

    @test get_saveat(experiment) == first(eachcol(data))
    @test DyadModelOptimizer.timespan(experiment) == (0.0, 1.0)

    @unpack k1, c1 = model

    @testset "Optimize all parameters" begin
        prob = InverseProblem(experiment, [k1 => (0, 5), c1 => (0, 5)])

        cost = objective(prob, SingleShooting(maxiters = 1))
        @test cost() == 0
    end

    @testset "Optimize one parameter" begin
        prob = InverseProblem(experiment, [k1 => (0, 5)])

        alg = SingleShooting(maxiters = 1)
        cost = objective(prob, alg)
        @test cost() == 0
        @test initial_state(alg, prob) == [ModelingToolkit.defaults(model)[k1]]
    end

    @testset "Update parameter depending on another parameter via pdesp" begin
        ps = @parameters k=2 a
        sts = @variables x(t) = 1.5

        eqs = [
            D(x) ~ -a * x
        ]
        model = structural_simplify(ODESystem(
            eqs, t, sts, ps; name = :model, parameter_dependencies = [a ~ k]))

        data = DataFrame(timestamp = [0.0, 1.0], x = [1.0, 0.35])
        tspan = (0.0, 1.0)

        experiment = Experiment(data, model)
        ss = [k => (0.1, 10.0)]
        prob = InverseProblem(experiment, ss)
        sol = simulate(experiment, prob, [1.0])

        @test sol.ps[a] == 1
    end

    @testset "Update parameter depending on another parameter via defaults" begin
        ps = @parameters k=2 a
        sts = @variables x(t) = 1.5

        eqs = [
            D(x) ~ -a * x
        ]
        model = structural_simplify(ODESystem(
            eqs, t, sts, ps; name = :model, defaults = Dict(a => k)))

        data = DataFrame(timestamp = [0.0, 1.0], x = [1.0, 0.35])
        tspan = (0.0, 1.0)

        experiment = Experiment(data, model)
        ss = [k => (0.1, 10.0)]
        prob = InverseProblem(experiment, ss)
        sol = simulate(experiment, prob, [1.0])

        @test sol.ps[a]==1 broken=true
    end

    @testset "Update parameter depending on another parameter via solvable params" begin
        ps = @parameters k=2 a=k [guess = k]
        sts = @variables x(t) = 1.5

        eqs = [
            D(x) ~ -a * x
        ]
        model = structural_simplify(ODESystem(
            eqs, t, sts, ps; name = :model))

        data = DataFrame(timestamp = [0.0, 1.0], x = [1.0, 0.35])
        tspan = (0.0, 1.0)

        experiment = Experiment(data, model)
        ss = [k => (0.1, 10.0)]
        prob = InverseProblem(experiment, ss)
        sol = simulate(experiment, prob, [1.0])

        @test sol.ps[a] == 1
    end

    @testset "Update parameter depending on another parameter via overrides" begin
        ps = @parameters k=2 a=5
        sts = @variables x(t) = 1.5

        eqs = [
            D(x) ~ -a * x
        ]
        model = structural_simplify(ODESystem(eqs, t, sts, ps; name = :model))

        data = DataFrame(timestamp = [0.0, 1.0], x = [1.0, 0.35])
        tspan = (0.0, 1.0)

        experiment = Experiment(data, model, overrides = [a => k])
        ss = [k => (0.1, 10.0)]
        prob = InverseProblem(experiment, ss)
        sol = simulate(experiment, prob, [1.0])

        @test sol.ps[a] == 1
    end

    @testset "Update initial condition" begin
        ps = @parameters k = 2
        sts = @variables x(t) = 1.5

        eqs = [
            D(x) ~ -k * x
        ]
        model = structural_simplify(ODESystem(eqs, t, sts, ps; name = :model))

        data = DataFrame(timestamp = [0.0, 1.0], x = [1.0, 0.35])
        tspan = (0.0, 1.0)

        experiment = Experiment(data, model)
        ss = [Initial(x) => (0.1, 10.0)]
        prob = InverseProblem(experiment, ss)
        sol = simulate(experiment, prob, [1.0])

        @test sol.u[1] == 1
    end

    @testset "Update initial condition, trivial initialization" begin
        ps = @parameters k=missing [guess = 2]
        sts = @variables x(t) = 1.5

        eqs = [
            D(x) ~ -k * x
        ]
        model = structural_simplify(ODESystem(
            eqs, t, sts, ps; name = :model, initialization_eqs = [k ~ 2]))

        data = DataFrame(timestamp = [0.0, 1.0], x = [1.0, 0.35])
        tspan = (0.0, 1.0)

        experiment = Experiment(data, model)
        ss = [Initial(x) => (0.1, 10.0)]
        prob = InverseProblem(experiment, ss)
        sol = simulate(experiment, prob, [1.0])

        @test sol.u[1] == 1
    end

    @testset "MTK defined search space" begin
        model = reactionsystem()
        data = generate_data(model)
        experiment = Experiment(data, model)

        prob = InverseProblem(experiment, [])
        cost = objective(prob, SingleShooting(maxiters = 1))
        # Run JET error analysis
        @test_call target_modules=(DyadModelOptimizer,) cost()==0
        @test cost() == 0
        @test isequal(DyadModelOptimizer.determine_search_space(model), [c1 => (0, 2)])
    end

    @testset "empty search space" begin
        prob = InverseProblem(experiment, [])
        sol = simulate(experiment, prob)
        @test successful_retcode(sol)
    end

    @testset "save_names with observed states" begin
        model = reactionsystem_obs()
        @unpack k1, c1, s1, s3 = model
        tspan = (0.0, 1.0)
        prob = ODEProblem(model, [], tspan, [])
        saveat = range(prob.tspan..., length = 10)
        sol = solve(prob; abstol = 1e-8, reltol = 1e-8, saveat)

        data = DataFrame(:timestamp => sol.t, :s1 => sol[s1], :s3 => sol[s3])
        experiment = Experiment(data, model, tspan = (0.0, 1.0))

        invprob = InverseProblem(experiment, [k1 => (0, 5), c1 => (0, 5)])
        alg = SingleShooting(maxiters = 1)

        cost = objective(invprob, SingleShooting(maxiters = 1))
        @test cost() == 0
    end

    @testset "simulate Vector{Pair} interface" begin
        prob = InverseProblem(experiment, [k1 => (0, 5), c1 => (0, 5)])

        sol1 = simulate(experiment, prob, [k1 => 1.5, c1 => 4.0])
        sol2 = simulate(experiment, prob, [c1 => 4.0, k1 => 1.5])
        sol3 = simulate(experiment,
            prob,
            [k1 => 1.5, c1 => 4.0],
            tspan = (0.0, 2.0),
            saveat = 0.1)
        sol = simulate(experiment, prob, [1.5, 4.0])

        @test sol.u == sol1.u == sol2.u
        @test sol.t == sol1.t == sol2.t
        @test sol3.t[begin] == 0
        @test sol3.t[end] == 2
    end

    @testset "simulate Vector{Pair} interface with remake" begin
        prob = InverseProblem(experiment, [k1 => (0, 5), c1 => (0, 5)])

        sol1 = simulate(experiment, prob, [k1 => 1.5, c1 => 4.0], alg = Vern6())
        re_ex = remake(experiment, alg = Vern6())
        re_prob = InverseProblem(re_ex, [k1 => (0, 5), c1 => (0, 5)])
        sol = simulate(re_ex, re_prob, [1.5, 4.0])

        @test sol1.u == sol.u
        @test sol1.t == sol.t
    end

    @testset "ODEFunction kwargs" begin
        model = reactionsystem()
        data = generate_data(model)

        # no jac in model
        @test isempty(ModelingToolkit.get_jac(model)[])
        experiment = Experiment(data, model, prob_kwargs = (jac = true,))

        @test !isempty(ModelingToolkit.get_jac(get_model(experiment))[])
    end

    @testset "DesignConfiguration" begin
        model = reactionsystem()

        dc = DesignConfiguration(model,
            tspan = (0.0, 1),
            saveat = 0.1,
            running_cost = model.s1^2,
            data = (a = [1],))

        @test get_saveat(dc) == 0.1
        @test DyadModelOptimizer.get_data(dc) == (a = [1],)

        dc = DesignConfiguration(model,
            tspan = (0.0, 1),
            saveat = 0.1,
            constraints = [model.s1 ≲ 5],
            running_cost = model.s1^2)

        experiment = Experiment(nothing, model;
            tspan = (0.0, 1),
            saveat = 0.1,
            constraints = [model.s1 ≲ 5],
            loss_func = (x, sol, data) -> mean(sol[model.s1^2]))

        @test DyadModelOptimizer.get_constraints_ts(dc) == range(0, 1, step = 0.1)

        prob = InverseProblem(dc, [model.k1 => (0, 5)])
        sol = simulate(dc, prob)

        @test successful_retcode(sol)

        r1 = calibrate(prob, SingleShooting(maxiters = 100))

        prob_ref = InverseProblem(experiment, [model.k1 => (0, 5)])

        r2 = calibrate(prob, SingleShooting(maxiters = 100))

        @test r1 == r2
    end

    @testset "error handling" begin
        data = generate_data(model)

        experiment = Experiment(data,
            model,
            loss_func = (x, sol, data) -> error("test"),
            tspan = (0.0, 1.0))
        invprob = InverseProblem(experiment, [k1 => (0, 2.0)])

        @test_throws ErrorException("test") cost_contribution(SingleShooting(maxiters = 1),
            experiment,
            invprob)

        # indexing error in loss function
        experiment = Experiment(data,
            model,
            loss_func = (x, sol, data) -> x["test"],
            tspan = (0.0, 1.0))
        invprob = InverseProblem(experiment, [k1 => (0, 2.0)])
        @test_throws KeyError("test") cost_contribution(SingleShooting(maxiters = 1),
            experiment,
            invprob)

        experiment = Experiment(data,
            model,
            loss_func = (x, sol, data) -> [1, 2][3],
            tspan = (0.0, 1.0))
        invprob = InverseProblem(experiment, [k1 => (0, 2.0)])
        @test_throws BoundsError cost_contribution(SingleShooting(maxiters = 1),
            experiment,
            invprob)

        # user error / typo

        experiment = Experiment(data,
            model,
            loss_func = (x, sol, data) -> so - data,
            tspan = (0.0, 1.0))
        invprob = InverseProblem(experiment, [k1 => (0, 2.0)])
        @test_throws UndefVarError(:so, @__MODULE__) cost_contribution(
            SingleShooting(maxiters = 1),
            experiment,
            invprob)

        experiment = Experiment(data,
            model,
            loss_func = (x, sol, data) -> sum(1, 1, 1),
            tspan = (0.0, 1.0))
        invprob = InverseProblem(experiment, [k1 => (0, 2.0)])
        @test_throws MethodError cost_contribution(SingleShooting(maxiters = 1),
            experiment,
            invprob)

        struct MyType
            a::Vector{Int}
            MyType() = new()
        end
        A = MyType()

        experiment = Experiment(data,
            model,
            loss_func = (x, sol, data) -> A.a,
            tspan = (0.0, 1.0))
        invprob = InverseProblem(experiment, [k1 => (0, 2.0)])
        @test_throws UndefRefError cost_contribution(SingleShooting(maxiters = 1),
            experiment,
            invprob)

        experiment = Experiment(data,
            model,
            loss_func = (x, sol, data) -> dropdims([1]),
            tspan = (0.0, 1.0))
        invprob = InverseProblem(experiment, [k1 => (0, 2.0)])
        @test_throws UndefKeywordError cost_contribution(SingleShooting(maxiters = 1),
            experiment,
            invprob)
    end

    @testset "Unused columns" begin
        @testset "No unused columns" begin
            @test_nowarn Experiment(data, model)
        end
        @testset "Warning" begin
            rename!(data, ["s1(t)" => "p1(t)", "s2(t)" => "p2(t)"])
            @test_logs (:warn,
                "Columns [Symbol(\"p1(t)\"), Symbol(\"p2(t)\")] in the data cannot be used as they are not present in the model.") Experiment(
                data, model)
        end
    end
end

# TODO: Uncomment when fixed
# @testset "explicit initial guess for missing defaults" begin
#     @variables x(t)=1 [description = "bunny"] y(t)=1.5 [description = "wolf"]
#     @parameters α [description = "🐰 growth rate"] β=0.9 γ=0.8 δ=1.8
#     eqs = [
#         D(x) ~ α * x - β * x * y,
#         D(y) ~ -δ * y + γ * x * y
#     ]
#     model = complete(ODESystem(eqs, t; name = :lotka))

#     data = generate_data(model, u0 = [x => 1.2], params = [α => 1.25], alg = Tsit5())
#     experiment = Experiment(data, model, alg = Tsit5())

#     prob = InverseProblem(experiment,
#         [
#             x => (1.2, 2, 5),
#             α => (1.25, 1.2, 4.0)
#         ])
#     sol = simulate(experiment, prob)
#     @test successful_retcode(sol)

#     cost = objective(prob, SingleShooting(maxiters = 1))
#     # initial guess is different from the defaults
#     @test cost() == 0

#     @test initial_guess(Any, prob) ==
#           [Symbol(x) => 1.2, :α => 1.25]
# end

@testset "initial guess missing" begin
    @variables x(t)=1 [description = "bunny"] y(t)=1.5 [description = "wolf"]
    @parameters α [guess = 0] β [guess = 0] γ=0.8 δ=1.8
    eqs = [
        D(x) ~ α * x - β * x * y,
        D(y) ~ -δ * y + γ * x * y
    ]
    model = structural_simplify(ODESystem(eqs, t; name = :lotka))

    data = generate_data(
        model, u0 = [x => 1.2], params = [α => 1.25, β => 0.9], alg = Tsit5())
    # initialization is required with v9
    experiment = Experiment(
        data, model, overrides = [α => 1, β => 2], alg = Tsit5())

    @parameters α β=0.9 γ=0.8 δ=1.8
    eqs = [
        D(x) ~ α * x - β * x * y,
        D(y) ~ -δ * y + γ * x * y
    ]
    model = structural_simplify(ODESystem(eqs, t; name = :lotka))
    # the error is surfaced earlier in v9
    @test_throws ModelingToolkit.MissingParametersError experiment=Experiment(
        data, model, alg = Tsit5())
end

@testset "description based search space" begin
    @testset "not all parameters have descriptions" begin
        @variables x(t)=3.1 [description = "bunny"] y(t)=1.5 [description = "wolf"]
        @parameters α=1.3 [description = "🐰 growth rate"] β=0.9 γ=0.8 δ=1.8

        eqs = [
            D(x) ~ α * x - β * x * y,
            D(y) ~ -δ * y + γ * x * y
        ]
        model = complete(ODESystem(eqs, t, name = :lotka))

        data = generate_data(model)
        experiment = Experiment(data, model)

        prob = InverseProblem(experiment,
            [
                Initial(x) => (1.2, 2, 5),
                "🐰 growth rate" => (1, 2, :log10),
                "wolf" => (1.25, 1.2, 4.0, :log)
            ])
        sol = simulate(experiment, prob)
        @test successful_retcode(sol)

        cost = objective(prob, SingleShooting(maxiters = 1))
        # initial guess is different from the defaults
        @test cost() ≠ 0

        @test Set(initial_guess(Any, prob)) ==
              Set([
            Symbol(Initial(x)) => 1.2, Symbol(Initial(y)) => log(1.25), :α => log10(1.3)])
    end
    @testset "duplicate description one model" begin
        @variables x(t)=3.1 [description = "bunny"] y(t)=1.5 [description = "wolf"]
        @parameters α=1.3 [description = "🐰 growth rate"] β=0.9 [description = "a"] γ=0.8 [
            description = "a"
        ] δ=1.8
        eqs = [
            D(x) ~ α * x - β * x * y,
            D(y) ~ -δ * y + γ * x * y
        ]
        model = complete(ODESystem(eqs, t; name = :lotka))

        data = generate_data(model)
        experiment = Experiment(data, model)

        @test_throws ErrorException("Description a is not unique!") prob=InverseProblem(
            experiment,
            [
                Initial(x) => (1.2, 2, 5),
                "🐰 growth rate" => (1, 2, :log10),
                "a" => (1.25, 1.2, 4.0, :log)
            ])
    end

    @testset "duplicate description 2 models" begin
        @variables x(t)=3.1 [description = "bunny"] y(t)=1.5 [description = "wolf"]
        @parameters α=1.3 [description = "🐰 growth rate"] β=0.9 [description = "b"] γ=0.8 δ=1.8

        eqs = [
            D(x) ~ α * x - β * x * y,
            D(y) ~ -δ * y + γ * x * y
        ]
        model = complete(ODESystem(eqs, t; name = :lotka))

        @parameters α=1.4 [description = "🐰 growth rate"]

        eqs = [
            D(x) ~ α * x - β * x * y,
            D(y) ~ -δ * y + γ * x * y
        ]

        model2 = complete(ODESystem(eqs, t; name = :lotka2))

        data = generate_data(model)
        experiment1 = Experiment(data, model)
        experiment2 = Experiment(data, model2)

        # it's not considered an error to have duplicates across models
        # but the initial guess can be either options
        prob = InverseProblem([
                experiment1,
                experiment2
            ],
            [
                Initial(x) => (1.2, 2, 5),
                "b" => (1.25, 1.2, 4.0, :log)
            ])

        # even if we have the same description for β, the guess is unique
        @test Dict(initial_guess(Any, prob))[:β] == log(1.25)

        sol1 = simulate(experiment1, prob)
        @test sol1.ps[:α] == 1.3
        sol2 = simulate(experiment2, prob)
        @test sol2.ps[:α] == 1.4
    end

    @testset "same parameter in 2 models" begin
        @variables x(t)=3.1 [description = "bunny"] y(t)=1.5 [description = "wolf"]
        @parameters α=1.3 [description = "🐰 growth rate"] β=0.9 [description = "b"] γ=0.8 δ=1.8

        eqs = [
            D(x) ~ α * x - β * x * y,
            D(y) ~ -δ * y + γ * x * y
        ]
        model = complete(ODESystem(eqs, t; name = :lotka))

        @parameters α=1.4 [description = "🐰 growth rate"]

        eqs = [
            D(x) ~ α * x - β * x * y,
            D(y) ~ -δ * y + γ * x * y
        ]

        model2 = complete(ODESystem(eqs, t; name = :lotka2))

        data = generate_data(model)
        experiment1 = Experiment(data, model)
        experiment2 = Experiment(data, model2)

        # There is only 1 α in the entire inverse problem, even if used in 2 models.
        # Since the initial guess is for the parameter, it doesn't matter that we have multiple experiments,
        # all will start with the same initial guess.
        prob = InverseProblem([
                experiment1,
                experiment2
            ],
            [
                Initial(x) => (1.2, 2, 5),
                "b" => (1.25, 1.2, 4.0, :log),
                "🐰 growth rate" => (1.55, 1, 3)
            ])

        # even if we have the same description for β, the guess is unique
        @test Dict(initial_guess(Any, prob))[:β] == log(1.25)

        sol1 = simulate(experiment1, prob)
        @test sol1.ps[:α] == 1.55
        sol2 = simulate(experiment2, prob)
        @test sol2.ps[:α] == 1.55
    end

    @testset "same parameter in 2 models with different descriptions" begin
        @variables x(t)=3.1 [description = "bunny"] y(t)=1.5 [description = "wolf"]
        @parameters α=1.3 [description = "🐰 growth rate 1"] β=0.9 [description = "b"] γ=0.8 δ=1.8

        eqs = [
            D(x) ~ α * x - β * x * y,
            D(y) ~ -δ * y + γ * x * y
        ]
        model = complete(ODESystem(eqs, t; name = :lotka))

        @parameters α=1.4 [description = "🐰 growth rate 2"]

        eqs = [
            D(x) ~ α * x - β * x * y,
            D(y) ~ -δ * y + γ * x * y
        ]

        model2 = complete(ODESystem(eqs, t; name = :lotka2))

        data = generate_data(model)
        experiment1 = Experiment(data, model)
        experiment2 = Experiment(data, model2)

        msg = "Found 2 or more contradictory descriptions for α: " *
              "🐰 growth rate 1 and 🐰 growth rate 2.\n" *
              "Please use different symbolic variables or different descriptions."
        @test_throws ErrorException(msg) prob=InverseProblem([
                experiment1,
                experiment2
            ],
            [
                Initial(x) => (1.2, 2, 5),
                "b" => (1.25, 1.2, 4.0, :log),
                "🐰 growth rate 1" => (1.35, 1, 3),
                "🐰 growth rate 2" => (1.45, 1, 3)
            ])
    end

    @testset "duplicate search space" begin
        @variables x(t)=3.1 [description = "bunny"] y(t)=1.5 [description = "wolf"]
        @parameters α=1.3 [description = "🐰 growth rate"] β=0.9 γ=0.8 [description = "a"] δ=1.8

        eqs = [
            D(x) ~ α * x - β * x * y,
            D(y) ~ -δ * y + γ * x * y
        ]
        model = complete(ODESystem(eqs, t; name = :lotka))

        data = generate_data(model)
        experiment = Experiment(data, model)

        msg = "Repeated elements in the search space are not allowed. Found 2 entries for γ."
        @test_throws AssertionError(msg) prob=InverseProblem(experiment,
            [
                Initial(x) => (1.2, 2, 5),
                "🐰 growth rate" => (1, 2, :log10),
                "a" => (1.25, 1.2, 4.0, :log),
                γ => (1, 2)])

        msg = "Description \"γ\" not found in any of the models."
        @test_throws ErrorException(msg) prob=InverseProblem(experiment,
            [
                Initial(x) => (1.2, 2, 5),
                "🐰 growth rate" => (1, 2, :log10),
                "a" => (1.25, 1.2, 4.0, :log),
                "γ" => (1, 2)])
    end

    @testset "distributions in search space" begin
        @variables x(t)=3.1 [description = "bunny"] y(t)=1.5 [description = "wolf"]
        @parameters α=1.3 [description = "🐰 growth rate"] β=0.9 [description = "b"] γ=0.8 δ=1.8

        eqs = [
            D(x) ~ α * x - β * x * y,
            D(y) ~ -δ * y + γ * x * y
        ]
        model = complete(ODESystem(eqs, t; name = :lotka))

        @parameters α=1.4 [description = "🐰 growth rate"]

        eqs = [
            D(x) ~ α * x - β * x * y,
            D(y) ~ -δ * y + γ * x * y
        ]

        model2 = complete(ODESystem(eqs, t; name = :lotka2))

        data = generate_data(model)
        experiment1 = Experiment(data, model)
        experiment2 = Experiment(data, model2)

        prob = InverseProblem([
                experiment1,
                experiment2
            ],
            [
                Initial(x) => Normal(2, 1),
                "b" => (LogNormal(0, 0.25), :log),
                "🐰 growth rate" => (1.55, 1, 3)
            ])

        # test that sampling works as expected
        x0_samples = [Dict(initial_guess(Any, prob)) for _ in 1:1_000_000]

        empirical_dist_x = fit(Normal, getindex.(x0_samples, Symbol(Initial(x))))
        @test mean(empirical_dist_x)≈2 rtol=1e-2
        @test std(empirical_dist_x)≈1 rtol=1e-2

        empirical_dist_β = fit(LogNormal, getindex.(x0_samples, :β))
        @test meanlogx(empirical_dist_β)≈0 atol=1e-2
        @test stdlogx(empirical_dist_β)≈0.25 rtol=1e-2
    end
end

@testset "Two experiments" begin
    model = reactionsystem()
    @unpack k1, c1 = model

    @testset "Iteration & indexing" begin
        overrides = [c1 => 3.0]
        data1 = generate_data(model)
        ex1 = Experiment(data1, model)
        data2 = generate_data(model, (0.0, 2.0), 10; params = overrides)
        ex2 = Experiment(data2, model; overrides)

        prob = InverseProblem([ex1, ex2], [k1 => (0, 5), c1 => (0, 5)])
        # This should solve the exact same problem as the data generation
        @test get_experiments(prob)[1] === ex1
        @test get_experiments(prob)[2] === ex2
        @test first(get_experiments(prob)) === ex1
        @test last(get_experiments(prob)) === ex2

        alg = SingleShooting(maxiters = 1)
        x = initial_state(alg, prob)
        @test cost_contribution(alg, get_experiments(prob)[1], prob) == 0
        @test cost_contribution(alg, get_experiments(prob)[2], prob) == 0
    end

    @testset "ex1 nothing fixed, ex2 one parameter fixed" begin
        overrides = [c1 => 3.0]
        data1 = generate_data(model)
        ex1 = Experiment(data1, model)
        data2 = generate_data(model, (0.0, 2.0), 10; params = overrides)
        ex2 = Experiment(data2, model; overrides)

        prob = InverseProblem([ex1, ex2], [k1 => (0, 5), c1 => (0, 5)])
        # This should solve the exact same problem as the data generation
        cost = objective(prob, SingleShooting(maxiters = 1))
        @test cost() == 0
    end

    @testset "ex1 fix 1 parameter, ex2 fix one initial condition" begin
        @unpack s2 = model
        overrides = [c1 => 3.0]
        u0 = [s2 => 1.0]
        data1 = generate_data(model; params = overrides)
        ex1 = Experiment(data1, model; overrides)
        data2 = generate_data(model, (0.0, 2.0), 10; u0)
        ex2 = Experiment(data2, model; overrides = u0)

        prob = InverseProblem([ex1, ex2], [k1 => (0, 5), c1 => (0, 5)])
        # This should solve the exact same problem as the data generation
        cost = objective(prob, SingleShooting(maxiters = 1))
        @test cost() == 0
    end

    @testset "ex1, ex2 fix 1 parameter, optimize one initial condition" begin
        @unpack s2 = model
        overrides = [c1 => 3.0]
        data1 = generate_data(model; params = overrides)
        ex1 = Experiment(data1, model; overrides)
        data2 = generate_data(model, (0.0, 2.0), 10; params = overrides)
        ex2 = Experiment(data2, model; overrides)

        prob = InverseProblem([ex1, ex2], [s2 => (1.5, 3), k1 => (0, 5)])
        # This should solve the exact same problem as the data generation
        cost = objective(prob, SingleShooting(maxiters = 1))
        @test cost() == 0
    end

    @testset "ex1 fix 1 parameter, ex2 fix 1 (param + u0), optimize one initial condition" begin
        @unpack s2 = model
        params = [c1 => 3.0]
        u0 = [s2 => 1.0]
        data1 = generate_data(model; params)
        ex1 = Experiment(data1, model; overrides = params)
        data2 = generate_data(model, (0.0, 2.0), 10; u0, params)
        ex2 = Experiment(data2, model; overrides = [u0; params])

        prob = InverseProblem([ex1, ex2], [s2 => (1.5, 3), k1 => (0, 5)])
        # This should solve the exact same problem as the data generation
        cost = objective(prob, SingleShooting(maxiters = 1))
        @test cost() == 0
    end

    @testset "save_idxs" begin
        @unpack s1, s2, s1s2 = model
        params = [c1 => 3.0]
        data1 = generate_data(model; params)
        data1_mod = data1[!, ["timestamp", "s1s2(t)", "s1(t)"]]
        ex1 = Experiment(data1_mod, model; overrides = params)
        data2 = generate_data(model, (0.0, 2.0), 10; params)
        data2_mod = data2[!, ["timestamp", "s2(t)", "s1s2(t)"]]
        ex2 = Experiment(data2_mod, model; overrides = params)

        prob = InverseProblem([ex1, ex2], [s2 => (1.5, 3), k1 => (0, 5)])
        # This should solve the exact same problem as the data generation
        cost = objective(prob, SingleShooting(maxiters = 1))
        @test cost() == 0
    end

    @testset "Optimize one parameter, model observed in loss function" begin
        model = reactionsystem_obs()
        @unpack s1, s2, s1s2, s3, c1, k1 = model
        overrides = [c1 => 3.0]

        prob = ODEProblem(model, [], (0.0, 1.0), overrides)
        saveat = range(prob.tspan..., length = 5)
        sol = solve(prob; saveat, abstol = 1e-8, reltol = 1e-8)

        data1 = DataFrame("timestamp" => sol.t, "s3" => sol[s3])

        ex1 = Experiment(data1, model; overrides)
        data2 = generate_data(model, (0.0, 2.0), 10; params = overrides)
        data2_mod = data2[!, ["timestamp", "s2(t)", "s1s2(t)"]]
        ex2 = Experiment(data2_mod, model; overrides)

        prob = InverseProblem([ex1, ex2], [s2 => (1.5, 3), k1 => (0, 5)])

        cost = objective(prob, SingleShooting(maxiters = 1))
        @test cost() == 0
    end
end

@testset "Experiment with no data" begin
    model = reactionsystem()
    data = nothing
    experiment = Experiment(data, model, tspan = (0, 1))
    @unpack k1, c1 = model

    prob = InverseProblem(experiment, [k1 => (0, 1)])

    sol = @test_nowarn simulate(experiment, prob)
    @test successful_retcode(sol.retcode)
end

@testset "Experiment with replicate data" begin
    model = reactionsystem()
    data1 = generate_data(model)
    ex1 = Experiment([data1, data1, data1], model)
    @test DyadModelOptimizer.timespan(ex1) == (0.0, 1.0)
    @test get_saveat(ex1) == data1[:, 1]
    @parameters k1, c1

    @testset "Optimize identical-time experiments" begin
        prob = InverseProblem([ex1], [k1 => (0, 5), c1 => (0, 5)])

        cost = objective(prob, SingleShooting(maxiters = 1))
        @test cost() == 0
    end

    data2 = generate_data(model, (0.0, 1.2), 4)
    ex2 = Experiment([data1, data2], model)
    @test DyadModelOptimizer.timespan(ex2) == (0.0, 1.2)
    @test get_saveat(ex2) == sort!(unique!(vcat(data1[:, 1], data2[:, 1])))
    @testset "Optimize different-time experiments" begin
        prob = InverseProblem([ex2], [k1 => (0, 5), c1 => (0, 5)])

        cost = objective(prob, SingleShooting(maxiters = 1))
        @test isapprox(cost(), 0, atol = 1e-11)  # Got 4.092197067247435e-12
    end
end

# TODO enable after timespan optimization works again
# @testset "timespan optimization" begin
#     t = t_nounits
#     sts = @variables s1(t)=2.0 s1s2(t)=2.0 s2(t)=2.0
#     ps = @parameters k1=1.0 c1=2.0 [bounds = (0, 2), tunable = true] Δt=2.5
#     eqs = [D_nounits(s1) ~ -0.25 * c1 * k1 * s1 * s2
#            D_nounits(s1s2) ~ 0.25 * c1 * k1 * s1 * s2
#            D_nounits(s2) ~ -0.25 * c1 * k1 * s1 * s2]

#     model = complete(ODESystem(eqs, t_nounits, sts, ps; name = :reactionsystem))
#     data = generate_data(model)
#     @unpack Δt = model
#     ex1 = Experiment(data, model, tspan = (0.0, 1.0 + Δt))
#     ex2 = Experiment(data, model, tspan = (Δt, 5.0))
#     ex3 = Experiment(data, model, tspan = (0.0 + Δt, 1.0 + Δt))
#     @unpack k1, c1 = model

#     prob = InverseProblem([ex1, ex2, ex3], [k1 => (0, 1)])

#     sol1 = @test_nowarn simulate(ex1, prob)
#     @test sol1.prob.tspan == (0.0, 3.5)
#     @test successful_retcode(sol1.retcode)
#     sol2 = @test_nowarn simulate(ex2, prob)
#     @test sol2.prob.tspan == (2.5, 5.0)
#     @test successful_retcode(sol2.retcode)
#     sol3 = @test_nowarn simulate(ex3, prob)
#     @test sol3.prob.tspan == (2.5, 3.5)
#     @test successful_retcode(sol3.retcode)
# end
