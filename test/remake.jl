using Test
using DyadModelOptimizer
using DyadModelOptimizer: get_saveat, get_save_idxs, get_overrides
using OrdinaryDiffEqTsit5, OrdinaryDiffEqSDIRK
using SciMLBase: successful_retcode
using SteadyStateDiffEq
using ModelingToolkit
using ModelingToolkit: D_nounits as D, t_nounits as t
using DataFrames
using SymbolicIndexingInterface

include("reactionsystem.jl")

@testset "Experiment" begin
    model = reactionsystem()
    data = generate_data(model)
    experiment = Experiment(data, model, tspan = (0.0, 1.0))

    overrides = [model.c1 => 3.0]

    t = remake(experiment; model, tspan = (0, 2))
    @test t.tspan == (0, 2)

    t = remake(experiment; model, overrides)
    @test get_overrides(t) ≢ get_overrides(experiment)
    @test get_saveat(experiment) == get_saveat(t)
    @test get_save_idxs(experiment) == get_save_idxs(t)

    prob = InverseProblem([t], [model.k1 => (0, 1)])
    sol = @test_nowarn simulate(t, prob)
    @test successful_retcode(sol.retcode)
end

@testset "remake Experiment with aliased parameter" begin
    ps = @parameters c1=2.0 k1=2.0 k2=2k1
    sts = @variables s1(t)=2.0 s1s2(t)=2.0 s2(t)=2 * k1
    eqs = [D_nounits(s1) ~ -0.25 * c1 * k1 * s1 * s2
           D_nounits(s1s2) ~ 0.25 * c1 * k1 * s1 * s2
           D_nounits(s2) ~ -0.25 * c1 * k2 * s1 * s2]

    model = structural_simplify(ODESystem(eqs, t; parameter_dependencies = [k2 => 2k1],
        name = :reactionsystem))

    data = generate_data(model, params = [c1 => k1])
    experiment = Experiment(data, model, tspan = (0.0, 1.0), overrides = [c1 => k1])

    alg = SingleShooting(maxiters = 1)
    prob = InverseProblem(experiment, [k1 => (0, 4)])

    ex = remake(experiment; model, tspan = (0, 2))
    @test ex.tspan == (0, 2)

    x = [1]
    new_prob = InverseProblem(ex, [k1 => (0, 4)])
    sol = simulate(ex, new_prob, x)
    @test sol[s2, 1] == 2 # 2*k1
    @test sol.prob.ps[[k1, c1, k2]] == [1, 1, 2]
end

@testset "SteadyStateExperiment" begin
    model = reactionsystem()
    @unpack k1, c1, s1, s2, s1s2 = model

    overrides = [c1 => 3.0]
    u0 = [s2 => 1.0, s1 => 2.0]
    data = generate_data(model; params = overrides)
    experiment = SteadyStateExperiment(data[end, :], model;
        alg = DynamicSS(Tsit5()),
        overrides,
        depvars = [s1s2, s1])

    ex = remake(experiment; model, overrides = [c1 => 2.0])
    @test get_overrides(ex) ≢ get_overrides(experiment)
    @test get_saveat(experiment) == get_saveat(ex)
    @test get_save_idxs(experiment) == get_save_idxs(ex)

    prob = InverseProblem(ex, [k1 => (0, 1)])
    # TODO: revive @test_nowarn
    sol = simulate(ex, prob)
    @test successful_retcode(sol.retcode)
end

@testset "remake SteadyStateExperiment with aliased parameter" begin
    ps = @parameters c1=2.0 k1=2.0 k2=2k1
    sts = @variables s1(t)=2.0 s1s2(t)=2.0 s2(t)=2 * k1
    eqs = [D_nounits(s1) ~ -0.25 * c1 * k1 * s1 * s2
           D_nounits(s1s2) ~ 0.25 * c1 * k1 * s1 * s2
           D_nounits(s2) ~ -0.25 * c1 * k2 * s1 * s2]

    model = structural_simplify(ODESystem(
        eqs, t; name = :reactionsystem))

    data = generate_data(model, params = [c1 => k1])
    experiment = SteadyStateExperiment(data[end, :], model;
        alg = DynamicSS(Tsit5()),
        overrides = [c1 => k1]
    )

    alg = SingleShooting(maxiters = 1)
    prob = InverseProblem(experiment, [k1 => (0, 4)])

    ex = remake(experiment; overrides = [k1 => 1.5])

    new_invprob = InverseProblem(ex, [])
    sol = simulate(ex, new_invprob)
    @test sol[s2]==2 broken=true # 2*k1
    # c1 is now 2, as the overrides are completely replaced, so c1 is no longer aliased
    @test sol.prob.ps[[k1, c1, k2]] == [1.5, 2, 3]
end

@testset "remake with special depvars" begin
    ps = @parameters k1 = 1
    sts = @variables x(t)=1 obs_x(t) __sigma_x(t)

    eqs = [D(x) ~ -k1 * x,
        obs_x ~ 2 * x,
        __sigma_x ~ 0.1 * obs_x]

    model = ODESystem(eqs, t; name = :reactionsystem)
    model = structural_simplify(model)

    data = DataFrame("timestamp" => [0, 1], "obs_x" => [1, 0.1])
    experiment = Experiment(data, model,
        loss_func = (x, sol, data) -> sum((sol[1, :] - data[1, :] ./ sol[2, :]) .^ 2),
        depvars = [:obs_x, :__sigma_x])
    invprob = InverseProblem(experiment, [k1 => (0, 5)])

    cost_true = cost_contribution(SingleShooting(maxiters = 1), experiment, invprob)

    ex = remake(experiment; model)
    new_invprob = InverseProblem(ex, [k1 => (0, 5)])
    cost_test = cost_contribution(SingleShooting(maxiters = 1), ex, new_invprob)
    @test isapprox(cost_test, cost_true)
end

@testset "ChainedExperiments" begin
    model = reactionsystem()
    @unpack k1, c1, s1, s2, s1s2 = model

    overrides = [c1 => 3.0]
    u0 = [s2 => 1.0, s1 => 2.0]
    data = generate_data(model; params = overrides)
    ss_data = data[end, 2:end]
    e1 = SteadyStateExperiment(ss_data, model;
        alg = DynamicSS(Tsit5()),
        overrides)
    @test nameof(e1) == "SteadyStateExperiment"

    e2 = Experiment(data, model; overrides = u0, dependency = e1)
    e3 = Experiment(data, model; overrides = u0, dependency = e2)

    prob = InverseProblem([e1, e2, e3],
        [s2 => (1.5, 3), k1 => (0, 5)])
    cost = objective(prob, SingleShooting(maxiters = 1))
    @test !iszero(cost())

    ex1 = remake(e1; model, overrides = [c1 => 2.0])
    @test get_overrides(ex1) ≢ get_overrides(e1)
    @test get_saveat(e1) == get_saveat(ex1)
    @test get_save_idxs(e1) == get_save_idxs(ex1)

    prob = InverseProblem(ex1, [k1 => (0, 1)])
    # TODO: revive @test_nowarn
    sol = simulate(ex1, prob)
    @test successful_retcode(sol.retcode)

    ex2 = remake(e2; model, overrides = [c1 => 2.0], dependency = ex1)
    @test get_overrides(ex2) ≢ get_overrides(e2)
    @test get_saveat(e2) == get_saveat(ex2)
    @test get_save_idxs(e2) == get_save_idxs(ex2)

    prob = InverseProblem([ex1, ex2], [k1 => (0, 1)])
    # TODO: revive @test_nowarn
    sol = simulate(ex2, prob)
    @test successful_retcode(sol.retcode)
end

@testset "Remake experiments" begin
    model = reactionsystem()
    data = generate_data(model)
    experiment = Experiment(data, model, tspan = (0.0, 1.0))
    @unpack k1, c1 = model

    invprob = InverseProblem(experiment, [k1 => (0, 1)])
    costfun = objective(invprob, SingleShooting(maxiters = 1))
    costval = costfun()

    invprob_trbdf2 = remake_experiments(invprob, alg = TRBDF2())
    costfun_trbdf2 = objective(invprob_trbdf2, SingleShooting(maxiters = 1))
    costval_trbdf2 = costfun_trbdf2()

    @test isapprox(costval, costval_trbdf2, atol = 2e-4)
end
