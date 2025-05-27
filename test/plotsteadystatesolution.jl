using DyadModelOptimizer
using ModelingToolkit
using OrdinaryDiffEqDefault
using DataFrames
using OptimizationOptimJL
using CSV
using Test # hide
using SteadyStateDiffEq
using Plots
using ModelingToolkit: t_nounits as t, D_nounits as D

@testset "Steady State Solution with more than 1 saved_states" begin
    function reactionsystem()
        sts = @variables s1(t)=2.0 s1s2(t)=2.0 s2(t)=2.0
        ps = @parameters k1=1.0 c1=2.0 [bounds = (0, 2), tunable = true] Δt=2.5
        eqs = [D(s1) ~ -0.25 * c1 * k1 * s1 * s2
               D(s1s2) ~ 0.25 * c1 * k1 * s1 * s2
               D(s2) ~ -0.25 * c1 * k1 * s1 * s2]

        return structural_simplify(ODESystem(eqs, t; name = :reactionsystem))
    end

    function generate_data(model, tspan = (0.0, 1.0), n = 5;
            params = [],
            u0 = [],
            kwargs...)
        prob = ODEProblem(model, u0, tspan, params)
        saveat = range(prob.tspan..., length = n)
        sol = solve(prob; saveat, kwargs...)

        return DataFrame(sol)
    end

    model = reactionsystem()
    @unpack k1, c1, s1, s2, s1s2 = model

    params = [c1 => 3.0]
    u0 = [s2 => 1.0, s1 => 2.0]
    data = generate_data(model; params)
    ss_data = data[end, 2:end]
    ex = SteadyStateExperiment(ss_data, model; overrides = params)

    prob = InverseProblem(ex, [Initial(s2) => (1.5, 3), k1 => (0, 5)])

    r = calibrate(prob, SingleShooting(maxiters = 10^4))

    # TODO: revive @test_nowarn for the three plots when fixed
    plot(ex, prob, r, legend = true, show_data = true)
    plot(ex, prob, r, legend = true, show_data = true, states = [s1, s2])
    plot(ex, prob, r, legend = true, show_data = true, states = [s1s2, s1])
end
