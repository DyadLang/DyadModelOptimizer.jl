using Test
using SciMLBase: EnsembleThreads
using DyadModelOptimizer
using ModelingToolkit
using DataFrames

include("reactionsystem.jl")

@testset "One experiment (StochGlobalOpt)" begin
    model = reactionsystem()
    @unpack k1, c1 = model
    data = generate_data(model, (0.0, 1.0), 10, params = [k1 => 2, c1 => 3])
    experiment = Experiment(data, model)
    prob = InverseProblem(experiment, [k1 => (0, 5), c1 => (0, 5)])
    sample_size = 200
    alg = StochGlobalOpt(parallel_type = EnsembleThreads())

    ps = parametric_uq(prob, alg; sample_size)
    # k1 and c1 are not identifiable, but k1 * c1 ≈ 6
    @test all(isapprox.(Tables.columns(ps)[:k1] .* Tables.columns(ps)[:c1], 6, rtol = 1e-4))
end

@testset "save_names with observed states" begin
    model = reactionsystem_obs()
    @unpack k1, c1, s1, s3 = model
    tspan = (0.0, 1.0)
    prob = ODEProblem(model, [k1 => 2, c1 => 3], tspan, [])
    saveat = range(prob.tspan..., length = 10)
    sol = solve(prob, Tsit5(); saveat)

    data = DataFrame(:timestamp => sol.t, :s1 => sol[s1], :s3 => sol[s3])
    experiment = Experiment(data, model, tspan = (0.0, 1.0))

    invprob = InverseProblem(experiment, [k1 => (0, 5), c1 => (0, 5)])
    alg = StochGlobalOpt(parallel_type = EnsembleThreads())

    sample_size = 20
    ps = parametric_uq(invprob, alg; sample_size)

    # k1 and c1 are not identifiable, but k1 * c1 ≈ 6
    @test all(isapprox.(Tables.columns(ps)[:k1] .* Tables.columns(ps)[:c1], 6, rtol = 1e-4))
end
