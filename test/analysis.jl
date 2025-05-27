using Test
using DyadModelOptimizer
using DataFrames

include("reactionsystem.jl")

@testset "One experiment" begin
    model = reactionsystem()
    data = generate_data(model)
    lc = DyadModelOptimizer.LossContribution(data)
    experiment = Experiment(data, model, tspan = (0.0, 1.0), loss_func = lc)

    prob = InverseProblem([experiment], [model.k1 => (0, 5), model.c1 => (0, 5)])

    @test iszero(lc)
end
