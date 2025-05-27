using DyadModelOptimizer
using Test

include("reactionsystem.jl")

@testset "Penalty" begin
    model = reactionsystem()

    data = generate_data(model)
    experiment = Experiment(data, model, tspan = (0.0, 1.0))

    prob = InverseProblem([experiment], [model.k1 => (0, 5), model.c1 => (0, 5)])
    cost = objective(prob, SingleShooting(maxiters = 1))

    penalty = labelled_x -> 2 * labelled_x.k1 + 3 * labelled_x.c1
    probp = InverseProblem([experiment],
        [model.k1 => (0, 5), model.c1 => (0, 5)],
        penalty = penalty)
    costp = objective(probp, SingleShooting(maxiters = 1))

    vals = [2 * ModelingToolkit.defaults(model)[model.k1],
        3 * ModelingToolkit.defaults(model)[model.c1]]
    @test costp() - cost() == sum(vals)
    @test DyadModelOptimizer.get_penalty(probp) === penalty
end
