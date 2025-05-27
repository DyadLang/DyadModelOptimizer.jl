using DyadModelOptimizer
using SciMLBase: successful_retcode
using Test

include("reactionsystem.jl")

@testset "IndependentExperiments" begin
    model = reactionsystem()
    data = generate_data(model)
    experiment = Experiment(data, model)

    prob = InverseProblem(experiment, [model.k1 => (0, 5), model.c1 => (0, 5)])
    cost = objective(prob, SingleShooting(maxiters = 1))

    @testset "$(typeof(x))" for x in [
        [1, 2],
        [model.k1 => 1, model.c1 => 2],
        (k1 = 1, c1 = 2),
        [model.k1 => 1.0, model.c1 => 2]
    ]
        @test iszero(cost(x))
    end
end
