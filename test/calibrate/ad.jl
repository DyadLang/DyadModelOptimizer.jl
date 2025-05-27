using Test
using DyadModelOptimizer
using DataFrames
using ModelingToolkit
using Optimization
using Zygote
using SciMLSensitivity

include("../reactionsystem.jl")

@testset "SingleShooting" begin
    model = reactionsystem()
    params = [model.c1 => 3.5]
    data = generate_data(model; params)

    experiment = Experiment(data, model, tspan = (0.0, 1.0))
    invprob = InverseProblem(experiment, [model.c1, model.k1] .=> ((0, 5),))

    @testset for adtype in [AutoFiniteDiff(), AutoForwardDiff(), AutoZygote()]
        r = calibrate(invprob, SingleShooting(maxiters = 100); adtype)
        @test r[1] * r[2]≈3.5 rtol=1e-4
    end
end
