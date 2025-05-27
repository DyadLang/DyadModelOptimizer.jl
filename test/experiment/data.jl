using DyadModelOptimizer
using DyadModelOptimizer: get_data, TimeSeriesData
using SciMLBase: successful_retcode
using Test
using JET

include("../reactionsystem.jl")

@testset "TimeSeriesData" begin
    model = reactionsystem()
    data = generate_data(model)
    experiment = Experiment(data, model)

    d = get_data(experiment)

    @test_call d[1] == 2
    @test_call d[1, 2] == data[2, 2]
    @test_call d[:, 1] == [2, 2, 2]
    @test_call d[[2, 3], 2] == d[2:3, 2]
    @test_call d[2:3, 2] == collect(data[2, 3:4])
    @test_call d[2, [2, 3]] == d[2, 2:3]
    @test_call d[2, 2:3] == collect(data[2:3, 3])
    @test_call d[1, :] == data."s1(t)"
    @test_call d[2, :] == data[:, 3]
    @test_call d[2, 2:3].time == data[2:3, 1]

    @test size(d) == (3, 5)
    @test axes(d) == (Base.OneTo(3), Base.OneTo(5))

    @test_opt d[:, 1] == [2, 2, 2]
    @test_opt d[2, :] == data[:, 3]
end

@testset "ReplicateData" begin
    model = reactionsystem()
    data1 = generate_data(model)
    ex1 = Experiment([data1, data1, data1], model)

    d1 = get_data(ex1)

    @test size(d1) == (3,)
    @test d1[1] isa TimeSeriesData
    @test_call d1[1] == d1[2]
end

@testset "SteadyStateData" begin
    model = reactionsystem()
    data = generate_data(model)
    ss_data = data[end, 2:end]
    ex = SteadyStateExperiment(ss_data, model)

    d = get_data(ex)

    @test_call length(d) == 3
    @test_call all(d .== collect(ss_data))
    @test_call all(d[:] .== collect(ss_data))
    @test_call d[1:2] == collect(ss_data)[1:2]
end
