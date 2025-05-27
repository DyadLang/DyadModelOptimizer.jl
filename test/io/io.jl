using Test
using DyadModelOptimizer
using ModelingToolkit
using ModelingToolkit: t_nounits as t
using DataFrames
using CSV
using DyadData
using Tables
using ProgressLogging
using OptimizationBBO

include("../reactionsystem.jl")

@testset "Import/Export - calibrate" begin
    model = reactionsystem()
    @parameters k1, c1
    data = DyadDataset("reaction_system_data.csv", independent_var = "t",
        dependent_vars = ["u1", "u2", "u3"])
    experiment = Experiment(data, model, tspan = (0.0, 1.0),
        depvars = [model.s1 => "u1", model.s1s2 => "u3", model.s2 => "u2"])

    prob = InverseProblem(experiment, [k1 => (0, 5), c1 => (0, 5)])

    res = calibrate(prob, SingleShooting(maxiters = 100))

    @test res[1]≈1 rtol=1e-4
    @test res[2]≈2 rtol=1e-4

    @test res[:k1]≈1 rtol=1e-4
    @test res[:c1]≈2 rtol=1e-4

    @test res[begin] == res[1] == first(res)
    @test res[end] == res[2] == last(res)

    @test size(res) == (2,)

    @test Tables.istable(typeof(res))
    row = first(Tables.rows(res))
    @test row.k1 == res[1]
    @test Tables.getcolumn(res, 1) == Tables.getcolumn(res, :k1) == res[1]

    ivec = import_res([res[1], res[2]], prob)
    @test all([all(res[i] == ivec[i]) for i in eachindex(res)])
    isol = simulate(experiment, prob, ivec)
    @test SciMLBase.successful_retcode(isol.retcode)
    ic = calibrate(ivec, SingleShooting(maxiters = 1))
    @test ic.retcode == ReturnCode.MaxIters

    df = DataFrame(res)
    @test names(df) == ["k1", "c1"]
    @test size(df) == (1, 2)
    @test df[1, :k1] == res[1]
    @test df[1, :c1] == res[2]

    idf = import_res(df, prob)
    @test all([all(res[i] == idf[i]) for i in eachindex(res)])
    isol = simulate(experiment, prob, idf)
    @test SciMLBase.successful_retcode(isol.retcode)
    ic = calibrate(idf, SingleShooting(maxiters = 1))
    @test ic.retcode == ReturnCode.MaxIters

    fn, io = mktemp()
    CSV.write(io, res)
    close(io)

    csv = CSV.File(fn)
    @test only(csv.k1) == res[1]
    @test only(csv.c1) == res[2]

    csv = CSV.File(fn)
    icsv = import_res(csv, prob)
    @test all([all(res[i] == icsv[i]) for i in eachindex(res)])
    isol = simulate(experiment, prob, icsv)
    @test SciMLBase.successful_retcode(isol.retcode)
    ic = calibrate(icsv, SingleShooting(maxiters = 1))
    @test ic.retcode == ReturnCode.MaxIters
end

@testset "Import/Export - parametric_uq" begin
    model = reactionsystem()
    @parameters k1, c1
    data = generate_data(model)
    experiment = Experiment(data, model, tspan = (0.0, 1.0))

    prob = InverseProblem(experiment, [k1 => (0, 5), c1 => (0, 5)])

    ps = parametric_uq(prob,
        StochGlobalOpt(method = SingleShooting(maxiters = 10,
            optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited())),
        sample_size = 50)
    cost = objective(prob, SingleShooting(maxiters = 1))
    c1 = cost.(ps)
    @test length(ps) == 50

    @test Tables.istable(typeof(ps))
    row = first(Tables.rows(ps))
    @test row[:k1] == row[1]
    @test Tables.getcolumn(row, :k1) == Tables.getcolumn(row, 1) == row[1]

    m = Tables.matrix(ps)
    @test all([m[i, :] == ps[i] for i in eachindex(ps)])
    @test all([m[:, i] == [ps[j][i] for j in eachindex(ps)] for i in 1:2])
    ivp = import_ps(m, prob)
    c2 = cost.(ivp)
    @test all([all(ps[i] .== values(ivp[i])) for i in eachindex(ps)])
    @test c1 == c2

    df = DataFrame(ps)
    idf = import_ps(df, prob)
    c3 = cost.(idf)
    @test all([all(ps[i] .== values(idf[i])) for i in eachindex(ps)])
    @test c1 == c3

    fn, io = mktemp()
    CSV.write(io, df)
    close(io)

    csv = CSV.File(fn)
    icsv = import_ps(csv, prob)
    c4 = cost.(icsv)
    @test all([all(ps[i] .== values(icsv[i])) for i in eachindex(ps)])
    @test c1 == c4
end

@testset "DataFrame export" begin
    @testset "One experiment, save all" begin
        model = reactionsystem()
        @parameters k1, c1
        data = generate_data(model)
        experiment = Experiment(data, model, tspan = (0.0, 1.0))

        prob = InverseProblem([experiment], [k1 => (0, 5), c1 => (0, 5)])

        ps = parametric_uq(prob,
            StochGlobalOpt(method = SingleShooting(maxiters = 10,
                optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited())),
            sample_size = 50)
        cost = objective(prob, SingleShooting(maxiters = 1))
        c1 = cost.(ps)
        @test length(ps) == 50

        df = DataFrame(ps)
        c2 = cost.(eachrow(df))
        @test df isa DataFrame
        @test all([all(collect(df[i, :]) .== ps[i]) for i in 1:50])
        @test c1 == c2
    end

    @testset "One experiment, not all params" begin
        model = reactionsystem()
        @parameters k1, c1
        @variables s1(t) s2(t)
        data = generate_data(model)
        experiment = Experiment(data, model, tspan = (0.0, 1.0))

        prob = InverseProblem([experiment], [c1 => (0, 5)])

        ps = parametric_uq(prob,
            StochGlobalOpt(method = SingleShooting(maxiters = 10,
                optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited())),
            sample_size = 50)
        @test length(ps) == 50

        df = DataFrame(ps)
        @test df isa DataFrame
        @test all([all(collect(df[i, :]) .== ps[i]) for i in 1:50])
    end
end

@testset "show" begin
    model = reactionsystem()
    @parameters k1, c1
    data = generate_data(model)
    experiment = Experiment(data, model, tspan = (0.0, 1.0))

    prob = InverseProblem(experiment, [k1 => (0, 5), c1 => (0, 5)])

    prob_ref = "InverseProblem with one experiment " *
               "with 2 elements in the search space.\n"
    ec_ref = "IndependentExperiments collection with one experiment."
    @test sprint(io -> show(io, MIME"text/plain"(), prob)) == prob_ref
    @test sprint(io -> show(io, MIME"text/plain"(), get_experiments(prob))) == ec_ref

    experiment_ref = "Experiment for reactionsystem with no overrides.\n" *
                     "The simulation of this experiment is given by:\n" *
                     "ODEProblem with uType Vector{Float64} and tType Float64. In-place: true\n" *
                     "Initialization status: FULLY_DETERMINED\nNon-trivial mass matrix: false\n" *
                     "timespan: (0.0, 1.0)"
    @test sprint(io -> show(io, MIME"text/plain"(), experiment)) == experiment_ref

    experiment2 = Experiment(data, model, overrides = [k1 => 1], tspan = (0.0, 1.0))

    prob = InverseProblem([experiment, experiment2], [k1 => (0, 5), c1 => (0, 5)])

    prob_ref = "InverseProblem with 2 experiments " *
               "with 2 elements in the search space.\n"
    ec_ref = "IndependentExperiments collection with 2 experiments."

    @test sprint(io -> show(io, MIME"text/plain"(), prob)) == prob_ref
    @test sprint(io -> show(io, MIME"text/plain"(), get_experiments(prob))) == ec_ref

    alg = SingleShooting(maxiters = 1, maxtime = 100)
    alg_ref = "SingleShooting method, optimizing with OptimizerWithAttributes.\nmaxiters = 1 and maxtime = 100.0\n"
    @test sprint(io -> show(io, MIME"text/plain"(), alg)) == alg_ref

    ref_c = "Calibration result computed in"
    r = calibrate(prob, alg)
    @test startswith(sprint(io -> show(io, MIME"text/plain"(), r)), ref_c)

    r = parametric_uq(prob,
        StochGlobalOpt(method = SingleShooting(maxiters = 1,
            optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited())),
        sample_size = 10)
    ref_p = "Parametric uncertainty ensemble of length 10 computed in"
    @test startswith(sprint(io -> show(io, MIME"text/plain"(), r)), ref_p)

    nodata = DyadModelOptimizer.NoData()
    @test sprint(io -> show(io, MIME"text/plain"(), nodata)) == "NoData()"
end

@testset "Progress logging" begin
    model = reactionsystem()
    @parameters k1, c1
    data = generate_data(model)
    experiment = Experiment(data, model, tspan = (0.0, 1.0))

    prob = InverseProblem(experiment, [k1 => (0, 5), c1 => (0, 5)])

    @testset "default" begin
        logs, r = Test.collect_test_logs(min_level = Base.LogLevel(-1)) do
            calibrate(prob, SingleShooting(maxiters = 10))
        end
        # both logs and loss_history
        m = first(logs)
        @test m.level == Base.LogLevel(-1)
        @test m.message isa ProgressLogging.Progress
        @test m.message.name == "calibrating"
        @test !m.message.done

        m_end = last(logs)
        @test m_end.level == Base.LogLevel(-1)
        @test m_end.message isa ProgressLogging.Progress
        @test m_end.message.done

        @test !isnothing(r.loss_history)
    end

    @testset "no progress" begin
        logs, r = Test.collect_test_logs(min_level = Base.LogLevel(-1)) do
            calibrate(prob, SingleShooting(maxiters = 10), progress = false)
        end
        # nol logs but we have loss_history
        @test isempty(logs)
        @test !isnothing(r.loss_history)
    end

    @testset "no tracking" begin
        logs, r = Test.collect_test_logs(min_level = Base.LogLevel(-1)) do
            calibrate(prob, SingleShooting(maxiters = 10), progress = nothing)
        end
        # nol logs and no loss_history
        @test isempty(logs)
        @test isnothing(r.loss_history)
    end
end
