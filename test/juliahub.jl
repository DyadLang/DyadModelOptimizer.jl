using JuliaHub, DyadModelOptimizer
using OptimizationBBO
using DataFrames, ModelingToolkit
using Serialization

include("reactionsystem.jl")
model = reactionsystem()

@testset "Parameteric UQ" begin
    @unpack k1, c1 = model
    data = generate_data(model, (0.0, 1.0), 10, params = [k1 => 2, c1 => 3])
    experiment = Experiment(data, model)
    prob = InverseProblem(experiment, [k1 => (0, 5), c1 => (0, 5)])

    auth = JuliaHub.authenticate()
    specs = (ncpu = 8,
        memory = 64,
        nnodes = 1,
        process_per_node = false,
        _image_sha256 = "sha256:88a868009d6b0733bd7821988dfae55f1d9d04ed22d99a5fa32f0fa221aef27a")

    alg = StochGlobalOpt(
        method = SingleShooting(maxiters = 10^4,
            optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited()),
        parallel_type = EnsembleDistributed())
    alg_juliahub = JuliaHubJob(; auth,
        batch_image = JuliaHub.batchimage("juliasim-batch", "JuliaSim - Next"),
        node_specs = specs, dataset_name = "dummy",
        alg = alg)

    sample_size = 100
    job = parametric_uq(prob, alg_juliahub; sample_size = sample_size)
    @test job isa JuliaHub.Job
    @show job.id
    job = JuliaHub.wait_job(job.id; auth)
    JuliaHub.download_dataset("dummy", "./ps"; auth)
    ps_juliahub = deserialize("ps")
    @test ps_juliahub isa ParameterEnsemble
end

@testset "Calibrate" begin
    @unpack k1 = model
    data = generate_data(model, (0.0, 1.0), 10, params = [k1 => 4.0])
    experiment = Experiment(data, model, abstol = 1e-8, reltol = 1e-6)
    prob = InverseProblem(experiment, [k1 => (0, 5)])
    alg_ss = SingleShooting(maxiters = 1000)
    alg_ms = MultipleShooting(maxiters = 1000, trajectories = 10)

    auth = JuliaHub.authenticate()
    specs = (ncpu = 8,
        memory = 64,
        nnodes = 1,
        process_per_node = true,
        _image_sha256 = "sha256:88a868009d6b0733bd7821988dfae55f1d9d04ed22d99a5fa32f0fa221aef27a")

    alg_juliahub_ss = JuliaHubJob(; auth,
        batch_image = JuliaHub.batchimage("juliasim-batch", "JuliaSim - Next"),
        node_specs = specs, dataset_name = "dummy2",
        alg = alg_ss)

    alg_juliahub_ms = JuliaHubJob(; auth,
        batch_image = JuliaHub.batchimage("juliasim-batch", "JuliaSim - Next"),
        node_specs = specs, dataset_name = "dummy2",
        alg = alg_ms)

    r_ss = calibrate(prob, alg_ss)
    r_ms = calibrate(prob, alg_ms)

    @testset "First Calibration" begin
        @testset "$type" for (type, alg_juliahub, r) in [
            ("SS", alg_juliahub_ss, r_ss), ("MS", alg_juliahub_ms, r_ms)]
            job = calibrate(prob, alg_juliahub)
            @test job isa JuliaHub.Job
            @show job.id
            job = JuliaHub.wait_job(job.id; auth)
            JuliaHub.download_dataset("dummy2", "./r"; auth)
            r_juliahub = deserialize("r")
            @test r_juliahub isa CalibrationResult
            @test only(r_juliahub)≈only(r) rtol=1e-5
            run(`rm ./r`)
        end
    end

    @testset "Continue Calibration" begin
        @testset "$type" for (type, alg_juliahub, alg) in [
            ("SS", alg_juliahub_ss, alg_ss), ("MS", alg_juliahub_ms, alg_ms)]
            @testset "first calibrated using $type2" for (type2, r) in [
                ("SS", r_ss), ("MS", r_ms)]
                job2 = calibrate(r, alg_juliahub)
                @test job2 isa JuliaHub.Job
                @show job2.id
                job2 = JuliaHub.wait_job(job2.id; auth)
                JuliaHub.download_dataset("dummy2", "./r2"; auth)
                r_juliahub = deserialize("r2")
                @test r_juliahub isa CalibrationResult
                r2 = calibrate(r, alg)
                @test only(r_juliahub)≈only(r2) rtol=1e-5
                run(`rm ./r2`)
            end
        end
    end
end
