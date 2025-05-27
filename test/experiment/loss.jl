using DyadModelOptimizer
using DyadModelOptimizer: initial_state, MatrixLike, VectorLike, get_data
using SciMLBase: successful_retcode
using DataFrames
using Test
using JET
using Statistics

include("../reactionsystem.jl")

model = reactionsystem()
data = generate_data(model)

zscore_test(x, mean, std) = @. (x - mean) / std

experiment = Experiment(data, model, tspan = (0.0, 1.0))
invprob = InverseProblem(experiment, [model.c1 => (3.5, 0, 5)])
alg = SingleShooting(maxiters = 10)
init_x = initial_state(alg, invprob)
sol = simulate(experiment, invprob, init_x)

@testset "Loss Functions" begin
    @testset "MatrixLike" begin
        @testset "squaredl2loss" begin
            loss = squaredl2loss(MatrixLike{Float64}(), sol, get_data(experiment))
            error = (Array(sol)' .- Array(data)[:, 2:end]) .^ 2
            @test isapprox(sum(error), loss, rtol = 1e-10)
        end
        @testset "meansquaredl2loss" begin
            loss = meansquaredl2loss(MatrixLike{Float64}(), sol, get_data(experiment))
            error = (Array(sol)' .- Array(data)[:, 2:end]) .^ 2
            mean_error = mean(error, dims = 1) |> sum
            @test isapprox(mean_error, loss, rtol = 1e-10)
        end
        @testset "norm_meansquaredl2loss" begin
            loss = norm_meansquaredl2loss(MatrixLike{Float64}(), sol, get_data(experiment))
            sol_arr = Array(sol)'
            data_arr = Array(data)[:, 2:end]
            sol_mean = mean(sol_arr, dims = 1)
            error = (sol_arr .- data_arr) .^ 2 ./ (sol_arr .- sol_mean) .^ 2
            mean_error = mean(error, dims = 1) |> sum
            @test isapprox(mean_error, loss, rtol = 1e-10)
        end
        @testset "zscore_meansquaredl2loss" begin
            loss = zscore_meansquaredl2loss(
                MatrixLike{Float64}(), sol, get_data(experiment))
            sol_arr = Array(sol)'
            data_arr = Array(data)[:, 2:end]
            data_mean = mean(data_arr, dims = 1)
            data_std = std(data_arr, dims = 1)
            sol_norm = zscore_test(sol_arr, data_mean, data_std)
            data_norm = zscore_test(data_arr, data_mean, data_std)
            error = (sol_norm .- data_norm) .^ 2
            mean_error = mean(error, dims = 1) |> sum
            @test isapprox(mean_error, loss, rtol = 1e-10)
        end
    end

    @testset "VectorLike" begin
        @testset "squaredl2loss" begin
            loss = squaredl2loss.(Ref(VectorLike{Float64}()), eachcol(Array(sol)),
                eachcol(get_data(experiment)))
            error = (Array(sol)' .- Array(data)[:, 2:end]) .^ 2
            @test all(isapprox.(sum(error, dims = 2), loss, rtol = 1e-10))
        end
        @testset "meansquaredl2loss" begin
            loss = meansquaredl2loss.(Ref(VectorLike{Float64}()), eachcol(Array(sol)),
                eachcol(get_data(experiment)))
            error = (Array(sol)' .- Array(data)[:, 2:end]) .^ 2
            mean_error = mean(error, dims = 2) |> vec
            @test all(isapprox.(mean_error, loss, rtol = 1e-10))
        end
        @testset "norm_meansquaredl2loss" begin
            loss = norm_meansquaredl2loss.(Ref(VectorLike{Float64}()), eachcol(Array(sol)),
                eachcol(get_data(experiment)))
            data_arr = Array(data)[:, 2:end]
            sol_arr = Array(sol)'
            sol_mean = mean(sol_arr, dims = 2)
            error = (sol_arr .- data_arr) .^ 2 ./ (sol_arr .- sol_mean) .^ 2
            mean_error = mean(error, dims = 2) |> vec
            # Since for the initial condition, sol == data == mean, we get `NaN`
            filtered_loss = filter(!isnan, loss)
            filtered_mean_error = filter(!isnan, mean_error)
            @test all(isapprox.(filtered_mean_error, filtered_loss, rtol = 1e-10))
        end
        @testset "zscore_meansquaredl2loss" begin
            @testset "with defaults" begin
                # Skipping the `first index` because std dev is 0.0
                loss = zscore_meansquaredl2loss.(Ref(VectorLike{Float64}()),
                    eachcol(Array(sol))[2:end],
                    eachcol(get_data(experiment))[2:end])
                sol_arr = Array(sol)'[2:end, :]
                data_arr = Array(data)[2:end, 2:end]
                data_mean = mean(data_arr, dims = 2)
                data_std = std(data_arr, dims = 2)
                sol_norm = zscore_test(sol_arr, data_mean, data_std)
                data_norm = zscore_test(data_arr, data_mean, data_std)
                error = (sol_norm .- data_norm) .^ 2
                mean_error = mean(error, dims = 2) |> vec
                @test isapprox(mean_error, loss, rtol = 1e-10)
            end
            @testset "With mean and std as kwarg" begin
                data_arr = Array(data)[:, 2:end]
                data_mean = mean(data_arr, dims = 1)
                data_std = std(data_arr, dims = 1)

                loss = zscore_meansquaredl2loss.(
                    Ref(VectorLike{Float64}()), eachcol(Array(sol)),
                    eachcol(get_data(experiment)); data_mean, data_std)
                sol_arr = Array(sol)'
                data_arr = Array(data)[:, 2:end]
                sol_norm = zscore_test(sol_arr, data_mean, data_std)
                data_norm = zscore_test(data_arr, data_mean, data_std)
                error = (sol_norm .- data_norm) .^ 2
                mean_error = mean(error, dims = 2) |> vec
                @test isapprox(mean_error, loss, rtol = 1e-10)
            end
        end
    end
end

@testset "cost_contribution" begin
    @testset "squaredl2loss" begin
        experiment = Experiment(data, model, tspan = (0.0, 1.0), loss_func = squaredl2loss)
        invprob = InverseProblem(experiment, [model.c1 => (3.5, 0, 5)])
        cost = cost_contribution(alg, experiment, invprob)
        squaredl2error = (Array(sol)' .- Array(data)[:, 2:end]) .^ 2
        @test isapprox(cost, sum(squaredl2error), rtol = 1e-10)
    end
    @testset "meansquaredl2loss" begin
        experiment = Experiment(
            data, model, tspan = (0.0, 1.0), loss_func = meansquaredl2loss)
        invprob = InverseProblem(experiment, [model.c1 => (3.5, 0, 5)])
        cost = cost_contribution(alg, experiment, invprob)
        squaredl2error = (Array(sol)' .- Array(data)[:, 2:end]) .^ 2
        @test isapprox(cost, sum(mean(squaredl2error, dims = 1)), rtol = 1e-10)
    end
    @testset "zscore_meansquaredl2loss" begin
        experiment = Experiment(
            data, model, tspan = (0.0, 1.0), loss_func = zscore_meansquaredl2loss)
        invprob = InverseProblem(experiment, [model.c1 => (3.5, 0, 5)])
        cost = cost_contribution(alg, experiment, invprob)

        data_arr = Array(data)[:, 2:end]
        data_mean = mean(data_arr, dims = 1)
        data_std = std(data_arr, dims = 1)

        zscore_squaredl2error = (zscore_test(Array(sol)', data_mean, data_std) .-
                                 zscore_test(Array(data)[:, 2:end], data_mean, data_std)) .^
                                2

        @test isapprox(cost, sum(mean(zscore_squaredl2error, dims = 1)), rtol = 1e-10)
    end
    @testset "norm_meansquaredl2loss" begin
        experiment = Experiment(
            data, model, tspan = (0.0, 1.0), loss_func = norm_meansquaredl2loss)
        invprob = InverseProblem(experiment, [model.c1 => (3.5, 0, 5)])
        cost = cost_contribution(alg, experiment, invprob)

        sol_mean = mean(Array(sol)', dims = 1)
        squaredl2error = (Array(sol)' .- Array(data)[:, 2:end]) .^ 2
        norm_squaredl2error = squaredl2error ./ (Array(sol)' .- sol_mean) .^ 2

        @test isapprox(cost, sum(mean(norm_squaredl2error, dims = 1)), rtol = 1e-10)
    end
end
