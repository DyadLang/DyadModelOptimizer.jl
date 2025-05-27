using SafeTestsets
using Test
const GROUP = get(ENV, "GROUP", "All")
run_juliahub_tests = false

@testset verbose=true "DyadModelOptimizer.jl" begin
    if GROUP == "All" || GROUP == "Basics"
        @testset verbose=true "Basics" begin
            @safetestset "QA" include("qa.jl")
            @safetestset "Latency" include("latency.jl")
            @safetestset "Short tests" include("short_tests.jl")
            @safetestset "Objective function" include("objective.jl")
            # @safetestset "Utils" include("utils.jl")

            @testset verbose=true "Experiments" begin
                @safetestset "Basics" include(joinpath("experiment", "experiment.jl"))
                @safetestset "Data indexing" include(joinpath("experiment",
                    "data.jl"))
                @safetestset "Vector parameters" include(joinpath(
                    "experiment", "vector_parameters.jl"))
                @safetestset "Experiment collections" include(joinpath("experiment",
                    "collections.jl"))
                @safetestset "Aliasing and symbolic variable expressions" include(joinpath(
                    "experiment",
                    "aliasing.jl"))
            end
            @safetestset "Losses" include(joinpath("experiment", "loss.jl"))
            @safetestset "Missing data" include("missing_data.jl")
            # @safetestset "Penalty" include("penalty.jl")
            @safetestset "Thermal conduction" include("thermal_conduction.jl")
            @safetestset "Remake" include("remake.jl")
            @safetestset "Subsampling" include("subsample.jl")
        end
    end

    if GROUP == "All" || GROUP == "Performance"
        @safetestset "Performance" include("perf.jl")
    end

    if GROUP == "All" || GROUP == "UQ"
        @testset verbose=true "Parametric uncertainty quantification" begin
            @safetestset "Stochastic global optimization" begin
                include(joinpath("parametric_uq", "stochglobalopt.jl"))
            end
            # TODO update MCMC
            # @safetestset "MCMC" begin
            #     include(joinpath("parametric_uq", "mcmc.jl"))
            # end
        end
    end

    if GROUP == "All" || GROUP == "Calibrate1"
        @testset verbose=true "Calibrate1" begin
            @safetestset "AD backends" include(joinpath("calibrate", "ad.jl"))
            @testset verbose=true "Calibrate" begin
                include(joinpath("calibrate", "calibrate.jl"))
            end
            @safetestset "Dyad interface" include(joinpath("dyad", "analysis.jl"))
            @safetestset "Search space transformations" include(joinpath("calibrate",
                "search_space_transformations.jl"))
        end
    end

    if GROUP == "All" || GROUP == "Calibrate2"
        @testset verbose=true "Calibrate2" begin
            @safetestset "Collocation" include(joinpath("calibrate", "collocation.jl"))
            @safetestset "Model tests" begin
                @safetestset "Circuit Model - oscillatory behavior" begin
                    include(joinpath("calibrate", "de_sauty_bridge.jl"))
                end
                @safetestset "Ball and Beam - unstable solution & real data" begin
                    include(joinpath("calibrate", "ball_and_beam.jl"))
                end
            end
            @testset verbose=true "Constraints" begin
                include(joinpath("calibrate", "constraints.jl"))
            end
        end
    end

    if GROUP == "All" || GROUP == "Calibrate3"
        @testset verbose=true "Calibrate3" begin
            @testset verbose=true "Prediction Error Method" begin
                include(joinpath("calibrate", "prediction_error.jl"))
            end
        end
    end

    if GROUP == "All" || GROUP == "ImportExport"
        @testset verbose=true "Import/Export" begin
            @safetestset "I/O" include(joinpath("io", "io.jl"))
            @safetestset "DataSets" include(joinpath("io", "datasets.jl"))

            # TODO update PEtab
            # @testset "PEtab import tests" begin
            #     @safetestset "Unit Tests" begin
            #         include(joinpath("petabimport", "petab_unit_tests.jl"))
            #     end
            #     @safetestset "Test Suite" begin
            #         include(joinpath("petabimport", "petab_test_suite.jl"))
            #     end
            # end

            @safetestset "Analysis" include("analysis.jl")

            @testset "Plots" begin
                @safetestset "Confidence Plot" include("confidence_plot.jl")
                @safetestset "Plot SteadyStateExperimentSolution" include("plotsteadystatesolution.jl")
                @safetestset "Plot References" include("plot_regression.jl")
            end
        end
    end

    if GROUP == "All" || GROUP == "Parallelization"
        @testset verbose=true "Parallelization" begin
            @safetestset "Threaded" include("threaded.jl")
            # Distributed goes last, must not be a safetestset because needs to be run in Main
            @testset "Distributed" verbose=true begin
                include("distributed.jl")
            end
        end
    end

    if GROUP == "All" || GROUP == "Downstream"
        # activate_downstream_env()
        @testset verbose=true "Downstream" begin
            @safetestset "HVAC" include("downstream/HVAC/cycle_model_calibration.jl")
        end
    end

    if GROUP == "All" || GROUP == "JuliaHub"
        if run_juliahub_tests
            @testset verbose=true "JuliaHub" begin
                @safetestset "JuliaHub_1" begin
                    include("juliahub.jl")
                end
            end
        else
            @info "JuliaHub tests are skipped!"
        end
    end
end
