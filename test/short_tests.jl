using Test
using DyadModelOptimizer
using ModelingToolkit
using Statistics
using OptimizationBBO
using OrdinaryDiffEqDefault
using JSON3
using DyadInterface, DyadData

json_str = """
{
    "name": "reactionsystemOptimize",
    "result": {
        "name": "reactionsystemTransient",
        "integrator": "auto",
        "abstol": 1e-6,
        "reltol": 1e-3,
        "start": 0.0,
        "stop": 10.0
    },
    "data": {
        "filename": "file://$(normpath(joinpath(@__DIR__, "reaction_system_data.csv")))",
        "ivar": "t",
        "dvar": ["u1", "u3", "u2"]
    },
    "method": "SingleShooting",
    "optimizer": "Ipopt",
    "loss": "l2loss",
    "maxiters": 1000,
    "parameters": {
        "c1": {
            "min": 0,
            "max": 5,
            "default": 3.1
        }
    },
    "signals": ["s1", "s1s2", "s2"]
}
"""

include("reactionsystem.jl")

@testset "DyadModelOptimizer.jl" begin
    @testset "CalibrationAnalysis with local file" begin
        model = reactionsystem()
        spec = CalibrationAnalysisSpec(;
            name = :test,
            model,
            abstol = 1e-6,
            reltol = 1e-3,
            data = DyadDataset(
                "$(normpath(joinpath(@__DIR__, "reaction_system_data.csv")))",
                independent_var = "t", dependent_vars = ["u1", "u3", "u2"]),
            N_cols = 3,
            depvars_cols = ["u1", "u3", "u2"],
            depvars_names = ["s1", "s1s2", "s2"],
            N_tunables = 1,
            search_space_names = ["c1"],
            search_space_lb = [0.0],
            search_space_ub = [5.0],
            calibration_alg = "SingleShooting",
            optimizer_maxiters = 1000
        )

        r = run_analysis(spec)

        @test r.r[1]≈2 rtol=1e-5
    end

    @testset "Parametric uncertainty quantification" begin
        model = reactionsystem()
        @unpack k1, c1 = model
        data = generate_data(model)
        experiment = Experiment(data, model, tspan = (0.0, 1.0))

        prob = InverseProblem(experiment, [k1 => (0, 5), c1 => (0, 5)])

        res = parametric_uq(prob,
            StochGlobalOpt(method = SingleShooting(maxiters = 10,
                optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited())),
            sample_size = 50)
        @test length(res) == 50
        @test !iszero(res)
    end
end
