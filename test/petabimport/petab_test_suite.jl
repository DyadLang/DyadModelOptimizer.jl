using Test
using CSV
using DataFrames
using OrdinaryDiffEq
using YAML
using DyadModelOptimizer
using DyadModelOptimizer: get_petab_problem,
                          get_petab_simulation_df,
                          problem_chi2, setup_problem, CalibrationResult

include("utils.jl")
function run_test_case(dir, case, hash)  # Derived from https://github.com/LCSB-BioCore/SBML.jl, LCSB-BioCore licensed, see repository for details
    download_case(dir, case, hash)

    @testset "Loading of $case" begin
        sol_true = YAML.load_file(joinpath(dir, "_$(case)_solution.yaml"))
        sim_true = CSV.read(joinpath(dir, sol_true["simulation_files"][1]), DataFrame)

        yaml_fn = joinpath(dir, "_$case.yaml")
        prob = import_petab(yaml_fn)
        res = calibrate(prob, SingleShooting(maxiters = 1))  # Check if calibrate runs
        sol = simulate(get_experiments(prob)[1], prob)
        # Test simulation result
        x0 = DyadModelOptimizer.initial_state(Any, prob)
        defres = CalibrationResult(x0, res.prob, nothing, x0, res.alg, [], missing)
        pp = get_petab_problem(yaml_fn)
        sim = get_petab_simulation_df(defres, pp)
        @test isequal(sim[!, 1:3], sim_true[!, 1:3])
        if case == "0012"  # Test suite uses concentration, we use amounts
            @test isapprox(sim[!, 4] ./ 3, sim_true[!, 4],
                atol = sol_true["tol_simulations"])
        else
            @test isapprox(sim[!, 4], sim_true[!, 4], atol = sol_true["tol_simulations"])

            if !in(case, ["0007", "0016"])  # Log transformed observable
                # Test chi2
                chi2 = problem_chi2(defres)
                @test isapprox(chi2, sol_true["chi2"], atol = sol_true["tol_chi2"])
            else
                @warn "Chi2 calculation for log-transformed observable not supported."
            end

            # Test log-likelihood
            cost = objective(prob, SingleShooting(maxiters = 1))
            @test isapprox(cost(), -sol_true["llh"], atol = sol_true["tol_llh"])
        end
    end
end

run_test_case(dir, "0002", "hash")

const cases = [
    ("0001", "cab76cd62aefd201556b1bf8b1c96f4c86a46eb619d70ce7524620233c79e015"),  # Nothing special.
    ("0002", "cab76cd62aefd201556b1bf8b1c96f4c86a46eb619d70ce7524620233c79e015"),  # Two conditions. Numeric parameter override.
    # ("0003", "bc89895af089ca421021b3d78d0a801e78e4014e0735c67381fef5a85853bc89"),  # Numeric observable parameter overrides in measurement table.
    ("0004", "bdd3abf6df5d248de531319e38093a6fffaf2ba5093276e682929448660a9db2"),  # Observable parameters only defined in parameter table.
    ("0005", "8432fdb8e961ce3a1b5b9a3a0f580571ec3341b3970763d74c9ff921039e147c"),  # Condition-specific parameters only defined in parameter table.
    # ("0006", "727b85e0266b9e84a68ef9b0391aca3a8131c1a5a9be534c9f02371f99fcfe0d"),  # Time-point specific numeric observable parameter overrides.
    ("0007", "104cec5b52fe1bbe3d69ca2daaa9f18f1f2cfa416420bb0f6ca8f448ffbdf8f5"),  # Observable transformation log10. (Single-time point data).)
    ("0008", "bdd3abf6df5d248de531319e38093a6fffaf2ba5093276e682929448660a9db2"),  # Replicate measurements.
    # ("0009", "bdd3abf6df5d248de531319e38093a6fffaf2ba5093276e682929448660a9db2"),  # Preequilibration.
    # ("0010", "bdd3abf6df5d248de531319e38093a6fffaf2ba5093276e682929448660a9db2"),  # One species reinitialized, one not. InitialAssignment to species overridden. (Preequilibration)
    ("0011", "bdd3abf6df5d248de531319e38093a6fffaf2ba5093276e682929448660a9db2"),  # InitialAssignment to species overridden.
    ("0012", "bdd3abf6df5d248de531319e38093a6fffaf2ba5093276e682929448660a9db2")  # Initial compartment size in condition table.    # ("0013", "bdd3abf6df5d248de531319e38093a6fffaf2ba5093276e682929448660a9db2"),  # Species with InitialAssignment overridden by parameter.    # ("0014", "bdd3abf6df5d248de531319e38093a6fffaf2ba5093276e682929448660a9db2"),  # Multiple numeric noise parameter overrides.    # ("0015", "bdd3abf6df5d248de531319e38093a6fffaf2ba5093276e682929448660a9db2"),  # Single parametric noise parameter override.    # ("0016", "bdd3abf6df5d248de531319e38093a6fffaf2ba5093276e682929448660a9db2"),  # Observable transformation log. (Single-time point data).    # ("00017", "hash"),  # Partial preequilibration with NaN's in the condition file  # Preequilibration. One species reinitialized, one not (NaN in condition table). InitialAssignment to species overridden.    # ("00018", "hash"),  # Preequilibration and RateRules  # Preequilibration and RateRules. One state reinitialized, one not (NaN in condition table). InitialAssignment to species overridden.
]

for (case, hash) in cases
    try
        run_test_case(dir, case, hash)
    finally
        rm(dir, recursive = true)
    end
end
