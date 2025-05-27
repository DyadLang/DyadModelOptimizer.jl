using DyadModelOptimizer
using Test
using DataFrames
using ModelingToolkit
using OptimizationBBO
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve

include("cycle_model_def.jl")

function sanity_check(patch_params)
    model = build_model(; patch_params)
    # test that unmodified model runs
    prob = ODEProblem(model, [], (0.0, 1400.0))
    @info "prob created"
    stats = @timed init(prob, Rodas5P())
    @info "init done in $(stats.time) seconds, $(100*stats.gctime / stats.time)% gctime ($(stats.bytes / 1e9) GiB)"
    solve(prob, Rodas5P())
end

function build_inverse_problem(sysRed)
    tunable_params = ModelingToolkit.tunable_parameters(sysRed; default = false)
    # @info tunable_params
    # defs_dict, _ = HVAC.evaluate_dict(MTK.defaults(sysRed))
    tunable_params_defaults = map(Float64, [defaults(sysRed)[p] for p in tunable_params])
    # @info tunable_params_defaults
    lb = tunable_params_defaults .- abs.(0.1 .* tunable_params_defaults)
    ub = tunable_params_defaults .+ abs.(0.1 .* tunable_params_defaults)

    prob = ODEProblem(sysRed, [], (0.0, 1400.0),
        tunable_params .=> 0.95 * tunable_params_defaults, use_scc = true)
    sol = solve(prob, Rodas5P())

    data1 = DataFrame(sol)

    experiment1 = Experiment(data1[:, 1:10], sysRed;
        alg = Rodas5P(),
        prob_kwargs = (use_scc = true,),
        loss_func = meansquaredl2loss)

    return InverseProblem(experiment1, tunable_params .=> zip(lb, ub))
end

@testset "cycle model with moist air" begin
    sol = sanity_check(false)
    @test SciMLBase.successful_retcode(sol)
    sol = sanity_check(true)
    @test SciMLBase.successful_retcode(sol)

    model = build_model()
    maxiters = 1000

    alg = SingleShooting(;
        maxiters,
        optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited()
    )

    invprob = build_inverse_problem(model)
    r1 = calibrate(invprob, alg)

    sol = simulate(invprob.experiments[1], r1)

    ca = sol.ps[model.evaporator.circuit.crossArea]
    ca1 = sol.ps[model.evaporator.circuit.crossAreas[1]]
    ca2 = sol.ps[model.evaporator.circuit.crossAreas[2]]

    @test ca == ca1
    @test ca1 == ca2
    @test ca == ca2
end
