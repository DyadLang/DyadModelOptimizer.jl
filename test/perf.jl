using JET
using DyadModelOptimizer
using OrdinaryDiffEqTsit5, OrdinaryDiffEqDefault
using Test
using BenchmarkTools

include("reactionsystem.jl")

function prepare(experiment, x, invprob, default_u0 = Val(:default))
    prob = DyadModelOptimizer.setup_problem(experiment, invprob, x, default_u0)
    alg = DyadModelOptimizer.get_solve_alg(experiment)
    saveat = DyadModelOptimizer.get_saveat(experiment, x, invprob)
    save_idxs = DyadModelOptimizer.get_save_idxs(experiment)
    kwargs = DyadModelOptimizer.get_kwargs(experiment)

    return prob, alg, save_idxs, saveat, kwargs
end

function judge_regression(b1, b2, estimator; tune = true, time_tolerance = 0.25,
        memory_tolerance = 0.05)
    GC.gc()
    tune && tune!(b1)
    tune && tune!(b2)
    r1 = run(b1)
    r2 = run(b2)
    # display(r1)
    # display(r2)
    e1 = estimator(r1)
    e2 = estimator(r2)

    return judge(e1, e2; memory_tolerance, time_tolerance)
end

@testset "cost overhead with 1 independent experiment" begin
    model = reactionsystem()
    data = generate_data(model)

    x = [1.0, 2.0]
    experiment1 = Experiment(data, model, tspan = (0.0, 1.0))
    prob1 = InverseProblem(experiment1, [model.k1 => (0, 5), model.c1 => (0, 5)])
    cost1 = objective(prob1, SingleShooting(maxiters = 1))
    p1 = DyadModelOptimizer.calibration_parameters(SingleShooting(maxiters = 1), prob1)

    experiment2 = Experiment(data, model, alg = Tsit5(), tspan = (0.0, 1.0))
    prob2 = InverseProblem(experiment2, [model.k1 => (0, 5), model.c1 => (0, 5)])
    cost2 = objective(prob2, SingleShooting(maxiters = 1))
    p2 = DyadModelOptimizer.calibration_parameters(SingleShooting(maxiters = 1), prob2)

    @testset "auto-alg" begin
        prob, alg, save_idxs, saveat, kwargs = prepare(experiment1, x, prob1)

        solve_bench = @benchmarkable solve($prob, $alg;
            save_idxs = $save_idxs,
            saveat = $saveat,
            $kwargs...)
        # @btime $cost1(): 6.900 μs (107 allocations: 12.47 KiB)
        simulate_bench = @benchmarkable simulate($experiment1, $prob1, $x)
        cost_bench = @benchmarkable $cost1($x, $p1)
        # Check with auto-alg
        @test_opt broken=false target_modules=(DyadModelOptimizer,) cost1(x, p1)
        # <25% overhead
        # More advanced profiling with Tracy shows that we spend ~80% of the time
        # in the DiffEq solve, 6% in setup, 4% in getting the solve args
        # and a 10% is not clearly accounted for
        solve_overhead = judge_regression(simulate_bench,
            solve_bench,
            minimum)
        display(solve_overhead)
        # TODO: setup env where it can be tested reliably and this can be uncommented
        @inferred simulate(experiment1, prob1, x)
        # check that the alg is prepared
        @test DyadModelOptimizer.get_solve_alg(experiment1) ≠
              DefaultODEAlgorithm()
        # @test solve_overhead.time !== :regression
        # @test solve_overhead.memory !== :regression

        # overhead of cost over solve
        cost_overhead = judge_regression(cost_bench,
            simulate_bench,
            minimum)
        display(cost_overhead)
        # there is an unexplained large allocation from using try/catch
        # it corresponds to the return type of the simulate call
        # TODO: setup env where it can be tested reliably and this can be uncommented
        # @test cost_overhead.time !== :regression
        # @test cost_overhead.memory !== :regression broken=true
    end

    @testset "alg provided" begin
        prob, alg, save_idxs, saveat, kwargs = prepare(experiment2, x, prob2)
        solve_bench = @benchmarkable solve($prob, $alg;
            save_idxs = $save_idxs,
            saveat = $saveat,
            $kwargs...)
        # @btime $cost2(): 4.600 μs (65 allocations: 8.94 KiB)
        simulate_bench = @benchmarkable simulate($experiment2, $prob2, $x)
        cost_bench = @benchmarkable $cost2($x, $p2)
        # Check with alg provided
        @test_opt broken=false target_modules=(DyadModelOptimizer,) cost2(x, p2)
        solve_overhead = judge_regression(simulate_bench,
            solve_bench,
            minimum)
        display(solve_overhead)
        # TODO: setup env where it can be tested reliably and this can be uncommented
        @inferred simulate(experiment2, prob2, x)
        @test solve_overhead.time!==:regression broken=true
        @test solve_overhead.memory!==:regression broken=true

        # overhead of cost over solve
        cost_overhead = judge_regression(cost_bench,
            simulate_bench,
            minimum,
            time_tolerance = 0.05)
        display(cost_overhead)
        # there is an unexplained large allocation from using try/catch
        # it corresponds to the return type of the simulate call
        # TODO: setup env where it can be tested reliably and this can be uncommented
        @test cost_overhead.time!==:regression broken=false
        @test cost_overhead.memory!==:regression broken=false
    end
end
