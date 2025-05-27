module PEMTests

using DyadModelOptimizer
using DyadModelOptimizer: timeseries_data,
                          determine_save_names, determine_saveat,
                          prediction_error_callback, NoData, setup_nonlinearsystem,
                          discretefixedgainpem_nonlinearproblem,
                          discretefixedgainpem_setp,
                          pem_metadata, data_shape
using DataFrames
using ModelingToolkit
import ModelingToolkit: D_nounits as D, t_nounits as t
using Optimization
using OptimizationNLopt, OptimizationOptimJL, OptimizationBBO
using LineSearches
using OrdinaryDiffEqTsit5, OrdinaryDiffEqRosenbrock, OrdinaryDiffEqVerner,
      OrdinaryDiffEqNonlinearSolve
using Test
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks
import DataInterpolations: CubicSpline
import ForwardDiff
using CSV
using SciMLBase
using SymbolicIndexingInterface: setp
using DyadData

include("../reactionsystem.jl")

@testset "Cases for observable inversion" begin
    @testset "Number of equations equal to unknowns" begin
        @testset "1 Equation" begin
            # u1 is an unknown and u2 is a parameter for the Non linear system
            @variables u1 u2
            @parameters k=1.0 r=1.0
            eq = [0 ~ -u2 + (k - u1) / r]
            eq = substitute(eq, u2 => ModelingToolkit.toparam(u2))
            sys = setup_nonlinearsystem(eq, [u1], [u2, k, r])
            @test length(unknowns(sys)) == 0
            u2_val = 2.0
            k_val = 0.0
            r_val = 1.0
            np = NonlinearLeastSquaresProblem(
                sys, [], [u2 => u2_val, k => k_val, r => r_val]; check_length = false)
            sol = solve(np)
            @test SciMLBase.successful_retcode(sol)
            @test sol[u1] == r_val * (-u2_val + k_val / r_val)
        end

        @testset ">1 Equations" begin
            # u1, u2 are unknowns and u3, u4 are parameters for the Non linear system
            @variables u1 u2 u3 u4
            @parameters k=1.0 r=1.0
            eqs = [u3 ~ (k - u1) / r, u4 ~ u1 + u2 + 2 * k]
            new_eqs = []
            for eq in eqs
                push!(new_eqs,
                    substitute(eq,
                        Dict([u3, u4] .=>
                            [ModelingToolkit.toparam(u3), ModelingToolkit.toparam(u4)])))
            end
            sys = setup_nonlinearsystem(
                new_eqs, [u1, u2], [u3, u4, k, r])
            @test length(unknowns(sys)) == 0
            u3_val = 1.0
            u4_val = 1.0
            k_val = 0.0
            r_val = 1.0
            np = NonlinearLeastSquaresProblem(
                sys, [], [u3 => u3_val, u4 => u4_val, k => k_val, r => r_val];
                check_length = false)
            sol = solve(np)
            @test SciMLBase.successful_retcode(sol)
            @test sol[u1] == r_val * (-u3_val + k_val / r_val)
            @test sol[u2] == -2 * k_val - sol[u1] + u4_val
        end
    end

    @testset "Number of unknowns more than equations" begin
        @testset "Infinite solutions" begin
            @testset "reduced to 1 unknowns" begin
                # u1, u2, u3 are unknowns and u4, u5 are parameters for the Non linear system
                @variables u1 u2 u3 u4 u5
                @parameters k=1.0 r=1.0
                eqs = [u4 ~ (k - u1) / r + u3, u5 ~ u1 + u2 + 2 * k]
                new_eqs = []
                for eq in eqs
                    push!(new_eqs,
                        substitute(eq,
                            Dict([u4, u5] .=>
                                [ModelingToolkit.toparam(u4), ModelingToolkit.toparam(u5)])))
                end
                sys = setup_nonlinearsystem(
                    new_eqs, [u1, u2, u3], [u4, u5, k, r])
                @test length(unknowns(sys)) == 1
                u4_val = 1.0
                u5_val = 1.0
                k_val = 0.0
                r_val = 1.0
                u2_val = 5.0
                u3_val = 5.0
                np = NonlinearLeastSquaresProblem(sys, [u2 => u2_val, u3 => u3_val],
                    [u4 => u4_val, u5 => u5_val, k => k_val, r => r_val];
                    check_length = false)
                sol = solve(np)
                @test SciMLBase.successful_retcode(sol)
                if Symbol(unknowns(sys)[1]) == :u3
                    # This is infinite solutions as any value of `u3` can be substituted
                    @test sol[u3] == u3_val
                    @test sol[u1] == r_val * (sol[u3] - u4_val + k_val / r_val)
                    @test sol[u2] == -2 * k_val - sol[u1] + u5_val
                elseif Symbol(unknowns(sys)[1]) == :u2
                    # This is infinite solutions as any value of `u2` can be substituted
                    @test sol[u2] == u2_val
                    @test sol[u1] == -2 * k_val - u2_val + u5_val
                    @test sol[u3] == u4_val + (-k_val + sol[u1]) / r_val
                end
            end
            @testset "reduced to 2 unknowns" begin
                # u1, u2, u3 are unknowns and u4, u5 are parameters for the Non linear system
                @variables u1 u2 u3 u4 u5
                eqs = [u4 ~ 2 + u1 + u2 + u3, u5 ~ 3 + u1 + u2 + u3]
                new_eqs = []
                for eq in eqs
                    push!(new_eqs,
                        substitute(eq,
                            Dict([u4, u5] .=>
                                [ModelingToolkit.toparam(u4), ModelingToolkit.toparam(u5)])))
                end
                sys = setup_nonlinearsystem(
                    new_eqs, [u1, u2, u3], [u4, u5])
                @test length(unknowns(sys)) == 2
                u4_val = 1.0
                u5_val = 2.0
                u2_val = -10.0
                u3_val = 0.0
                np = NonlinearLeastSquaresProblem(sys, [u2 => u2_val, u3 => u3_val],
                    [u4 => u4_val, u5 => u5_val]; check_length = false)
                sol = solve(np)
                @test SciMLBase.successful_retcode(sol)
                @test sol[u1] == -2 - u2_val - u3_val + u4_val
                @test sol[u2] == u2_val
                @test sol[u3] == u3_val
            end
        end

        @testset "No solutions" begin
            @testset "reduced to 2 unknowns" begin
                # u1, u2, u3 are unknowns and u4, u5 are parameters for the Non linear system
                @variables u1 u2 u3 u4 u5
                eqs = [u4 ~ 2 + u1 + u2 + u3, u5 ~ 3 + u1 + u2 + u3]
                new_eqs = []
                for eq in eqs
                    push!(new_eqs,
                        substitute(eq,
                            Dict([u4, u5] .=>
                                [ModelingToolkit.toparam(u4), ModelingToolkit.toparam(u5)])))
                end
                sys = setup_nonlinearsystem(
                    new_eqs, [u1, u2, u3], [u4, u5])
                @test length(unknowns(sys)) == 2
                u4_val = 1.0
                u5_val = 0.0
                u2_val = -10.0
                u3_val = 0.0
                np = NonlinearLeastSquaresProblem(sys, [u2 => u2_val, u3 => u3_val],
                    [u4 => u4_val, u5 => u5_val]; check_length = false)
                sol = solve(np)
                @test SciMLBase.successful_retcode(sol)
            end
        end
    end

    @testset "Number of unknowns less than equations" begin
        @testset "Unique solution" begin
            # u1, u2 are unknowns and u3, u4 are parameters for the Non linear system
            @variables u1 u2 u3 u4
            eqs = [u3 ~ u1 + 1.0, u4 ~ u2 + 2.0, u3 + u4 ~ u1 + u2 + 3.0]
            new_eqs = []
            for eq in eqs
                push!(new_eqs,
                    substitute(eq,
                        Dict([u3, u4] .=>
                            [ModelingToolkit.toparam(u3), ModelingToolkit.toparam(u4)])))
            end
            sys = setup_nonlinearsystem(new_eqs, [u1, u2], [u3, u4])
            @test length(unknowns(sys)) == 0
            u3_val = 1.0
            u4_val = 1.0
            np = NonlinearLeastSquaresProblem(sys, [], [u3 => u3_val, u4 => u4_val])
            sol = solve(np)
            @test SciMLBase.successful_retcode(sol)
            @test sol[u1] == -1.0 + u3_val
            @test sol[u2] == -2.0 + u4_val
        end

        @testset "No solution" begin
            # u1, u2 are unknowns and u3, u4 are parameters for the Non linear system
            @variables u1 u2 u3 u4
            eqs = [u3 ~ u1 + 1.0, u4 ~ u2 + 2.0, u3 + u4 ~ u1 + u2 + 4.0]
            new_eqs = []
            for eq in eqs
                push!(new_eqs,
                    substitute(eq,
                        Dict([u3, u4] .=>
                            [ModelingToolkit.toparam(u3), ModelingToolkit.toparam(u4)])))
            end
            sys = setup_nonlinearsystem(new_eqs, [u1, u2], [u3, u4])
            @test length(unknowns(sys)) == 0
            u3_val = 1.0
            u4_val = 1.0
            np = NonlinearLeastSquaresProblem(sys, [], [u3 => u3_val, u4 => u4_val])
            sol = solve(np)
            # Inconsistent
            @test !SciMLBase.successful_retcode(sol)
        end

        @testset "Infinite solutions" begin
            # u1, u2 are unknowns and u3, u4 are parameters for the Non linear system
            @variables u1 u2 u3 u4
            eqs = [u3 ~ u1 + u2, u4 ~ 2 * (u1 + u2), u3 + u4 ~ 3 * (u1 + u2)]
            new_eqs = []
            for eq in eqs
                push!(new_eqs,
                    substitute(eq,
                        Dict([u3, u4] .=>
                            [ModelingToolkit.toparam(u3), ModelingToolkit.toparam(u4)])))
            end
            sys = setup_nonlinearsystem(new_eqs, [u1, u2], [u3, u4])
            @test length(unknowns(sys)) == 1
            u3_val = 1.0
            u4_val = 2.0
            u2_val = 10.0
            np = NonlinearLeastSquaresProblem(
                sys, [u2 => u2_val], [u3 => u3_val, u4 => u4_val])
            sol = solve(np)
            @test SciMLBase.successful_retcode(sol)
            @test sol[u2] == u2_val
            @test sol[u1] == -u2_val + u3_val
        end
    end
end

@testset verbose=true "Return type" begin
    model = reactionsystem()
    data = generate_data(model)
    tspan = (0.0, 1.0)
    experiment_data = timeseries_data(data, determine_save_names(data, model),
        :timestamp, determine_saveat(data), tspan)
    @testset "Normal usage" begin
        pe_callback = prediction_error_callback(
            DiscreteFixedGainPEM(0.1), experiment_data, model)
        @test typeof(pe_callback) <: DiscreteCallback
    end

    @testset "Data is nothing" begin
        pe_callback = @test_logs (:error,
            "Cannot use Predictive Error Method as there is no data") prediction_error_callback(
            DiscreteFixedGainPEM(0.1),
            NoData(), model)
        @test isnothing(pe_callback)
    end
end

@testset verbose=true "Reaction System" begin
    model = reactionsystem()
    @unpack c1 = model
    true_c = 4
    data = generate_data(model, params = [c1 => true_c])

    experiment = Experiment(data, model, tspan = (0.0, 1.0);
        model_transformations = [DiscreteFixedGainPEM(0.1)])
    invprob = InverseProblem([experiment], [c1 => (0, 5)])

    @testset "SingleShooting" begin
        alg = SingleShooting(maxiters = 10^3)
        r = calibrate(invprob, alg)
        @test only(r)≈true_c rtol=1e-4
    end

    @testset "MultipleShooting" begin
        alg = MultipleShooting(maxiters = 10^5,
            trajectories = 3,
            continuitylossweight = 10^2)
        r = calibrate(invprob, alg)
        @test only(r)≈true_c rtol=2e-4
    end

    single_state_data = data[!, ["timestamp", "s1(t)"]]
    experiment = Experiment(single_state_data, model, tspan = (0.0, 1.0);
        model_transformations = [DiscreteFixedGainPEM(1.0)])
    invprob = InverseProblem(experiment, [c1 => (0, 5)])

    @testset "Single State SingleShooting" begin
        alg = SingleShooting(maxiters = 10^3,
            optimizer = NLopt.G_MLSL_LDS(),
            local_method = NLopt.LD_LBFGS())
        r = calibrate(invprob, alg)
        @test only(r)≈true_c rtol=1e-4
    end
end

@testset verbose=true "De-Sauty bridge" begin
    function create_model()
        @named resistor1 = Resistor(R = 5.0)
        @named resistor2 = Resistor(R = 2.0)
        @named capacitor1 = Capacitor(C = 2.4)
        @named capacitor2 = Capacitor(C = 60.0)
        @named source = Voltage()
        @named input_signal = Sine(frequency = 1.0)
        @named ground = Ground()
        @named ampermeter = CurrentSensor()

        eqs = [connect(input_signal.output, source.V)
               connect(source.p, capacitor1.n, capacitor2.n)
               connect(source.n, resistor1.p, resistor2.p, ground.g)
               connect(resistor1.n, capacitor1.p, ampermeter.n)
               connect(resistor2.n, capacitor2.p, ampermeter.p)]

        @named circuit_model = ODESystem(eqs, t,
            systems = [
                resistor1, resistor2, capacitor1, capacitor2,
                source, input_signal, ground, ampermeter
            ])
    end

    model = create_model()
    sys = structural_simplify(model)
    t0 = 0.0
    tend = 3.0
    @unpack ampermeter, resistor1, capacitor1, capacitor2 = model

    prob = ODEProblem(
        sys, [capacitor2.v => 0.0], (t0, tend), [capacitor2.C => 6.0])
    sol = solve(prob, Rodas5P())
    data1 = DataFrame("timestamp" => sol.t,
        "capacitor2.v" => sol[capacitor2.v],
        "capacitor1.i" => sol[capacitor1.i])
    data2 = DataFrame("timestamp" => sol.t,
        "ampermeter.i" => sol[ampermeter.i],
        "capacitor1.i" => sol[capacitor1.i])
    data3 = DataFrame("timestamp" => sol.t,
        "ampermeter.i" => sol[ampermeter.i],
        "resistor1.i" => sol[resistor1.i])

    experiment1 = Experiment(data1, sys;
        overrides = [capacitor2.v => 0.0],
        tspan = (t0, tend),
        model_transformations = [DiscreteFixedGainPEM(0.2)],
        alg = Rodas5P())
    experiment2 = Experiment(data2, sys;
        overrides = [capacitor2.v => 0.0],
        tspan = (t0, tend),
        model_transformations = [DiscreteFixedGainPEM(0.2)],
        alg = Rodas5P())
    experiment3 = Experiment(data3, sys;
        overrides = [capacitor2.v => 0.0],
        tspan = (t0, tend),
        model_transformations = [DiscreteFixedGainPEM(0.2)],
        alg = Rodas5P())
    experiments = [experiment1, experiment2, experiment3]

    ss = [capacitor2.C => (1.0, 100.0)]

    invprob1 = InverseProblem(experiment1, ss)
    invprob2 = InverseProblem(experiment2, ss)
    invprob3 = InverseProblem(experiment3, ss)
    invprobs = [invprob1, invprob2, invprob3]

    @testset "Inversion" begin
        @testset "$s" for (i, s) in zip(2:3, ["states + observed", "all observed"])
            experiment_data = experiments[i].config.data
            md = pem_metadata(experiment_data, sys)
            np, estimatable_unknowns, estimatable_unknowns_idxs = discretefixedgainpem_nonlinearproblem(
                sys, md, experiment_data)
            setp_func = discretefixedgainpem_setp(
                np, md, data_shape(typeof(experiment_data)))
            for i in eachindex(experiment_data.time)
                setp_func(np,
                    [experiment_data[md.observed_idxs_data, i]...,
                        experiment_data[md.unknowns_idxs_data, i]..., experiment_data.time[i]])
                sol_np = solve(np)
                @test sol_np[estimatable_unknowns] ≈ sol[estimatable_unknowns][i]
            end
        end
    end

    @testset "SingleShooting" begin
        alg = SingleShooting(maxiters = 10^5,
            optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited())
        @testset "$s" for (i, s) in enumerate([
            "all states", "states + observed", "all observed"])
            r = calibrate(invprobs[i], alg)
            # C2 = C1 * R1/R2
            @test only(r)≈6 rtol=1e-2
        end
    end

    @testset verbose=true "MultipleShooting" begin
        alg = MultipleShooting(
            maxtime = 400, trajectories = 11, continuitylossweight = 100,
            initialization = DataInitialization())
        @testset "$s" for (i, s) in enumerate([
            "all states", "states + observed", "all observed"])
            r = calibrate(invprobs[i], alg)
            # C2 = C1 * R1/R2
            @test only(r)≈6 rtol=1e-2
        end
    end

    @testset "Error Conditions" begin
        data = DataFrame("timestamp" => sol.t,
            "ampermeter.i" => sol[ampermeter.i])
        @test_throws ErrorException Experiment(data, sys;
            tspan = (t0, tend),
            model_transformations = [DiscreteFixedGainPEM(0.2)],
            alg = Rodas5P())
    end
end

@testset verbose=true "Ball and Beam" begin
    function ballandbeamsystem(x0, ϕ_vec, t_vec)
        ϕ0 = ϕ_vec[1]
        @variables x(t)=x0 ϕ(t)=ϕ0 ball_position(t)
        @parameters g=9.81 I=0.1 Fv=1

        @named src = Interpolation(CubicSpline, ϕ_vec, t_vec)
        @named clk = ContinuousClock()
        @named Dϕ_input = RealInput()

        eqs = [
            connect(clk.output, src.input),
            connect(src.output, Dϕ_input),
            D(ϕ) ~ Dϕ_input.u,
            # I = J/(m*r^2)
            0 ~ (1 + I) * D(D(x)) - x * (D(ϕ))^2 - g * sin(ϕ) + Fv * D(x)
        ]
        @named ballandbeam = ODESystem(eqs, t; systems = [Dϕ_input, src, clk])
    end

    training_dataset = DyadDataset("juliasimtutorials", "ball_beam",
        independent_var = "timestamp", dependent_vars = ["ϕ", "x"])
    df = build_dataframe(training_dataset)

    t_vec = df[!, "timestamp"]
    ϕ_vec = df[:, "ϕ"]

    input_func = CubicSpline(ϕ_vec, t_vec)
    dinput_vec = map(Base.Fix1(ForwardDiff.derivative, input_func), t_vec)

    model = ballandbeamsystem(df.x[1], dinput_vec, t_vec)
    sys = structural_simplify(model)

    # check that we don't get any warnings or errors (like PEM not applicable)
    experiment = @test_nowarn Experiment(df,
        sys,
        overrides = [D(sys.x) => dinput_vec[1]],
        reltol = 1e-10,
        abstol = 1e-8,
        dense = false,
        model_transformations = [DiscreteFixedGainPEM(0.3)])

    experiment_no_pem = Experiment(df,
        sys,
        overrides = [D(sys.x) => dinput_vec[1]],
        reltol = 1e-10,
        abstol = 1e-8,
        dense = false)

    prob = InverseProblem(experiment,
        [sys.I => (0.0, 5.0), sys.Fv => (0, 2)])

    prob_no_pem = InverseProblem(experiment_no_pem,
        [sys.I => (0.0, 5.0), sys.Fv => (0, 2)])

    @testset "SingleShooting" begin
        alg = SingleShooting(maxiters = 10^5, maxtime = 100)
        r = calibrate(prob, alg)
        @test r[:I] < 10 # test that it doesn't hit the upper bound
        @test r[:I] > 0.1 # we want to check that we don't just get the initial value
        @test r[:I] < 3
        @test r.original.objective < 0.1
        r_no_pem = calibrate(prob_no_pem, alg)
        @test r[:I] < r_no_pem[:I] # test if the method worked
    end

    @testset "MultipleShooting" begin
        alg = MultipleShooting(maxiters = 10^3, maxtime = 350, trajectories = 100,
            optimizer = NLopt.LD_LBFGS(),
            continuitylossweight = 1e3,
            initialization = DataInitialization())
        r = calibrate(prob, alg)
        @test r[:I] < 5
        @test r[:I] > 0.1
        @test r[:I] < 3
        @test r.original.objective < 0.1
    end
end

@testset verbose=true "Pendulum" begin
    @variables x₁(t)=0.0 x₂(t)=3.0 m(t)
    @parameters L = 1.5
    @constants g = 9.81

    tspan = (0.1, 20.0)
    tsteps = range(tspan[1], tspan[2], length = 1000)
    true_L = 0.2
    eqs = [
        D(x₁) ~ x₂,
        D(x₂) ~ -(g / L) * sin(x₁)
    ]
    eqs2 = [
        D(x₁) ~ x₂,
        D(x₂) ~ -(g / L) * sin(x₁),
        m ~ 3 * x₁
    ]

    model = complete(ODESystem(eqs, t, [x₁, x₂], [L]; tspan, name = :model))
    odeprob = ODEProblem(model, [L => true_L])
    sol = solve(odeprob,
        Tsit5(),
        saveat = tsteps,
        abstol = 1e-8,
        reltol = 1e-8
    )
    data1 = DataFrame(sol)
    experiment1 = @test_nowarn Experiment(data1,
        model,
        alg = Tsit5(),
        reltol = 1e-8,
        abstol = 1e-8,
        tspan = tspan,
        model_transformations = [DiscreteFixedGainPEM(0.5)])
    @unpack L = model
    invprob1 = InverseProblem(experiment1, [L => (0.01, 2.0)])

    @named model2 = ODESystem(eqs2, t, [x₁, x₂], [L]; tspan)
    sys2 = structural_simplify(model2)
    odeprob2 = ODEProblem(sys2, [L => true_L])
    sol2 = solve(odeprob2,
        Tsit5(),
        saveat = tsteps,
        abstol = 1e-8,
        reltol = 1e-8
    )
    data2 = DataFrame(
        "timestamp" => sol2.t,
        "m" => sol2[m],
        "x₂" => sol2[x₂]
    )
    data3 = DataFrame(
        "timestamp" => sol2.t,
        "m" => sol2[m]
    )
    experiment2 = Experiment(data2,
        sys2,
        alg = Tsit5(),
        reltol = 1e-8,
        abstol = 1e-8,
        tspan = tspan,
        model_transformations = [DiscreteFixedGainPEM(0.5)])
    experiment3 = Experiment(data3,
        sys2,
        alg = Tsit5(),
        reltol = 1e-8,
        abstol = 1e-8,
        tspan = tspan,
        model_transformations = [DiscreteFixedGainPEM(0.35)])
    @unpack L = model2
    invprob2 = InverseProblem(experiment2, [L => (0.01, 2.0)])
    invprob3 = InverseProblem(experiment3, [L => (0.01, 2.0)])

    invprobs = [invprob1, invprob2, invprob3]

    @testset "SingleShooting" begin
        alg = SingleShooting(maxiters = 10^3)
        @testset "$s" for (i, s) in enumerate([
            "all states", "states + observed", "all observed"])
            r = calibrate(invprobs[i], alg)
            @test only(r)≈true_L rtol=1e-7
        end
    end

    @testset "MultipleShooting" begin
        alg = MultipleShooting(maxtime = 400,
            trajectories = 20,
            continuitylossweight = 100,
            initialization = DataInitialization())
        @testset "$s" for (i, s) in enumerate([
            "all states", "states + observed", "all observed"])
            r = calibrate(invprobs[i], alg)
            @test only(r)≈true_L rtol=1e-3
        end
    end
end

end # PEMTests
