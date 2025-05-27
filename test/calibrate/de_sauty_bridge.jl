using ModelingToolkit
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks: Sine
using DyadModelOptimizer
using DyadModelOptimizer: initial_state, internal_params
using OptimizationBBO
using DataFrames
using Test
using DyadInterface
using DyadData

using ModelingToolkit: t_nounits as t

function create_model(; C₁ = 2.4, C₂ = 60.0, f = 1.0)
    # V = 10.0
    @named resistor1 = Resistor(R = 5.0)
    @named resistor2 = Resistor(R = 2.0)
    @named capacitor1 = Capacitor(C = C₁)
    @named capacitor2 = Capacitor(C = C₂)
    @named source = Voltage()
    @named input_signal = Sine(frequency = f)
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

@testset "Design optimization with 1 experiment" begin
    model = create_model()
    sys = structural_simplify(model)
    t0 = 0.0
    tend = 3.0

    @unpack ampermeter, capacitor2, capacitor1, resistor2 = model

    experiment = DesignConfiguration(sys;
        tspan = (t0, tend),
        overrides = [capacitor2.v => 0.0],
        alg = Rodas5P(),
        abstol = 1e-8,
        reltol = 1e-8,
        running_cost = abs2(ampermeter.i),
        reduction = sum)

    ss = [capacitor2.C => (1.0, 100.0)]

    invprob = InverseProblem(experiment, ss)

    @testset "SingleShooting" begin
        alg = SingleShooting(maxiters = 10^3,
            optimizer = IpoptOptimizer(
            # verbose = true,
                hessian_approximation = "exact")
        )
        r = calibrate(invprob, alg)

        # C2 = C1 * R1/R2
        @test only(r)≈6 rtol=1e-4
    end

    @testset "MultipleShooting" begin
        alg = MultipleShooting(maxtime = 400,
            trajectories = 11,
            optimizer = IpoptOptimizer(
            # verbose = true,
                hessian_approximation = "exact"),
            continuitytype = ConstraintBased,
            initialization = DefaultSimulationInitialization())
        r = calibrate(invprob, alg)

        # C2 = C1 * R1/R2
        @test only(r)≈6 rtol=1e-3

        # test only cost differences due to system parameters
        # alg = MultipleShooting(maxiters = 1, trajectories = 11,
        #     continuitylossweight = 0)

        # initial_cost = cost_contribution(alg, experiment, invprob,
        #     initial_state(alg, invprob))
        # correct_cost = cost_contribution(alg, experiment, invprob,
        #     vcat([6.0], internal_params(alg, invprob)))
        # @test correct_cost < initial_cost # broken=true
    end
end

@testset "Data for states that are simplified away" begin
    model = create_model()
    sys = structural_simplify(model)
    t0 = 0.0
    tend = 3.0
    @unpack ampermeter, capacitor1, capacitor2 = model

    prob = ODEProblem(
        sys, [capacitor2.v => 0.0], (t0, tend), [capacitor2.C => 6.0])
    sol = solve(prob, Rodas5P(), abstol = 1e-6, reltol = 1e-6)
    data = DataFrame("timestamp" => sol.t,
        "ampermeter.i" => sol[ampermeter.i],
        "capacitor1.i" => sol[capacitor1.i])

    experiment = Experiment(data, sys;
        overrides = [capacitor2.v => 0.0],
        tspan = (t0, tend),
        abstol = 1e-6,
        reltol = 1e-6,
        alg = Rodas5P())
    ss = [capacitor2.C => (1.0, 100.0)]

    invprob = InverseProblem(experiment, ss)

    @testset "SingleShooting" begin
        alg = SingleShooting(maxiters = 10^5,
            optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited())
        r = calibrate(invprob, alg)

        # C2 = C1 * R1/R2
        @test only(r) ≈ 6
    end

    @testset "MultipleShooting" begin
        alg = MultipleShooting(maxtime = 400,
            trajectories = 19,
            continuitylossweight = 100,
            initialization = DataInitialization())
        r = calibrate(invprob, alg)

        # C2 = C1 * R1/R2
        @test only(r)≈6 rtol=5e-3
    end
end
