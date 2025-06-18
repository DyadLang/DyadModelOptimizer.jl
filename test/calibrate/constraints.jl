module ConstraintTests

using Test
using DyadModelOptimizer

using DataFrames
using ModelingToolkit
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Mechanical.Rotational
using ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit: t_nounits as t
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using Plots
using OptimizationOptimJL
using OptimizationMOI
using Ipopt
using SymbolicIndexingInterface

include("../reactionsystem.jl")

function circuit_model()
    # V = 10.0
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

function dc_motor(R1 = 0.5)
    R = R1 # [Ohm] armature resistance
    L = 4.5e-3 # [H] armature inductance
    k = 0.5 # [N.m/A] motor constant
    J = 0.02 # [kg.m²] inertia
    f = 0.01 # [N.m.s/rad] friction factor
    tau_L_step = -0.3 # [N.m] amplitude of the load torque step

    @named ground = Ground()
    @named source = Voltage()
    @named ref = Blocks.Step(height = 0.2, start_time = 0)
    @named pi_controller = Blocks.LimPI(k = 1.1, T = 0.035, u_max = 10, Ta = 0.035)
    @named feedback = Blocks.Feedback()
    @named R1 = Resistor(R = R)
    @named L1 = Inductor(L = L)
    @named emf = EMF(k = k)
    @named fixed = Fixed()
    @named load = Torque()
    @named load_step = Blocks.Step(height = tau_L_step, start_time = 3)
    @named inertia = Inertia(J = J)
    @named friction = Damper(d = f)
    @named speed_sensor = SpeedSensor()

    connections = [connect(fixed.flange, emf.support, friction.flange_b)
                   connect(emf.flange, friction.flange_a, inertia.flange_a)
                   connect(inertia.flange_b, load.flange)
                   connect(inertia.flange_b, speed_sensor.flange)
                   connect(load_step.output, load.tau)
                   connect(ref.output, feedback.input1)
                   connect(speed_sensor.w, :y, feedback.input2)
                   connect(feedback.output, pi_controller.err_input)
                   connect(pi_controller.ctr_output, :u, source.V)
                   connect(source.p, R1.p)
                   connect(R1.n, L1.p)
                   connect(L1.n, emf.p)
                   connect(emf.n, source.n, ground.g)]

    @named model = ODESystem(connections, t,
        systems = [
            ground,
            ref,
            pi_controller,
            feedback,
            source,
            R1,
            L1,
            emf,
            fixed,
            load,
            load_step,
            inertia,
            friction,
            speed_sensor
        ])
end

maxiters = 10^3

@testset "parametrized constraints" begin
    model = complete(reactionsystem())
    @unpack c1, k1 = model

    @testset "trivial case" begin
        data = generate_data(model, params = [k1 => 1.5, c1 => 3.0])

        experiment = Experiment(data, model, tspan = (0.0, 1.0), constraints = [k1 ~ 1.5])
        invprob = InverseProblem(experiment, [c1 => (0.0, 5), k1 => (1.0, 2)])
        alg = SingleShooting(; maxiters)

        r = calibrate(invprob, alg)

        @test r≈[3.0, 1.5] rtol=1e-5
    end

    @testset "simple parametrized constraint" begin
        data = generate_data(model, params = [k1 => 1.4, c1 => 3.0])

        experiment = Experiment(data, model, tspan = (0.0, 1.0),
            overrides = [k1 => 1.4], constraints = [k1 ≲ c1 / 2])
        invprob = InverseProblem(experiment, [c1 => (0, 5)])
        alg = SingleShooting(; maxiters)

        r = calibrate(invprob, alg)

        @test r≈[3] rtol=1e-5
    end

    @testset "insatisfiable case" begin
        data = generate_data(model, params = [k1 => 1.5, c1 => 3.0])

        experiment = Experiment(data,
            model,
            tspan = (0.0, 1.0),
            overrides = [c1 => 3.0],
            constraints = [c1 ~ 1.5])
        invprob = InverseProblem(experiment, [k1 => (1.0, 2)])
        alg = SingleShooting(; maxiters = 10)

        r = calibrate(invprob, alg)

        @test r≈[1.5] rtol=1e-5
        @test r.retcode == ReturnCode.Infeasible
    end

    @testset "ConstraintBased MultipleShooting" begin
        data = generate_data(model, params = [c1 => 3.0])

        experiment = Experiment(data, model, tspan = (0.0, 1.0))
        invprob = InverseProblem(experiment, [c1 => (0, 5)])

        alg = MultipleShooting(; maxiters, trajectories = 5,
            continuitytype = ConstraintBased)

        r = calibrate(invprob, alg)
        @test r≈[3] rtol=1e-4
    end

    @testset "ConstraintBased MultipleShooting & experiment constraints" begin
        data = generate_data(model, params = [c1 => 3.0, k1 => 1.4])

        experiment = Experiment(data, model, tspan = (0.0, 1.0),
            overrides = [k1 => 1.4], constraints = [k1 ≲ c1 / 2])
        invprob = InverseProblem(experiment, [c1 => (0, 5)])

        alg = MultipleShooting(; maxiters, trajectories = 5,
            continuitytype = ConstraintBased)

        r = calibrate(invprob, alg)
        @test r≈[3] rtol=1e-4
    end
end

huber(x, a = 0.1) = ifelse(abs(x) < a, x^2 / 2a, (abs(x) - a / 2))

@testset "SingleShooting constraint on observed" begin
    model = dc_motor()
    sys = structural_simplify(model)
    V_max = 0.41
    constraints = [sys.pi_controller.ctr_output.u ≲ V_max]

    experiment = Experiment(nothing,
        sys;
        alg = Rodas5P(),
        overrides = [sys.L1.i => 0.0, sys.inertia.w => 0],
        abstol = 1e-6,
        reltol = 1e-6,
        tspan = (0, 6.0),
        saveat = 0:0.01:6,
        constraints,
        loss_func = (x, sol, data) -> sum(huber.(
            sol[sys.inertia.w] .-
            sol[sys.ref.output.u], 0.05)))
    ss = [
        sys.inertia.J => (0.02, 0.04),
        sys.pi_controller.gainPI.k => (0.1, 10),
        sys.pi_controller.int.k => (1 / 0.35, 1 / 0.0035)
    ]
    prob = InverseProblem(experiment, ss)

    maxiters = 1000
    alg = SingleShooting(; maxiters)
    r = calibrate(prob, alg)

    r_sol = simulate(experiment, r)

    ϵ = 1e-6
    @test all(<(V_max + ϵ), r_sol[sys.pi_controller.ctr_output.u])
end

@testset "Design optimization" begin
    model = circuit_model()
    sys = structural_simplify(model)
    t0 = 0.0
    tend = 3.0

    @unpack ampermeter, capacitor1, capacitor2 = model

    dc = DesignConfiguration(sys;
        overrides = [capacitor2.v => 0.0],
        tspan = (t0, tend),
        alg = Rodas5P(),
        abstol = 1e-8,
        reltol = 1e-8,
        constraints_ts = range(t0, tend, length = 100),
        loss_func = (x, sol, data) -> 0,
        constraints = [sys.ampermeter.i ≲ 1e-6])

    ss = [capacitor2.C => (1.0, 100.0)]

    invprob = InverseProblem(dc, ss)
    optimizer = IpoptOptimizer(constr_viol_tol = 1e-5)
    maxiters = 1000
    alg = SingleShooting(; maxiters, optimizer)
    r = calibrate(invprob, alg)

    # C2 = C1 * R1/R2
    @test only(r)≈6 rtol=1e-4

    # TODO: add MultipleShooting implementation such that what follows will work
    # alg = MultipleShooting(; maxtime = 100, trajectories = 3,
    #     continuitylossweight = 1, optimizer, continuitytype = ConstraintBased)

end
@testset "Design optimization with fixed params" begin
    model = dc_motor()
    sys = structural_simplify(model)
    I_max = 0.9
    constraints = [sys.pi_controller.ctr_output.u ≲ sys.R1.R * I_max]
    experiment = DesignConfiguration(sys;
        overrides = [sys.L1.i => 0.0, sys.inertia.w => 0, sys.R1.R => 0.4],
        alg = Rodas5P(),
        abstol = 1e-8,
        reltol = 1e-8,
        tspan = (0, 6.0),
        saveat = 0:0.01:6,
        constraints,
        loss_func = (x, sol, data) -> sum(huber.(
            sol[sys.inertia.w] .-
            sol[sys.ref.output.u], 0.05)))
    ss = [
        sys.inertia.J => (0.02, 0.04),
        sys.pi_controller.gainPI.k => (0.1, 10),
        sys.pi_controller.int.k => (1 / 0.35, 1 / 0.0035)
    ]
    prob = InverseProblem(experiment, ss)

    maxiters = 1000
    alg = SingleShooting(; maxiters)
    r = calibrate(prob, alg)

    r_sol = simulate(experiment, r)

    ϵ = 1e-6
    @test getp(r_sol, sys.R1.R)(r_sol) == 0.4
    V_max = I_max * getp(r_sol, sys.R1.R)(r_sol)
    @test all(<(V_max + ϵ), r_sol[sys.pi_controller.ctr_output.u])
end

end # ConstraintTests
