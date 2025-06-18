# Design Optimization of a DC Motor

In this tutorial, we will perform _control and plant co-design_, that is, we will jointly optimize over both physical parameters of an engineering system and the parameters of the control system controlling it. The system we will consider is a DC motor with a PI controller. The goal of the control system is to meet a desired velocity reference signal, subject to load torque disturbances. This setup simulates a common scenario in robotics and machining, where load torques may arise from either the weight of the tool, or from contact with the environment. The perhaps simplest example to have in mind during this tutorial is that of a _motorized saw_ cutting through a piece of wood. When idling, the control system keeps the saw blade spinning at a constant speed. When the saw blade encounters the wood, the load torque increases and the control system must increase the torque to the saw blade in order to maintain the desired speed.

## Julia Environment

For this tutorial we will need the following packages:

| Module                                                                                         | Description                                                                                |
|:---------------------------------------------------------------------------------------------- |:------------------------------------------------------------------------------------------ |
| [DyadModelOptimizer](https://help.juliahub.com/jsmo/stable/)                               | The high-level library used to formulate our problem and perform automated model discovery |
| [ModelingToolkit](https://docs.sciml.ai/ModelingToolkit/stable/)                               | The symbolic modeling environment                                                          |
| [ModelingToolkitStandardLibrary](https://docs.sciml.ai/ModelingToolkitStandardLibrary/stable/) | Library for using standard modeling components                                             |
| [OrdinaryDiffEq](https://docs.sciml.ai/DiffEqDocs/stable/)                                     | The numerical differential equation solvers                                                |
| [Plots](https://docs.juliaplots.org/stable/)                                                   | The plotting and visualization library                                                     |

```@example dcmotor
using DyadModelOptimizer
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Mechanical.Rotational
using ModelingToolkitStandardLibrary.Blocks
using OrdinaryDiffEq
using Plots
gr(fmt = :png) # hide
using Test # hide
```

## Model Setup

The model is assembled out of components from [ModelingToolkitStandardLibrary](https://docs.sciml.ai/ModelingToolkitStandardLibrary/stable/) and consists of three primary parts:

  - A model of the electrical circuit of the DC motor
  - A model of the mechanical dynamics of the DC motor
  - A model of the control system and the input-output signals

The electrical circuit consists of a voltage source, resistive and inductive properties of the motor, as well as an electromotive force (EMF) that is proportional to the angular velocity of the motor. The EMF serves as the interface between the electrical domain and the mechanical domain. The mechanical dynamics consists of a fixed support to which the motor is mounted, an inertia, and a frictional damper. The control system consists of a reference signal, a PI controller, a sensor for the angular velocity, and the feedback connection.

```@example dcmotor
function dc_motor(R1 = 0.5)
    R = R1 # [Ohm] armature resistance
    L = 4.5e-3 # [H] armature inductance
    k = 0.5 # [N.m/A] motor constant
    J = 0.02 # [kg.m²] inertia
    f = 0.01 # [N.m.s/rad] friction factor
    tau_L_step = -0.3 # [N.m] amplitude of the load torque step

    @named ground = Ground()
    @named source = Voltage()
    @named ref = Blocks.Step(height = 0.2, start_time = 0.0)
    @named pi_controller = Blocks.LimPI(k = 1.1, T = 0.035, u_max = 10.0, Ta = 0.035)
    @named feedback = Blocks.Feedback()
    @named R1 = Resistor(R = R)
    @named L1 = Inductor(L = L)
    @named emf = EMF(k = k)
    @named fixed = Fixed()
    @named load = Torque()
    @named load_step = Blocks.Step(height = tau_L_step, start_time = 3.0)
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
            speed_sensor,
        ])
end

model = dc_motor()
sys = structural_simplify(model)
```

Although the model contains a relatively large number of components, the number of resulting equations after simplification is relatively small.

## Control objective

We will tune the parameters of the control system to meet a specific control objective. In addition to the control parameters, we are also free to choose the inertia of an inertial wheel that is mounted to the load side of the motor. The effect of this inertial wheel is to smooth out the effect of any load-torque disturbances, at the expense of making the response to changes in the velocity reference slower.

The tuning objectives are

  - The current through the motor winding (`I_max`) may not exceed 0.82A.
  - The integrated absolute error in response to a reference angular velocity step of size 0.2 rad/s is to be minimized.
  - The integrated absolute error in response to a load-torque step of size 0.3Nm that comes from encountering the wood is to be minimized.
  - The tunable motor-side inertia is constrained to be between 0.02 and 0.04 kg.m², where the lower bound corresponds to having no additional inertia, and the upper bound corresponds to having the largest physically realizable inertial wheel given the geometrical design of the physical system.

Since we have two competing minimization objectives, we will minimize the sum of these objectives in a single simulation. We construct the simulation such that we start with a reference step, and apply the load step after sufficiently long time has passed for the error in response to the reference step to have dissipated.

## Defining DesignConfiguration and InverseProblem

The next step is to define the [`DesignConfiguration`](@ref). The corresponding running cost to be integrated can be the absolute error between the motor velocity and the reference, but since that's a non-smooth function, it can be difficult to optimize. We can instead use a smoother version of it, the `Huber` loss function as our running cost as it is similar to the absolute value, but behaves like a differentiable quadratic function very close to 0. We also specify the constraint on the maximum current through the motor winding as a vector of symbolic constraints (in this case the vector has a single element only).

```@example dcmotor
I_max = 0.82
constraints = [sys.pi_controller.ctr_output.u ≲ sys.R1.R * I_max]
huber(x, a = 0.1) = ifelse(abs(x) < a, x^2 / 2a, (abs(x) - a / 2))
designconfig = DesignConfiguration(sys;
    overrides = [sys.L1.i => 0.0, sys.inertia.w => 0],
    alg = Rodas5P(),
    abstol = 1e-6,
    reltol = 1e-6,
    tspan = (0.0, 6.0),
    saveat = 0.0:0.01:6.0,
    constraints,
    running_cost = (x, sol, data) -> huber.(sol[sys.inertia.w] .- sol[sys.ref.output.u], 0.05),
    reduction = sum
)
```

The bounds on the tunable parameters, while also a form of constraints, are specified separately. This is because many solves can handle parameter bounds more efficiently than general constraints. The parameter bounds are specified as a vector of pairs, each pair containing a parameter and a tuple with its lower and upper bounds. The parameter bounds are then packaged into an [`InverseProblem`](@ref) together with the experiment definition and the system model.

```@example dcmotor
ss = [
    sys.inertia.J => (0.02, 0.04),
    sys.pi_controller.gainPI.k => (0.1, 10),
    sys.pi_controller.int.k => (1 / 0.35, 1 / 0.0035),
]
prob = InverseProblem(designconfig, ss)
```

## Design Optimization

We run the design optimization using [`calibrate`](@ref) function. We specify the [`SingleShooting`](@ref) algorithm to solve this optimization problem.

```@example dcmotor
alg = SingleShooting(maxiters = 1000)
r = calibrate(prob, alg)
```

## Visualization

When the problem is solved, we can call [`simulate`](@ref) to simulate the system with the optimized parameters. We can then plot the results to see how the system behaves.

```@example dcmotor
r_sol = simulate(designconfig, r)
V_max = r_sol.ps[sys.R1.R] * I_max
ϵ = 1e-6 # hide
@test r[:inertia₊J] ≈ 0.04 atol = 1e-8 # hide
@test all(<(V_max + ϵ), r_sol[sys.pi_controller.ctr_output.u]) # hide
plot(r_sol, idxs = [sys.inertia.w, sys.pi_controller.ctr_output.u], label = ["Motor velocity" "output of PI Controller"], layout = (2, 1), size = (1000, 600))
hline!([V_max], label = "Constraint", sp = 2, l = (:dash, :black))
```

We see that the constraint on the maximum current becomes active during the load step taking place at `t = 3`, however, the step in the velocity reference did not lead to as large currents through the motor. We also notice that the optimizer chose to maximize the inertia of the inertial wheel (`inertia.J = 0.04`). These two facts are linked, it was apparently harder for the control system to handle the step in the load torque, and therefore beneficial to increase the inertia of the inertial wheel in order to make this transient easier to handle.
