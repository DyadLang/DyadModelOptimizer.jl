using Test
using DyadModelOptimizer
using OrdinaryDiffEqTsit5
using ModelingToolkit
using ModelingToolkitStandardLibrary.Thermal
using ModelingToolkit: t_nounits as t
using Statistics
using OptimizationBBO
using DataFrames

C1 = 15
C2 = 15
@named mass1 = HeatCapacitor(C = C1, T = 373.15)
@named mass2 = HeatCapacitor(C = C2, T = 273.15)
@named conduction = ThermalConductor(G = 10)
@named Tsensor1 = TemperatureSensor()
@named Tsensor2 = TemperatureSensor()

connections = [
    connect(mass1.port, conduction.port_a),
    connect(conduction.port_b, mass2.port),
    connect(mass1.port, Tsensor1.port),
    connect(mass2.port, Tsensor2.port)
]

@named model = ODESystem(connections,
    t,
    systems = [mass1, mass2, conduction, Tsensor1, Tsensor2])
sys = structural_simplify(model)

prob = ODEProblem(sys, Pair[], (0, 5.0))
sol = solve(prob, Tsit5())

data = DataFrame(sol)

experiment = Experiment(data, sys, tspan = (0.0, 5.0))

invprob = InverseProblem([experiment], [mass1.C => (10, 20), mass2.C => (10, 20)])

ps = parametric_uq(invprob,
    StochGlobalOpt(method = SingleShooting(maxiters = 10,
        optimizer = BBO_adaptive_de_rand_1_bin_radiuslimited())),
    sample_size = 50)

params = Tables.columntable(ps)
@test isapprox(mean(Tables.getcolumn(params, :mass1₊C)), C1, rtol = 1e-1)
@test isapprox(mean(Tables.getcolumn(params, :mass2₊C)), C2, rtol = 1e-1)
