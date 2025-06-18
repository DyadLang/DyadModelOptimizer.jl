using Test
using ModelingToolkit, OrdinaryDiffEqTsit5
using ModelingToolkit: t_nounits as t, D_nounits as D
using DyadModelOptimizer
using DataFrames

function experiment1_sys()
    @parameters p1=0.5 [tunable = true] (p23[1:2]=[1, 3.0]) [tunable = true] p4=3 * p1 [tunable = false] y0=1.2 [tunable = true]
    @variables x(t)=2p1 y(t)=y0 z(t)=x + y

    eqs = [D(x) ~ p1 * x - p23[1] * x * y
           D(y) ~ -p23[2] * y + p4 * x * y
           z ~ x + y]

    structural_simplify(ODESystem(eqs, t, tspan = (0, 3.0), name = :sys))
end

function experiment2_sys()
    @parameters p1=1 [tunable = false] (p23[1:2]=[1, 3.0]) [tunable = true] p4=3 * p1
    @variables x(t)=2 * p1 y(t)=0.2 z(t)

    eqs = [D(x) ~ p1 * x - p23[1] * x * y
           D(y) ~ -p23[2] * y + p4 * x * y
           z ~ x + y]

    structural_simplify(ODESystem(eqs, t, tspan = (0, 3.0), name = :sys))
end

sys1 = experiment1_sys()
prob1 = ODEProblem(sys1, [sys1.p23 => [2, 4.0]])
sol1 = solve(prob1, abstol = 1e-8, reltol = 1e-8)
data1 = DataFrame(sol1)

sys2 = experiment2_sys()
prob2 = ODEProblem(sys2, [sys2.p1 => 2.0])
sol2 = solve(prob2, abstol = 1e-8, reltol = 1e-8)
data2 = DataFrame(sol2)

experiment1 = Experiment(data1, sys1)
experiment2 = Experiment(data2, sys2)

invprob = InverseProblem([experiment1, experiment2], [sys1.p1 => (1, 3.0)])
initial_guess(Any, invprob)
cost_contribution(SingleShooting(maxiters = 1), experiment2, invprob, [2.0])

invprob = InverseProblem(
    experiment1, [sys1.p1 => (1.1, 2), sys1.p23 => ([1, 1.3], [3.0, 4.5])])

@test lowerbound(invprob) == [1.1, 1, 1.3]

@test length(initial_guess(Any, invprob)) == 3

s = simulate(experiment1, invprob)
@test SciMLBase.successful_retcode(s)

r = calibrate(invprob, SingleShooting(maxiters = 1000))
@test SciMLBase.successful_retcode(r.retcode)

@testset "Element of a vector in search space" begin
    invprob = InverseProblem(
        experiment1, [sys1.p23[1] => (1.1, 3), sys1.p23[2] => (2.7, 4.5)])

    r = calibrate(invprob, SingleShooting(maxiters = 1000))
    @test r≈[2, 4] rtol=1e-4
    @test SciMLBase.successful_retcode(r.retcode)
end

############

using Test
using JET
using ModelingToolkitNeuralNets
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using ModelingToolkitStandardLibrary.Blocks
using OrdinaryDiffEqVerner
using SymbolicIndexingInterface
using Optimization
using OptimizationOptimisers: Adam
using SciMLStructures
using SciMLStructures: Tunable
using ForwardDiff
using StableRNGs
using DataFrames
using DyadModelOptimizer

@testset "ModelingToolkitNeuralNets" begin
    function lotka_ude()
        @variables x(t)=3.1 y(t)=1.5
        @parameters α=1.3 [tunable = false] δ=1.8 [tunable = false]
        Dt = ModelingToolkit.D_nounits
        @named nn_in = RealInputArray(nin = 2)
        @named nn_out = RealOutputArray(nout = 2)

        eqs = [
            Dt(x) ~ α * x + nn_in.u[1],
            Dt(y) ~ -δ * y + nn_in.u[2],
            nn_out.u[1] ~ x,
            nn_out.u[2] ~ y
        ]
        return ODESystem(
            eqs, ModelingToolkit.t_nounits, name = :lotka, systems = [nn_in, nn_out])
    end

    function lotka_true()
        @variables x(t)=3.1 y(t)=1.5
        @parameters α=1.3 β=0.9 γ=0.8 δ=1.8
        Dt = ModelingToolkit.D_nounits

        eqs = [
            Dt(x) ~ α * x - β * x * y,
            Dt(y) ~ -δ * y + δ * x * y
        ]
        return ODESystem(eqs, ModelingToolkit.t_nounits, name = :lotka_true)
    end

    model = lotka_ude()

    chain = ModelingToolkitNeuralNets.multi_layer_feed_forward(2, 2)
    @named nn = NeuralNetworkBlock(2, 2; chain, rng = StableRNG(42))

    eqs = [connect(model.nn_in, nn.output)
           connect(model.nn_out, nn.input)]

    ude_sys = ODESystem(eqs, t, systems = [model, nn], name = :ude_sys)
    sys = structural_simplify(ude_sys)

    prob = ODEProblem{true, SciMLBase.FullSpecialize}(sys, [], (0, 1.0), [])

    model_true = structural_simplify(lotka_true())
    prob_true = ODEProblem{true, SciMLBase.FullSpecialize}(model_true, [], (0, 1.0), [])
    sol_ref = solve(prob_true, Vern9(), reltol = 1e-8, abstol = 1e-8)

    data = DataFrame(sol_ref)

    experiment = Experiment(data, sys;
        alg = Vern9(), reltol = 1e-8, abstol = 1e-8,
        depvars = [sys.lotka.x => "x(t)", sys.lotka.y => "y(t)"])
    invprob = InverseProblem(experiment, [sys.nn.p => (-Inf, Inf)])

    s = simulate(experiment, invprob)
    @test SciMLBase.successful_retcode(s)

    cost_contribution(SingleShooting(maxiters = 1), experiment, invprob)
    r = calibrate(invprob,
        SingleShooting(
            maxiters = 25000, optimizer = Adam(1e-3)))

    @test r.original.objective < 1e-4
end
