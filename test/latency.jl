using Test
using SnoopCompile, SnoopCompileCore

using DyadModelOptimizer

using OrdinaryDiffEqTsit5
using ModelingToolkit
using DataFrames

include("lotka_volterra.jl")

model = lotka()
@unpack α, β = model

overrides = [β => 3.0]

data1 = generate_data(model; params = overrides)

tinf_e = @snoop_inference Experiment(data1, model; overrides, tspan = (0.0, 1.0))
@show tinf_e

# inference time for Experiment should be under 10s
@test inclusive(tinf_e)<10 broken=false

experiment = Experiment(data1, model; overrides, tspan = (0.0, 1.0))
ss = [α => (0.0, 5.0), β => (1.5, 3.0)]

tinf_p = @snoop_inference InverseProblem(experiment, ss)
itrigs = inference_triggers(tinf_p)

prob = InverseProblem(experiment, ss)

using Plots

tinf_plt = @snoop_inference plot(experiment, prob)
@show tinf_plt

# inference time for plot should be under 10s
@test inclusive(tinf_plt) < 20

alg = SingleShooting(maxiters = 1)
# InferenceTimingNode: 5.622281/11.103190 on Core.Compiler.Timings.ROOT() with 93 direct children
tinf_c = @snoop_inference calibrate(prob, alg)
@show tinf_c

@test inclusive(tinf_plt) < 30
