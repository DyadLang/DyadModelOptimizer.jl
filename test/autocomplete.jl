using DyadModelOptimizer
using SciMLSensitivity
using OrdinaryDiffEq
using ModelingToolkit
using Optimization
using OptimizationOptimisers: Adam
using OptimizationOptimJL: LBFGS
using LineSearches
using LineSearches: BackTracking
using DataFrames
using Test
using JET

# Derived from https://docs.sciml.ai/Overview/stable/showcase/missing_physics/ in https://github.com/SciML/SciMLDocs, MIT licensed, see repository for details
include("lotka_volterra.jl")

model = lotka()
rng = StableRNG(1111)
data = generate_noisy_data(model,
    (0.0, 5),
    21;
    alg = Vern7(),
    abstol = 1e-12,
    reltol = 1e-12,
    rng)

@testset "Train the neural network" begin
    experiment = Experiment(data, model, alg = Vern7(), abstol = 1e-6,
        reltol = 1e-3)
    prob = InverseProblem(experiment,
        [],
        neural_network = multi_layer_feed_forward(2, 2))
    alg = SingleShooting(maxiters = 10, optimizer = Adam())
    cost = objective(prob, alg)
    @test cost() ≠ 0 # we have and added term from the nn
    r = calibrate(prob, alg)
end

incomplete_model = complete(incomplete_lotka())
@unpack x, y = incomplete_model
experiment = Experiment(
    data, incomplete_model, alg = Vern7(), abstol = 1e-6, reltol = 1e-6,
    u0 = [x => data."x"[1], y => data."y"[1]])

prob = InverseProblem(experiment, [],
    neural_network = multi_layer_feed_forward(2, 2), nn_rng = rng)
r1 = calibrate(prob, SingleShooting(maxiters = 5000, optimizer = Adam()))
r2 = calibrate(r1,
    SingleShooting(maxiters = 1000, optimizer = LBFGS(linesearch = BackTracking())))

cost = objective(prob, SingleShooting(maxiters = 1))

@test cost(r2) < cost(r1)
@test_call cost(r1)

alg = MultipleShooting(trajectories = 10, initialization = DataInitialization(),
    maxiters = 1000, optimizer = LBFGS(linesearch = BackTracking()))

# test the combination of MultipleShooting and neural networks
# both need internal parameters, but they have to be considered
# separately
r1_ms = calibrate(prob, alg)
@test cost(r1_ms) < cost(r1)

# test that we can reinitialize the the initial state in calibration
# to add additional internal parameters that are not present in the
# initial alg that was used in for the calibration result
r2_ms = calibrate(r1, alg)
@test cost(r2_ms) < cost(r1)

using DataDrivenDiffEq
using DataDrivenSparse
using StableRNGs

X̂ = simulate(experiment, prob, r2)
β = ModelingToolkit.defaults(incomplete_model)[incomplete_model.β]
γ = ModelingToolkit.defaults(incomplete_model)[incomplete_model.γ]
# Ideal unknown interactions of the predictor
Ȳ = [-β * (X̂[1, :] .* X̂[2, :])'; γ * (X̂[1, :] .* X̂[2, :])']
# Neural network guess
Ŷ = network_prediction(prob)(X̂, r2)

nn_problem = DirectDataDrivenProblem(reduce(hcat, X̂.u), Ȳ)

@variables u[1:2]
b = polynomial_basis(u, 4)
basis = Basis(b, u);

λ = exp10.(-3:0.01:3)
opt = ADMM(λ)

options = DataDrivenCommonOptions(maxiters = 10_000,
    normalize = DataNormalization(ZScoreTransform),
    selector = bic, digits = 1,
    data_processing = DataProcessing(split = 0.9,
        batchsize = 30,
        shuffle = true,
        rng = StableRNG(1111)))

nn_res = solve(nn_problem, basis, opt, options = options)
nn_eqs = get_basis(nn_res)
println(nn_res)

equations(nn_eqs)
