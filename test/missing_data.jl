using DyadModelOptimizer
using DataFrames

include("reactionsystem.jl")

data = DataFrame(:timestamp => [1, 2, 3, 4], :s1 => [missing, 1, 2, 3],
    :s2 => [missing, missing, 4, 5])

model = reactionsystem()
experiment = Experiment(data, model)

invprob = InverseProblem(experiment, [])
cost = objective(invprob, SingleShooting(maxiters = 1))

@test !iszero(cost())
