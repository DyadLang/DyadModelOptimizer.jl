using ModelingToolkit
using ModelingToolkit: D_nounits as D, t_nounits
using StableRNGs
using Statistics
using OrdinaryDiffEqDefault
using SymbolicIndexingInterface

function lotka()
    t = t_nounits
    @variables x(t)=3.1 y(t)=1.5
    @parameters α=1.3 β=0.9 γ=0.8 δ=1.8
    eqs = [
        D(x) ~ α * x - β * x * y,
        D(y) ~ -δ * y + γ * x * y
    ]
    return complete(ODESystem(eqs, t, name = :lotka))
end

function incomplete_lotka()
    t = t_nounits
    sts = @variables x(t)=5.0 y(t)=5.0
    ps = @parameters α=1.3 β=0.9 γ=0.8 δ=1.8
    eqs = [
        D(x) ~ α * x,
        D(y) ~ -δ * y
    ]
    return complete(ODESystem(eqs, t, name = :lotka))
end

function generate_data(model, tspan = (0.0, 1.0), n = 5;
        params = [],
        u0 = [],
        kwargs...)
    prob = ODEProblem(model, u0, tspan, params)
    saveat = range(prob.tspan..., length = n)
    sol = solve(prob; saveat, kwargs...)
    return DataFrame(sol)
end

function generate_noisy_data(model, tspan = (0.0, 1.0), n = 5;
        params = [],
        u0 = [],
        rng = StableRNG(1111),
        kwargs...)
    prob = ODEProblem(model, u0, tspan, params)
    prob = remake(prob, u0 = 5.0f0 * rand(rng, length(prob.u0)))
    saveat = range(prob.tspan..., length = n)
    sol = solve(prob; saveat, kwargs...)
    X = Array(sol)
    x̄ = mean(X, dims = 2)
    noise_magnitude = 5e-3
    Xₙ = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))
    return DataFrame(hcat(sol.t, transpose(Xₙ)),
        vcat(:timestamp, Symbol.(variable_symbols(sol))))
end
