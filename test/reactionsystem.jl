using ModelingToolkit
using ModelingToolkit: defaults, D_nounits, t_nounits
using OrdinaryDiffEqTsit5
using DataFrames

getparams(model) = getindex.((defaults(model),), parameters(model))

function reactionsystem()
    t = t_nounits
    sts = @variables s1(t)=2.0 s1s2(t)=2.0 s2(t)=2.0
    ps = @parameters k1=1.0 c1=2.0 [bounds = (0, 2), tunable = true]
    eqs = [D_nounits(s1) ~ -0.25 * c1 * k1 * s1 * s2
           D_nounits(s1s2) ~ 0.25 * c1 * k1 * s1 * s2
           D_nounits(s2) ~ -0.25 * c1 * k1 * s1 * s2]

    return complete(ODESystem(eqs, t_nounits; name = :reactionsystem))
end

function reactionsystem_obs()
    t = t_nounits
    sts = @variables s1(t)=2.0 s1s2(t)=2.0 s2(t)=2.0 s3(t)
    ps = @parameters k1=1.0 c1=2.0 [bounds = (0, 2), tunable = true] Δt=2.5
    eqs = [D_nounits(s1) ~ -0.25 * c1 * k1 * s1 * s2
           D_nounits(s1s2) ~ 0.25 * c1 * k1 * s1 * s2
           D_nounits(s2) ~ -0.25 * c1 * k1 * s1 * s2
           s3 ~ s1 + s2]
    return structural_simplify(ODESystem(eqs, t_nounits,
        sts,
        ps;
        name = :reactionsystem))
end

function extended_reactionsystem()
    t = t_nounits
    sts = @variables s1(t)=2.0 s1s2(t)=2.0 s2(t)=2.0 o1(t) [guess = 4.0]
    ps = @parameters k1=1.0 c1=2.0 [bounds = (0, 2), tunable = true] Δt=2.5 inf_s1=0.0 inf_s1s2=0.0 inf_s2=0.0
    eqs = [D_nounits(s1) ~ -0.25 * c1 * k1 * s1 * s2 + inf_s1
           D_nounits(s1s2) ~ 0.25 * c1 * k1 * s1 * s2 + inf_s1s2
           D_nounits(s2) ~ -0.15 * c1 * k1 * s1 * s2 + inf_s2
           o1 ~ s1 * s2]

    @named model = ODESystem(eqs, t, sts, ps)

    return structural_simplify(model)
end

function reactionsystem_obs_local_alias()
    t = t_nounits
    ps = @parameters k1=1.0 c1=2.0 c1_cond1=2.0 c1_cond2=2
    sts = @variables s1(t)=2.0 s1s2(t)=2.0 s2(t)=2c1 s3(t)
    eqs = [D_nounits(s1) ~ -0.25 * c1 * k1 * s1 * s2
           D_nounits(s1s2) ~ 0.25 * c1 * k1 * s1 * s2
           D_nounits(s2) ~ -0.25 * c1 * k1 * s1 * s2
           s3 ~ s1 + s2]
    return structural_simplify(ODESystem(eqs,
        t,
        sts,
        ps;
        name = :reactionsystem))
end

function reactionsystem_non_independent_params()
    t = t_nounits
    sts = @variables s1(t)=2.0 s1s2(t)=2.0 s2(t)=2.0
    ps = @parameters k1=1.0 c1=2.0 k2=2 * k1 [bounds = (0, 2), tunable = true] Δt=2.5
    eqs = [D_nounits(s1) ~ -0.25 * c1 * k1 * s1 * s2
           D_nounits(s1s2) ~ 0.25 * c1 * k1 * s1 * s2
           D_nounits(s2) ~ -0.25 * c1 * k2 * s1 * s2]

    return complete(ODESystem(eqs,
        t,
        sts,
        ps;
        name = :reactionsystem))
end

function reactionsystem_obs_non_independent_params()
    t = t_nounits
    sts = @variables s1(t)=2.0 s1s2(t)=2.0 s2(t)=2.0 s3(t)
    ps = @parameters k1=1.0 c1=2.0 k2=2 * k1 [bounds = (0, 2), tunable = true] Δt=2.5
    eqs = [D_nounits(s1) ~ -0.25 * c1 * k1 * s1 * s2
           D_nounits(s1s2) ~ 0.25 * c1 * k1 * s1 * s2
           D_nounits(s2) ~ -0.25 * c1 * k2 * s1 * s2
           s3 ~ s1 + s2]
    return structural_simplify(ODESystem(eqs,
        t,
        sts,
        ps;
        name = :reactionsystem))
end

function generate_data(model, tspan = (0.0, 1.0), n = 5;
        params = [],
        u0 = [],
        kwargs...)
    prob = ODEProblem(model, u0, tspan, params)
    saveat = range(prob.tspan..., length = n)
    sol = solve(prob, Tsit5(); saveat, kwargs...)

    return DataFrame(sol)
end

function generate_noisy_data(model, tspan = (0.0, 1.0), n = 5;
        params = getparams(model),
        u0 = [],
        noise_std = 0.5)
    prob = ODEProblem(model, u0, tspan, params)
    saveat = range(prob.tspan..., length = n)
    sol = solve(prob, Tsit5(); saveat)

    df = DataFrame(sol)

    # draw an independent noise sample for each (saveat, state)
    rd = randn(size(df[:, Not("timestamp")])) .* noise_std
    df[:, Not("timestamp")] .+= rd
    return df
end

function generate_observed_data(model, tspan = (0.0, 1.0), n = 5;
        params = [],
        u0 = [],
        kwargs...)
    prob = ODEProblem(model, u0, tspan, params)
    saveat = range(prob.tspan..., length = n)
    sol = solve(prob, Tsit5(); saveat, kwargs...)
    @unpack s3 = model
    return hcat(DataFrame(sol), DataFrame("s3(t)" => sol[s3]))
end
