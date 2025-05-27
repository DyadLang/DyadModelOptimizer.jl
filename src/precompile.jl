using ModelingToolkit: @variables, @parameters, @unpack, D_nounits, t_nounits

@setup_workload begin
    function __reactionsystem()
        sts = @variables s1(t_nounits)=2.0 s1s2(t_nounits)=2.0 s2(t_nounits)=2.0
        ps = @parameters k1=1.0 c1=2.0
        eqs = [D_nounits(s1) ~ -0.25 * c1 * k1 * s1 * s2
               D_nounits(s1s2) ~ 0.25 * c1 * k1 * s1 * s2
               D_nounits(s2) ~ -0.25 * c1 * k1 * s1 * s2]

        return ModelingToolkit.structural_simplify(ODESystem(
            eqs, t_nounits; name = :reactionsystem))
    end

    function __generate_data(model, tspan = (0.0, 1.0), n = 5;
            params = [],
            u0 = [],
            kwargs...)
        prob = ODEProblem(model, u0, tspan, params)
        saveat = range(prob.tspan..., length = n)
        sol = solve(prob; saveat, kwargs...)

        return DataFrame(sol)
    end

    const _model = __reactionsystem()

    @compile_workload begin
        _data = __generate_data(_model; params = [_model.c1 => 3.0])
        ex = Experiment(_data, _model)

        invprob = InverseProblem(ex,
            [_model.c1 => (1.5, 3.0), _model.k1 => (0.0, 5.0)])

        alg1 = SingleShooting(maxiters = 1)
        alg2 = MultipleShooting(maxiters = 1, trajectories = 2)
        OptimizationProblem(invprob, alg1)
        OptimizationProblem(invprob, alg2)
    end
end
