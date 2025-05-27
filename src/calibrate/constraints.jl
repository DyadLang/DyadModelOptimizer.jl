function build_constraints(alg,
        prob,
        internal_params = internal_alg_params(alg, prob))
    internal_residual! = compute_residual(alg, prob)
    cstr = reduce(vcat, get_constraints.(get_experiments(prob)))

    isnothing(internal_residual!) && isempty(cstr) && return (nothing, nothing, ())

    if isempty(cstr)
        # we have an internal residual only
        residual = internal_residual!
        @assert !isnothing(internal_params)
        # assuming that the internal constraints are equality constraints
        lcons = fill(zero(eltype(internal_params)), length(internal_params))
        ucons = fill(zero(eltype(internal_params)), length(internal_params))
    elseif isnothing(internal_residual!)
        # we have symbolic constraints only
        sym_residual!, lcons, ucons = interpret_symbolic_constraints(cstr, alg, prob)

        residual = sym_residual!
    else
        # we need to merge both type of constraints
        sym_residual!, _lcons, _ucons = interpret_symbolic_constraints(cstr, alg, prob)

        residual = function (res, x, p)
            sym_residual!(view(res, axes(_lcons, 1)), x, p)
            internal_residual!(
                view(res,
                    (length(_lcons) + 1):(length(_lcons) + length(internal_params))),
                x,
                p)
        end
        # assuming that the internal constraints are equality constraints
        lcons = vcat(_lcons, fill(zero(eltype(_lcons)), length(internal_params)))
        ucons = vcat(_ucons, fill(zero(eltype(_ucons)), length(internal_params)))
    end

    return lcons, ucons, (cons = residual,)
end

compute_residual(::AbstractCalibrationAlgorithm, prob) = nothing

function cstr_canonical_form(eqs)
    ModelingToolkit.subs_constants([Symbolics.canonical_form(eq).lhs for eq in eqs])
end

function interpret_symbolic_constraints(cstr, alg, prob)
    experiments = get_experiments(prob)
    models = get_model.(experiments)
    model_variables = unique(reduce(vcat, all_variable_symbols.(models)))
    @debug "Variables from all models: $(join(string.(model_variables), ", ", " and "))"
    model_parameters = get_all_parameters(models, full = true)
    # we need to determine if the constraints are using only elements of the search
    # space (i.e. time independent constraints) or if variables or observed are also
    # involved
    # FIXME: get_variables seems to drop metadata for variables
    # and the results are no longer compatible with what all_variable_symbols
    # returns
    cstr_vars = unique(reduce(vcat, get_variables.(cstr)))
    dvs = constraint_var_names(alg, prob)
    @debug "Constraint vars: $(join(string.(cstr_vars), ", ", " and "))"

    continuous_cons = false
    parametrized_cons = false

    for v in cstr_vars
        if ispresent(v, dvs)
            # the variable is in the search space
            continue
        elseif ispresent(v, model_variables)
            continuous_cons = true
        elseif ispresent(v, model_parameters)
            parametrized_cons = true
        else
            error("Unable to formulate constraints using $v")
        end
    end

    canon_cstr = cstr_canonical_form(cstr)

    if continuous_cons
        all_ts = reduce(vcat, get_constraints_ts.(get_experiments(prob)))
        # @debug all_ts
        lcons = fill(-Inf, length(cstr))
        lcons[findall(Base.Fix2(isa, Equation), cstr)] .= 0.0
        lcons = repeat(lcons, length(all_ts))
        ucons = zeros(length(cstr) * length(all_ts))
        residual! = residual_formulation(alg, canon_cstr)
    else
        @named cons_sys = ConstraintsSystem(cstr, dvs, [])
        cons, lcons, ucons = generate_function(cons_sys, expression = Val{true})
        oop_residual! = cons[2] # we only need the oop version of the function

        # handle only parametrized constraint vars that are not in the search space
        for p in setdiff(intersect(cstr_vars, model_parameters), dvs)
            @debug "substituting $p"
            sub = substitute_p_in_expr(p)
            # figure out where is this params from
            ex = corresponding_experiment(p, prob)
            # (ˍ₋out, __mtk_arg_1, __mtk_arg_2) is (res, x, p)
            # p contains only the prob
            oop_residual! = sub(
                _ -> :(get_updated_param($p, $ex, first(__mtk_arg_2), __mtk_arg_1)),
                oop_residual!)
            # @debug oop_residual!
        end

        residual! = Symbolics._build_and_inject_function(@__MODULE__, oop_residual!)
    end

    residual!, lcons, ucons
end

function residual_formulation(::SingleShooting, canon_cstr)
    function symbolic_residual!(res, x, p)
        # @debug length(res)
        prob = first(p)
        experiments = get_experiments(prob)
        idx = 1
        for experiment in experiments
            n_cons = length(get_constraints(experiment))
            ts = get_constraints_ts(experiment)
            n = length(ts)
            # @debug "idx: $(idx:(n * n_cons * (idx))); n: $n * $(length(canon_cstr))"
            sol = simulate(experiment, prob, x, save_idxs = nothing)

            # TODO cache this as optimization problem parameter
            obs = SymbolicIndexingInterface.observed(sol, canon_cstr)
            view(res, idx:(n * n_cons * idx)) .= mapreduce(
                t -> obs(sol(t),
                    parameter_values(sol),
                    t),
                vcat,
                ts)
            idx += n_cons * n
        end
        # @debug res
    end
end

function residual_formulation(a::AbstractCalibrationAlgorithm, canon_cstr)
    error("$(nameof(typeof(a))) is not compatible with time dependent constraints.")
end

function constraint_var_names(alg, prob)
    ist = get_internal_storage(prob)
    ss_names = collect(search_space_names(ist))
    # findfirst doesn't work on ComponentArrays because it changes what keys means
    # so ispresent would fail silently if used with this
    @debug "Search space names: $(join(string.(ss_names), ", ", " and "))"

    ss_names
end

function substitute_p_in_expr(v)
    Substitute() do expr
        # @debug "$expr::$(typeof(expr)) == $(getname(v))): $(expr == getname(v))"
        expr isa Symbol && expr == :($(getname(v))) && return true
        return false
    end
end

function corresponding_experiment(p, prob)
    experiments = get_experiments(prob)
    ex = nothing
    idx = nothing

    for (i, e) in enumerate(experiments)
        pb = get_prob(e)
        # dependent parameters don't have an index in MTK@9.33+
        # from the SII point of view they are observed
        is_p = is_parameter(pb, p) || is_observed(pb, p)
        if !isnothing(is_p)
            # found the parameter
            if isnothing(idx)
                idx = i
                ex = e
                @debug "Parameter $p found in experiment $(find_index(ex, prob))"
            else
                # ambiguity
                error("Parameter $p was found in both experiment $idx " *
                      "and experiment $(find_index(e, prob)).\nIt is not clear which value to pick in the constraints.")
            end
        end
    end
    if isnothing(idx)
        error("Parameter $p not found!")
    end

    return ex
end
