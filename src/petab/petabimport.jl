struct PetabProblem{P <: DataFrame, C <: Vector{DataFrame}, M <: Vector{DataFrame},
    O <: Vector{DataFrame}, S <: Vector,
    V <: Vector{DataFrame}, D <: Vector{DataFrame}}
    parameter_df::P
    condition_dfs::C  # For now we just support length 1 vectors, i.e. one model that is optimized
    measurement_dfs::M
    observable_dfs::O
    models::S
    visualization_dfs::V
    dosing_dfs::D
end

const sbml_promote_expand = SBML.libsbml_convert([
    "promoteLocalParameters",
    "expandFunctionDefinitions"
] .=> Ref(Dict{String, String}()))  # We need to use this converter as it does not silently replace symbolic parameters with numeric values
# in some locations.

const IV = (toparam)((wrap)((setmetadata)((Sym){Real}(:t),
    VariableSource,
    (:parameters, :t))))  # Derived from https://github.com/SciML/Catalyst.jl, MIT licensed, see repository for details

function import_petab(yamlfile::String;
        models::Union{ODESystem, Vector{ODESystem}} = ODESystem[])
    pp = get_petab_problem(yamlfile, models = models)
    ppm = petab2mtk(pp)
    return InverseProblem(ppm)
end

function get_petab_problem(yamlfile::String;
        models::Union{ODESystem, Vector{ODESystem}} = ODESystem[],
        collapse_timepoint_overrides::Bool = true)
    petab_yaml = YAML.load_file(yamlfile)
    if typeof(petab_yaml["parameter_file"]) == String
        parameter_path = joinpath(dirname(yamlfile), petab_yaml["parameter_file"])
        parameter_df = CSV.read(parameter_path, DataFrame)
    else
        error("Multiple files not supported")
    end

    condition_df = [CSV.read(joinpath(dirname(yamlfile), fn), DataFrame)
                    for fn in petab_yaml["problems"][1]["condition_files"]]

    measurement_df = [CSV.read(joinpath(dirname(yamlfile), fn), DataFrame)
                      for fn in petab_yaml["problems"][1]["measurement_files"]]

    observable_df = [CSV.read(joinpath(dirname(yamlfile), fn), DataFrame)
                     for fn in petab_yaml["problems"][1]["observable_files"]]

    if models == ODESystem[]
        models = map(petab_yaml["problems"][1]["sbml_files"]) do fn
            readSBML(joinpath(dirname(yamlfile), fn),
                doc -> begin
                    SBML.set_level_and_version(3, 2)(doc)
                    sbml_promote_expand(doc)
                end)
        end
    elseif models isa ODESystem
        models = [models]
    else
        @warn "Models provided as argument, overriding `sbml_files`. Validity of the PEtab problem is not checked."
    end

    if haskey(petab_yaml["problems"][1], "visualization_files")
        visualization_df = [CSV.read(joinpath(dirname(yamlfile), fn), DataFrame)
                            for fn in petab_yaml["problems"][1]["visualization_files"]]
    else
        visualization_df = DataFrame[]
    end

    if haskey(petab_yaml["problems"][1], "dosing_files")
        dosing_df = [CSV.read(joinpath(dirname(yamlfile), fn), DataFrame)
                     for fn in petab_yaml["problems"][1]["dosing_files"]]
    else
        dosing_df = [DataFrame(:simulationConditionId => [])]
    end

    ns = map(l -> length(l),
        [condition_df, measurement_df, observable_df, models, visualization_df, dosing_df])
    any(>(1), ns) && error("Multiple files not supported")

    collapse_timepoint_overrides &&
        collapse_timepoint_overrides!(only(observable_df), only(measurement_df))

    pp = PetabProblem(parameter_df,
        condition_df,
        measurement_df,
        observable_df,
        models,
        visualization_df,
        dosing_df)
    return pp
end

"""
    collapse_timepoint_overrides!(observable_df, measurement_df)

    Internal function to flatten redundant observable and noise parameter overrides.

    The PEtab measurement table allows you to override observable and noise parameters at every single time point.
    We currently don't need such fine granularity. It would also make the objective function much more complicated.
    So for now, we don't support this.

    However, in the PEtab test suite, a lot/all cases use observable and noise parameter overrides (i.e. have the
    optional observableParameter and noiseParameter columns in the measurement table), although in many cases this
    would not be needed, since all time points use the same overrides. These are the cases we can get to work easily
    with the internal collapse_timepoint_overrides function.
"""
function collapse_timepoint_overrides!(observable_df, measurement_df)
    prefixes = String[]
    "observableParameters" in names(measurement_df) && push!(prefixes, "observable")
    "noiseParameters" in names(measurement_df) && push!(prefixes, "noise")
    for (i, row) in enumerate(eachrow(observable_df))
        observableId = row[:observableId]
        for prefix in prefixes
            formula = row["$(prefix)Formula"]

            this_mdf = measurement_df[measurement_df[:, "observableId"] .== observableId, :]
            length(unique(this_mdf[:, "$(prefix)Parameters"])) == 1 ||
                error("Multiple different $(prefix)Parameter overrides not supported")
            overrides = this_mdf[1, "$(prefix)Parameters"]
            ismissing(overrides) && continue
            if overrides isa AbstractString
                overrides = strip.(split(overrides, ";"))
            end
            for (j, override) in enumerate(overrides)
                pattern = Regex("$(prefix)Parameter$(j)+_$(observableId)")
                formula = replace(formula, pattern => override)
            end
            observable_df[i, "$(prefix)Formula"] = formula
        end
    end
end

function _get_module(pp::PetabProblem, sys::ODESystem)  # Derived from https://github.com/SciML/Catalyst.jl, MIT licensed, see repository for details
    # We evaluate all the parameters and species names in a module
    opmod = Module()
    Base.eval(opmod, :(using ModelingToolkit))
    Base.eval(opmod, :(const t = (@parameters t)[1]))

    df_pars = pp.parameter_df.parameterId
    model_pars = string.(ModelingToolkit.parameters(sys))
    pars = vcat(df_pars, model_pars)
    pars = sort(unique(pars))
    for p in pars
        ex = Meta.parse("@parameters $p")
        Base.eval(opmod, ex)
    end

    specs = ModelingToolkit.unknowns(sys)
    for s in specs
        ex = Meta.parse("@variables $s")
        Base.eval(opmod, ex)
    end

    obs = pp.observable_dfs[1].observableId
    for o in obs
        ex = Meta.parse("@variables $o(t)")
        Base.eval(opmod, ex)
    end
    return opmod
end

function petab2mtk(pp::PetabProblem)  # This is where the real work happens
    if all(isequal(1),
        map(length,
            [pp.condition_dfs, pp.measurement_dfs,
                pp.observable_dfs, pp.models]))
        if first(pp.models) isa SBML.Model
            initial_assignments = get_initial_assignments(pp.models[1])
            defs = defaults(SBMLToolkit.ODESystem(pp.models[1]))
            # @debug "defs: $defs"
            sys = SBMLToolkit.ODESystem(pp.models[1],
                defaults = merge(defs, initial_assignments))
        else
            sys = first(pp.models)
        end
        opmod = _get_module(pp, sys)
        observations = get_observations(pp, opmod)
        noiseformulas = get_noiseformulas(pp, opmod)
        neg_logprior = get_neg_logprior(pp)
        measurements = get_measurements(pp.condition_dfs[1], pp.measurement_dfs[1])
        parameters = get_parameters(pp.parameter_df)
        sys = addparams(sys, parameters)
        sys = extend(sys, ODESystem([observations..., noiseformulas...], name = :obssys))
        ssys = structural_simplify(sys)
        conditions = get_conditions(pp.condition_dfs[1], ssys)
        doses = Dict{String, DataFrame}(cid => pp.dosing_dfs[1][
                                            pp.dosing_dfs[1].simulationConditionId .== cid,
                                            :]
        for cid in unique(pp.dosing_dfs[1].simulationConditionId))
        issubset(keys(doses), keys(conditions)) ||
            error("Dosing contains conditions that are not in the condition table")
        neg_llh = get_neg_llh(pp, [keys(conditions)...][1])
    else
        println("Multiple files not supported")
    end

    return PetabProblemMTK(ssys,
        observations,
        conditions,
        neg_llh,
        neg_logprior,
        parameters,
        measurements,
        doses)
end

function addparams(sys::ODESystem, parameters::Dict{String, Parameter})
    for pn in keys(parameters)
        sys = @set sys.ps = [get_ps(sys)..., create_par(pn)]
    end
    sys
end

function get_initial_assignments(model)
    initial_assignments = model.initial_assignments
    symbolic_assignments = Dict()
    for (k, v) in initial_assignments
        var, ass = get_var_and_assignment(model, k, v)
        symbolic_assignments[var] = ass
    end
    symbolic_assignments
end

function get_var_and_assignment(model::SBML.Model, species::String, term::SBML.Math)
    if !haskey(merge(model.species, model.compartments, model.parameters), species)
        error("Cannot find target for rule with ID `$(species)`")
    end
    var = create_var(species, IV)
    math = SBML.extensive_kinetic_math(model, term)
    vc = SBMLToolkit.get_volume_correction(model, species)
    if !isnothing(vc)
        math = SBML.MathApply("*", [SBML.MathIdent(vc), math])
    end
    assignment = SBMLToolkit.interpret_as_num(math, model)
    var, assignment
end

function get_measurements(df_cond, df_meas)
    measurements = Dict{String, DataFrame}()
    for cid in df_cond.conditionId
        !in(cid, df_meas.simulationConditionId) &&
            error("Measurement table does not contain condition $(cid).")
        df = df_meas[df_meas.simulationConditionId .== cid, :]
        select!(df, Not(:simulationConditionId))
        measurements[cid] = df
    end
    return measurements
end

function get_parameters(df)
    parameters = Dict{String, Parameter}()
    for row in eachrow(df)
        parameters[row.parameterId] = Parameter(parameterScale = row.parameterScale,
            lowerBound = float(row.lowerBound),
            upperBound = float(row.upperBound),
            nominalValue = float(row.nominalValue),
            estimate = Bool(row.estimate))
    end
    return parameters
end

function get_conditions(df, ssys)
    c_ids = df.conditionId

    u0_idxs = map(x -> in(x * "(t)", string.(ModelingToolkit.unknowns(ssys))), names(df))
    u0s = map(id -> SBMLToolkit.create_var(id, IV), names(df)[u0_idxs])
    u0vals = Array(df[:, u0_idxs])
    u0vals = [x isa AbstractString ? create_par(x) : float(x) for x in u0vals]

    par_idxs = map(x -> in(x, string.(ModelingToolkit.parameters(ssys))), names(df))
    params = map(id -> create_par(id), names(df)[par_idxs])
    pvals = Array(df[:, par_idxs])
    pvals = [x isa AbstractString ? create_par(x) : float(x) for x in pvals]

    conditions = Dict{String, Dict{String, Vector{Pair{Num, Any}}}}()
    for (i, c) in enumerate(c_ids)
        conditions[c] = Dict{String, Vector{Pair{Num, Any}}}()
        conditions[c]["u0"] = isempty(u0s) ?
                              [] .=> [] :
                              [u0s[j] => u0vals[i, j]
                               for
                               j in 1:length(u0s) if
                               !ismissing(u0vals[i, j])]
        conditions[c]["params"] = isempty(params) ?
                                  [] .=> [] :
                                  [params[j] => pvals[i, j]
                                   for
                                   j in 1:length(params) if
                                   !ismissing(pvals[i, j])]
    end
    return conditions
end

function normal_lossfun(obs, data, sigma)
    sum(0.5 * log.(2 * pi * sigma .^ 2) + 0.5 * (obs - data) .^ 2 ./ sigma .^ 2)
end

function lognormal_lossfun(obs, data, sigma)
    sum(0.5 * log.(2 * pi * sigma .^ 2 .* data .^ 2) +
        0.5 * (log.(obs) - log.(data)) .^ 2 ./ sigma .^ 2)
end

function log10normal_lossfun(obs, data, sigma)
    sum(0.5 * log.(2 * pi * log(10)^2 * sigma .^ 2 .* data .^ 2) +
        0.5 * (log10.(obs) - log10.(data)) .^ 2 ./
        sigma .^ 2)
end

function laplace_lossfun(obs, data, sigma)
    sum(log.(2 * sigma) + abs.(obs - data) ./ sigma)
end

function loglaplace_lossfun(obs, data, sigma)
    sum(log.(2 .* sigma .* data) + abs.(log.(obs) - log.(data)) ./ sigma)
end

function log10laplace_lossfun(obs, data, sigma)
    sum(log.(2 .* sigma .* log(10) .* data) + abs.(log10.(obs) - log10.(data)) ./ sigma)
end

function uniform_lossfun(lb, data, ub)
    ifelse(!all(lb < data < ub), Inf, sum(1 ./ (ub .- lb)))
end

function chi2_lossfun(obs, data, sigma)
    sum((obs - data) .^ 2 ./ sigma .^ 2)
end

function get_neg_llh(pp::PetabProblem, cond::AbstractString)
    lossfuns = Function[]
    odf = copy(pp.observable_dfs[1])
    nrows = nrow(odf)
    if !("noiseDistribution" in names(odf))  # Fill in defaults if optional columns not defined
        odf.noiseDistribution = fill("normal", nrows)
    end
    if !("noiseFormula" in names(odf))
        odf.noiseFormula = fill(1.0, nrows)
    end
    if !("observableTransformation" in names(odf))
        odf.observableTransformation = fill("lin", nrows)
    end

    # Parsing loss function types for each observable
    for row in eachrow(odf)
        if row.noiseDistribution == "normal" && row.observableTransformation == "lin"
            lossfun = normal_lossfun
        elseif row.noiseDistribution == "normal" && row.observableTransformation == "log"
            lossfun = lognormal_lossfun
        elseif row.noiseDistribution == "normal" && row.observableTransformation == "log10"
            lossfun = log10normal_lossfun
        elseif row.noiseDistribution == "laplace" && row.observableTransformation == "lin"
            lossfun = laplace_lossfun
        elseif row.noiseDistribution == "laplace" && row.observableTransformation == "log"
            lossfun = loglaplace_lossfun
        elseif row.noiseDistribution == "laplace" && row.observableTransformation == "log10"
            lossfun = log10laplace_lossfun
        else
            error("Noise distribution and observable transformation not supported")
        end
        push!(lossfuns, lossfun)
    end

    function lossfun(::Any, x, sol, data)
        loss = zero(eltype(sol))
        @views for (i, f) in enumerate(lossfuns)
            loss += f(sol[i, :], data[i, :], sol[i + nrows, :])
        end
        return loss
    end

    function lossfun(::MismatchedMatrixLike{T}, x, sol, data) where {T}
        idxs = map(x -> in(x, data.time), sol.t)
        sol_matched = @view sol[:, idxs]
        lossfun(MatrixLike{T}(), x, sol_matched, data)
    end

    function lossfun(::VectorLike{A},
            x,
            sol,
            data) where {A <: AbstractExperimentData{T}} where {T}  # Todo: think about collocation cost, where we don't solve.
        sum(lossfun.((x,), (sol,), data))
    end

    lossfun(x, sol, data::T) where {T} = lossfun(data_shape(T), x, sol, data)

    return lossfun
end

function get_neg_logprior(pp::PetabProblem)
    if !in("objectivePriorType", names(pp.parameter_df)) ||
       !in("objectivePriorParameters", names(pp.parameter_df))
        return (labelled_x) -> zero(eltype(labelled_x))
    end

    neg_llh_map = Dict("uniform" => uniform_lossfun,  # Posterior costs are not tested in the petab_test_suite yet. Created issue #60 there.
        "normal" => normal_lossfun,
        "laplace" => laplace_lossfun,
        "logNormal" => lognormal_lossfun,
        "logLaplace" => loglaplace_lossfun,
        "parameterScaleUniform" => (lb, data, ub) -> zero(eltype(data)),
        "parameterScaleNormal" => normal_lossfun,
        "parameterScaleLaplace" => laplace_lossfun,
        missing => (lb, data, ub) -> zero(eltype(data)))
    scale_map = Dict("lin" => identity,
        "log" => exp,
        "log10" => x -> 10^x)

    neg_logpriors = Dict{Symbol, Function}()
    parameter_scales = Dict{Symbol, Function}()
    for (id, scale, opt, opp) in eachrow(pp.parameter_df[pp.parameter_df.estimate .== 1,
        ["parameterId", "parameterScale",
            "objectivePriorType", "objectivePriorParameters"]])
        if !ismissing(opp)
            m, s = parse.(Float64, strip.(split(opp, ";")))  # Todo: not sure if this is the best way.
        else
            m, s = missing, missing  # Should not matter, as these cases should also have neg_llh_map[missing] and thus zero penalty.
        end
        f = neg_llh_map[opt]
        neg_logpriors[Symbol(id)] = labelled_x -> f(m, labelled_x, s)
        parameter_scales[Symbol(id)] = scale_map[scale]
    end

    function neg_logprior(labelled_x)
        loss = zero(eltype(labelled_x))
        for (k, v) in pairs(labelled_x)
            loss += neg_logpriors[k](parameter_scales[k](v))
        end
        loss
    end
    return neg_logprior
end

function get_observations(pp::PetabProblem, opmod::Module)
    if any(!isequal(1),
        map(length,
            [pp.condition_dfs, pp.measurement_dfs,
                pp.observable_dfs, pp.models]))
        error("Multiple files not supported")
    end
    observations = Vector{Equation}()
    for (id, term) in eachrow(pp.observable_dfs[1][:,
        [
            "observableId",
            "observableFormula"
        ]])
        push!(observations,
            create_var(string(id), IV) ~ str2sim(string(term), opmod))
    end
    return observations
end

function get_noiseformulas(pp::PetabProblem, opmod::Module)
    noiseformulas = Vector{Equation}()
    if length(pp.observable_dfs) == 1
        df = copy(pp.observable_dfs[1])
        println(df)
        "noiseFormula" in names(df) || (df.noiseFormula = fill(1.0, size(df, 1)))
        for (id, term) in eachrow(df[:, ["observableId", "noiseFormula"]])
            push!(noiseformulas,
                create_var("__sigma_" * string(id), IV) ~ str2sim(string(term), opmod))
        end
    else
        error("Multiple files not supported")
    end

    return noiseformulas
end

function InverseProblem(ppm::PetabProblemMTK)
    search_space = [create_par(k) => (v.lowerBound,
                        v.upperBound,
                        get_parameterscale(v.parameterScale))
                    for (k, v) in ppm.parameters if v.estimate]
    experiments = get_experiments(ppm)
    # @debug "sys defaults: $(defaults(ppm.sys))"
    return InverseProblem(experiments, ppm.sys, search_space,
        defaults_override = PetabOverride(ppm),
        penalty = ppm.neg_logprior)
end

function get_parameterscale(x)
    if x == "lin"
        return :identity
    elseif x == "log"
        return :log
    elseif x == "log10"
        return :log10
    else
        error("Unknown parameterScale $x. Only lin, log and log10 are supported.")
    end
end

function get_experiments(ppm::PetabProblemMTK)
    experiments = Vector{AbstractExperiment}()
    for cid in keys(ppm.conditions)
        measurements = ppm.measurements[cid]
        rename!(measurements, :time => :timestamp)
        obsnames = unique(measurements.observableId)
        sigmas = map(x -> "__sigma_" * x, obsnames)
        replicates = split_replicates(measurements)
        data = [unstack(r, :timestamp, :observableId, :measurement) for r in replicates]
        haskey(ppm.perturbations, cid) && error("Perturbations not supported.")
        u0 = ppm.conditions[cid]["u0"]
        params = ppm.conditions[cid]["params"]
        loss_func = ppm.neg_llh
        push!(experiments,
            Experiment(data, ppm.sys;
                u0,
                params,
                loss_func,
                save_names = Symbol.([obsnames..., sigmas...]),
                name = cid))
    end
    return experiments
end

function create_par(name::AbstractString)  # Todo: maybe replace at some point with SBMLToolkit.create_param
    sym = Symbol(name)
    first(@parameters $sym)
end

function str2sim(term::AbstractString, opmod)
    ex = Meta.parse(term)
    ex = Base.eval(opmod, ex)  # @Sebastian: what to do about this?
    return ex
end

function get_petab_simulation_df(res::CalibrationResult, pp::PetabProblem)
    simulation_df = copy(pp.measurement_dfs[1])
    rename!(simulation_df, (:measurement => :simulation))
    simulation_df[!, :simulation] .= NaN

    for cond in pp.condition_dfs[1].conditionId
        experiment = get_experiment(string(cond), res.prob)
        sol = simulate(experiment, res.prob, res.u)
        for obs in pp.observable_dfs[1].observableId
            idxs = (simulation_df.simulationConditionId .== cond) .&
                   (simulation_df.observableId .== obs)
            for i in 1:sum(idxs)
                t = simulation_df[idxs, :time][i]
                j = map(x -> x == t, sol.t)
                df = @view simulation_df[idxs, :simulation]
                df[i] = only(sol[create_var(obs, IV)][j])
            end
        end
    end
    any(isnan.(simulation_df.simulation)) && error("NaNs in simulation_df")
    return simulation_df
end

function get_experiment(name::AbstractString, prob::AbstractInverseProblem)
    names = [nameof(experiment) for experiment in get_experiments(prob)]
    idxs = findall(names .== name)
    str(p) = printname(first(get_experiments(prob)), plural = p)
    length(idxs) > 1 && error("Multiple $(str(true)) with the same name!")
    isempty(idxs) && error("No $(str(false)) with that name")

    return first(iterate(get_experiments(prob), only(idxs)))
end

function get_chi2_function(experiment::AbstractExperiment)
    n_obs = Int(length(get_saved_model_variables(experiment)) / 2)  # The other half of the saved model variables are sigmas.
    function lossfun(::Any, x, sol, data)
        loss = zero(eltype(sol))
        @views for i in 1:n_obs
            loss += chi2_lossfun(sol[i, :], data[i, :], sol[i + n_obs, :])
        end
        return loss
    end

    function lossfun(::MismatchedMatrixLike{T}, x, sol, data) where {T}
        idxs = map(x -> in(x, data.time), sol.t)
        sol_matched = @view sol[:, idxs]
        lossfun(MatrixLike{T}(), x, sol_matched, data)
    end

    function lossfun(::VectorLike{A},
            x,
            sol,
            data) where {A <: AbstractExperimentData{T}} where {T} # Todo: think about collocation cost, where we don't solve.
        sum(lossfun.((x,), (sol,), data))
    end

    lossfun(x, sol, data::T) where {T} = begin
        lossfun(data_shape(T), x, sol, data)
    end

    return lossfun
end

function problem_chi2(res::CalibrationResult)
    # @Sebastian: atm this outputs garbage if sol is not split in observables and sigmas (i.e. most problems not imported via petab_import).
    # Chi2 is generally useful (e.g. to do a Chi2 test for goodness of fit. So it would be
    # nice to get this to work for any CalibrationResult (e.g. with ones as default for the std).
    # Once done, we can make this public API.
    loss = zero(eltype(res.u))
    for experiment in get_experiments(res.prob)
        loss += compute_chi2(experiment, res)
    end
    loss
end

function compute_chi2(experiment::AbstractExperiment, res::CalibrationResult)  # can be exported if we generally support sigmas for InverseProblems/Experiments.
    loss_func = get_chi2_function(experiment)
    sol = trysolve(experiment, res.prob, res)
    compute_error(experiment, res.prob, res.u, sol; loss_func)
end

function split_replicates(df::DataFrame; v = [])
    nrow(df) == 0 && return v
    idx = nonunique(df, Not(:measurement))
    push!(v, df[.!idx, :])
    split_replicates(df[idx, :], v = v)
end

function create_petab_template(outdir::String = joinpath(pwd(), "petab_problem"); kwargs...)
    @info "Downloading PEtab template"
    PkgAuthentication.authenticate()
    JuliaHubData.connect_juliahub_data_project()
    PEtab_blob = dataset("juliasimtutorials/PEtab_template")
    PEtab_tmp = open(IO, PEtab_blob) do io
        Tar.extract(io)
    end
    cp(PEtab_tmp, outdir; kwargs...)
end
